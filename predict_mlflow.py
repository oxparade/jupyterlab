"""TP M02TP01 — evaluate: charge un modèle existant et l'évalue sans réentraîner."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from config import TARGET
from figures import residual_breakdown, residual_diagnostics
from pipeline_steps import evaluate

try:
    from evidently import Report
    from evidently.metric_preset import DataDriftPreset
except Exception:  # pragma: no cover
    Report = None
    DataDriftPreset = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _extract_run_id(model_uri: str) -> str | None:
    if not model_uri.startswith("runs:/"):
        return None
    parts = model_uri.split("/")
    if len(parts) < 3:
        return None
    return parts[1].replace("runs:", "")


def _log_report_to_evidently_ui(
    report,
    *,
    project_name: str,
    api_url: str,
    secret: str | None = None,
) -> None:
    if not api_url:
        return

    def _publish() -> None:
        if api_url == "https://app.evidently.cloud":
            if not secret:
                raise ValueError("Evidently Cloud requires EVIDENTLY_SECRET")
            from evidently.ui.workspace import CloudWorkspace

            workspace = CloudWorkspace(token=secret, url=api_url)
        else:
            from evidently.ui.workspace import Workspace

            workspace = Workspace.create(api_url)

        report.name = project_name
        snapshot = report.to_snapshot()
        project = next((item for item in workspace.list_projects() if item.name == project_name), None)
        if project is None:
            project = workspace.create_project(project_name)
        project.add_snapshot(snapshot)
        logger.info("logged Evidently snapshot to project=%s", project_name)

    try:
        _publish()
    except Exception as first_error:
        insecure_tls_requested = os.environ.get("EVIDENTLY_INSECURE_TLS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if not (api_url.startswith("https://") or insecure_tls_requested):
            logger.exception("failed to log snapshot to Evidently UI at %s", api_url)
            return

        try:
            import requests

            original_send = requests.Session.send

            def insecure_send(self, request, **kwargs):
                kwargs.setdefault("verify", False)
                return original_send(self, request, **kwargs)

            requests.Session.send = insecure_send
            try:
                _publish()
            finally:
                requests.Session.send = original_send
        except Exception:
            logger.exception("failed to log snapshot to Evidently UI at %s", api_url)
            logger.debug("initial Evidently UI error: %s", first_error)


def main(
    model: str,
    data: Path = Path("data/processed/test.parquet"),
    reference_data: Path | None = Path("data/processed/train.parquet"),
    evidently_api_url: str = os.environ.get("EVIDENTLY_API_URL", ""),
    evidently_project_name: str = os.environ.get("EVIDENTLY_PROJECT_NAME", "electricity_forecaster"),
    evidently_secret: str | None = os.environ.get("EVIDENTLY_SECRET"),
) -> None:
    """Charge le modèle, score sur test, log les métriques dans le run source."""
    logger.info("loading %s", model)
    regressor = mlflow.sklearn.load_model(model)

    test_data = pd.read_parquet(data)
    features = [
        column
        for column in test_data.columns
        if column != TARGET and pd.api.types.is_numeric_dtype(test_data[column])
    ]
    X_test = test_data[features]
    y_test = test_data[TARGET]

    metrics = evaluate(regressor, X_test, y_test)
    predictions = pd.Series(regressor.predict(X_test), index=y_test.index)

    client_series: pd.Series | None = None
    if "client" in test_data.columns:
        client_series = test_data["client"]
    elif isinstance(test_data.index, pd.MultiIndex) and "individual" in test_data.index.names:
        client_series = pd.Series(
            test_data.index.get_level_values("individual").astype(str),
            index=test_data.index,
        )

    breakdown_figures = residual_breakdown(y_test, predictions, client=client_series)

    drift_report_html: str | None = None
    if reference_data is not None and Report is not None and DataDriftPreset is not None and reference_data.exists():
        reference_df = pd.read_parquet(reference_data)
        ref_features = reference_df[features].copy()
        cur_features = X_test.copy()
        numeric_columns = [
            column
            for column in features
            if pd.api.types.is_numeric_dtype(ref_features[column])
            and pd.api.types.is_numeric_dtype(cur_features[column])
        ]
        if numeric_columns:
            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=ref_features[numeric_columns],
                current_data=cur_features[numeric_columns],
            )
            _log_report_to_evidently_ui(
                report,
                project_name=evidently_project_name,
                api_url=evidently_api_url,
                secret=evidently_secret,
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "drift_report.html"
                report.save_html(str(temp_path))
                drift_report_html = temp_path.read_text(encoding="utf-8")

    source_run_id = _extract_run_id(model)
    if source_run_id is None:
        if mlflow.active_run() is None:
            with mlflow.start_run(run_name="evaluate-existing-model"):
                mlflow.log_param("model_uri", model)
                mlflow.log_param("test_path", str(data))
                mlflow.log_metrics({"test_rmse": metrics["rmse"], "test_mae": metrics["mae"]})
                mlflow.log_figure(
                    residual_diagnostics(y_test, predictions),
                    "residual_diagnostics.html",
                )
                for artifact_name, figure in breakdown_figures.items():
                    mlflow.log_figure(figure, f"{artifact_name}.html")
                if drift_report_html is not None:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        path = Path(temp_dir) / "drift_report.html"
                        path.write_text(drift_report_html, encoding="utf-8")
                        mlflow.log_artifact(str(path), artifact_path="drift")
        else:
            mlflow.log_param("model_uri", model)
            mlflow.log_param("test_path", str(data))
            mlflow.log_metrics({"test_rmse": metrics["rmse"], "test_mae": metrics["mae"]})
            mlflow.log_figure(
                residual_diagnostics(y_test, predictions),
                "residual_diagnostics.html",
            )
            for artifact_name, figure in breakdown_figures.items():
                mlflow.log_figure(figure, f"{artifact_name}.html")
            if drift_report_html is not None:
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "drift_report.html"
                    path.write_text(drift_report_html, encoding="utf-8")
                    mlflow.log_artifact(str(path), artifact_path="drift")
    else:
        client = MlflowClient()
        client.log_param(source_run_id, "test_path", str(data))
        client.log_metric(source_run_id, "test_rmse", float(metrics["rmse"]))
        client.log_metric(source_run_id, "test_mae", float(metrics["mae"]))
        if reference_data is not None:
            client.log_param(source_run_id, "drift_reference_path", str(reference_data))
        figure = residual_diagnostics(y_test, predictions)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "residual_diagnostics.html"
            figure.write_html(path)
            client.log_artifact(source_run_id, str(path), artifact_path=None)
            for artifact_name, artifact_figure in breakdown_figures.items():
                artifact_path = Path(temp_dir) / f"{artifact_name}.html"
                artifact_figure.write_html(artifact_path)
                client.log_artifact(source_run_id, str(artifact_path), artifact_path=None)
            if drift_report_html is not None:
                drift_path = Path(temp_dir) / "drift_report.html"
                drift_path.write_text(drift_report_html, encoding="utf-8")
                client.log_artifact(source_run_id, str(drift_path), artifact_path="drift")

    if drift_report_html is None and reference_data is not None:
        logger.info("drift report skipped (reference missing or evidently unavailable)")

    logger.info("test_rmse=%.4f test_mae=%.4f", metrics["rmse"], metrics["mae"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue un modèle MLflow existant")
    parser.add_argument("--model", required=True, help="Model URI, ex: runs:/<id>/model")
    parser.add_argument("--data", type=Path, default=Path("data/processed/test.parquet"))
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=Path("data/processed/train.parquet"),
        help="Reference dataset for Evidently drift report",
    )
    parser.add_argument(
        "--evidently-api-url",
        default=os.environ.get("EVIDENTLY_API_URL", ""),
        help="Evidently UI base URL for publishing snapshots",
    )
    parser.add_argument(
        "--evidently-project-name",
        default=os.environ.get("EVIDENTLY_PROJECT_NAME", "electricity_forecaster"),
        help="Evidently project name to create or update",
    )
    parser.add_argument(
        "--evidently-secret",
        default=os.environ.get("EVIDENTLY_SECRET"),
        help="Secret header/token for Evidently UI writes",
    )
    args = parser.parse_args()
    main(
        model=args.model,
        data=args.data,
        reference_data=args.reference_data,
        evidently_api_url=args.evidently_api_url,
        evidently_project_name=args.evidently_project_name,
        evidently_secret=args.evidently_secret,
    )
