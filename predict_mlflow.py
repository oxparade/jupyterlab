"""TP M02TP01 — evaluate: charge un modèle existant et l'évalue sans réentraîner."""

from __future__ import annotations

import argparse
import logging
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


def main(
    model: str,
    data: Path = Path("data/processed/test.parquet"),
    reference_data: Path | None = Path("data/processed/train.parquet"),
) -> None:
    """Charge le modèle, score sur test, log les métriques dans le run source."""
    logger.info("loading %s", model)
    regressor = mlflow.sklearn.load_model(model)

    test_data = pd.read_parquet(data)
    features = [column for column in test_data.columns if column != TARGET]
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
    args = parser.parse_args()
    main(model=args.model, data=args.data, reference_data=args.reference_data)
