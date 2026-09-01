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
from figures import residual_diagnostics
from pipeline_steps import evaluate

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
        else:
            mlflow.log_param("model_uri", model)
            mlflow.log_param("test_path", str(data))
            mlflow.log_metrics({"test_rmse": metrics["rmse"], "test_mae": metrics["mae"]})
            mlflow.log_figure(
                residual_diagnostics(y_test, predictions),
                "residual_diagnostics.html",
            )
    else:
        client = MlflowClient()
        client.log_param(source_run_id, "test_path", str(data))
        client.log_metric(source_run_id, "test_rmse", float(metrics["rmse"]))
        client.log_metric(source_run_id, "test_mae", float(metrics["mae"]))
        figure = residual_diagnostics(y_test, predictions)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "residual_diagnostics.html"
            figure.write_html(path)
            client.log_artifact(source_run_id, str(path), artifact_path=None)

    logger.info("test_rmse=%.4f test_mae=%.4f", metrics["rmse"], metrics["mae"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue un modèle MLflow existant")
    parser.add_argument("--model", required=True, help="Model URI, ex: runs:/<id>/model")
    parser.add_argument("--data", type=Path, default=Path("data/processed/test.parquet"))
    args = parser.parse_args()
    main(model=args.model, data=args.data)
