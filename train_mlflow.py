"""TP M02TP01 — train: sélectionne alpha sur validation, log le modèle avec signature."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature

from config import EXPERIMENT, RIDGE_ALPHAS, TARGET
from figures import coefficient_weights
from pipeline_steps import evaluate, train_ridge

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main(
    data: Path = Path("data/processed/train.parquet"),
    validation: Path = Path("data/processed/validation.parquet"),
) -> None:
    """Fit un Ridge par alpha, conserve le meilleur sur validation, log dans MLflow."""
    active_run = mlflow.active_run()
    project_run_id = os.getenv("MLFLOW_RUN_ID")
    if active_run is None and project_run_id is None:
        mlflow.set_experiment(EXPERIMENT)

    train_data = pd.read_parquet(data)
    validation_data = pd.read_parquet(validation)
    features = [column for column in train_data.columns if column != TARGET]

    X_train, y_train = train_data[features], train_data[TARGET]
    X_validation, y_validation = validation_data[features], validation_data[TARGET]

    best_alpha: float | None = None
    best_model = None
    best_metrics: dict[str, float] | None = None

    if active_run is not None:
        run_context = nullcontext()
    elif project_run_id is not None:
        run_context = mlflow.start_run(run_id=project_run_id)
    else:
        run_context = mlflow.start_run(run_name="ridge-train")

    with run_context:
        mlflow.log_param("train_path", str(data))
        mlflow.log_param("validation_path", str(validation))
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("n_train_rows", int(len(X_train)))
        mlflow.log_param("n_validation_rows", int(len(X_validation)))

        for alpha in RIDGE_ALPHAS:
            with mlflow.start_run(run_name=f"alpha={alpha}", nested=True):
                model = train_ridge(X_train, y_train, alpha=alpha)
                metrics = evaluate(model, X_validation, y_validation)
                mlflow.log_param("alpha", float(alpha))
                mlflow.log_metric("validation_rmse", metrics["rmse"])
                mlflow.log_metric("validation_mae", metrics["mae"])

                figure = coefficient_weights(model, features)
                mlflow.log_figure(figure, "coefficient_weights.html")

            if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
                best_alpha = float(alpha)
                best_model = model
                best_metrics = metrics

        if best_model is None or best_alpha is None or best_metrics is None:
            raise RuntimeError("Aucun modèle entraîné — vérifier RIDGE_ALPHAS.")

        signature = infer_signature(X_validation, best_model.predict(X_validation))
        input_example = X_validation.head(5)

        mlflow.log_param("selected_alpha", best_alpha)
        mlflow.log_metric("best_validation_rmse", best_metrics["rmse"])
        mlflow.log_metric("best_validation_mae", best_metrics["mae"])
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            signature=signature,
            input_example=input_example,
        )
        logger.info(
            "Best alpha=%s validation_rmse=%.4f validation_mae=%.4f",
            best_alpha,
            best_metrics["rmse"],
            best_metrics["mae"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ridge avec suivi MLflow")
    parser.add_argument("--data", type=Path, default=Path("data/processed/train.parquet"))
    parser.add_argument(
        "--validation", type=Path, default=Path("data/processed/validation.parquet")
    )
    args = parser.parse_args()
    main(data=args.data, validation=args.validation)
