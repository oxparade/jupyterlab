"""TP M02TP01 — train: sélectionne alpha sur validation, log le modèle avec signature."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import EXPERIMENT, RIDGE_ALPHAS, TARGET
from figures import coefficient_weights
from pipeline_steps import evaluate, train_ridge

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _run_cmd(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(args, check=True, capture_output=True, text=True)
    except Exception:
        return None
    value = (completed.stdout or "").strip()
    return value or None


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _log_repro_context() -> None:
    git_commit = _run_cmd(["git", "rev-parse", "HEAD"])
    git_branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_dirty = _run_cmd(["git", "status", "--porcelain"])
    dvc_version = _run_cmd(["dvc", "--version"])
    dvc_lock_hash = _file_sha256(Path("dvc.lock"))

    if git_commit:
        mlflow.set_tag("git_commit", git_commit)
    if git_branch:
        mlflow.set_tag("git_branch", git_branch)
    mlflow.set_tag("git_dirty", "true" if git_dirty else "false")
    if dvc_version:
        mlflow.set_tag("dvc_version", dvc_version)
    if dvc_lock_hash:
        mlflow.set_tag("dvc_lock_sha256", dvc_lock_hash)


def _extract_ridge_coefficients(model: Any) -> Any | None:
    if hasattr(model, "coef_"):
        return model
    if isinstance(model, Pipeline) and "ridge" in model.named_steps:
        ridge_step = model.named_steps["ridge"]
        if hasattr(ridge_step, "coef_"):
            return ridge_step
    return None


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

    training_dataset = mlflow.data.from_pandas(
        train_data,
        source=str(data),
        name="electricity_train",
    )
    validation_dataset = mlflow.data.from_pandas(
        validation_data,
        source=str(validation),
        name="electricity_validation",
    )
    features = [
        column
        for column in train_data.columns
        if column != TARGET and pd.api.types.is_numeric_dtype(train_data[column])
    ]

    X_train, y_train = train_data[features], train_data[TARGET]
    X_validation, y_validation = validation_data[features], validation_data[TARGET]

    best_alpha: float | None = None
    best_standardize: bool | None = None
    best_model = None
    best_metrics: dict[str, float] | None = None

    if active_run is not None:
        run_context = nullcontext()
    elif project_run_id is not None:
        run_context = mlflow.start_run(run_id=project_run_id)
    else:
        run_context = mlflow.start_run(run_name="ridge-train")

    with run_context:
        _log_repro_context()
        mlflow.log_input(training_dataset, context="training")
        mlflow.log_input(validation_dataset, context="validation")
        mlflow.log_param("train_path", str(data))
        mlflow.log_param("validation_path", str(validation))
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("n_train_rows", int(len(X_train)))
        mlflow.log_param("n_validation_rows", int(len(X_validation)))
        mlflow.log_param("ridge_standardization_options", "false,true")

        for standardize in (False, True):
            for alpha in RIDGE_ALPHAS:
                with mlflow.start_run(
                    run_name=f"alpha={alpha}|standardize={str(standardize).lower()}",
                    nested=True,
                ):
                    if standardize:
                        model = Pipeline(
                            [
                                ("scaler", StandardScaler()),
                                ("ridge", Ridge(alpha=alpha)),
                            ]
                        )
                        model.fit(X_train, y_train)
                    else:
                        model = train_ridge(X_train, y_train, alpha=alpha)
                    metrics = evaluate(model, X_validation, y_validation)
                    mlflow.log_param("alpha", float(alpha))
                    mlflow.log_param("standardize", str(standardize).lower())
                    mlflow.log_metric("validation_rmse", metrics["rmse"])
                    mlflow.log_metric("validation_mae", metrics["mae"])

                    coef_model = _extract_ridge_coefficients(model)
                    if coef_model is not None:
                        figure = coefficient_weights(coef_model, features)
                        mlflow.log_figure(
                            figure,
                            f"coefficient_weights_standardize_{str(standardize).lower()}_alpha_{alpha}.html",
                        )

                if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
                    best_alpha = float(alpha)
                    best_model = model
                    best_metrics = metrics
                    best_standardize = standardize

        if best_model is None or best_alpha is None or best_metrics is None or best_standardize is None:
            raise RuntimeError("Aucun modèle entraîné — vérifier RIDGE_ALPHAS.")

        signature = infer_signature(X_validation, best_model.predict(X_validation))
        input_example = X_validation.head(5)

        mlflow.log_param("selected_alpha", best_alpha)
        mlflow.log_param("selected_standardize", str(best_standardize).lower())
        mlflow.log_metric("best_validation_rmse", best_metrics["rmse"])
        mlflow.log_metric("best_validation_mae", best_metrics["mae"])
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            signature=signature,
            input_example=input_example,
        )
        logger.info(
            "Best alpha=%s standardize=%s validation_rmse=%.4f validation_mae=%.4f",
            best_alpha,
            best_standardize,
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
