"""TP02 — Enregistre deux versions dans le Model Registry et pose champion/challenger."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature

from config import EXPERIMENT, MODELLING_FEATURES, RIDGE_ALPHAS, ModellingStrategy, TARGET
from mlflow_helpers import step_run
from pipeline_steps import evaluate, train_ridge

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _best_model_for_strategy(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    strategy: ModellingStrategy,
) -> tuple[float, object, dict[str, float], list[str]]:
    features = MODELLING_FEATURES[strategy]
    missing = [column for column in features if column not in train_data.columns]
    if missing:
        raise KeyError(f"Missing feature(s) for {strategy.value}: {missing}")

    X_train = train_data[features]
    y_train = train_data[TARGET]
    X_validation = validation_data[features]
    y_validation = validation_data[TARGET]

    best_alpha: float | None = None
    best_model = None
    best_metrics: dict[str, float] | None = None

    for alpha in RIDGE_ALPHAS:
        model = train_ridge(X_train, y_train, alpha=alpha)
        metrics = evaluate(model, X_validation, y_validation)
        logger.info(
            "strategy=%s alpha=%s validation_rmse=%.4f",
            strategy.value,
            alpha,
            metrics["rmse"],
        )
        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_alpha = float(alpha)
            best_model = model
            best_metrics = metrics

    assert best_model is not None and best_alpha is not None and best_metrics is not None
    return best_alpha, best_model, best_metrics, features


def _wait_model_ready(client: MlflowClient, model_name: str, version: str, timeout_s: int = 60) -> None:
    start = time.time()
    while True:
        mv = client.get_model_version(model_name, version)
        status = (mv.status or "").upper()
        if status == "READY" or status == "":
            return
        if status == "FAILED_REGISTRATION":
            raise RuntimeError(f"Model registration failed: {model_name} v{version}")
        if time.time() - start > timeout_s:
            return
        time.sleep(1)


def main(
    train_path: Path,
    validation_path: Path,
    model_name: str,
    first_strategy: ModellingStrategy,
    second_strategy: ModellingStrategy,
) -> None:
    if mlflow.active_run() is None and os.getenv("MLFLOW_RUN_ID") is None:
        mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    train_data = pd.read_parquet(train_path)
    validation_data = pd.read_parquet(validation_path)

    training_dataset = mlflow.data.from_pandas(
        train_data,
        source=str(train_path),
        name="electricity_train",
    )
    validation_dataset = mlflow.data.from_pandas(
        validation_data,
        source=str(validation_path),
        name="electricity_validation",
    )

    created_versions: list[tuple[str, str]] = []

    with step_run("tp02-register", nested=False):
        mlflow.log_input(training_dataset, context="training")
        mlflow.log_input(validation_dataset, context="validation")
        mlflow.log_param("train_digest", training_dataset.digest)
        mlflow.log_param("validation_digest", validation_dataset.digest)

        for strategy in (first_strategy, second_strategy):
            with step_run(f"register-{strategy.value}", nested=True):
                alpha, model, metrics, features = _best_model_for_strategy(
                    train_data=train_data,
                    validation_data=validation_data,
                    strategy=strategy,
                )

                X_validation = validation_data[features]
                signature = infer_signature(X_validation, model.predict(X_validation))

                mlflow.log_param("strategy", strategy.value)
                mlflow.log_param("selected_alpha", alpha)
                mlflow.log_param("features", ",".join(features))
                mlflow.log_metric("validation_rmse", metrics["rmse"])
                mlflow.log_metric("validation_mae", metrics["mae"])

                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    signature=signature,
                    input_example=X_validation.head(5),
                )

                registered = mlflow.register_model(
                    model_uri=model_info.model_uri,
                    name=model_name,
                )
                version = str(registered.version)
                _wait_model_ready(client, model_name, version)

                client.set_model_version_tag(model_name, version, "strategy", strategy.value)
                client.set_model_version_tag(model_name, version, "selected_alpha", str(alpha))
                client.set_model_version_tag(model_name, version, "train_digest", training_dataset.digest)
                client.set_model_version_tag(model_name, version, "validation_digest", validation_dataset.digest)
                client.set_model_version_tag(
                    model_name,
                    version,
                    "rmse",
                    f"{metrics['rmse']:.6f}",
                )
                client.set_model_version_tag(
                    model_name,
                    version,
                    "mae",
                    f"{metrics['mae']:.6f}",
                )
                client.set_model_version_tag(model_name, version, "passed_validation", "true")
                client.set_model_version_tag(model_name, version, "validated", "true")
                client.update_model_version(
                    name=model_name,
                    version=version,
                    description=(
                        f"Strategy={strategy.value}, alpha={alpha}, "
                        f"validation_rmse={metrics['rmse']:.4f}, validation_mae={metrics['mae']:.4f}"
                    ),
                )
                created_versions.append((strategy.value, version))
                logger.info("Registered %s version=%s", strategy.value, version)

        champion_version = created_versions[0][1]
        challenger_version = created_versions[1][1]
        client.update_registered_model(
            name=model_name,
            description=(
                "Prévision de consommation électrique multi-clients. "
                "Versions entraînées sur données préparées avec features de lag/rolling."
            ),
        )
        client.set_registered_model_alias(model_name, "champion", champion_version)
        client.set_registered_model_alias(model_name, "challenger", challenger_version)
        logger.info(
            "Aliases set: champion=v%s challenger=v%s",
            champion_version,
            challenger_version,
        )

    print("Registered versions:")
    for strategy_name, version in created_versions:
        print(f"- {strategy_name}: v{version}")
    print(f"- champion -> v{champion_version}")
    print(f"- challenger -> v{challenger_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP02 register: versions + champion/challenger")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.parquet"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.parquet"))
    parser.add_argument("--model-name", default="electricity_forecaster")
    parser.add_argument(
        "--first-strategy",
        choices=[strategy.value for strategy in ModellingStrategy],
        default=ModellingStrategy.SHORT_MEMORY.value,
    )
    parser.add_argument(
        "--second-strategy",
        choices=[strategy.value for strategy in ModellingStrategy],
        default=ModellingStrategy.MIXED.value,
    )
    args = parser.parse_args()

    main(
        train_path=args.train,
        validation_path=args.validation,
        model_name=args.model_name,
        first_strategy=ModellingStrategy(args.first_strategy),
        second_strategy=ModellingStrategy(args.second_strategy),
    )
