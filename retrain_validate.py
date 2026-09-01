from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mlflow


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def main(
    strategy: str,
    split_strategy: str,
    tracking_uri: str | None,
    register_model: bool,
    promote_model: bool,
    model_name: str,
    min_gain: float,
) -> None:
    root = Path(__file__).resolve().parent

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    _run(
        [
            "dvc",
            "exp",
            "run",
            "prep_mlflow_data",
            "--set-param",
            f"prep_mlflow_data.strategy={strategy}",
            "--set-param",
            f"prep_mlflow_data.split_strategy={split_strategy}",
        ]
    )

    train_run = mlflow.run(
        uri=str(root),
        entry_point="train",
        parameters={
            "data": str(root / "data" / "processed" / "train.parquet"),
            "validation": str(root / "data" / "processed" / "validation.parquet"),
        },
        env_manager="local",
        synchronous=True,
    )

    mlflow.run(
        uri=str(root),
        entry_point="predict",
        parameters={
            "run_id": train_run.run_id,
            "data": str(root / "data" / "processed" / "test.parquet"),
            "reference_data": str(root / "data" / "processed" / "train.parquet"),
        },
        env_manager="local",
        synchronous=True,
    )

    if register_model:
        mlflow.run(
            uri=str(root),
            entry_point="register",
            parameters={
                "train": str(root / "data" / "processed" / "train.parquet"),
                "validation": str(root / "data" / "processed" / "validation.parquet"),
                "model_name": model_name,
            },
            env_manager="local",
            synchronous=True,
        )

    if register_model and promote_model:
        mlflow.run(
            uri=str(root),
            entry_point="promote",
            parameters={"model_name": model_name, "min_gain": min_gain},
            env_manager="local",
            synchronous=True,
        )

    print(f"train_run_id={train_run.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated retraining + validation pipeline")
    parser.add_argument("--strategy", default="mixed")
    parser.add_argument("--split-strategy", default="full_history")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--register-model", default="false")
    parser.add_argument("--promote-model", default="false")
    parser.add_argument("--model-name", default="electricity_forecaster")
    parser.add_argument("--min-gain", type=float, default=0.02)
    args = parser.parse_args()

    main(
        strategy=args.strategy,
        split_strategy=args.split_strategy,
        tracking_uri=args.tracking_uri,
        register_model=_to_bool(args.register_model),
        promote_model=_to_bool(args.promote_model),
        model_name=args.model_name,
        min_gain=args.min_gain,
    )
