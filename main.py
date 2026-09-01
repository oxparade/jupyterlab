"""Orchestrateur TP01: prep_data -> train -> serve."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mlflow

from config import ModellingStrategy, SplitStrategy


def main(
    strategy: ModellingStrategy = ModellingStrategy.MIXED,
    split_strategy: SplitStrategy = SplitStrategy.FULL_HISTORY,
    host: str = "127.0.0.1",
    port: int = 5001,
) -> None:
    root = Path(__file__).resolve().parent

    subprocess.run(
        [
            sys.executable,
            str(root / "prep.py"),
            "--strategy",
            str(strategy),
            "--split-strategy",
            str(split_strategy),
        ],
        check=True,
    )

    submitted = mlflow.run(
        uri=str(root),
        entry_point="train",
        parameters={
            "data": str(root / "data" / "processed" / "train.parquet"),
            "validation": str(root / "data" / "processed" / "validation.parquet"),
        },
        env_manager="local",
        synchronous=True,
    )

    model_uri = f"runs:/{submitted.run_id}/model"
    subprocess.run(
        [
            "mlflow",
            "models",
            "serve",
            "-m",
            model_uri,
            "-h",
            host,
            "-p",
            str(port),
            "--env-manager",
            "local",
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrateur MLflow: prep -> train -> serve")
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in ModellingStrategy],
        default=ModellingStrategy.MIXED.value,
    )
    parser.add_argument(
        "--split-strategy",
        choices=[split.value for split in SplitStrategy],
        default=SplitStrategy.FULL_HISTORY.value,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    main(
        strategy=ModellingStrategy(args.strategy),
        split_strategy=SplitStrategy(args.split_strategy),
        host=args.host,
        port=args.port,
    )
