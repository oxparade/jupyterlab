from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from config import EXPERIMENT


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _resolve_trigger_source(trigger_source: str) -> str:
    if trigger_source != "auto":
        return trigger_source

    if os.getenv("GITHUB_ACTIONS") != "true":
        return "manual"

    event_name = os.getenv("GITHUB_EVENT_NAME", "")
    if event_name == "schedule":
        return "scheduled"
    if event_name == "workflow_dispatch":
        return "manual"
    if event_name in {"push", "pull_request", "pull_request_target", "workflow_call"}:
        return "ci"
    return "manual"


def _comparison_filter(train_digest: str | None, validation_digest: str | None) -> str:
    parts: list[str] = []
    if train_digest:
        parts.append(f"params.train_digest = '{train_digest}'")
    if validation_digest:
        parts.append(f"params.validation_digest = '{validation_digest}'")
    return " and ".join(parts)


def _log_cycle_decision(train_run_id: str, trigger: str, strategy: str, split_strategy: str) -> dict[str, str] | None:
    client = MlflowClient()
    run = client.get_run(train_run_id)
    train_digest = run.data.params.get("train_digest")
    validation_digest = run.data.params.get("validation_digest")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return None

    filter_string = _comparison_filter(train_digest, validation_digest)
    runs = mlflow.search_runs([experiment.experiment_id], filter_string=filter_string)
    if runs.empty:
        return None

    sort_columns = [
        column
        for column in (
            "metrics.test_rmse",
            "metrics.test_mae",
            "metrics.best_validation_rmse",
            "metrics.best_validation_mae",
            "start_time",
        )
        if column in runs.columns
    ]
    if sort_columns:
        runs = runs.sort_values(by=sort_columns, ascending=[True] * len(sort_columns))

    comparison_columns = [
        column
        for column in (
            "run_id",
            "params.train_path",
            "params.validation_path",
            "params.train_digest",
            "params.validation_digest",
            "metrics.test_rmse",
            "metrics.test_mae",
            "metrics.best_validation_rmse",
            "metrics.best_validation_mae",
            "start_time",
        )
        if column in runs.columns
    ]
    comparison = runs.loc[:, comparison_columns].copy()
    comparison.insert(0, "rank", range(1, len(comparison) + 1))

    best_run_id = str(runs.iloc[0]["run_id"])
    best_test_rmse = float(runs.iloc[0]["metrics.test_rmse"]) if "metrics.test_rmse" in runs.columns else float("nan")
    best_test_mae = float(runs.iloc[0]["metrics.test_mae"]) if "metrics.test_mae" in runs.columns else float("nan")

    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.set_tag("cycle_trigger", trigger)
        mlflow.set_tag("cycle_strategy", strategy)
        mlflow.set_tag("cycle_split_strategy", split_strategy)
        if train_digest:
            mlflow.set_tag("cycle_train_digest", train_digest)
            mlflow.log_param("cycle_train_digest", train_digest)
        if validation_digest:
            mlflow.set_tag("cycle_validation_digest", validation_digest)
            mlflow.log_param("cycle_validation_digest", validation_digest)
        mlflow.set_tag("cycle_best_run_id", best_run_id)
        mlflow.log_param("cycle_best_run_id", best_run_id)
        mlflow.set_tag("cycle_decision", "best-candidate-selected")
        mlflow.log_metric("cycle_best_test_rmse", best_test_rmse)
        mlflow.log_metric("cycle_best_test_mae", best_test_mae)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "retrain_comparison.csv"
            comparison.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_path="comparison")

            summary_path = temp_path / "retrain_decision.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "trigger": trigger,
                        "strategy": strategy,
                        "split_strategy": split_strategy,
                        "train_run_id": train_run_id,
                        "best_run_id": best_run_id,
                        "same_run_selected": best_run_id == train_run_id,
                        "train_digest": train_digest,
                        "validation_digest": validation_digest,
                        "n_candidates": int(len(runs)),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(summary_path), artifact_path="comparison")

    return {
        "best_run_id": best_run_id,
        "best_test_rmse": f"{best_test_rmse:.6f}",
        "best_test_mae": f"{best_test_mae:.6f}",
    }


def main(
    strategy: str,
    split_strategy: str,
    tracking_uri: str | None,
    register_model: bool,
    promote_model: bool,
    model_name: str,
    min_gain: float,
    trigger_source: str,
) -> None:
    root = Path(__file__).resolve().parent
    trigger = _resolve_trigger_source(trigger_source)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(run_name=f"retrain-cycle|{strategy}|{split_strategy}"):
        mlflow.log_param("cycle_trigger", trigger)
        mlflow.log_param("strategy", strategy)
        mlflow.log_param("split_strategy", split_strategy)
        mlflow.log_param("register_model", str(register_model).lower())
        mlflow.log_param("promote_model", str(promote_model).lower())
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("min_gain", f"{min_gain:.6f}")

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

        decision = _log_cycle_decision(train_run.run_id, trigger, strategy, split_strategy)

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

        if decision is not None:
            print(
                "comparison_best_run_id="
                f"{decision['best_run_id']} rmse={decision['best_test_rmse']} mae={decision['best_test_mae']}"
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
    parser.add_argument(
        "--trigger-source",
        choices=["auto", "manual", "scheduled", "ci"],
        default="auto",
        help="Retraining trigger category used for tracing and decision logging",
    )
    args = parser.parse_args()

    main(
        strategy=args.strategy,
        split_strategy=args.split_strategy,
        tracking_uri=args.tracking_uri,
        register_model=_to_bool(args.register_model),
        promote_model=_to_bool(args.promote_model),
        model_name=args.model_name,
        min_gain=args.min_gain,
        trigger_source=args.trigger_source,
    )
