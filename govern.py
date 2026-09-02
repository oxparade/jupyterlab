"""Governance workflow: compare validated versions and decide whether one becomes champion."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _tag_bool(tags: dict[str, str], key: str, default: bool = False) -> bool:
    value = tags.get(key)
    if value is None:
        return default
    return value.strip().lower() == "true"


def _tag_float(tags: dict[str, str], key: str) -> float:
    if key not in tags:
        raise KeyError(f"Required tag '{key}' is missing")
    return float(tags[key])


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _validated(mv) -> bool:
    tags = mv.tags
    return (
        _tag_bool(tags, "validated", default=False)
        and _tag_bool(tags, "passed_validation", default=False)
    ) or tags.get("validation_status", "").strip().lower() == "passed"


def _rmse(mv) -> float:
    return _tag_float(mv.tags, "rmse")


def _mae(mv) -> float:
    return _tag_float(mv.tags, "mae")


def _best_validated_versions(client: MlflowClient, model_name: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    validated_versions = [mv for mv in versions if _validated(mv)]
    if not validated_versions:
        raise RuntimeError(f"No validated versions found for model '{model_name}'")
    validated_versions.sort(key=lambda mv: (_rmse(mv), _mae(mv), int(mv.version)))
    return validated_versions


def main(model_name: str, min_gain: float, dry_run: bool) -> None:
    client = MlflowClient()
    champion_mv = client.get_model_version_by_alias(model_name, "champion")
    champion_version = str(champion_mv.version)
    champion_rmse = _rmse(champion_mv)
    champion_mae = _mae(champion_mv)

    validated_versions = _best_validated_versions(client, model_name)
    rows: list[dict[str, object]] = []
    best_mv = validated_versions[0]
    best_version = str(best_mv.version)
    best_rmse = _rmse(best_mv)
    best_mae = _mae(best_mv)
    gain = (champion_rmse - best_rmse) / champion_rmse
    mae_gain = (champion_mae - best_mae) / champion_mae

    decision_ok = best_version != champion_version and gain >= min_gain
    decision_reason = (
        f"accepted: validated best v{best_version} beats champion v{champion_version} "
        f"with rmse_gain={gain:.2%} >= {min_gain:.2%}"
        if decision_ok
        else (
            "rejected: "
            + (
                "best already is current champion"
                if best_version == champion_version
                else f"rmse_gain={gain:.2%} < {min_gain:.2%}"
            )
        )
    )

    for rank, mv in enumerate(validated_versions, start=1):
        row = {
            "rank": rank,
            "version": str(mv.version),
            "rmse": _rmse(mv),
            "mae": _mae(mv),
            "validated": _validated(mv),
            "is_best": str(mv.version) == best_version,
            "is_current_champion": str(mv.version) == champion_version,
            "validation_status": mv.tags.get("validation_status", "passed"),
        }
        rows.append(row)
        client.set_model_version_tag(model_name, str(mv.version), "governance_rank", str(rank))
        client.set_model_version_tag(model_name, str(mv.version), "governance_validated", "true")
        client.set_model_version_tag(model_name, str(mv.version), "governance_is_best", str(row["is_best"]).lower())

    client.set_model_version_tag(model_name, best_version, "governance_selected", "true")
    client.set_model_version_tag(model_name, best_version, "governance_decision", "accepted" if decision_ok else "rejected")
    client.set_model_version_tag(model_name, best_version, "governance_reason", decision_reason)
    client.set_model_version_tag(model_name, best_version, "governance_gain", f"{gain:.6f}")
    client.set_model_version_tag(model_name, best_version, "governance_mae_gain", f"{mae_gain:.6f}")
    client.set_model_version_tag(model_name, best_version, "governance_compared_champion", champion_version)

    if decision_ok and not dry_run:
        client.set_registered_model_alias(model_name, "challenger", best_version)
        client.set_registered_model_alias(model_name, "champion", best_version)
        logger.info("Promoted validated version v%s to champion", best_version)
    else:
        client.set_registered_model_alias(model_name, "challenger", best_version)
        logger.info("Kept champion at v%s", champion_version)

    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = Path(temp_dir) / "validated_versions.csv"
        pd.DataFrame(rows).to_csv(report_path, index=False)

        summary_path = Path(temp_dir) / "governance_decision.json"
        summary_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "champion_version": champion_version,
                    "best_validated_version": best_version,
                    "decision": "accepted" if decision_ok else "rejected",
                    "reason": decision_reason,
                    "rmse_gain": gain,
                    "mae_gain": mae_gain,
                    "dry_run": dry_run,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"champion={champion_version} best_validated={best_version} rmse_gain={gain:.2%} mae_gain={mae_gain:.2%}")
        print(f"decision={'accepted' if decision_ok else 'rejected'}")
        print(decision_reason)
        print(pd.DataFrame(rows).to_string(index=False))

        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.log_param("governance_model_name", model_name)
            mlflow.log_param("governance_champion_version", champion_version)
            mlflow.log_param("governance_best_validated_version", best_version)
            mlflow.log_param("governance_min_gain", f"{min_gain:.6f}")
            mlflow.log_param("governance_decision", "accepted" if decision_ok else "rejected")
            mlflow.log_artifact(str(report_path), artifact_path="governance")
            mlflow.log_artifact(str(summary_path), artifact_path="governance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Govern validated MLflow versions")
    parser.add_argument("--model-name", default="electricity_forecaster")
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--dry-run", default="true", help="Do not change aliases")
    args = parser.parse_args()
    main(model_name=args.model_name, min_gain=args.min_gain, dry_run=_to_bool(args.dry_run))