"""TP02 — Charge @champion, promeut @challenger, rollback, puis pose des alias multi-env."""

from __future__ import annotations

import argparse
import logging

import mlflow
from mlflow import MlflowClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _alias_version(client: MlflowClient, model_name: str, alias: str) -> str:
    return str(client.get_model_version_by_alias(model_name, alias).version)


def _tag_bool(tags: dict[str, str], key: str, default: bool = False) -> bool:
    value = tags.get(key)
    if value is None:
        return default
    return value.strip().lower() == "true"


def _tag_float(tags: dict[str, str], key: str) -> float:
    if key not in tags:
        raise KeyError(f"Required tag '{key}' is missing")
    return float(tags[key])


def main(model_name: str, min_gain: float) -> None:
    client = MlflowClient()

    champion_before_mv = client.get_model_version_by_alias(model_name, "champion")
    challenger_mv = client.get_model_version_by_alias(model_name, "challenger")
    champion_before = str(champion_before_mv.version)
    challenger_version = str(challenger_mv.version)

    champion_rmse = _tag_float(champion_before_mv.tags, "rmse")
    challenger_rmse = _tag_float(challenger_mv.tags, "rmse")
    gain = (champion_rmse - challenger_rmse) / champion_rmse
    challenger_validated = _tag_bool(challenger_mv.tags, "validated", default=False)
    challenger_passed = _tag_bool(challenger_mv.tags, "passed_validation", default=False)

    decision_ok = challenger_validated and challenger_passed and gain >= min_gain
    if decision_ok:
        client.set_model_version_tag(model_name, challenger_version, "promotion_status", "accepted")
        client.set_model_version_tag(
            model_name,
            challenger_version,
            "decision_note",
            f"accepted: gain={gain:.2%} >= {min_gain:.2%}",
        )
    else:
        reasons: list[str] = []
        if not challenger_validated:
            reasons.append("validated=false")
        if not challenger_passed:
            reasons.append("passed_validation=false")
        if gain < min_gain:
            reasons.append(f"gain {gain:.2%} < {min_gain:.2%}")
        reason = " ; ".join(reasons)
        client.set_model_version_tag(model_name, challenger_version, "promotion_status", "rejected")
        client.set_model_version_tag(model_name, challenger_version, "rejected_reason", reason)
        client.set_model_version_tag(model_name, challenger_version, "decision_note", f"rejected: {reason}")

    champion_uri = f"models:/{model_name}@champion"
    champion_model_before = mlflow.pyfunc.load_model(champion_uri)
    logger.info("Loaded champion before promotion from %s", champion_uri)

    if decision_ok:
        client.set_registered_model_alias(model_name, "champion", challenger_version)
    champion_after = _alias_version(client, model_name, "champion")

    champion_model_after = mlflow.pyfunc.load_model(champion_uri)
    logger.info("Loaded champion after promotion from %s", champion_uri)

    client.set_registered_model_alias(model_name, "champion", champion_before)
    champion_rollback = _alias_version(client, model_name, "champion")

    for alias in ("prod-eu", "prod-us", "shadow"):
        client.set_registered_model_alias(model_name, alias, challenger_version)

    print("Alias transitions:")
    print(f"- champion initial: v{champion_before}")
    print(f"- challenger: v{challenger_version}")
    print(f"- quality gate: {'accepted' if decision_ok else 'rejected'}")
    print(f"- gain vs champion: {gain:.2%} (minimum required: {min_gain:.2%})")
    print(f"- champion after promotion: v{champion_after}")
    print(f"- champion after rollback: v{champion_rollback}")
    print("- multi-env aliases -> challenger version:")
    for alias in ("prod-eu", "prod-us", "shadow"):
        print(f"  - {alias}: v{_alias_version(client, model_name, alias)}")

    print("- loaded model objects:")
    print(f"  - before promotion: {type(champion_model_before).__name__}")
    print(f"  - after promotion: {type(champion_model_after).__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP02 promote: alias promotion + rollback")
    parser.add_argument("--model-name", default="electricity_forecaster")
    parser.add_argument("--min-gain", type=float, default=0.02)
    args = parser.parse_args()
    main(model_name=args.model_name, min_gain=args.min_gain)
