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


def main(model_name: str) -> None:
    client = MlflowClient()

    champion_before = _alias_version(client, model_name, "champion")
    challenger_version = _alias_version(client, model_name, "challenger")

    champion_uri = f"models:/{model_name}@champion"
    champion_model_before = mlflow.pyfunc.load_model(champion_uri)
    logger.info("Loaded champion before promotion from %s", champion_uri)

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
    args = parser.parse_args()
    main(model_name=args.model_name)
