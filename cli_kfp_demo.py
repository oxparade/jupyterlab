"""Typer CLI for Module 3 Kubeflow demo.

Usage:
  /opt/venvs/mlops/bin/python cli_kfp_demo.py compile
  /opt/venvs/mlops/bin/python cli_kfp_demo.py run --features full
"""

from __future__ import annotations

import os
from typing import Annotated

import typer
from kfp import Client, compiler

from pipeline_first import FeatureSet, feature_columns, first_pipeline

PIPELINE_YAML = "pipeline_first.yaml"
HOST = os.environ.get("KFP_HOST", "http://localhost:8081/pipeline")
NAMESPACE = os.environ.get("KFP_NAMESPACE", "kubeflow-user-example-com")
EXPERIMENT = os.environ.get("KFP_EXPERIMENT", "electricity")

app = typer.Typer(help="Compile or run the first KFP pipeline.")


@app.command()
def compile() -> None:
    compiler.Compiler().compile(first_pipeline, PIPELINE_YAML)
    typer.echo(f"{PIPELINE_YAML} compiled OK")


@app.command()
def run(
    features: Annotated[FeatureSet, typer.Option(help="Feature set to build.")] = FeatureSet.FULL,
) -> None:
    token = os.environ.get("KF_TOKEN") or os.environ.get("KFP_TOKEN")
    if not token:
        raise typer.BadParameter("Set KF_TOKEN or KFP_TOKEN before running.")

    client = Client(host=HOST, existing_token=token, namespace=NAMESPACE)
    run_result = client.create_run_from_pipeline_func(
        first_pipeline,
        arguments={"features": feature_columns(features)},
        experiment_name=EXPERIMENT,
    )
    typer.echo(f"run submitted ({features.value}): {run_result.run_id}")


if __name__ == "__main__":
    app()
