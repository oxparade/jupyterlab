"""Helpers TP02 pour runs MLflow imbriqués et robustes sous `mlflow run`."""

from __future__ import annotations

import os
from contextlib import nullcontext

import mlflow


def step_run(run_name: str, nested: bool = True):
    """Retourne un context manager de run compatible script local + MLflow Projects.

    - si un run actif existe: ouvre un run nested (ou no-op si nested=False)
    - sinon, si MLFLOW_RUN_ID existe: rattache le run parent de project
    - sinon: démarre un run normal
    """
    active_run = mlflow.active_run()
    project_run_id = os.getenv("MLFLOW_RUN_ID")

    if active_run is not None:
        if nested:
            return mlflow.start_run(run_name=run_name, nested=True)
        return nullcontext()

    if project_run_id:
        return mlflow.start_run()

    return mlflow.start_run(run_name=run_name)
