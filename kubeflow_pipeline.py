"""Kubeflow Pipelines v2 scaffold for the electricity forecasting lab.

This pipeline is intentionally thin: each step reuses the existing project scripts.
It assumes the container image already contains:

- this repository source code mounted or baked into the image;
- the runtime dependencies required by the scripts;
- access to the MLflow tracking server through MLFLOW_TRACKING_URI.

The goal is to keep the Kubeflow graph explicit while preserving the current MLflow/DVC
codebase almost untouched.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Output

COMPONENT_IMAGE = os.environ.get("KUBEFLOW_COMPONENT_IMAGE", "jupyterlab-kfp:0.1.0")
DEFAULT_SOURCE_DIR = os.environ.get("KUBEFLOW_SOURCE_DIR", "/workspace/jupyterlab")
DEFAULT_RAW_DATASET = os.environ.get(
    "KUBEFLOW_RAW_DATASET",
    "s3://models/datasets/LD2011_2014_kwh.parquet",
)


def _build_env(mlflow_tracking_uri: str) -> dict[str, str]:
    env = os.environ.copy()
    if mlflow_tracking_uri:
        env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    return env


def _run_script(source_dir: str, script_name: str, args: list[str], mlflow_tracking_uri: str) -> None:
    script_path = Path(source_dir) / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    subprocess.run(
        [sys.executable, str(script_path), *args],
        check=True,
        env=_build_env(mlflow_tracking_uri),
    )


@dsl.component(base_image=COMPONENT_IMAGE)
def prepare_data(
    source_dir: str,
    raw_dataset: Input[Dataset],
    strategy: str,
    split_strategy: str,
    mlflow_tracking_uri: str,
    train_path: Output[Dataset],
    validation_path: Output[Dataset],
    test_path: Output[Dataset],
) -> None:
    """Run prep.py and expose the split parquet files as KFP artifacts."""

    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    def run_script(script_name: str, args: list[str]) -> None:
        script_path = Path(source_dir) / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        env = os.environ.copy()
        if mlflow_tracking_uri:
            env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        subprocess.run([sys.executable, str(script_path), *args], check=True, env=env)

    with tempfile.TemporaryDirectory() as temp_dir:
        run_script(
            script_name="prep.py",
            args=[
                "--input",
                raw_dataset.path,
                "--output",
                temp_dir,
                "--strategy",
                strategy,
                "--split-strategy",
                split_strategy,
            ],
        )

        produced = {
            "train.parquet": train_path.path,
            "validation.parquet": validation_path.path,
            "test.parquet": test_path.path,
        }
        for filename, destination in produced.items():
            src = Path(temp_dir) / filename
            if not src.exists():
                raise FileNotFoundError(f"Expected output missing: {src}")
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, destination)


@dsl.component(base_image=COMPONENT_IMAGE)
def register_candidates(
    source_dir: str,
    train_path: Input[Dataset],
    validation_path: Input[Dataset],
    model_name: str,
    first_strategy: str,
    second_strategy: str,
    mlflow_tracking_uri: str,
    registry_summary_path: Output[Dataset],
) -> None:
    """Register two candidate strategies and write a tiny summary artifact."""

    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    def run_script(script_name: str, args: list[str]) -> None:
        script_path = Path(source_dir) / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        env = os.environ.copy()
        if mlflow_tracking_uri:
            env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        subprocess.run([sys.executable, str(script_path), *args], check=True, env=env)

    run_script(
        script_name="register.py",
        args=[
            "--train",
            train_path.path,
            "--validation",
            validation_path.path,
            "--model-name",
            model_name,
            "--first-strategy",
            first_strategy,
            "--second-strategy",
            second_strategy,
        ],
    )

    summary = {
        "model_name": model_name,
        "first_strategy": first_strategy,
        "second_strategy": second_strategy,
        "train_path": train_path.path,
        "validation_path": validation_path.path,
    }
    Path(registry_summary_path.path).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@dsl.component(base_image=COMPONENT_IMAGE)
def govern_champion(
    source_dir: str,
    model_name: str,
    min_gain: float,
    dry_run: str,
    evaluation_summary_path: Input[Dataset],
    mlflow_tracking_uri: str,
    governance_summary_path: Output[Dataset],
) -> None:
    """Run governance decision and persist a simple decision artifact."""

    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    def run_script(script_name: str, args: list[str]) -> None:
        script_path = Path(source_dir) / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        env = os.environ.copy()
        if mlflow_tracking_uri:
            env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        subprocess.run([sys.executable, str(script_path), *args], check=True, env=env)

    run_script(
        script_name="govern.py",
        args=[
            "--model-name",
            model_name,
            "--min-gain",
            str(min_gain),
            "--dry-run",
            dry_run,
        ],
    )

    summary = {
        "model_name": model_name,
        "min_gain": min_gain,
        "dry_run": dry_run,
        "evaluation_summary_path": evaluation_summary_path.path,
        "decision": "logged in MLflow registry tags by govern.py",
    }
    Path(governance_summary_path.path).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@dsl.component(base_image=COMPONENT_IMAGE)
def evaluate_champion(
    source_dir: str,
    model_name: str,
    test_path: Input[Dataset],
    reference_path: Input[Dataset],
    registry_summary_path: Input[Dataset],
    mlflow_tracking_uri: str,
    evaluation_summary_path: Output[Dataset],
) -> None:
    """Evaluate the promoted champion through predict_mlflow.py."""

    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    def run_script(script_name: str, args: list[str]) -> None:
        script_path = Path(source_dir) / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        env = os.environ.copy()
        if mlflow_tracking_uri:
            env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        subprocess.run([sys.executable, str(script_path), *args], check=True, env=env)

    model_uri = f"models:/{model_name}@champion"
    run_script(
        script_name="predict_mlflow.py",
        args=[
            "--model",
            model_uri,
            "--data",
            test_path.path,
            "--reference-data",
            reference_path.path,
        ],
    )

    summary = {
        "model_uri": model_uri,
        "test_path": test_path.path,
        "reference_path": reference_path.path,
        "registry_summary_path": registry_summary_path.path,
        "note": "Detailed metrics are logged in MLflow by predict_mlflow.py",
    }
    Path(evaluation_summary_path.path).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@dsl.pipeline(name="electricity-forecaster-kubeflow")
def electricity_forecaster_pipeline(
    source_dir: str = DEFAULT_SOURCE_DIR,
    raw_dataset_uri: str = DEFAULT_RAW_DATASET,
    mlflow_tracking_uri: str = "",
    model_name: str = "electricity_forecaster",
    first_strategy: str = "short_memory",
    second_strategy: str = "mixed",
    strategy: str = "mixed",
    split_strategy: str = "full_history",
    min_gain: float = 0.02,
    governance_dry_run: str = "false",
) -> None:
    """Kubeflow-native orchestration for the current MLflow/DVC project."""

    source = dsl.importer(
        artifact_uri=raw_dataset_uri,
        artifact_class=Dataset,
        reimport=False,
    )

    data = prepare_data(
        source_dir=source_dir,
        raw_dataset=source.output,
        strategy=strategy,
        split_strategy=split_strategy,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    registered = register_candidates(
        source_dir=source_dir,
        train_path=data.outputs["train_path"],
        validation_path=data.outputs["validation_path"],
        model_name=model_name,
        first_strategy=first_strategy,
        second_strategy=second_strategy,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    evaluation = evaluate_champion(
        source_dir=source_dir,
        model_name=model_name,
        test_path=data.outputs["test_path"],
        reference_path=data.outputs["train_path"],
        registry_summary_path=registered.outputs["registry_summary_path"],
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    govern_champion(
        source_dir=source_dir,
        model_name=model_name,
        min_gain=min_gain,
        dry_run=governance_dry_run,
        evaluation_summary_path=evaluation.outputs["evaluation_summary_path"],
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


def compile_pipeline(output_path: Path) -> None:
    compiler.Compiler().compile(
        pipeline_func=electricity_forecaster_pipeline,
        package_path=str(output_path),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile the Kubeflow pipeline spec")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kubeflow_pipeline.yaml"),
        help="Path of the compiled pipeline spec",
    )
    args = parser.parse_args()
    compile_pipeline(args.output)
