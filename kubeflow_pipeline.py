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
DEFAULT_EVIDENTLY_API_URL = os.environ.get("EVIDENTLY_API_URL", "")
DEFAULT_EVIDENTLY_PROJECT_NAME = os.environ.get("EVIDENTLY_PROJECT_NAME", "electricity_forecaster")
DEFAULT_EVIDENTLY_SECRET = os.environ.get("EVIDENTLY_SECRET", "")


def _build_env(mlflow_tracking_uri: str, evidently_api_url: str = "", evidently_secret: str = "") -> dict[str, str]:
    env = os.environ.copy()
    if mlflow_tracking_uri:
        env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        if mlflow_tracking_uri.startswith("https://"):
            env["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    if evidently_api_url:
        env["EVIDENTLY_API_URL"] = evidently_api_url
    if evidently_secret:
        env["EVIDENTLY_SECRET"] = evidently_secret
    return env


def _run_script(
    source_dir: str,
    script_name: str,
    args: list[str],
    mlflow_tracking_uri: str,
    evidently_api_url: str = "",
    evidently_secret: str = "",
) -> None:
    script_path = Path(source_dir) / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    subprocess.run(
        [sys.executable, str(script_path), *args],
        check=True,
        env=_build_env(mlflow_tracking_uri, evidently_api_url=evidently_api_url, evidently_secret=evidently_secret),
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
            if mlflow_tracking_uri.startswith("https://"):
                env["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
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
    training_summary_path: Input[Dataset],
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
            if mlflow_tracking_uri.startswith("https://"):
                env["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
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
        "training_summary_path": training_summary_path.path,
    }
    Path(registry_summary_path.path).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@dsl.component(base_image=COMPONENT_IMAGE)
def train_model(
    source_dir: str,
    train_path: Input[Dataset],
    validation_path: Input[Dataset],
    strategy: str,
    mlflow_tracking_uri: str,
    training_summary_path: Output[Dataset],
) -> None:
    """Run train_mlflow.py as a dedicated training node in the DAG."""

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
            if mlflow_tracking_uri.startswith("https://"):
                env["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        subprocess.run([sys.executable, str(script_path), *args], check=True, env=env)

    run_script(
        script_name="train_mlflow.py",
        args=[
            "--data",
            train_path.path,
            "--validation",
            validation_path.path,
        ],
    )

    summary = {
        "strategy": strategy,
        "train_path": train_path.path,
        "validation_path": validation_path.path,
        "note": "Detailed training metrics and model artifacts are logged by train_mlflow.py",
    }
    Path(training_summary_path.path).write_text(
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
            if mlflow_tracking_uri.startswith("https://"):
                env["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
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
    evidently_api_url: str,
    evidently_project_name: str,
    evidently_secret: str,
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
            if mlflow_tracking_uri.startswith("https://"):
                env["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        if evidently_api_url:
            env["EVIDENTLY_API_URL"] = evidently_api_url
        if evidently_project_name:
            env["EVIDENTLY_PROJECT_NAME"] = evidently_project_name
        if evidently_secret:
            env["EVIDENTLY_SECRET"] = evidently_secret
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
            "--evidently-api-url",
            evidently_api_url,
            "--evidently-project-name",
            evidently_project_name,
            "--evidently-secret",
            evidently_secret,
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
    evidently_api_url: str = DEFAULT_EVIDENTLY_API_URL,
    evidently_project_name: str = DEFAULT_EVIDENTLY_PROJECT_NAME,
    evidently_secret: str = DEFAULT_EVIDENTLY_SECRET,
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

    trained = train_model(
        source_dir=source_dir,
        train_path=data.outputs["train_path"],
        validation_path=data.outputs["validation_path"],
        strategy=strategy,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    registered = register_candidates(
        source_dir=source_dir,
        train_path=data.outputs["train_path"],
        validation_path=data.outputs["validation_path"],
        training_summary_path=trained.outputs["training_summary_path"],
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
        evidently_api_url=evidently_api_url,
        evidently_project_name=evidently_project_name,
        evidently_secret=evidently_secret,
    )

    govern_champion(
        source_dir=source_dir,
        model_name=model_name,
        min_gain=min_gain,
        dry_run=governance_dry_run,
        evaluation_summary_path=evaluation.outputs["evaluation_summary_path"],
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


def visualize_pipeline() -> str:
    """Generate a Mermaid diagram of the pipeline DAG."""
    diagram = """graph TD
    A["📥 Import Raw Dataset<br/>(S3)"] --> B["🔧 Prepare Data<br/>(prep.py)"]
    B --> |train.parquet| C["🤖 Train Model<br/>(train_mlflow.py)"]
    B --> |validation.parquet| C
    C --> D["📋 Register Candidates<br/>(register.py)"]
    B --> |test.parquet| E["🎯 Evaluate Champion<br/>(predict_mlflow.py)"]
    D --> E
    E --> F["✅ Governance Decision<br/>(govern.py)"]
    
    style A fill:#90EE90
    style B fill:#87CEEB
    style C fill:#FFB6C1
    style D fill:#DDA0DD
    style E fill:#F0E68C
    style F fill:#FFA07A
    
    A -.MLflow Tracking.-> MLF["🔍 MLflow Server<br/>Train/Register/Log"]
    B -.logs to.-> MLF
    C -.logs to.-> MLF
    D -.logs to.-> MLF
    E -.logs to.-> MLF
    F -.logs to.-> MLF
    
    style MLF fill:#FFE4B5"""
    return diagram


def compile_pipeline(output_path: Path) -> None:
    compiler.Compiler().compile(
        pipeline_func=electricity_forecaster_pipeline,
        package_path=str(output_path),
    )
    print(f"\n✅ Pipeline compiled to {output_path}")
    print("\n📊 DAG Visualization:\n")
    print(visualize_pipeline())


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
