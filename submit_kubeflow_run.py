"""Submit the Kubeflow pipeline to an existing KFP deployment on the VM.

Usage (env-based):
    export KFP_HOST="https://<your-kfp-endpoint>"
    export KFP_TOKEN="<optional-bearer-token>"
    /opt/venvs/mlops/bin/python submit_kubeflow_run.py --compile-if-missing
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from kfp import Client
from kfp_server_api.exceptions import ApiException


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_host(cli_value: str | None) -> str:
    host = (cli_value or os.environ.get("KFP_HOST") or "http://localhost:8081/pipeline").strip()
    if not host:
        raise ValueError(
            "KFP host is required. Set --host or KFP_HOST (for example: https://pipelines.<domain>)."
        )
    return host


def _maybe_compile(pipeline_package: Path) -> None:
    from kubeflow_pipeline import compile_pipeline

    pipeline_package.parent.mkdir(parents=True, exist_ok=True)
    compile_pipeline(pipeline_package)


def _build_arguments(args: argparse.Namespace) -> dict[str, object]:
    return {
        "source_dir": args.source_dir,
        "raw_dataset_uri": args.raw_dataset_uri,
        "mlflow_tracking_uri": args.mlflow_tracking_uri,
        "model_name": args.model_name,
        "first_strategy": args.first_strategy,
        "second_strategy": args.second_strategy,
        "strategy": args.strategy,
        "split_strategy": args.split_strategy,
        "min_gain": args.min_gain,
        "governance_dry_run": args.governance_dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Kubeflow run for electricity_forecaster_pipeline")
    parser.add_argument("--host", default=None, help="KFP API host; fallback to KFP_HOST env var")
    parser.add_argument(
        "--token",
        default=os.environ.get("KFP_TOKEN") or os.environ.get("KF_TOKEN"),
        help="Bearer token; fallback to KFP_TOKEN then KF_TOKEN env vars",
    )
    parser.add_argument("--cookies", default=os.environ.get("KFP_COOKIES"), help="Cookie header string (authservice_session=...)")
    parser.add_argument("--namespace", default=os.environ.get("KFP_NAMESPACE", "kubeflow"))
    parser.add_argument("--experiment-name", default=os.environ.get("KFP_EXPERIMENT", "electricity-forecaster"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--pipeline-package", type=Path, default=Path("kubeflow_pipeline.yaml"))
    parser.add_argument("--compile-if-missing", action="store_true")
    parser.add_argument("--enable-caching", default="true", help="true/false")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification")
    parser.add_argument("--ssl-ca-cert", default=os.environ.get("KFP_SSL_CA_CERT"), help="Custom CA certificate path")

    parser.add_argument("--source-dir", default=os.environ.get("KUBEFLOW_SOURCE_DIR", "/workspace/jupyterlab"))
    parser.add_argument(
        "--raw-dataset-uri",
        default=os.environ.get("KUBEFLOW_RAW_DATASET", "s3://models/datasets/LD2011_2014_kwh.parquet"),
    )
    parser.add_argument("--mlflow-tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", ""))
    parser.add_argument("--model-name", default="electricity_forecaster")
    parser.add_argument("--first-strategy", default="short_memory")
    parser.add_argument("--second-strategy", default="mixed")
    parser.add_argument("--strategy", default="mixed")
    parser.add_argument("--split-strategy", default="full_history")
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--governance-dry-run", default="false", help="true/false")

    args = parser.parse_args()

    host = _resolve_host(args.host)
    pipeline_package = args.pipeline_package

    if args.compile_if_missing and not pipeline_package.exists():
        _maybe_compile(pipeline_package)

    if not pipeline_package.exists():
        raise FileNotFoundError(
            f"Pipeline package not found: {pipeline_package}. Compile first with kubeflow_pipeline.py or use --compile-if-missing."
        )

    run_name = args.run_name or f"electricity-forecaster-{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    client = Client(
        host=host,
        existing_token=(args.token or None),
        cookies=(args.cookies or None),
        namespace=args.namespace,
        verify_ssl=not args.insecure,
        ssl_ca_cert=args.ssl_ca_cert,
    )

    pipeline_arguments = _build_arguments(args)
    try:
        submitted = client.create_run_from_pipeline_package(
            pipeline_file=str(pipeline_package),
            arguments=pipeline_arguments,
            run_name=run_name,
            experiment_name=args.experiment_name,
            namespace=args.namespace,
            enable_caching=_to_bool(args.enable_caching),
        )
    except ApiException as exc:
        if exc.status == 401:
            raise RuntimeError(
                "Unauthorized by KFP API. Provide identity via --token (KFP_TOKEN) or --cookies (KFP_COOKIES), "
                "and use a profile namespace such as kubeflow-user-example-com."
            ) from exc
        raise

    run_id = getattr(submitted, "run_id", None)
    details = {
        "host": host,
        "namespace": args.namespace,
        "experiment_name": args.experiment_name,
        "run_name": run_name,
        "run_id": run_id,
        "pipeline_package": str(pipeline_package),
        "verify_ssl": not args.insecure,
        "ssl_ca_cert": args.ssl_ca_cert,
        "auth_mode": "token" if args.token else ("cookies" if args.cookies else "anonymous"),
        "arguments": pipeline_arguments,
    }
    print(json.dumps(details, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
