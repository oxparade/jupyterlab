#!/usr/bin/env python3
"""Talk to in-cluster Garage S3 from the VM host.

Examples:
  /opt/venvs/mlops/bin/python garage_s3.py ls
  /opt/venvs/mlops/bin/python garage_s3.py upload /tmp/LD2011_2014_kwh.parquet --bucket models --key datasets/LD2011_2014_kwh.parquet
"""

from __future__ import annotations

import base64
import contextlib
import socket
import subprocess
import time
from pathlib import Path
from typing import Annotated

import boto3
import typer

NS_GARAGE = "garage"
NS_CREDS = "kubeflow-user-example-com"
SECRET = "garage-s3-creds"
REGION = "garage"
DEFAULT_BUCKET = "models"

app = typer.Typer(add_completion=False)


def _kubectl(*args: str) -> str:
    return subprocess.run(["kubectl", *args], check=True, capture_output=True, text=True).stdout.strip()


def _creds() -> tuple[str, str]:
    template = "{.data.AWS_ACCESS_KEY_ID}|{.data.AWS_SECRET_ACCESS_KEY}"
    raw = _kubectl("get", "secret", SECRET, "-n", NS_CREDS, "-o", f"jsonpath={template}")
    access_key, secret_key = (base64.b64decode(value).decode() for value in raw.split("|", 1))
    return access_key, secret_key


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(port: int, proc: subprocess.Popen[str], timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("garage port-forward exited early")
        with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 0.5):
            return
        time.sleep(0.25)
    raise TimeoutError(f"garage port-forward not ready on port {port}")


@contextlib.contextmanager
def s3_client():
    port = _free_port()
    proc = subprocess.Popen(
        ["kubectl", "port-forward", "-n", NS_GARAGE, "svc/garage", f"{port}:3900"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _wait_port(port, proc)
        access_key, secret_key = _creds()
        client = boto3.client(
            "s3",
            endpoint_url=f"http://127.0.0.1:{port}",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=REGION,
            config=boto3.session.Config(s3={"addressing_style": "path"}),
        )
        yield client
    finally:
        proc.terminate()


@app.command()
def ls(bucket: Annotated[str | None, typer.Argument()] = None) -> None:
    with s3_client() as s3:
        if bucket is None:
            for entry in s3.list_buckets().get("Buckets", []):
                typer.echo(entry["Name"])
            return

        paginator = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket)
        empty = True
        for page in paginator:
            for obj in page.get("Contents", []):
                empty = False
                typer.echo(f"{obj['Size']:>12}  {obj['Key']}")
        if empty:
            typer.echo(f"(no objects in {bucket})")


@app.command()
def upload(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    bucket: Annotated[str, typer.Option()] = DEFAULT_BUCKET,
    key: Annotated[str | None, typer.Option()] = None,
) -> None:
    target_key = key or f"datasets/{file.name}"
    with s3_client() as s3:
        s3.upload_file(str(file), bucket, target_key)
    typer.echo(f"uploaded -> s3://{bucket}/{target_key}")


if __name__ == "__main__":
    app()
