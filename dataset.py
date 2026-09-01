#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["typer>=0.12", "polars>=1.43.2", "pyarrow>=25.0.0"]
# ///

"""Fetch and convert the shared electricity dataset.

Usage:
    uv run dataset.py download
    uv run dataset.py convert
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

DATASET_URL: str = (
    "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
)


def _dataset_dir() -> Path:
    """Locate shared/dataset/ by walking up, so the script runs from anywhere."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "shared" / "dataset").is_dir():
            return parent / "shared" / "dataset"
    return Path(__file__).resolve().parents[1] / "shared" / "dataset"


DATASET_DIR: Path = _dataset_dir()
RAW_FILE: Path = DATASET_DIR / "LD2011_2014.txt"
PARQUET_FILE: Path = DATASET_DIR / "LD2011_2014_kwh.parquet"
INTERVALS_PER_HOUR: int = 4
TIMESTAMP: str = "timestamp"

app = typer.Typer(help="Download and convert the reference electricity dataset.")


@app.command()
def download(
    url: Annotated[str, typer.Option(help="Source archive.")] = DATASET_URL,
    output: Annotated[Path, typer.Option(help="Where to write the extracted file.")] = RAW_FILE,
) -> None:
    """Download the UCI archive and extract the raw text file it contains."""
    typer.echo(f"downloading {url}")
    with (
        urllib.request.urlopen(url) as response,
        zipfile.ZipFile(io.BytesIO(response.read())) as archive,
    ):
        names = [name for name in archive.namelist() if name.endswith(".txt")]
        if not names:
            raise typer.BadParameter(f"no .txt file inside {url}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(archive.read(names[0]))
    typer.echo(f"{output} ({output.stat().st_size / 1024**2:.0f} MB)")


@app.command()
def convert(
    source: Annotated[Path, typer.Option(help="Raw text file to read.")] = RAW_FILE,
    output: Annotated[Path, typer.Option(help="Parquet file to write.")] = PARQUET_FILE,
) -> None:
    """Convert the raw file to Parquet, in kWh, one column per client."""
    if not source.exists():
        raise typer.BadParameter(f"{source} not found — run `download` first")

    clients = pl.read_csv(source, separator=";", n_rows=1, infer_schema_length=0).columns[1:]
    frame = pl.read_csv(
        source,
        separator=";",
        decimal_comma=True,
        try_parse_dates=True,
        schema_overrides=dict.fromkeys(clients, pl.Float64),
    )
    frame = frame.rename({frame.columns[0]: TIMESTAMP})
    frame = frame.with_columns((pl.col(clients) / INTERVALS_PER_HOUR).cast(pl.Float32)).with_columns(
        pl.col(TIMESTAMP).cast(pl.Datetime("ns"))
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output)
    typer.echo(
        f"{output} ({output.stat().st_size / 1024**2:.0f} MB) — "
        f"{frame.height} rows, {len(clients)} clients, kWh per 15 min"
    )


if __name__ == "__main__":
    app()
