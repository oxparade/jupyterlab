"""Configuration partagée pour le TP MLflow (features, splits, alphas)."""

from __future__ import annotations

import datetime
from enum import StrEnum
from pathlib import Path

INTERVALS_PER_HOUR: int = 4
STEPS_PER_DAY: int = 96

DATA_DIR: Path = Path("data")
EXPERIMENT: str = "electricity_consumption_forecasting"
TARGET: str = "consumption"


def _shared_dataset() -> Path:
    """Locate shared/dataset/ by walking up from this file."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "shared" / "dataset" / "LD2011_2014_kwh.parquet"
        if candidate.exists():
            return candidate
    return Path("shared/dataset/LD2011_2014_kwh.parquet")


DATASET_FILE: Path = _shared_dataset()


class ModellingStrategy(StrEnum):
    SHORT_MEMORY = "short_memory"
    SEASONALITY = "seasonality"
    TENDENCY = "tendency"
    MIXED = "mixed"
    ALL = "all"


MODELLING_FEATURES: dict[ModellingStrategy, list[str]] = {
    ModellingStrategy.SHORT_MEMORY: ["lag_1d"],
    ModellingStrategy.SEASONALITY: ["lag_7d", "lag_30d"],
    ModellingStrategy.TENDENCY: ["rolling_mean_7d", "rolling_mean_30d"],
    ModellingStrategy.MIXED: ["lag_1d", "lag_7d", "lag_30d", "rolling_mean_30d"],
    ModellingStrategy.ALL: [
        "lag_1d",
        "lag_7d",
        "lag_30d",
        "lag_365d",
        "rolling_mean_7d",
        "rolling_mean_30d",
    ],
}


class SplitStrategy(StrEnum):
    FULL_HISTORY = "full_history"
    RECENT_HISTORY = "recent_history"


DATASET_SPLIT_DATES: dict[
    SplitStrategy, dict[str, tuple[datetime.date, datetime.date]]
] = {
    SplitStrategy.FULL_HISTORY: {
        "train": (datetime.date(2011, 1, 1), datetime.date(2012, 12, 31)),
        "validation": (datetime.date(2013, 1, 1), datetime.date(2013, 12, 31)),
        "test": (datetime.date(2014, 1, 1), datetime.date(2014, 12, 31)),
    },
    SplitStrategy.RECENT_HISTORY: {
        "train": (datetime.date(2013, 1, 1), datetime.date(2013, 12, 31)),
        "validation": (datetime.date(2014, 1, 1), datetime.date(2014, 5, 31)),
        "test": (datetime.date(2014, 6, 1), datetime.date(2014, 12, 31)),
    },
}

RIDGE_ALPHAS: list[float] = [0.0, 1.0, 1e3, 1e9]
