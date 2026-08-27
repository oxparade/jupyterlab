from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Mapping

from .settings import get_settings


class FeatureNotFoundError(KeyError):
    """Raised when a client or timestamp cannot be resolved in the feature store."""


def _timestamp_key(timestamp: datetime) -> str:
    normalized = timestamp.replace(second=0, microsecond=0, tzinfo=None)
    return normalized.isoformat(timespec="minutes")


DEFAULT_FEATURE_STORE: dict[str, dict[str, dict[str, float]]] = {
    "MT_001": {
        "2014-07-01T00:00": {
            "lag_1d": 6.2,
            "lag_7d": 6.0,
            "lag_30d": 5.8,
            "lag_365d": 5.5,
            "rolling_mean_7d": 6.1,
            "rolling_mean_30d": 5.9,
        },
        "2014-07-01T00:15": {
            "lag_1d": 6.4,
            "lag_7d": 6.1,
            "lag_30d": 5.9,
            "lag_365d": 5.6,
            "rolling_mean_7d": 6.2,
            "rolling_mean_30d": 6.0,
        },
    },
    "MT_002": {
        "2014-07-01T00:00": {
            "lag_1d": 3.8,
            "lag_7d": 3.7,
            "lag_30d": 3.6,
            "lag_365d": 3.4,
            "rolling_mean_7d": 3.75,
            "rolling_mean_30d": 3.65,
        }
    },
    "MT_003": {
        "2014-07-01T00:00": {
            "lag_1d": 9.1,
            "lag_7d": 8.9,
            "lag_30d": 8.6,
            "lag_365d": 8.2,
            "rolling_mean_7d": 9.0,
            "rolling_mean_30d": 8.7,
        }
    },
}


@dataclass(frozen=True)
class InMemoryFeatureStore:
    data: Mapping[str, Mapping[str, Mapping[str, float]]]

    def get_features(self, individual: str, timestamp: datetime) -> dict[str, float]:
        timestamp_key = _timestamp_key(timestamp)
        try:
            feature_row = self.data[individual][timestamp_key]
        except KeyError as exc:
            raise FeatureNotFoundError(
                f"No features available for individual={individual!r} at timestamp={timestamp_key!r}."
            ) from exc

        required_features = get_settings().required_features
        missing = [name for name in required_features if name not in feature_row]
        if missing:
            raise FeatureNotFoundError(
                f"Feature record for individual={individual!r} at timestamp={timestamp_key!r} is missing: {missing}."
            )

        return {name: float(feature_row[name]) for name in required_features}


@lru_cache(maxsize=1)
def get_feature_store() -> InMemoryFeatureStore:
    return InMemoryFeatureStore(DEFAULT_FEATURE_STORE)
