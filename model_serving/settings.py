from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ._bootstrap import ensure_data_code_path

ensure_data_code_path()

import ml_pipeline  # type: ignore[import-not-found]


DEFAULT_MLFLOW_TRACKING_URI = "https://mlflow.10-53-101-61.nip.io"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    mlflow_tracking_uri: str = Field(
        default=DEFAULT_MLFLOW_TRACKING_URI,
        validation_alias="MLFLOW_TRACKING_URI",
    )
    registered_model_name: str = Field(
        default="modelregistrytest",
        validation_alias="MLFLOW_REGISTERED_MODEL_NAME",
    )
    champion_alias: str = Field(
        default="champion",
        validation_alias="MLFLOW_CHAMPION_ALIAS",
    )
    api_title: str = "Electricity load prediction API"
    api_version: str = "0.1.0"
    required_features: list[str] = Field(default_factory=lambda: list(ml_pipeline.DEFAULT_FEATURES))

    @property
    def model_uri(self) -> str:
        return f"models:/{self.registered_model_name}@{self.champion_alias}"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
