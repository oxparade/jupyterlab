from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from .settings import AppSettings, get_settings


class RegistryModelLoadError(RuntimeError):
    """Raised when the promoted model cannot be loaded from MLflow."""


@dataclass
class RegistryModelService:
    settings: AppSettings
    _model: Any | None = None
    _load_error: Exception | None = None

    @property
    def model_uri(self) -> str:
        return self.settings.model_uri

    def is_loaded(self) -> bool:
        return self._model is not None and self._load_error is None

    def load(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            self._model = mlflow.pyfunc.load_model(self.model_uri)
            self._load_error = None
            return self._model
        except Exception as exc:  # pragma: no cover - depends on external MLflow state
            self._load_error = exc
            raise RegistryModelLoadError(
                f"Unable to load model from {self.model_uri!r}. Check the registry alias and tracking URI."
            ) from exc

    def last_error(self) -> Exception | None:
        return self._load_error

    def predict(self, features: dict[str, float]) -> float:
        model = self.load()
        frame = pd.DataFrame([features], columns=self.settings.required_features)
        prediction = model.predict(frame)
        return float(np.asarray(prediction).ravel()[0])


@lru_cache(maxsize=1)
def get_model_service() -> RegistryModelService:
    return RegistryModelService(settings=get_settings())
