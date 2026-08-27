from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_uri: str
    detail: str | None = None


class PredictionRequest(BaseModel):
    individual: str = Field(..., examples=["MT_001"])
    timestamp: datetime = Field(..., examples=["2014-07-01T00:00:00"])


class PredictionResponse(BaseModel):
    individual: str
    timestamp: datetime
    model_uri: str
    prediction: float
    features: dict[str, float]
