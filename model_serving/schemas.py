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


class PredictionError(BaseModel):
    individual: str
    timestamp: datetime
    detail: str


class BatchPredictionItem(BaseModel):
    individual: str
    timestamp: datetime
    prediction: float | None = None
    features: dict[str, float] | None = None
    error: str | None = None


class BatchPredictionRequestItem(BaseModel):
    individual: str = Field(..., examples=["MT_001"])
    timestamp: datetime = Field(..., examples=["2014-07-01T00:00:00"])


class BatchPredictionRequest(BaseModel):
    items: list[BatchPredictionRequestItem]


class BatchPredictionResponse(BaseModel):
    model_uri: str
    items: list[BatchPredictionItem]
