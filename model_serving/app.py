from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, status

from .feature_store import FeatureNotFoundError, InMemoryFeatureStore, get_feature_store
from .model_service import RegistryModelLoadError, RegistryModelService, get_model_service
from .schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from .settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="FastAPI service that serves the promoted MLflow model through the champion alias.",
    )

    @app.on_event("startup")
    def _warm_up_model() -> None:
        service = get_model_service()
        try:
            service.load()
        except RegistryModelLoadError:
            # Keep the app available for docs and health checks even if MLflow is temporarily unavailable.
            pass

    @app.get("/", tags=["health"])
    def root() -> dict[str, str]:
        return {"message": "Electricity load prediction API is running."}

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health(service: RegistryModelService = Depends(get_model_service)) -> HealthResponse:
        loaded = service.is_loaded()
        detail = None if loaded else "The model could not be loaded yet. Check MLflow and the champion alias."
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            model_uri=service.model_uri,
            detail=detail,
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(
        payload: PredictionRequest,
        feature_store: InMemoryFeatureStore = Depends(get_feature_store),
        service: RegistryModelService = Depends(get_model_service),
    ) -> PredictionResponse:
        try:
            features = feature_store.get_features(payload.individual, payload.timestamp)
        except FeatureNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        try:
            prediction = service.predict(features)
        except RegistryModelLoadError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

        return PredictionResponse(
            individual=payload.individual,
            timestamp=payload.timestamp,
            model_uri=service.model_uri,
            prediction=prediction,
            features=features,
        )

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
    def predict_batch(
        payload: BatchPredictionRequest,
        feature_store: InMemoryFeatureStore = Depends(get_feature_store),
        service: RegistryModelService = Depends(get_model_service),
    ) -> BatchPredictionResponse:
        if not payload.items:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="items must not be empty.")

        try:
            service.load()
        except RegistryModelLoadError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

        batch_items: list[BatchPredictionItem] = []
        for item in payload.items:
            try:
                features = feature_store.get_features(item.individual, item.timestamp)
                prediction = service.predict(features)
            except FeatureNotFoundError as exc:
                batch_items.append(
                    BatchPredictionItem(
                        individual=item.individual,
                        timestamp=item.timestamp,
                        error=str(exc),
                    )
                )
                continue

            batch_items.append(
                BatchPredictionItem(
                    individual=item.individual,
                    timestamp=item.timestamp,
                    prediction=prediction,
                    features=features,
                )
            )

        return BatchPredictionResponse(model_uri=service.model_uri, items=batch_items)

    return app


app = create_app()
