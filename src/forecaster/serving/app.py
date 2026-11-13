from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forecaster.config import Config, load_config
from forecaster.serving.predictor import PredictorRegistry


class ForecastRequest(BaseModel):
    frequency: str = Field(min_length=1)
    history: list[float] = Field(min_length=2)
    series_id: str | None = None


class ForecastResponse(BaseModel):
    frequency: str
    horizon: int
    point: list[float]
    lower: dict[str, list[float]]
    upper: dict[str, list[float]]
    interval_source: str


def create_app(config: Config | None = None) -> FastAPI:
    settings = config if config is not None else load_config()
    registry = PredictorRegistry(settings)
    app = FastAPI(title="calibrated-timeseries-forecaster")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "frequencies": list(settings.run.frequencies)}

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest) -> ForecastResponse:
        try:
            predictor = registry.get(request.frequency)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown frequency") from None
        history = np.asarray(request.history, dtype=np.float32)
        if not np.isfinite(history).all():
            raise HTTPException(status_code=422, detail="history must be finite")
        prediction = predictor.predict(history, request.series_id)
        known = request.series_id is not None and request.series_id in predictor.series_lookup
        return ForecastResponse(
            frequency=request.frequency,
            horizon=int(prediction.point.size),
            point=[float(value) for value in prediction.point],
            lower={key: [float(v) for v in value] for key, value in prediction.lower.items()},
            upper={key: [float(v) for v in value] for key, value in prediction.upper.items()},
            interval_source="adaptive" if known else "split",
        )

    return app
