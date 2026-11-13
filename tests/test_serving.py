from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from forecaster.data.panel import load_panel
from forecaster.evaluation.paths import Artifacts
from forecaster.serving.app import create_app
from forecaster.serving.predictor import FrequencyPredictor


def test_serving_reproduces_the_offline_forecast(pipeline) -> None:
    artifacts = Artifacts(pipeline)
    stored = np.load(artifacts.forecasts("Hourly"))["blend"]
    panel = load_panel(pipeline, "Hourly")
    predictor = FrequencyPredictor(pipeline, "Hourly")
    for index in (0, 5, 17):
        served = predictor.predict(panel.series(index))
        assert np.allclose(served.point, stored[index], rtol=1e-3, atol=1e-3)


def test_serving_pads_a_short_history(pipeline) -> None:
    predictor = FrequencyPredictor(pipeline, "Hourly")
    window, context = predictor.prepare_window(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    assert window.shape == (1, predictor.spec.lookback)
    assert context[0] == 3
    assert window[0, 0] == 4.0


def test_health_endpoint_lists_frequencies(pipeline) -> None:
    client = TestClient(create_app(pipeline))
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert "Hourly" in payload["frequencies"]


def test_forecast_endpoint_returns_intervals(pipeline) -> None:
    panel = load_panel(pipeline, "Hourly")
    client = TestClient(create_app(pipeline))
    response = client.post(
        "/forecast",
        json={"frequency": "Hourly", "history": panel.series(0).tolist()},
    )
    assert response.status_code == 200
    payload = response.json()
    horizon = pipeline.frequency["Hourly"].horizon
    assert payload["horizon"] == horizon
    assert len(payload["point"]) == horizon
    for level in ("0.80", "0.95"):
        assert len(payload["lower"][level]) == horizon
        lower = np.asarray(payload["lower"][level])
        upper = np.asarray(payload["upper"][level])
        point = np.asarray(payload["point"])
        assert np.all(lower <= point) and np.all(point <= upper)
    assert payload["interval_source"] == "split"


def test_known_series_identifier_selects_adaptive_intervals(pipeline) -> None:
    panel = load_panel(pipeline, "Hourly")
    client = TestClient(create_app(pipeline))
    response = client.post(
        "/forecast",
        json={
            "frequency": "Hourly",
            "history": panel.series(0).tolist(),
            "series_id": panel.series_ids[0],
        },
    )
    assert response.json()["interval_source"] == "adaptive"


def test_unknown_frequency_is_rejected(pipeline) -> None:
    client = TestClient(create_app(pipeline))
    response = client.post("/forecast", json={"frequency": "Fortnightly", "history": [1.0, 2.0]})
    assert response.status_code == 404


def test_non_finite_history_is_rejected(pipeline) -> None:
    client = TestClient(create_app(pipeline))
    response = client.post(
        "/forecast",
        content='{"frequency": "Hourly", "history": [1.0, NaN, 3.0]}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
