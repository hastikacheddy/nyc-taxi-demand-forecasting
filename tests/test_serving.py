"""
Serving API tests. The inference engine and the recent-history loader are
mocked so these run anywhere (incl. CI) without a live model registry or data.
Verifies the wiring: auth gate, input validation, and the real engine call path
(no more zero-vector stub).
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

pytest.importorskip("slowapi", reason="slowapi not installed")
pytest.importorskip("httpx", reason="httpx/TestClient not installed")

os.environ.setdefault("API_KEY", "testsecret")
_HEADERS = {"X-API-Key": "testsecret"}


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    import src.serving.api as api

    engine = MagicMock()
    engine.error_sigma = 100.0
    engine.forecast_next.return_value = {
        "Forecast_Period": pd.Timestamp("2026-04-01"),
        "Point_Forecast": 500.0,
        "Safety_Buffer_99": 750.0,
        "Capacity_Target": 750,
    }
    history = pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=60, freq="D"),
        "Volume": range(60),
    })
    with patch.object(api, "DemandForecastEngine", return_value=engine), \
         patch.object(api, "_load_recent_history", return_value=history):
        with TestClient(api.app) as c:
            yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] is True


def test_predict_requires_valid_api_key(client):
    r = client.post("/predict", headers={"X-API-Key": "wrong"},
                    json={"time_period": "2026-03-31", "granularity": "D"})
    assert r.status_code == 401


def test_predict_rejects_bad_granularity(client):
    r = client.post("/predict", headers=_HEADERS,
                    json={"time_period": "2026-03-31", "granularity": "X"})
    assert r.status_code == 422


def test_predict_returns_forecast(client):
    r = client.post("/predict", headers=_HEADERS,
                    json={"time_period": "2026-03-31", "granularity": "D", "confidence_level": 0.99})
    assert r.status_code == 200
    body = r.json()
    assert body["point_forecast"] == 500.0
    assert body["capacity_target"] == 750
    assert body["safety_buffer_99"] >= body["point_forecast"]
    assert body["forecast_period"].startswith("2026-04-01")


def test_predict_uses_forecaster_not_stub(client):
    """The endpoint must call the forecaster's forecast_next (real forward
    forecast), not fabricate a zero vector."""
    import src.serving.api as api
    client.post("/predict", headers=_HEADERS,
                json={"time_period": "2026-03-31", "granularity": "D"})
    api._engines["D"].forecast_next.assert_called()


def test_metrics_endpoint_exposes_prometheus(client):
    client.post("/predict", headers=_HEADERS,
                json={"time_period": "2026-03-31", "granularity": "D"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "predict_requests_total" in r.text
    assert "predict_latency_seconds" in r.text
