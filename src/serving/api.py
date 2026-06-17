import logging
import os
import secrets
import time
import hashlib
import yaml
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, field_validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.forecasting.engine import DemandForecastEngine
from src.common.mlflow_config import resolve_tracking_uri

logger = logging.getLogger(__name__)

# ── Telemetry (Prometheus) ─────────────────────────────────────
PREDICT_REQUESTS = Counter("predict_requests_total", "Predict requests",
                           ["granularity", "outcome"])
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Predict latency (s)")

# ── App & Rate Limiter ─────────────────────────────────────────
# 60 requests/minute per IP — prevents model extraction attacks
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(_app):
    load_models()      # warm the forecast engines on startup
    yield


app = FastAPI(
    title="NYC Taxi Demand Forecasting API",
    version="1.0.0",
    # Disable automatic OpenAPI docs in production to avoid leaking schema
    docs_url=None if os.environ.get("ENV") == "production" else "/docs",
    redoc_url=None,
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Config ─────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
with open(_CONFIG_PATH, 'r') as f:
    _raw = f.read()
config = yaml.safe_load(os.path.expandvars(_raw))

mlflow.set_tracking_uri(resolve_tracking_uri(config))

# ── API Key Auth ───────────────────────────────────────────────
_API_KEY = os.environ.get("API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Security(_api_key_header)):
    if not _API_KEY:
        raise RuntimeError("API_KEY environment variable is not set")
    if not secrets.compare_digest(key or "", _API_KEY):
        logger.warning("Rejected request with invalid API key")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

# ── Forecast Engines ───────────────────────────────────────────
# Same forecaster the batch DAGs use: loads the @champion forecast model and
# its VaR sigma(s), and computes features from recent history via the exact
# production code — so the API and batch jobs give identical forward forecasts.
_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
_RECENT_WINDOW = 60  # rows of history; covers the longest lag (28 daily / 24 hourly)
_engines = {}


def load_models():
    for gran in ('D', 'H'):
        try:
            _engines[gran] = DemandForecastEngine(granularity=gran)
            logger.info("Loaded %s forecast engine (garch_sigma=%.4f)",
                        gran, _engines[gran].error_sigma)
        except Exception as e:
            logger.error("Failed to load %s forecast engine: %s", gran, e)
            _engines[gran] = None


def _load_recent_history(granularity):
    """Most recent cleaned aggregates — the same DVC-tracked artifact the batch
    DAGs consume. In a fuller deployment this would read the Feast online store."""
    fname = 'daily_demand.csv' if granularity == 'D' else 'hourly_demand.csv'
    path = os.path.join(_DATA_DIR, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['TimePeriod'])
    if df.empty:
        return None
    return df.sort_values('TimePeriod').tail(_RECENT_WINDOW).reset_index(drop=True)

# ── Request / Response Schemas ─────────────────────────────────
class ForecastRequest(BaseModel):
    time_period: str
    granularity: str
    confidence_level: float = 0.99

    @field_validator('granularity')
    @classmethod
    def granularity_must_be_valid(cls, v):
        if v.upper() not in ('D', 'H'):
            raise ValueError("granularity must be 'D' or 'H'")
        return v.upper()

    @field_validator('confidence_level')
    @classmethod
    def confidence_in_range(cls, v):
        if not (0.5 <= v <= 0.9999):
            raise ValueError("confidence_level must be between 0.5 and 0.9999")
        return v

    @field_validator('time_period')
    @classmethod
    def time_period_safe(cls, v):
        # Prevent injection via the time_period string
        if len(v) > 20 or not all(c in '0123456789T:-Z' for c in v):
            raise ValueError("time_period contains invalid characters")
        return v

class ForecastResponse(BaseModel):
    time_period: str
    granularity: str
    forecast_period: str          # the FUTURE period this forecast is for
    point_forecast: float
    safety_buffer_99: float
    capacity_target: int

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health_check():
    ready = any(e is not None for e in _engines.values())
    return {"status": "healthy" if ready else "degraded", "models_loaded": ready}


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint (request counts, latency histogram)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=ForecastResponse)
@limiter.limit("60/minute")
def predict(req: ForecastRequest, request: Request, _: None = Security(require_api_key)):
    start_time = time.monotonic()
    # Anonymised request fingerprint for audit log (no PII)
    request_id = hashlib.sha256(
        f"{get_remote_address(request)}:{req.time_period}:{time.time()}".encode()
    ).hexdigest()[:12]

    logger.info(
        "[AUDIT] request_id=%s granularity=%s time_period=%s",
        request_id, req.granularity, req.time_period,
    )

    engine = _engines.get(req.granularity)
    if engine is None:
        PREDICT_REQUESTS.labels(req.granularity, "unavailable").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    history = _load_recent_history(req.granularity)
    if history is None:
        PREDICT_REQUESTS.labels(req.granularity, "no_history").inc()
        raise HTTPException(status_code=503, detail="Recent history unavailable for inference")

    # Forecast the NEXT period from recent history (pandera validation runs
    # inside forecast_next). The API and batch jobs share this exact code path.
    try:
        results = engine.forecast_next(history, confidence_level=req.confidence_level)
    except Exception as e:
        PREDICT_REQUESTS.labels(req.granularity, "invalid").inc()
        logger.error("[AUDIT] request_id=%s forecast failed: %s", request_id, e)
        raise HTTPException(status_code=422, detail="Input data failed validation")

    point_forecast = results['Point_Forecast']

    # Sanity-check: reject wildly anomalous predictions before returning them
    MAX_REASONABLE_VOLUME = 100_000
    if point_forecast > MAX_REASONABLE_VOLUME:
        PREDICT_REQUESTS.labels(req.granularity, "anomalous").inc()
        logger.error("[AUDIT] request_id=%s anomalous prediction=%.2f — fallback required", request_id, point_forecast)
        raise HTTPException(status_code=500, detail="Model returned an anomalous prediction. Fallback required.")

    elapsed = time.monotonic() - start_time
    PREDICT_LATENCY.observe(elapsed)
    PREDICT_REQUESTS.labels(req.granularity, "ok").inc()
    elapsed_ms = elapsed * 1000
    logger.info(
        "[AUDIT] request_id=%s forecast_period=%s point_forecast=%.2f capacity_target=%d latency_ms=%.1f",
        request_id, results['Forecast_Period'], point_forecast, results['Capacity_Target'], elapsed_ms,
    )

    return ForecastResponse(
        time_period      = req.time_period,
        granularity      = req.granularity,
        forecast_period  = str(results['Forecast_Period']),
        point_forecast   = point_forecast,
        safety_buffer_99 = results['Safety_Buffer_99'],
        capacity_target  = results['Capacity_Target'],
    )


if __name__ == "__main__":
    # Binding all interfaces is required inside the container; access is fronted
    # by the K8s Service + API-key auth, not exposed directly.
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8080, reload=False)  # nosec B104
