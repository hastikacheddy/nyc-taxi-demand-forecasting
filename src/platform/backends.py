"""
Inference backends for the AI platform.

A *backend* is the thing that actually executes a model. The platform's job is to
pick the right one for a ModelVersion's framework and give the router a single,
uniform `predict()` call — so the control plane never branches on model type.

Backends shipped:

  * EchoBackend               deterministic, dependency-free. Demos + tests.
  * LightGBMForecastBackend   wraps the repo's real DemandForecastEngine (the same
                              code path the batch DAGs and /predict use). CPU pool.
  * VLLMBackend               calls an OpenAI-compatible vLLM server over HTTP. GPU
                              pool. Runnable the moment you point it at a served
                              model; degrades to a clear "unavailable" otherwise.

The abstract contract is intentionally tiny — `load`, `predict`, `health`, `kind`
— because that is all the router needs, and a small contract is what makes adding
a Triton or KServe backend later a 40-line file, not a refactor.
"""
from __future__ import annotations

import abc
import os
from typing import Any, Dict

from src.platform.registry import Framework, ModelVersion


class BackendUnavailable(RuntimeError):
    """Raised when a backend's dependency (GPU server, weights) isn't reachable.
    The router treats this as a health failure and can fall back to another
    version — never a 500 that leaks a stack trace to the caller."""


class InferenceBackend(abc.ABC):
    def __init__(self, model_version: ModelVersion) -> None:
        self.mv = model_version
        self._loaded = False

    @property
    def kind(self) -> str:
        return self.mv.framework.value

    @abc.abstractmethod
    def load(self) -> None:
        """Warm the backend (load weights / open a client). Idempotent."""

    @abc.abstractmethod
    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run one inference. Raises BackendUnavailable on infra failure."""

    def health(self) -> bool:
        try:
            return self._health()
        except Exception:
            return False

    def _health(self) -> bool:
        return self._loaded


class EchoBackend(InferenceBackend):
    """Deterministic backend: returns a hash of the payload. Lets the whole
    platform (registry → deployment → routing → metrics) be exercised end to end
    with zero model dependencies."""

    def load(self) -> None:
        self._loaded = True

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded:
            self.load()
        token = abs(hash(frozenset(payload.items()))) % 100_000 if payload else 0
        return {"model": self.mv.name, "version": self.mv.version,
                "framework": "echo", "echo": payload, "token": token}


class LightGBMForecastBackend(InferenceBackend):
    """Wraps the repo's DemandForecastEngine. This is the honest integration: the
    platform serves the *real* leakage-free forecaster over the same code path as
    the batch pipeline, so there is no platform/batch skew.

    payload: {"granularity": "D"|"H", "confidence_level": float}
    The recent-history load mirrors src/serving/api.py so the two agree."""

    _DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    _RECENT_WINDOW = 60

    def load(self) -> None:
        if self._loaded:
            return
        # Lazy import: keeps the control plane importable without mlflow/lightgbm
        # installed (e.g. a lightweight gateway-only container).
        from src.forecasting.engine import DemandForecastEngine
        gran = (self.mv.tags.get("granularity") or "D").upper()
        try:
            self._engine = DemandForecastEngine(granularity=gran)
        except Exception as e:  # model not in registry yet, mlflow down, etc.
            raise BackendUnavailable(f"LightGBM engine load failed: {e}") from e
        self._gran = gran
        self._loaded = True

    def _recent_history(self, gran: str):
        import pandas as pd
        fname = "daily_demand.csv" if gran == "D" else "hourly_demand.csv"
        path = os.path.join(self._DATA_DIR, fname)
        if not os.path.exists(path):
            raise BackendUnavailable(f"recent history missing: {fname}")
        df = pd.read_csv(path, parse_dates=["TimePeriod"])
        if df.empty:
            raise BackendUnavailable("recent history empty")
        return df.sort_values("TimePeriod").tail(self._RECENT_WINDOW).reset_index(drop=True)

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded:
            self.load()
        gran = (payload.get("granularity") or self._gran).upper()
        conf = float(payload.get("confidence_level", 0.99))
        history = self._recent_history(gran)
        result = self._engine.forecast_next(history, confidence_level=conf)
        return {
            "model": self.mv.name,
            "version": self.mv.version,
            "framework": "lightgbm",
            "forecast_period": str(result["Forecast_Period"]),
            "point_forecast": result["Point_Forecast"],
            "safety_buffer_99": result["Safety_Buffer_99"],
            "capacity_target": result["Capacity_Target"],
        }

    def _health(self) -> bool:
        return self._loaded and self._engine is not None


class VLLMBackend(InferenceBackend):
    """Adapter to a vLLM server exposing the OpenAI-compatible API. The platform
    treats an LLM as just another registered model on the GPU pool.

    The server URL comes from the model version's artifact_uri when it is an
    http(s) endpoint, else the VLLM_BASE_URL env var. If neither resolves, the
    backend is honestly `unavailable` rather than faking a response — the whole
    reason to have a real vLLM path is to benchmark it (see benchmarks/)."""

    def __init__(self, model_version: ModelVersion) -> None:
        super().__init__(model_version)
        uri = model_version.artifact_uri
        self.base_url = (uri if uri.startswith("http") else os.environ.get("VLLM_BASE_URL", "")).rstrip("/")
        self.served_model = model_version.tags.get("served_model_name", model_version.name)
        self.timeout = float(os.environ.get("VLLM_TIMEOUT", "60"))

    def load(self) -> None:
        if not self.base_url:
            raise BackendUnavailable(
                "no vLLM endpoint: set model artifact_uri to an http(s) URL or "
                "export VLLM_BASE_URL (e.g. http://vllm.mlops.svc:8000/v1)"
            )
        self._loaded = True

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded:
            self.load()
        import httpx
        body = {
            "model": self.served_model,
            "messages": payload.get("messages")
            or [{"role": "user", "content": payload.get("prompt", "")}],
            "max_tokens": int(payload.get("max_tokens", 128)),
            "temperature": float(payload.get("temperature", 0.0)),
        }
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout)
            r.raise_for_status()
        except Exception as e:
            raise BackendUnavailable(f"vLLM request failed: {e}") from e
        data = r.json()
        return {
            "model": self.mv.name,
            "version": self.mv.version,
            "framework": "vllm",
            "completion": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
        }

    def _health(self) -> bool:
        if not self.base_url:
            return False
        try:
            import httpx
            r = httpx.get(f"{self.base_url}/models", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


_REGISTRY = {
    Framework.ECHO: EchoBackend,
    Framework.LIGHTGBM: LightGBMForecastBackend,
    Framework.SKLEARN: LightGBMForecastBackend,  # same pyfunc path
    Framework.VLLM: VLLMBackend,
    Framework.TRANSFORMERS: VLLMBackend,          # OpenAI-compatible TGI/vLLM
}


def build_backend(mv: ModelVersion) -> InferenceBackend:
    """Factory: pick the backend class for a version's framework."""
    cls = _REGISTRY.get(mv.framework)
    if cls is None:
        raise BackendUnavailable(f"no backend for framework {mv.framework}")
    return cls(mv)
