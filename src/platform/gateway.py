"""
AI Platform API gateway (control plane + inference data plane).

This is the single internal entrypoint developers hit instead of talking to a
model container directly:

    POST /v1/models                 register a model version
    GET  /v1/models                 list registered versions
    POST /v1/models/{name}/alias    move an alias (champion/canary) to a version
    POST /v1/deployments            create/replace a deployment (CPU vs GPU pool)
    GET  /v1/deployments            list deployments
    GET  /v1/deployments/{id}       one deployment's status/health
    POST /v1/deployments/{id}/promote   promote canary → stable
    POST /v1/workloads              apply a declarative workload spec (YAML)
    POST /v1/inference              route an inference to a live deployment
    GET  /v1/costs                  cost attribution (FinOps)
    GET  /v1/metrics                Prometheus scrape

Auth mirrors src/serving/api.py: an optional X-API-Key with a constant-time
compare (set PLATFORM_API_KEY to require it; unset = open, for local dev).

The gateway holds process-global ModelRegistry + DeploymentManager singletons.
In a multi-replica production deployment these would be backed by the shared
MLflow registry + a control-plane DB; the abstraction boundary is already here,
so that swap is a store implementation, not an API change."""
from __future__ import annotations

import os
import secrets
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.platform.backends import BackendUnavailable
from src.platform.deployments import DeploymentManager
from src.platform.finops import CostModel
from src.platform.registry import Framework, ModelRegistry, ResourceProfile
from src.platform.workloads import WorkloadManager, WorkloadSpec, WorkloadError

# ── Telemetry ──────────────────────────────────────────────────────
INFER_REQUESTS = Counter(
    "platform_inference_requests_total", "Platform inference requests",
    ["model", "version", "outcome"],
)
INFER_LATENCY = Histogram(
    "platform_inference_latency_seconds", "Platform inference latency (s)", ["model"],
)
DEPLOYMENTS = Gauge(
    "platform_deployments", "Deployments by status", ["status"],
)
REGISTERED_MODELS = Gauge(
    "platform_registered_model_versions", "Registered model versions",
)

# ── Process-global control plane ───────────────────────────────────
registry = ModelRegistry()
deployments = DeploymentManager(registry)
cost_model = CostModel()

# ── Auth (optional, constant-time) ─────────────────────────────────
_API_KEY = os.environ.get("PLATFORM_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str = Security(_api_key_header)) -> None:
    if not _API_KEY:
        return  # auth disabled for local/dev
    if not secrets.compare_digest(key or "", _API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing API key")


# ── Schemas ────────────────────────────────────────────────────────
class RegisterModelRequest(BaseModel):
    name: str = Field(..., examples=["taxi-demand-daily"])
    framework: Framework = Field(..., examples=["lightgbm"])
    artifact_uri: str = Field(..., examples=["models:/TaxiDemand_Daily_Forecast@champion"])
    description: str = ""
    sha256: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    # Optional explicit resource ask; defaults are derived from framework.
    cpu: Optional[str] = None
    memory: Optional[str] = None
    gpu: Optional[int] = None
    gpu_type: Optional[str] = None


class AliasRequest(BaseModel):
    version: int
    alias: str = Field(..., examples=["champion"])


class CreateDeploymentRequest(BaseModel):
    model_name: str
    selector: str = "champion"
    replicas: int = Field(1, ge=0, le=100)
    canary_selector: Optional[str] = None
    canary_traffic: int = Field(0, ge=0, le=100)


class InferenceRequest(BaseModel):
    model: str
    inputs: Dict[str, Any] = Field(default_factory=dict)


# ── App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Internal AI Platform API",
    version="1.0.0",
    description="Framework-agnostic control plane over registry, deployments, and inference.",
    docs_url=None if os.environ.get("ENV") == "production" else "/docs",
    redoc_url=None,
)


def _refresh_gauges() -> None:
    REGISTERED_MODELS.set(len(registry.list()))
    counts: Dict[str, int] = {}
    for dep in deployments.list():
        counts[dep.status.value] = counts.get(dep.status.value, 0) + 1
    for st in ("pending", "loading", "ready", "degraded", "failed"):
        DEPLOYMENTS.labels(st).set(counts.get(st, 0))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models": len(registry.list()),
            "deployments": len(deployments.list())}


@app.get("/v1/metrics")
def metrics() -> Response:
    _refresh_gauges()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Models ─────────────────────────────────────────────────────────
@app.post("/v1/models", status_code=201, dependencies=[Depends(require_api_key)])
def register_model(req: RegisterModelRequest) -> dict:
    resources = None
    if any(v is not None for v in (req.cpu, req.memory, req.gpu, req.gpu_type)):
        base = ResourceProfile.for_framework(req.framework)
        resources = ResourceProfile(
            cpu=req.cpu or base.cpu,
            memory=req.memory or base.memory,
            gpu=req.gpu if req.gpu is not None else base.gpu,
            gpu_type=req.gpu_type or base.gpu_type,
        )
    mv = registry.register(
        name=req.name, framework=req.framework, artifact_uri=req.artifact_uri,
        resources=resources, sha256=req.sha256, description=req.description, tags=req.tags,
    )
    return mv.to_public()


@app.get("/v1/models")
def list_models() -> List[dict]:
    return [mv.to_public() for mv in registry.list()]


@app.post("/v1/models/{name}/alias", dependencies=[Depends(require_api_key)])
def set_alias(name: str, req: AliasRequest) -> dict:
    try:
        mv = registry.set_alias(name, req.version, req.alias)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return mv.to_public()


# ── Deployments ────────────────────────────────────────────────────
@app.post("/v1/deployments", status_code=201, dependencies=[Depends(require_api_key)])
def create_deployment(req: CreateDeploymentRequest) -> dict:
    try:
        dep = deployments.create(
            req.model_name, selector=req.selector, replicas=req.replicas,
            canary_selector=req.canary_selector, canary_traffic=req.canary_traffic,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    _refresh_gauges()
    return dep.to_public()


@app.get("/v1/deployments")
def list_deployments() -> List[dict]:
    return [d.to_public() for d in deployments.list()]


@app.get("/v1/deployments/{deployment_id}")
def get_deployment(deployment_id: str) -> dict:
    dep = deployments.get(deployment_id)
    if dep is None:
        raise HTTPException(status_code=404, detail="deployment not found")
    return dep.to_public()


@app.post("/v1/deployments/{deployment_id}/promote", dependencies=[Depends(require_api_key)])
def promote(deployment_id: str) -> dict:
    try:
        dep = deployments.promote_canary(deployment_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    _refresh_gauges()
    return dep.to_public()


# ── Workloads (declarative) ────────────────────────────────────────
@app.post("/v1/workloads", status_code=201, dependencies=[Depends(require_api_key)])
async def apply_workload(request: Request) -> dict:
    """Apply a declarative workload spec. Accepts a YAML body (Content-Type
    text/plain or application/yaml) or a JSON object. Registers the model version
    and reconciles a deployment — the 'deploy any AI workload' entrypoint."""
    raw = (await request.body()).decode("utf-8").strip()
    if not raw:
        raise HTTPException(status_code=422, detail="empty workload spec")
    try:
        # try JSON first (application/json), fall back to YAML
        ctype = request.headers.get("content-type", "")
        if "json" in ctype:
            import json
            spec = WorkloadSpec.from_dict(json.loads(raw))
        else:
            spec = WorkloadSpec.from_yaml(raw)
        # build from current globals so state stays consistent
        return WorkloadManager(registry, deployments).apply(spec)
    except WorkloadError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Inference ──────────────────────────────────────────────────────
@app.post("/v1/inference", dependencies=[Depends(require_api_key)])
def inference(req: InferenceRequest) -> dict:
    start = time.monotonic()
    try:
        result, slot = deployments.route(req.model, req.inputs)
    except KeyError as e:
        INFER_REQUESTS.labels(req.model, "none", "not_found").inc()
        raise HTTPException(status_code=404, detail=str(e))
    except BackendUnavailable as e:
        INFER_REQUESTS.labels(req.model, "none", "unavailable").inc()
        raise HTTPException(status_code=503, detail=str(e))

    elapsed = time.monotonic() - start
    INFER_LATENCY.labels(req.model).observe(elapsed)
    INFER_REQUESTS.labels(req.model, str(slot.version.version), "ok").inc()
    return {
        "served_by": {"version": slot.version.version, "framework": slot.version.framework.value},
        "latency_ms": round(elapsed * 1000, 2),
        "result": result,
    }


# ── FinOps ─────────────────────────────────────────────────────────
def _observed_rps(dep) -> float:
    """Rough sustained rps = total served requests / deployment age. In-process
    proxy for what Prometheus rate() would give in production."""
    served = 0
    for slot in (dep.stable, dep.canary):
        if slot is not None:
            served += slot.requests
    age = max(1e-6, time.time() - dep.created_at)
    return served / age


@app.get("/v1/costs")
def costs() -> dict:
    """Cost attribution across deployments: hourly/monthly burn, $/1k requests per
    model, and the idle-GPU waste list. GPU utilization is None here (it comes
    from the DCGM exporter in-cluster); pass it in a fuller integration."""
    return cost_model.report(deployments.list(), rps_of=_observed_rps)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.platform.gateway:app", host="0.0.0.0", port=8090, reload=False)  # nosec B104
