"""
Deployment manager for the AI platform.

A *deployment* is a live serving endpoint for one model. It owns:

  * which registry versions are serving (a stable version + an optional canary),
  * the traffic split between them,
  * which compute pool the replicas land on (CPU vs GPU),
  * lifecycle state (pending → loading → ready / failed),
  * routing an inference request to a backend, with canary fallback.

This is the piece that makes the platform feel like SageMaker/Vertex endpoints:
you `create` a deployment against "model X @champion", optionally shift 10% of
traffic to a canary version, and the manager handles placement + routing + a
safe fallback when the canary misbehaves.

It is deliberately an *in-process simulation* of a control plane — it does not
itself talk to the Kubernetes API. The production mapping (KServe
InferenceService, GPU node pools, HPA) is in kubernetes/serving/ and documented
in docs/platform/GPU_DESIGN.md. Keeping the control logic here, framework-free,
is what lets it be unit-tested without a cluster (mirrors the repo's existing
"src/ plays, Airflow conducts" split)."""
from __future__ import annotations

import enum
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.platform.backends import BackendUnavailable, InferenceBackend, build_backend
from src.platform.registry import Framework, ModelRegistry, ModelVersion


class ComputePool(str, enum.Enum):
    """Physical pool a replica is scheduled onto. GPU frameworks land on the GPU
    pool (node affinity + tolerations for the GPU taint); everything else on the
    cheaper CPU pool. See docs/platform/GPU_DESIGN.md."""
    CPU = "cpu"
    GPU = "gpu"

    @staticmethod
    def for_framework(fw: Framework) -> "ComputePool":
        return ComputePool.GPU if fw.needs_gpu else ComputePool.CPU


class DeploymentStatus(str, enum.Enum):
    PENDING = "pending"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"   # serving on stable only; canary or a replica failed
    FAILED = "failed"


@dataclass
class VersionSlot:
    """One version serving inside a deployment, plus its warmed backend."""
    version: ModelVersion
    backend: InferenceBackend
    traffic: int              # percent of traffic (stable + canary sum to 100)
    healthy: bool = True
    requests: int = 0
    failures: int = 0


@dataclass
class Deployment:
    id: str
    model_name: str
    pool: ComputePool
    status: DeploymentStatus = DeploymentStatus.PENDING
    replicas: int = 1
    created_at: float = field(default_factory=time.time)
    stable: Optional[VersionSlot] = None
    canary: Optional[VersionSlot] = None
    last_error: Optional[str] = None

    def to_public(self) -> dict:
        def slot(s: Optional[VersionSlot]):
            if s is None:
                return None
            return {
                "version": s.version.version,
                "framework": s.version.framework.value,
                "traffic_percent": s.traffic,
                "healthy": s.healthy,
                "requests": s.requests,
                "failures": s.failures,
            }
        return {
            "id": self.id,
            "model_name": self.model_name,
            "pool": self.pool.value,
            "status": self.status.value,
            "replicas": self.replicas,
            "created_at": self.created_at,
            "endpoint": f"/v1/inference (model={self.model_name})",
            "stable": slot(self.stable),
            "canary": slot(self.canary),
            "last_error": self.last_error,
        }


class DeploymentManager:
    def __init__(self, registry: ModelRegistry, rng: Optional[random.Random] = None) -> None:
        self._registry = registry
        self._deployments: Dict[str, Deployment] = {}
        self._by_model: Dict[str, str] = {}   # model_name -> deployment id (latest)
        self._lock = threading.RLock()
        self._rng = rng or random.Random()

    # ── lifecycle ─────────────────────────────────────────────────
    def create(
        self,
        model_name: str,
        *,
        selector: str = "champion",
        replicas: int = 1,
        canary_selector: Optional[str] = None,
        canary_traffic: int = 0,
    ) -> Deployment:
        """Create (or replace) the deployment for `model_name`.

        Resolves the stable version via `selector` (alias / version / "latest").
        Optionally attaches a canary version taking `canary_traffic`% of requests.
        Warms both backends; a backend that fails to load leaves the deployment
        DEGRADED (canary) or FAILED (stable) rather than crashing the platform."""
        stable_mv = self._registry.resolve(model_name, selector)
        if stable_mv is None:
            raise KeyError(f"model '{model_name}' selector '{selector}' not found")

        if not (0 <= canary_traffic <= 100):
            raise ValueError("canary_traffic must be within [0, 100]")

        pool = ComputePool.for_framework(stable_mv.framework)
        dep = Deployment(id=f"dep-{uuid.uuid4().hex[:8]}", model_name=model_name,
                         pool=pool, replicas=replicas, status=DeploymentStatus.LOADING)

        dep.stable = self._make_slot(stable_mv, 100 - (canary_traffic if canary_selector else 0))

        if canary_selector and canary_traffic > 0:
            canary_mv = self._registry.resolve(model_name, canary_selector)
            if canary_mv is None:
                raise KeyError(f"canary selector '{canary_selector}' not found")
            dep.canary = self._make_slot(canary_mv, canary_traffic)

        # Derive status from what actually warmed.
        if dep.stable is None or not dep.stable.healthy:
            dep.status = DeploymentStatus.FAILED
            dep.last_error = "stable version failed to load"
        elif dep.canary is not None and not dep.canary.healthy:
            dep.status = DeploymentStatus.DEGRADED
            dep.last_error = "canary failed to load; serving stable only"
        else:
            dep.status = DeploymentStatus.READY

        with self._lock:
            self._deployments[dep.id] = dep
            self._by_model[model_name] = dep.id
        return dep

    def _make_slot(self, mv: ModelVersion, traffic: int) -> VersionSlot:
        backend = build_backend(mv)
        healthy = True
        try:
            backend.load()
        except BackendUnavailable:
            healthy = False
        return VersionSlot(version=mv, backend=backend, traffic=traffic, healthy=healthy)

    def get(self, deployment_id: str) -> Optional[Deployment]:
        return self._deployments.get(deployment_id)

    def list(self) -> List[Deployment]:
        return list(self._deployments.values())

    def scale(self, deployment_id: str, replicas: int) -> Deployment:
        dep = self._require(deployment_id)
        if replicas < 0:
            raise ValueError("replicas must be >= 0")
        dep.replicas = replicas
        return dep

    def promote_canary(self, deployment_id: str) -> Deployment:
        """Cut the canary over to 100% stable — the 'canary looks good, ship it'
        action. Idempotent if there is no canary."""
        dep = self._require(deployment_id)
        if dep.canary is not None and dep.canary.healthy:
            dep.stable = dep.canary
            dep.stable.traffic = 100
            dep.canary = None
            dep.status = DeploymentStatus.READY
            dep.last_error = None
        return dep

    # ── routing ───────────────────────────────────────────────────
    def route(self, model_name: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], VersionSlot]:
        """Route one inference to a live deployment for `model_name`.

        Splits traffic stable/canary by the configured weights, then executes.
        Reliability contract: if the *canary* errors, the request transparently
        retries on stable (the caller still gets an answer, the canary's failure
        counter ticks, and repeated failures flip the deployment to DEGRADED).
        This is the platform-level expression of 'bad model deployed → automatic
        fallback' from docs/platform/RELIABILITY.md."""
        dep_id = self._by_model.get(model_name)
        if dep_id is None:
            raise KeyError(f"no deployment for model '{model_name}'")
        dep = self._deployments[dep_id]
        if dep.stable is None or not dep.stable.healthy:
            raise BackendUnavailable(f"deployment {dep.id} has no healthy stable version")

        slot = self._pick_slot(dep)
        try:
            result = slot.backend.predict(payload)
            slot.requests += 1
            return result, slot
        except BackendUnavailable:
            slot.failures += 1
            # Canary failure → fall back to stable. Stable failure → propagate.
            if slot is dep.canary:
                self._maybe_degrade(dep)
                result = dep.stable.backend.predict(payload)
                dep.stable.requests += 1
                return result, dep.stable
            raise

    def _pick_slot(self, dep: Deployment) -> VersionSlot:
        if dep.canary is not None and dep.canary.healthy and dep.canary.traffic > 0:
            if self._rng.randint(1, 100) <= dep.canary.traffic:
                return dep.canary
        return dep.stable

    def _maybe_degrade(self, dep: Deployment, threshold: int = 3) -> None:
        if dep.canary is not None and dep.canary.failures >= threshold:
            dep.canary.healthy = False
            dep.status = DeploymentStatus.DEGRADED
            dep.last_error = "canary auto-disabled after repeated failures"

    def _require(self, deployment_id: str) -> Deployment:
        dep = self._deployments.get(deployment_id)
        if dep is None:
            raise KeyError(f"deployment '{deployment_id}' not found")
        return dep
