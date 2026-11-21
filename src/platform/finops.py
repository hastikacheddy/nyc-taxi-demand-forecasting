"""
FinOps — cost attribution and waste detection for the platform.

AI-infrastructure spend is dominated by *idle accelerators*, not compute you use, so the
platform treats cost as a first-class, per-model observable (see
docs/platform/COST_MODEL.md). This module turns the deployment state the control
plane already tracks — pool, GPU type/count, replicas — plus request counts into:

    * hourly $ per deployment and per model,
    * $ per 1k requests (the number tied to throughput / inference optimization),
    * idle-GPU detection (low utilization → wasted accelerator).

Rates are illustrative on-demand list prices; override with real numbers (or a
cloud billing export) via CostRates. The point is the *method* — attribution +
waste detection — not the specific dollar figures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.platform.deployments import ComputePool, Deployment


@dataclass
class CostRates:
    """$/hour. GPU rates are per-GPU; multiply by the GPU count on the replica."""
    cpu_node_hourly: float = 0.30           # a general 8-vCPU node
    gpu_hourly: Dict[str, float] = field(default_factory=lambda: {
        "nvidia-a100": 3.60,
        "nvidia-l4": 0.70,
        "default": 3.00,
    })

    def gpu_rate(self, gpu_type: Optional[str]) -> float:
        return self.gpu_hourly.get(gpu_type or "default", self.gpu_hourly["default"])


@dataclass
class DeploymentCost:
    deployment_id: str
    model_name: str
    pool: str
    replicas: int
    hourly_usd: float
    per_1k_requests_usd: Optional[float]   # None if no traffic observed
    rps: float
    gpu_util_pct: Optional[float]
    idle: bool

    def as_dict(self) -> dict:
        return self.__dict__


class CostModel:
    def __init__(self, rates: Optional[CostRates] = None,
                 idle_util_threshold_pct: float = 5.0) -> None:
        self.rates = rates or CostRates()
        self.idle_util_threshold = idle_util_threshold_pct

    def replica_hourly(self, pool: ComputePool, gpu_type: Optional[str], gpu_count: int) -> float:
        """Cost of one replica for an hour."""
        if pool is ComputePool.GPU and gpu_count > 0:
            return self.rates.gpu_rate(gpu_type) * gpu_count
        return self.rates.cpu_node_hourly

    def deployment_hourly(self, dep: Deployment) -> float:
        """Total hourly cost of a deployment = replicas × per-replica rate,
        derived from the stable version's resource profile."""
        if dep.stable is None:
            return 0.0
        res = dep.stable.version.resources
        return dep.replicas * self.replica_hourly(dep.pool, res.gpu_type, res.gpu)

    def cost_per_1k_requests(self, dep: Deployment, rps: float) -> Optional[float]:
        """$/1000 requests at the given sustained requests/sec. This is the dollar
        translation of throughput: double the rps (e.g. via vLLM batching) and
        this halves — see COST_MODEL.md."""
        if rps <= 0:
            return None
        hourly = self.deployment_hourly(dep)
        cost_per_request = hourly / (rps * 3600.0)
        return round(cost_per_request * 1000.0, 6)

    def is_idle_gpu(self, dep: Deployment, gpu_util_pct: Optional[float]) -> bool:
        """A GPU deployment whose utilization is below threshold is burning money.
        CPU deployments and unknown-utilization ones are never flagged idle."""
        if dep.pool is not ComputePool.GPU:
            return False
        if gpu_util_pct is None:
            return False
        return gpu_util_pct < self.idle_util_threshold

    def price_deployment(
        self,
        dep: Deployment,
        rps: float = 0.0,
        gpu_util_pct: Optional[float] = None,
    ) -> DeploymentCost:
        hourly = self.deployment_hourly(dep)
        return DeploymentCost(
            deployment_id=dep.id,
            model_name=dep.model_name,
            pool=dep.pool.value,
            replicas=dep.replicas,
            hourly_usd=round(hourly, 4),
            per_1k_requests_usd=self.cost_per_1k_requests(dep, rps),
            rps=round(rps, 3),
            gpu_util_pct=gpu_util_pct,
            idle=self.is_idle_gpu(dep, gpu_util_pct),
        )

    def report(
        self,
        deployments: List[Deployment],
        rps_of: Optional[Callable[[Deployment], float]] = None,
        util_of: Optional[Callable[[Deployment], Optional[float]]] = None,
    ) -> Dict[str, object]:
        """Cost report across all deployments: per-deployment rows, total burn,
        and the idle-GPU waste list (the actionable FinOps output)."""
        rows = [
            self.price_deployment(
                d,
                rps=(rps_of(d) if rps_of else 0.0),
                gpu_util_pct=(util_of(d) if util_of else None),
            )
            for d in deployments
        ]
        total_hourly = round(sum(r.hourly_usd for r in rows), 4)
        idle = [r for r in rows if r.idle]
        idle_waste_hourly = round(sum(r.hourly_usd for r in idle), 4)
        return {
            "deployments": [r.as_dict() for r in rows],
            "total_hourly_usd": total_hourly,
            "total_monthly_usd": round(total_hourly * 730, 2),
            "idle_gpu_deployments": [r.deployment_id for r in idle],
            "idle_gpu_waste_hourly_usd": idle_waste_hourly,
            "idle_gpu_waste_monthly_usd": round(idle_waste_hourly * 730, 2),
        }
