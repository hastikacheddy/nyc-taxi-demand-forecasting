"""
Declarative workloads — "deploy any AI workload" from a spec.

The gateway's JSON API is imperative ("register this, then deploy that"). A real
platform also takes a *declarative* workload spec — the user describes the desired
end state and the platform reconciles to it. This is the mini-CRD at the heart of
the platform: the same spec shape describes a CPU LightGBM model and a GPU LLM,
and the WorkloadManager turns either into a live deployment via the existing
registry + DeploymentManager.

    name: taxi-forecast          name: llama-chat
    type: ml-model               type: llm
    runtime: lightgbm            runtime: vllm
    artifact_uri: models:/...    artifact_uri: http://vllm:8000/v1
    resources: {cpu, memory}     resources: {gpu: 1}
    scaling: {min, max}          scaling: {min: 1, max: 4}

One declarative interface, two radically different workloads — which is exactly
what makes it a platform rather than a service. Example specs live in
src/platform/schemas/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import yaml

from src.platform.deployments import DeploymentManager
from src.platform.registry import Framework, ModelRegistry, ResourceProfile

# workload runtime → serving framework/backend
_RUNTIME_TO_FRAMEWORK = {
    "lightgbm": Framework.LIGHTGBM,
    "sklearn": Framework.SKLEARN,
    "onnx": Framework.ONNX,
    "vllm": Framework.VLLM,
    "transformers": Framework.TRANSFORMERS,
    "echo": Framework.ECHO,
}

# workload type is the human-facing category; each maps to allowed runtimes
_VALID_TYPES = {"ml-model", "llm", "embedding"}


class WorkloadError(ValueError):
    """Invalid workload spec (bad type/runtime, missing field, etc.)."""


@dataclass
class WorkloadSpec:
    name: str
    type: str
    runtime: str
    artifact_uri: str
    resources: ResourceProfile
    min_replicas: int = 1
    max_replicas: int = 1
    canary_selector: Optional[str] = None
    canary_traffic: int = 0
    alias: str = "champion"
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def framework(self) -> Framework:
        fw = _RUNTIME_TO_FRAMEWORK.get(self.runtime)
        if fw is None:
            raise WorkloadError(f"unknown runtime '{self.runtime}' "
                                f"(known: {sorted(_RUNTIME_TO_FRAMEWORK)})")
        return fw

    @classmethod
    def from_dict(cls, d: dict) -> "WorkloadSpec":
        if not isinstance(d, dict):
            raise WorkloadError("workload spec must be a mapping")
        # tolerate an optional top-level 'workload:' wrapper
        if "workload" in d and isinstance(d["workload"], dict):
            inner = dict(d["workload"])
            inner.update({k: v for k, v in d.items() if k != "workload"})
            d = inner

        for req in ("name", "type", "runtime", "artifact_uri"):
            if not d.get(req):
                raise WorkloadError(f"workload spec missing required field '{req}'")
        if d["type"] not in _VALID_TYPES:
            raise WorkloadError(f"type '{d['type']}' must be one of {sorted(_VALID_TYPES)}")

        res = d.get("resources") or {}
        gpu = int(res.get("gpu", 0) or 0)
        resources = ResourceProfile(
            cpu=str(res.get("cpu", "500m")),
            memory=str(res.get("memory", "512Mi")),
            gpu=gpu,
            gpu_type=res.get("gpu_type") or ("nvidia-a100" if gpu else None),
        )
        scaling = d.get("scaling") or {}
        rollout = d.get("rollout") or {}
        return cls(
            name=d["name"],
            type=d["type"],
            runtime=d["runtime"],
            artifact_uri=d["artifact_uri"],
            resources=resources,
            min_replicas=int(scaling.get("min", scaling.get("min_replicas", 1))),
            max_replicas=int(scaling.get("max", scaling.get("max_replicas", 1))),
            canary_selector=rollout.get("canary_selector"),
            canary_traffic=int(rollout.get("canary_traffic", 0) or 0),
            alias=d.get("alias", "champion"),
            tags={k: str(v) for k, v in (d.get("tags") or {}).items()},
        )

    @classmethod
    def from_yaml(cls, text: str) -> "WorkloadSpec":
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise WorkloadError(f"invalid YAML: {e}") from e
        return cls.from_dict(data)


class WorkloadManager:
    """Reconciles a WorkloadSpec into a live deployment. `apply` is idempotent at
    the level of 'register a new version + (re)deploy' — calling it again ships a
    new version of the same workload, which is exactly rolling-update semantics."""

    def __init__(self, registry: ModelRegistry, deployments: DeploymentManager) -> None:
        self.registry = registry
        self.deployments = deployments

    def apply(self, spec: WorkloadSpec) -> dict:
        # 1. register the model version described by the workload
        mv = self.registry.register(
            name=spec.name,
            framework=spec.framework,
            artifact_uri=spec.artifact_uri,
            resources=spec.resources,
            description=f"{spec.type} workload ({spec.runtime})",
            tags=spec.tags,
        )
        # 2. move the alias the deployment selects onto this new version
        self.registry.set_alias(spec.name, mv.version, spec.alias)
        # 3. create/replace the deployment (pool is derived from the framework)
        dep = self.deployments.create(
            spec.name,
            selector=spec.alias,
            replicas=spec.min_replicas,
            canary_selector=spec.canary_selector,
            canary_traffic=spec.canary_traffic,
        )
        return {
            "workload": spec.name,
            "type": spec.type,
            "runtime": spec.runtime,
            "version": mv.version,
            "pool": dep.pool.value,
            "min_replicas": spec.min_replicas,
            "max_replicas": spec.max_replicas,
            "deployment": dep.to_public(),
        }
