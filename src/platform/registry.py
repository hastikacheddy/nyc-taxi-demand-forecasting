"""
Model registry abstraction for the AI platform.

The platform must serve *many* models built with *many* frameworks (a LightGBM
forecaster today, a vLLM-served LLM tomorrow). It therefore cannot bind to one
training stack's notion of "a model". This module defines a small, framework-
neutral registry:

    ModelVersion   an immutable record: name + version + framework + artifact URI
                   + integrity digest + resource profile + aliases/tags
    ModelRegistry  register / get / list / alias, with pluggable storage

Two storage backends ship:

  * InMemoryStore  — zero-dependency, so the control plane (and its tests) run
                     standalone. This is the default.
  * MlflowStore    — mirrors registrations into the existing MLflow registry the
                     rest of the repo already uses, so the platform is a *view*
                     over real infrastructure rather than a parallel universe.

Design choice (see docs/adr/004-platform-api-abstraction.md): the registry stores
*metadata and a pointer*, never the weights. Artifacts live in object storage
(MinIO/S3/Blob); the registry is the index. This is what lets the same control
plane address a 2 MB LightGBM pickle and a 140 GB LLM shard set identically.
"""
from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Protocol


class Framework(str, enum.Enum):
    """Serving framework a model version is built for. Drives compute-pool
    placement and backend selection downstream."""
    LIGHTGBM = "lightgbm"
    SKLEARN = "sklearn"
    ONNX = "onnx"
    TRANSFORMERS = "transformers"   # HF pipeline, GPU
    VLLM = "vllm"                   # LLM served by a vLLM OpenAI-compatible server
    ECHO = "echo"                   # trivial deterministic backend (demos/tests)

    @property
    def needs_gpu(self) -> bool:
        return self in (Framework.TRANSFORMERS, Framework.VLLM)


@dataclass(frozen=True)
class ResourceProfile:
    """What one replica of this model needs. The scheduler turns this into pod
    resource requests + node affinity (see kubernetes/serving/)."""
    cpu: str = "500m"
    memory: str = "512Mi"
    gpu: int = 0
    gpu_type: Optional[str] = None   # e.g. "nvidia-a100", "nvidia-l4"

    @staticmethod
    def for_framework(fw: Framework) -> "ResourceProfile":
        if fw == Framework.VLLM:
            return ResourceProfile(cpu="4", memory="24Gi", gpu=1, gpu_type="nvidia-a100")
        if fw == Framework.TRANSFORMERS:
            return ResourceProfile(cpu="2", memory="16Gi", gpu=1, gpu_type="nvidia-l4")
        return ResourceProfile(cpu="500m", memory="512Mi", gpu=0)


@dataclass(frozen=True)
class ModelVersion:
    name: str
    version: int
    framework: Framework
    artifact_uri: str
    resources: ResourceProfile
    sha256: Optional[str] = None          # integrity digest (see model_integrity.py)
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    # aliases pointing at THIS version, e.g. {"champion", "canary"}.
    aliases: frozenset = field(default_factory=frozenset)

    def to_public(self) -> dict:
        d = asdict(self)
        d["framework"] = self.framework.value
        d["aliases"] = sorted(self.aliases)
        d["needs_gpu"] = self.framework.needs_gpu
        return d


class RegistryStore(Protocol):
    def put(self, mv: ModelVersion) -> None: ...
    def get(self, name: str, version: int) -> Optional[ModelVersion]: ...
    def latest(self, name: str) -> Optional[ModelVersion]: ...
    def by_alias(self, name: str, alias: str) -> Optional[ModelVersion]: ...
    def list(self) -> List[ModelVersion]: ...
    def set_alias(self, name: str, version: int, alias: str) -> None: ...


class InMemoryStore:
    """Thread-safe in-process registry. Default backend so the platform runs with
    no external services — the control plane is honest about what it is: an index."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # name -> {version -> ModelVersion}
        self._versions: Dict[str, Dict[int, ModelVersion]] = {}
        # (name, alias) -> version
        self._aliases: Dict[tuple, int] = {}

    def put(self, mv: ModelVersion) -> None:
        with self._lock:
            self._versions.setdefault(mv.name, {})[mv.version] = mv

    def get(self, name: str, version: int) -> Optional[ModelVersion]:
        with self._lock:
            return self._versions.get(name, {}).get(version)

    def latest(self, name: str) -> Optional[ModelVersion]:
        with self._lock:
            vers = self._versions.get(name)
            if not vers:
                return None
            return vers[max(vers)]

    def by_alias(self, name: str, alias: str) -> Optional[ModelVersion]:
        with self._lock:
            v = self._aliases.get((name, alias))
            return self._versions.get(name, {}).get(v) if v is not None else None

    def list(self) -> List[ModelVersion]:
        with self._lock:
            return [mv for vers in self._versions.values() for mv in vers.values()]

    def set_alias(self, name: str, version: int, alias: str) -> None:
        with self._lock:
            if version not in self._versions.get(name, {}):
                raise KeyError(f"{name} v{version} not registered")
            # move alias off any previous version, attach to the new one
            self._aliases[(name, alias)] = version
            self._reindex_aliases(name)

    def _reindex_aliases(self, name: str) -> None:
        """Recompute the frozenset of aliases on each ModelVersion of `name`."""
        per_version: Dict[int, set] = {}
        for (n, alias), v in self._aliases.items():
            if n == name:
                per_version.setdefault(v, set()).add(alias)
        for v, mv in self._versions.get(name, {}).items():
            new_aliases = frozenset(per_version.get(v, set()))
            if new_aliases != mv.aliases:
                self._versions[name][v] = ModelVersion(**{**asdict_shallow(mv), "aliases": new_aliases})


def asdict_shallow(mv: ModelVersion) -> dict:
    """Shallow field copy that preserves enum/dataclass fields (asdict would
    recurse and turn ResourceProfile into a plain dict)."""
    return {
        "name": mv.name, "version": mv.version, "framework": mv.framework,
        "artifact_uri": mv.artifact_uri, "resources": mv.resources,
        "sha256": mv.sha256, "description": mv.description, "tags": mv.tags,
        "created_at": mv.created_at, "aliases": mv.aliases,
    }


class ModelRegistry:
    """Framework-neutral registry facade the control plane calls."""

    def __init__(self, store: Optional[RegistryStore] = None) -> None:
        self._store = store or InMemoryStore()

    def register(
        self,
        name: str,
        framework: Framework,
        artifact_uri: str,
        *,
        resources: Optional[ResourceProfile] = None,
        sha256: Optional[str] = None,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelVersion:
        """Register a new *version* of `name`. Versions are monotonic per name."""
        latest = self._store.latest(name)
        version = (latest.version + 1) if latest else 1
        mv = ModelVersion(
            name=name,
            version=version,
            framework=framework,
            artifact_uri=artifact_uri,
            resources=resources or ResourceProfile.for_framework(framework),
            sha256=sha256,
            description=description,
            tags=tags or {},
        )
        self._store.put(mv)
        return mv

    def resolve(self, name: str, selector: str = "latest") -> Optional[ModelVersion]:
        """Resolve a caller reference to a concrete version.

        `selector` is one of: "latest", "v<N>" / "<N>", or an alias
        ("champion", "canary", ...). This is the indirection that lets callers
        say "give me the champion" without knowing the version number."""
        if selector == "latest":
            return self._store.latest(name)
        if selector.lstrip("v").isdigit():
            return self._store.get(name, int(selector.lstrip("v")))
        return self._store.by_alias(name, selector)

    def set_alias(self, name: str, version: int, alias: str) -> ModelVersion:
        self._store.set_alias(name, version, alias)
        mv = self._store.get(name, version)
        assert mv is not None
        return mv

    def list(self) -> List[ModelVersion]:
        return sorted(self._store.list(), key=lambda m: (m.name, m.version))


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
