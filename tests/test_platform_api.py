"""
Tests for the internal AI Platform layer (src/platform/).

These exercise the control plane end to end — registry → deployment → routing →
canary split → fallback → gateway HTTP — using the dependency-free `echo`
backend, so they need neither MLflow nor a GPU. The real LightGBM/vLLM backends
are integration-tested elsewhere (they wrap code already covered by the existing
suites); here we prove the *platform abstraction* is correct.
"""
import random

import pytest

from src.platform.registry import ModelRegistry, Framework, ResourceProfile
from src.platform.deployments import DeploymentManager, ComputePool, DeploymentStatus
from src.platform.backends import BackendUnavailable, build_backend, InferenceBackend


# ── Registry ───────────────────────────────────────────────────────
def test_register_increments_versions():
    reg = ModelRegistry()
    v1 = reg.register("m", Framework.ECHO, "s3://a")
    v2 = reg.register("m", Framework.ECHO, "s3://b")
    assert (v1.version, v2.version) == (1, 2)
    assert reg.resolve("m", "latest").version == 2
    assert reg.resolve("m", "1").version == 1
    assert reg.resolve("m", "v2").version == 2


def test_alias_resolution_and_reindex():
    reg = ModelRegistry()
    reg.register("m", Framework.ECHO, "s3://a")
    reg.register("m", Framework.ECHO, "s3://b")
    reg.set_alias("m", 1, "champion")
    assert reg.resolve("m", "champion").version == 1
    assert "champion" in reg.resolve("m", "1").aliases
    # moving the alias re-points it and clears the old version's alias set
    reg.set_alias("m", 2, "champion")
    assert reg.resolve("m", "champion").version == 2
    assert "champion" not in reg.resolve("m", "1").aliases


def test_resource_profile_gpu_by_framework():
    assert ResourceProfile.for_framework(Framework.VLLM).gpu == 1
    assert ResourceProfile.for_framework(Framework.LIGHTGBM).gpu == 0
    assert Framework.VLLM.needs_gpu and not Framework.LIGHTGBM.needs_gpu


# ── Compute-pool placement ─────────────────────────────────────────
def test_pool_selection_follows_framework():
    reg = ModelRegistry()
    reg.register("cpu-model", Framework.ECHO, "s3://a")
    reg.register("gpu-model", Framework.VLLM, "http://vllm:8000/v1")
    mgr = DeploymentManager(reg)
    cpu_dep = mgr.create("cpu-model", selector="latest")
    assert cpu_dep.pool is ComputePool.CPU
    assert cpu_dep.status is DeploymentStatus.READY
    # vLLM with no reachable server → stable fails to load → FAILED, not a crash
    gpu_dep = mgr.create("gpu-model", selector="latest")
    assert gpu_dep.pool is ComputePool.GPU


# ── Routing + canary ───────────────────────────────────────────────
def test_route_returns_served_version():
    reg = ModelRegistry()
    reg.register("m", Framework.ECHO, "s3://a")
    mgr = DeploymentManager(reg)
    mgr.create("m", selector="latest")
    result, slot = mgr.route("m", {"x": 1})
    assert slot.version.version == 1
    assert result["framework"] == "echo"
    assert slot.requests == 1


def test_canary_traffic_split_is_roughly_honored():
    reg = ModelRegistry()
    reg.register("m", Framework.ECHO, "s3://a")   # v1 stable
    reg.register("m", Framework.ECHO, "s3://b")   # v2 canary
    # deterministic RNG so the split is reproducible
    mgr = DeploymentManager(reg, rng=random.Random(0))
    dep = mgr.create("m", selector="1", canary_selector="2", canary_traffic=30)
    assert dep.status is DeploymentStatus.READY
    hits = {1: 0, 2: 0}
    for _ in range(2000):
        _, slot = mgr.route("m", {"x": 1})
        hits[slot.version.version] += 1
    canary_share = hits[2] / sum(hits.values())
    assert 0.24 < canary_share < 0.36     # ~30% ± noise


def test_promote_canary_cuts_over_to_stable():
    reg = ModelRegistry()
    reg.register("m", Framework.ECHO, "s3://a")
    reg.register("m", Framework.ECHO, "s3://b")
    mgr = DeploymentManager(reg)
    dep = mgr.create("m", selector="1", canary_selector="2", canary_traffic=50)
    dep = mgr.promote_canary(dep.id)
    assert dep.canary is None
    assert dep.stable.version.version == 2
    assert dep.stable.traffic == 100


class _FlakyBackend(InferenceBackend):
    """Always raises — used to prove canary failures fall back to stable."""
    def load(self): self._loaded = True
    def predict(self, payload): raise BackendUnavailable("boom")


def test_canary_failure_falls_back_to_stable_then_degrades(monkeypatch):
    reg = ModelRegistry()
    reg.register("m", Framework.ECHO, "s3://a")   # v1 stable (healthy echo)
    reg.register("m", Framework.ECHO, "s3://b")   # v2 canary (will be flaky)
    mgr = DeploymentManager(reg, rng=random.Random(1))

    real_build = build_backend

    def fake_build(mv):
        if mv.version == 2:
            return _FlakyBackend(mv)
        return real_build(mv)

    # patch the symbol the manager actually calls
    monkeypatch.setattr("src.platform.deployments.build_backend", fake_build)
    dep = mgr.create("m", selector="1", canary_selector="2", canary_traffic=100)

    # every request routes to the flaky canary, fails, and falls back to stable
    for _ in range(5):
        result, slot = mgr.route("m", {"x": 1})
        assert result["framework"] == "echo"    # answered by stable
        assert slot.version.version == 1
    # after repeated canary failures the deployment is DEGRADED
    assert dep.status is DeploymentStatus.DEGRADED
    assert dep.canary.failures >= 3


# ── Gateway HTTP surface ───────────────────────────────────────────
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from src.platform import gateway
    # isolate global state per test
    gateway.registry = ModelRegistry()
    gateway.deployments = DeploymentManager(gateway.registry)
    return TestClient(gateway.app)


def test_gateway_full_lifecycle(client):
    # register
    r = client.post("/v1/models", json={
        "name": "demo", "framework": "echo", "artifact_uri": "s3://demo"})
    assert r.status_code == 201, r.text
    assert r.json()["version"] == 1

    # list
    assert len(client.get("/v1/models").json()) == 1

    # deploy
    r = client.post("/v1/deployments", json={"model_name": "demo", "selector": "latest"})
    assert r.status_code == 201, r.text
    dep = r.json()
    assert dep["pool"] == "cpu" and dep["status"] == "ready"

    # status
    assert client.get(f"/v1/deployments/{dep['id']}").json()["id"] == dep["id"]

    # inference
    r = client.post("/v1/inference", json={"model": "demo", "inputs": {"a": 1}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["served_by"]["version"] == 1
    assert body["result"]["framework"] == "echo"
    assert "latency_ms" in body

    # metrics expose the platform counters
    m = client.get("/v1/metrics")
    assert m.status_code == 200
    assert "platform_inference_requests_total" in m.text
    assert "platform_registered_model_versions" in m.text


def test_gateway_inference_unknown_model_404(client):
    r = client.post("/v1/inference", json={"model": "nope", "inputs": {}})
    assert r.status_code == 404


def test_gateway_alias_flow(client):
    client.post("/v1/models", json={"name": "m", "framework": "echo", "artifact_uri": "s3://a"})
    client.post("/v1/models", json={"name": "m", "framework": "echo", "artifact_uri": "s3://b"})
    r = client.post("/v1/models/m/alias", json={"version": 2, "alias": "champion"})
    assert r.status_code == 200
    assert "champion" in r.json()["aliases"]
    # a deployment against @champion resolves to v2
    dep = client.post("/v1/deployments", json={"model_name": "m", "selector": "champion"}).json()
    assert dep["stable"]["version"] == 2
