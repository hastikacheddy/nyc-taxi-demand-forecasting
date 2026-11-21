"""
Tests for the declarative workload abstraction (src/platform/workloads.py).

Proves the platform's "deploy any AI workload" contract: one spec shape drives a
CPU LightGBM model and a GPU LLM onto the right pool, and the shipped example
specs parse into exactly what they claim.
"""
import os

import pytest

from src.platform.registry import ModelRegistry, Framework
from src.platform.deployments import DeploymentManager
from src.platform.workloads import WorkloadSpec, WorkloadManager, WorkloadError

_SCHEMAS = os.path.join(os.path.dirname(__file__), "..", "src", "platform", "schemas")


def _mgr():
    reg = ModelRegistry()
    return reg, DeploymentManager(reg), WorkloadManager(reg, DeploymentManager(reg))


# ── Spec parsing ───────────────────────────────────────────────────
def test_shipped_examples_parse():
    taxi = WorkloadSpec.from_yaml(open(os.path.join(_SCHEMAS, "taxi-forecast.yaml")).read())
    assert taxi.type == "ml-model" and taxi.framework is Framework.LIGHTGBM
    assert taxi.resources.gpu == 0 and taxi.resources.cpu == "4"
    assert (taxi.min_replicas, taxi.max_replicas) == (2, 10)

    llm = WorkloadSpec.from_yaml(open(os.path.join(_SCHEMAS, "llm-chat.yaml")).read())
    assert llm.type == "llm" and llm.framework is Framework.VLLM
    assert llm.resources.gpu == 1 and llm.resources.gpu_type == "nvidia-a100"


def test_workload_wrapper_key_tolerated():
    spec = WorkloadSpec.from_yaml(
        "workload:\n  name: w\n  type: ml-model\n  runtime: echo\nartifact_uri: s3://a\n")
    assert spec.name == "w" and spec.framework is Framework.ECHO


def test_invalid_specs_rejected():
    with pytest.raises(WorkloadError):
        WorkloadSpec.from_dict({"name": "x", "type": "ml-model", "runtime": "echo"})  # no artifact_uri
    with pytest.raises(WorkloadError):
        WorkloadSpec.from_dict({"name": "x", "type": "bogus", "runtime": "echo", "artifact_uri": "s3://a"})
    with pytest.raises(WorkloadError):
        WorkloadSpec.from_yaml(": : not yaml : :")
    bad_rt = WorkloadSpec.from_dict({"name": "x", "type": "llm", "runtime": "nope", "artifact_uri": "s3://a"})
    with pytest.raises(WorkloadError):
        _ = bad_rt.framework


# ── Apply → register + deploy on the right pool ────────────────────
def test_apply_cpu_workload():
    reg = ModelRegistry()
    mgr = WorkloadManager(reg, DeploymentManager(reg))
    spec = WorkloadSpec.from_yaml(
        "name: demo\ntype: ml-model\nruntime: echo\nartifact_uri: s3://a\n"
        "resources: {cpu: '2', memory: 4Gi}\nscaling: {min: 3, max: 8}\n")
    out = mgr.apply(spec)
    assert out["pool"] == "cpu"
    assert out["version"] == 1
    assert out["deployment"]["status"] == "ready"
    assert out["deployment"]["replicas"] == 3       # min_replicas
    # the model is now registered and champion-aliased
    assert "champion" in reg.resolve("demo", "champion").aliases


def test_apply_gpu_workload_lands_on_gpu_pool():
    reg = ModelRegistry()
    mgr = WorkloadManager(reg, DeploymentManager(reg))
    spec = WorkloadSpec.from_yaml(
        "name: llm\ntype: llm\nruntime: vllm\nartifact_uri: http://vllm:8000/v1\n"
        "resources: {gpu: 1}\n")
    out = mgr.apply(spec)
    assert out["pool"] == "gpu"                      # derived from runtime, not requested


def test_apply_twice_ships_new_version():
    reg = ModelRegistry()
    mgr = WorkloadManager(reg, DeploymentManager(reg))
    spec = WorkloadSpec.from_yaml("name: m\ntype: ml-model\nruntime: echo\nartifact_uri: s3://a\n")
    assert mgr.apply(spec)["version"] == 1
    assert mgr.apply(spec)["version"] == 2           # rolling update = new version


# ── Gateway endpoint ───────────────────────────────────────────────
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from src.platform import gateway
    gateway.registry = ModelRegistry()
    gateway.deployments = DeploymentManager(gateway.registry)
    return TestClient(gateway.app)


def test_gateway_apply_workload_yaml(client):
    yaml_body = ("name: demo\ntype: ml-model\nruntime: echo\n"
                 "artifact_uri: s3://a\nscaling: {min: 2, max: 5}\n")
    r = client.post("/v1/workloads", content=yaml_body,
                    headers={"Content-Type": "application/yaml"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["pool"] == "cpu" and body["version"] == 1
    # the workload is now inferable through the same gateway
    inf = client.post("/v1/inference", json={"model": "demo", "inputs": {"x": 1}})
    assert inf.status_code == 200
    assert inf.json()["result"]["framework"] == "echo"


def test_gateway_apply_workload_invalid_422(client):
    r = client.post("/v1/workloads", content="name: x\ntype: bogus\nruntime: echo\nartifact_uri: s3://a\n",
                    headers={"Content-Type": "application/yaml"})
    assert r.status_code == 422
