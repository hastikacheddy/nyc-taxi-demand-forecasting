"""
Tests for FinOps cost attribution (src/platform/finops.py).

Verifies the numbers that matter in a cost review: GPU deployments cost far more
than CPU, $/1k-requests falls as throughput rises, and idle-GPU detection flags
exactly the wasteful accelerators (and never a CPU box).
"""
import random

from src.platform.registry import ModelRegistry, Framework
from src.platform.deployments import DeploymentManager
from src.platform.finops import CostModel, CostRates


def _mgr_with(model_name, framework):
    reg = ModelRegistry()
    reg.register(model_name, framework, "s3://a" if framework is Framework.ECHO else "http://vllm:8000/v1")
    return DeploymentManager(reg, rng=random.Random(0)), reg


def test_gpu_costs_more_than_cpu_per_hour():
    cm = CostModel()
    cpu_mgr, _ = _mgr_with("cpu-model", Framework.ECHO)
    cpu_dep = cpu_mgr.create("cpu-model", selector="latest")

    gpu_reg = ModelRegistry()
    gpu_reg.register("llm", Framework.VLLM, "http://vllm:8000/v1")
    gpu_mgr = DeploymentManager(gpu_reg)
    gpu_dep = gpu_mgr.create("llm", selector="latest")

    cpu_hourly = cm.deployment_hourly(cpu_dep)
    gpu_hourly = cm.deployment_hourly(gpu_dep)
    assert gpu_hourly > cpu_hourly
    # vLLM default profile is 1× A100 → matches the A100 rate
    assert abs(gpu_hourly - CostRates().gpu_hourly["nvidia-a100"]) < 1e-6


def test_cost_per_1k_requests_falls_with_throughput():
    cm = CostModel()
    mgr, _ = _mgr_with("m", Framework.ECHO)
    dep = mgr.create("m", selector="latest")
    low = cm.cost_per_1k_requests(dep, rps=10)
    high = cm.cost_per_1k_requests(dep, rps=100)
    assert low is not None and high is not None
    # 10× the throughput → ~1/10th the cost per request (values are rounded to
    # 6 dp, so compare with a tolerance that reflects that, not exact equality)
    assert high < low
    assert abs(high * 10 - low) < 1e-4
    # no traffic → undefined, not a divide-by-zero
    assert cm.cost_per_1k_requests(dep, rps=0) is None


def test_idle_gpu_detection():
    cm = CostModel(idle_util_threshold_pct=5.0)
    gpu_reg = ModelRegistry()
    gpu_reg.register("llm", Framework.VLLM, "http://vllm:8000/v1")
    gpu_dep = DeploymentManager(gpu_reg).create("llm", selector="latest")

    assert cm.is_idle_gpu(gpu_dep, gpu_util_pct=2.0) is True     # below threshold → wasteful
    assert cm.is_idle_gpu(gpu_dep, gpu_util_pct=80.0) is False   # busy → fine
    assert cm.is_idle_gpu(gpu_dep, gpu_util_pct=None) is False   # unknown → don't fabricate

    # a CPU deployment is never flagged idle-GPU, even at 0% util
    cpu_dep = _mgr_with("c", Framework.ECHO)[0].create("c", selector="latest")
    assert cm.is_idle_gpu(cpu_dep, gpu_util_pct=0.0) is False


def test_report_totals_and_waste():
    cm = CostModel()
    reg = ModelRegistry()
    reg.register("llm", Framework.VLLM, "http://vllm:8000/v1")
    mgr = DeploymentManager(reg)
    dep = mgr.create("llm", selector="latest")
    report = cm.report([dep], util_of=lambda d: 1.0)   # 1% util → idle
    assert report["total_hourly_usd"] > 0
    assert dep.id in report["idle_gpu_deployments"]
    assert report["idle_gpu_waste_monthly_usd"] > 0
    assert report["total_monthly_usd"] == round(report["total_hourly_usd"] * 730, 2)


def test_gateway_costs_endpoint():
    from fastapi.testclient import TestClient
    from src.platform import gateway
    gateway.registry = ModelRegistry()
    gateway.deployments = DeploymentManager(gateway.registry)
    client = TestClient(gateway.app)

    client.post("/v1/models", json={"name": "demo", "framework": "echo", "artifact_uri": "s3://d"})
    client.post("/v1/deployments", json={"model_name": "demo", "selector": "latest"})
    r = client.get("/v1/costs")
    assert r.status_code == 200
    body = r.json()
    assert "total_hourly_usd" in body and "deployments" in body
    assert body["deployments"][0]["pool"] == "cpu"
