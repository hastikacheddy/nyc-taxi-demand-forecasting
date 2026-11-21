# AI Platform — Architecture

> How a single demand-forecasting model became a **multi-tenant internal AI
> platform**: one control plane that serves a 2 MB LightGBM forecaster and a
> GPU-served LLM through the *same* API, scheduler, and observability stack.

This document is the map. Deep-dives live in the sibling docs:
[SCALING](SCALING.md) · [GPU_DESIGN](GPU_DESIGN.md) · [RELIABILITY](RELIABILITY.md) ·
[COST_MODEL](COST_MODEL.md) · [DISASTER_RECOVERY](DISASTER_RECOVERY.md). Decisions
are recorded as [ADRs](../adr/).

---

## 1. The shift: from *a model* to *a platform*

The original repo shipped one model well — leakage-free forecaster, VaR risk
bands, a promotion gate, a drift loop, a hardened `/predict`. That is a strong
**MLOps** story. It is not yet a **platform** story.

The difference is the question each answers:

| MLOps asks | Platform asks |
|---|---|
| "Is *my* model in production and healthy?" | "Can *any team* ship *any* model without me?" |
| One serving path (`/predict`) | One control plane, N model types, N teams |
| I deploy | Self-service deploy + canary + rollback |
| CPU inference | CPU **and** GPU pools, scheduled by cost |

The platform layer (`src/platform/`) is the answer to the right-hand column. It
does **not** replace the model — it *wraps* it, and makes room for the next ten.

```
        Developers / internal services
                     │
                     ▼
        ┌─────────────────────────────┐
        │   AI Platform API (gateway) │   src/platform/gateway.py
        │   /v1/models  /v1/deployments│
        │   /v1/inference  /v1/metrics │
        └───────┬───────────────┬──────┘
      control   │               │  data plane
      plane     ▼               ▼
   ┌──────────────────┐  ┌────────────────────┐
   │ ModelRegistry    │  │ DeploymentManager  │  src/platform/
   │ framework-neutral│  │ pool placement,    │
   │ index + aliases  │  │ canary, fallback   │
   └───────┬──────────┘  └─────────┬──────────┘
           │                       │ build_backend()
           │                       ▼
           │            ┌────────────────────────────┐
           │            │ InferenceBackend (uniform)  │  src/platform/backends.py
           │            │  Echo · LightGBM · vLLM     │
           │            └───────┬──────────┬──────────┘
           ▼                    ▼          ▼
   MLflow registry      DemandForecast   vLLM server (GPU)
   (source of truth)    Engine (CPU)     OpenAI-compatible API
```

The critical property: **the caller never branches on model type.** A team POSTs
`{"model": "risk-copilot", "inputs": {...}}` and the platform resolves the
version, picks the pool, routes traffic (including canary), executes the right
backend, records metrics, and falls back on failure. Adding Triton or KServe is a
new `InferenceBackend` subclass — ~40 lines — not an API change.

---

## 2. The three planes

### Control plane — *what should be running*
- **`ModelRegistry`** (`registry.py`) — a framework-neutral index. A
  `ModelVersion` is `name + version + framework + artifact_uri + sha256 +
  ResourceProfile + aliases`. It stores **metadata and a pointer, never weights**
  ([ADR-004](../adr/004-platform-api-abstraction.md)); the same record addresses
  a 2 MB pickle and a 140 GB shard set. Default storage is in-process (so the
  control plane runs standalone); the production binding mirrors into the
  existing MLflow registry.
- **`DeploymentManager`** (`deployments.py`) — owns live endpoints: which
  versions serve, the stable/canary traffic split, the compute pool, lifecycle
  status, and routing with fallback. It is an in-process **simulation of a
  control loop** — the real actuation (KServe `InferenceService`, HPA, GPU node
  pools) lives in [`kubernetes/serving/`](../../kubernetes/serving/). Keeping the
  control *logic* framework-free is what lets it be unit-tested without a cluster
  — the same "`src/` plays, Airflow conducts" split the rest of the repo uses.

### Data plane — *serving the request*
- **`InferenceBackend`** (`backends.py`) — a 4-method contract (`load`,
  `predict`, `health`, `kind`). `LightGBMForecastBackend` wraps the repo's real
  `DemandForecastEngine` (same code path as the batch DAGs — no train/serve
  skew). `VLLMBackend` speaks the OpenAI-compatible API to a vLLM server and is
  **honestly `unavailable`** when no endpoint is configured, rather than faking a
  response.

### Management plane — *is it healthy, and what does it cost*
- Prometheus metrics on `/v1/metrics`: `platform_inference_requests_total`,
  `platform_inference_latency_seconds`, `platform_deployments{status}`,
  `platform_registered_model_versions`. These compose with the existing
  Grafana stack. Cost attribution is layered on top in [COST_MODEL](COST_MODEL.md).

---

## 3. Request lifecycle (an inference)

1. `POST /v1/inference {"model": "taxi-demand-daily", "inputs": {...}}`
2. Gateway authenticates (constant-time API key, optional) and starts a latency timer.
3. `DeploymentManager.route()` finds the live deployment for the model.
4. Traffic split: with probability = `canary_traffic%`, the canary version is
   chosen; else stable.
5. The version's warmed `InferenceBackend.predict()` runs.
6. **Fallback:** if the *canary* raises `BackendUnavailable`, the request
   transparently retries on stable; the canary's failure counter ticks and, past
   a threshold, the deployment flips to `DEGRADED` and stops sending it traffic.
   ([RELIABILITY](RELIABILITY.md) — "bad model deployed → automatic fallback".)
7. Metrics recorded; response includes `served_by.version` and `latency_ms` so
   canary vs stable performance is directly comparable.

---

## 4. Compute placement (why a request lands where it does)

Placement is derived from the model's framework, not chosen by the caller:

| Framework | Pool | Node ask | Rationale |
|---|---|---|---|
| `lightgbm`, `sklearn`, `onnx` | **CPU** | `500m` / `512Mi` | µs-scale trees; a GPU would idle |
| `transformers` | **GPU** | 1× L4, `16Gi` | mid-size models, cost-efficient GPU |
| `vllm` | **GPU** | 1× A100, `24Gi` | KV-cache + paged attention want HBM |

`ComputePool.for_framework()` encodes this; the K8s translation (node affinity,
the `nvidia.com/gpu` taint toleration, `priorityClass`, GPU `ResourceQuota`) is in
[GPU_DESIGN](GPU_DESIGN.md). The reason to separate pools is **economic**: a
LightGBM forecast must never sit on an A100. See [COST_MODEL](COST_MODEL.md).

---

## 5. What is real vs. simulated (honesty ledger)

Being explicit about the line between the two is part of doing this honestly.

| Component | Status | Notes |
|---|---|---|
| Platform API (register/deploy/infer/metrics) | **Real, runs, tested** | `pytest tests/test_platform_api.py` — 11 tests |
| LightGBM backend | **Real** | wraps production `DemandForecastEngine` |
| Canary split + fallback + degrade | **Real, tested** | deterministic-RNG test asserts ~30% split |
| vLLM backend | **Real when pointed at a server** | set `VLLM_BASE_URL`; else honest `unavailable` |
| DeploymentManager control loop | **Simulated in-process** | production actuation = KServe manifests |
| GPU node pools / A100s | **Designed, not provisioned** | [GPU_DESIGN](GPU_DESIGN.md) + Terraform (planned) |

Nothing in this repo pretends to own 8 A100s. The engineering that *is* here — the
abstraction boundary, the routing/fallback logic, the pool model, the metrics —
is the part that transfers directly to a cluster that does.

---

## 6. Where each capability lives

| Concern | Path |
|---|---|
| Platform API gateway | [`src/platform/gateway.py`](../../src/platform/gateway.py) |
| Registry abstraction | [`src/platform/registry.py`](../../src/platform/registry.py) |
| Deployment + routing | [`src/platform/deployments.py`](../../src/platform/deployments.py) |
| Backends (LightGBM/vLLM) | [`src/platform/backends.py`](../../src/platform/backends.py) |
| GPU/CPU serving manifests | [`kubernetes/serving/`](../../kubernetes/serving/) |
| Existing model + MLOps | [`src/forecasting/`](../../src/forecasting/), [`src/serving/`](../../src/serving/) |
