# AI Platform — Cost Model (FinOps)

> AI infrastructure is expensive and the cost is dominated by **idle
> accelerators**, not compute you use. This document defines how the platform
> attributes, measures, and controls cost — cost/request, cost/model, idle-GPU
> detection, and automatic scale-down.

Illustrative rates (US on-demand, order-of-magnitude — verify against your bill):

| Resource | ~$/hr | Note |
|---|---|---|
| A100 80GB node | ~$3.6 | LLM serving / fine-tuning |
| L4 node | ~$0.7 | embeddings / mid models |
| General CPU node (8 vCPU) | ~$0.30 | APIs, Airflow, LightGBM |

The 10× gap between an idle A100 and an idle CPU node is the entire reason GPU
FinOps exists.

---

## 1. The one equation that matters

```
cost/request = (node $/hr) / (3600 × requests_per_second_per_replica × utilization)
```

Two levers dominate the denominator:

- **utilization** — an A100 at 15% utilization makes every request ~6× its floor
  cost. This is why batching (vLLM continuous batching) and right-sizing matter
  more than raw model speed.
- **requests_per_second_per_replica** — higher throughput per replica amortizes
  the fixed node cost. This is the direct dollar translation of the
  inference-optimization benchmark (`benchmarks/`): vLLM's higher throughput isn't
  just a latency win, it's a **cost/request division**.

---

## 2. Worked cost/request (illustrative)

| Path | Node | Throughput/replica | Utilization | ≈ cost / 1k req |
|---|---|---|---|---|
| LightGBM forecast (CPU) | $0.30/hr | ~500 rps | high | ~$0.0002 |
| LLM vanilla HF (1 GPU) | $3.6/hr | ~20 rps | low | ~$0.05 |
| LLM vLLM (1 GPU) | $3.6/hr | ~150 rps | high | ~$0.007 |

The LLM path is **~250× more expensive per request** than the forecast — which is
why the platform refuses to let a LightGBM model land on a GPU, and why vLLM vs
vanilla is a ~7× cost swing on the *same hardware*. FinOps and inference
optimization are the same lever viewed from two angles.

---

## 3. Attribution: cost/model and cost/team

Every deployment carries its `ComputePool`, `ResourceProfile`, and replica count —
enough to compute a running cost even before a cloud billing export:

```
model_hourly_cost = replicas × node_rate(pool, gpu_type)
```

Joined with `platform_inference_requests_total{model}` (already exported), this
yields **cost/request per model** and, via the tenant namespace, **cost/team** —
the numbers a platform owner is asked for in every quarterly review. A Grafana
panel over these two series is the "GPU cost dashboard."

---

## 4. Idle-GPU detection & auto scale-down (the biggest save)

| Control | Mechanism | Saves |
|---|---|---|
| **Scale-to-zero** dev/eval GPU endpoints | KServe `minReplicas: 0` | full node cost off-hours |
| **Idle detection** | DCGM GPU-util < 5% for 15 min → alert / cordon | reclaims stranded A100s |
| **Preempt training onto idle serving GPUs** | PriorityClasses | training runs "for free" on slack |
| **Right-sizing** | L4 not A100 for small models | ~5× per-hour |
| **Spot for training** | pre-emptible + checkpointing | up to ~70% on training |
| **MIG partitioning** | slice A100 into 7 | packs small models |

The single highest-ROI control is **scale-to-zero + accepting cold starts** for
everything that isn't latency-critical. The second is **not buying an A100 for an
L4 job.** Neither requires clever code — they require the placement discipline the
platform already encodes.

---

## 5. Cost as a first-class metric

The platform treats cost like latency: a labelled Prometheus series, an SLO-style
budget, and an alert when a model's cost/request regresses (e.g. a config change
drops vLLM utilization). "This model got 3× more expensive last week" should page
the same way "this model got 3× slower" does. That framing — cost as an
observable, not a monthly surprise — is the FinOps posture a mature platform
brings.
