# GPU Infrastructure Design

> How the platform schedules scarce, expensive accelerators so that a CPU-cheap
> LightGBM forecast never lands on an A100, and GPU work is packed, prioritized,
> and pre-emptible.

Paired manifests: [`kubernetes/serving/`](../../kubernetes/serving/). Cost
consequences: [COST_MODEL](COST_MODEL.md).

---

## 1. Two pools, one cluster

```
                         AKS / EKS cluster
   ┌───────────────────────────┬───────────────────────────────┐
   │        CPU pool           │           GPU pool             │
   │  (general Standard nodes) │   (A100 / L4, tainted)         │
   ├───────────────────────────┼───────────────────────────────┤
   │ • platform gateway        │ • vLLM LLM serving             │
   │ • LightGBM /predict (HPA) │ • Transformers embeddings      │
   │ • Airflow workers         │ • fine-tuning / LoRA jobs      │
   │ • Prometheus / Grafana    │ • batch offline inference      │
   └───────────────────────────┴───────────────────────────────┘
        no taint                  taint: nvidia.com/gpu=present:NoSchedule
```

**Why a taint on the GPU pool.** Without it, the Kubernetes scheduler will happily
place a stateless CPU pod (the gateway, a Prometheus sidecar) onto a GPU node
because it has spare CPU — silently occupying a $3/hr node with $0.10/hr work.
The `NoSchedule` taint makes GPU nodes **opt-in**: only pods that explicitly
*tolerate* it land there. The platform adds that toleration exactly when
`ResourceProfile.gpu > 0` (`ComputePool.for_framework` in `registry.py`).

---

## 2. The four scheduling primitives

| Primitive | What it does here | Manifest |
|---|---|---|
| **Taints / tolerations** | GPU nodes repel non-GPU pods; GPU pods opt in | `kserve-llm-vllm.yaml` |
| **Node affinity** | Pin vLLM to A100 nodes, embeddings to L4 (`accelerator` label) | `kserve-llm-vllm.yaml` |
| **Resource requests/limits** | `nvidia.com/gpu: 1` — whole-GPU allocation (K8s can't fractionalize by default) | both |
| **PriorityClass + preemption** | Online serving out-ranks batch/fine-tuning; can evict it | `gpu-priorityclasses.yaml` |
| **ResourceQuota** | Cap total GPUs a namespace/team may hold — the multi-tenant guardrail | `gpu-resourcequota.yaml` |

### Priority: who gets the GPU when they're all busy

```
system-critical      (kube-system)                    2_000_000_000
platform-online-serving   vLLM, embeddings  ← evicts →   1_000_000
platform-batch-inference  offline scoring                  100_000
platform-training         fine-tuning / LoRA                10_000   (pre-emptible)
```

A fine-tuning job runs on otherwise-idle A100s and is **evicted** the moment an
online-serving pod needs the GPU. This is the single most important GPU-economics
decision: training soaks up idle capacity but never starves production latency.

---

## 3. Why whole-GPU, and the fractional-GPU escape hatch

A vanilla cluster allocates GPUs in whole units — you cannot request `0.3` of an
A100. For an LLM that fills HBM with KV-cache, whole-GPU is correct. But a small
embedding model wasting 80% of an A100 is the second-biggest waste after idle
nodes. Three mitigations, in increasing order of complexity:

1. **Right-size the node.** Embeddings → L4 (24 GB, ~$0.7/hr), not A100
   (80 GB, ~$3+/hr). Node affinity by `accelerator` label does this today.
2. **Pack multiple models per GPU with MIG** (A100 Multi-Instance GPU): slice one
   A100 into up to 7 isolated instances, expose as `nvidia.com/mig-1g.10gb`.
3. **Time-slice** via the NVIDIA device plugin for bursty, latency-tolerant work.

The platform's `ResourceProfile.gpu_type` is the seam where this decision is
expressed; the manifests start with option (1) because it is the highest ROI and
lowest operational risk.

---

## 4. Autoscaling GPU serving (the cold-start problem)

GPU serving cannot autoscale like the CPU API (HPA 2→6 on CPU%). Two differences
dominate:

- **Scale-to-zero matters more.** An idle A100 is ~30× the cost of an idle CPU
  node, so KServe/Knative scales LLM deployments **to zero** when idle.
- **Cold start is brutal.** Loading a multi-GB model + CUDA context is 30–120 s,
  not 2 s. So: keep `minReplicas: 1` for latency-critical models (pay to avoid
  cold starts), and only scale-to-zero for dev/eval endpoints. This is a *cost vs
  latency* dial, set per-model in the InferenceService.

Scaling signal is **concurrency / queue depth**, not CPU% — a saturated A100 can
sit at modest CPU while requests queue. See [SCALING](SCALING.md).

---

## 5. Failure modes specific to GPUs

| Event | Detection | Response |
|---|---|---|
| GPU node disappears (spot reclaim) | node NotReady | pods rescheduled; PriorityClass decides who lands first |
| `CUDA out of memory` | pod crashloop, OOM in logs | lower vLLM `--gpu-memory-utilization` / `--max-model-len`; alert |
| GPU falls off the bus (Xid errors) | DCGM exporter metric | cordon node, drain, replace; DCGM alert to on-call |
| Model too big for one GPU | fails to load | tensor-parallel across GPUs (`--tensor-parallel-size`) |

GPU observability needs the **DCGM exporter** (GPU util, memory, temperature, Xid
errors) alongside the app Prometheus metrics — app metrics alone can't see a
degrading GPU. This is the GPU analog of the existing Prometheus/Grafana stack.

---

## 6. Provisioning (Terraform, planned)

The Azure mapping (see [ADR-001](../adr/001-why-kubernetes.md)) is an AKS cluster
with a **system CPU node pool** plus a **GPU node pool** (`Standard_NC24ads_A100_v4`)
that is tainted, labeled `accelerator=nvidia-a100`, and set to `enableAutoScaling`
with `minCount: 0`. Terraform for this lands in `infra/azure/` (tracked
separately). The Kubernetes-side contract — taints, labels, quotas, priorities —
is already committed in `kubernetes/serving/` so the cluster is a fill-in, not a
redesign.
