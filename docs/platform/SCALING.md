# AI Platform — Scaling

> How the platform grows from one CPU model to N models across CPU + GPU pools,
> and the scaling *signals* that differ fundamentally between the two.

---

## 1. The core insight: CPU and GPU scale on different signals

| | CPU serving (LightGBM) | GPU serving (vLLM) |
|---|---|---|
| Scaling signal | CPU utilization (HPA @70%) | **in-flight concurrency / queue depth** |
| Why | tree inference is CPU-bound | a saturated A100 can be at *low* CPU while its queue grows |
| Min replicas | 2 (always-on) | 1 latency-critical, 0 for dev (scale-to-zero) |
| Cold start | ~2 s | **30–120 s** (weights + CUDA context) |
| Scale unit | pod | whole GPU |

Using CPU% to autoscale a GPU model is a classic mistake — it under-scales under
load (GPU busy, CPU idle) and the latency SLO blows before a replica is added.
KServe/Knative concurrency scaling (`kserve-llm-vllm.yaml`) is the correct signal.

---

## 2. Three axes of scale

**Scale up (bigger model).** A model too large for one GPU shards across several
with tensor parallelism (`--tensor-parallel-size`). This trades inter-GPU
communication for capacity and is why the GPU pool co-locates GPUs on high-bandwidth
nodes.

**Scale out (more traffic).** Add replicas. CPU: HPA 2→6. GPU: Knative 1→4 on
concurrency, bounded by the GPU `ResourceQuota` so scale-out can't starve other
tenants.

**Scale wide (more models).** The platform's reason to exist. Each model is an
independent deployment addressed by name; adding the 50th model is a registry
`register` + `deployments.create`, not new infra. The gateway, scheduler, and
metrics are shared; only backends differ.

---

## 3. Bottlenecks, in the order you'll hit them

1. **GPU cold start** → keep `minReplicas≥1` for hot paths; pre-pull images;
   consider model-weight caching on node-local NVMe.
2. **Whole-GPU allocation waste** → MIG / right-sized nodes (L4 for embeddings).
   See [GPU_DESIGN §3](GPU_DESIGN.md).
3. **KV-cache memory** (LLM) → `--max-model-len` and `--gpu-memory-utilization`
   bound it; batch size trades throughput for latency.
4. **Registry / control-plane DB** → the in-process store must become a shared DB
   before running >1 gateway replica (the abstraction boundary is already there —
   swap `InMemoryStore` for a DB-backed `RegistryStore`).
5. **Object storage egress** on large weights → node-local cache + pre-warm.

---

## 4. Load evidence available today

The repo ships a Locust profile (`loadtest/`) against the CPU serving path and a
backend-agnostic inference benchmark (`benchmarks/`, see
[COST_MODEL](COST_MODEL.md)) that measures p50/p95/p99 and throughput. The vLLM
side is measured with the same harness once a GPU endpoint is available — the
methodology is committed even where the A100 is not.

---

## 5. Capacity planning (worked example)

The production forecast is ~**117k trips/day** with a peak-hour target of ~2.6k
trips/h. If each downstream planning cycle issues one forecast per zone per hour,
that's a bounded, low-QPS workload → the CPU pool at 2 replicas is over-provisioned
for correctness/HA, not throughput. The LLM copilot path is the opposite: bursty,
expensive per token, latency-sensitive → concurrency-scaled GPU with scale-to-zero
off-hours. **Same platform, opposite capacity shapes** — which is exactly why the
placement decision is derived from the model, not hard-coded.
