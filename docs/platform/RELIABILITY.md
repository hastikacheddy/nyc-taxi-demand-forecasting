# AI Platform — Reliability Design Document

> The document you write before an incident, not during one. It states, for every
> component, **what happens when it fails** —
> the detection signal, the automatic response, the blast radius, and the manual
> fallback — and it names the code or manifest that makes the claim true.

Scope: the platform serving path (`src/platform/`), the model lifecycle it
inherits from the existing MLOps repo, and the GPU/cluster substrate in
[GPU_DESIGN](GPU_DESIGN.md). Recovery-time targets are in [DISASTER_RECOVERY](DISASTER_RECOVERY.md).

---

## 1. Reliability model in one line

**Every dependency is assumed to fail; the platform degrades in defined steps
rather than falling over.** Concretely: a failed *canary* degrades to *stable*; a
failed *stable* degrades to the *last-known-good artifact*; a failed *registry*
degrades to the *cached artifact*; a failed *GPU node* degrades to *reschedule +
priority preemption*. Each step is a smaller promise kept, never a 500.

---

## 2. "What happens if…" — the failure catalogue

| # | Failure | Detection | Automatic response | Blast radius | Owner action |
|---|---|---|---|---|---|
| 1 | **Model server pod dies** | liveness probe fails | K8s restarts the pod; other replicas serve | one replica; none if `replicas≥2` | none (alert only if crashloop) |
| 2 | **GPU node disappears** (spot reclaim / Xid) | node `NotReady`, DCGM Xid | pods rescheduled; `platform-online-serving` priority preempts training to free a GPU | GPU models on that node, seconds | replace node; check spot strategy |
| 3 | **Bad model deployed** (canary regresses) | canary error rate / `BackendUnavailable` | request falls back to **stable**; after N failures canary auto-disabled → `DEGRADED` | requests routed to canary (e.g. 10%) | inspect canary, fix or drop |
| 4 | **Bad model, but it *returns* garbage** | anomaly guard (output > sane bound) | request rejected 500 → caller retries stable path; promotion gate would've blocked it | single request | investigate model |
| 5 | **Registry unavailable** (MLflow down) | resolve() error at deploy | serving uses the **already-warmed backend** (weights cached in-pod); no new deploys | *deploys blocked*, serving unaffected | restore MLflow |
| 6 | **Concept drift** | monitoring_dag: realised MAE +15% | closed loop **triggers retraining**; promotion gate decides if it ships | forecast quality, hours | review retrain |
| 7 | **Tampered model artifact** | SHA-256 mismatch pre-load | load **refused**; deployment stays on prior version | zero (fails closed) | security review |
| 8 | **Traffic spike / GPU saturated** | Knative concurrency > target | scale up to `maxReplicas`; excess requests queue then shed | latency rises before scale-out completes | raise max / add nodes |
| 9 | **CUDA OOM** | pod OOMKilled | restart; if persistent, lower `--gpu-memory-utilization` | one replica | tune vLLM args |
| 10 | **Whole region down** | health checks | out of scope for single-region MVP | total | see DISASTER_RECOVERY multi-region |

Rows 1–3, 5, 7 are the ones worth scrutinising hardest. Note which are **already
enforced in code** below.

---

## 3. Claims backed by code (not aspirations)

Reliability docs lose credibility when they describe behavior the code doesn't
have. These are the ones that are real today:

- **Row 3 — canary → stable fallback.** `DeploymentManager.route()`
  (`src/platform/deployments.py`) catches `BackendUnavailable` from the canary,
  retries on stable, increments the canary's failure counter, and flips the
  deployment to `DEGRADED` past a threshold. Proven by
  `test_canary_failure_falls_back_to_stable_then_degrades`.
- **Row 7 — artifact integrity fail-closed.** `src/inference/model_integrity.py`
  verifies SHA-256 before load; a mismatch raises rather than deserializing a
  possibly-tampered pickle.
- **Row 6 — closed drift→retrain loop.** `src/monitoring/scoring.py` joins matured
  forecasts to actuals and triggers retraining at +15% realised MAE.
- **Row 4 — output anomaly guard + promotion gate.** `src/serving/api.py` rejects
  predictions beyond a sane bound; `src/common/promotion.py` blocks a retrain from
  taking `@champion` unless it beats the incumbent's holdout MAE.

Rows 1, 2, 5, 8, 9 are **substrate guarantees** — provided by Kubernetes / KServe /
the manifests in `kubernetes/serving/`, documented in [GPU_DESIGN](GPU_DESIGN.md),
and true once deployed to a cluster.

---

## 4. Degradation ladders (explicit, per subsystem)

```
SERVING          stable+canary → stable only (DEGRADED) → last-good artifact → 503 (shed)
REGISTRY         live MLflow  → in-pod cached weights → block new deploys (serving OK)
GPU CAPACITY     dedicated GPU → preempt training → queue (concurrency scaler) → shed
DATA/FEATURES    fresh online store → cached recent history (CSV) → stale-but-valid → refuse
```

The design rule: **each arrow is a smaller, still-correct promise.** Never jump
straight from "healthy" to "down".

---

## 5. Multi-tenancy blast-radius isolation

A platform serves many teams; one team must not sink another.

- **GPU `ResourceQuota`** (`kubernetes/serving/gpu-resourcequota.yaml`) caps GPUs
  per namespace — a greedy vLLM config can't consume the whole pool.
- **PriorityClasses** ensure one team's fine-tuning never preempts another team's
  online serving (training is globally lowest, `preemptionPolicy: Never`).
- **Rate limiting** (existing `src/serving/api.py`, 60 req/min/IP) bounds a single
  caller; the platform gateway inherits the same pattern.
- **Per-model deployments** mean a crashlooping model affects only its own
  replicas, not the gateway or sibling models.

---

## 6. Observability that makes failures *visible*

Reliability requires that every row above emits a signal:

| Signal | Source | Alerts on |
|---|---|---|
| `platform_inference_requests_total{outcome}` | gateway | rising `unavailable` / `not_found` |
| `platform_deployments{status="degraded\|failed"}` | gateway | any degraded/failed deployment |
| `platform_inference_latency_seconds` p95/p99 | gateway | SLO breach |
| DCGM GPU util / Xid / memory | DCGM exporter | Xid errors, thermal, OOM |
| realised MAE, coverage | monitoring_dag | drift threshold |

These compose with the existing Prometheus + Grafana stack — the platform did not
invent a parallel observability system, it extended the one already here.

---

## 7. What is *not* covered (stated honestly)

- **Multi-region failover** — single-region MVP; the path is sketched in
  DISASTER_RECOVERY but not built.
- **Exactly-once inference** — the platform is at-least-once; idempotency is the
  caller's concern for non-idempotent side effects (forecasting is read-only, so
  moot here).
- **Automated canary *analysis*** — promotion is currently a human decision on top
  of metrics; automated statistical canary analysis (e.g. Kayenta-style) is a
  named next step, not a claim.
