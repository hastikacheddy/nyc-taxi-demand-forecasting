# ADR-001 — Kubernetes (AKS) as the platform substrate

- **Status:** Accepted
- **Date:** 2026-07
- **Deciders:** Platform eng

## Context

The platform must run heterogeneous workloads — CPU APIs, GPU LLM serving,
batch/distributed training, Airflow — for multiple teams, with autoscaling,
isolation, and a declarative recovery story. The existing repo already deploys to
Kubernetes for the CPU serving path; the question is whether to stay on it as the
*platform* substrate or move serving to a managed, higher-level product (e.g. a
serverless GPU endpoint service).

## Decision

Use **Kubernetes**, specifically **Azure AKS**, as the substrate for all planes.
Managed control plane; a system **CPU node pool** plus a tainted, autoscaling
**GPU node pool**; ACR for images, Blob for artifacts, Key Vault for secrets,
Azure Monitor alongside the in-cluster Prometheus/Grafana.

Azure specifically because it fits the repo's existing direction and gives
first-class private networking + Key Vault integration; the design is portable
(nothing depends on an Azure-only primitive) so EKS/GKE are a node-pool and
IAM remap, not a rewrite.

## Consequences

**Positive**
- One scheduler for CPU + GPU + batch → GPU FinOps (priorities, preemption,
  quotas) is *possible at all* (see GPU_DESIGN, COST_MODEL).
- Declarative topology → DR is "re-apply config" (DISASTER_RECOVERY).
- KServe/Knative/Ray all run natively on it → serving and distributed-training
  choices stay open (ADR-002/003).
- Reuses the repo's existing hardening (OPA, non-root, seccomp).

**Negative / accepted cost**
- Kubernetes operational complexity is real; mitigated by using *managed* AKS and
  keeping cluster state in Git.
- GPU scheduling has sharp edges (whole-GPU allocation, cold starts) — addressed
  explicitly in GPU_DESIGN rather than wished away.

## Alternatives considered

- **Managed serverless GPU endpoints (e.g. SageMaker/Vertex endpoints).** Lower
  ops burden, but weak multi-tenant GPU economics, vendor lock-in, and no shared
  substrate for training + Airflow + serving. Rejected: the platform's value is
  the *shared* substrate.
- **VMs + systemd.** Simplest, but no autoscaling, no bin-packing, no declarative
  DR, no GPU scheduling primitives. Rejected.
- **Nomad.** Lighter than K8s, but far smaller AI-serving ecosystem (no KServe).
  Rejected on ecosystem.
