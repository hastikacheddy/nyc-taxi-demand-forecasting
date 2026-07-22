# ADR-003 — KServe for model serving

- **Status:** Accepted
- **Date:** 2026-07
- **Deciders:** Platform eng

## Context

The existing CPU serving path is a hand-rolled FastAPI `Deployment` + `Service` +
HPA — excellent, hardened, but per-model bespoke. Scaling to N models with canary,
versioned rollouts, scale-to-zero, and GPU concurrency scaling by hand would mean
re-implementing a serving controller. The platform needs a serving layer that
provides these as declarative primitives across both CPU and GPU.

## Decision

Adopt **KServe** (on Knative) as the Kubernetes-native model-serving layer,
expressed as `InferenceService` resources (`kubernetes/serving/`). Keep the
existing hardened FastAPI Deployment as the *lower tier* for the case where a bespoke
API surface is wanted; KServe is the default for standardized model serving.

KServe gives, as declarative fields, exactly the capabilities the brief calls for:
- **autoscaling** (incl. **scale-to-zero**, essential for GPU cost),
- **canary** via `canaryTrafficPercent`,
- **rolling** revision-based deploys,
- **model versions** via `storageUri`,
- **concurrency-based scaling** (correct signal for GPUs, see SCALING).

The platform's in-process `DeploymentManager` mirrors these semantics so control
logic is unit-testable without a cluster; KServe is the production actuator.

## Consequences

**Positive**
- One serving abstraction spans CPU (LightGBM) and GPU (vLLM) — see the two
  parallel manifests.
- Canary + scale-to-zero + concurrency scaling out of the box → less bespoke code.
- Standard, inspectable resources (`kubectl get inferenceservices`).

**Negative / accepted cost**
- KServe + Knative add cluster components to operate. Accepted: they replace code
  we'd otherwise own and get wrong.
- Some loss of fine-grained control vs a hand-rolled server; the FastAPI tier
  remains for the exceptions.

## Alternatives considered

- **Seldon Core** — comparable capability, strong for inference graphs; KServe
  chosen for tighter Knative scale-to-zero and simpler single-model UX.
- **Ray Serve** — excellent for Python-composed multi-model / ensemble logic and
  shares the Ray substrate used for distributed training; a strong candidate and a
  likely *second* serving backend for composition-heavy workloads. Not the default
  because KServe's declarative K8s-native model is a better fit for standardized,
  ops-owned serving.
- **Raw Deployments (status quo)** — what we have; doesn't scale to N models with
  canary/versioning without reinventing a controller. Kept as the lower tier only.
- **Triton Inference Server** — best-in-class multi-framework/GPU perf; heavier
  model-repository workflow. A candidate backend under KServe, not a replacement
  for it.
