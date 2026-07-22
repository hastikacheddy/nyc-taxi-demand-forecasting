# ADR-004 — A framework-neutral platform API that stores metadata + pointers, not weights

- **Status:** Accepted
- **Date:** 2026-07
- **Deciders:** Platform eng

## Context

The platform must address radically different models — a 2 MB LightGBM pickle and
a 140 GB LLM shard set — through one API, and let teams self-serve register →
deploy → infer without platform-team involvement. The central design question:
what is the platform's unit of currency, and where do model weights live relative
to the control plane?

## Decision

The platform API is **framework-neutral** and the registry stores **metadata and a
pointer (`artifact_uri`), never the weights**. A `ModelVersion`
(`src/platform/registry.py`) is `name + version + framework + artifact_uri +
sha256 + ResourceProfile + aliases`. Weights live in object storage; the registry
is the *index*. Backends (`src/platform/backends.py`) are selected by `framework`
behind a 4-method contract (`load / predict / health / kind`), so the control
plane never branches on model type.

Corollaries:
- **Compute placement is derived, not requested** — `framework.needs_gpu` →
  `ComputePool` → node affinity/tolerations. Callers don't pick GPUs.
- **Selectors, not versions, in the call** — callers say `@champion`; the registry
  resolves to a concrete version. Enables canary/rollback without caller changes.

## Consequences

**Positive**
- The same control plane, scheduler, and metrics serve any framework; adding
  Triton/ONNX is a ~40-line backend, not an API change.
- Weights-as-pointers means the registry scales independently of model size and
  stays cheap/fast; artifact immutability gives instant rollback (RELIABILITY).
- Testable without heavy deps — the `echo` backend exercises the whole platform;
  MLflow/GPU aren't needed to prove the abstraction (11 passing tests).

**Negative / accepted cost**
- An extra indirection vs. calling a model server directly. Justified: the
  indirection *is* the platform (placement, canary, fallback, metrics, multi-tenancy).
- The default in-process `InMemoryStore` must become a shared DB before running >1
  gateway replica. Accepted and bounded: it's a `RegistryStore` implementation swap
  behind an interface that already exists.

## Alternatives considered

- **One serving stack, one framework** (e.g. everything as MLflow pyfunc). Simple,
  but can't serve a vLLM LLM well and couples the platform to a training tool.
  Rejected.
- **Registry stores the weights.** Ties registry cost/latency to model size, makes
  the control plane a data-plane bottleneck, and duplicates object storage.
  Rejected.
- **Caller specifies compute** (asks for a GPU explicitly). Leaks infra into every
  team's code and defeats central FinOps. Rejected — placement is the platform's
  job.
