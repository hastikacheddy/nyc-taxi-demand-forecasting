# Architecture Decision Records

Short, dated records of the decisions that shaped the platform — the *why*, the
alternatives weighed, and the trade-off accepted. Good architecture is largely the
act of making these decisions legible.

Format: [MADR](https://adr.github.io/madr/)-lite — Context · Decision · Consequences · Alternatives.

| ADR | Decision | Status |
|---|---|---|
| [001](001-why-kubernetes.md) | Kubernetes (AKS) as the platform substrate | Accepted |
| [002](002-why-vllm.md) | vLLM as the default LLM serving engine | Accepted |
| [003](003-model-serving-choice.md) | KServe for model serving over Seldon / raw Deployments / Ray Serve | Accepted |
| [004](004-platform-api-abstraction.md) | A framework-neutral platform API that stores metadata + pointers, not weights | Accepted |
