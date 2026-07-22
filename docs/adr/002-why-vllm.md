# ADR-002 — vLLM as the default LLM serving engine

- **Status:** Accepted
- **Date:** 2026-07
- **Deciders:** Platform eng

## Context

The platform needs to serve generative LLMs (a "risk copilot" over the forecasts,
future GenAI workloads). LLM serving is dominated by two costs: **GPU memory**
(KV-cache) and **throughput per GPU** (which sets cost/request, COST_MODEL). Naïve
serving with vanilla HuggingFace `transformers.generate()` gives low throughput
(~20 rps) and poor GPU utilization because it can't batch requests of different
lengths efficiently.

## Decision

Default LLM serving engine is **vLLM**, behind its OpenAI-compatible server, on
the GPU pool. The platform's `VLLMBackend` (`src/platform/backends.py`) speaks that
API, so an LLM is "just another registered model."

The two properties that decide it:
- **PagedAttention** — near-zero KV-cache fragmentation → more concurrent
  sequences per GPU → higher throughput at fixed memory.
- **Continuous batching** — new requests join the running batch each step instead
  of waiting for it to drain → high utilization under mixed load.

Together these are a ~5–8× throughput gain over vanilla HF on the same GPU — which
COST_MODEL shows is a ~7× cost/request reduction, not merely a latency win.

## Consequences

**Positive**
- Higher throughput → lower cost/request and better GPU utilization.
- OpenAI-compatible API → drop-in, and the platform backend is thin.
- Tensor-parallel sharding for models too big for one GPU.

**Negative / accepted cost**
- vLLM is optimized for throughput; ultra-low-latency single-request use cases may
  prefer TensorRT-LLM. Accepted: the platform's LLM use is concurrent serving.
- Another moving part to operate; mitigated by the maintained `vllm/vllm-openai`
  image and KServe management.

## Alternatives considered

- **Vanilla HF `transformers`** — the baseline the benchmark (`benchmarks/`)
  measures *against*; kept precisely to quantify the gap, not to serve.
- **TensorRT-LLM / Triton** — potentially lower latency, but heavier build/compile
  workflow and less flexible. A candidate second backend, not the default.
- **TGI (Text Generation Inference)** — comparable; also OpenAI-ish. vLLM chosen
  for PagedAttention maturity and ecosystem. The backend abstraction means
  swapping to TGI later is a config change (its adapter reuses the OpenAI path).
