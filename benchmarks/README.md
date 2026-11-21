# `benchmarks/` — Inference optimization

Measures the numbers that decide production serving cost and latency — **p50 / p95 /
p99 latency, throughput, and GPU utilization** — and renders a before/after table
comparing a baseline (vanilla HF Transformers) against an optimized server (vLLM).

The *methodology* is the deliverable and it is fully committed. The harness runs
three ways:

| Mode | Deps | Produces |
|---|---|---|
| **Self-demo** (`--demo`) | none | mock "vanilla vs vLLM" — proves the harness, shows the result shape |
| **Real servers** (`--before-url/--after-url`) | a running vLLM / TGI endpoint | **real** p50/p95/p99 + throughput on your hardware |
| **In-process HF** (`HFPipelineBackend`) | `torch`, `transformers` | the literal vanilla baseline |

## Run it

```bash
# zero-dependency self-demo (CI-safe, no GPU)
python -m benchmarks.run_benchmark --demo --requests 200 --concurrency 16

# LLM streaming metrics — TTFT (time-to-first-token) + tokens/sec, vanilla vs vLLM
python -m benchmarks.run_benchmark --stream-demo --requests 100 --concurrency 8

# real comparison: point 'before' at a vanilla/TGI server, 'after' at vLLM
python -m benchmarks.run_benchmark \
    --before-url http://hf-tgi:8080/v1 --before-model my-model \
    --after-url  http://vllm:8000/v1   --after-model  my-model \
    --requests 500 --concurrency 32 --max-tokens 128 \
    --out benchmarks/results/vllm-vs-hf.md
```

## Illustrative result (mock, [`results/demo-mock.md`](results/demo-mock.md))

| Config | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (rps) | GPU util |
|---|---:|---:|---:|---:|---:|
| Vanilla HF (mock) | ~187 | high tail | high tail | ~11 | N/A (CPU box) |
| vLLM (mock) | ~67 | ~67 | ~67 | ~220 | N/A (CPU box) |

> ⚠️ **These are mock numbers**, generated on a CPU-only machine to demonstrate the
> harness — the mock models the *shape* of the difference (continuous batching →
> higher effective concurrency → flat tail latency + higher throughput), not real
> model performance. On a real A100 the axes are identical; only the magnitudes are
> real. The harness reports GPU utilization as `N/A` here precisely because there is
> no GPU — it never fabricates a number.

## Why these metrics

- **p99, not mean** — tail latency is what users feel and what SLOs are written
  against; a good mean can hide a terrible p99 from queuing.
- **throughput under concurrency** — measured with a wall-clock window at fixed
  concurrency, so it captures queuing, not just per-request service time.
- **GPU utilization** — a fast model on a 15%-utilized A100 is a *cost* failure;
  throughput × utilization is the bridge from latency to **$/request**
  ([COST_MODEL](../docs/platform/COST_MODEL.md)).
- **cold start, reported separately** — the first request (weights + CUDA context)
  is 30–120 s and must not pollute steady-state percentiles; it's its own number
  because it drives the scale-to-zero vs min-replicas decision ([SCALING](../docs/platform/SCALING.md)).
- **TTFT + tokens/sec for LLMs** (`--stream-demo`, `benchmark_streaming`) — for a
  streaming LLM, total latency hides the two numbers that matter: **time-to-first-token**
  (what the user feels before text appears) and **tokens/sec** decode throughput
  (what sets $/token). vLLM's continuous batching wins both; the harness measures
  them over the same OpenAI-compatible streaming API vLLM and TGI expose.
