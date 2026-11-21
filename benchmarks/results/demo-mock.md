# Inference benchmark results

| Config | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (rps) | GPU util | Errors |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla HF (mock) | 186.8 | 17750.1 | 18496.7 | 10.7 | N/A | 0 |
| vLLM (mock) | 66.9 | 67.1 | 67.2 | 229.0 | N/A | 0 |

**vLLM (mock) vs Vanilla HF (mock):** ~2.8× lower p50 latency, ~21.4× higher throughput.
