"""Inference optimization benchmark suite.

`harness` is the measurement engine (percentiles, throughput, GPU sampling);
`backends` are the things measured (mock / vLLM / vanilla HF). See README.md."""
from benchmarks.harness import (
    benchmark, compare, BenchmarkResult, GpuSampler,
    benchmark_streaming, compare_streaming, StreamResult,
)
from benchmarks.backends import build_backend

__all__ = [
    "benchmark", "compare", "BenchmarkResult", "GpuSampler",
    "benchmark_streaming", "compare_streaming", "StreamResult",
    "build_backend",
]
