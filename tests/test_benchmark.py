"""
Tests for the inference benchmark harness.

Uses the MockBackend so it runs anywhere (no GPU, no model server). Proves the
*measurement* is correct: percentiles are ordered, throughput scales with
concurrency, GPU sampling degrades honestly to N/A, and the compare() table
renders.
"""
import math

from benchmarks.harness import (
    benchmark, compare, _percentile, GpuSampler,
    benchmark_streaming, compare_streaming,
)
from benchmarks.backends import build_backend, MockBackend, MockStreamBackend


def test_percentile_nearest_rank():
    vals = [float(i) for i in range(1, 101)]  # 1..100 sorted
    assert _percentile(vals, 50) == 50
    assert _percentile(vals, 99) == 99
    assert _percentile(vals, 100) == 100
    assert math.isnan(_percentile([], 50))


def test_benchmark_reports_ordered_percentiles():
    backend = MockBackend(base_latency_ms=20, max_concurrency=8)
    res = benchmark(backend, {"max_tokens": 16}, label="unit",
                    n_requests=60, concurrency=8, warmup=3)
    lat = res.latency_ms
    assert lat["p50"] <= lat["p95"] <= lat["p99"] <= lat["max"]
    assert res.errors == 0
    assert res.n_requests == 60
    assert res.throughput_rps > 0


def test_higher_concurrency_raises_throughput():
    # A batching backend (high concurrency ceiling) should serve more rps when
    # driven with more concurrency — the core inference-optimization signal.
    backend = MockBackend(base_latency_ms=40, max_concurrency=16)
    low = benchmark(backend, {}, label="c2", n_requests=40, concurrency=2, warmup=2)
    high = benchmark(backend, {}, label="c16", n_requests=40, concurrency=16, warmup=2)
    assert high.throughput_rps > low.throughput_rps


def test_gpu_sampler_honest_when_absent():
    s = GpuSampler()
    summ = s.summary()
    # On a CPU-only box this must be a clean N/A, never a fabricated number.
    assert "available" in summ
    if not summ["available"]:
        assert summ["util_mean_pct"] is None


def test_compare_table_and_delta():
    b = MockBackend(base_latency_ms=120, max_concurrency=2)
    a = MockBackend(base_latency_ms=40, max_concurrency=16)
    rb = benchmark(b, {}, label="Vanilla", n_requests=30, concurrency=8, warmup=2)
    ra = benchmark(a, {}, label="vLLM", n_requests=30, concurrency=8, warmup=2)
    table = compare([rb, ra])
    assert "p50 (ms)" in table and "Throughput (rps)" in table
    assert "Vanilla" in table and "vLLM" in table
    assert "higher throughput" in table  # delta line rendered


def test_build_backend_factory():
    assert isinstance(build_backend("mock"), MockBackend)
    assert isinstance(build_backend("mock-stream"), MockStreamBackend)
    oai = build_backend("vllm", base_url="http://x:8000/v1", model="m")
    assert oai.base_url == "http://x:8000/v1"


# ── Streaming (LLM) metrics: TTFT + tokens/sec ─────────────────────
def test_streaming_reports_ttft_and_throughput():
    be = MockStreamBackend(ttft_ms=30, per_token_ms=2, tokens=16, max_concurrency=8)
    res = benchmark_streaming(be.stream, {"max_tokens": 16}, label="s",
                              n_requests=24, concurrency=8, warmup=2)
    assert res.ttft_ms["p50"] <= res.ttft_ms["p95"] <= res.ttft_ms["p99"]
    assert res.ttft_ms["p50"] >= 25          # ~ the configured 30ms prefill
    assert res.tokens_per_sec > 0
    assert res.errors == 0


def test_streaming_ttft_reflects_prefill():
    slow = MockStreamBackend(ttft_ms=120, per_token_ms=1, tokens=8, max_concurrency=8)
    fast = MockStreamBackend(ttft_ms=20, per_token_ms=1, tokens=8, max_concurrency=8)
    slow_r = benchmark_streaming(slow.stream, {}, label="slow", n_requests=16, concurrency=8, warmup=1)
    fast_r = benchmark_streaming(fast.stream, {}, label="fast", n_requests=16, concurrency=8, warmup=1)
    assert fast_r.ttft_ms["p50"] < slow_r.ttft_ms["p50"]   # lower prefill → lower TTFT


def test_compare_streaming_table():
    a = MockStreamBackend(ttft_ms=200, per_token_ms=20, tokens=16, max_concurrency=2)
    b = MockStreamBackend(ttft_ms=40, per_token_ms=4, tokens=16, max_concurrency=8)
    ra = benchmark_streaming(a.stream, {}, label="Vanilla", n_requests=16, concurrency=8, warmup=1)
    rb = benchmark_streaming(b.stream, {}, label="vLLM", n_requests=16, concurrency=8, warmup=1)
    table = compare_streaming([ra, rb])
    assert "TTFT p50 (ms)" in table and "Throughput (tok/s)" in table
    assert "token throughput" in table
