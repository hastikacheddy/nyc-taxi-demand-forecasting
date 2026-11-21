"""
Inference benchmark harness — the measurement engine.

Backend-agnostic on purpose: it hammers *any* callable that maps a request to a
response and reports the numbers that decide production serving cost and latency — p50/p95/p99
latency, throughput under concurrency, and (when a GPU is present) utilization.

The methodology is the deliverable. Point it at a vLLM endpoint and it produces
the "after" column; point it at a vanilla HF server and it produces the "before".
Point it at the built-in mock and it self-demonstrates with zero dependencies —
which is how the harness is unit-tested without a GPU.

Design notes:
  * Percentiles use nearest-rank on the *measured* sample — no interpolation, no
    hidden smoothing, so a p99 is a real observed request.
  * Throughput is measured under a fixed concurrency with a wall-clock window, not
    inferred from mean latency (which would ignore queuing).
  * A warmup phase is discarded so cold-start (CUDA/JIT) doesn't pollute steady
    state — cold start is measured separately and reported on its own.
"""
from __future__ import annotations

import math
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

Request = Dict[str, Any]
InferFn = Callable[[Request], Any]


# ── GPU utilization sampling ───────────────────────────────────────
class GpuSampler:
    """Polls `nvidia-smi` in a background thread. If there's no GPU (or no
    nvidia-smi), it reports availability=False and utilization stays None —
    honest N/A rather than a fabricated number."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self._samples: List[float] = []
        self._mem: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.available = self._probe()

    @staticmethod
    def _probe() -> bool:
        try:
            subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=5, check=True)
            return True
        except Exception:
            return False

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=True,
                ).stdout.strip().splitlines()
                # average across GPUs present
                utils, mems = [], []
                for line in out:
                    u, m = line.split(",")
                    utils.append(float(u))
                    mems.append(float(m))
                if utils:
                    self._samples.append(sum(utils) / len(utils))
                    self._mem.append(sum(mems) / len(mems))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        if self.available:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> Dict[str, Optional[float]]:
        if not self.available or not self._samples:
            return {"available": self.available, "util_mean_pct": None,
                    "util_max_pct": None, "mem_used_mb_max": None}
        return {
            "available": True,
            "util_mean_pct": round(statistics.mean(self._samples), 1),
            "util_max_pct": round(max(self._samples), 1),
            "mem_used_mb_max": round(max(self._mem), 1) if self._mem else None,
        }


# ── Results ────────────────────────────────────────────────────────
def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Nearest-rank percentile on already-sorted values (ms).

    Rank (1-based) = ceil(P/100 · N); 0-based index = rank − 1, clamped. So the
    p99 of 100 samples is the 99th-ranked (index 98) observed value — a real
    request, no interpolation."""
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    k = math.ceil(pct / 100 * n) - 1
    k = max(0, min(n - 1, k))
    return sorted_vals[k]


@dataclass
class BenchmarkResult:
    label: str
    n_requests: int
    concurrency: int
    errors: int
    wall_time_s: float
    throughput_rps: float
    latency_ms: Dict[str, float]
    cold_start_ms: Optional[float] = None
    gpu: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_row(self) -> str:
        lat = self.latency_ms
        g = self.gpu.get("util_mean_pct")
        gpu = f"{g:.0f}%" if g is not None else "N/A"
        return (f"| {self.label} | {lat['p50']:.1f} | {lat['p95']:.1f} | {lat['p99']:.1f} "
                f"| {self.throughput_rps:.1f} | {gpu} | {self.errors} |")

    def as_dict(self) -> dict:
        return asdict(self)


TABLE_HEADER = (
    "| Config | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (rps) | GPU util | Errors |\n"
    "|---|---:|---:|---:|---:|---:|---:|"
)


# ── The benchmark ──────────────────────────────────────────────────
def benchmark(
    infer: InferFn,
    request: Request,
    *,
    label: str,
    n_requests: int = 200,
    concurrency: int = 10,
    warmup: int = 10,
) -> BenchmarkResult:
    """Drive `infer(request)` `n_requests` times at the given `concurrency`.

    Returns latency percentiles (ms), throughput (rps) under load, cold-start
    (first call, measured before warmup), and GPU utilization if available."""
    # cold start: the very first call, timed on its own
    t0 = time.perf_counter()
    try:
        infer(request)
        cold_start_ms = (time.perf_counter() - t0) * 1000
    except Exception:
        cold_start_ms = None

    # warmup (discarded) — reach steady state
    for _ in range(max(0, warmup)):
        try:
            infer(request)
        except Exception:
            pass

    latencies: List[float] = []
    errors = 0
    lock = threading.Lock()

    def one_call(_i: int) -> None:
        nonlocal errors
        s = time.perf_counter()
        try:
            infer(request)
            dt = (time.perf_counter() - s) * 1000
            with lock:
                latencies.append(dt)
        except Exception:
            with lock:
                errors += 1

    with GpuSampler() as gpu:
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one_call, i) for i in range(n_requests)]
            for f in as_completed(futures):
                f.result()
        wall = time.perf_counter() - wall_start

    latencies.sort()
    ok = len(latencies)
    return BenchmarkResult(
        label=label,
        n_requests=n_requests,
        concurrency=concurrency,
        errors=errors,
        wall_time_s=round(wall, 4),
        throughput_rps=round(ok / wall, 2) if wall > 0 else 0.0,
        latency_ms={
            "p50": round(_percentile(latencies, 50), 2),
            "p95": round(_percentile(latencies, 95), 2),
            "p99": round(_percentile(latencies, 99), 2),
            "mean": round(statistics.mean(latencies), 2) if ok else float("nan"),
            "max": round(latencies[-1], 2) if ok else float("nan"),
        },
        cold_start_ms=round(cold_start_ms, 2) if cold_start_ms is not None else None,
        gpu=gpu.summary(),
    )


# ── Streaming (LLM) benchmark: TTFT + tokens/sec ───────────────────
@dataclass
class StreamResult:
    label: str
    n_requests: int
    concurrency: int
    errors: int
    wall_time_s: float
    tokens_per_sec: float                 # aggregate decode throughput under load
    ttft_ms: Dict[str, float]             # time-to-first-token percentiles
    total_ms: Dict[str, float]            # full-generation latency percentiles
    gpu: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_row(self) -> str:
        t, tot = self.ttft_ms, self.total_ms
        g = self.gpu.get("util_mean_pct")
        gpu = f"{g:.0f}%" if g is not None else "N/A"
        return (f"| {self.label} | {t['p50']:.0f} | {t['p95']:.0f} | {t['p99']:.0f} "
                f"| {self.tokens_per_sec:.0f} | {tot['p50']:.0f} | {gpu} | {self.errors} |")

    def as_dict(self) -> dict:
        return asdict(self)


STREAM_TABLE_HEADER = (
    "| Config | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | Throughput (tok/s) "
    "| Total p50 (ms) | GPU util | Errors |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|"
)

# stream_fn(request) -> iterator of token strings (one item per generated token)
StreamFn = Callable[[Request], Any]


def benchmark_streaming(
    stream: StreamFn,
    request: Request,
    *,
    label: str,
    n_requests: int = 100,
    concurrency: int = 8,
    warmup: int = 5,
) -> StreamResult:
    """Drive a streaming (token-by-token) backend and report the metrics that
    define LLM-serving quality: TTFT (time to first token — the latency a user
    feels) and aggregate tokens/sec (decode throughput — what sets $/token)."""
    for _ in range(max(0, warmup)):                    # warm up (discarded)
        try:
            for _tok in stream(request):
                pass
        except Exception:
            pass

    ttfts: List[float] = []
    totals: List[float] = []
    token_counts: List[int] = []
    errors = 0
    lock = threading.Lock()

    def one(_i: int) -> None:
        nonlocal errors
        start = time.perf_counter()
        first_at: Optional[float] = None
        n_tok = 0
        try:
            for _tok in stream(request):
                if first_at is None:
                    first_at = time.perf_counter()
                n_tok += 1
            end = time.perf_counter()
            if first_at is None:                       # produced nothing
                with lock:
                    errors += 1
                return
            with lock:
                ttfts.append((first_at - start) * 1000)
                totals.append((end - start) * 1000)
                token_counts.append(n_tok)
        except Exception:
            with lock:
                errors += 1

    with GpuSampler() as gpu:
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for f in as_completed([pool.submit(one, i) for i in range(n_requests)]):
                f.result()
        wall = time.perf_counter() - wall_start

    ttfts.sort()
    totals.sort()
    total_tokens = sum(token_counts)
    return StreamResult(
        label=label,
        n_requests=n_requests,
        concurrency=concurrency,
        errors=errors,
        wall_time_s=round(wall, 4),
        tokens_per_sec=round(total_tokens / wall, 1) if wall > 0 else 0.0,
        ttft_ms={"p50": round(_percentile(ttfts, 50), 1),
                 "p95": round(_percentile(ttfts, 95), 1),
                 "p99": round(_percentile(ttfts, 99), 1)},
        total_ms={"p50": round(_percentile(totals, 50), 1),
                  "p95": round(_percentile(totals, 95), 1),
                  "p99": round(_percentile(totals, 99), 1)},
        gpu=gpu.summary(),
    )


def compare_streaming(results: List["StreamResult"]) -> str:
    lines = [STREAM_TABLE_HEADER] + [r.to_row() for r in results]
    if len(results) == 2:
        before, after = results
        if before.ttft_ms["p50"] and after.tokens_per_sec and before.tokens_per_sec:
            ttft = before.ttft_ms["p50"] / max(after.ttft_ms["p50"], 1e-9)
            tput = after.tokens_per_sec / max(before.tokens_per_sec, 1e-9)
            lines.append("")
            lines.append(f"**{after.label} vs {before.label}:** "
                         f"~{ttft:.1f}× faster TTFT, ~{tput:.1f}× higher token throughput.")
    return "\n".join(lines)


def compare(results: List[BenchmarkResult]) -> str:
    """Render a before/after markdown table plus the headline deltas."""
    lines = [TABLE_HEADER] + [r.to_row() for r in results]
    if len(results) == 2:
        before, after = results
        if before.latency_ms["p50"] and after.throughput_rps and before.throughput_rps:
            lat = before.latency_ms["p50"] / max(after.latency_ms["p50"], 1e-9)
            tput = after.throughput_rps / max(before.throughput_rps, 1e-9)
            lines.append("")
            lines.append(f"**{after.label} vs {before.label}:** "
                         f"~{lat:.1f}× lower p50 latency, ~{tput:.1f}× higher throughput.")
    return "\n".join(lines)
