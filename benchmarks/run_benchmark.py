"""
Run an inference benchmark and emit the before/after table.

Examples
--------
# Zero-dependency self-demo: mock 'vanilla' (low concurrency) vs mock 'vLLM'
# (continuous batching → higher effective concurrency). Proves the harness end to
# end and shows the shape of the result without a GPU.
    python -m benchmarks.run_benchmark --demo

# Real comparison against two OpenAI-compatible servers:
    python -m benchmarks.run_benchmark \
        --before-url http://hf-tgi:8080/v1  --before-model my-model \
        --after-url  http://vllm:8000/v1    --after-model  my-model \
        --requests 500 --concurrency 32 --max-tokens 128 \
        --out benchmarks/results/latest.md

The numbers the harness reports (p50/p95/p99, throughput, GPU util) are the exact
axes that matter for production LLM serving; see docs/platform/COST_MODEL.md for how
throughput maps to $/request.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from benchmarks.backends import build_backend
from benchmarks.harness import benchmark, compare, benchmark_streaming, compare_streaming


def _run(args) -> None:
    request = {"prompt": args.prompt, "max_tokens": args.max_tokens}
    results = []

    if args.stream_demo:
        # LLM streaming metrics: vanilla (high TTFT, low concurrency) vs vLLM
        # (low TTFT via continuous batching, high concurrency).
        vanilla = build_backend("mock-stream", ttft_ms=900, per_token_ms=45,
                                tokens=args.max_tokens, max_concurrency=2)
        vllm = build_backend("mock-stream", ttft_ms=250, per_token_ms=8,
                             tokens=args.max_tokens, max_concurrency=16)
        sr = [
            benchmark_streaming(vanilla.stream, request, label="Vanilla HF (mock)",
                                n_requests=args.requests, concurrency=args.concurrency),
            benchmark_streaming(vllm.stream, request, label="vLLM (mock)",
                                n_requests=args.requests, concurrency=args.concurrency),
        ]
        table = compare_streaming(sr)
        print("\n" + table + "\n")
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                f.write("# LLM streaming benchmark (TTFT + tokens/sec)\n\n" + table + "\n")
            print(f"Wrote {args.out}")
        return

    if args.demo:
        # 'vanilla': serves ~2 concurrent (no batching); 'vLLM': ~16 (batched).
        before = build_backend("mock", base_latency_ms=180, max_concurrency=2)
        after = build_backend("mock", base_latency_ms=60, max_concurrency=16)
        results.append(benchmark(before, request, label="Vanilla HF (mock)",
                                 n_requests=args.requests, concurrency=args.concurrency))
        results.append(benchmark(after, request, label="vLLM (mock)",
                                 n_requests=args.requests, concurrency=args.concurrency))
    else:
        if args.before_url:
            b = build_backend("openai", base_url=args.before_url, model=args.before_model,
                              api_key=args.api_key)
            results.append(benchmark(b, request, label=f"Before ({args.before_model})",
                                     n_requests=args.requests, concurrency=args.concurrency))
        if args.after_url:
            a = build_backend("vllm", base_url=args.after_url, model=args.after_model,
                              api_key=args.api_key)
            results.append(benchmark(a, request, label=f"After/vLLM ({args.after_model})",
                                     n_requests=args.requests, concurrency=args.concurrency))
        if not results:
            raise SystemExit("Nothing to run: pass --demo, or --before-url/--after-url.")

    table = compare(results)
    print("\n" + table + "\n")
    for r in results:
        cs = f"{r.cold_start_ms:.0f} ms" if r.cold_start_ms is not None else "n/a"
        print(f"  {r.label}: cold_start={cs}  wall={r.wall_time_s:.2f}s  "
              f"errors={r.errors}  gpu={r.gpu.get('util_mean_pct')}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# Inference benchmark results\n\n" + table + "\n")
        with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump([r.as_dict() for r in results], f, indent=2)
        print(f"\nWrote {args.out} (+ .json)")


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description="Inference optimization benchmark")
    p.add_argument("--demo", action="store_true", help="zero-dependency mock comparison")
    p.add_argument("--stream-demo", action="store_true",
                   help="zero-dependency streaming demo (TTFT + tokens/sec)")
    p.add_argument("--before-url")
    p.add_argument("--before-model", default="baseline")
    p.add_argument("--after-url")
    p.add_argument("--after-model", default="vllm")
    p.add_argument("--api-key", default=None)
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prompt", default="Summarize NYC taxi demand for tomorrow.")
    p.add_argument("--out", default=None, help="write markdown+json here")
    _run(p.parse_args(argv))


if __name__ == "__main__":
    main()
