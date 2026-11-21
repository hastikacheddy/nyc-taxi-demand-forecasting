"""
Benchmark backends — the things being measured.

Each backend is just a callable `infer(request) -> response`. The harness doesn't
care what's behind it, which is the whole point: the *same* measurement drives a
vLLM server, a vanilla HF server, or a local mock.

  * MockBackend            a deterministic sleeper with configurable latency +
                           concurrency ceiling. Models the qualitative difference
                           between vanilla (per-request, low concurrency) and
                           batched (higher effective concurrency) serving, so the
                           harness is demoable and testable with zero deps.
  * OpenAICompatBackend    hits any OpenAI-compatible /chat/completions — this is
                           the REAL path for both vLLM and HF TGI. Same code,
                           different URL, so the comparison is apples-to-apples.
  * HFPipelineBackend      an in-process HuggingFace pipeline (optional import).
                           The literal "vanilla transformers" baseline when torch
                           + transformers are installed.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


class MockBackend:
    """Deterministic stand-in. `base_latency_ms` is the per-request service time;
    `max_concurrency` caps how many requests are truly served in parallel (extra
    requests queue) — that ceiling is precisely what separates naïve per-request
    serving from continuous batching, so the mock reproduces the *shape* of the
    real result without a GPU."""

    def __init__(self, base_latency_ms: float = 40.0, max_concurrency: int = 8) -> None:
        self.base_latency_ms = base_latency_ms
        self._sema = threading.Semaphore(max_concurrency)

    def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with self._sema:
            # scale a little with requested tokens to feel realistic
            tokens = int(request.get("max_tokens", 64))
            time.sleep((self.base_latency_ms + 0.1 * tokens) / 1000.0)
            return {"completion": "ok", "tokens": tokens}


class MockStreamBackend:
    """Deterministic streaming stand-in for a token-by-token LLM server.

    `ttft_ms` is the prefill time before the first token; `per_token_ms` the
    decode interval; `max_concurrency` the batching ceiling. A vanilla server is
    modelled as high TTFT + low concurrency; a batched (vLLM-like) server as low
    TTFT + high concurrency — so TTFT and tokens/sec show the real shape without a
    GPU."""

    def __init__(self, ttft_ms: float = 200.0, per_token_ms: float = 20.0,
                 tokens: int = 64, max_concurrency: int = 8) -> None:
        self.ttft_ms = ttft_ms
        self.per_token_ms = per_token_ms
        self.tokens = tokens
        self._sema = threading.Semaphore(max_concurrency)

    def stream(self, request: Dict[str, Any]):
        n = int(request.get("max_tokens", self.tokens))
        with self._sema:
            time.sleep(self.ttft_ms / 1000.0)          # prefill → first token
            yield "tok"
            for _ in range(max(0, n - 1)):
                time.sleep(self.per_token_ms / 1000.0)
                yield "tok"


class OpenAICompatBackend:
    """Real backend: POST /chat/completions to a vLLM or TGI OpenAI-compatible
    server. Set base_url to the vLLM service for the 'after' run and to the
    vanilla/TGI server for the 'before' run."""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0,
                 api_key: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        import httpx
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model,
            "messages": request.get("messages")
            or [{"role": "user", "content": request.get("prompt", "Hello")}],
            "max_tokens": int(request.get("max_tokens", 64)),
            "temperature": float(request.get("temperature", 0.0)),
        }
        r = httpx.post(f"{self.base_url}/chat/completions", json=body,
                       headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return {"completion": data["choices"][0]["message"]["content"],
                "usage": data.get("usage", {})}

    def stream(self, request: Dict[str, Any]):
        """Yield content tokens as they arrive (OpenAI SSE, stream=True) — the
        real path for measuring TTFT + tokens/sec against vLLM/TGI."""
        import json as _json
        import httpx
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model,
            "messages": request.get("messages")
            or [{"role": "user", "content": request.get("prompt", "Hello")}],
            "max_tokens": int(request.get("max_tokens", 64)),
            "temperature": float(request.get("temperature", 0.0)),
            "stream": True,
        }
        with httpx.stream("POST", f"{self.base_url}/chat/completions", json=body,
                          headers=headers, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = _json.loads(payload)["choices"][0]["delta"].get("content")
                except (KeyError, ValueError, IndexError):
                    continue
                if delta:
                    yield delta


class HFPipelineBackend:
    """In-process HuggingFace text-generation pipeline — the literal 'vanilla
    transformers' baseline. Optional: requires torch + transformers. Kept out of
    requirements so the repo stays light; installed only when you actually run a
    real GPU benchmark."""

    def __init__(self, model: str = "sshleifer/tiny-gpt2", device: int = -1) -> None:
        from transformers import pipeline  # noqa: F401 (optional dep)
        self.pipe = pipeline("text-generation", model=model, device=device)

    def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        prompt = request.get("prompt", "Hello")
        out = self.pipe(prompt, max_new_tokens=int(request.get("max_tokens", 64)),
                        do_sample=False)
        return {"completion": out[0]["generated_text"]}


def build_backend(kind: str, **kwargs):
    kind = kind.lower()
    if kind == "mock":
        return MockBackend(**{k: v for k, v in kwargs.items()
                              if k in ("base_latency_ms", "max_concurrency")})
    if kind in ("mock-stream", "mockstream"):
        return MockStreamBackend(**{k: v for k, v in kwargs.items()
                                    if k in ("ttft_ms", "per_token_ms", "tokens", "max_concurrency")})
    if kind in ("vllm", "openai", "tgi"):
        return OpenAICompatBackend(kwargs["base_url"], kwargs.get("model", "default"),
                                   timeout=kwargs.get("timeout", 60.0),
                                   api_key=kwargs.get("api_key"))
    if kind in ("hf", "transformers"):
        return HFPipelineBackend(kwargs.get("model", "sshleifer/tiny-gpt2"),
                                 device=kwargs.get("device", -1))
    raise ValueError(f"unknown backend kind: {kind}")
