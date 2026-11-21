"""
Embeddings — pluggable text→vector, with a dependency-free default.

The platform must embed text (for RAG retrieval) without hard-wiring a heavyweight
model into every container. `HashingEmbedder` is a deterministic, dependency-free
embedder (the hashing-trick / feature-hashing): good enough to make retrieval
*work and be tested* anywhere, with the exact same interface a real embedder
exposes. Swap in sentence-transformers or an embedding API in production by
implementing the same `embed(texts) -> np.ndarray` call.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    dim: int
    def embed(self, texts: List[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Feature-hashing embedder. Deterministic, no model download, cosine-mean-
    ingful: documents sharing tokens get similar vectors. Signed hashing reduces
    collision bias. Vectors are L2-normalized so dot product == cosine similarity."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)  # nosec B324 (non-crypto use)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            v[idx] += sign
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim))


_DEFAULT = HashingEmbedder()


def embed(texts: List[str], embedder: Embedder = _DEFAULT) -> np.ndarray:
    return embedder.embed(texts)
