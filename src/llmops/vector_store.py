"""
Vector store — the "vector DB" abstraction.

An in-memory cosine-similarity store: enough to build and test RAG end to end, and
a clean seam to swap for pgvector / Qdrant / Pinecone in production (same
add/search interface). Vectors from HashingEmbedder are L2-normalized, so cosine
similarity is a single matrix-vector dot product — the same operation a real ANN
index approximates at scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.llmops.embeddings import Embedder, HashingEmbedder


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


class InMemoryVectorStore:
    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._docs: List[Document] = []
        self._matrix: Optional[np.ndarray] = None   # (n_docs, dim)

    def add(self, docs: List[Document]) -> int:
        if not docs:
            return 0
        vecs = self.embedder.embed([d.text for d in docs])
        self._docs.extend(docs)
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])
        return len(docs)

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Top-k documents by cosine similarity (dot product on normalized vecs)."""
        if not self._docs or self._matrix is None:
            return []
        q = self.embedder.embed([query])[0]
        scores = self._matrix @ q
        k = min(k, len(self._docs))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._docs[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self._docs)
