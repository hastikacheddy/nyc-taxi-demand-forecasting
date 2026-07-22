"""
LLMOps layer — the GenAI half of the AI platform.

AI infrastructure in 2026 almost always includes generative AI, so the platform
carries the pieces a GenAI workload needs, built to the same standards as the rest
of the repo (framework-neutral, dep-free-runnable, tested):

    Prompt Registry   versioned prompt templates with aliases (champion/canary),
                      the LLM analog of the model registry.
    Guardrails        input + output safety: PII, prompt-injection heuristics,
                      output length/format/blocklist checks.
    Embeddings        a pluggable embedding function (dep-free hashing embedder
                      for tests; sentence-transformers/OpenAI in production).
    Vector Store      cosine-similarity retrieval over embedded documents — the
                      "vector DB" abstraction, swappable for pgvector/Qdrant/etc.
    RAG Service       retrieve → assemble grounded prompt → (guardrails) → LLM.
    Evaluation        run a suite of cases, score, and report — an LLM's CI gate.

The LLM itself is served by vLLM on the GPU pool via the platform's VLLMBackend
(src/platform/backends.py) — LLMOps is the layer *around* the model, not another
serving stack.
"""
from src.llmops.prompt_registry import PromptRegistry, PromptTemplate
from src.llmops.guardrails import GuardrailPipeline, GuardrailResult
from src.llmops.embeddings import HashingEmbedder, embed
from src.llmops.vector_store import InMemoryVectorStore, Document
from src.llmops.rag import RAGService
from src.llmops.evaluation import EvaluationSuite, EvalCase
from src.llmops.ops_copilot import TaxiOpsCopilot

__all__ = [
    "PromptRegistry", "PromptTemplate",
    "GuardrailPipeline", "GuardrailResult",
    "HashingEmbedder", "embed",
    "InMemoryVectorStore", "Document",
    "RAGService",
    "EvaluationSuite", "EvalCase",
    "TaxiOpsCopilot",
]
