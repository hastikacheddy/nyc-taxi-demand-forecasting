"""
RAG service — retrieval-augmented generation, wired through the platform.

Flow:  query
         → input guardrails (PII redaction, injection block)
         → retrieve top-k documents from the vector store
         → assemble a grounded prompt from a registered template
         → LLM generation (any callable; the platform's VLLMBackend in prod)
         → output guardrails (PII echo, blocklist)

Every dependency is injected, so the whole pipeline runs and is tested with the
dep-free embedder + a stub LLM, and swaps to a real vector DB + vLLM by passing
different objects — no code change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.llmops.guardrails import GuardrailPipeline
from src.llmops.prompt_registry import PromptRegistry
from src.llmops.vector_store import InMemoryVectorStore

# An LLM is any callable: prompt -> completion. In production this wraps the
# platform's VLLMBackend; in tests it's a stub.
LLMFn = Callable[[str], str]

_DEFAULT_TEMPLATE = (
    "You are a helpful analyst. Answer the question using ONLY the context below.\n"
    "If the context is insufficient, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


@dataclass
class RAGResponse:
    answer: str
    allowed: bool
    sources: List[str]
    input_guardrail: dict
    output_guardrail: dict
    retrieved: List[Tuple[str, float]]   # (doc_id, score)


class RAGService:
    def __init__(
        self,
        store: InMemoryVectorStore,
        llm: LLMFn,
        *,
        prompts: Optional[PromptRegistry] = None,
        prompt_name: str = "rag_default",
        guardrails: Optional[GuardrailPipeline] = None,
        top_k: int = 3,
    ) -> None:
        self.store = store
        self.llm = llm
        self.guardrails = guardrails or GuardrailPipeline()
        self.top_k = top_k
        self.prompts = prompts or PromptRegistry()
        self.prompt_name = prompt_name
        if self.prompts.get(prompt_name) is None:
            self.prompts.register(prompt_name, _DEFAULT_TEMPLATE, "default RAG template")
            self.prompts.set_alias(prompt_name, 1, "champion")

    def answer(self, question: str) -> RAGResponse:
        # 1. input guardrails (may redact PII, may block injection)
        gin = self.guardrails.check_input(question)
        if not gin.allowed:
            return RAGResponse(
                answer="Request blocked by input guardrails.",
                allowed=False, sources=[], input_guardrail=gin.as_dict(),
                output_guardrail={}, retrieved=[])
        safe_q = gin.redacted_text or question

        # 2. retrieve
        hits = self.store.search(safe_q, k=self.top_k)
        context = "\n".join(f"- {doc.text}" for doc, _ in hits) or "(no documents found)"

        # 3. assemble grounded prompt from the registered (champion) template
        prompt = self.prompts.render(self.prompt_name, "champion",
                                     context=context, question=safe_q)

        # 4. generate
        completion = self.llm(prompt)

        # 5. output guardrails
        gout = self.guardrails.check_output(completion)
        answer = completion if gout.allowed else "Response withheld by output guardrails."
        return RAGResponse(
            answer=answer,
            allowed=gout.allowed,
            sources=[doc.id for doc, _ in hits],
            input_guardrail=gin.as_dict(),
            output_guardrail=gout.as_dict(),
            retrieved=[(doc.id, score) for doc, score in hits],
        )
