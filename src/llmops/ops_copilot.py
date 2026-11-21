"""
Taxi Operations Copilot — a grounded RAG assistant over the platform's own data.

This is what makes the LLM workload *coherent* with the rest of the repo instead
of "GenAI bolted on": the copilot answers operational questions ("why did demand
spike yesterday?", "what's tomorrow's capacity target?") strictly from the
forecasting workload's own signals — the demand history the forecaster consumes.

    question → guardrails → retrieve ops facts (vector store) → grounded prompt
             → LLM → guardrails → answer, with the facts it used as sources

It reuses the whole LLMOps stack (RAGService, guardrails, vector store) and grounds
retrieval in *real* data (data/daily_demand.csv), so the answers cite actual
numbers. The LLM itself is injected — a stub in tests, the platform's vLLM backend
in production — so the copilot runs and is tested without a GPU.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional

from src.llmops.rag import RAGService, RAGResponse
from src.llmops.vector_store import Document, InMemoryVectorStore
from src.llmops.embeddings import Embedder
from src.llmops.prompt_registry import PromptRegistry

LLMFn = Callable[[str], str]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

_COPILOT_PROMPT = (
    "You are the NYC Taxi Operations Copilot. Answer the operator's question using "
    "ONLY the demand facts in the context. Cite the specific numbers and dates. If "
    "the context doesn't contain the answer, say you don't have that data.\n\n"
    "Context:\n{context}\n\nOperator question: {question}\nAnswer:"
)


def build_ops_documents(daily_csv: Optional[str] = None, window: int = 30,
                        spike_pct: float = 15.0) -> List[Document]:
    """Turn recent demand history into grounded, retrievable ops facts.

    Produces one document per recent day (with day-over-day change and spike/drop
    flags) plus summary + peak facts — so a question about a specific day's spike
    retrieves the real number behind it."""
    import pandas as pd

    path = daily_csv or os.path.join(_DATA_DIR, "daily_demand.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, parse_dates=["TimePeriod"]).sort_values("TimePeriod")
    if df.empty:
        return []
    df["pct"] = df["Volume"].pct_change() * 100.0
    recent = df.tail(window).reset_index(drop=True)

    docs: List[Document] = [
        Document("summary-daily",
                 f"Over the last {len(recent)} days, daily NYC taxi demand averaged "
                 f"{recent.Volume.mean():.0f} trips, ranging {recent.Volume.min():.0f} "
                 f"to {recent.Volume.max():.0f}."),
    ]
    peak = recent.loc[recent.Volume.idxmax()]
    docs.append(Document(
        "peak-daily",
        f"The highest-demand day in this window was {peak.TimePeriod.date()} "
        f"with {peak.Volume:.0f} trips."))

    for _, row in recent.iterrows():
        date = row.TimePeriod.date()
        sentence = f"On {date}, demand was {row.Volume:.0f} trips"
        if pd.notna(row.pct):
            direction = "up" if row.pct >= 0 else "down"
            sentence += f", {direction} {abs(row.pct):.1f}% from the prior day"
            if row.pct >= spike_pct:
                sentence += " — a notable spike"
            elif row.pct <= -spike_pct:
                sentence += " — a notable drop"
        docs.append(Document(f"day-{date}", sentence + "."))
    return docs


class TaxiOpsCopilot:
    def __init__(
        self,
        llm: LLMFn,
        *,
        daily_csv: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        top_k: int = 4,
    ) -> None:
        self.store = InMemoryVectorStore(embedder)
        docs = build_ops_documents(daily_csv)
        self.n_facts = self.store.add(docs)

        prompts = PromptRegistry()
        prompts.register("ops_copilot", _COPILOT_PROMPT, "taxi ops copilot")
        prompts.set_alias("ops_copilot", 1, "champion")

        self.rag = RAGService(self.store, llm, prompts=prompts,
                              prompt_name="ops_copilot", top_k=top_k)

    def ask(self, question: str) -> RAGResponse:
        return self.rag.answer(question)


if __name__ == "__main__":
    # Stub LLM so the flow runs without a GPU; in production this is the platform's
    # vLLM backend. It just echoes the grounded context it was given.
    def _stub_llm(prompt: str) -> str:
        ctx = prompt.split("Context:", 1)[-1].split("Operator question:", 1)[0].strip()
        first = ctx.splitlines()[0] if ctx else "(no data)"
        return f"Based on the demand data: {first}"

    copilot = TaxiOpsCopilot(_stub_llm)
    print(f"Indexed {copilot.n_facts} ops facts from demand history.\n")
    for q in ("What was the highest-demand day recently?",
              "What is average daily demand?"):
        r = copilot.ask(q)
        print(f"Q: {q}\nA: {r.answer}\n   sources: {r.sources}\n")
