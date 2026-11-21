# `src/llmops/` — the GenAI layer

The LLM platform pieces from the brief, built to the repo's standard
(framework-neutral, dependency-free-runnable, tested). The LLM *model* is served
by vLLM on the GPU pool via the platform's `VLLMBackend`; this package is
everything *around* the model.

```
        Prompt Registry ──┐
                          │
 query → Guardrails(in) → RAG ──→ retrieve (Vector Store ← Embeddings)
                          │          │
                          │     grounded prompt (registered template)
                          │          │
                          └────→   LLM  ──→ Guardrails(out) → answer
                                    │
                              Evaluation suite  (CI gate)
```

| Component | File | Prod swap |
|---|---|---|
| Prompt Registry (versioned + aliases) | `prompt_registry.py` | back with a DB; same interface |
| Guardrails (PII / injection / output) | `guardrails.py` | add a policy model; reuses `src/data/pii.py` |
| Embeddings (hashing, dep-free) | `embeddings.py` | sentence-transformers / embedding API |
| Vector Store (cosine, in-memory) | `vector_store.py` | pgvector / Qdrant / Pinecone |
| RAG Service | `rag.py` | inject the real store + vLLM LLM |
| **Taxi Ops Copilot** (grounded in own data) | `ops_copilot.py` | swap stub LLM → platform vLLM backend |
| Evaluation (deterministic scorers) | `evaluation.py` | add an LLM-judge `Scorer` |

The **Ops Copilot** (`ops_copilot.py`) is what makes the LLM *coherent* with the
rest of the repo: it indexes real facts from the forecasting workload's demand
history (`data/daily_demand.csv`) and answers operator questions ("why did demand
spike on X?", "what's the average daily demand?") strictly from those grounded
facts — so it's an assistant over the platform's own data, not GenAI bolted on.
Run it: `python -m src.llmops.ops_copilot`.

## Run it

```python
from src.llmops import InMemoryVectorStore, Document, RAGService

store = InMemoryVectorStore()
store.add([Document("d1", "NYC taxi demand peaks during morning rush hour")])

def llm(prompt: str) -> str:          # in prod: platform VLLMBackend.predict
    return "Demand peaks at morning rush hour."

rag = RAGService(store, llm)
print(rag.answer("When does taxi demand peak?").answer)
```

```bash
pytest tests/test_llmops.py           # 12 tests, dep-free
```

## Why these pieces (the design stance)

- **Prompts are versioned artifacts**, not string literals — a wording change can
  regress quality like a model swap, so it gets versions + a champion alias,
  mirroring the model registry.
- **Guardrails treat input and output as untrusted** (OWASP LLM Top 10,
  [SECURITY.md §2.2](../../docs/platform/SECURITY.md)): PII is redacted before it
  reaches the model, injection is blocked *before* the LLM is even called (tested),
  and output is checked before it reaches the user.
- **The vector store is an interface, not a product** — RAG is built against
  `add`/`search`, so pgvector vs Qdrant is a deployment decision, not a rewrite.
- **Evaluation is a gate, not a vibe** — a deterministic suite with a pass-rate
  bar, the LLM analog of the repo's champion-challenger promotion gate. You can
  fail CI on a prompt regression.
