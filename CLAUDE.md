# CLAUDE.md

Lab: build a RAG Q&A assistant over company policy docs, from scratch. Full brief + 100-pt rubric: `docs/lab-brief.md` (source of truth — check it before claiming a part is done). Personal dev preferences: `CLAUDE.local.md` (gitignored).
Corpus: `data/policies/` (5 docs; `expense_reimbursement_policy_2023.md` is the PLANTED stale duplicate — see `docs/data-quality-issue.md`). Never "fix" it.

## Stack
- Python 3.12, `uv` (src layout: `src/rag_from_scratch/`, CLI `rag-from-scratch` → `rag_from_scratch:main`).
- Embeddings: `sentence-transformers` (local). Vector store: `chromadb` (local). Reranker: sentence-transformers `CrossEncoder`.
- Generation: an LLM via API. Tests: `pytest`. CI must run the eval harness.
- Hooks (`.claude/`): every Stop runs `ruff format` + `pytest` (failures block); `.env` files are off-limits.

## Build order (incremental — verify each step before the next)
1. Generate 3-4 policy docs (500-800 words) with ONE planted data-quality issue (outdated conflicting duplicate, or doc missing a referenced section).
2. Minimal loop: embed + store one text, add a second, query returns the more relevant one.
3. Chunking (document size/overlap rationale) → embed → chroma; query → top-k → LLM answer.
4. Hybrid: add substring/keyword search, merge with vector results; show a query where hybrid beats vector-only.
5. Rerank candidates with a CrossEncoder before picking final chunks.
6. Eval harness: ≥8 fixed questions; pytest checks retrieval recall AND answer key-info.
7. Diagnose planted issue: retrieval vs. source data — show it's the source data.
8. Source attribution: chunk metadata (doc name, section) surfaced with every answer.

## Commands
```sh
uv sync                  # install
uv run rag-from-scratch  # run CLI
uv add <pkg> / uv add --dev <pkg>
uv run pytest            # all tests; single: uv run pytest path::test_name
```

Deliverable is a screenshot PDF (see brief) — keep terminal output readable and screenshot-friendly.
