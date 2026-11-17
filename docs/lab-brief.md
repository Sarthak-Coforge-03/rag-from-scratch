# Lab: Build a RAG System from Scratch — Brief & Rubric

Verbatim copy of the assignment. `CLAUDE.md` holds a 30-line summary; this file is the source of truth.

## Overview
This week you'll build a working RAG pipeline from the ground up, applying every concept covered this week: chunking, embeddings, vector search, hybrid/agentic retrieval, failure modes, reranking, an evaluation harness, and data quality/lineage. You'll also debug a deliberately broken version of your own pipeline where the root cause is bad source data, not retrieval logic. No provided corpus, no starter files.

If you're unsure how a tool or concept works beyond what the lesson covered, ask Claude (in a regular conversation, separate from your script) to explain it, or to help you research an unfamiliar library, technique, or error message before you implement it.

## Suggested Approach (Read This Before Starting)
1. **Install a real embedding library and vector store.** `sentence-transformers` for local, free embeddings (no API key) paired with `chromadb` as a local vector store (no external service/account). A hosted embedding API (OpenAI/Anthropic) is also fine, but the local option avoids cost and setup.
2. **Get one chunk embedded and stored before building the full pipeline.** Smallest possible script: one short text → embedding via sentence-transformers → stored in a chromadb collection. Confirm end to end before writing chunking logic.
3. **Confirm retrieval works on that same minimal example.** Add a second, different text, run a query, confirm the store returns the more relevant of the two. Proves the embed-store-retrieve loop works before layering chunking, hybrid search, or reranking.
4. **Only then** build the chunking function, run it on the real generated documents, and repeat steps 2-3 at scale.
5. **Add hybrid retrieval, reranking, and the eval harness incrementally**, one at a time, confirming each works on its own before adding the next.
6. **If stuck**, go back to step 3 — a working retrieval loop on two known texts — before adding complexity.

## The Scenario
An internal Q&A assistant over a set of company policy documents. You generate your own source material rather than being handed a corpus.

## What You'll Do

### Part 1: Generate Source Documents
Use an LLM to generate 3-4 policy documents (e.g., remote work, expense reimbursement, PTO), each 500-800 words. Deliberately introduce one realistic data quality issue: either an outdated duplicate of one document (old version with conflicting details) or an incomplete document missing a section it references.

### Part 2: Build the Pipeline
Following the Suggested Approach: implement chunking (with documented chunk size and overlap rationale), embed each chunk with sentence-transformers, store embeddings in chromadb. Write a script that takes a natural-language question, embeds it, retrieves the top-k most similar chunks, and passes them to an LLM (via its API) to generate an answer.

### Part 3: Add Hybrid Retrieval
Add keyword-based search alongside vector search (simple substring/grep-style match against chunk text is sufficient; no dedicated keyword library needed), and combine/re-rank results from both. Demonstrate at least one query where hybrid succeeds where vector-only would likely miss (e.g., a query containing an exact policy section name or code).

### Part 4: Add Reranking
Add a reranking step after initial retrieval. Practical option: a sentence-transformers `CrossEncoder` model to re-score the initial candidate set before selecting the final top few chunks.

### Part 5: Build an Evaluation Harness
A fixed test set of at least 8 questions with known expected answers or expected source chunks. Automated tests (pytest) that measure retrieval recall (did the right chunk get retrieved) and check that generated answers contain expected key information. Run the harness and record results.

### Part 6: Diagnose the Planted Data Quality Issue
Ask at least one question that should surface the planted issue. Using the two-question debugging approach (was it retrieval, or was it the source data?), diagnose and document what went wrong, and confirm it traces back to the source data, not retrieval or generation logic.

### Part 7: Source Attribution
Add source metadata (document name, section) to each chunk, and have the system surface which source(s) informed each generated answer.

### Part 8: Pipeline Submission
Push everything through the CI pipeline, with the evaluation harness tests running as part of it.

## Deliverable
A single PDF containing screenshots, in order, of:
1. Generated source documents, with the planted data quality issue identified
2. Terminal output of the minimal embed-store-retrieve loop working on two known texts (Suggested Approach steps 2-3)
3. Chunking, embedding, and vector store code (full pipeline)
4. Basic RAG pipeline running end to end on a sample question
5. Hybrid retrieval code, and a query demonstrating it outperforming vector-only search
6. Reranking code
7. Evaluation test set and harness code
8. Terminal output of the evaluation harness running, with recall/accuracy results
9. The question that surfaced the planted issue, the system's (flawed) answer, and the written diagnosis tracing it to the source data
10. Source attribution output, showing an answer with its cited source(s)
11. A passing pipeline run confirming the full submission

## Rubric (100 points)
| Criteria | Points |
|---|---|
| Source documents generated with a realistic, identifiable data quality issue planted | 8 |
| Minimal embed-store-retrieve loop confirmed working before full build (Suggested Approach) | 7 |
| Chunking, embedding, and vector store implemented correctly at full scale | 10 |
| Basic RAG pipeline runs end to end and produces reasonable answers | 10 |
| Hybrid retrieval implemented and demonstrated to outperform vector-only search on at least one query | 12 |
| Reranking implemented correctly | 8 |
| Evaluation harness includes 8+ test cases and measures both retrieval recall and answer accuracy | 15 |
| Data quality issue correctly diagnosed, with clear reasoning tracing it to source data rather than retrieval/generation | 15 |
| Source attribution correctly implemented and surfaced with answers | 5 |
| Submitted and passing through the pipeline | 10 |
| **Total** | **100** |
