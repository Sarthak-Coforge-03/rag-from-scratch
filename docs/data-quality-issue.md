# Planted Data-Quality Issue: Outdated Duplicate Policy

**Type:** outdated duplicate document with conflicting details (Lab Part 1).
**Diagnosis:** see [Diagnosis (Part 6)](#diagnosis-part-6) at the end. In short, the root cause is the **source data**, not retrieval or generation.

## What was planted

The corpus in `data/policies/` contains **two versions of the same policy**:

| File | Policy ID | Version | Effective | Status |
|---|---|---|---|---|
| `expense_reimbursement_policy.md` | EXP-001 | 3.0 | 2026-01-01 | **Current** (its front matter says `supersedes: EXP-001 v2.1 (2023)`) |
| `expense_reimbursement_policy_2023.md` | EXP-001 | 2.1 | 2023-03-01 | **Stale**: never removed, no "superseded" marker |

Both files have the same title, the same Policy ID, the same section codes (`EXP-1` … `EXP-9`) and mostly identical wording. Only the facts below differ.

## Conflicting facts

| Section | Fact | v2.1 (2023, stale) | v3.0 (2026, current) |
|---|---|---|---|
| EXP-4.2 | Daily meal limit | $50 | **$75** |
| EXP-5 | Receipt required above | $25 | **$50** |
| EXP-6 | Claim submission deadline | 60 days | **30 days** |
| EXP-7 | Director approval above | $1,000 | **$2,500** |
| EXP-3 | Flights of 6 hours or more | economy only | **premium economy with approval** |
| EXP-3 | Airport parking | full, up to 5 days | **capped at round-trip ride-share fare** |
| EXP-4.2 | Same-day trip of more than 10 hours | no meal allowance | **meal allowance allowed** |

## Why it's realistic

Policy owners often publish a new version without deleting the old file from the shared drive or wiki. Both copies look authoritative. An ingestion job that crawls the whole folder picks up both, and nothing in the stale file says it has been replaced.

## Questions that should surface it

- "What is the daily meal limit when traveling?" (expected: $75)
- "How many days do I have to submit an expense claim?" (expected: 30 days)
- "Above what amount do I need an itemized receipt?" (expected: $50)

## Expected flawed behaviour

Retrieval should work correctly. For these questions, the top-k chunks will include the `EXP-4.2` / `EXP-5` / `EXP-6` section from **both** files, because they're near-identical in wording and embedding. The LLM then either:

1. answers with the **stale** value ($50 / 60 days / $25), or
2. gives a hedged answer ("between $50 and $75"), or
3. switches between the two values from one run to the next.

The point of Part 6 is that the retrieved chunks *are* relevant and the generator faithfully reports what it was given. The root cause is that the source corpus contains a superseded document. The evidence is the `version` / `effective_date` metadata on every retrieved chunk and the `supersedes` field in the current file's front matter.

## Diagnosis (Part 6)

**Setup:** fresh index from the policy folder (structural chunker, size 800), `all-MiniLM-L6-v2` embeddings, hybrid retrieval, Claude Opus 5.5 on Foundry. Same pipeline as `uv run rag-from-scratch`.

### The question that surfaced it

> **How many days do I have to submit an expense claim?**
> Correct answer (current policy EXP-001 v3.0, section EXP-6): **30 days**.

**Run 1, `TOP_K=1` — the system answers wrongly:**

```
Answer:
You have 60 days from the date the expense was incurred to submit a claim via form
FIN-REQ-17. For trips, submit one claim for the whole trip within 60 days of your return
date. Claims submitted after 60 days need written approval from a director and may be
rejected.

Sources (* = cited):
 * [1] expense_reimbursement_policy_2023    EXP-001 v2.1   EXP-6 Submitting a Claim
```

The receipt question fails the same way at `TOP_K=1`. It answers "any single expense over **$25**", but the current threshold is $50, and its only source is `expense_reimbursement_policy_2023 · EXP-001 v2.1 · EXP-5`.

**Run 2, `TOP_K=5` (the default) — right, but only by luck of the prompt:**

```
Answer:
You have 30 days from the date the expense was incurred to submit a claim on form
FIN-REQ-17. [...] (An older 2023 version of the policy, v2.1, allowed 60 days, but the
current v3.0 sets 30 days.)

Sources (* = cited):
   [1] expense_reimbursement_policy_2023    EXP-001 v2.1   EXP-6 Submitting a Claim
 * [2] expense_reimbursement_policy         EXP-001 v3.0   EXP-6 Submitting a Claim
 * [3] employee_faq                                        Employee FAQ
 * [4] travel_booking_howto                                After the trip
   [5] employee_faq                                        Employee FAQ
```

The stale copy is retrieved **first**, ahead of the current one. The answer is right only because both versions happened to be in the prompt, each labelled with its version, and the model chose the newer one. The same pattern held for all six conflicting facts in the table above:
- the stale chunk was retrieved every time;
- it ranked first for the deadline, receipt and airport-parking questions.

### Question 1: was it retrieval?

**No.** Retrieval did exactly its job:
- For a question about the claim deadline, it returned the chunks for section **EXP-6 Submitting a Claim**, the section that states the deadline.
- The two EXP-6 chunks are near-identical text, so they score almost the same, 0.032 each in run 2. Neither embedding similarity nor keyword matching can tell a current rule from a superseded one, because the words barely differ.
- Any retriever, vector, keyword or hybrid, with or without reranking, will keep returning both.

### Question 2: was it the source data?

**Yes.** The corpus contains two live, authoritative-looking documents for the same policy that state different rules:

```
expense_reimbursement_policy.md:39       Claims must be submitted **within 30 days** ...
expense_reimbursement_policy_2023.md:38  Claims must be submitted **within 60 days** ...
```

| | `expense_reimbursement_policy.md` | `expense_reimbursement_policy_2023.md` |
|---|---|---|
| policy_id | EXP-001 | EXP-001 |
| version | 3.0 | 2.1 |
| effective_date | 2026-01-01 | 2023-03-01 |
| marks the other as replaced? | `supersedes: "EXP-001 v2.1 (2023)"` | nothing: no "superseded" or "archived" marker |

Nothing in the data stops the 2023 file from being used. It sits in the same folder and has the same title, policy id and section codes. The only signal, the `supersedes` key in the *new* file's front matter, isn't something our loader reads.

### And generation?

**Not the cause.** In run 1, the model received one chunk that says 60 days, answered "60 days" and cited that chunk. That's a faithful answer to its sources, not a hallucination. In run 2, it received both versions and resolved them correctly. The LLM never invented a wrong value.

### Root cause

A **superseded document was left live in the source corpus**. The pipeline faithfully retrieves and quotes it. Whether a user gets the right answer depends on how many chunks are retrieved and how they are ranked, not on the system knowing which rule is current.

### Fixes (recommended, not applied — the stale file stays as planted)

1. **At the source (the real fix):** remove or archive `expense_reimbursement_policy_2023.md`, or at least mark it superseded in its own front matter. Old policy versions belong in an archive, not in the live folder.
2. **At ingestion (defence in depth):** when several documents share a `policy_id`, index only the one with the highest `version` / latest `effective_date`, or drop any document named in another document's `supersedes` field. Log what was skipped.
3. **At answer time (last resort):** tell the model to prefer the newest version when sources conflict, and to say so. Run 2 shows the model can do this, but it only works when both versions are retrieved, so it doesn't replace fixes 1 and 2.
