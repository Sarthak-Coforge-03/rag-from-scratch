# Planted Data-Quality Issue: Outdated Duplicate Policy

**Type:** outdated duplicate document with conflicting details (Lab Part 1).
**Diagnosis:** to be written in Part 6. This file records only what was planted and what we expect to happen.

## What was planted

The corpus in `data/policies/` contains **two versions of the same policy**:

| File | Policy ID | Version | Effective | Status |
|---|---|---|---|---|
| `expense_reimbursement_policy.md` | EXP-001 | 3.0 | 2026-01-01 | **Current** (its header says "Supersedes v2.1") |
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

The point of Part 6 is that the retrieved chunks *are* relevant and the generator faithfully reports what it was given. The root cause is that the source corpus contains a superseded document. The evidence is the `Version` / `Effective date` metadata and the "Supersedes v2.1" line in the current file.
