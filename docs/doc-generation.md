# How the Source Documents Were Generated

The lab's Part 1 asks for LLM-generated policy documents. All five files in `data/policies/` were written by **Claude (Opus 5.5)** in Claude Code on 2026-11-17, for the fictional company *Halvorsen Analytics*. No real company policies were used.

## Prompt / spec given to the model

> Write four internal policy documents for a fictional mid-size company, "Halvorsen Analytics": a Remote Work policy, an Expense Reimbursement policy, a PTO policy, and an IT Security policy. Each should be 500–800 words of Markdown.
>
> Format: an `#` title, a metadata list (Policy ID, Version, Effective date, Owner), and numbered sections as `## <PREFIX>-<n> <Name>` (prefixes RW, EXP, PTO, SEC), with `###` subsections where useful. Include concrete numbers (limits, deadlines, thresholds) and a few exact form or section codes (for example `FIN-REQ-17`, `HR-RW-04`, `LV-PTO`, `SEC-7.3 Incident Escalation`) so the documents can be queried precisely.
>
> Then write a fifth file: an **older version (v2.1, 2023)** of the Expense policy that is nearly identical but has different values for the meal limit, receipt threshold, submission deadline, director-approval threshold and some travel rules. It must not say that it has been superseded.

## Design choices for later lab parts

| Choice | Supports |
|---|---|
| `## CODE Name` headings | Chunking by section and recording section metadata (Part 7, source attribution) |
| Rare exact tokens (`FIN-REQ-17`, `HR-RW-04`, `LV-PTO`, `SEC-7.3`) | Queries where keyword search beats vector search (Part 3, hybrid) |
| Concrete numbers in every doc | Checking answers for key facts (Part 5, evaluation) |
| Stale duplicate with identical section codes | Planted data-quality issue (Parts 1 and 6), see `data-quality-issue.md` |

## Word counts

| File | Words |
|---|---|
| expense_reimbursement_policy.md | 740 |
| expense_reimbursement_policy_2023.md (planted) | 673 |
| it_security_policy.md | 648 |
| pto_policy.md | 681 |
| remote_work_policy.md | 663 |
