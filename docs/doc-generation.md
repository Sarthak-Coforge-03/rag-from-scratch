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

## Expansion: production-style corpus (2026-11-18)

To make retrieval behave more like a real company wiki, eight more documents were added by **Claude (Opus 5.5)**, and the original five were converted from bullet metadata to optional YAML front matter (bodies unchanged).

> Write eight more markdown documents for Halvorsen Analytics in deliberately *varied* shapes, as found in a real policy wiki: a benefits overview with a table, an employee FAQ in Q&A form, an onboarding checklist, a numbered travel-booking how-to, a mostly-prose code of conduct, a memo-style holiday announcement with no headings, a glossary, and a README index. Some should have YAML front matter (title, policy_id, version, effective_date, owner), some none. Every fact that overlaps an existing policy must match the **current** versions (e.g. $75 daily meal limit, receipts over $50, 30-day claim deadline) so the stale 2023 expense policy remains the only conflicting source.

| File | Shape | Front matter | Words |
|---|---|---|---|
| README.md | index of links | none | 203 |
| benefits_overview.md | table + short sections | full | 566 |
| code_of_conduct.md | prose, one heading | full | 500 |
| employee_faq.md | bold Q / A pairs | none | 490 |
| glossary.md | bold term — definition | none | 371 |
| holiday_schedule_2026_announcement.md | email memo, no headings | none | 339 |
| onboarding_guide.md | checklist (`- [ ]`) | title + owner only | 458 |
| travel_booking_howto.md | numbered steps, `###` headings | none | 451 |

With these, the corpus is 13 documents and about 6,800 words, which chunk into 66 chunks at 800/150 characters.
