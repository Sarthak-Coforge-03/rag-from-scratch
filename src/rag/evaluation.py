"""Scoring rules for the eval harness: retrieval recall and answer key facts.

    cases = load_eval_cases(EVAL_SET_FILE)
    retrieval_hit(case, retriever.retrieve(case.question, top_k))   # recall
    missing_key_facts(case, answer.text) == []                      # accuracy

Used by tests/test_eval_harness.py (pytest, also in CI) and scripts/04_evaluate.py
(the printed results), so both always score the same way.
"""

from pathlib import Path

import yaml

from rag.models import EvalCase, RetrievedChunk

EVAL_SET_FILE = Path("evals/eval_set.yaml")


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Read the eval questions from a YAML list; each entry is validated as an EvalCase."""
    raw_cases = yaml.safe_load(path.read_text())
    return [EvalCase.model_validate(raw_case) for raw_case in raw_cases]


def retrieval_hit(case: EvalCase, retrieved: list[RetrievedChunk]) -> bool:
    """True if any retrieved chunk comes from one of the case's expected documents."""
    retrieved_doc_ids = {result.chunk.metadata.doc_id for result in retrieved}
    return any(doc_id in retrieved_doc_ids for doc_id in case.expected_docs)


def missing_key_facts(case: EvalCase, answer_text: str) -> list[str]:
    """The key facts the answer doesn't contain (ignoring case); empty means correct."""
    answer_lower = answer_text.lower()
    return [fact for fact in case.key_facts if fact.lower() not in answer_lower]
