"""Tests for the eval scoring rules: what counts as a retrieval hit and a correct answer."""

from rag.evaluation import missing_key_facts, retrieval_hit
from rag.models import Chunk, ChunkMetadata, EvalCase, RetrievedChunk

MEAL_LIMIT_CASE = EvalCase(
    id="meal-limit",
    question="What is the daily meal limit on overnight trips?",
    expected_docs=["expense_reimbursement_policy", "employee_faq"],
    key_facts=["75", "per day"],
)


def make_result(doc_id: str) -> RetrievedChunk:
    """A retrieved chunk from document `doc_id`; only the doc_id matters here."""
    metadata = ChunkMetadata(
        doc_id=doc_id,
        title=doc_id,
        policy_id="",
        version="",
        effective_date="",
        source_path=f"{doc_id}.md",
        chunk_index=0,
        start_index=0,
        end_index=0,
        token_count=0,
    )
    chunk = Chunk(chunk_id=f"{doc_id}::0", text="", metadata=metadata)
    return RetrievedChunk(chunk=chunk, score=1.0)


def test_retrieval_hit_needs_any_one_expected_document():
    """Retrieving any expected doc is a hit; only other docs is a miss."""
    with_faq = [make_result("pto_policy"), make_result("employee_faq")]
    without_expected = [make_result("pto_policy"), make_result("glossary")]

    assert retrieval_hit(MEAL_LIMIT_CASE, with_faq)
    assert not retrieval_hit(MEAL_LIMIT_CASE, without_expected)


def test_missing_key_facts_lists_what_the_answer_lacks_ignoring_case():
    """A correct answer has every fact (any case); a wrong one lists the facts it lacks."""
    assert missing_key_facts(MEAL_LIMIT_CASE, "Up to $75 PER DAY for meals.") == []
    assert missing_key_facts(MEAL_LIMIT_CASE, "Up to $50 per day.") == ["75"]
