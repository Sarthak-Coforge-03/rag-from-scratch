"""Tests for loading policy markdown files into Document objects."""

from datetime import date
from pathlib import Path

from rag.documents import load_documents

POLICY_FOLDER = Path(__file__).parent.parent / "data" / "policies"


def find_document(documents, doc_id):
    """Return the document whose doc_id matches, so tests can look one up by file name."""
    matching = [document for document in documents if document.doc_id == doc_id]
    return matching[0]


def test_loads_every_policy_with_its_metadata():
    """All five files load, and the PTO policy's header metadata and body text are parsed."""
    documents = load_documents(POLICY_FOLDER)
    assert len(documents) == 5

    pto_policy = find_document(documents, "pto_policy")
    assert pto_policy.title == "Paid Time Off (PTO) Policy"
    assert pto_policy.policy_id == "PTO-001"
    assert pto_policy.version == "4.0"
    assert pto_policy.effective_date == date(2026, 1, 1)
    assert "## PTO-5 Carryover" in pto_policy.text


def test_stale_expense_duplicate_is_distinguishable_by_version():
    """Both expense files share a policy ID; only version and date reveal the planted stale copy."""
    documents = load_documents(POLICY_FOLDER)

    current_policy = find_document(documents, "expense_reimbursement_policy")
    stale_policy = find_document(documents, "expense_reimbursement_policy_2023")

    assert current_policy.policy_id == stale_policy.policy_id == "EXP-001"
    assert current_policy.version == "3.0"
    assert stale_policy.version == "2.1"
    assert stale_policy.effective_date < current_policy.effective_date
