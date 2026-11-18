"""Tests for loading markdown files into Document objects."""

from datetime import date
from pathlib import Path

from rag.documents import load_document, load_documents

POLICY_FOLDER = Path(__file__).parent.parent / "data" / "policies"


def find_document(documents, doc_id):
    """Return the document whose doc_id matches, so tests can look one up by file name."""
    matching = [document for document in documents if document.doc_id == doc_id]
    return matching[0]


def test_front_matter_is_parsed_and_removed_from_text(tmp_path):
    """YAML front matter fills the metadata fields and is not part of the chunkable text."""
    path = tmp_path / "travel_policy.md"
    path.write_text(
        "---\n"
        "title: Travel Policy\n"
        "policy_id: TRV-001\n"
        'version: "1.2"\n'
        "effective_date: 2026-02-01\n"
        "---\n"
        "# Travel\n"
        "Book trains for trips under three hours.\n"
    )

    document = load_document(path)

    assert document.title == "Travel Policy"
    assert document.policy_id == "TRV-001"
    assert document.version == "1.2"
    assert document.effective_date == date(2026, 2, 1)
    assert document.text.startswith("# Travel")
    assert "policy_id" not in document.text


def test_missing_front_matter_falls_back_to_heading_then_file_name(tmp_path):
    """Without front matter: title comes from the first '# ' heading, else the file name."""
    with_heading = tmp_path / "faq.md"
    with_heading.write_text("Intro line\n# Employee FAQ\nQ: Who do I ask?\n")
    without_heading = tmp_path / "loose_notes.md"
    without_heading.write_text("Just some notes with no heading at all.\n")

    document_with_heading = load_document(with_heading)
    document_without_heading = load_document(without_heading)

    assert document_with_heading.title == "Employee FAQ"
    assert document_with_heading.policy_id == ""
    assert document_with_heading.version == ""
    assert document_with_heading.effective_date is None
    assert document_without_heading.title == "loose_notes"


def test_stale_expense_duplicate_is_distinguishable_by_version():
    """Both expense files share a policy ID; only version and date reveal the planted stale copy."""
    documents = load_documents(POLICY_FOLDER)

    current_policy = find_document(documents, "expense_reimbursement_policy")
    stale_policy = find_document(documents, "expense_reimbursement_policy_2023")

    assert current_policy.policy_id == stale_policy.policy_id == "EXP-001"
    assert current_policy.version == "3.0"
    assert stale_policy.version == "2.1"
    assert stale_policy.effective_date < current_policy.effective_date
