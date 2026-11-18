"""Tests for storing documents and chunks in SQLite and reading chunks back by id."""

from pathlib import Path

from rag.chunk import create_chunker
from rag.documents import DocumentRepository, load_document

PTO_POLICY_PATH = Path(__file__).parent.parent / "data" / "policies" / "pto_policy.md"
IN_MEMORY_SQLITE = "sqlite://"


def make_repository_and_pto_chunks():
    """Return an empty in-memory repository, the PTO policy Document and its Chunks."""
    repository = DocumentRepository(IN_MEMORY_SQLITE)
    document = load_document(PTO_POLICY_PATH)
    chunker = create_chunker(strategy="token", chunk_size=800, chunk_overlap=150)
    chunks = chunker.chunk_document(document)
    return repository, document, chunks


def test_get_chunks_returns_saved_chunks_in_requested_order():
    """Chunks round-trip through the database unchanged, in the order the ids were asked for."""
    repository, document, chunks = make_repository_and_pto_chunks()
    repository.save_document(document, chunks)

    # Reverse order stands in for Chroma's ranking, which is not the order chunks were saved in.
    requested_ids = [chunk.chunk_id for chunk in reversed(chunks)]
    loaded_chunks = repository.get_chunks(requested_ids)

    assert [chunk.chunk_id for chunk in loaded_chunks] == requested_ids
    assert loaded_chunks == list(reversed(chunks))


def test_saving_a_document_again_replaces_its_chunks():
    """Re-saving a document leaves no duplicates and reports the chunk ids it replaced."""
    repository, document, chunks = make_repository_and_pto_chunks()
    first_save_ids = [chunk.chunk_id for chunk in chunks]

    repository.save_document(document, chunks)
    replaced_ids = repository.save_document(document, chunks)

    assert replaced_ids == first_save_ids
    assert len(repository.get_chunks(first_save_ids)) == len(chunks)
