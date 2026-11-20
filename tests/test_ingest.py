"""Tests for the chunk step: syncing a folder of markdown files into the document database."""

from rag.chunking import create_chunker
from rag.documents import DocumentRepository, iter_documents
from rag.ingest import Ingestor

IN_MEMORY_SQLITE = "sqlite://"
NO_SIZE_LIMIT = 5_000_000


def make_ingestor():
    """Return an Ingestor over an empty in-memory database (reach it via .repository)."""
    repository = DocumentRepository(IN_MEMORY_SQLITE)
    chunker = create_chunker(strategy="token", chunk_size=800, chunk_overlap=150)
    return Ingestor(repository=repository, chunker=chunker)


def write_docs(folder):
    """Write two small policy files into folder."""
    (folder / "pto.md").write_text("# PTO\n\nEmployees get 20 days of paid time off.")
    (folder / "travel.md").write_text("# Travel\n\nBook flights through the portal.")


def test_second_run_on_unchanged_files_rechunks_nothing(tmp_path):
    """First run adds every file; running again on the same files leaves them all unchanged."""
    ingestor = make_ingestor()
    write_docs(tmp_path)

    first_report = ingestor.ingest_folder(tmp_path, NO_SIZE_LIMIT)
    second_report = ingestor.ingest_folder(tmp_path, NO_SIZE_LIMIT)

    assert first_report.added == ["pto", "travel"]
    assert second_report.added == []
    assert second_report.updated == []
    assert second_report.unchanged == ["pto", "travel"]


def test_edited_file_is_rechunked_and_deleted_file_is_removed(tmp_path):
    """An edit replaces that document's chunks; a deleted file's document and chunks are removed."""
    ingestor = make_ingestor()
    write_docs(tmp_path)
    ingestor.ingest_folder(tmp_path, NO_SIZE_LIMIT)
    old_chunk_ids = []
    for document in iter_documents(tmp_path, NO_SIZE_LIMIT):
        for chunk in ingestor.chunker.chunk_document(document):
            old_chunk_ids.append(chunk.chunk_id)

    (tmp_path / "pto.md").write_text("# PTO\n\nEmployees get 25 days of paid time off.")
    (tmp_path / "travel.md").unlink()
    report = ingestor.ingest_folder(tmp_path, NO_SIZE_LIMIT)

    assert report.updated == ["pto"]
    assert report.removed == ["travel"]
    assert list(ingestor.repository.get_content_hashes()) == ["pto"]
    # Every old id is gone: pto's ids changed with its content hash, travel's were deleted.
    assert ingestor.repository.get_chunks(old_chunk_ids) == []
