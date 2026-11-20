"""Ingestor: the chunk step. Syncs a folder of markdown files into the document database.

Writes SQL only. Vector folders are not touched here: after a run they may be stale,
and the embed step syncs each one by comparing its chunk ids with the database.
"""

import logging
from pathlib import Path

from rag.chunking import BaseChunker
from rag.documents import DocumentRepository, iter_documents
from rag.models import IngestReport

logger = logging.getLogger(__name__)


class Ingestor:
    """Chunks new or edited documents into the repository and removes deleted ones."""

    def __init__(self, repository: DocumentRepository, chunker: BaseChunker):
        """Ingest into `repository`, splitting documents with `chunker`."""
        self.repository = repository
        self.chunker = chunker

    def ingest_folder(self, folder: Path, max_bytes: int) -> IngestReport:
        """Make the database match the .md files in `folder`; return what changed.

        Each file is compared with the stored content hash:
          - not stored yet      -> added   (chunked and saved)
          - different hash      -> updated (re-chunked; old chunks replaced in one transaction)
          - same hash           -> unchanged (skipped, not re-chunked)
        Stored documents with no file anymore are removed with their chunks.

        Note: a file skipped for being over max_bytes counts as gone, so an older
        stored copy of it is removed (the loader logs a warning for the skip).
        """
        stored_hashes = self.repository.get_content_hashes()
        report = IngestReport()
        doc_ids_on_disk = set()

        for document in iter_documents(folder, max_bytes):
            doc_ids_on_disk.add(document.doc_id)
            stored_hash = stored_hashes.get(document.doc_id)
            if stored_hash == document.content_hash:
                report.unchanged.append(document.doc_id)
                continue

            chunks = self.chunker.chunk_document(document)
            self.repository.save_document(document, chunks)
            if stored_hash is None:
                report.added.append(document.doc_id)
            else:
                report.updated.append(document.doc_id)
            logger.info("Saved %s (%d chunks)", document.doc_id, len(chunks))

        # sorted() so the report (and log) order doesn't depend on dict order.
        for doc_id in sorted(stored_hashes):
            if doc_id not in doc_ids_on_disk:
                self.repository.delete_document(doc_id)
                report.removed.append(doc_id)
                logger.info("Removed %s (file no longer in %s)", doc_id, folder)

        return report
