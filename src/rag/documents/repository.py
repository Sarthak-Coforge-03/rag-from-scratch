"""Save documents and chunks to the database and read chunks back by id."""

from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload

from rag.documents.orm import Base, ChunkRow, DocumentRow
from rag.models import Chunk, ChunkMetadata, Document


class DocumentRepository:
    """The only place that reads or writes the documents and chunks tables.

    The repository owns its database connection: give it a URL and it creates the
    engine and the tables. Switching databases (e.g. SQLite to Postgres) is just a
    different URL, such as IndexLayout.database_url; callers never touch the engine.
    """

    def __init__(self, database_url: str):
        """Connect to `database_url` and create the tables if they don't exist yet.

        Use an index's IndexLayout.database_url (e.g. "sqlite:///indexes/token-800/sql/docs.db")
        for a file, or "sqlite://" for a throwaway in-memory database (handy in tests).
        """
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)

    def close(self) -> None:
        """Release the database connections held by this repository."""
        self.engine.dispose()

    def save_document(self, document: Document, chunks: list[Chunk]) -> list[str]:
        """Store a document and its chunks, replacing any earlier copy of that document.

        Runs in one transaction, so a failure leaves the old copy untouched.
        Returns the chunk ids of the earlier copy (empty if there was none), so the
        caller can remove those ids from the vector store as well.
        """
        with Session(self.engine) as session, session.begin():
            replaced_chunk_ids = []
            existing_row = session.get(DocumentRow, document.doc_id)
            if existing_row is not None:
                replaced_chunk_ids = [chunk.chunk_id for chunk in existing_row.chunks]
                session.delete(existing_row)
                # Send the delete now, before inserting a new row with the same doc_id.
                session.flush()

            document_row = document_to_row(document)
            document_row.chunks = [chunk_to_row(chunk) for chunk in chunks]
            session.add(document_row)
        return replaced_chunk_ids

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Load chunks by id, in the same order as `chunk_ids`; unknown ids are skipped.

        Keeping the caller's order matters because the ids come ranked from the
        vector search, best match first.
        """
        query = (
            select(ChunkRow)
            .where(ChunkRow.chunk_id.in_(chunk_ids))
            .options(selectinload(ChunkRow.document))
        )
        with Session(self.engine) as session:
            rows = session.scalars(query).all()
            chunks_by_id = {row.chunk_id: row_to_chunk(row) for row in rows}

        return [
            chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id
        ]

    def get_all_chunk_ids(self) -> list[str]:
        """Return every stored chunk id, sorted.

        The embed step compares these with the ids in a vector folder to find what
        to embed and what to delete.
        """
        query = select(ChunkRow.chunk_id).order_by(ChunkRow.chunk_id)
        with Session(self.engine) as session:
            return list(session.scalars(query).all())

    def get_content_hashes(self) -> dict[str, str]:
        """Return {doc_id: content_hash} for every stored document.

        Ingestion compares these with the files on disk: same hash means the file is
        unchanged and can be skipped; a doc_id with no file anymore means it was
        deleted or renamed and should be removed.
        """
        query = select(DocumentRow.doc_id, DocumentRow.content_hash)
        with Session(self.engine) as session:
            rows = session.execute(query).all()
        return {row.doc_id: row.content_hash for row in rows}

    def delete_document(self, doc_id: str) -> list[str]:
        """Delete a document and its chunks; return the deleted chunk ids.

        The ids are returned so the caller can delete the same chunks from the
        vector store. An unknown doc_id deletes nothing and returns [].
        """
        with Session(self.engine) as session, session.begin():
            existing_row = session.get(DocumentRow, doc_id)
            if existing_row is None:
                return []
            removed_chunk_ids = [chunk.chunk_id for chunk in existing_row.chunks]
            session.delete(existing_row)
        return removed_chunk_ids


def document_to_row(document: Document) -> DocumentRow:
    """Copy a pydantic Document into a new DocumentRow (without its chunks)."""
    return DocumentRow(
        doc_id=document.doc_id,
        title=document.title,
        policy_id=document.policy_id,
        version=document.version,
        effective_date=document.effective_date,
        source_path=str(document.source_path),
        text=document.text,
        content_hash=document.content_hash,
    )


def chunk_to_row(chunk: Chunk) -> ChunkRow:
    """Copy a pydantic Chunk into a new ChunkRow; document fields live on DocumentRow."""
    return ChunkRow(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.metadata.doc_id,
        chunk_index=chunk.metadata.chunk_index,
        start_index=chunk.metadata.start_index,
        end_index=chunk.metadata.end_index,
        token_count=chunk.metadata.token_count,
        text=chunk.text,
    )


def row_to_chunk(row: ChunkRow) -> Chunk:
    """Rebuild a pydantic Chunk from a ChunkRow and its DocumentRow."""
    document_row = row.document
    if document_row.effective_date is None:
        effective_date = ""
    else:
        effective_date = document_row.effective_date.isoformat()

    metadata = ChunkMetadata(
        doc_id=document_row.doc_id,
        title=document_row.title,
        policy_id=document_row.policy_id,
        version=document_row.version,
        effective_date=effective_date,
        source_path=Path(document_row.source_path),
        chunk_index=row.chunk_index,
        start_index=row.start_index,
        end_index=row.end_index,
        token_count=row.token_count,
    )
    return Chunk(chunk_id=row.chunk_id, text=row.text, metadata=metadata)
