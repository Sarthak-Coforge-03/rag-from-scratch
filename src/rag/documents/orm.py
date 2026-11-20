"""Database tables for documents and chunks

These rows are storage only. The rest of the app works with the pydantic models in
rag.models; repository.py converts between the two.

Chroma stores just chunk ids and embeddings; the text and metadata for each id
live here, in the `chunks` and `documents` tables.
"""

from datetime import date

from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared base class that collects every table definition below."""


class DocumentRow(Base):
    """One source document: its metadata and full body text."""

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str]
    policy_id: Mapped[str]  # "" when the document has none
    version: Mapped[str]  # "" when the document has none
    effective_date: Mapped[date | None]
    source_path: Mapped[str]
    text: Mapped[str] = mapped_column(Text)
    # SHA-256 of the raw file; an unchanged hash means ingestion can skip the file.
    content_hash: Mapped[str]

    # Deleting a document deletes its chunks too ("delete-orphan").
    chunks: Mapped[list["ChunkRow"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ChunkRow.chunk_index",
    )


class ChunkRow(Base):
    """One chunk of a document: its text and where it sits in the document."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(primary_key=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id"), index=True)
    chunk_index: Mapped[int]
    start_index: Mapped[int]
    end_index: Mapped[int]
    token_count: Mapped[int]
    section: Mapped[str]  # nearest heading above the chunk; "" if none
    text: Mapped[str] = mapped_column(Text)

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")
