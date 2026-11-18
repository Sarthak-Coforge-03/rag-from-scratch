"""Everything about source documents: reading them from disk (and, later, storing them)."""

from rag.documents.loader import iter_documents, load_document

__all__ = ["iter_documents", "load_document"]
