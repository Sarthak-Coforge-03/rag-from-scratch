"""BaseVectorStore: the interface every vector database implements.

A vector store only knows chunk ids and their embeddings. Chunk text and metadata
live in the document database (rag.documents), so swapping the vector database
(e.g. Chroma for pgvector or Qdrant) means writing one new subclass.

Writes are upsert-only (there is no "add"): writing an id that already exists
replaces its vector, so running the same write twice never creates duplicates.
"""

from abc import ABC, abstractmethod

from rag.models import VectorMatch


class BaseVectorStore(ABC):
    """Stores one embedding per chunk id and finds the ids nearest to a query vector."""

    @abstractmethod
    def upsert(self, chunk_ids: list[str], embeddings: list[list[float]]) -> None:
        """Insert or replace the embedding for each chunk id (same order in both lists)."""

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Remove these chunk ids; ids that aren't stored are ignored."""

    @abstractmethod
    def query(self, embedding: list[float], top_k: int) -> list[VectorMatch]:
        """Return up to top_k chunk ids nearest to the embedding, closest first."""

    @abstractmethod
    def list_ids(self) -> list[str]:
        """Return every stored chunk id, sorted (used to sync with the document database)."""

    @abstractmethod
    def count(self) -> int:
        """Return how many chunk ids are stored."""
