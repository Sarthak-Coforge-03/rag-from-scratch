"""BaseRetriever: the interface every search strategy implements.

A retriever turns a question into a ranked list of chunks. Whatever it searches
(vectors, keywords, or both), it returns full chunks from the document database,
so the answer step can cite each chunk's source.
"""

from abc import ABC, abstractmethod

from rag.models import RetrievedChunk


class BaseRetriever(ABC):
    """Finds the chunks most relevant to a question."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to top_k chunks for the query, most relevant (highest score) first."""
