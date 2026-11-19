"""BaseEmbedder: the interface every embedding provider implements.

Two methods instead of one, because some models (e5, bge, most API embedders) expect
questions and stored passages to be prepared differently. With MiniLM both do the
same thing; only a provider's subclass needs to know the difference.
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Turns text into vectors. Subclasses wrap one embedding library or API."""

    def __init__(self, model_name: str, batch_size: int):
        """Remember which model to use and how many texts to embed per batch.

        batch_size caps how many texts go to the model at once, so even a document
        with thousands of chunks never has to be embedded in a single call.
        """
        self.model_name = model_name
        self.batch_size = batch_size

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many chunk texts, for storing; returns one vector per text, in order."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed one user question, for searching."""
