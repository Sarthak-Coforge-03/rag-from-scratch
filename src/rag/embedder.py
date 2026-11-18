from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer


class BaseEmbedder(ABC):
    """Turns text into vectors. Subclasses wrap one embedding library or API."""

    def __init__(self, model_name: str):
        """Remember which model this embedder uses."""
        self.model_name = model_name

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many chunk texts at once, for storing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed one user question, for searching."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings with sentence-transformers; the model is loaded once."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()
