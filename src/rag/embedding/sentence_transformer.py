"""SentenceTransformerEmbedder: local embeddings (provider "sentence-transformers").

Runs the model on this machine — no API key, no cost. The default model,
all-MiniLM-L6-v2, produces 384-number vectors and reads at most 256 word-pieces per
text, which is why chunks are kept to about 800 characters (see chunking/).
"""

from sentence_transformers import SentenceTransformer

from rag.embedding.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings with sentence-transformers; the model is loaded once."""

    def __init__(self, model_name: str, batch_size: int):
        """Load the model once here; loading takes seconds, so never do it per call."""
        super().__init__(model_name, batch_size)
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts batch_size at a time, as unit-length vectors.

        normalize_embeddings=True scales every vector to length 1, which makes
        cosine distance (what the Chroma collection uses) well behaved.
        """
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed one question the same way as documents (MiniLM needs no query prefix)."""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()
