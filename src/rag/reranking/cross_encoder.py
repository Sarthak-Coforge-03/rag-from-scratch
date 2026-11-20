"""CrossEncoderReranker: local reranking with a sentence-transformers cross-encoder.

The default model, cross-encoder/ms-marco-MiniLM-L-6-v2, was trained on real search
queries to judge whether a passage answers a question. It runs on this machine (no
API key) and downloads about 90 MB the first time.
"""

from sentence_transformers import CrossEncoder

from rag.reranking.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """Scores each (question, chunk text) pair with a cross-encoder model, loaded once."""

    def __init__(self, model_name: str):
        """Load the model once here; loading takes seconds, so never do it per call."""
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score every (query, text) pair; higher = more relevant (unbounded, not 0-1)."""
        pairs = [(query, text) for text in texts]
        scores = self.model.predict(pairs)
        return [float(score) for score in scores]
