"""Turn text into vectors.

Layout — one class per file:
- base.py                  BaseEmbedder: embed_documents (many texts, batched) and
                           embed_query (one question).
- sentence_transformer.py  SentenceTransformerEmbedder, provider "sentence-transformers":
                           local model, normalized vectors.
- factory.py               EMBEDDER_CLASSES and create_embedder(), which picks a provider by name.

To add a provider: write a BaseEmbedder subclass in its own file and register it in
EMBEDDER_CLASSES (factory.py).
"""

from rag.embedding.base import BaseEmbedder
from rag.embedding.factory import EMBEDDER_CLASSES, create_embedder
from rag.embedding.sentence_transformer import SentenceTransformerEmbedder

__all__ = [
    "EMBEDDER_CLASSES",
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "create_embedder",
]
