"""Reorder retrieved chunks with a model that reads the question and each chunk together.

Layout — one class per file:
- base.py           BaseReranker: rerank(query, candidates, top_k); subclasses only score.
- cross_encoder.py  CrossEncoderReranker, provider "cross-encoder": local sentence-transformers
                    cross-encoder.
- factory.py        RERANKER_CLASSES and create_reranker(), which picks a provider by name.

To use a reranker in search, wrap a retriever in retrieval.RerankedRetriever.
To add a reranker: write a BaseReranker subclass in its own file and register it in
RERANKER_CLASSES (factory.py).
"""

from rag.reranking.base import BaseReranker
from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.reranking.factory import RERANKER_CLASSES, create_reranker

__all__ = [
    "RERANKER_CLASSES",
    "BaseReranker",
    "CrossEncoderReranker",
    "create_reranker",
]
