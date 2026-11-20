"""Find the chunks most relevant to a question.

Layout — one class per file:
- base.py     BaseRetriever: retrieve(query, top_k) -> list[RetrievedChunk], best first.
- vector.py   VectorRetriever, strategy "vector": semantic search over the vector store,
              chunk text and metadata loaded from the document database.
- keyword.py  KeywordRetriever, strategy "keyword": exact-word BM25 search (rank_bm25)
              over chunk text loaded from the document database.
- factory.py  RETRIEVER_STRATEGIES and create_retriever(), which picks a strategy by name.

To add a strategy: write a BaseRetriever subclass in its own file, add its name to
RETRIEVER_STRATEGIES and build it in create_retriever (factory.py).
"""

from rag.retrieval.base import BaseRetriever
from rag.retrieval.factory import RETRIEVER_STRATEGIES, create_retriever
from rag.retrieval.keyword import KeywordRetriever
from rag.retrieval.vector import VectorRetriever

__all__ = [
    "RETRIEVER_STRATEGIES",
    "BaseRetriever",
    "KeywordRetriever",
    "VectorRetriever",
    "create_retriever",
]
