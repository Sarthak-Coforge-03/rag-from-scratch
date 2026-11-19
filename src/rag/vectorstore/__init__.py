"""Store chunk embeddings and find the chunks nearest to a query.

Layout — one class per file:
- base.py    BaseVectorStore: upsert / delete / query / count by chunk id. Upsert-only, so
             repeating a write never duplicates. Text and metadata live in rag.documents.
- chroma.py  ChromaVectorStore: a persistent local Chroma collection, cosine distance.

To add a vector database: write a BaseVectorStore subclass in its own file.
"""

from rag.vectorstore.base import BaseVectorStore
from rag.vectorstore.chroma import ChromaVectorStore

__all__ = ["BaseVectorStore", "ChromaVectorStore"]
