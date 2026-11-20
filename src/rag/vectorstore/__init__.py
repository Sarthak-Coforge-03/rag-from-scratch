"""Store chunk embeddings and find the chunks nearest to a query.

Layout — one class per file:
- base.py     BaseVectorStore: upsert / delete / query / list_ids / count by chunk id. Upsert-only, so
              repeating a write never duplicates. Text and metadata live in rag.documents.
- chroma.py   ChromaVectorStore: a persistent local Chroma collection, cosine distance.
- factory.py  VECTOR_STORE_CLASSES and create_vector_store(), which picks a store by name.

To add a vector database: write a BaseVectorStore subclass in its own file that takes
(location, collection_name), and register it in VECTOR_STORE_CLASSES (factory.py).
"""

from rag.vectorstore.base import BaseVectorStore
from rag.vectorstore.chroma import ChromaVectorStore
from rag.vectorstore.factory import VECTOR_STORE_CLASSES, create_vector_store

__all__ = [
    "VECTOR_STORE_CLASSES",
    "BaseVectorStore",
    "ChromaVectorStore",
    "create_vector_store",
]
