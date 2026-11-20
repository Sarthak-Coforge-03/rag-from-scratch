"""Pick a vector database by name (VECTOR_STORE in config.py)."""

from rag.vectorstore.base import BaseVectorStore
from rag.vectorstore.chroma import ChromaVectorStore

# Store name (VECTOR_STORE) -> vector store class. Add new databases here.
# Every class takes the same two arguments: (location, collection_name).
VECTOR_STORE_CLASSES = {
    "chroma": ChromaVectorStore,
}


def create_vector_store(
    store: str, location: str, collection_name: str
) -> BaseVectorStore:
    """Build the vector store for a store name, e.g. settings.vector_store.

    location says where the store lives, as plain text so every database can use it:
    a folder for Chroma (e.g. IndexLayout.vector_dir(...)), or a connection URL for a server-based
    database added later. collection_name is the collection or table to use.

    Raises ValueError listing the valid names if the store is unknown, so a typo in
    VECTOR_STORE fails immediately instead of silently using a different database.
    """
    if store not in VECTOR_STORE_CLASSES:
        valid_names = ", ".join(VECTOR_STORE_CLASSES)
        raise ValueError(
            f"Unknown vector store {store!r}; choose one of: {valid_names}"
        )
    store_class = VECTOR_STORE_CLASSES[store]
    return store_class(location=location, collection_name=collection_name)
