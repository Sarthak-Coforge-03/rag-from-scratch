"""ChromaVectorStore: chunk ids + embeddings in a local, on-disk Chroma collection."""

from pathlib import Path

import chromadb

from rag.models import VectorMatch
from rag.vectorstore.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """A persistent Chroma collection holding only chunk ids and their embeddings.

    No documents or metadata are stored in Chroma; they live in the document database.
    """

    def __init__(self, location: str | Path, collection_name: str):
        """Open (or create) the collection stored in the folder `location`.

        `location` is the folder Chroma keeps its files in (e.g. ".chroma"); every
        vector store takes (location, collection_name) so the factory can build any
        of them the same way.

        embedding_function=None stops Chroma from embedding text with its own built-in
        model: every vector comes from our embedder. Cosine distance matches the
        normalized vectors our embedder produces.
        """
        self.client = chromadb.PersistentClient(path=str(location))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def upsert(self, chunk_ids: list[str], embeddings: list[list[float]]) -> None:
        """Insert or replace the embedding for each chunk id."""
        if not chunk_ids:
            return
        self.collection.upsert(ids=chunk_ids, embeddings=embeddings)

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove these chunk ids; an empty list does nothing.

        The early return matters: Chroma raises ValueError for an empty id list, and
        callers often have nothing to delete (e.g. a document saved for the first time).
        """
        if not chunk_ids:
            return
        self.collection.delete(ids=chunk_ids)

    def query(self, embedding: list[float], top_k: int) -> list[VectorMatch]:
        """Return up to top_k chunk ids nearest to the embedding, closest first."""
        # Chroma can search for several query vectors at once, so it takes a list of
        # them; we always send exactly one.
        # include=["distances"] asks only for ids (always returned) and distances —
        # not the stored embeddings, documents or metadata, which we don't need here.
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["distances"],
        )

        # Results come back as one list per query vector: results["ids"] looks like
        # [["x_chunk", "y_chunk"]]. Index [0] picks the lists for our single query.
        # The two lists are parallel: position 0 in both is the closest hit.
        ids_for_this_query = results["ids"][0]
        distances_for_this_query = results["distances"][0]

        # Pair each id with its distance by position, keeping Chroma's closest-first
        # order, so callers get VectorMatch objects instead of indexing parallel lists.
        matches = []
        for position in range(len(ids_for_this_query)):
            match = VectorMatch(
                chunk_id=ids_for_this_query[position],
                distance=distances_for_this_query[position],
            )
            matches.append(match)
        return matches

    def count(self) -> int:
        """Return how many chunk ids are stored in the collection."""
        return self.collection.count()
