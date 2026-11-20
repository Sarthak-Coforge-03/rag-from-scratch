"""Tests for retrievers: given a question, return the most relevant chunks with metadata."""

import pytest

from rag.chunking import create_chunker
from rag.documents import DocumentRepository
from rag.embed import sync_vectors
from rag.embedding import BaseEmbedder
from rag.ingest import Ingestor
from rag.retrieval import create_retriever
from rag.vectorstore import ChromaVectorStore

NO_SIZE_LIMIT = 5_000_000


class TopicEmbedder(BaseEmbedder):
    """Fake embedder: one axis per topic word, so the expected ranking is obvious.

    Texts mentioning "meal" point along x, "password" along y, anything else along z.
    No model is downloaded.
    """

    def __init__(self):
        """No real model; batch size only matters for sync_vectors."""
        super().__init__(model_name="topic", batch_size=8)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """One topic vector per text."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """[1,0,0] for meals, [0,1,0] for passwords, [0,0,1] otherwise."""
        lowered = text.lower()
        if "meal" in lowered:
            return [1.0, 0.0, 0.0]
        if "password" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def make_search_parts(tmp_path):
    """Ingest two one-chunk docs into in-memory SQL, embed them into Chroma.

    Returns a dict of the parts a retriever needs, so tests pick what they use.
    """
    docs_folder = tmp_path / "docs"
    docs_folder.mkdir()
    (docs_folder / "expenses.md").write_text("# Expenses\n\nMeal allowance is $75/day.")
    (docs_folder / "security.md").write_text("# Security\n\nPasswords need 14 chars.")
    repository = DocumentRepository("sqlite://")
    chunker = create_chunker(strategy="token", chunk_size=800, chunk_overlap=0)
    Ingestor(repository=repository, chunker=chunker).ingest_folder(
        docs_folder, NO_SIZE_LIMIT
    )
    embedder = TopicEmbedder()
    vector_store = ChromaVectorStore(
        location=tmp_path / "vectors", collection_name="chunks"
    )
    sync_vectors(repository, embedder, vector_store)
    return {
        "repository": repository,
        "embedder": embedder,
        "vector_store": vector_store,
    }


def test_vector_retriever_ranks_the_relevant_chunk_first_with_its_metadata(tmp_path):
    """A meals question returns the expenses chunk first, best score first, at most top_k."""
    parts = make_search_parts(tmp_path)
    retriever = create_retriever(
        strategy="vector",
        repository=parts["repository"],
        embedder=parts["embedder"],
        vector_store=parts["vector_store"],
    )

    results = retriever.retrieve("What is the meal limit?", top_k=2)
    top_result_only = retriever.retrieve("What is the meal limit?", top_k=1)

    assert [result.chunk.metadata.doc_id for result in results] == [
        "expenses",
        "security",
    ]
    assert results[0].chunk.metadata.title == "Expenses"  # metadata for attribution
    assert results[0].score >= results[1].score
    assert len(top_result_only) == 1


def test_unknown_retriever_strategy_is_rejected(tmp_path):
    """A typo in RETRIEVER fails immediately instead of silently using another strategy."""
    parts = make_search_parts(tmp_path)

    with pytest.raises(ValueError):
        create_retriever(
            strategy="nope",
            repository=parts["repository"],
            embedder=parts["embedder"],
            vector_store=parts["vector_store"],
        )
