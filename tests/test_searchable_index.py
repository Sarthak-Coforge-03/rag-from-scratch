"""Tests for loading an index for search: picking the vector folder and checking it is current."""

import pytest

from rag.chunking import create_chunker
from rag.documents import DocumentRepository
from rag.embed import sync_vectors
from rag.embedding import BaseEmbedder
from rag.indexes.searchable_index import ensure_vectors_match_sql, pick_embedding_model
from rag.ingest import Ingestor
from rag.models import EmbedderConfig
from rag.vectorstore import ChromaVectorStore

NO_SIZE_LIMIT = 5_000_000
MINILM = EmbedderConfig(provider="sentence-transformers", model_name="all-MiniLM-L6-v2")
BGE = EmbedderConfig(
    provider="sentence-transformers", model_name="BAAI/bge-small-en-v1.5"
)


class ConstantEmbedder(BaseEmbedder):
    """Fake embedder: the same 2-number vector for every text (no model download)."""

    def __init__(self):
        """No real model."""
        super().__init__(model_name="constant", batch_size=8)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """[1.0, 0.0] for every text; only the ids matter to these tests."""
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Not used by these tests."""
        raise NotImplementedError


def test_pick_embedding_model_needs_an_unambiguous_choice():
    """One vector folder is picked automatically; with several, a model name decides."""
    assert pick_embedding_model([MINILM], model_name=None) == MINILM
    assert pick_embedding_model([BGE, MINILM], model_name="all-MiniLM-L6-v2") == MINILM

    with pytest.raises(ValueError):
        pick_embedding_model([BGE, MINILM], model_name=None)  # which one?
    with pytest.raises(ValueError):
        pick_embedding_model([MINILM], model_name="not-embedded-yet")
    with pytest.raises(ValueError):
        pick_embedding_model([], model_name=None)  # nothing embedded at all


def test_vectors_must_match_sql_before_searching(tmp_path):
    """Matching ids pass; a chunk added to SQL but not embedded is refused."""
    docs_folder = tmp_path / "docs"
    docs_folder.mkdir()
    (docs_folder / "pto.md").write_text("# PTO\n\nEmployees get 20 days off.")
    repository = DocumentRepository("sqlite://")
    chunker = create_chunker(strategy="token", chunk_size=800, chunk_overlap=0)
    ingestor = Ingestor(repository=repository, chunker=chunker)
    ingestor.ingest_folder(docs_folder, NO_SIZE_LIMIT)
    vector_store = ChromaVectorStore(
        location=tmp_path / "vectors", collection_name="chunks"
    )
    sync_vectors(repository, ConstantEmbedder(), vector_store)

    ensure_vectors_match_sql(repository, vector_store)  # in sync: no error

    (docs_folder / "travel.md").write_text("# Travel\n\nBook flights early.")
    # The chunk step runs again, but the embed step doesn't.
    ingestor.ingest_folder(docs_folder, NO_SIZE_LIMIT)
    with pytest.raises(ValueError):
        ensure_vectors_match_sql(repository, vector_store)
