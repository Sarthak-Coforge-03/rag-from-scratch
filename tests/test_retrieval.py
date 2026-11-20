"""Tests for retrievers: given a question, return the most relevant chunks with metadata."""

import pytest

from rag.chunking import create_chunker
from rag.documents import DocumentRepository
from rag.embed import sync_vectors
from rag.embedding import BaseEmbedder
from rag.ingest import Ingestor
from rag.models import Chunk, ChunkMetadata, RetrievedChunk
from rag.retrieval import BaseRetriever, HybridRetriever, create_retriever
from rag.retrieval.keyword import split_into_words
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


def make_keyword_retriever(parts):
    """Keyword search only needs the document database; the other parts are ignored."""
    return create_retriever(
        strategy="keyword",
        repository=parts["repository"],
        embedder=parts["embedder"],
        vector_store=parts["vector_store"],
    )


def test_keyword_retriever_finds_an_exact_term_and_ignores_punctuation(tmp_path):
    """ "$75" appears only in the expenses doc; a query with no words returns nothing."""
    parts = make_search_parts(tmp_path)
    retriever = make_keyword_retriever(parts)

    results = retriever.retrieve("$75", top_k=2)
    no_words_results = retriever.retrieve("?!", top_k=2)

    assert [result.chunk.metadata.doc_id for result in results] == ["expenses"]
    assert results[0].chunk.metadata.title == "Expenses"
    assert no_words_results == []


def test_keyword_words_keep_codes_and_numbers_whole():
    """Section codes and amounts stay one word, so "SEC-4" can't match any chunk with a 4.

    Joining characters (- . ,) only count inside a word; sentence punctuation and
    filler words ("who", "or") are dropped, so they can't outweigh the code.
    """
    words = split_into_words("Who approves $2,500 under SEC-7.2 or EXP-4? Ask HR.")

    assert words == ["approves", "2,500", "under", "sec-7.2", "exp-4", "ask", "hr"]


def test_keyword_retriever_built_after_rechunk_sees_the_new_text(tmp_path):
    """A keyword retriever built after re-ingesting finds new text and not old text.

    Its BM25 index is a snapshot of the chunks in SQL when it is created, so after the
    chunk step a new retriever is built (each script run creates its own anyway).
    """
    parts = make_search_parts(tmp_path)
    docs_folder = tmp_path / "docs"
    (docs_folder / "expenses.md").write_text("# Expenses\n\nAsk your manager first.")
    (docs_folder / "security.md").unlink()
    chunker = create_chunker(strategy="token", chunk_size=800, chunk_overlap=0)
    Ingestor(repository=parts["repository"], chunker=chunker).ingest_folder(
        docs_folder, NO_SIZE_LIMIT
    )

    retriever = make_keyword_retriever(parts)

    assert retriever.retrieve("$75", top_k=2) == []
    assert retriever.retrieve("passwords", top_k=2) == []
    new_text_results = retriever.retrieve("manager", top_k=2)
    assert [result.chunk.metadata.doc_id for result in new_text_results] == ["expenses"]


def make_chunk(chunk_id: str) -> Chunk:
    """A placeholder chunk; only its id matters to the fusion arithmetic."""
    metadata = ChunkMetadata(
        doc_id=chunk_id,
        title=chunk_id,
        policy_id="",
        version="",
        effective_date="",
        source_path="placeholder.md",
        chunk_index=0,
        start_index=0,
        end_index=0,
        token_count=0,
    )
    return Chunk(chunk_id=chunk_id, text=chunk_id, metadata=metadata)


class FixedListRetriever(BaseRetriever):
    """Fake retriever: returns the same ranked chunk ids for every query."""

    def __init__(self, ranked_chunk_ids: list[str]):
        """Remember the ranking to return, best first."""
        self.ranked_chunk_ids = ranked_chunk_ids

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the fixed ranking, cut to top_k; scores are unused by fusion."""
        return [
            RetrievedChunk(chunk=make_chunk(chunk_id), score=0.0)
            for chunk_id in self.ranked_chunk_ids[:top_k]
        ]


def test_hybrid_ranks_a_chunk_found_by_both_retrievers_first():
    """Reciprocal rank fusion: each list gives a chunk 1 / (60 + rank); the sums decide.

    vector A,B,C and keyword D,A -> A gets 1/61 + 1/62 and wins; D (1/61) beats
    B (1/62) beats C (1/63).
    """
    vector_like = FixedListRetriever(["A", "B", "C"])
    keyword_like = FixedListRetriever(["D", "A"])
    hybrid = HybridRetriever(retrievers=[vector_like, keyword_like])

    results = hybrid.retrieve("any question", top_k=4)

    assert [result.chunk.chunk_id for result in results] == ["A", "D", "B", "C"]
    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_hybrid_strategy_combines_vector_and_keyword_results(tmp_path):
    """ "meal $75": vector finds expenses by meaning ("meal"), keyword by the exact "75"."""
    parts = make_search_parts(tmp_path)
    retriever = create_retriever(
        strategy="hybrid",
        repository=parts["repository"],
        embedder=parts["embedder"],
        vector_store=parts["vector_store"],
    )

    results = retriever.retrieve("meal $75", top_k=1)

    assert [result.chunk.metadata.doc_id for result in results] == ["expenses"]
