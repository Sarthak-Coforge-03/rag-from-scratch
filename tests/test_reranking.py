"""Tests for reranking: reordering retrieved candidates by a second, closer look at each one."""

import pytest

from rag.models import Chunk, ChunkMetadata, RetrievedChunk
from rag.reranking import BaseReranker, create_reranker
from rag.retrieval import BaseRetriever, RerankedRetriever


def make_result(text: str, retriever_score: float) -> RetrievedChunk:
    """A retrieved chunk whose id and text are `text`; metadata is placeholder."""
    metadata = ChunkMetadata(
        doc_id=text,
        title=text,
        policy_id="",
        version="",
        effective_date="",
        source_path="placeholder.md",
        chunk_index=0,
        start_index=0,
        end_index=0,
        token_count=0,
    )
    chunk = Chunk(chunk_id=text, text=text, metadata=metadata)
    return RetrievedChunk(chunk=chunk, score=retriever_score)


class LengthReranker(BaseReranker):
    """Fake reranker: longer text = more relevant, so the expected order is obvious."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One score per text: its length."""
        return [float(len(text)) for text in texts]


class RecordingRetriever(BaseRetriever):
    """Fake retriever: returns fixed results and remembers how many it was asked for."""

    def __init__(self, results: list[RetrievedChunk]):
        """Return `results` (cut to top_k) on every call."""
        self.results = results
        self.requested_top_k = None

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Record top_k and return the first top_k fixed results."""
        self.requested_top_k = top_k
        return self.results[:top_k]


def test_rerank_reorders_by_reranker_score_and_keeps_top_k():
    """The retriever's order and scores are replaced by the reranker's."""
    candidates = [
        make_result("short", retriever_score=0.9),
        make_result("much longer text", retriever_score=0.5),
        make_result("medium text", retriever_score=0.7),
    ]

    reranked = LengthReranker().rerank("any question", candidates, top_k=2)

    assert [result.chunk.text for result in reranked] == [
        "much longer text",
        "medium text",
    ]
    assert reranked[0].score == len("much longer text")
    assert LengthReranker().rerank("any question", [], top_k=2) == []


def test_reranked_retriever_reranks_more_candidates_than_it_returns():
    """It fetches `candidates` results so a chunk ranked low can still be promoted."""
    results = [
        make_result("a", 0.9),
        make_result("bb", 0.8),
        make_result("ccc", 0.7),
        make_result("dddd", 0.6),
    ]
    inner_retriever = RecordingRetriever(results)
    retriever = RerankedRetriever(
        retriever=inner_retriever, reranker=LengthReranker(), candidates=4
    )

    top_results = retriever.retrieve("any question", top_k=1)

    assert inner_retriever.requested_top_k == 4  # asked for candidates, not top_k
    # "dddd" was last for the retriever but is first after reranking.
    assert [result.chunk.text for result in top_results] == ["dddd"]


def test_unknown_reranker_provider_is_rejected():
    """A typo in RERANK_PROVIDER fails immediately instead of loading another model."""
    with pytest.raises(ValueError):
        create_reranker(provider="nope", model_name="any-model")
