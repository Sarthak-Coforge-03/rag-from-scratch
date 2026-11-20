"""RerankedRetriever: a retriever whose results are reordered by a reranker.

It is a retriever (not a reranker): anything that takes a retriever, such as the
answer step or the eval harness, can use "hybrid + rerank" without special cases.
"""

from rag.models import RetrievedChunk
from rag.reranking import BaseReranker
from rag.retrieval.base import BaseRetriever

# How many candidates to fetch before reranking: more than top_k, so a chunk the
# retriever ranked e.g. 12th can still be promoted, but few enough for the slower
# reranker to read each one.
DEFAULT_CANDIDATES = 20


class RerankedRetriever(BaseRetriever):
    """Fetches candidates with one retriever, then keeps the reranker's top_k."""

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        candidates: int = DEFAULT_CANDIDATES,
    ):
        """Rerank the first `candidates` results of `retriever` (e.g. a HybridRetriever)."""
        self.retriever = retriever
        self.reranker = reranker
        self.candidates = candidates

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top_k chunks by reranker score (see BaseReranker.rerank)."""
        candidates = self.retriever.retrieve(query, self.candidates)
        return self.reranker.rerank(query, candidates, top_k)
