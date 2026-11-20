"""HybridRetriever: merges several retrievers' rankings with reciprocal rank fusion (RRF).

Each retriever returns a ranked list. Every chunk earns 1 / (RRF_K + rank) from each
list it appears in (rank 1 = best), and the points are added up by chunk id:

    vector : A(1) B(2) C(3)        A = 1/61 + 1/62   <- found by both: ranked first
    keyword: D(1) A(2)             D = 1/61, B = 1/62, C = 1/63
    fused  : A, D, B, C

Only ranks are used, so cosine similarities and BM25 scores never need to share a scale.
"""

from rag.models import Chunk, RetrievedChunk
from rag.retrieval.base import BaseRetriever

# The standard constant from the RRF paper (Cormack et al., 2009). It keeps rank 1 from
# dominating (1/61 vs 1/62 is a small gap), so a chunk that several retrievers agree on
# beats a chunk that is top of only one list.
RRF_K = 60

# How many results to take from each retriever before fusing. More than top_k, so a
# chunk ranked e.g. 8th by vector and 2nd by keyword can still win.
CANDIDATES_PER_RETRIEVER = 20


class HybridRetriever(BaseRetriever):
    """Runs every retriever on the question and fuses their rankings with RRF."""

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        candidates_per_retriever: int = CANDIDATES_PER_RETRIEVER,
    ):
        """Fuse the results of `retrievers` (e.g. vector + keyword)."""
        self.retrievers = retrievers
        self.candidates_per_retriever = candidates_per_retriever

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to top_k chunks, highest fused score first.

        score = the chunk's summed RRF points: higher means ranked well by more retrievers.
        """
        fused_scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}

        for retriever in self.retrievers:
            results = retriever.retrieve(query, self.candidates_per_retriever)
            # enumerate from 1 so the best result has rank 1, as in the RRF formula.
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk.chunk_id
                points = 1 / (RRF_K + rank)
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + points
                chunks_by_id[chunk_id] = result.chunk

        best_first = sorted(
            fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True
        )
        return [
            RetrievedChunk(chunk=chunks_by_id[chunk_id], score=fused_scores[chunk_id])
            for chunk_id in best_first[:top_k]
        ]
