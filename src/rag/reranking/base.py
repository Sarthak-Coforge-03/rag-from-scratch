"""BaseReranker: the interface every reranking model implements.

A retriever scores the question and each chunk separately (embeddings, word counts).
A reranker reads the question and a chunk together and judges how well that chunk
answers it: more accurate, but too slow for the whole index, so it only reorders the
few candidates a retriever already found.

The base class does the shared work (score, sort, keep top_k); a subclass only says
how to score.
"""

from abc import ABC, abstractmethod

from rag.models import RetrievedChunk


class BaseReranker(ABC):
    """Reorders retrieved chunks by how relevant each one is to the question."""

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text, in the same order; higher = more relevant."""

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the top_k candidates, best first, each scored by this reranker.

        The retriever's scores are replaced: they were on a different scale and are
        no longer what decides the order.
        """
        if not candidates:
            return []

        # 1. Score every candidate's text against the question (one score per text,
        #    in the same order as the candidates).
        texts = [candidate.chunk.text for candidate in candidates]
        reranker_scores = self.score(query, texts)

        # 2. Pair each chunk with its new score, dropping the retriever's score.
        rescored_candidates = []
        for position in range(len(candidates)):
            rescored = RetrievedChunk(
                chunk=candidates[position].chunk,
                score=float(reranker_scores[position]),
            )
            rescored_candidates.append(rescored)

        # 3. Best new score first, then keep only top_k.
        best_first = sorted(
            rescored_candidates, key=lambda result: result.score, reverse=True
        )
        return best_first[:top_k]
