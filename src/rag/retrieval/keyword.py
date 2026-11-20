"""KeywordRetriever: exact-word search over chunk text, ranked with BM25 (rank_bm25)."""

import re

from rank_bm25 import BM25Okapi

from rag.documents import DocumentRepository
from rag.models import RetrievedChunk
from rag.retrieval.base import BaseRetriever

# A run of letters or digits (any language). "$2,500" -> "2", "500"; "EXP-4" -> "exp", "4".
WORD = re.compile(r"[^\W_]+")


def split_into_words(text: str) -> list[str]:
    """Lowercase the text and return its words, the units BM25 counts and matches."""
    return WORD.findall(text.lower())


class KeywordRetriever(BaseRetriever):
    """Finds chunks containing the question's words, using only the document database.

    Catches what embeddings blur: exact amounts ($2,500), codes (EXP-4), acronyms (VPN).

    The BM25 index is built in memory from every chunk in SQL when the retriever is
    created, so it is a snapshot: after the chunk step, create a new retriever. SQL
    stays the only stored copy of the text.
    """

    def __init__(self, repository: DocumentRepository):
        """Load every chunk from `repository` and build the BM25 index over their words."""
        self.chunks = repository.get_chunks(repository.get_all_chunk_ids())
        self.words_per_chunk = [split_into_words(chunk.text) for chunk in self.chunks]
        # BM25Okapi can't be built from zero chunks; an empty database just finds nothing.
        self.bm25 = BM25Okapi(self.words_per_chunk) if self.chunks else None

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to top_k chunks containing the query's words, best BM25 score first.

        Chunks with none of the query's words are left out. The score alone can't tell
        that: BM25 gives a word that appears in half the chunks a weight of 0.
        """
        query_words = split_into_words(query)
        if self.bm25 is None or not query_words:
            return []

        # One score per chunk, in the same order as self.chunks.
        scores = self.bm25.get_scores(query_words)

        matching_positions = []
        for position, chunk_words in enumerate(self.words_per_chunk):
            if any(word in chunk_words for word in query_words):
                matching_positions.append(position)
        best_first = sorted(
            matching_positions, key=lambda position: scores[position], reverse=True
        )

        return [
            RetrievedChunk(chunk=self.chunks[position], score=float(scores[position]))
            for position in best_first[:top_k]
        ]
