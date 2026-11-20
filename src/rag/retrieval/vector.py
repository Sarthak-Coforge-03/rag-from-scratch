"""VectorRetriever: semantic search, matching the question's meaning to chunk embeddings."""

import logging

from rag.documents import DocumentRepository
from rag.embedding import BaseEmbedder
from rag.models import RetrievedChunk
from rag.retrieval.base import BaseRetriever
from rag.vectorstore import BaseVectorStore

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """Embeds the question, finds the nearest chunk ids, then loads those chunks from SQL.

    The embedder must be the model the vector store was filled with, otherwise
    question and chunk vectors aren't comparable.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        repository: DocumentRepository,
    ):
        """Search `vector_store` with `embedder`, reading chunk text from `repository`."""
        self.embedder = embedder
        self.vector_store = vector_store
        self.repository = repository

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to top_k chunks nearest to the query, closest first.

        score = 1 - cosine distance (cosine similarity): 1 means same direction.
        """
        query_embedding = self.embedder.embed_query(query)
        matches = self.vector_store.query(query_embedding, top_k)

        # The vector store only knows ids; text and metadata come from SQL.
        # get_chunks keeps the order of the ids it's given, i.e. closest first.
        matched_ids = [match.chunk_id for match in matches]
        chunks_by_id = {
            chunk.chunk_id: chunk for chunk in self.repository.get_chunks(matched_ids)
        }

        results = []
        for match in matches:
            if match.chunk_id not in chunks_by_id:
                # The vector store has an id SQL no longer has: the chunk step ran
                # after the last embed. Skip it so results only contain real chunks.
                logger.warning(
                    "Chunk %s is in the vector store but not in SQL; vectors are out "
                    "of sync, run 03_embed.py --sync",
                    match.chunk_id,
                )
                continue
            result = RetrievedChunk(
                chunk=chunks_by_id[match.chunk_id], score=1.0 - match.distance
            )
            results.append(result)
        return results
