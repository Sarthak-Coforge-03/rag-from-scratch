"""sync_vectors: the embed step. Makes one vector folder hold exactly the chunks in SQL.

Chunk ids contain the document's content hash, so comparing ids is enough:
an edited document has all-new ids (embedded) and its old ids disappear from SQL
(deleted); untouched documents keep their ids (skipped, not embedded again).
"""

import logging

from rag.documents import DocumentRepository
from rag.embedding import BaseEmbedder
from rag.models import VectorSyncReport
from rag.vectorstore import BaseVectorStore

logger = logging.getLogger(__name__)


def sync_vectors(
    repository: DocumentRepository,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
) -> VectorSyncReport:
    """Embed chunks missing from the vector store and delete ids no longer in SQL.

    Missing chunks are loaded and embedded embedder.batch_size at a time, so only
    one batch of chunk text is in memory at once.
    """
    ids_in_sql = set(repository.get_all_chunk_ids())
    ids_in_store = set(vector_store.list_ids())
    missing_ids = sorted(ids_in_sql - ids_in_store)
    extra_ids = sorted(ids_in_store - ids_in_sql)

    vector_store.delete(extra_ids)

    batch_size = embedder.batch_size
    for batch_start in range(0, len(missing_ids), batch_size):
        batch_ids = missing_ids[batch_start : batch_start + batch_size]
        chunks = repository.get_chunks(batch_ids)
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_documents(texts)
        vector_store.upsert(chunk_ids=batch_ids, embeddings=embeddings)
        logger.info(
            "Embedded %d / %d chunks", batch_start + len(batch_ids), len(missing_ids)
        )

    return VectorSyncReport(
        embedded=len(missing_ids),
        deleted=len(extra_ids),
        unchanged=len(ids_in_sql & ids_in_store),
    )
