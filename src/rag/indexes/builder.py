"""build_index: create a complete index (chunked + embedded) in one call.

Does what scripts/02_chunk.py and then scripts/03_embed.py --create do, using the
CHUNK_* and EMBED_* settings. The CLI uses it to build a throwaway index in a
temporary folder, so a fresh run needs no prepared index.
"""

from pathlib import Path

from rag.chunking import create_chunker
from rag.config import settings
from rag.documents import DocumentRepository
from rag.embed import sync_vectors
from rag.embedding import create_embedder
from rag.indexes.fingerprints import chunker_fingerprint, embedder_fingerprint
from rag.indexes.manager import IndexManager
from rag.ingest import Ingestor
from rag.models import ChunkerConfig, EmbedderConfig
from rag.vectorstore import create_vector_store


def build_index(index_root: Path, docs_folder: Path) -> str:
    """Chunk and embed every document in docs_folder into a new index under index_root.

    Returns the new index's name (e.g. "structural-800"), ready for
    load_searchable_index(name, index_root=index_root).
    """
    manager = IndexManager(index_root)

    # 1. Create the index folder for the CHUNK_* settings.
    chunker_config = ChunkerConfig(
        strategy=settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunker = create_chunker(
        strategy=chunker_config.strategy,
        chunk_size=chunker_config.chunk_size,
        chunk_overlap=chunker_config.chunk_overlap,
    )
    index_name = manager.next_free_name(
        f"{chunker_config.strategy}-{chunker_config.chunk_size}"
    )
    layout = manager.create_index(
        index_name, chunker_fingerprint(chunker), chunker_config
    )

    # 2. Chunk every document into the index's SQL database.
    repository = DocumentRepository(layout.database_url)
    ingestor = Ingestor(repository=repository, chunker=chunker)
    ingestor.ingest_folder(docs_folder, settings.max_doc_bytes)

    # 3. Embed every chunk into a vector folder for EMBED_MODEL.
    embedder_config = EmbedderConfig(
        provider=settings.embed_provider, model_name=settings.embed_model
    )
    embedder = create_embedder(
        provider=embedder_config.provider,
        model_name=embedder_config.model_name,
        batch_size=settings.embed_batch_size,
    )
    vector_dir = manager.create_vector_dir(
        index_name, embedder_fingerprint(embedder), embedder_config
    )
    vector_store = create_vector_store(
        store=settings.vector_store,
        location=str(vector_dir),
        collection_name=settings.vector_collection,
    )
    sync_vectors(repository, embedder, vector_store)

    repository.close()
    return index_name
