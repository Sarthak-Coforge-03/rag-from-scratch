"""SearchableIndex: one index loaded and verified, ready to build retrievers from.

IndexManager answers every question about folders (is it chunked? which models are
embedded? where is each folder?). This module turns those answers into loaded objects:
the SQL database, the embedding model and the vector store. It loads and checks only;
searching is done by the retrievers it makes.

    index = load_searchable_index("structural-800")
    results = index.make_retriever("hybrid").retrieve("What is SEC-4 about?", top_k=3)
    index.close()
"""

from pathlib import Path

from rag.config import settings
from rag.documents import DocumentRepository
from rag.embedding import BaseEmbedder, create_embedder
from rag.indexes.fingerprints import embedder_fingerprint
from rag.indexes.manager import IndexManager
from rag.models import EmbedderConfig
from rag.retrieval import BaseRetriever, create_retriever
from rag.vectorstore import BaseVectorStore, create_vector_store


class SearchableIndex:
    """Everything needed to search one index, loaded once.

    Loading the embedding model takes seconds, so build it once here and make as
    many retrievers from it as needed (e.g. vector, keyword and hybrid side by side).
    """

    def __init__(
        self,
        name: str,
        repository: DocumentRepository,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ):
        """Hold the loaded parts of index `name`; use load_searchable_index to create one."""
        self.name = name
        self.repository = repository
        self.embedder = embedder
        self.vector_store = vector_store

    def make_retriever(self, strategy: str) -> BaseRetriever:
        """Build a retriever ("vector", "keyword" or "hybrid") over this index."""
        return create_retriever(
            strategy=strategy,
            repository=self.repository,
            embedder=self.embedder,
            vector_store=self.vector_store,
        )

    def close(self) -> None:
        """Release the database connection."""
        self.repository.close()


def load_searchable_index(
    index_name: str,
    model_name: str | None = None,
    index_root: Path = settings.index_root,
) -> SearchableIndex:
    """Load an index's SQL data, embedding model and vectors, checking they belong together.

    model_name picks the vector folder when the index has several (e.g.
    "all-MiniLM-L6-v2"); leave it out when there is only one. The embedder is built
    from that folder's saved settings, so questions are always embedded by the same
    model as the chunks (EMBED_MODEL is not used here).

    Raises ValueError if the index isn't chunked, the model choice is unclear, or the
    vectors are out of date with SQL. index_root is where the index folder lives
    (INDEX_ROOT by default; the CLI passes a temporary folder for a fresh index).
    """
    manager = IndexManager(index_root)
    layout = manager.open_chunked(index_name)
    embedder_config = pick_embedding_model(
        manager.read_embedder_configs(index_name), model_name
    )

    embedder = create_embedder(
        provider=embedder_config.provider,
        model_name=embedder_config.model_name,
        batch_size=settings.embed_batch_size,
    )
    vector_dir = layout.vector_dir(embedder_fingerprint(embedder))
    vector_store = create_vector_store(
        store=settings.vector_store,
        location=str(vector_dir),
        collection_name=settings.vector_collection,
    )
    repository = DocumentRepository(layout.database_url)
    ensure_vectors_match_sql(repository, vector_store)

    return SearchableIndex(
        name=index_name,
        repository=repository,
        embedder=embedder,
        vector_store=vector_store,
    )


def pick_embedding_model(
    configs: list[EmbedderConfig], model_name: str | None
) -> EmbedderConfig:
    """Choose which vector folder (embedding model) to search; ValueError if unclear.

    No folders -> nothing to search yet. A model name picks that folder. Without a
    name, an index with exactly one folder uses it; with several, the caller must choose.
    """
    available_names = [config.model_name for config in configs]
    if not configs:
        raise ValueError("This index has no vectors yet; run 03_embed.py --create")

    if model_name is not None:
        for config in configs:
            if config.model_name == model_name:
                return config
        raise ValueError(
            f"No vectors for model {model_name!r}; available: {', '.join(available_names)}"
        )

    if len(configs) == 1:
        return configs[0]
    raise ValueError(
        f"This index has vectors for several models ({', '.join(available_names)}); "
        "pass a model name to choose one"
    )


def ensure_vectors_match_sql(
    repository: DocumentRepository, vector_store: BaseVectorStore
) -> None:
    """Raise ValueError unless the vector store holds exactly the chunk ids in SQL.

    They differ when the chunk step ran after the last embed step: new chunks would be
    unsearchable by meaning, and removed chunks could still be returned.
    """
    ids_in_sql = set(repository.get_all_chunk_ids())
    ids_in_store = set(vector_store.list_ids())
    not_embedded = ids_in_sql - ids_in_store
    stale = ids_in_store - ids_in_sql
    if not_embedded or stale:
        raise ValueError(
            f"Vectors are out of date: {len(not_embedded)} chunks not embedded, "
            f"{len(stale)} stale vectors; run 03_embed.py --sync"
        )
