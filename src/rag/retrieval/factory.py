"""Pick a retriever by name (RETRIEVER in config.py)."""

from rag.documents import DocumentRepository
from rag.embedding import BaseEmbedder
from rag.retrieval.base import BaseRetriever
from rag.retrieval.vector import VectorRetriever
from rag.vectorstore import BaseVectorStore

# Strategy names accepted by create_retriever (RETRIEVER). Add new strategies here
# and build them in create_retriever below.
RETRIEVER_STRATEGIES = ("vector",)


def create_retriever(
    strategy: str,
    repository: DocumentRepository,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
) -> BaseRetriever:
    """Build the retriever for a strategy name, e.g. settings.retriever_strategy.

    Every strategy is given all the search parts and uses the ones it needs, so
    callers build retrievers the same way whichever strategy is chosen.

    Raises ValueError listing the valid names if the strategy is unknown, so a typo
    in RETRIEVER fails immediately instead of silently searching differently.
    """
    if strategy == "vector":
        return VectorRetriever(
            embedder=embedder, vector_store=vector_store, repository=repository
        )
    valid_names = ", ".join(RETRIEVER_STRATEGIES)
    raise ValueError(f"Unknown retriever {strategy!r}; choose one of: {valid_names}")
