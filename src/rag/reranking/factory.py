"""Pick a reranker by name (RERANK_PROVIDER in config.py)."""

from rag.reranking.base import BaseReranker
from rag.reranking.cross_encoder import CrossEncoderReranker

# Provider name (RERANK_PROVIDER) -> reranker class. Add new rerankers here.
# Every class takes one argument: model_name.
RERANKER_CLASSES = {
    "cross-encoder": CrossEncoderReranker,
}


def create_reranker(provider: str, model_name: str) -> BaseReranker:
    """Build the reranker for a provider name, e.g. settings.rerank_provider.

    Raises ValueError listing the valid names if the provider is unknown, so a typo
    in RERANK_PROVIDER fails immediately instead of loading a different model.
    """
    if provider not in RERANKER_CLASSES:
        valid_names = ", ".join(RERANKER_CLASSES)
        raise ValueError(f"Unknown reranker {provider!r}; choose one of: {valid_names}")
    reranker_class = RERANKER_CLASSES[provider]
    return reranker_class(model_name=model_name)
