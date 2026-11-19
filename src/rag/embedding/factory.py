"""Pick an embedding provider by name (EMBED_PROVIDER in config.py)."""

from rag.embedding.base import BaseEmbedder
from rag.embedding.sentence_transformer import SentenceTransformerEmbedder

# Provider name (EMBED_PROVIDER) -> embedder class. Add new providers here.
EMBEDDER_CLASSES = {
    "sentence-transformers": SentenceTransformerEmbedder,
}


def create_embedder(provider: str, model_name: str, batch_size: int) -> BaseEmbedder:
    """Build the embedder for a provider name, e.g. settings.embed_provider.

    Raises ValueError listing the valid names if the provider is unknown, so a typo in
    EMBED_PROVIDER fails immediately, before any model is downloaded or loaded.
    """
    if provider not in EMBEDDER_CLASSES:
        valid_names = ", ".join(EMBEDDER_CLASSES)
        raise ValueError(
            f"Unknown embedding provider {provider!r}; choose one of: {valid_names}"
        )
    embedder_class = EMBEDDER_CLASSES[provider]
    return embedder_class(model_name=model_name, batch_size=batch_size)
