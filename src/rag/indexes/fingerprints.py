"""Readable descriptions of the settings that shape an index's chunks and vectors.

A fingerprint is recorded inside an index folder so a later run can tell whether
the folder was built with the same settings. Fingerprints are plain text, not
hashes, so a mismatch message shows exactly which setting differs.
"""

from rag.chunking import BaseChunker
from rag.embedding import BaseEmbedder


def chunker_fingerprint(chunker: BaseChunker) -> str:
    """Describe the chunker settings that decide the chunk text.

    Example: "TokenWindowChunker(size=800, overlap=150)". The class name identifies
    the strategy (each strategy is its own class).
    """
    chunker_name = type(chunker).__name__
    return f"{chunker_name}(size={chunker.chunk_size}, overlap={chunker.chunk_overlap})"


def embedder_fingerprint(embedder: BaseEmbedder) -> str:
    """Describe the embedding provider and model that decide the vectors.

    Example: "SentenceTransformerEmbedder(all-MiniLM-L6-v2)". batch_size is left out:
    it changes how fast vectors are made, not the vectors themselves.
    """
    embedder_name = type(embedder).__name__
    return f"{embedder_name}({embedder.model_name})"
