"""Pick a chunking strategy by name (CHUNK_STRATEGY in config.py)."""

from rag.chunking.base import BaseChunker
from rag.chunking.recursive_split import RecursiveSplitChunker
from rag.chunking.token_window import TokenWindowChunker

# Strategy name (CHUNK_STRATEGY) -> chunker class. Add new strategies here.
CHUNKER_CLASSES = {
    "token": TokenWindowChunker,
    "recursive": RecursiveSplitChunker,
}


def create_chunker(strategy: str, chunk_size: int, chunk_overlap: int) -> BaseChunker:
    """Build the chunker for a strategy name, e.g. settings.chunk_strategy.

    Raises ValueError listing the valid names if the strategy is unknown, so a typo in
    CHUNK_STRATEGY fails immediately instead of silently using a different strategy.
    """
    if strategy not in CHUNKER_CLASSES:
        valid_names = ", ".join(CHUNKER_CLASSES)
        raise ValueError(
            f"Unknown chunk strategy {strategy!r}; choose one of: {valid_names}"
        )
    chunker_class = CHUNKER_CLASSES[strategy]
    return chunker_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
