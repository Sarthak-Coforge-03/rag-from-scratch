"""RecursiveSplitChunker: split at natural boundaries (strategy "recursive").

How it works: chonkie's RecursiveChunker first splits the text into paragraphs; any
paragraph still longer than chunk_size is split into sentences, then at pauses
(commas, semicolons...), then into words. Neighbouring pieces are then merged back
together up to chunk_size characters. The result is chunks that never cut a word in
half and usually end at a paragraph or sentence break.

No overlap: chonkie's recursive chunker has no overlap option, so chunk_overlap is
accepted (to keep one constructor for every strategy) but ignored, and a warning is
logged when it is non-zero. Overlap mainly exists to repair sentences cut at a
boundary, and this strategy doesn't cut them.

chunk_size keeps the same meaning and rationale as the "token" strategy (characters,
sized to stay under the embedding model's input limit); see token_window.py.

Named RecursiveSplitChunker so it can't be confused with chonkie.RecursiveChunker.
"""

import logging

import chonkie

from rag.chunking.base import BaseChunker

logger = logging.getLogger(__name__)


class RecursiveSplitChunker(BaseChunker):
    """Paragraph -> sentence -> pause -> word splitting, merged up to chunk_size characters."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Configure chonkie's RecursiveChunker (default rules) to count characters.

        Logs a warning if chunk_overlap is non-zero, because this strategy can't apply it.
        """
        super().__init__(chunk_size, chunk_overlap)
        if chunk_overlap > 0:
            logger.warning(
                "Chunk strategy 'recursive' does not support overlap; "
                "chunk_overlap=%d is ignored",
                chunk_overlap,
            )
        self.chonkie_chunker = chonkie.RecursiveChunker(
            tokenizer="character",
            chunk_size=chunk_size,
        )

    def split_text(self, text: str) -> list[chonkie.Chunk]:
        """Split the text at the most natural boundaries that keep chunks under chunk_size."""
        return self.chonkie_chunker.chunk(text)
