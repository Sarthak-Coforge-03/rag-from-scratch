"""TokenWindowChunker: fixed-size character windows with overlap (strategy "token").

Chunk size and overlap rationale
--------------------------------
- Chunks are measured in characters (chonkie's "character" tokenizer).
- chunk_size = 800 characters is roughly 130-180 words, which stays under the
  256 word-piece input limit of the all-MiniLM-L6-v2 embedding model. Longer
  chunks would be silently truncated by the embedder and their tail never searched.
- 800 characters is also about one policy section, so each vector holds one
  cluster of related facts (e.g. only meal limits), which keeps retrieval precise.
- chunk_overlap = 150 characters (~19%) repeats the end of each chunk at the start
  of the next, so a sentence cut at a boundary is still readable in one chunk.

Both values come from config.py (CHUNK_SIZE / CHUNK_OVERLAP) and are passed in as
arguments, so tests and experiments can try other sizes.
"""

import chonkie

from rag.chunking.base import BaseChunker


class TokenWindowChunker(BaseChunker):
    """Fixed-size character windows with overlap, using chonkie's TokenChunker.

    Simple and predictable; the trade-off is that a window can start or end mid-word.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Configure chonkie's TokenChunker to count characters."""
        super().__init__(chunk_size, chunk_overlap)
        self.chonkie_chunker = chonkie.TokenChunker(
            tokenizer="character",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_text(self, text: str) -> list[chonkie.Chunk]:
        """Cut the text into overlapping character windows."""
        return self.chonkie_chunker.chunk(text)
