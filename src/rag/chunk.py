"""Split Documents into Chunks (with metadata).

Structure
---------
- BaseChunker holds everything shared: building Chunk/ChunkMetadata, chunk ids and
  headings. Subclasses only decide how raw text is split (split_text).
- TokenWindowChunker is the one strategy today: fixed-size character windows with
  overlap, using chonkie's TokenChunker.
- create_chunker() picks the strategy by name (CHUNK_STRATEGY in config.py). To add a
  strategy, write a BaseChunker subclass and add it to CHUNKER_CLASSES.

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

import re
from abc import ABC, abstractmethod

import chonkie

from rag.models import Chunk, ChunkMetadata, Document

# Any markdown heading line ("# Title" ... "###### Title"); captures the heading text.
HEADING_LINE = re.compile(r"^#{1,6}\s+(?P<heading>.+?)\s*$", re.MULTILINE)


def find_heading(text: str, start: int, end: int) -> str:
    """Return the heading that best describes text[start:end].

    If a heading line begins inside the chunk, the first such heading is used,
    because the chunk is mostly about that section. Otherwise the nearest heading
    above the chunk is used. Returns "" when the text has no heading before `end`.
    """
    heading_above_chunk = ""
    for match in HEADING_LINE.finditer(text):
        heading_start = match.start()
        if heading_start >= end:
            break
        if heading_start >= start:
            return match.group("heading")
        heading_above_chunk = match.group("heading")
    return heading_above_chunk


class BaseChunker(ABC):
    """Turns Documents into Chunks; subclasses only decide how the text is split.

    Everything that does not depend on the splitting strategy (metadata, chunk ids,
    headings) lives here, so every strategy produces Chunks the same way.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Store the size limits every strategy is configured with."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> list[chonkie.Chunk]:
        """Split raw text into pieces, each with text, start/end offsets and a token count."""

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split one Document into Chunks, copying its metadata onto every chunk.

        chunk_id is "<doc_id>::v<version>::<index>" so it is unique across versions
        of the same document and identical every time the document is re-chunked.
        """
        pieces = self.split_text(document.text)

        chunks = []
        for index, piece in enumerate(pieces):
            metadata = ChunkMetadata(
                doc_id=document.doc_id,
                title=document.title,
                policy_id=document.policy_id,
                version=document.version,
                effective_date=document.effective_date,
                source_path=document.source_path,
                heading=find_heading(document.text, piece.start_index, piece.end_index),
                chunk_index=index,
                start_index=piece.start_index,
                end_index=piece.end_index,
                token_count=piece.token_count,
            )
            chunk = Chunk(
                chunk_id=f"{document.doc_id}::v{document.version}::{index}",
                text=piece.text,
                metadata=metadata,
            )
            chunks.append(chunk)
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk every Document and return all chunks in one list, in document order."""
        all_chunks = []
        for document in documents:
            all_chunks.extend(self.chunk_document(document))
        return all_chunks


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


# Strategy name (CHUNK_STRATEGY) -> chunker class. Add new strategies here.
CHUNKER_CLASSES = {
    "token": TokenWindowChunker,
}


def create_chunker(strategy: str, chunk_size: int, chunk_overlap: int) -> BaseChunker:
    """Build the chunker for a strategy name, e.g. settings.chunk_strategy.

    Raises ValueError listing the valid names if the strategy is unknown, so a typo in
    CHUNK_STRATEGY fails immediately instead of silently using a different strategy.
    """
    if strategy not in CHUNKER_CLASSES:
        valid_names = ", ".join(CHUNKER_CLASSES)
        raise ValueError(f"Unknown chunk strategy {strategy!r}; choose one of: {valid_names}")
    chunker_class = CHUNKER_CLASSES[strategy]
    return chunker_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
