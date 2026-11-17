"""Split Documents into Chunks (with metadata) using chonkie's TokenChunker.

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

from chonkie import TokenChunker

from rag.models import Chunk, ChunkMetadata, Document

# Any markdown heading line ("# Title" ... "###### Title"); captures the heading text.
HEADING_LINE = re.compile(r"^#{1,6}\s+(?P<heading>.+?)\s*$", re.MULTILINE)


def create_chunker(chunk_size: int, chunk_overlap: int) -> TokenChunker:
    """Build a character-based chonkie chunker with the given size and overlap."""
    return TokenChunker(
        tokenizer="character",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


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


def chunk_document(document: Document, chunker: TokenChunker) -> list[Chunk]:
    """Split one Document into Chunks, copying its metadata onto every chunk.

    chunk_id is "<doc_id>::v<version>::<index>" so it is unique across versions
    of the same document and identical every time the document is re-chunked.
    """
    chonkie_chunks = chunker.chunk(document.text)

    chunks = []
    for index, chonkie_chunk in enumerate(chonkie_chunks):
        metadata = ChunkMetadata(
            doc_id=document.doc_id,
            title=document.title,
            policy_id=document.policy_id,
            version=document.version,
            effective_date=document.effective_date,
            source_path=document.source_path,
            heading=find_heading(
                document.text, chonkie_chunk.start_index, chonkie_chunk.end_index
            ),
            chunk_index=index,
            start_index=chonkie_chunk.start_index,
            end_index=chonkie_chunk.end_index,
            token_count=chonkie_chunk.token_count,
        )
        chunk = Chunk(
            chunk_id=f"{document.doc_id}::v{document.version}::{index}",
            text=chonkie_chunk.text,
            metadata=metadata,
        )
        chunks.append(chunk)
    return chunks


def chunk_documents(documents: list[Document], chunker: TokenChunker) -> list[Chunk]:
    """Chunk every Document and return all chunks in one list, in document order."""
    all_chunks = []
    for document in documents:
        all_chunks.extend(chunk_document(document, chunker))
    return all_chunks
