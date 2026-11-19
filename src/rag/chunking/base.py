"""BaseChunker: the shared part of every chunking strategy.

Subclasses only decide how raw text is split (split_text). Everything else —
building Chunk/ChunkMetadata and chunk ids — lives here, so every strategy produces
Chunks the same way. No assumptions are made about markdown structure: chunks carry
document-level metadata plus their character offsets.
"""

from abc import ABC, abstractmethod

import chonkie

from rag.models import Chunk, ChunkMetadata, Document


class BaseChunker(ABC):
    """Turns Documents into Chunks; subclasses only decide how the text is split.

    Everything that does not depend on the splitting strategy (metadata and chunk
    ids) lives here, so every strategy produces Chunks the same way.
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

        # Chroma metadata can't hold None, so a missing date is stored as "".
        if document.effective_date is None:
            effective_date = ""
        else:
            effective_date = document.effective_date.isoformat()

        chunks = []
        for index, piece in enumerate(pieces):
            metadata = ChunkMetadata(
                doc_id=document.doc_id,
                title=document.title,
                policy_id=document.policy_id,
                version=document.version,
                effective_date=effective_date,
                source_path=document.source_path,
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
