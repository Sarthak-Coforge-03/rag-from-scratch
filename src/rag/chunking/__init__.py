"""Split Documents into Chunks (with metadata).

Layout — one class per file:
- base.py             BaseChunker: shared logic (metadata, chunk ids); subclasses only split text.
- token_window.py     TokenWindowChunker, strategy "token": fixed character windows with overlap.
- recursive_split.py  RecursiveSplitChunker, strategy "recursive": paragraph -> sentence ->
                      word splits merged up to chunk_size; never cuts a word.
- structural.py       StructuralChunker, strategy "structural": splits before markdown
                      headings first, then like "recursive"; chunks start with their heading.
- factory.py          CHUNKER_CLASSES and create_chunker(), which picks a strategy by name.

To add a strategy: write a BaseChunker subclass in its own file and register it in
CHUNKER_CLASSES (factory.py).
"""

from rag.chunking.base import BaseChunker
from rag.chunking.factory import CHUNKER_CLASSES, create_chunker
from rag.chunking.recursive_split import RecursiveSplitChunker
from rag.chunking.structural import StructuralChunker
from rag.chunking.token_window import TokenWindowChunker

__all__ = [
    "CHUNKER_CLASSES",
    "BaseChunker",
    "RecursiveSplitChunker",
    "StructuralChunker",
    "TokenWindowChunker",
    "create_chunker",
]
