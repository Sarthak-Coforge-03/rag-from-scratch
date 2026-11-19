"""Split Documents into Chunks (with metadata).

Layout — one class per file:
- base.py          BaseChunker: shared logic (metadata, chunk ids); subclasses only split text.
- token_window.py  TokenWindowChunker, strategy "token": fixed character windows with overlap.
- factory.py       CHUNKER_CLASSES and create_chunker(), which picks a strategy by name.

To add a strategy: write a BaseChunker subclass in its own file and register it in
CHUNKER_CLASSES (factory.py).
"""

from rag.chunking.base import BaseChunker
from rag.chunking.factory import CHUNKER_CLASSES, create_chunker
from rag.chunking.token_window import TokenWindowChunker

__all__ = ["CHUNKER_CLASSES", "BaseChunker", "TokenWindowChunker", "create_chunker"]
