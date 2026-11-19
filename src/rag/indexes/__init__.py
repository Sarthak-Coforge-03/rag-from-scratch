"""Index folders: one chunked SQL database plus one vector store per embedding model.

Layout — one class per file:
- fingerprints.py  chunker_fingerprint() / embedder_fingerprint(): readable descriptions of
                   the settings that shape chunks and vectors.
- layout.py        IndexLayout: every path inside one index folder (no disk access).
"""

from rag.indexes.fingerprints import chunker_fingerprint, embedder_fingerprint
from rag.indexes.layout import IndexLayout

__all__ = ["IndexLayout", "chunker_fingerprint", "embedder_fingerprint"]
