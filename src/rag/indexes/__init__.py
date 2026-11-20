"""Index folders: one chunked SQL database plus one vector store per embedding model.

Layout — one class per file:
- fingerprints.py  chunker_fingerprint() / embedder_fingerprint(): readable descriptions of
                   the settings that shape chunks and vectors.
- layout.py        IndexLayout: every path inside one index folder (no disk access).
- manager.py       IndexManager: the only code that creates, inspects or deletes index
                   folders (by name under INDEX_ROOT); see its docstring for the rules.
"""

from rag.indexes.fingerprints import chunker_fingerprint, embedder_fingerprint
from rag.indexes.layout import IndexLayout
from rag.indexes.manager import IndexManager

__all__ = ["IndexLayout", "IndexManager", "chunker_fingerprint", "embedder_fingerprint"]
