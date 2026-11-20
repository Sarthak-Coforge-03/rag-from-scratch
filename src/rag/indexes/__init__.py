"""Index folders: one chunked SQL database plus one vector store per embedding model.

Layout — one class per file:
- builder.py       build_index(): chunk and embed a new index from CHUNK_* / EMBED_* in one call.
- fingerprints.py  chunker_fingerprint() / embedder_fingerprint(): readable descriptions of
                   the settings that shape chunks and vectors.
- layout.py        IndexLayout: every path inside one index folder (no disk access).
- manager.py       IndexManager: the only code that creates, inspects or deletes index
                   folders (by name under INDEX_ROOT); see its docstring for the rules.
- searchable_index.py  load_searchable_index() -> SearchableIndex: an index's SQL data,
                   embedding model and vectors loaded and checked, ready for retrievers.
"""

from rag.indexes.builder import build_index
from rag.indexes.fingerprints import chunker_fingerprint, embedder_fingerprint
from rag.indexes.layout import IndexLayout
from rag.indexes.manager import IndexManager
from rag.indexes.searchable_index import SearchableIndex, load_searchable_index

__all__ = [
    "IndexLayout",
    "IndexManager",
    "SearchableIndex",
    "build_index",
    "chunker_fingerprint",
    "embedder_fingerprint",
    "load_searchable_index",
]
