from datetime import date
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field


class User(BaseModel):
    """Class to define the current user of the app

    Args:
        BaseModel (User): pydantic basemodel to enforce limits
    """

    id: Annotated[int, Field(gt=0)]
    name: Annotated[str, Field(min_length=1)]


class Document(BaseModel):
    """One markdown file loaded into memory: optional front-matter metadata plus its body.

    Metadata comes from an optional YAML front-matter block, so it may be missing.
    When present, policy ID, version and effective date let later steps tell two
    versions of the same document apart and cite where an answer came from.
    """

    doc_id: str  # file name without extension, e.g. "pto_policy"
    title: str  # front-matter title, else first "# " heading, else doc_id
    policy_id: str = ""  # e.g. "PTO-001"; "" when the file doesn't say
    version: str = ""  # e.g. "4.0"; "" when the file doesn't say
    effective_date: date | None = None  # None when the file doesn't say
    source_path: Path
    text: str  # the file body without front matter; chunking decides how to split it
    # SHA-256 of the whole raw file (front matter included). Ingestion compares it with
    # the stored hash to skip files that haven't changed since the last run.
    content_hash: str


class ChunkMetadata(BaseModel):
    """Where a chunk came from, stored alongside it in Chroma.

    Kept flat with no optional fields because Chroma metadata only accepts
    str/int/float/bool values (no None). Unknown values are stored as "".
    Use model_dump(mode="json") to turn the path into a string before storing.
    """

    doc_id: str
    title: str
    policy_id: str
    version: str
    effective_date: str  # ISO date like "2026-01-01", or "" when the document has none
    source_path: Path
    chunk_index: int  # 0, 1, 2… position of the chunk within its document
    start_index: int  # character offsets of the chunk inside Document.text
    end_index: int
    token_count: int


class Chunk(BaseModel):
    """A piece of a document to embed and store, together with its metadata.

    chunk_id combines doc_id, the start of the document's content hash and
    chunk_index: identical on every re-ingest of the same content, and new for
    every edit, so an edited document never reuses an old chunk's id.
    """

    chunk_id: str  # e.g. "pto_policy::3f9a1c0b7d2e::0"
    text: str
    metadata: ChunkMetadata


class VectorMatch(BaseModel):
    """One hit from a vector search: which chunk matched and how far it was from the query.

    distance is cosine distance: 0 means the same direction, larger means less similar.
    The chunk's text and metadata are looked up separately in the document database.
    """

    chunk_id: str
    distance: float


class IndexSummary(BaseModel):
    """What one index folder holds, as reported by IndexManager.list_indexes()."""

    name: str  # folder name under the index root, e.g. "token-800"
    chunker_fingerprint: str  # settings that built its SQL data
    embedder_fingerprints: list[str]  # one per vector folder, sorted
    is_chunked: bool  # True once the chunk step has created sql/docs.db
