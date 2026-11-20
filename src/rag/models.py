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


class ChunkerConfig(BaseModel):
    """The settings an index was created with, saved so an update rebuilds the same chunker.

    Stored as chunker_config.json in the index folder. chunker.txt holds the readable
    fingerprint; this holds the same settings in a form that can be read back.
    """

    strategy: str  # a key of CHUNKER_CLASSES, e.g. "structural"
    chunk_size: int
    chunk_overlap: int


class EmbedderConfig(BaseModel):
    """The model a vector folder was created with, saved so --sync rebuilds the same embedder.

    Stored as embedder_config.json inside the vector folder, next to embedder.txt.
    """

    provider: str  # a key of EMBEDDER_CLASSES, e.g. "sentence-transformers"
    model_name: str  # e.g. "all-MiniLM-L6-v2"


class VectorSyncReport(BaseModel):
    """What one embed-step run did to one vector folder, counted in chunks."""

    embedded: int  # in SQL but not in the folder: embedded and added
    deleted: int  # in the folder but no longer in SQL: removed
    unchanged: int  # in both: left alone


class IngestReport(BaseModel):
    """What one chunk-step run did to each document, by doc_id (each list sorted)."""

    added: list[str] = []  # new files, chunked and saved
    updated: list[str] = []  # content hash changed, re-chunked and replaced
    unchanged: list[str] = []  # same content hash, skipped
    removed: list[str] = []  # stored before but the file is gone, deleted
