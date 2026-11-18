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
    """One policy file loaded into memory: its header metadata plus the full, untouched text.

    The metadata (policy ID, version, effective date) is kept so later steps can tell
    two versions of the same policy apart and cite where an answer came from.
    """

    doc_id: str  # file name without extension, e.g. "pto_policy"
    title: str  # the "# " heading on the first line
    policy_id: str  # e.g. "PTO-001"
    version: str  # e.g. "4.0"
    effective_date: date
    source_path: Path
    text: str  # the whole file; chunking decides how to split it


class ChunkMetadata(BaseModel):
    """Where a chunk came from, stored alongside it in Chroma.

    Kept flat with no optional fields because Chroma metadata only accepts
    str/int/float/bool values. Use model_dump(mode="json") to turn the date
    and path into strings before storing.
    """

    doc_id: str
    title: str
    policy_id: str
    version: str
    effective_date: date
    source_path: Path
    heading: (
        str  # nearest markdown heading above the chunk, "" if the document has none
    )
    chunk_index: int  # 0, 1, 2… position of the chunk within its document
    start_index: int  # character offsets of the chunk inside Document.text
    end_index: int
    token_count: int


class Chunk(BaseModel):
    """A piece of a document to embed and store, together with its metadata.

    chunk_id combines doc_id, version and chunk_index so it stays unique across
    versions of the same document and is identical on every re-ingest.
    """

    chunk_id: str  # e.g. "pto_policy::v4.0::0"
    text: str
    metadata: ChunkMetadata
