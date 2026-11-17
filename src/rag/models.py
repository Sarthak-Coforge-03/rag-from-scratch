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
