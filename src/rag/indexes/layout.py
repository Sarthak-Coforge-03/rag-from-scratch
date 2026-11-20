"""IndexLayout: where everything lives inside one index folder.

Paths only — this class never reads or writes the disk. The IndexManager decides
when folders are created or removed; everything else asks the layout for paths
instead of building them by hand, so the structure is defined in one place:

    <root>/
      chunker.txt                  fingerprint of the chunker that built sql/
      chunker_config.json          the same settings as JSON, to rebuild that chunker
      sql/docs.db                  documents and chunk text
      vectors/<embedder>/          one folder per embedding model
        embedder.txt               fingerprint of the model whose vectors live here
"""

import re
from pathlib import Path

from pydantic import BaseModel

CHUNKER_FILE_NAME = "chunker.txt"
CHUNKER_CONFIG_FILE_NAME = "chunker_config.json"
EMBEDDER_FILE_NAME = "embedder.txt"
SQL_FOLDER_NAME = "sql"
DATABASE_FILE_NAME = "docs.db"
VECTORS_FOLDER_NAME = "vectors"

# Characters allowed in a folder name made from a fingerprint. Anything else
# (e.g. "/" in "BAAI/bge-small", or the parentheses) becomes "_", so a model name
# can never create nested folders or awkward names.
UNSAFE_FOLDER_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]")


class IndexLayout(BaseModel):
    """The fixed folder structure of one index, derived entirely from its root."""

    root: Path

    @property
    def chunker_file(self) -> Path:
        """File recording which chunker built this index's SQL data."""
        return self.root / CHUNKER_FILE_NAME

    @property
    def chunker_config_file(self) -> Path:
        """File with the chunker settings as JSON, read back when the index is updated."""
        return self.root / CHUNKER_CONFIG_FILE_NAME

    @property
    def sql_dir(self) -> Path:
        """Folder holding the SQLite database."""
        return self.root / SQL_FOLDER_NAME

    @property
    def database_file(self) -> Path:
        """The SQLite database file with documents and chunk text."""
        return self.sql_dir / DATABASE_FILE_NAME

    @property
    def database_url(self) -> str:
        """SQLAlchemy URL for the database file, ready for DocumentRepository."""
        return f"sqlite:///{self.database_file}"

    @property
    def vectors_root(self) -> Path:
        """Folder holding one vector-store folder per embedding model."""
        return self.root / VECTORS_FOLDER_NAME

    def vector_dir(self, embedder_fingerprint: str) -> Path:
        """Folder for one embedding model's vectors, named safely from its fingerprint."""
        folder_name = UNSAFE_FOLDER_CHARACTERS.sub("_", embedder_fingerprint)
        return self.vectors_root / folder_name

    def embedder_file(self, embedder_fingerprint: str) -> Path:
        """File inside a model's vector folder recording that model's full fingerprint."""
        return self.vector_dir(embedder_fingerprint) / EMBEDDER_FILE_NAME
