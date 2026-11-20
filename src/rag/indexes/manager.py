"""IndexManager: the only code that creates, inspects or deletes index folders.

Indexes are addressed by name under one root folder (INDEX_ROOT), e.g.
indexes/token-800. One index = one chunker config; different configs live side by
side under different names.

Rules:
- chunk step, new name                      -> create sql/ and chunker.txt (no vectors)
- chunk step, existing name, same config    -> keep everything
- chunk step, existing name, other config   -> wipe that index (SQL and all vectors),
                                               recreate only sql/ and chunker.txt
- vector folders are created only when asked for (prepare_vector_dir), never rebuilt
  automatically
- a folder without chunker.txt is never wiped: every write operation refuses it
"""

import logging
import re
import shutil
from pathlib import Path

from rag.indexes.layout import EMBEDDER_FILE_NAME, IndexLayout
from rag.models import IndexSummary

logger = logging.getLogger(__name__)

# Index names become folder names, so only plain characters are allowed: no "/" or
# "..", which would let a name (e.g. a CLI argument) point outside the index root.
VALID_INDEX_NAME = re.compile(r"[A-Za-z0-9._-]+")


class IndexManager:
    """Creates, opens, lists and deletes the index folders under one root folder."""

    def __init__(self, root_dir: Path):
        """Manage the index folders inside root_dir (e.g. settings.index_root)."""
        self.root_dir = Path(root_dir)

    def layout(self, name: str) -> IndexLayout:
        """Return the paths of the index called `name`; ValueError for an unsafe name."""
        if not VALID_INDEX_NAME.fullmatch(name) or name in (".", ".."):
            raise ValueError(
                f"Invalid index name {name!r}: use letters, digits, '.', '_' or '-'"
            )
        return IndexLayout(root=self.root_dir / name)

    def prepare_for_chunking(self, name: str, chunker_fp: str) -> IndexLayout:
        """Get an index ready for the chunk step with this chunker config.

        Creates a new index, keeps one built with the same config, or wipes and
        recreates one built with a different config. Refuses a non-empty folder
        that isn't an index, so unrelated files are never deleted.
        """
        layout = self.layout(name)

        if layout.root.exists():
            if not layout.chunker_file.exists():
                folder_is_empty = not any(layout.root.iterdir())
                if not folder_is_empty:
                    raise ValueError(
                        f"{layout.root} is not empty and is not an index "
                        "(no chunker.txt); refusing to use it. Pick another name."
                    )
            else:
                stored_chunker_fp = layout.chunker_file.read_text()
                if stored_chunker_fp == chunker_fp:
                    return layout
                dropped_models = self._embedder_fingerprints(layout)
                logger.warning(
                    "Index %r was chunked with %s; rebuilding it for %s "
                    "(SQL data and %d vector folder(s) removed)",
                    name,
                    stored_chunker_fp,
                    chunker_fp,
                    len(dropped_models),
                )
                shutil.rmtree(layout.root)

        layout.sql_dir.mkdir(parents=True, exist_ok=True)
        layout.chunker_file.write_text(chunker_fp)
        return layout

    def open_chunked(self, name: str) -> IndexLayout:
        """Return an index that the chunk step has already filled; never writes.

        Used by the embed step and retrieval. Raises ValueError if the index is
        missing or hasn't been chunked yet.
        """
        layout = self.layout(name)
        if not layout.chunker_file.exists() or not layout.database_file.exists():
            raise ValueError(
                f"Index {name!r} has no chunked data; run the chunk step first"
            )
        return layout

    def prepare_vector_dir(self, name: str, embedder_fp: str) -> Path:
        """Create (or reuse) the vector folder for one embedding model in a chunked index.

        This is the only place vector folders are created. Raises ValueError if the
        folder already belongs to a different model (two model names that become
        the same safe folder name).
        """
        layout = self.open_chunked(name)
        embedder_file = layout.embedder_file(embedder_fp)

        if embedder_file.exists():
            stored_embedder_fp = embedder_file.read_text()
            if stored_embedder_fp != embedder_fp:
                raise ValueError(
                    f"{embedder_file.parent} holds vectors from {stored_embedder_fp}, "
                    f"not {embedder_fp}"
                )

        embedder_file.parent.mkdir(parents=True, exist_ok=True)
        embedder_file.write_text(embedder_fp)
        return embedder_file.parent

    def list_indexes(self) -> list[IndexSummary]:
        """Describe every index under the root, sorted by name (folders that aren't indexes are skipped)."""
        if not self.root_dir.exists():
            return []

        summaries = []
        for folder in sorted(self.root_dir.iterdir()):
            layout = IndexLayout(root=folder)
            if not folder.is_dir() or not layout.chunker_file.exists():
                continue
            summary = IndexSummary(
                name=folder.name,
                chunker_fingerprint=layout.chunker_file.read_text(),
                embedder_fingerprints=self._embedder_fingerprints(layout),
                is_chunked=layout.database_file.exists(),
            )
            summaries.append(summary)
        return summaries

    def delete_index(self, name: str) -> None:
        """Delete an index folder with all its SQL and vector data.

        Raises ValueError if the folder doesn't exist or isn't an index.
        """
        layout = self.layout(name)
        if not layout.chunker_file.exists():
            raise ValueError(
                f"{layout.root} is not an index (no chunker.txt); not deleting it"
            )
        shutil.rmtree(layout.root)
        logger.info("Deleted index %r", name)

    def _embedder_fingerprints(self, layout: IndexLayout) -> list[str]:
        """Fingerprints of the models that have a vector folder in this index, sorted."""
        if not layout.vectors_root.exists():
            return []
        fingerprints = []
        for vector_folder in sorted(layout.vectors_root.iterdir()):
            embedder_file = vector_folder / EMBEDDER_FILE_NAME
            if embedder_file.exists():
                fingerprints.append(embedder_file.read_text())
        return fingerprints
