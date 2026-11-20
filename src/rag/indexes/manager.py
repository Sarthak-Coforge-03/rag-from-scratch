"""IndexManager: the only code that creates index folders.

Indexes are addressed by name under one root folder (INDEX_ROOT), e.g.
indexes/structural-800. One index = one chunker config, fixed when it is created:

- create: always a brand-new folder (next_free_name picks structural-800, then
  structural-800_1, ...); writes sql/, chunker.txt and chunker_config.json
- update: the chunk step rebuilds the chunker from chunker_config.json, so an
  index's settings can never change and nothing is ever wiped
- vector folders are created only when asked for (prepare_vector_dir)
"""

import re
from pathlib import Path

from rag.indexes.layout import IndexLayout
from rag.models import ChunkerConfig

# Index names become folder names, so only plain characters are allowed: no "/" or
# "..", which would let a name (e.g. a CLI argument) point outside the index root.
VALID_INDEX_NAME = re.compile(r"[A-Za-z0-9._-]+")


class IndexManager:
    """Creates and opens the index folders under one root folder."""

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

    def next_free_name(self, base_name: str) -> str:
        """Return base_name if no folder has it yet, else base_name_1, base_name_2, ...

        Used when creating an index from env settings, so an existing index is never
        reused or overwritten by accident.
        """
        candidate = base_name
        counter = 0
        while self.layout(candidate).root.exists():
            counter += 1
            candidate = f"{base_name}_{counter}"
        return candidate

    def create_index(
        self, name: str, chunker_fp: str, config: ChunkerConfig
    ) -> IndexLayout:
        """Create a new index folder with its chunker settings; ValueError if it exists.

        Writes sql/ (empty until the chunk step runs), chunker.txt (readable
        fingerprint) and chunker_config.json (settings to rebuild the chunker).
        Refusing any existing folder means nothing is ever overwritten.
        """
        layout = self.layout(name)
        if layout.root.exists():
            raise ValueError(f"{layout.root} already exists; pick another name")
        layout.sql_dir.mkdir(parents=True)
        layout.chunker_file.write_text(chunker_fp)
        layout.chunker_config_file.write_text(config.model_dump_json(indent=2))
        return layout

    def read_chunker_config(self, name: str) -> ChunkerConfig:
        """Return the settings an index was created with; ValueError if it has none."""
        layout = self.layout(name)
        if not layout.chunker_config_file.exists():
            raise ValueError(
                f"Index {name!r} not found (no {layout.chunker_config_file}); "
                "run without --index to create one"
            )
        return ChunkerConfig.model_validate_json(layout.chunker_config_file.read_text())

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
