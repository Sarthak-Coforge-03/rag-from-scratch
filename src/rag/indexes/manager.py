"""IndexManager: the only code that creates index folders.

Indexes are addressed by name under one root folder (INDEX_ROOT), e.g.
indexes/structural-800. One index = one chunker config, fixed when it is created:

- create: always a brand-new folder (next_free_name picks structural-800, then
  structural-800_1, ...); writes sql/, chunker.txt and chunker_config.json
- update: the chunk step rebuilds the chunker from chunker_config.json, so an
  index's settings can never change and nothing is ever wiped
- vector folders are created only when asked for (create_vector_dir); creating one
  for a model that already has a folder deletes that folder and starts it over
"""

import re
import shutil
from pathlib import Path

from rag.indexes.layout import EMBEDDER_CONFIG_FILE_NAME, IndexLayout
from rag.models import ChunkerConfig, EmbedderConfig

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
                f"Index {name!r} not found (no {layout.chunker_config_file})"
            )
        return ChunkerConfig.model_validate_json(layout.chunker_config_file.read_text())

    def list_index_names(self) -> list[str]:
        """Names of the indexes the chunk step has filled, sorted (for the CLI menu).

        Uses the same check as open_chunked, so every listed name can be opened.
        """
        if not self.root_dir.exists():
            return []
        index_names = []
        for folder in sorted(self.root_dir.iterdir()):
            layout = IndexLayout(root=folder)
            is_chunked = layout.chunker_file.exists() and layout.database_file.exists()
            if folder.is_dir() and is_chunked:
                index_names.append(folder.name)
        return index_names

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

    def create_vector_dir(
        self, name: str, embedder_fp: str, config: EmbedderConfig
    ) -> Path:
        """Create an empty vector folder for one model in a chunked index.

        If that model already has a folder it is deleted first, so the model is
        embedded again from scratch; other models' folders are not touched.
        Writes embedder.txt (readable fingerprint) and embedder_config.json
        (settings to rebuild the embedder on --sync).

        Raises ValueError if the index hasn't been chunked, or if the folder belongs
        to a different model (two model names that become the same safe folder name).
        """
        layout = self.open_chunked(name)
        vector_dir = layout.vector_dir(embedder_fp)
        embedder_file = layout.embedder_file(embedder_fp)

        if vector_dir.exists():
            if not embedder_file.exists():
                raise ValueError(
                    f"{vector_dir} is not a vector folder; not deleting it"
                )
            stored_embedder_fp = embedder_file.read_text()
            if stored_embedder_fp != embedder_fp:
                raise ValueError(
                    f"{vector_dir} holds vectors from {stored_embedder_fp}, "
                    f"not {embedder_fp}"
                )
            shutil.rmtree(vector_dir)

        vector_dir.mkdir(parents=True)
        embedder_file.write_text(embedder_fp)
        layout.embedder_config_file(embedder_fp).write_text(
            config.model_dump_json(indent=2)
        )
        return vector_dir

    def read_embedder_configs(self, name: str) -> list[EmbedderConfig]:
        """Return the settings of every vector folder in a chunked index, by folder name.

        Folders without embedder_config.json are skipped. An empty list means the
        index has no vectors yet.
        """
        layout = self.open_chunked(name)
        if not layout.vectors_root.exists():
            return []
        configs = []
        for vector_folder in sorted(layout.vectors_root.iterdir()):
            config_file = vector_folder / EMBEDDER_CONFIG_FILE_NAME
            if config_file.exists():
                config = EmbedderConfig.model_validate_json(config_file.read_text())
                configs.append(config)
        return configs
