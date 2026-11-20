"""Tests for IndexManager: the only code that creates index folders.

One index = one chunker config, fixed at creation. Creating never touches an existing
folder (a counter picks a free name), and updating rebuilds the chunker from the
settings saved in the index.
"""

import pytest

from rag.chunking import create_chunker
from rag.indexes import IndexManager, chunker_fingerprint
from rag.models import ChunkerConfig, EmbedderConfig

STRUCTURAL_CONFIG = ChunkerConfig(
    strategy="structural", chunk_size=800, chunk_overlap=0
)
STRUCTURAL_CHUNKER = "StructuralChunker(size=800, overlap=0)"
MINILM_EMBEDDER = "SentenceTransformerEmbedder(all-MiniLM-L6-v2)"
BGE_EMBEDDER = "SentenceTransformerEmbedder(BAAI/bge-small-en-v1.5)"
MINILM_CONFIG = EmbedderConfig(
    provider="sentence-transformers", model_name="all-MiniLM-L6-v2"
)
BGE_CONFIG = EmbedderConfig(
    provider="sentence-transformers", model_name="BAAI/bge-small-en-v1.5"
)


def test_saved_chunker_config_rebuilds_the_same_chunker(tmp_path):
    """An update run rebuilds the chunker from the index's saved settings, not from env.

    The config read back must equal what was saved, and the chunker built from it
    must have the same fingerprint as the one that created the index.
    """
    manager = IndexManager(tmp_path)
    manager.create_index("structural-800", STRUCTURAL_CHUNKER, STRUCTURAL_CONFIG)

    loaded_config = manager.read_chunker_config("structural-800")
    rebuilt_chunker = create_chunker(
        strategy=loaded_config.strategy,
        chunk_size=loaded_config.chunk_size,
        chunk_overlap=loaded_config.chunk_overlap,
    )

    assert loaded_config == STRUCTURAL_CONFIG
    assert chunker_fingerprint(rebuilt_chunker) == STRUCTURAL_CHUNKER
    with pytest.raises(ValueError):
        manager.read_chunker_config("never-created")


def test_next_free_name_adds_a_counter_when_the_name_is_taken(tmp_path):
    """Creating from env never reuses a folder: structural-800, then _1, then _2."""
    manager = IndexManager(tmp_path)

    first_name = manager.next_free_name("structural-800")
    manager.create_index(first_name, STRUCTURAL_CHUNKER, STRUCTURAL_CONFIG)
    second_name = manager.next_free_name("structural-800")
    manager.create_index(second_name, STRUCTURAL_CHUNKER, STRUCTURAL_CONFIG)
    third_name = manager.next_free_name("structural-800")

    assert [first_name, second_name, third_name] == [
        "structural-800",
        "structural-800_1",
        "structural-800_2",
    ]


def test_existing_folders_are_never_overwritten(tmp_path):
    """create_index refuses any existing folder, and unsafe names are rejected."""
    manager = IndexManager(tmp_path)
    unrelated_folder = tmp_path / "my_documents"
    unrelated_folder.mkdir()
    precious_file = unrelated_folder / "notes.md"
    precious_file.write_text("do not delete me")

    with pytest.raises(ValueError):
        manager.create_index("my_documents", STRUCTURAL_CHUNKER, STRUCTURAL_CONFIG)
    with pytest.raises(ValueError):
        manager.layout("../escape")

    assert precious_file.read_text() == "do not delete me"


def test_vectors_need_a_chunked_index_and_each_model_gets_its_own_folder(tmp_path):
    """Before the chunk step has run there is nothing to embed; afterwards one folder per model.

    Each folder saves its model's settings, so --sync can rebuild every embedder.
    """
    manager = IndexManager(tmp_path)
    layout = manager.create_index(
        "structural-800", STRUCTURAL_CHUNKER, STRUCTURAL_CONFIG
    )  # no docs.db yet

    with pytest.raises(ValueError):
        manager.create_vector_dir("structural-800", MINILM_EMBEDDER, MINILM_CONFIG)

    layout.database_file.write_text("chunks")  # the chunk step has now run
    minilm_dir = manager.create_vector_dir(
        "structural-800", MINILM_EMBEDDER, MINILM_CONFIG
    )
    bge_dir = manager.create_vector_dir("structural-800", BGE_EMBEDDER, BGE_CONFIG)

    assert minilm_dir != bge_dir
    assert (bge_dir / "embedder.txt").read_text() == BGE_EMBEDDER
    # Folders are listed in name order: BAAI... (bge) sorts before all-MiniLM.
    assert manager.read_embedder_configs("structural-800") == [
        BGE_CONFIG,
        MINILM_CONFIG,
    ]


def test_create_vector_dir_again_starts_that_model_from_scratch(tmp_path):
    """--create on a model that already has vectors deletes its folder; other models stay."""
    manager = IndexManager(tmp_path)
    layout = manager.create_index(
        "structural-800", STRUCTURAL_CHUNKER, STRUCTURAL_CONFIG
    )
    layout.database_file.write_text("chunks")
    minilm_dir = manager.create_vector_dir(
        "structural-800", MINILM_EMBEDDER, MINILM_CONFIG
    )
    bge_dir = manager.create_vector_dir("structural-800", BGE_EMBEDDER, BGE_CONFIG)
    old_minilm_vectors = minilm_dir / "old_vectors.bin"
    old_minilm_vectors.write_text("stale")
    bge_vectors = bge_dir / "vectors.bin"
    bge_vectors.write_text("keep")

    manager.create_vector_dir("structural-800", MINILM_EMBEDDER, MINILM_CONFIG)

    assert not old_minilm_vectors.exists()
    assert (minilm_dir / "embedder.txt").read_text() == MINILM_EMBEDDER
    assert bge_vectors.exists()
