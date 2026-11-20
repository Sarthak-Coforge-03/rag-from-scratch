"""Tests for IndexManager: the only code that creates, inspects or deletes index folders.

One index = one chunker config. Different configs live side by side under different
index names:

    indexes/token-800/        TokenWindowChunker(size=800, overlap=150)
    indexes/structural-800/   StructuralChunker(size=800, overlap=0)

Running the chunk step with...
    a new index name                         -> new index, nothing else touched
    an existing name, same config            -> kept
    an existing name, different config       -> that index wiped and rebuilt; others untouched
"""

import pytest

from rag.indexes import IndexManager

TOKEN_CHUNKER = "TokenWindowChunker(size=800, overlap=150)"
TOKEN_CHUNKER_400 = "TokenWindowChunker(size=400, overlap=150)"
STRUCTURAL_CHUNKER = "StructuralChunker(size=800, overlap=0)"
MINILM_EMBEDDER = "SentenceTransformerEmbedder(all-MiniLM-L6-v2)"
BGE_EMBEDDER = "SentenceTransformerEmbedder(BAAI/bge-small-en-v1.5)"


def build_index_with_data(manager, name, chunker_fp):
    """Chunk an index (stand-in docs.db) and embed it with one model.

    Returns the database file and the vector folder, so tests can check what survives.
    """
    layout = manager.prepare_for_chunking(name, chunker_fp)
    layout.database_file.write_text("chunks")
    vector_dir = manager.prepare_vector_dir(name, MINILM_EMBEDDER)
    return layout.database_file, vector_dir


def test_new_index_name_creates_a_new_index_without_touching_others(tmp_path):
    """Two strategies side by side: indexes/token-800 and indexes/structural-800."""
    manager = IndexManager(tmp_path)
    token_database, token_vectors = build_index_with_data(
        manager, "token-800", TOKEN_CHUNKER
    )

    structural_layout = manager.prepare_for_chunking(
        "structural-800", STRUCTURAL_CHUNKER
    )

    assert structural_layout.chunker_file.read_text() == STRUCTURAL_CHUNKER
    assert not structural_layout.vectors_root.exists()  # vectors only when asked for
    assert token_database.exists()
    assert token_vectors.exists()


def test_existing_name_with_same_config_is_kept(tmp_path):
    """token-800 chunked again with the same TokenWindowChunker settings -> nothing wiped."""
    manager = IndexManager(tmp_path)
    database_file, vector_dir = build_index_with_data(
        manager, "token-800", TOKEN_CHUNKER
    )

    manager.prepare_for_chunking("token-800", TOKEN_CHUNKER)

    assert database_file.exists()
    assert vector_dir.exists()


@pytest.mark.parametrize(
    "new_config",
    [TOKEN_CHUNKER_400, STRUCTURAL_CHUNKER],
    ids=["different-size", "different-strategy"],
)
def test_existing_name_with_different_config_is_wiped_others_untouched(
    tmp_path, new_config
):
    """token-800 rechunked with another size OR strategy -> only token-800 is wiped.

    Its SQL and vectors are removed and no vector folder is recreated; the other
    index (structural-800) keeps all its data.
    """
    manager = IndexManager(tmp_path)
    database_file, vector_dir = build_index_with_data(
        manager, "token-800", TOKEN_CHUNKER
    )
    other_database, other_vectors = build_index_with_data(
        manager, "structural-800", STRUCTURAL_CHUNKER
    )

    layout = manager.prepare_for_chunking("token-800", new_config)

    assert not database_file.exists()
    assert not vector_dir.exists()
    assert layout.sql_dir.is_dir()
    assert layout.chunker_file.read_text() == new_config
    assert other_database.exists()
    assert other_vectors.exists()


def test_folders_that_are_not_indexes_are_never_touched(tmp_path):
    """A non-empty folder without chunker.txt is refused by every write operation."""
    manager = IndexManager(tmp_path)
    unrelated_folder = tmp_path / "my_documents"
    unrelated_folder.mkdir()
    precious_file = unrelated_folder / "notes.md"
    precious_file.write_text("do not delete me")

    with pytest.raises(ValueError):
        manager.prepare_for_chunking("my_documents", TOKEN_CHUNKER)
    with pytest.raises(ValueError):
        manager.delete_index("my_documents")
    with pytest.raises(ValueError):
        manager.layout("../escape")

    assert precious_file.read_text() == "do not delete me"


def test_vectors_need_a_chunked_index_and_each_model_gets_its_own_folder(tmp_path):
    """Before the chunk step has run there is nothing to embed; afterwards one folder per model."""
    manager = IndexManager(tmp_path)
    layout = manager.prepare_for_chunking("token-800", TOKEN_CHUNKER)  # no docs.db yet

    with pytest.raises(ValueError):
        manager.open_chunked("token-800")
    with pytest.raises(ValueError):
        manager.prepare_vector_dir("token-800", MINILM_EMBEDDER)

    layout.database_file.write_text("chunks")  # the chunk step has now run
    minilm_dir = manager.prepare_vector_dir("token-800", MINILM_EMBEDDER)
    bge_dir = manager.prepare_vector_dir("token-800", BGE_EMBEDDER)

    assert manager.open_chunked("token-800").root == tmp_path / "token-800"
    assert minilm_dir != bge_dir
    assert (bge_dir / "embedder.txt").read_text() == BGE_EMBEDDER


def test_list_and_delete_indexes(tmp_path):
    """list_indexes reports each index's state; delete_index removes one completely."""
    manager = IndexManager(tmp_path)
    build_index_with_data(manager, "token-800", TOKEN_CHUNKER)
    manager.prepare_for_chunking("structural-800", STRUCTURAL_CHUNKER)

    summaries = manager.list_indexes()

    assert [summary.name for summary in summaries] == ["structural-800", "token-800"]
    assert summaries[0].chunker_fingerprint == STRUCTURAL_CHUNKER
    assert summaries[0].embedder_fingerprints == []
    assert summaries[0].is_chunked is False
    assert summaries[1].embedder_fingerprints == [MINILM_EMBEDDER]
    assert summaries[1].is_chunked is True

    manager.delete_index("token-800")

    assert [summary.name for summary in manager.list_indexes()] == ["structural-800"]
    assert not (tmp_path / "token-800").exists()
