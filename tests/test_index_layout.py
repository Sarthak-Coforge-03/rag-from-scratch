"""Tests for index fingerprints and the paths of an index folder (no disk access)."""

from pathlib import Path

from rag.chunking import create_chunker
from rag.embedding import BaseEmbedder
from rag.indexes import IndexLayout, chunker_fingerprint, embedder_fingerprint


class NamedOnlyEmbedder(BaseEmbedder):
    """An embedder with a model name but no model, so the fingerprint tests load nothing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Not used by these tests."""
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        """Not used by these tests."""
        raise NotImplementedError


def test_fingerprints_name_the_class_and_every_setting():
    """Fingerprints are readable, so a mismatch shows exactly which setting differs."""
    chunker = create_chunker(strategy="token", chunk_size=800, chunk_overlap=150)
    embedder = NamedOnlyEmbedder(model_name="all-MiniLM-L6-v2", batch_size=64)

    assert chunker_fingerprint(chunker) == "TokenWindowChunker(size=800, overlap=150)"
    assert embedder_fingerprint(embedder) == "NamedOnlyEmbedder(all-MiniLM-L6-v2)"


def test_same_config_rebuilt_from_scratch_finds_the_same_paths():
    """Objects rebuilt from the same settings (e.g. in a later run) land in the same folders.

    This is what lets a second run find and update the index the first run created;
    different settings must give a different fingerprint and vector folder.
    """
    first_run_chunker = create_chunker(
        strategy="token", chunk_size=800, chunk_overlap=150
    )
    later_run_chunker = create_chunker(
        strategy="token", chunk_size=800, chunk_overlap=150
    )
    other_chunker = create_chunker(strategy="token", chunk_size=400, chunk_overlap=150)
    first_run_embedder = NamedOnlyEmbedder(model_name="all-MiniLM-L6-v2", batch_size=64)
    later_run_embedder = NamedOnlyEmbedder(model_name="all-MiniLM-L6-v2", batch_size=8)
    other_embedder = NamedOnlyEmbedder(model_name="bge-small-en-v1.5", batch_size=64)
    first_run_layout = IndexLayout(root=Path("indexes/token-800"))
    later_run_layout = IndexLayout(root=Path("indexes/token-800"))

    first_run_vector_dir = first_run_layout.vector_dir(
        embedder_fingerprint(first_run_embedder)
    )
    later_run_vector_dir = later_run_layout.vector_dir(
        embedder_fingerprint(later_run_embedder)
    )
    other_model_vector_dir = first_run_layout.vector_dir(
        embedder_fingerprint(other_embedder)
    )

    assert chunker_fingerprint(first_run_chunker) == chunker_fingerprint(
        later_run_chunker
    )
    assert chunker_fingerprint(first_run_chunker) != chunker_fingerprint(other_chunker)
    assert first_run_layout == later_run_layout
    # batch_size only changes speed, not the vectors, so it must not change the folder.
    assert first_run_vector_dir == later_run_vector_dir
    assert first_run_vector_dir != other_model_vector_dir


def test_layout_derives_every_path_from_the_index_root():
    """All locations inside an index come from its root; nothing is guessed elsewhere."""
    layout = IndexLayout(root=Path("indexes/token-800"))

    assert layout.chunker_file == Path("indexes/token-800/chunker.txt")
    assert layout.database_file == Path("indexes/token-800/sql/docs.db")
    assert layout.database_url == "sqlite:///indexes/token-800/sql/docs.db"
    assert layout.vectors_root == Path("indexes/token-800/vectors")


def test_vector_folder_name_is_safe_for_any_model_name():
    """A "/" in a model name (e.g. BAAI/bge-small) must not create a nested folder."""
    layout = IndexLayout(root=Path("indexes/token-800"))

    vector_dir = layout.vector_dir(
        "SentenceTransformerEmbedder(BAAI/bge-small-en-v1.5)"
    )

    assert vector_dir.parent == layout.vectors_root
    assert vector_dir.name == "SentenceTransformerEmbedder_BAAI_bge-small-en-v1.5_"
