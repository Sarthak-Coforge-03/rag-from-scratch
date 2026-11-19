"""Tests for splitting Documents into Chunks with metadata."""

from pathlib import Path

import pytest

from rag.chunking import create_chunker
from rag.documents import load_document

PTO_POLICY_PATH = Path(__file__).parent.parent / "data" / "policies" / "pto_policy.md"


@pytest.mark.parametrize("strategy", ["token", "recursive", "structural"])
def test_chunk_document_builds_ids_and_offsets(strategy):
    """Every strategy gives chunks versioned ids and offsets that point back into the text."""
    document = load_document(PTO_POLICY_PATH)
    chunker = create_chunker(strategy=strategy, chunk_size=800, chunk_overlap=150)

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"pto_policy::v4.0::{index}"
        assert chunk.metadata.chunk_index == index
        start = chunk.metadata.start_index
        end = chunk.metadata.end_index
        assert document.text[start:end] == chunk.text


def test_recursive_strategy_never_ends_a_chunk_mid_word():
    """Recursive chunks end at a natural boundary, unlike fixed character windows."""
    document = load_document(PTO_POLICY_PATH)
    chunker = create_chunker(strategy="recursive", chunk_size=800, chunk_overlap=150)

    chunks = chunker.chunk_document(document)

    for chunk in chunks:
        end = chunk.metadata.end_index
        is_end_of_text = end == len(document.text)
        ends_on_whitespace = chunk.text[-1].isspace()
        next_character_is_whitespace = (
            not is_end_of_text and document.text[end].isspace()
        )
        assert is_end_of_text or ends_on_whitespace or next_character_is_whitespace


def test_structural_strategy_starts_a_chunk_at_a_heading(tmp_path):
    """When two sections don't fit in one chunk, the second section's chunk begins with its heading."""
    section_body = "This sentence pads the section so it fills most of a chunk. " * 2
    path = tmp_path / "two_sections.md"
    path.write_text(
        f"# Title\n\n## First\n\n{section_body}\n\n## Second\n\n{section_body}\n"
    )
    document = load_document(path)
    chunker = create_chunker(strategy="structural", chunk_size=150, chunk_overlap=0)

    chunks = chunker.chunk_document(document)

    chunk_starts = [chunk.text.lstrip("\n")[:9] for chunk in chunks]
    assert "## Second" in chunk_starts


def test_unknown_strategy_is_rejected():
    """A typo in CHUNK_STRATEGY should fail at startup, not fall back to some default silently."""
    with pytest.raises(ValueError):
        create_chunker(strategy="no-such-strategy", chunk_size=800, chunk_overlap=150)
