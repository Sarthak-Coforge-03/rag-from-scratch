"""Tests for splitting Documents into Chunks with metadata."""

from pathlib import Path

from rag.chunk import chunk_document, create_chunker, find_heading
from rag.documents import load_document

PTO_POLICY_PATH = Path(__file__).parent.parent / "data" / "policies" / "pto_policy.md"


def test_find_heading_prefers_heading_inside_chunk_then_nearest_above():
    """A heading inside the chunk wins; otherwise the nearest one above it; otherwise ""."""
    text = "# Title\nintro\n## First\nalpha\n## Second\nbeta"
    second_heading_start = text.index("## Second")
    beta_start = text.index("beta")

    chunk_spanning_second_heading = find_heading(text, second_heading_start - 3, len(text))
    chunk_after_all_headings = find_heading(text, beta_start, len(text))
    text_without_headings = find_heading("no headings here", 0, 16)

    assert chunk_spanning_second_heading == "Second"
    assert chunk_after_all_headings == "Second"
    assert text_without_headings == ""


def test_chunk_document_builds_ids_offsets_and_headings():
    """Chunks get versioned ids, offsets that point back into the text, and their section heading."""
    document = load_document(PTO_POLICY_PATH)
    chunker = create_chunker(chunk_size=800, chunk_overlap=150)

    chunks = chunk_document(document, chunker)

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"pto_policy::v4.0::{index}"
        assert chunk.metadata.chunk_index == index
        start = chunk.metadata.start_index
        end = chunk.metadata.end_index
        assert document.text[start:end] == chunk.text

    # Overlap means the PTO-5 heading can also appear at the end of the previous chunk,
    # which is labelled PTO-4 because it starts in that section. Pick the chunk where
    # PTO-5 is the only heading present.
    carryover_chunks = [
        chunk
        for chunk in chunks
        if "## PTO-5 Carryover" in chunk.text and "## PTO-4" not in chunk.text
    ]
    assert carryover_chunks[0].metadata.heading == "PTO-5 Carryover"
