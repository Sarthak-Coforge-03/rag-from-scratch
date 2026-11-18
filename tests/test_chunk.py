"""Tests for splitting Documents into Chunks with metadata."""

from pathlib import Path

import pytest

from rag.chunk import create_chunker
from rag.documents import load_document

PTO_POLICY_PATH = Path(__file__).parent.parent / "data" / "policies" / "pto_policy.md"


def test_chunk_document_builds_ids_and_offsets():
    """Chunks get versioned ids and offsets that point back into the document text."""
    document = load_document(PTO_POLICY_PATH)
    chunker = create_chunker(strategy="token", chunk_size=800, chunk_overlap=150)

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"pto_policy::v4.0::{index}"
        assert chunk.metadata.chunk_index == index
        start = chunk.metadata.start_index
        end = chunk.metadata.end_index
        assert document.text[start:end] == chunk.text


def test_unknown_strategy_is_rejected():
    """A typo in CHUNK_STRATEGY should fail at startup, not fall back to some default silently."""
    with pytest.raises(ValueError):
        create_chunker(strategy="no-such-strategy", chunk_size=800, chunk_overlap=150)
