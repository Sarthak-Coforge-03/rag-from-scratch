"""Tests for creating embedders and the vectors they produce."""

import math

import pytest

from rag.embedding import create_embedder

MINILM_MODEL = "all-MiniLM-L6-v2"
MINILM_DIMENSIONS = 384


def test_unknown_provider_is_rejected():
    """A typo in EMBED_PROVIDER should fail at startup, before any model is loaded."""
    with pytest.raises(ValueError):
        create_embedder(
            provider="no-such-provider", model_name=MINILM_MODEL, batch_size=2
        )


def test_sentence_transformer_embedder_returns_normalized_vectors():
    """Documents and queries become 384-number unit vectors, one per text, even across batches.

    Loads the real model (cached after the first download).
    """
    embedder = create_embedder(
        provider="sentence-transformers", model_name=MINILM_MODEL, batch_size=2
    )
    texts = [
        "Meals are reimbursed up to $75 per day.",
        "Passwords must be at least 14 characters.",
        "Hybrid employees work on-site on Tuesdays and Thursdays.",
    ]

    document_vectors = embedder.embed_documents(texts)
    query_vector = embedder.embed_query("How much can I spend on food?")

    # 3 texts with batch_size=2 means two batches; all three vectors must come back.
    assert len(document_vectors) == 3
    for vector in document_vectors:
        assert len(vector) == MINILM_DIMENSIONS
        # A normalized vector has length (Euclidean norm) exactly 1: sqrt(sum of squares).
        # The model computes in 32-bit floats, so the norm comes out as something like
        # 0.99999994 rather than exactly 1.0. Comparing with == would fail on that rounding,
        # so math.isclose accepts any value within a relative tolerance of 0.0001 (0.01%).
        vector_length = math.sqrt(sum(value * value for value in vector))
        assert math.isclose(vector_length, 1.0, rel_tol=1e-4)
    assert len(query_vector) == MINILM_DIMENSIONS
