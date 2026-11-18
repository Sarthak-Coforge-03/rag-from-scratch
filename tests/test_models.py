"""Tests for the pydantic models' validation and serialisation rules."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.models import ChunkMetadata, User


def test_valid_user_is_created():
    """A positive id and a non-empty name produce a User with those values."""
    user = User(id=1, name="Ada")

    assert user.id == 1
    assert user.name == "Ada"


def test_id_must_be_positive():
    """An id of zero breaks the gt=0 rule and is rejected."""
    with pytest.raises(ValidationError):
        User(id=0, name="Ada")


def test_name_must_not_be_empty():
    """An empty name breaks the min_length=1 rule and is rejected."""
    with pytest.raises(ValidationError):
        User(id=1, name="")


def test_chunk_metadata_dumps_to_chroma_safe_values():
    """Chroma metadata only accepts str/int/float/bool, so a JSON dump must contain nothing else."""
    metadata = ChunkMetadata(
        doc_id="pto_policy",
        title="Paid Time Off (PTO) Policy",
        policy_id="PTO-001",
        version="4.0",
        effective_date=date(2026, 1, 1),
        source_path=Path("data/policies/pto_policy.md"),
        heading="",
        chunk_index=0,
        start_index=0,
        end_index=120,
        token_count=30,
    )

    dumped = metadata.model_dump(mode="json")

    assert dumped["effective_date"] == "2026-01-01"
    for value in dumped.values():
        assert isinstance(value, (str, int, float, bool))
