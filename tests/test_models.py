"""Tests for the User model's validation rules."""

import pytest
from pydantic import ValidationError

from rag.models import User


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
