"""Tests for generation: turning the model's response into answer text (no network calls)."""

import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, Usage

from rag.generation import create_generator
from rag.generation.anthropic_foundry import text_from_message


def make_message(blocks: list, stop_reason: str) -> Message:
    """A real SDK Message, built by hand as if the API had returned it."""
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-opus-5-5-fast",
        content=blocks,
        stop_reason=stop_reason,
        usage=Usage(input_tokens=10, output_tokens=10),
    )


def test_answer_text_skips_thinking_blocks():
    """Opus 5.5 always thinks, so a response mixes thinking and text; only text is the answer."""
    blocks = [
        ThinkingBlock(type="thinking", thinking="", signature="sig"),
        TextBlock(type="text", text="Paris is the capital"),
        TextBlock(type="text", text=" of France."),
    ]

    answer = text_from_message(make_message(blocks, stop_reason="end_turn"))

    assert answer == "Paris is the capital of France."


def test_refused_or_truncated_response_is_an_error():
    """A refusal or a cut-off answer is never returned as if it were complete."""
    partial_answer = [TextBlock(type="text", text="Employees may")]

    with pytest.raises(ValueError):
        text_from_message(make_message([], stop_reason="refusal"))
    with pytest.raises(ValueError):
        text_from_message(make_message(partial_answer, stop_reason="max_tokens"))


def test_unknown_generator_provider_is_rejected():
    """A typo in LLM_PROVIDER fails immediately, before any client is built."""
    with pytest.raises(ValueError):
        create_generator(provider="nope", model_name="any-model")
