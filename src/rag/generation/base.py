"""BaseGenerator: the interface every LLM provider implements.

A generator only sends a prompt and returns the model's reply, either as plain text
or as a filled-in pydantic model (structured output). It knows nothing about chunks
or citations; the answer step builds the prompt from the retrieved chunks and hands
it to whichever generator the factory picked.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseGenerator(ABC):
    """Calls one LLM. Subclasses wrap one provider's client library."""

    def __init__(self, model_name: str):
        """Remember which model (or deployment) to call."""
        self.model_name = model_name

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        """Send one question to the model and return its answer text."""

    @abstractmethod
    def generate_structured(
        self, system_prompt: str, user_message: str, output_type: type[BaseModel]
    ) -> BaseModel:
        """Send one question and return the reply as a validated `output_type` instance."""
