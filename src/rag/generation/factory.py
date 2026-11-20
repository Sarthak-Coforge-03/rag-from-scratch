"""Pick an LLM provider by name (LLM_PROVIDER in config.py)."""

from rag.generation.anthropic_foundry import AnthropicFoundryGenerator
from rag.generation.base import BaseGenerator

# Provider name (LLM_PROVIDER) -> generator class. Add new providers here.
GENERATOR_CLASSES = {
    "anthropic-foundry": AnthropicFoundryGenerator,
}


def create_generator(provider: str, model_name: str) -> BaseGenerator:
    """Build the generator for a provider name, e.g. settings.llm_provider.

    Raises ValueError listing the valid names if the provider is unknown, so a typo in
    LLM_PROVIDER fails immediately, before any client is built.
    """
    if provider not in GENERATOR_CLASSES:
        valid_names = ", ".join(GENERATOR_CLASSES)
        raise ValueError(
            f"Unknown LLM provider {provider!r}; choose one of: {valid_names}"
        )
    generator_class = GENERATOR_CLASSES[provider]
    return generator_class(model_name=model_name)
