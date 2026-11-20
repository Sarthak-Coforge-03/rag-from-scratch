"""Generation: send a prompt to an LLM and get its answer text back.

base.py               BaseGenerator, the interface (generate(system_prompt, user_message))
anthropic_foundry.py  AnthropicFoundryGenerator, Claude on a Microsoft Foundry deployment
factory.py            create_generator(provider, model_name), picks a class by LLM_PROVIDER

generator = create_generator(settings.llm_provider, settings.llm_model)
answer = generator.generate("Answer briefly.", "What is the capital of France?")
"""

from rag.generation.anthropic_foundry import AnthropicFoundryGenerator
from rag.generation.base import BaseGenerator
from rag.generation.factory import GENERATOR_CLASSES, create_generator

__all__ = [
    "AnthropicFoundryGenerator",
    "BaseGenerator",
    "GENERATOR_CLASSES",
    "create_generator",
]
