"""AnthropicFoundryGenerator: Claude served from a Microsoft Foundry (Azure) deployment.

Settings (from .env): LLM_ENDPOINT, ANTHROPIC_FOUNDRY_API_KEY, LLM_MAX_TOKENS,
LLM_EFFORT. The model name is the Foundry deployment name, e.g.
"claude-opus-5-5-fast"; on Foundry the "-fast" deployment is what turns on fast
mode, so no speed setting is sent with the request.
"""

from anthropic import AnthropicFoundry
from anthropic.types import Message

from rag.config import settings
from rag.generation.base import BaseGenerator


class AnthropicFoundryGenerator(BaseGenerator):
    """Answers with Claude through the Anthropic SDK's Foundry client."""

    def __init__(self, model_name: str):
        """Build the Foundry client; ValueError if the endpoint or API key is missing."""
        super().__init__(model_name=model_name)
        if not settings.llm_endpoint:
            raise ValueError(
                "LLM_ENDPOINT is not set; add your Foundry endpoint to .env"
            )
        if not settings.llm_api_key:
            raise ValueError(
                "ANTHROPIC_FOUNDRY_API_KEY is not set; add your Foundry key to .env"
            )
        self.client = AnthropicFoundry(
            api_key=settings.llm_api_key, base_url=settings.llm_endpoint
        )
        self.max_tokens = settings.llm_max_tokens
        self.effort = settings.llm_effort

    def generate(self, system_prompt: str, user_message: str) -> str:
        """Send one question to Claude and return its answer text."""
        message = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=self.max_tokens,
            # How hard Claude thinks before answering; "low" suits short Q&A.
            output_config={"effort": self.effort},
        )
        return text_from_message(message)


def text_from_message(message: Message) -> str:
    """Join the answer's text blocks; ValueError if the answer is refused or cut off.

    Claude Opus 5.5 always thinks before answering, so the response holds thinking
    blocks as well as text blocks. Only the text blocks are the answer.
    """
    if message.stop_reason == "refusal":
        raise ValueError("The model declined to answer this question")
    if message.stop_reason == "max_tokens":
        raise ValueError(
            "The answer was cut off at LLM_MAX_TOKENS; raise it in .env and retry"
        )

    answer_parts = [block.text for block in message.content if block.type == "text"]
    return "".join(answer_parts)
