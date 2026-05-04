"""Standard return type for an agent's LLM call."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Result of one LLM call.

    Attributes:
        content: The model's visible response text — the JSON action our
            prompts request.
        reasoning: Native API reasoning trace captured from the provider
            (Claude extended thinking blocks, OpenAI reasoning summaries,
            Gemini thought parts, OpenAI-compatible `reasoning_content`,
            etc.). Empty string when the provider does not expose one.
        prompt_tokens: Input tokens billed by the provider for this call.
            0 if not reported.
        completion_tokens: Output tokens (visible content + native reasoning,
            depending on provider). 0 if not reported.
    """

    content: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
