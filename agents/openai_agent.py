"""OpenAI agent for The Sandbox.

When use_native_reasoning=True, uses the Responses API to capture reasoning
summaries. When False (default), uses Chat Completions for fast responses
and the model writes its reasoning into the prompted JSON `thinking` field.
"""

from __future__ import annotations

import logging
import os

import openai

from agents.base_agent import BaseAgent
from agents.llm_response import LLMResponse

log = logging.getLogger(__name__)


class OpenAIAgent(BaseAgent):
    """AI agent backed by OpenAI (default: GPT-5.5)."""

    def __init__(
        self,
        player_id: str = "openai_agent",
        display_name: str = "GPT-5.5",
        server_url: str = "ws://localhost:8000",
        api_key: str | None = None,
        model: str = "gpt-5.5",
        reasoning_effort: str = "medium",
        use_native_reasoning: bool = False,
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
            use_native_reasoning=use_native_reasoning,
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required")
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self._model = model
        self._reasoning_effort = reasoning_effort

    async def call_llm(self, prompt: str) -> LLMResponse:
        if self.use_native_reasoning:
            resp = await self.client.responses.create(
                model=self._model,
                input=prompt,
                reasoning={"effort": self._reasoning_effort, "summary": "auto"},
                max_output_tokens=4000,
            )
            text = getattr(resp, "output_text", "") or ""
            reasoning_parts: list[str] = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "reasoning":
                    for s in getattr(item, "summary", []) or []:
                        reasoning_parts.append(getattr(s, "text", "") or "")
            usage = getattr(resp, "usage", None)
            return LLMResponse(
                content=text,
                reasoning="\n".join(p for p in reasoning_parts if p),
                prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            )

        # Default fast path — Chat Completions, model writes prompted "thinking".
        # GPT-5.x reasoning-class models require max_completion_tokens (not max_tokens).
        resp = await self.client.chat.completions.create(
            model=self._model,
            max_completion_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )
