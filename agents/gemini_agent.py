"""Gemini (Google) agent for The Sandbox.

When use_native_reasoning=True, enables include_thoughts and captures
thought parts. When False (default), uses minimum thinking_budget so calls
stay fast; the model writes reasoning into the prompted JSON `thinking`
field instead.
"""

from __future__ import annotations

import logging
import os

from google import genai
from google.genai import types

from agents.base_agent import BaseAgent
from agents.llm_response import LLMResponse

log = logging.getLogger(__name__)


class GeminiAgent(BaseAgent):
    """AI agent backed by Google Gemini."""

    def __init__(
        self,
        player_id: str = "gemini_agent",
        display_name: str = "Gemini 2.5 Flash",
        server_url: str = "ws://localhost:8000",
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        thinking_budget: int = 4000,
        use_native_reasoning: bool = False,
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
            use_native_reasoning=use_native_reasoning,
        )
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required")
        self.client = genai.Client(api_key=self.api_key)
        self._model = model
        self._thinking_budget = thinking_budget

    async def call_llm(self, prompt: str) -> LLMResponse:
        if self.use_native_reasoning:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self._thinking_budget,
                    include_thoughts=True,
                ),
            )
            resp = await self.client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
            text_parts: list[str] = []
            thought_parts: list[str] = []
            try:
                for cand in (resp.candidates or []):
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    for part in getattr(content, "parts", []) or []:
                        if getattr(part, "thought", False):
                            thought_parts.append(getattr(part, "text", "") or "")
                        else:
                            text_parts.append(getattr(part, "text", "") or "")
            except Exception:  # noqa: BLE001 — fall back to whole-response text
                text_parts = [getattr(resp, "text", "") or ""]
            usage = getattr(resp, "usage_metadata", None)
            return LLMResponse(
                content="".join(text_parts),
                reasoning="\n".join(p for p in thought_parts if p),
                prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
                completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            )

        # Default fast path — minimum allowed thinking, no thoughts captured
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=128),
        )
        resp = await self.client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            content=getattr(resp, "text", "") or "",
            prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
        )
