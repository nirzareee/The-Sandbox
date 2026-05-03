"""Grok (xAI) agent for The Sandbox.

OpenAI-compatible API at https://api.x.ai/v1. When use_native_reasoning=True,
captures `reasoning_content` from the response. When False (default),
ignores reasoning if present and runs fast.
"""

from __future__ import annotations

import logging
import os

import openai

from agents.base_agent import BaseAgent
from agents.llm_response import LLMResponse

log = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1"


class GrokAgent(BaseAgent):
    """AI agent backed by xAI Grok."""

    def __init__(
        self,
        player_id: str = "grok_agent",
        display_name: str = "Grok 4.3",
        server_url: str = "ws://localhost:8000",
        api_key: str | None = None,
        model: str = "grok-4.3",
        use_native_reasoning: bool = False,
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
            api_timeout=30,
            use_native_reasoning=use_native_reasoning,
        )
        self.api_key = api_key or os.getenv("XAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("XAI_API_KEY required")
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=XAI_BASE_URL,
        )
        self._model = model

    async def call_llm(self, prompt: str) -> LLMResponse:
        resp = await self.client.chat.completions.create(
            model=self._model,
            max_tokens=4096 if self.use_native_reasoning else 2000,
            messages=[{"role": "user", "content": prompt}],
        )
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        if self.use_native_reasoning:
            return LLMResponse(
                content=msg.content or "",
                reasoning=getattr(msg, "reasoning_content", "") or "",
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
            )
        return LLMResponse(
            content=msg.content or "",
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
        )
