"""Llama (Together AI) agent for The Sandbox.

Together-hosted Llama doesn't expose native reasoning, so the
use_native_reasoning flag has no effect — the model always uses the
prompted JSON `thinking` field.
"""

from __future__ import annotations

import logging
import os

import together

from agents.base_agent import BaseAgent
from agents.llm_response import LLMResponse

log = logging.getLogger(__name__)


class LlamaAgent(BaseAgent):
    """AI agent backed by Together AI (Llama)."""

    def __init__(
        self,
        player_id: str = "llama_agent",
        display_name: str = "Llama",
        server_url: str = "ws://localhost:8000",
        api_key: str | None = None,
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        use_native_reasoning: bool = False,
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
            use_native_reasoning=use_native_reasoning,
        )
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY", "")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY required")
        self.client = together.AsyncTogether(api_key=self.api_key)
        self._model = model

    async def call_llm(self, prompt: str) -> LLMResponse:
        resp = await self.client.chat.completions.create(
            model=self._model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=msg.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )
