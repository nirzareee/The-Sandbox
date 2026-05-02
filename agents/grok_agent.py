"""Grok (xAI) agent for The Sandbox."""

from __future__ import annotations

import logging
import os

import openai

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class GrokAgent(BaseAgent):
    """AI agent backed by xAI's Grok via the OpenAI-compatible API."""

    def __init__(
        self,
        player_id: str = "grok_agent",
        display_name: str = "Grok",
        server_url: str = "ws://localhost:8000",
        api_key: str | None = None,
        model: str = "grok-3-mini",
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
        )
        self.api_key = api_key or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROK_API_KEY or XAI_API_KEY required")
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1",
        )
        self._model = model

    async def call_llm(self, prompt: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self._model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
