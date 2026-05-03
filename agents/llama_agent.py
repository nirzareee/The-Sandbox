"""Llama (Together AI) agent for The Sandbox."""

from __future__ import annotations

import logging
import os

import together

from agents.base_agent import BaseAgent

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
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
        )
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY", "")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY required")
        self.client = together.AsyncTogether(api_key=self.api_key)
        self._model = model

    async def call_llm(self, prompt: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self._model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
