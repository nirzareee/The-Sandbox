"""Qwen 3.5 (Together AI) agent for The Sandbox."""

from __future__ import annotations

import asyncio
import logging
import os

import together

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class QwenAgent(BaseAgent):
    """AI agent backed by Together AI (Qwen 3.5)."""

    def __init__(
        self,
        player_id: str = "qwen_agent",
        display_name: str = "Qwen 3.5",
        server_url: str = "ws://localhost:8000",
        api_key: str | None = None,
        model: str = "Qwen/Qwen3.5-397B-A17B",
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
            api_timeout=25,
            max_retries=1,
        )
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY", "")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY required")
        self.client = together.Together(api_key=self.api_key, max_retries=0)
        self._model = model

    async def call_llm(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._call_sync, prompt)

    def _call_sync(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
