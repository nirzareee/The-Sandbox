"""Claude (Anthropic) agent for The Sandbox."""

from __future__ import annotations

import asyncio
import logging
import os

import anthropic

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class ClaudeAgent(BaseAgent):
    """AI agent backed by Anthropic Claude."""

    def __init__(
        self,
        player_id: str = "claude_agent",
        display_name: str = "Claude",
        server_url: str = "ws://localhost:8000",
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250514",
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
        )
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self._model = model

    async def call_llm(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._call_sync, prompt)

    def _call_sync(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self._model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
