"""Gemini (Google) agent for The Sandbox."""

from __future__ import annotations

import asyncio
import logging
import os

import google.generativeai as genai

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class GeminiAgent(BaseAgent):
    """AI agent backed by Google Gemini."""

    def __init__(
        self,
        player_id: str = "gemini_agent",
        display_name: str = "Gemini",
        server_url: str = "ws://localhost:8000",
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
        )
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required")
        genai.configure(api_key=self.api_key)
        self._genai_model = genai.GenerativeModel(model)

    async def call_llm(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, self._genai_model.generate_content, prompt,
        )
        return resp.text
