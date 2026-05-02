"""Groq-hosted LLM agent for The Sandbox."""

from __future__ import annotations

import logging
import os

import groq

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class GroqAgent(BaseAgent):
    """AI agent backed by Groq's fast inference API."""

    def __init__(
        self,
        player_id: str = "groq_agent",
        display_name: str = "Groq Agent",
        server_url: str = "ws://localhost:8000",
        api_key: str | None = None,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
        )
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY required")
        self.client = groq.AsyncGroq(api_key=self.api_key)
        self._model = model

    async def call_llm(self, prompt: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self._model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": "Respond only with valid JSON. Keep your thinking brief."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content
