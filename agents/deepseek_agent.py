"""DeepSeek agent for The Sandbox.

OpenAI-compatible API at https://api.deepseek.com/v1. When
use_native_reasoning=True, native thinking stays on and reasoning_content
is captured. When False (default), thinking is explicitly disabled for
speed.
"""

from __future__ import annotations

import logging
import os

import openai

from agents.base_agent import BaseAgent
from agents.llm_response import LLMResponse

log = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekAgent(BaseAgent):
    """AI agent backed by DeepSeek's native API."""

    def __init__(
        self,
        player_id: str = "deepseek_agent",
        display_name: str = "DeepSeek V4 Pro",
        server_url: str = "ws://localhost:8000",
        api_key: str | None = None,
        model: str = "deepseek-v4-pro",
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
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY required")
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=DEEPSEEK_BASE_URL,
        )
        self._model = model

    async def call_llm(self, prompt: str) -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096 if self.use_native_reasoning else 2000,
            "messages": [{"role": "user", "content": prompt}],
            # Force strict JSON output. Without this, V4 Pro occasionally
            # emits malformed JSON (raw escape sequences outside strings,
            # premature closing quotes) that drops to fallback action and
            # silently randomizes votes/hints.
            "response_format": {"type": "json_object"},
        }
        if not self.use_native_reasoning:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        resp = await self.client.chat.completions.create(**kwargs)
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
