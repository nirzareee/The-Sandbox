"""Grok (xAI) agent for The Sandbox.

OpenAI-compatible API at https://api.x.ai/v1. xAI ships separate model variants
for reasoning vs non-reasoning play, so we pick the variant from
`use_native_reasoning` rather than trying to disable thinking on a single model:

- default (use_native_reasoning=False) → `grok-4.20-0309-non-reasoning`
- native reasoning ON                  → `grok-4.20-0309-reasoning`,
  `reasoning_content` captured into LLMResponse.reasoning

Callers can override the model explicitly (e.g. for pinning to grok-4.3).
"""

from __future__ import annotations

import logging
import os

import openai

from agents.base_agent import BaseAgent
from agents.llm_response import LLMResponse

log = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1"

DEFAULT_MODEL_NON_REASONING = "grok-4.20-0309-non-reasoning"
DEFAULT_MODEL_REASONING = "grok-4.20-0309-reasoning"


class GrokAgent(BaseAgent):
    """AI agent backed by xAI Grok."""

    def __init__(
        self,
        player_id: str = "grok_agent",
        display_name: str = "Grok 4.20",
        server_url: str = "ws://localhost:8000",
        api_key: str | None = None,
        model: str | None = None,
        use_native_reasoning: bool = False,
    ) -> None:
        if model is None:
            model = (
                DEFAULT_MODEL_REASONING
                if use_native_reasoning
                else DEFAULT_MODEL_NON_REASONING
            )
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
            api_timeout=60 if use_native_reasoning else 30,
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
