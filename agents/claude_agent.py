"""Claude (Anthropic) agent for The Sandbox.

When use_native_reasoning=True, extended thinking is enabled and reasoning
is captured from the API's `thinking` content blocks. When False (default),
plain messages.create runs fast and the model uses the prompted JSON
`thinking` field instead.
"""

from __future__ import annotations

import logging
import os

import anthropic

from agents.base_agent import BaseAgent
from agents.llm_response import LLMResponse

log = logging.getLogger(__name__)


class ClaudeAgent(BaseAgent):
    """AI agent backed by Anthropic Claude."""

    def __init__(
        self,
        player_id: str = "claude_agent",
        display_name: str = "Claude Opus 4.6",
        server_url: str = "ws://localhost:8000",
        api_key: str | None = None,
        model: str = "claude-opus-4-6",
        thinking_budget: int = 4000,
        use_native_reasoning: bool = False,
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
            use_native_reasoning=use_native_reasoning,
        )
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self._model = model
        self._thinking_budget = thinking_budget

    async def call_llm(self, prompt: str) -> LLMResponse:
        if self.use_native_reasoning:
            msg = await self.client.messages.create(
                model=self._model,
                max_tokens=self._thinking_budget + 4000,
                thinking={"type": "enabled", "budget_tokens": self._thinking_budget},
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            for block in msg.content:
                btype = getattr(block, "type", None)
                if btype == "thinking":
                    thinking_parts.append(getattr(block, "thinking", "") or "")
                elif btype == "text":
                    text_parts.append(getattr(block, "text", "") or "")
            usage = getattr(msg, "usage", None)
            return LLMResponse(
                content="".join(text_parts),
                reasoning="\n".join(p for p in thinking_parts if p),
                prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            )

        # Default fast path — plain text response, model writes prompted "thinking"
        msg = await self.client.messages.create(
            model=self._model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text if msg.content else ""
        usage = getattr(msg, "usage", None)
        return LLMResponse(
            content=text,
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )
