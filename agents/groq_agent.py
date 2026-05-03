"""Groq-hosted LLM agent for The Sandbox."""

from __future__ import annotations

import logging

import groq

from agents.base_agent import BaseAgent
from agents.groq_key_pool import get_pool
from agents.llm_response import LLMResponse

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
        use_native_reasoning: bool = False,
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name=model,
            server_url=server_url,
            use_native_reasoning=use_native_reasoning,
        )
        self._pool = get_pool()
        self._api_key = api_key or self._pool.key_for_round(1)
        self.client = groq.AsyncGroq(api_key=self._api_key)
        self._model = model

    # ------------------------------------------------------------------
    # Key rotation helpers
    # ------------------------------------------------------------------

    def _rotate_key_for_round(self) -> None:
        """Switch client to the key assigned to the current round."""
        new_key = self._pool.key_for_round(max(self.current_round, 1))
        if new_key != self._api_key:
            self._api_key = new_key
            self.client = groq.AsyncGroq(api_key=new_key)
            key_idx = (max(self.current_round, 1) - 1) % self._pool.num_keys + 1
            log.info("%s: rotated to key #%d for round %d",
                     self.display_name, key_idx, self.current_round)

    def _fallback_to_next_key(self) -> bool:
        """Try switching to the next key after a 429. Returns True if switched."""
        next_key = self._pool.next_key(self._api_key)
        if next_key is None:
            return False
        self._api_key = next_key
        self.client = groq.AsyncGroq(api_key=next_key)
        log.warning("%s: 429 fallback — switched to next key", self.display_name)
        return True

    # ------------------------------------------------------------------
    # LLM interface
    # ------------------------------------------------------------------

    async def _do_chat(self, prompt: str) -> LLMResponse:
        """Issue a single chat completion request."""
        resp = await self.client.chat.completions.create(
            model=self._model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": "Respond only with valid JSON. Keep your thinking brief."},
                {"role": "user", "content": prompt},
            ],
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

    async def call_llm(self, prompt: str) -> LLMResponse:
        self._rotate_key_for_round()
        try:
            return await self._do_chat(prompt)
        except groq.RateLimitError as exc:
            retry_after = getattr(exc, "response", None)
            if retry_after is not None:
                retry_after = exc.response.headers.get("retry-after")
            log.warning("%s: Groq 429 (retry-after=%s)", self.display_name, retry_after)
            if self._fallback_to_next_key():
                return await self._do_chat(prompt)
            raise
