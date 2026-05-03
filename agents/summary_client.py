"""Dedicated Groq client for LLM-based round summarization.

Uses a separate API key (GROQ_API_KEY_SUMMARY) so summarization calls
don't compete with gameplay API calls for rate limits.
"""

from __future__ import annotations

import logging
import os

import groq

log = logging.getLogger(__name__)

_SUMMARY_MODEL = "llama-3.1-8b-instant"

_SUMMARY_PROMPT_TEMPLATE = """\
You are an AI agent's memory system in a social deduction game. Summarize this round \
in 4-6 sentences for the agent to reference in future rounds.

Focus on:
- Social dynamics: who accused whom, who defended whom, who was quiet
- Deals and alliances: any promises made in DMs, whether they were kept (vote coordination)
- Suspicious behavior: whose hints didn't fit, who deflected, who changed their story
- Strategic takeaways: who to trust, who to watch, what worked/didn't

Round {round_number}: Topic "{topic}" (Secret word: {secret_word}, My role: {role})

HINTS:
{hints}

GROUP CHAT:
{messages}

MY PRIVATE DMs:
{dms}

VOTES: {votes}
RESULT: {result}

Write a concise strategic summary. No bullet points — flowing prose. Max 150 words."""


_THINKING_SUMMARY_PROMPT_TEMPLATE = """\
You are summarizing an AI agent's INTERNAL REASONING from one round of a social deduction game. \
The agent will read this summary in FUTURE rounds to recall what it was thinking. \
Write in first-person past tense ("I noticed", "I thought", "I suspected") so the agent recognizes it as their own reasoning.

Capture (in flowing prose, 100-150 words):
- Key suspicions formed about other players (and why)
- Theories that evolved or were abandoned mid-round
- Strategic decisions made and the reasoning behind them
- Any contradictions or shifts in thinking — moments where you accepted then doubted, or doubted then accepted
- Subtle signals you picked up on (timing, tone, deflection, vote patterns)

Round {round_number}. My role: {role}.

THINKING TRACES (in chronological order, one per phase):
{traces}

Write the first-person summary now. Max 150 words. Flowing prose, no bullets."""


class _ThinkingSummaryMixin:
    """Adds thinking-trace summarization to SummaryClient (mixed in below)."""

    async def summarize_thinking(
        self, round_number: int, role: str, thinking_traces: list[str],
    ) -> str | None:
        """Summarize a list of thinking traces from one round.

        Returns the summary string, or None on error / empty traces.
        """
        if not thinking_traces:
            return None
        traces_str = "\n".join(
            f"  [phase {i + 1}] {t}" for i, t in enumerate(thinking_traces) if t
        ) or "(none)"

        prompt = _THINKING_SUMMARY_PROMPT_TEMPLATE.format(
            round_number=round_number,
            role=role,
            traces=traces_str,
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self._model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = resp.choices[0].message.content.strip()
            log.info("LLM thinking summary R%d (%d chars)", round_number, len(summary))
            return summary
        except Exception as exc:  # noqa: BLE001
            log.warning("Thinking summary LLM call failed: %s", exc)
            return None


class SummaryClient(_ThinkingSummaryMixin):
    """Async Groq client dedicated to round and thinking summarization."""

    def __init__(self, api_key: str) -> None:
        self.client = groq.AsyncGroq(api_key=api_key)
        self._model = _SUMMARY_MODEL

    async def summarize_round(self, round_data: dict) -> str | None:
        """Summarize a round's data into a strategic narrative.

        Returns the summary string, or None on any error (caller falls
        back to structured extraction).
        """
        hints = round_data.get("hints", {})
        hints_str = "\n".join(f"  {pid}: \"{h}\"" for pid, h in hints.items()) or "(none)"

        messages = round_data.get("messages", [])
        msg_lines = []
        for m in messages[-20:]:  # cap to avoid huge prompts
            msg_lines.append(f"  {m.get('player_id', '?')}: {m.get('content', '')}")
        messages_str = "\n".join(msg_lines) or "(none)"

        dms = round_data.get("dms", [])
        dm_lines = []
        for dm in dms:
            dm_lines.append(
                f"  [{dm.get('sender', '?')} -> {dm.get('recipient', '?')}]: "
                f"{dm.get('content', '')}"
            )
        dms_str = "\n".join(dm_lines) or "(none)"

        votes = round_data.get("votes", {})
        votes_str = ", ".join(f"{v}->{t}" for v, t in votes.items()) or "(none)"

        result = round_data.get("result", {})
        imp = result.get("imposter_id", "?")
        caught = result.get("imposter_caught", False)
        result_str = f"{imp} was the imposter ({'caught' if caught else 'survived'})"

        prompt = _SUMMARY_PROMPT_TEMPLATE.format(
            round_number=round_data.get("round_number", "?"),
            topic=round_data.get("topic", "?"),
            secret_word=round_data.get("secret_word", "UNKNOWN"),
            role=round_data.get("role", "?"),
            hints=hints_str,
            messages=messages_str,
            dms=dms_str,
            votes=votes_str,
            result=result_str,
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self._model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = resp.choices[0].message.content.strip()
            log.info("LLM round summary (%d chars)", len(summary))
            return summary
        except Exception as exc:  # noqa: BLE001
            log.warning("Summary LLM call failed, falling back to structured: %s", exc)
            return None


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_client: SummaryClient | None = None
_checked = False


def get_summary_client() -> SummaryClient | None:
    """Return the shared summary client, or None if GROQ_API_KEY_SUMMARY is not set."""
    global _client, _checked
    if not _checked:
        _checked = True
        key = os.getenv("GROQ_API_KEY_SUMMARY", "")
        if key:
            _client = SummaryClient(key)
            log.info("SummaryClient initialized with dedicated key")
        else:
            log.info("GROQ_API_KEY_SUMMARY not set — using structured fallback for round summaries")
    return _client
