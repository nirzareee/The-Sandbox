"""
phase2_ai_agent.py

Phase 2 — AI Agent Client (standalone)
========================================
Connects to the Phase 2 hint server via WebSocket, listens for hint_request
events, generates human-like hints using an LLM provider, and handles the
server's validation feedback (truncation, rejection + retry).

Designed to be used standalone for testing, or imported and extended from
base_agent.py in the full game integration (override _call_llm).

LLM Providers
─────────────
  MockProvider   — no API keys; returns plausible fake hints (use for testing)
  ClaudeProvider — Anthropic Claude via ANTHROPIC_API_KEY env var
  OpenAIProvider — OpenAI GPT-4o via OPENAI_API_KEY env var

Usage examples
──────────────
  # Run a single mock agent:
  python phase2_ai_agent.py --player-id Player_1 --provider mock

  # Run a full mock session (6 agents, no server needed — use with phase2_test.py):
  python phase2_ai_agent.py --mock-session

  # Run with real Claude API:
  ANTHROPIC_API_KEY=... python phase2_ai_agent.py --player-id Player_2 --provider claude
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

logger = logging.getLogger(__name__)

MAX_HINT_CHARS   = 60
PER_HINT_TIMEOUT = 15   # mirrors server constant


# ─── LLM Provider interface ───────────────────────────────────────────────────

class LLMProvider:
    """Abstract base. Subclass and implement generate()."""

    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Call the underlying model.
        Must return a dict:
            {
              "thinking": "private reasoning string",
              "action":   {"type": "hint", "content": "the hint text"}
            }
        """
        raise NotImplementedError


class MockProvider(LLMProvider):
    """
    Simulates an LLM without calling any real API.
    Useful for end-to-end testing of the server-agent protocol.

    Insider hints are relatively specific; imposter hints are generic enough
    to plausibly fit several grid items (PRD Phase 2 spec).
    """

    _INSIDER_HINTS = [
        "tropical",
        "yellow inside",
        "smoothie king",
        "sweet + sour",
        "aamras vibes",
        "not citrus",
        "🌞🏝️",
        "juicy drip",
        "summer staple",
        "goes with chili powder",
        "king of fruits",
        "pulpy goodness",
    ]

    _IMPOSTER_HINTS = [
        "sweet",
        "colorful",
        "summer favorite",
        "fresh pick",
        "popular choice",
        "seasonal",
        "bright colors",
        "natural sugar",
        "fruity",
        "juicy",
    ]

    # Hints that will intentionally trigger the server's rejection rule —
    # used by a small fraction of mock agents to exercise the retry path.
    _SENTENCE_HINTS = [
        "This is the sweetest fruit I know",
        "I would describe it as very tropical",
    ]

    def __init__(self, inject_bad_hint: bool = False) -> None:
        """
        inject_bad_hint: if True, first attempt will be a sentence-like hint
                         to exercise server rejection + client retry.
        """
        self._inject_bad_hint = inject_bad_hint
        self._first_call      = True

    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        # Simulate realistic API latency
        await asyncio.sleep(random.uniform(0.4, 2.0))

        # Detect role from prompt (imposter has "secret_word: None")
        is_imposter = "secret_word: None" in user_prompt or "imposter" in user_prompt.lower()

        if self._inject_bad_hint and self._first_call:
            self._first_call = False
            hint = random.choice(self._SENTENCE_HINTS)
            return {
                "thinking": "Testing rejection path — sending a sentence-like hint.",
                "action":   {"type": "hint", "content": hint},
            }

        self._first_call = False
        hints = self._IMPOSTER_HINTS if is_imposter else self._INSIDER_HINTS
        hint  = random.choice(hints)
        return {
            "thinking": (
                f"Mock reasoning — role={'imposter' if is_imposter else 'insider'}. "
                f"Picking {'generic' if is_imposter else 'specific'} hint."
            ),
            "action": {"type": "hint", "content": hint},
        }


class ClaudeProvider(LLMProvider):
    """Anthropic Claude. Requires: pip install anthropic"""

    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        self.model = model
        try:
            import anthropic  # noqa: F401
            self._anthropic = anthropic
        except ImportError as exc:
            raise RuntimeError("Run: pip install anthropic") from exc
        self._client = self._anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.messages.create(
                model      = self.model,
                max_tokens = 300,
                system     = system_prompt,
                messages   = [{"role": "user", "content": user_prompt}],
            ),
        )
        raw = response.content[0].text
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        import re
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\n?```$", "", cleaned.strip()).strip()
        return json.loads(cleaned)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4o. Requires: pip install openai"""

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        except ImportError as exc:
            raise RuntimeError("Run: pip install openai") from exc

    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        response = await self._client.chat.completions.create(
            model           = self.model,
            max_tokens      = 300,
            response_format = {"type": "json_object"},
            messages        = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return json.loads(response.choices[0].message.content)


# ─── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a player in a multiplayer social deduction game called "The Sandbox."

GAME RULES:
- 6 players (some human, some AI). You don't know who is who.
- Each round, one player is the IMPOSTER; everyone else is an INSIDER.
- Insiders see the topic + secret word. The imposter sees the topic and 10 possible
  words but does NOT know which one is the secret word.
- Each player gives a SHORT HINT. Then open discussion. Then everyone votes.

HINT RULES — CRITICAL:
- Hints MUST be SHORT: 1–4 words max, often just 1–3 words.
- Good examples: "tropical", "yellow inside", "smoothie king", "🌞"
- Bad examples: "This is a tropical fruit" (too long + reads like AI)
- Use slang, emojis, abbreviations, lowercase. Vary your style naturally.
- NEVER start with "This is" or "I would describe".
- NEVER write a dictionary-style definition.

IF YOU ARE THE IMPOSTER:
- You see 10 possible words but don't know the secret one.
- Use other players' hints to narrow it down.
- Give a hint that could plausibly fit several grid items.

RESPONSE FORMAT — always return valid JSON, no markdown fences:
{
  "thinking": "Your private reasoning. Never shown to other players.",
  "action":   {"type": "hint", "content": "your hint here"}
}"""


def build_hint_prompt(
    *,
    topic:       str,
    image_grid:  List[str],
    role:        str,
    secret_word: Optional[str],
    hints_so_far: List[Dict],
    player_id:   str,
    force_short: bool = False,
) -> str:
    """
    Construct the per-turn user prompt for the LLM.

    Parameters
    ----------
    force_short : bool
        Set True on a retry after rejection — injects a strong instruction to
        give an ultra-short hint.
    """
    grid_str  = ", ".join(image_grid)
    word_line = (
        f"Secret word: {secret_word}"
        if secret_word
        else "Secret word: None  ← YOU ARE THE IMPOSTER"
    )
    if hints_so_far:
        hints_text = "\n".join(
            f"  {h['player_id']}: \"{h['hint']}\"" for h in hints_so_far
        )
    else:
        hints_text = "  (none yet — you are first)"

    short_notice = (
        "\n\n⚠️  RETRY: Your previous hint was rejected for being too long. "
        "Give a SINGLE WORD or TWO-WORD hint only."
        if force_short else ""
    )

    return (
        f"CURRENT ROUND\n"
        f"Topic     : {topic}\n"
        f"Grid      : {grid_str}\n"
        f"Your role : {role.upper()}\n"
        f"{word_line}\n\n"
        f"Hints given so far:\n{hints_text}\n\n"
        f"It is your turn ({player_id}). Give your hint now."
        f"{short_notice}\n\n"
        f"Respond with JSON only."
    )


# ─── Human-like delay simulation ─────────────────────────────────────────────

async def simulate_reading_delay(hints_count: int) -> None:
    """
    Pause to simulate a human reading previous hints before typing.
    Later players (more hints to read) take slightly longer.
    Clamped to [1 s, 10 s] to stay within PER_HINT_TIMEOUT.
    """
    base        = random.uniform(1.5, 4.0)
    read_time   = hints_count * random.uniform(0.3, 0.8)
    total       = max(1.0, min(10.0, base + read_time))
    await asyncio.sleep(total)


# ─── Agent state ──────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    player_id:    str
    role:         str            = "insider"
    secret_word:  Optional[str] = None
    topic:        str            = ""
    image_grid:   List[str]     = field(default_factory=list)
    hints_so_far: List[Dict]    = field(default_factory=list)
    thinking_log: List[Dict]    = field(default_factory=list)  # research only


# ─── AI Agent client ──────────────────────────────────────────────────────────

class AIAgentHintClient:
    """
    Standalone AI agent for Phase 2 (Hint Phase).

    Lifecycle
    ─────────
      connect_and_run()
        └─ _handle_message()
              ├─ joined              → store role/secret_word
              ├─ hint_phase_start    → store topic/grid, reset hints_so_far
              ├─ hint_turn           → log whose turn it is
              ├─ hint_request        → _on_hint_request() → generate + submit
              ├─ hint_submitted      → append to local hints_so_far
              ├─ hint_truncated      → log notification
              ├─ hint_rejected       → _on_hint_rejected() → retry with force_short
              ├─ hint_skipped        → log
              └─ hint_phase_complete → log final summary
    """

    def __init__(
        self,
        server_url: str,
        player_id:  str,
        provider:   LLMProvider,
    ) -> None:
        self.server_url = server_url
        self.player_id  = player_id
        self.provider   = provider
        self.state      = AgentState(player_id=player_id)
        self._ws        = None   # websockets connection

    # ── Public entry ─────────────────────────────────────────────────────────

    async def connect_and_run(self) -> None:
        uri = f"{self.server_url}/{self.player_id}"
        logger.info("[%s] Connecting to %s", self.player_id, uri)
        try:
            async with websockets.connect(uri, ping_interval=20) as ws:
                self._ws = ws
                logger.info("[%s] Connected.", self.player_id)
                async for raw in ws:
                    try:
                        await self._handle_message(json.loads(raw))
                    except json.JSONDecodeError:
                        logger.warning("[%s] Non-JSON: %r", self.player_id, raw[:80])
                    except Exception as exc:
                        logger.error("[%s] Handler error: %s", self.player_id, exc, exc_info=True)
        except (ConnectionClosedOK, ConnectionClosedError):
            logger.info("[%s] Connection closed.", self.player_id)
        except Exception as exc:
            logger.error("[%s] Connection error: %s", self.player_id, exc)
        finally:
            self._ws = None

    # ── Event dispatch ────────────────────────────────────────────────────────

    async def _handle_message(self, msg: dict) -> None:
        event = msg.get("event", "")
        handlers = {
            "joined":              self._on_joined,
            "hint_phase_start":    self._on_hint_phase_start,
            "hint_turn":           self._on_hint_turn,
            "hint_request":        self._on_hint_request,
            "hint_submitted":      self._on_hint_submitted,
            "hint_truncated":      self._on_hint_truncated,
            "hint_rejected":       self._on_hint_rejected,
            "hint_skipped":        self._on_hint_skipped,
            "hint_phase_complete": self._on_hint_phase_complete,
        }
        handler = handlers.get(event)
        if handler:
            await handler(msg)
        else:
            logger.debug("[%s] Unhandled event: %s", self.player_id, event)

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _on_joined(self, msg: dict) -> None:
        self.state.role        = msg.get("role", "insider")
        self.state.secret_word = msg.get("secret_word")
        waiting                = msg.get("waiting_for", "?")
        logger.info(
            "[%s] Joined as %s. Secret: %s. Waiting for %s more.",
            self.player_id,
            self.state.role,
            self.state.secret_word or "??? (imposter)",
            waiting,
        )

    async def _on_hint_phase_start(self, msg: dict) -> None:
        self.state.topic      = msg.get("topic", "")
        self.state.image_grid = msg.get("image_grid", [])
        self.state.hints_so_far.clear()
        order    = msg.get("hint_order", [])
        my_pos   = order.index(self.player_id) + 1 if self.player_id in order else "?"
        logger.info(
            "[%s] Hint phase started. Topic: %s. My turn: %s/%d.",
            self.player_id, self.state.topic, my_pos, len(order),
        )

    async def _on_hint_turn(self, msg: dict) -> None:
        current = msg.get("player_id")
        pos     = msg.get("position")
        total   = msg.get("total")
        if current != self.player_id:
            logger.info("[%s] Waiting — it's %s's turn (%s/%s).", self.player_id, current, pos, total)

    async def _on_hint_request(self, msg: dict) -> None:
        """Server is requesting our hint. Simulate reading, generate, submit."""
        # Refresh state from server payload (authoritative)
        self.state.topic       = msg.get("topic",       self.state.topic)
        self.state.image_grid  = msg.get("image_grid",  self.state.image_grid)
        self.state.hints_so_far = msg.get("hints_so_far", self.state.hints_so_far)
        if msg.get("secret_word") is not None:
            self.state.secret_word = msg["secret_word"]

        logger.info("[%s] Hint requested. Timeout: %ss.", self.player_id, msg.get("timeout"))
        await simulate_reading_delay(len(self.state.hints_so_far))
        hint = await self._generate_hint(force_short=False)
        await self._submit_hint(hint)

    async def _on_hint_submitted(self, msg: dict) -> None:
        """Record a hint submitted by ANY player (including ourselves)."""
        submitter = msg.get("player_id")
        if submitter != self.player_id:
            self.state.hints_so_far.append({
                "player_id": submitter,
                "hint":      msg.get("hint", ""),
            })
            logger.info(
                "[%s] ← %s: \"%s\"",
                self.player_id, submitter, msg.get("hint"),
            )

    async def _on_hint_truncated(self, msg: dict) -> None:
        logger.info(
            "[%s] Our hint was truncated: \"%s\" → \"%s\".",
            self.player_id, msg.get("original"), msg.get("accepted"),
        )

    async def _on_hint_rejected(self, msg: dict) -> None:
        """Server rejected our hint. Regenerate with force_short=True."""
        logger.warning(
            "[%s] Hint rejected: \"%s\". Reason: %s. Retry: %s.",
            self.player_id,
            msg.get("original"),
            msg.get("reason"),
            msg.get("retry"),
        )
        if msg.get("retry"):
            await asyncio.sleep(random.uniform(0.5, 1.5))  # brief re-think
            hint = await self._generate_hint(force_short=True)
            await self._submit_hint(hint)

    async def _on_hint_skipped(self, msg: dict) -> None:
        pid = msg.get("player_id")
        if pid == self.player_id:
            logger.warning("[%s] Our turn was skipped (timeout).", self.player_id)
        else:
            logger.info("[%s] %s was skipped.", self.player_id, pid)

    async def _on_hint_phase_complete(self, msg: dict) -> None:
        logger.info("[%s] ── Hint phase complete ──", self.player_id)
        for h in msg.get("hints", []):
            marker = " ← me" if h["player_id"] == self.player_id else ""
            logger.info(
                "  %s (%s): \"%s\"%s",
                h["player_id"], h["role"], h["hint"], marker,
            )

    # ── Hint generation ───────────────────────────────────────────────────────

    async def _generate_hint(self, force_short: bool = False) -> str:
        """
        Build a prompt, call the LLM with a 15-second timeout, parse the result.
        Falls back to a safe short default on any error.
        """
        prompt = build_hint_prompt(
            topic        = self.state.topic,
            image_grid   = self.state.image_grid,
            role         = self.state.role,
            secret_word  = self.state.secret_word,
            hints_so_far = self.state.hints_so_far,
            player_id    = self.player_id,
            force_short  = force_short,
        )

        try:
            response = await asyncio.wait_for(
                self.provider.generate(SYSTEM_PROMPT, prompt),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[%s] LLM timed out — using fallback hint.", self.player_id)
            return self._fallback_hint()
        except Exception as exc:
            logger.error("[%s] LLM error: %s", self.player_id, exc)
            return self._fallback_hint()

        # Log thinking trace for research (never sent to server)
        thinking = response.get("thinking", "")
        self.state.thinking_log.append({
            "timestamp":   time.time(),
            "thinking":    thinking,
            "action":      response.get("action", {}),
            "force_short": force_short,
        })
        if thinking:
            logger.debug("[%s] Thinking: %s", self.player_id, thinking[:120])

        hint = response.get("action", {}).get("content", "")
        if not hint:
            hint = self._fallback_hint()

        logger.info("[%s] Generated hint: \"%s\"", self.player_id, hint)
        return hint

    def _fallback_hint(self) -> str:
        """Safe 1-word default if the LLM call fails entirely."""
        return random.choice(["hmm", "nice", "this", "🤔", "tasty"])

    # ── Outbound ──────────────────────────────────────────────────────────────

    async def _submit_hint(self, content: str) -> None:
        if self._ws is None:
            logger.error("[%s] Cannot submit hint — not connected.", self.player_id)
            return
        payload = json.dumps({"type": "hint", "content": content})
        await self._ws.send(payload)
        logger.info("[%s] → Submitted: \"%s\"", self.player_id, content)


# ─── Session runner ───────────────────────────────────────────────────────────

async def run_mock_session(
    server_url: str = "ws://localhost:8000/ws",
    num_agents: int = REQUIRED_PLAYERS,
) -> None:
    """
    Spin up `num_agents` MockProvider agents and connect them to the server.
    One agent (Player_2) has inject_bad_hint=True to exercise rejection + retry.
    Stagger connections by 300 ms to simulate human join timing.
    """
    from phase2_hint_server import REQUIRED_PLAYERS  # noqa: F401

    agents: List[AIAgentHintClient] = []
    for i in range(num_agents):
        inject = (i == 1)   # second agent will trigger rejection path
        agents.append(
            AIAgentHintClient(
                server_url = server_url,
                player_id  = f"Player_{i + 1}",
                provider   = MockProvider(inject_bad_hint=inject),
            )
        )

    tasks: List[asyncio.Task] = []
    for agent in agents:
        await asyncio.sleep(0.3)
        tasks.append(asyncio.create_task(agent.connect_and_run()))

    await asyncio.gather(*tasks, return_exceptions=True)


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Phase 2 AI Agent Client")
    parser.add_argument("--server",       default="ws://localhost:8000/ws",
                        help="WebSocket server URL prefix")
    parser.add_argument("--player-id",    default="Player_1")
    parser.add_argument("--provider",     choices=["mock", "claude", "openai"], default="mock")
    parser.add_argument("--inject-bad",   action="store_true",
                        help="Send a sentence-like hint first (test rejection path)")
    parser.add_argument("--mock-session", action="store_true",
                        help="Launch 6 mock agents (requires server to be running)")
    args = parser.parse_args()

    if args.mock_session:
        asyncio.run(run_mock_session(args.server))
    else:
        if args.provider == "claude":
            prov: LLMProvider = ClaudeProvider()
        elif args.provider == "openai":
            prov = OpenAIProvider()
        else:
            prov = MockProvider(inject_bad_hint=args.inject_bad)

        agent = AIAgentHintClient(args.server, args.player_id, prov)
        asyncio.run(agent.connect_and_run())
