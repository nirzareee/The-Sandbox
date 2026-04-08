"""
phase5_reflection_agent.py

Phase 5 — AI Agent Reflection Client (standalone)
==================================================
Connects to the Phase 5 reflection server via WebSocket, waits for
reflection_start, performs private reasoning about the round, and broadcasts
a structured reflection covering:

  • How each player's hint related to the grid vs the secret word
  • Whether any hint pattern suggests grid-guessing (imposter behaviour)
  • META-GAME analysis: who seems human vs AI based on hint style

LLM Providers
─────────────
  MockProvider   — no API keys; returns plausible fake reflection (use for testing)
  ClaudeProvider — Anthropic Claude via ANTHROPIC_API_KEY env var
  OpenAIProvider — OpenAI GPT-4o via OPENAI_API_KEY env var

Usage examples
──────────────
  # Single mock AI agent:
  python phase5_reflection_agent.py --player-id Player_1 --provider mock

  # Full mock session (6 agents):
  python phase5_reflection_agent.py --mock-session

  # With real Claude API:
  ANTHROPIC_API_KEY=... python phase5_reflection_agent.py --player-id Player_1 --provider claude
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

REFLECTION_DURATION = 90   # mirrors server constant


# ─── LLM Provider interface ───────────────────────────────────────────────────

class LLMProvider:
    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        raise NotImplementedError


class MockProvider(LLMProvider):
    """
    Returns plausible hard-coded reflections without calling any real API.
    Randomly picks from a bank of mock analysis responses.
    """

    _GRID_ANALYSES = [
        (
            "Player_1's hint 'tropical' is very specific to Mango — strong insider signal. "
            "Player_3's 'yellow inside' is also precise. "
            "Player_4's 'aamras vibes' (a mango-based Indian drink) is highly targeted. "
            "Player_2's 'sweet' and Player_5's 'colorful' could fit almost any grid item — "
            "they might be guessing from the grid rather than knowing the secret word."
        ),
        (
            "Most hints cluster tightly around Mango properties (tropical, yellow inside, 🌞🏝️). "
            "However 'sweet' and 'colorful' are suspiciously broad — both apply equally to "
            "Strawberry, Watermelon, and Apple. These players may not know the secret word."
        ),
    ]

    _META_ANALYSES = [
        (
            "META-GAME: Player_4's 'aamras vibes' uses a cultural reference that feels very human — "
            "an AI rarely invokes regional food contexts unprompted. "
            "Player_6's emoji-only hint 🌞🏝️ could go either way. "
            "Player_2 and Player_5's generic hints ('sweet', 'colorful') read more like AI fallbacks — "
            "safe enough to fit anything, but lacking personality."
        ),
        (
            "META-GAME: Emoji-only hints and culturally specific references lean human. "
            "Generic adjectives like 'sweet' or 'colorful' with no flair lean AI. "
            "Player_1's lowercase 'tropical' with no punctuation could be either. "
            "I'd flag Player_2 and Player_5 as likely AI based on hint blandness."
        ),
    ]

    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        await asyncio.sleep(random.uniform(2.0, 8.0))   # simulate thinking time
        grid_analysis = random.choice(self._GRID_ANALYSES)
        meta_analysis = random.choice(self._META_ANALYSES)
        return {
            "thinking": "Analysing each hint against the secret word and grid...",
            "reflection": {
                "grid_analysis": grid_analysis,
                "meta_analysis": meta_analysis,
            },
        }


class ClaudeProvider(LLMProvider):
    """Anthropic Claude. Requires: pip install anthropic"""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        except ImportError as exc:
            raise RuntimeError("Run: pip install anthropic") from exc

    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        import re
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.messages.create(
                model      = self.model,
                max_tokens = 600,
                system     = system_prompt,
                messages   = [{"role": "user", "content": user_prompt}],
            ),
        )
        raw = response.content[0].text
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
            max_tokens      = 600,
            response_format = {"type": "json_object"},
            messages        = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return json.loads(response.choices[0].message.content)


# ─── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI player in a multiplayer social deduction game called "The Sandbox."

GAME RECAP:
- 6 players (some human, some AI). Identities are hidden.
- Each round: one IMPOSTER doesn't know the secret word; everyone else (INSIDERS) does.
- Players give short hints. Then discussion. Then voting to find the imposter.

YOUR TASK — BETWEEN-ROUND REFLECTION:
You now see the full round result (topic, grid, secret word, all hints + roles revealed).
Write a reflection covering TWO parts:

PART 1 — GRID ANALYSIS:
- For each player's hint, assess how well it targets the secret word vs being vague enough
  to fit multiple grid items.
- Flag any hints that seem like the player was guessing from the grid (imposter pattern).

PART 2 — META-GAME ANALYSIS:
- Based ONLY on hint STYLE (length, vocabulary, creativity, cultural references, emoji use,
  punctuation), estimate who among the players is most likely human vs AI.
- Do NOT use the revealed roles for this — pretend you don't see them.
- Give a brief reasoning for each player.

RESPONSE FORMAT — valid JSON only, no markdown:
{
  "thinking": "your private reasoning chain",
  "reflection": {
    "grid_analysis": "paragraph about how hints relate to secret word vs grid",
    "meta_analysis": "paragraph about who seems human vs AI based on style"
  }
}"""


def build_reflection_prompt(
    *,
    round_number: int,
    topic:        str,
    secret_word:  str,
    image_grid:   List[str],
    hints:        List[Dict],
    player_id:    str,
) -> str:
    grid_str   = ", ".join(image_grid)
    hints_text = "\n".join(
        f"  {h['player_id']}: \"{h['hint']}\"  [role: {h.get('role', 'hidden')}]"
        for h in hints
    )
    return (
        f"ROUND {round_number} RESULTS\n"
        f"Topic      : {topic}\n"
        f"Grid       : [{grid_str}]\n"
        f"Secret word: {secret_word}\n\n"
        f"Hints submitted:\n{hints_text}\n\n"
        f"You are {player_id}. Write your reflection now.\n"
        f"Respond with JSON only."
    )


# ─── Agent state ──────────────────────────────────────────────────────────────

@dataclass
class ReflectionAgentState:
    player_id:    str
    round_number: int            = 0
    topic:        str            = ""
    secret_word:  str            = ""
    image_grid:   List[str]      = field(default_factory=list)
    hints:        List[Dict]     = field(default_factory=list)
    thinking_log: List[Dict]     = field(default_factory=list)


# ─── AI Agent client ──────────────────────────────────────────────────────────

class AIReflectionAgent:
    """
    AI agent for Phase 5 (Between-Round Reflection).

    Lifecycle
    ─────────
      connect_and_run()
        └─ _handle_message()
              ├─ joined               → log connection
              ├─ reflection_start     → store context, generate + submit reflection
              ├─ reflection_submitted → log another agent's reflection
              ├─ chat_message         → log chat
              ├─ dm_message           → log DM
              └─ reflection_complete  → log final summary
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
        self.state      = ReflectionAgentState(player_id=player_id)
        self._ws        = None

    # ── Public entry ─────────────────────────────────────────────────────────

    async def connect_and_run(self) -> None:
        uri = f"{self.server_url}/{self.player_id}?type=ai"
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
            "joined":               self._on_joined,
            "reflection_start":     self._on_reflection_start,
            "reflection_submitted": self._on_reflection_submitted,
            "chat_message":         self._on_chat_message,
            "dm_message":           self._on_dm_message,
            "reflection_complete":  self._on_reflection_complete,
        }
        handler = handlers.get(event)
        if handler:
            await handler(msg)
        else:
            logger.debug("[%s] Unhandled event: %s", self.player_id, event)

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _on_joined(self, msg: dict) -> None:
        logger.info(
            "[%s] Joined. Waiting for %s more players.",
            self.player_id, msg.get("waiting_for", "?"),
        )

    async def _on_reflection_start(self, msg: dict) -> None:
        self.state.round_number = msg.get("round", 0)
        self.state.topic        = msg.get("topic", "")
        self.state.secret_word  = msg.get("secret_word", "")
        self.state.image_grid   = msg.get("image_grid", [])
        self.state.hints        = msg.get("hints", [])
        duration                = msg.get("duration", REFLECTION_DURATION)

        logger.info(
            "[%s] Reflection started — Round %d, Topic: %s, Secret: %s (%ds window).",
            self.player_id, self.state.round_number,
            self.state.topic, self.state.secret_word, duration,
        )

        # Stagger reflections so all agents don't respond simultaneously
        delay = random.uniform(5.0, 30.0)
        logger.info("[%s] Thinking for %.1fs before submitting reflection.", self.player_id, delay)
        await asyncio.sleep(delay)

        reflection_text = await self._generate_reflection()
        await self._submit_reflection(reflection_text)

    async def _on_reflection_submitted(self, msg: dict) -> None:
        submitter = msg.get("player_id")
        if submitter != self.player_id:
            preview = msg.get("content", "")[:80]
            logger.info("[%s] ← %s reflection: %s...", self.player_id, submitter, preview)

    async def _on_chat_message(self, msg: dict) -> None:
        sender  = msg.get("player_id")
        content = msg.get("content", "")
        logger.info("[%s] [CHAT] %s: %s", self.player_id, sender, content[:80])

    async def _on_dm_message(self, msg: dict) -> None:
        sender    = msg.get("from")
        recipient = msg.get("to")
        content   = msg.get("content", "")
        if recipient == self.player_id:
            logger.info("[%s] [DM] ← %s: %s", self.player_id, sender, content[:80])
        else:
            logger.info("[%s] [DM] → %s: %s", self.player_id, recipient, content[:80])

    async def _on_reflection_complete(self, msg: dict) -> None:
        logger.info("[%s] ── Reflection phase complete ──", self.player_id)
        for r in msg.get("reflections", []):
            marker  = " ← me" if r["player_id"] == self.player_id else ""
            preview = r["content"][:100].replace("\n", " ")
            logger.info("  %s: %s...%s", r["player_id"], preview, marker)

    # ── Reflection generation ─────────────────────────────────────────────────

    async def _generate_reflection(self) -> str:
        prompt = build_reflection_prompt(
            round_number = self.state.round_number,
            topic        = self.state.topic,
            secret_word  = self.state.secret_word,
            image_grid   = self.state.image_grid,
            hints        = self.state.hints,
            player_id    = self.player_id,
        )

        try:
            response = await asyncio.wait_for(
                self.provider.generate(SYSTEM_PROMPT, prompt),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[%s] LLM timed out — using fallback reflection.", self.player_id)
            return self._fallback_reflection()
        except Exception as exc:
            logger.error("[%s] LLM error: %s", self.player_id, exc)
            return self._fallback_reflection()

        thinking = response.get("thinking", "")
        self.state.thinking_log.append({
            "timestamp": time.time(),
            "thinking":  thinking,
        })
        if thinking:
            logger.debug("[%s] Thinking: %s", self.player_id, thinking[:120])

        reflection = response.get("reflection", {})
        grid_part  = reflection.get("grid_analysis", "")
        meta_part  = reflection.get("meta_analysis", "")

        if not grid_part and not meta_part:
            return self._fallback_reflection()

        combined = ""
        if grid_part:
            combined += f"[Grid Analysis] {grid_part}"
        if meta_part:
            if combined:
                combined += "\n\n"
            combined += f"[Meta-Game] {meta_part}"

        logger.info("[%s] Generated reflection (%d chars).", self.player_id, len(combined))
        return combined

    def _fallback_reflection(self) -> str:
        return (
            "[Grid Analysis] Hard to tell from hints alone — some were very specific, others vague.\n\n"
            "[Meta-Game] Could not determine human vs AI with confidence this round."
        )

    # ── Outbound ──────────────────────────────────────────────────────────────

    async def _submit_reflection(self, content: str) -> None:
        if self._ws is None:
            logger.error("[%s] Cannot submit reflection — not connected.", self.player_id)
            return
        payload = json.dumps({"type": "reflection", "content": content})
        await self._ws.send(payload)
        logger.info("[%s] → Submitted reflection.", self.player_id)

    async def send_chat(self, content: str) -> None:
        """Optional: AI agent sends a free-form chat message."""
        if self._ws is None:
            return
        await self._ws.send(json.dumps({"type": "chat", "content": content}))

    async def send_dm(self, to: str, content: str) -> None:
        """Optional: AI agent sends a DM to a specific player."""
        if self._ws is None:
            return
        await self._ws.send(json.dumps({"type": "dm", "to": to, "content": content}))


# ─── Session runner ───────────────────────────────────────────────────────────

async def run_mock_session(
    server_url: str = "ws://localhost:8001/ws",
    num_agents: int = REQUIRED_PLAYERS,
) -> None:
    """
    Spin up `num_agents` MockProvider agents and connect them to the server.
    Stagger connections by 300 ms to simulate human join timing.
    """
    agents: List[AIReflectionAgent] = [
        AIReflectionAgent(
            server_url = server_url,
            player_id  = f"Player_{i + 1}",
            provider   = MockProvider(),
        )
        for i in range(num_agents)
    ]

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

    parser = argparse.ArgumentParser(description="Phase 5 AI Reflection Agent")
    parser.add_argument("--server",       default="ws://localhost:8001/ws",
                        help="WebSocket server URL prefix")
    parser.add_argument("--player-id",    default="Player_1")
    parser.add_argument("--provider",     choices=["mock", "claude", "openai"], default="mock")
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
            prov = MockProvider()

        agent = AIReflectionAgent(args.server, args.player_id, prov)
        asyncio.run(agent.connect_and_run())
