"""
Full-game LLM agent for The Sandbox.

Connects to the game server as an AI WebSocket player and handles every phase:
  Assignment → Hint → Discussion → Voting → Classification → Survey

One instance per agent. Spawn via: asyncio.create_task(agent.run())
"""
from __future__ import annotations

import asyncio
import json
import logging
import random

import websockets

from .providers import LLMProvider
from .prompts import (
    SYSTEM_PROMPT,
    hint_prompt,
    discussion_prompt,
    vote_prompt,
    classification_prompt,
)

logger = logging.getLogger(__name__)


class LLMGameAgent:
    def __init__(
        self,
        server_url: str,          # e.g. "ws://127.0.0.1:7777"
        agent_id: str,
        provider: LLMProvider,
        name: str | None = None,
        timer_multiplier: float = 1.0,
    ):
        self.server_url       = server_url
        self.agent_id         = agent_id
        self.provider         = provider
        self.display_name     = name or agent_id
        self.timer_mul        = timer_multiplier

        # ── Per-round state ──────────────────────────────────────────
        self.role:         str        = "insider"
        self.secret_word:  str | None = None
        self.topic:        str        = ""
        self.grid:         list[str]  = []
        self.hints:        list[dict] = []    # [{player_id, hint}] this round
        self.chat_history: list[dict] = []    # [{sender, content}] this round
        self.hint_submitted  = False
        self.vote_submitted  = False
        self.roster: dict[str, str] = {}      # pid → display_name

        # ── Cross-round memory for classification ────────────────────
        self.all_hints: dict[str, list[str]] = {}   # pid → hints all rounds
        self.all_chat:  dict[str, list[str]] = {}   # pid → messages all rounds

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._discussion_task: asyncio.Task | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def run(self):
        from urllib.parse import quote
        uri = f"{self.server_url}/ws/{self.agent_id}?name={quote(self.display_name)}&type=AI"
        logger.info("[%s/%s] Connecting to %s", self.provider.name, self.agent_id, uri)
        try:
            async with websockets.connect(uri) as ws:
                self._ws = ws
                async for raw in ws:
                    try:
                        await self._dispatch(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
                    except Exception as exc:
                        logger.error("[%s] Handler error: %s", self.agent_id, exc, exc_info=True)
        except Exception as exc:
            logger.error("[%s] Connection error: %s", self.agent_id, exc)
        finally:
            self._ws = None
            if self._discussion_task and not self._discussion_task.done():
                self._discussion_task.cancel()

    # ── Event dispatch ────────────────────────────────────────────────────────

    async def _dispatch(self, msg: dict):
        event = msg.get("event", "")
        data  = msg.get("data", {})

        handlers = {
            "player_roster":   self._on_roster,
            "round_start":     self._on_round_start,
            "hint_phase_start":self._on_hint_phase_start,
            "hint_submitted":  self._on_hint_submitted,
            "discussion_start":self._on_discussion_start,
            "group_message":   self._on_group_message,
            "dm":              self._on_dm,
            "discussion_end":  self._on_discussion_end,
            "voting_start":    self._on_voting_start,
            "meta_game_start": self._on_meta_game_start,
            "game_finished":   self._on_game_finished,
        }
        handler = handlers.get(event)
        if handler:
            await handler(data)

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _on_roster(self, data: dict):
        self.roster = data.get("roster", {})
        self.display_name = self.roster.get(self.agent_id, self.display_name)

    async def _on_round_start(self, data: dict):
        self.role        = data.get("role", "insider")
        self.secret_word = data.get("secret_word")   # None if imposter
        self.topic       = data.get("topic", "")
        self.grid        = data.get("image_grid", [])
        self.hints       = []
        self.chat_history= []
        self.hint_submitted = False
        self.vote_submitted = False
        logger.info("[%s] Round start — role=%s topic=%s", self.agent_id, self.role, self.topic)

    async def _on_hint_phase_start(self, data: dict):
        if not self.hint_submitted:
            delay = random.uniform(3, 9) * self.timer_mul
            await asyncio.sleep(delay)
            await self._submit_hint()

    async def _on_hint_submitted(self, data: dict):
        pid  = data.get("player_id", "")
        hint = data.get("hint", "")
        if pid:
            self.hints.append({"player_id": pid, "hint": hint})
            self.all_hints.setdefault(pid, []).append(hint)

    async def _on_discussion_start(self, data: dict):
        duration = data.get("duration", 180)
        self._discussion_task = asyncio.create_task(
            self._run_discussion(duration)
        )

    async def _on_group_message(self, data: dict):
        sender  = data.get("player_id", "")
        content = data.get("content", "")
        self.chat_history.append({"sender": sender, "content": content})
        self.all_chat.setdefault(sender, []).append(content)

    async def _on_dm(self, data: dict):
        sender  = data.get("sender", "")
        content = data.get("content", "")
        if sender and sender != self.agent_id:
            asyncio.create_task(self._reply_dm(sender, content))

    async def _on_discussion_end(self, _data: dict):
        if self._discussion_task and not self._discussion_task.done():
            self._discussion_task.cancel()

    async def _on_voting_start(self, data: dict):
        if not self.vote_submitted:
            delay = random.uniform(4, 11) * self.timer_mul
            await asyncio.sleep(delay)
            candidates = [c for c in data.get("candidates", []) if c != self.agent_id]
            await self._submit_vote(candidates)

    async def _on_meta_game_start(self, data: dict):
        delay = random.uniform(3, 8) * self.timer_mul
        await asyncio.sleep(delay)
        await self._submit_classification(data.get("players", {}))

    async def _on_game_finished(self, _data: dict):
        logger.info("[%s] Game finished.", self.agent_id)

    # ── Outbound helper ───────────────────────────────────────────────────────

    async def _send(self, payload: dict):
        if self._ws:
            try:
                await self._ws.send(json.dumps(payload))
            except Exception as exc:
                logger.warning("[%s] Send failed: %s", self.agent_id, exc)

    # ── Hint ──────────────────────────────────────────────────────────────────

    async def _submit_hint(self, retry: bool = False):
        prompt = hint_prompt(
            role=self.role,
            topic=self.topic,
            grid=self.grid,
            secret_word=self.secret_word,
            hints_so_far=self.hints,
            player_name=self.display_name,
            retry=retry,
        )
        hint = await self._llm_str(prompt, key="hint",
                                   fallback=random.choice(["interesting", "classic", "seasonal"]),
                                   timeout=15.0)
        # Word-count safety
        words = hint.split()
        if len(words) > 8:
            hint = " ".join(words[:4])

        await self._send({"action": "HINT", "content": hint})
        self.hint_submitted = True
        logger.info("[%s] Hint → %r", self.agent_id, hint)

    # ── Discussion ────────────────────────────────────────────────────────────

    async def _run_discussion(self, duration: int):
        num_messages = random.randint(3, 5)
        for i in range(num_messages):
            # Spread messages through the discussion window
            if i == 0:
                delay = random.uniform(2, 7)
            elif i == num_messages - 1:
                delay = random.uniform(10, 20)
            else:
                delay = random.uniform(7, 16)

            try:
                await asyncio.sleep(delay * self.timer_mul)
            except asyncio.CancelledError:
                return

            prompt = discussion_prompt(
                role=self.role,
                topic=self.topic,
                grid=self.grid,
                secret_word=self.secret_word,
                hints=self.hints,
                chat_history=self.chat_history,
                player_name=self.display_name,
            )
            message = await self._llm_str(prompt, key="message", fallback="", timeout=12.0)
            if not message:
                continue
            message = message[:200]  # server hard limit

            try:
                await self._send({"action": "GROUP_MESSAGE", "content": message})
                logger.info("[%s] Chat → %r", self.agent_id, message)
            except asyncio.CancelledError:
                return

    # ── DM reply ──────────────────────────────────────────────────────────────

    async def _reply_dm(self, sender: str, content: str):
        await asyncio.sleep(random.uniform(2, 5) * self.timer_mul)
        prompt = (
            f'Someone sent you a private message: "{content}"\n'
            f"You are a {self.role} in a social deduction game. Write a short casual reply.\n"
            f'Respond JSON: {{"reply": "..."}}'
        )
        reply = await self._llm_str(prompt, key="reply",
                                    fallback="yeah interesting point",
                                    timeout=8.0)
        await self._send({"action": "DM", "target_player": sender, "content": reply[:200]})

    # ── Vote ──────────────────────────────────────────────────────────────────

    async def _submit_vote(self, candidates: list[str]):
        if not candidates:
            return
        prompt = vote_prompt(
            role=self.role,
            topic=self.topic,
            hints=self.hints,
            chat_history=self.chat_history,
            candidates=candidates,
            player_name=self.display_name,
        )
        target = await self._llm_str(prompt, key="vote",
                                     fallback=random.choice(candidates),
                                     timeout=12.0)
        if target not in candidates:
            target = random.choice(candidates)

        await self._send({"action": "VOTE", "target_player": target})
        self.vote_submitted = True
        logger.info("[%s] Vote → %s", self.agent_id, target)

    # ── Classification ────────────────────────────────────────────────────────

    async def _submit_classification(self, players: dict):
        player_names = {
            pid: info.get("display_name", pid)
            for pid, info in players.items()
            if pid != self.agent_id
        }
        if not player_names:
            return

        prompt = classification_prompt(
            players=player_names,
            hints_by_player=self.all_hints,
            chat_by_player=self.all_chat,
            player_name=self.display_name,
        )
        try:
            result = await asyncio.wait_for(
                self.provider.complete_json(SYSTEM_PROMPT, prompt, max_tokens=600),
                timeout=20.0,
            )
            classifications = {}
            for pid in player_names:
                entry = result.get(pid, {})
                if isinstance(entry, dict) and "guess" in entry:
                    classifications[pid] = {
                        "guess":      entry.get("guess", "human"),
                        "confidence": entry.get("confidence", 3),
                        "reasoning":  entry.get("reasoning", ""),
                    }
                else:
                    classifications[pid] = {
                        "guess": "human", "confidence": 2, "reasoning": "unsure"
                    }
        except Exception:
            classifications = {
                pid: {
                    "guess": random.choice(["human", "ai"]),
                    "confidence": 2,
                    "reasoning": "gut feeling",
                }
                for pid in player_names
            }

        await self._send({"action": "CLASSIFICATION", "content": json.dumps(classifications)})
        logger.info("[%s] Classification submitted.", self.agent_id)

    # ── LLM helper ────────────────────────────────────────────────────────────

    async def _llm_str(
        self,
        user_prompt: str,
        key: str,
        fallback: str,
        timeout: float = 12.0,
    ) -> str:
        """Call provider.complete_json, extract `key` from result, or return fallback."""
        try:
            result = await asyncio.wait_for(
                self.provider.complete_json(SYSTEM_PROMPT, user_prompt),
                timeout=timeout,
            )
            value = str(result.get(key, "")).strip()
            return value if value else fallback
        except asyncio.TimeoutError:
            logger.warning("[%s] LLM timeout for key=%r", self.agent_id, key)
            return fallback
        except Exception as exc:
            logger.error("[%s] LLM error: %s", self.agent_id, exc)
            return fallback
