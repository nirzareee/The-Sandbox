"""Abstract base agent for The Sandbox game."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import random
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import websockets

from server.models import ActionType, GamePhase, PlayerRole

from agents.delay_simulator import DelaySimulator
from agents.memory import AgentMemory
from agents.prompt_builder import PromptBuilder

log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all AI agents."""

    def __init__(
        self,
        player_id: str,
        display_name: str,
        model_name: str,
        server_url: str = "ws://localhost:8000",
        api_timeout: int = 15,
    ) -> None:
        self.player_id = player_id
        self.display_name = display_name
        self.model_name = model_name
        self.server_url = server_url
        self.api_timeout = api_timeout

        self.ws: websockets.WebSocketClientProtocol | None = None
        self.connected = False

        self.memory = AgentMemory(player_id)
        self.current_phase: GamePhase | None = None
        self.current_round: int = 0
        self._num_rounds: int = 5
        self._discussion_task: asyncio.Task | None = None
        self._all_players: list[str] = []
        self._incoming_event = asyncio.Event()  # Fires when agent is mentioned or DM'd
        self._pending_dm_queue: list[str] = []  # Queue of player_ids who DM'd us
        self._dm_history: dict[str, list[str]] = {}  # {target_player_id: [sent messages]}
        self._session_id: str | None = None
        self._outbox: list[dict] = []

        # Identity mapping: raw player_id ↔ anonymous display_name ("Player 1")
        self._id_to_name: dict[str, str] = {}
        self._name_to_id: dict[str, str] = {}
        self._my_display_name: str | None = None

    # ------------------------------------------------------------------
    # Game params helper
    # ------------------------------------------------------------------

    def _get_game_params(self) -> tuple[int, int, list[str]]:
        """Return (num_players, num_rounds, player_display_names)."""
        names = [self._dn(p) for p in self._all_players]
        return len(names), self._num_rounds, names

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the game server with retry logic."""
        params = f"type=AI&name={self.display_name}&model={self.model_name}"
        url = f"{self.server_url}/ws/{self.player_id}?{params}"

        for attempt in range(3):
            try:
                self.ws = await websockets.connect(url)
                self.connected = True
                log.info("%s connected to %s", self.display_name, url)
                await self._flush_outbox()
                await self._event_loop()
                return
            except Exception as exc:
                log.error("Connection attempt %d failed: %s", attempt + 1, exc)
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    async def _event_loop(self) -> None:
        """Receive and dispatch server events."""
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                await self._dispatch(msg)
        except websockets.exceptions.ConnectionClosed:
            log.info("%s disconnected", self.display_name)
        finally:
            self.connected = False

    async def _dispatch(self, msg: dict) -> None:
        event = msg.get("event", "")
        data = msg.get("data", {})

        handler = getattr(self, f"_on_{event}", None)
        if handler:
            try:
                await handler(data)
            except Exception:
                log.exception("Error handling event '%s'", event)
        else:
            log.debug("%s: unhandled event '%s'", self.display_name, event)

    # ------------------------------------------------------------------
    # Sending actions
    # ------------------------------------------------------------------

    async def _send(self, action: str, content: str = "", target_player: str | None = None) -> None:
        # --- Secret word / grid word filter (safety net) ---
        if action in ("GROUP_MESSAGE", "DM") and content:
            rs = self.memory.current_round_state
            if rs:
                forbidden: list[str] = []
                sw = rs.get("secret_word")
                if sw:
                    forbidden.append(sw)
                forbidden.extend(rs.get("image_grid", []))
                content_lower = content.lower()
                for word in forbidden:
                    if word and re.search(r"\b" + re.escape(word.lower()) + r"\b", content_lower):
                        log.warning(
                            "%s: blocked message containing forbidden word '%s'",
                            self.display_name, word,
                        )
                        return  # silently drop — looks like the agent just paused

        payload: dict = {"action": action, "content": content}
        if target_player:
            payload["target_player"] = target_player
        if not self.ws:
            if len(self._outbox) < 50:  # bounded queue
                self._outbox.append(payload)
            return
        await self.ws.send(json.dumps(payload))

    async def _flush_outbox(self) -> None:
        while self._outbox and self.ws:
            payload = self._outbox.pop(0)
            try:
                await self.ws.send(json.dumps(payload))
            except Exception:
                self._outbox.insert(0, payload)
                break

    # ------------------------------------------------------------------
    # Identity translation helpers
    # ------------------------------------------------------------------

    def _dn(self, player_id: str) -> str:
        """Translate a player_id to its anonymous display name."""
        return self._id_to_name.get(player_id, player_id)

    def _pid(self, name: str) -> str:
        """Translate a display name back to a player_id."""
        return self._name_to_id.get(name, name)

    def _translate_hints(self, hints: dict[str, str]) -> dict[str, str]:
        return {self._dn(k): v for k, v in hints.items()}

    def _translate_hint_tuples(self, hints: list[tuple[str, str]]) -> list[tuple[str, str]]:
        return [(self._dn(pid), hint) for pid, hint in hints]

    def _translate_messages(self, msgs: list[dict]) -> list[dict]:
        out = []
        for m in msgs:
            m2 = dict(m)
            for key in ("player_id", "sender", "recipient"):
                if key in m2:
                    m2[key] = self._dn(m2[key])
            out.append(m2)
        return out

    def _translate_context(self, context: str) -> str:
        """Replace raw player_ids with display names in a text block."""
        for pid, name in self._id_to_name.items():
            context = context.replace(pid, name)
        return context

    def _resolve_target(self, target: str | None) -> str | None:
        """Resolve an LLM-generated target to a player_id.

        Handles variations like 'Player_1', 'Player 1', 'player_1',
        'Player1', 'P1', etc.
        """
        if not target:
            return None
        # Exact match on display name
        if target in self._name_to_id:
            return self._name_to_id[target]
        # Already a valid player_id
        if target in self._id_to_name:
            return target
        # Normalize: lowercase, strip spaces/underscores for fuzzy match
        t = target.lower().strip()
        for name, pid in self._name_to_id.items():
            # "Player_1" matches "player 1", "player_1", "Player1"
            if name.lower() == t:
                return pid
            if name.lower().replace("_", " ") == t:
                return pid
            if name.lower().replace("_", "") == t.replace(" ", "").replace("_", ""):
                return pid
        # Try extracting just the number: "P1", "player 2", "1" etc.
        m = re.search(r"(\d+)", t)
        if m:
            num = m.group(1)
            for name, pid in self._name_to_id.items():
                if name.endswith(f"_{num}") or name.endswith(f" {num}"):
                    return pid
        return target

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_game_created(self, data: dict) -> None:
        self.memory.reset()
        self._session_id = data.get("session_id")
        if "num_rounds" in data:
            self._num_rounds = data["num_rounds"]
        log.info("%s: game created (session=%s)", self.display_name, self._session_id)

    async def _on_player_joined(self, data: dict) -> None:
        connected = data.get("connected", [])
        self._all_players = connected
        # Build player_id → display_name mapping
        pid = data.get("player_id")
        name = data.get("display_name")
        if pid and name:
            self._id_to_name[pid] = name
            self._name_to_id[name] = pid
            if pid == self.player_id:
                self._my_display_name = name
                log.info("%s identity: %s", self.player_id, name)

    async def _on_player_left(self, data: dict) -> None:
        self._all_players = data.get("connected", [])

    async def _on_player_roster(self, data: dict) -> None:
        """Handle shuffled roster — rebuild identity maps."""
        roster = data.get("roster", {})
        if not roster:
            return
        self._id_to_name.clear()
        self._name_to_id.clear()
        for pid, display_name in roster.items():
            self._id_to_name[pid] = display_name
            self._name_to_id[display_name] = pid
            if pid == self.player_id:
                self._my_display_name = display_name
                log.info("%s identity (shuffled): %s", self.player_id, display_name)
        self._all_players = list(roster.keys())
        self.memory.set_name_mapping(self._id_to_name)

        # Now that names are assigned, set game start context
        np, nr, pl = self._get_game_params()
        my_name = self._my_display_name or self.player_id
        if np > 0:
            ctx = PromptBuilder.build_game_start_message(np, nr, pl, my_name)
            self.memory.set_game_start_context(ctx)

    async def _on_round_start(self, data: dict) -> None:
        self.current_round = data["round_number"]
        self.current_phase = GamePhase.ASSIGNMENT

        role = PlayerRole(data["role"])
        self.memory.start_round(
            round_number=self.current_round,
            role=role,
            topic=data["topic"],
            secret_word=data.get("secret_word"),
            image_grid=data.get("image_grid", []),
            hint_order=data.get("hint_order", []),
        )
        log.info(
            "%s round %d as %s (topic=%s)",
            self.display_name, self.current_round, role.value, data["topic"],
        )

    async def _on_your_turn(self, data: dict) -> None:
        pass  # No longer used for hints — hints are submitted on phase start

    async def _on_hint_submitted(self, data: dict) -> None:
        self.memory.record_hint(data["player_id"], data["hint"])

    async def _on_hint_phase_start(self, data: dict) -> None:
        self.current_phase = GamePhase.HINT
        # Generate and submit hint asynchronously
        asyncio.ensure_future(self._generate_and_submit_hint())

    async def _generate_and_submit_hint(self) -> None:
        """Generate a hint using the LLM and submit it."""
        rs = self.memory.current_round_state
        if not rs or "role" not in rs:
            log.warning("%s: no round state — submitting fallback hint", self.display_name)
            await asyncio.sleep(DelaySimulator.hint())
            await self._send("HINT", "...")
            return
        my_name = self._my_display_name or self.player_id
        np, nr, pl = self._get_game_params()

        prompt = PromptBuilder.build_hint_prompt(
            role=rs["role"],
            topic=rs["topic"],
            secret_word=rs.get("secret_word"),
            image_grid=rs["image_grid"],
            past_hints=self._translate_hint_tuples(self.memory.get_past_hints_this_round()),
            memory_context=self._translate_context(self.memory.get_context_for_prompt()),
            player_id=my_name,
            num_players=np,
            num_rounds=nr,
            player_list=pl,
        )

        t0 = asyncio.get_event_loop().time()
        try:
            raw = await self._call_llm_with_retry(prompt)
            parsed = self._parse_response(raw)
            hint = parsed.get("action", {}).get("content", "...")
            thinking = parsed.get("thinking", "")
            self.memory.log_thinking(self.current_round, "hint", thinking)
            self._flush_thinking(self.current_round, "hint", thinking)
        except (asyncio.TimeoutError, Exception) as exc:
            log.warning("%s hint fallback: %s", self.display_name, exc)
            hint = "..."

        target_delay = DelaySimulator.hint()
        remaining = max(0, target_delay - (asyncio.get_event_loop().time() - t0))
        await asyncio.sleep(remaining)
        await self._send("HINT", hint)

    async def _on_hint_phase_end(self, data: dict) -> None:
        pass

    def _is_repetitive_dm(self, target: str, content: str, threshold: float = 0.55) -> bool:
        """Check if this DM is too similar to one already sent to this target."""
        history = self._dm_history.get(target, [])
        content_lower = content.lower().strip()
        for prev in history:
            if difflib.SequenceMatcher(None, content_lower, prev.lower().strip()).ratio() > threshold:
                return True
        return False

    def _record_dm_sent(self, target: str, content: str) -> None:
        self._dm_history.setdefault(target, []).append(content)

    async def _on_discussion_start(self, data: dict) -> None:
        self.current_phase = GamePhase.DISCUSSION
        self._dm_history.clear()
        duration = data.get("duration", 180)
        self._discussion_task = asyncio.create_task(self._discussion_behavior(duration))

    async def _on_discussion_end(self, data: dict) -> None:
        if self._discussion_task and not self._discussion_task.done():
            self._discussion_task.cancel()

    def _compute_chat_state(
        self,
        my_msg_count: int,
        group_msg_count: int,
        time_remaining: float,
    ) -> str:
        """Return a chat state label for context-aware prompting.

        Returns one of: "empty", "quiet_self", "ending", "active".
        """
        if group_msg_count == 0:
            return "empty"
        if time_remaining < 30:
            return "ending"
        if my_msg_count == 0 and group_msg_count >= 2:
            return "quiet_self"
        return "active"

    async def _discussion_behavior(self, duration: float) -> None:
        """State-driven discussion loop with human-like timing.

        Uses wall-clock time to track elapsed/remaining time accurately.
        Wakes up early when someone DMs this agent or mentions them
        in group chat, so the agent can respond promptly.

        Timing states (all local variables):
        - is_first_turn: True on first iteration → fast opener (1.5-4s)
        - last_spoke + not burst_used: burst follow-up (1-3s)
        - incoming_event triggered: reactive reply (1-2.5s)
        - otherwise: normal discussion_turn (4-12s)
        """
        loop = asyncio.get_event_loop()
        start_time = loop.time()
        my_msg_count = 0
        is_first_turn = True
        last_spoke = False
        burst_used = False
        try:
            while True:
                wall_elapsed = loop.time() - start_time
                time_remaining = max(0.0, duration - wall_elapsed)
                if time_remaining <= 0:
                    break

                # --- Choose wait delay based on state ---
                if is_first_turn:
                    wait = min(DelaySimulator.first_discussion_turn(), time_remaining)
                elif last_spoke and not burst_used:
                    wait = min(DelaySimulator.burst_turn(), time_remaining)
                else:
                    wait = min(DelaySimulator.discussion_turn(), time_remaining)

                # Wait for either the normal timer OR an incoming event
                self._incoming_event.clear()
                was_triggered = False
                try:
                    await asyncio.wait_for(self._incoming_event.wait(), timeout=wait)
                    # Someone mentioned us or sent a DM — respond after a brief reactive delay
                    was_triggered = True
                    await asyncio.sleep(DelaySimulator.reactive_turn())
                except asyncio.TimeoutError:
                    pass  # Normal timer expired

                is_first_turn = False

                # Capture and clear pending DM trigger
                dm_from = self._pending_dm_queue.pop(0) if self._pending_dm_queue else None

                # Re-check time after waiting
                wall_elapsed = loop.time() - start_time
                time_remaining = max(0.0, duration - wall_elapsed)
                if time_remaining <= 0:
                    break

                rs = self.memory.current_round_state
                if not rs or "topic" not in rs:
                    continue
                my_name = self._my_display_name or self.player_id
                np, nr, pl = self._get_game_params()

                # Translate dm_from to display name for the prompt
                reply_to_dm_from = self._dn(dm_from) if dm_from else None
                recent_thinking = self.memory.get_recent_thinking()

                # Separate group messages and DM threads so DMs are never
                # pushed out of context by a busy group chat.
                group_msgs = self._translate_messages(self.memory.get_group_messages_this_round())
                raw_threads = self.memory.get_dm_threads_this_round()
                dm_threads = {self._dn(k): self._translate_messages(v) for k, v in raw_threads.items()}

                # Compute chat state for context-aware prompting
                chat_state = self._compute_chat_state(
                    my_msg_count, len(group_msgs), time_remaining,
                )

                prompt = PromptBuilder.build_discussion_prompt(
                    topic=rs["topic"],
                    secret_word=rs.get("secret_word"),
                    image_grid=rs["image_grid"],
                    hints=self._translate_hints(self.memory.get_all_hints_this_round()),
                    recent_group_messages=group_msgs,
                    dm_threads=dm_threads,
                    memory_context=self._translate_context(self.memory.get_context_for_prompt()),
                    player_id=my_name,
                    num_players=np,
                    num_rounds=nr,
                    player_list=pl,
                    reply_to_dm_from=reply_to_dm_from,
                    recent_thinking=recent_thinking,
                    my_msg_count=my_msg_count,
                    time_remaining=time_remaining,
                    chat_state=chat_state,
                )

                t0 = loop.time()
                spoke_this_turn = False
                try:
                    raw = await self._call_llm_with_retry(prompt)
                    parsed = self._parse_response(raw)
                    action = parsed.get("action", {})
                    atype = action.get("type", "wait")
                    content = action.get("content", "")
                    target = self._resolve_target(action.get("target_player"))
                    thinking = parsed.get("thinking", "")
                    self.memory.log_thinking(self.current_round, "discussion", thinking)
                    self._flush_thinking(self.current_round, "discussion", thinking)

                    if atype == "group_message" and content:
                        target_delay = DelaySimulator.group_message()
                        remaining = max(0, target_delay - (loop.time() - t0))
                        await asyncio.sleep(remaining)
                        await self._send("GROUP_MESSAGE", content)
                        my_msg_count += 1
                        spoke_this_turn = True
                    elif atype == "dm" and content and target:
                        if target == self.player_id:
                            log.warning("%s tried to DM itself, skipping", self.display_name)
                        elif self._is_repetitive_dm(target, content):
                            log.info("%s blocked repetitive DM to %s", self.display_name, self._dn(target))
                        else:
                            self._record_dm_sent(target, content)
                            target_delay = DelaySimulator.dm()
                            remaining = max(0, target_delay - (loop.time() - t0))
                            await asyncio.sleep(remaining)
                            await self._send("DM", content, target)
                            my_msg_count += 1
                            spoke_this_turn = True
                    # else: wait — do nothing
                except Exception as exc:
                    log.warning("%s discussion LLM error: %s", self.display_name, exc)

                # Update burst state: allow max 2 consecutive fast messages
                if spoke_this_turn:
                    if last_spoke:
                        burst_used = True  # already did a burst follow-up
                    last_spoke = True
                else:
                    last_spoke = False
                    burst_used = False
        except asyncio.CancelledError:
            pass

    async def _on_group_message(self, data: dict) -> None:
        self.memory.record_group_message(data["player_id"], data["content"])
        # Wake up discussion loop if someone mentions us by any name variant
        content = data.get("content", "").lower()
        names_to_check = [self.player_id.lower(), self.display_name.lower()]
        if self._my_display_name:
            name = self._my_display_name.lower()
            names_to_check.append(name)                       # Player_2
            names_to_check.append(name.replace("_", " "))     # player 2
            names_to_check.append(name.replace("_", ""))      # player2
        if any(name in content for name in names_to_check):
            self._incoming_event.set()

    async def _on_dm(self, data: dict) -> None:
        sender = data.get("sender")
        recipient = data.get("recipient")
        content = data.get("content", "")
        if recipient == self.player_id or sender == self.player_id:
            self.memory.record_dm(sender, recipient, content)
            # Wake up discussion loop to reply when we receive a DM
            if recipient == self.player_id and sender != self.player_id:
                if sender not in self._pending_dm_queue:
                    self._pending_dm_queue.append(sender)
                self._incoming_event.set()

    async def _on_voting_start(self, data: dict) -> None:
        self.current_phase = GamePhase.VOTING
        candidates = data.get("candidates", [])
        rs = self.memory.current_round_state
        if not rs or "topic" not in rs:
            # No round state — vote randomly
            choices = [c for c in candidates if c != self.player_id]
            target = random.choice(choices) if choices else candidates[0] if candidates else None
            if target:
                await asyncio.sleep(DelaySimulator.vote())
                await self._send("VOTE", "", target)
            return
        my_name = self._my_display_name or self.player_id
        np, nr, pl = self._get_game_params()
        recent_thinking = self.memory.get_recent_thinking()

        group_msgs = self._translate_messages(self.memory.get_group_messages_this_round())
        raw_threads = self.memory.get_dm_threads_this_round()
        dm_threads = {self._dn(k): self._translate_messages(v) for k, v in raw_threads.items()}

        prompt = PromptBuilder.build_voting_prompt(
            topic=rs["topic"],
            secret_word=rs.get("secret_word"),
            image_grid=rs["image_grid"],
            hints=self._translate_hints(self.memory.get_all_hints_this_round()),
            recent_group_messages=group_msgs,
            dm_threads=dm_threads,
            candidates=[self._dn(c) for c in candidates],
            memory_context=self._translate_context(self.memory.get_context_for_prompt()),
            player_id=my_name,
            num_players=np,
            num_rounds=nr,
            player_list=pl,
            recent_thinking=recent_thinking,
        )

        target = None
        t0 = asyncio.get_event_loop().time()
        try:
            raw = await self._call_llm_with_retry(prompt)
            parsed = self._parse_response(raw)
            raw_target = parsed.get("action", {}).get("target_player")
            target = self._resolve_target(raw_target)
            thinking = parsed.get("thinking", "")
            self.memory.log_thinking(self.current_round, "voting", thinking)
            self._flush_thinking(self.current_round, "voting", thinking)
        except Exception as exc:
            log.warning("%s vote fallback: %s", self.display_name, exc)

        if not target or target == self.player_id or target not in candidates:
            choices = [c for c in candidates if c != self.player_id]
            target = random.choice(choices) if choices else candidates[0]

        target_delay = DelaySimulator.vote()
        remaining = max(0, target_delay - (asyncio.get_event_loop().time() - t0))
        await asyncio.sleep(remaining)
        await self._send("VOTE", "", target)

    async def _on_voting_end(self, data: dict) -> None:
        pass

    async def _on_round_results(self, data: dict) -> None:
        self.memory.record_round_result(data)
        # Extract vote breakdown into memory
        votes = data.get("votes", {})
        for voter_id, target_id in votes.items():
            self.memory.record_vote(voter_id, target_id)
        log.info(
            "%s round %d: imposter=%s caught=%s",
            self.display_name, self.current_round,
            data.get("imposter_id"), data.get("imposter_caught"),
        )

    async def _on_reflection_start(self, data: dict) -> None:
        self.current_phase = GamePhase.REFLECTION

        rs = self.memory.current_round_state
        my_name = self._my_display_name or self.player_id
        np, nr, pl = self._get_game_params()

        # Build list of other players (display names) for the prompt
        other_players = [name for name in pl if name != my_name]

        # Translate imposter_id in round result
        result = self.memory.get_round_result()
        if result and "imposter_id" in result:
            result = dict(result)
            result["imposter_id"] = self._dn(result["imposter_id"])

        prompt = PromptBuilder.build_reflection_prompt(
            round_number=self.current_round,
            topic=rs["topic"],
            secret_word=rs.get("secret_word"),
            image_grid=rs["image_grid"],
            hints=self._translate_hints(self.memory.get_all_hints_this_round()),
            discussion=self._translate_messages(self.memory.get_all_discussion_this_round()),
            result=result,
            memory_context=self._translate_context(self.memory.get_context_for_prompt()),
            player_id=my_name,
            other_players=other_players,
            num_players=np,
            num_rounds=nr,
            player_list=pl,
        )

        try:
            raw = await self._call_llm_with_retry(prompt)
            parsed = self._parse_response(raw)
            thinking = parsed.get("thinking", "")
            action = parsed.get("action", {})
            self.memory.log_reflection(self.current_round, thinking)
            self._flush_reflection(self.current_round, thinking)

            # Feed structured suspicion updates into meta_game_tracker
            suspicion_updates = action.get("suspicion_updates", {})
            for player_name, update in suspicion_updates.items():
                pid = self._resolve_target(player_name)
                if pid and pid != self.player_id:
                    self.memory.update_meta_tracker(
                        pid,
                        update.get("confidence", 0.5),
                        update.get("note", ""),
                    )
        except Exception as exc:
            log.warning("%s reflection error: %s", self.display_name, exc)

        # Compress AFTER reflection so the summary includes the reflection text
        self.memory.compress_round(self.current_round)

    async def _on_reflection_end(self, data: dict) -> None:
        pass

    async def _on_meta_game_start(self, data: dict) -> None:
        self.current_phase = GamePhase.META_GAME
        players_data = data.get("players", {})
        # players_data is a dict: {player_id: {player_id, display_name}}
        if isinstance(players_data, dict):
            player_ids = list(players_data.keys())
        else:
            player_ids = list(players_data)
        my_name = self._my_display_name or self.player_id
        np, nr, pl = self._get_game_params()

        # Build display-name lists for the prompt
        display_players = [self._dn(p) for p in player_ids]

        prompt = PromptBuilder.build_classification_prompt(
            all_rounds_memory=self._translate_context(self.memory.get_all_rounds_summary()),
            players=display_players,
            player_id=my_name,
            memory_tracker={self._dn(k): v for k, v in self.memory.meta_game_tracker.items()},
            num_players=np,
            num_rounds=nr,
            player_list=pl,
        )

        t0 = asyncio.get_event_loop().time()
        try:
            raw = await self._call_llm_with_retry(prompt)
            parsed = self._parse_response(raw)
            # Unified format: action.classifications, with top-level fallback
            action = parsed.get("action", {})
            raw_classifications = action.get("classifications", {})
            if not raw_classifications:
                raw_classifications = parsed.get("classifications", {})  # fallback
            thinking = parsed.get("thinking", "")
            self.memory.log_thinking(self.current_round, "classification", thinking)
            self._flush_thinking(self.current_round, "classification", thinking)
            # Translate display names back to player_ids
            classifications = {}
            for key, val in raw_classifications.items():
                pid = self._resolve_target(key) or key
                classifications[pid] = val
        except Exception as exc:
            log.warning("%s classification fallback: %s", self.display_name, exc)
            classifications = {}
            for pid in player_ids:
                if pid != self.player_id:
                    classifications[pid] = {
                        "guess": "human",
                        "confidence": 3,
                        "reasoning": "default guess",
                    }

        # Make sure we don't classify ourselves
        classifications.pop(self.player_id, None)
        # Also pop our display name in case LLM used it
        if my_name in classifications:
            classifications.pop(my_name, None)

        # Ensure all other players are classified
        for pid in player_ids:
            if pid != self.player_id and pid not in classifications:
                classifications[pid] = {
                    "guess": "human",
                    "confidence": 3,
                    "reasoning": "no observation",
                }

        target_delay = DelaySimulator.classification()
        remaining = max(0, target_delay - (asyncio.get_event_loop().time() - t0))
        await asyncio.sleep(remaining)
        await self._send("CLASSIFICATION", json.dumps(classifications))

    async def _on_meta_game_end(self, data: dict) -> None:
        pass

    async def _on_survey_start(self, data: dict) -> None:
        pass  # AI agents don't fill surveys

    async def _on_survey_end(self, data: dict) -> None:
        pass

    async def _on_grand_reveal(self, data: dict) -> None:
        log.info("%s: grand reveal received", self.display_name)

    async def _on_game_finished(self, data: dict) -> None:
        log.info("%s: game finished", self.display_name)
        self.connected = False
        if self.ws:
            await self.ws.close()

    async def _on_error(self, data: dict) -> None:
        log.error("%s server error: %s", self.display_name, data.get("message"))

    # ------------------------------------------------------------------
    # Thinking / reflection persistence
    # ------------------------------------------------------------------

    def _flush_thinking(self, round_num: int, phase: str, thinking: str) -> None:
        """Append a thinking trace to the session's ai_internals directory."""
        if not self._session_id or not thinking:
            return
        ai_dir = Path("game_sessions") / self._session_id / "ai_internals" / f"{self.player_id}_{self.model_name}"
        ai_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "round_number": round_num,
            "phase": phase,
            "thinking": thinking,
        }
        with open(ai_dir / "thinking_traces.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _flush_reflection(self, round_num: int, reflection: str) -> None:
        """Append a reflection to the session's ai_internals directory."""
        if not self._session_id or not reflection:
            return
        ai_dir = Path("game_sessions") / self._session_id / "ai_internals" / f"{self.player_id}_{self.model_name}"
        ai_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "round_number": round_num,
            "reflection": reflection,
        }
        with open(ai_dir / "reflections.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # LLM interface
    # ------------------------------------------------------------------

    async def _call_llm_with_retry(self, prompt: str, max_retries: int = 2) -> str:
        for attempt in range(max_retries + 1):
            try:
                return await asyncio.wait_for(self.call_llm(prompt), timeout=self.api_timeout)
            except (asyncio.TimeoutError, Exception) as exc:
                if attempt == max_retries:
                    raise
                wait = min(2 ** attempt, 5)
                log.warning("%s: LLM retry %d/%d after %s", self.display_name, attempt + 1, max_retries, exc)
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable")  # satisfy type checker

    @abstractmethod
    async def call_llm(self, prompt: str) -> str:
        """Call the underlying LLM. Subclasses must implement."""
        ...

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> dict:
        """Parse JSON response from LLM. Tries to extract JSON from markdown fences."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            log.warning("Failed to parse LLM response: %.200s", text)
            return {"thinking": text, "action": {"type": "wait"}}
