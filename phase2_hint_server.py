"""
phase2_hint_server.py

Phase 2 — Hint Phase Server (standalone)
=========================================
Manages the hint-collection phase for one game round:

  1. Receives player WebSocket connections.
  2. On game start, broadcasts hint_phase_start with a randomised turn order.
  3. Sends a hint_request to each player in turn.
  4. Enforces rules per the PRD:
       • Hard 60-character limit  → truncate silently (AI agents) or warn (humans)
       • Sentence-like hint        → reject + request retry once; accept on 2nd attempt
  5. Broadcasts hint_submitted after each accepted hint.
  6. Advances automatically when all players have submitted or the 60-second
     phase timer expires.
  7. Broadcasts hint_phase_complete with the full hint list.

Run standalone for testing (requires 6 WebSocket clients, see phase2_test.py):
    python phase2_hint_server.py

Protocol (server → client):
    hint_phase_start   — phase begins; includes randomised hint_order
    hint_turn          — whose turn it is right now (broadcast)
    hint_request       — sent ONLY to the current player; contains full context
    hint_submitted     — a player's accepted hint (broadcast to all)
    hint_truncated     — sent to player whose hint was shortened
    hint_rejected      — sent to player whose hint was rejected (retry=True once)
    hint_skipped       — player timed out (broadcast)
    hint_phase_complete — all hints collected; transitions to Phase 3

Protocol (client → server):
    {"type": "hint", "content": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

HINT_PHASE_DURATION   = 60    # total seconds for entire hint phase
PER_HINT_TIMEOUT      = 15    # seconds each player has to submit before skip
MAX_HINT_CHARS        = 60    # hard character limit (PRD FR-01.8)
REQUIRED_PLAYERS      = 6

# Patterns that trigger a "sounds like a sentence" rejection (PRD phase 2)
SENTENCE_PATTERNS: tuple[str, ...] = ("this is", "i would describe")


# ─── Mock word bank entry (replace with real word_bank.json in production) ───

MOCK_ROUND = {
    "round_id":   1,
    "topic":      "Fruits",
    "secret_word": "Mango",
    "image_grid": [
        "Mango", "Apple", "Banana", "Strawberry", "Watermelon",
        "Grape", "Pineapple", "Cherry", "Orange", "Kiwi",
    ],
}


# ─── Enums & data classes ─────────────────────────────────────────────────────

class HintStatus(str, Enum):
    ACCEPTED  = "accepted"
    TRUNCATED = "truncated"   # over 60 chars → silently shortened
    REJECTED  = "rejected"    # sentence-like → retry requested
    SKIPPED   = "skipped"     # player timed out


@dataclass
class HintRecord:
    """One resolved hint entry stored in the phase results."""
    player_id: str
    role:      str
    content:   str
    status:    HintStatus
    original:  Optional[str] = None  # only set when content was altered
    timestamp: float = field(default_factory=time.time)


@dataclass
class PlayerConn:
    """Wraps a live WebSocket with per-player metadata."""
    player_id:   str
    websocket:   WebSocket
    role:        str          # "insider" | "imposter"
    player_type: str          # "human"   | "ai"
    secret_word: Optional[str] = None   # None for imposter


@dataclass
class HintPhaseConfig:
    """
    All information needed to run one hint phase.
    Populated from the round_start payload (Phase 1 hand-off).
    """
    round_id:           int
    topic:              str
    image_grid:         List[str]
    player_roles:       Dict[str, str]           # player_id → "insider"|"imposter"
    player_secret_words: Dict[str, Optional[str]] # player_id → word or None
    player_types:       Dict[str, str]           # player_id → "human"|"ai"


# ─── Hint validation ──────────────────────────────────────────────────────────

def validate_hint(raw: str) -> tuple[bool, str, HintStatus]:
    """
    Apply the two server-side hint rules in order:

    Rule 1 — sentence rejection (checked BEFORE truncation so the full text is
             inspected):
             If the hint contains "this is" or "i would describe" (case-insensitive)
             → return (False, raw, REJECTED).  Caller handles retry.

    Rule 2 — length truncation:
             If len > MAX_HINT_CHARS → truncate and return (True, truncated, TRUNCATED).

    Otherwise → return (True, stripped, ACCEPTED).
    """
    content = raw.strip()
    lower   = content.lower()

    # Rule 1: sentence-like rejection
    for pattern in SENTENCE_PATTERNS:
        if pattern in lower:
            return False, content, HintStatus.REJECTED

    # Rule 2: hard length cap
    if len(content) > MAX_HINT_CHARS:
        return True, content[:MAX_HINT_CHARS], HintStatus.TRUNCATED

    return True, content, HintStatus.ACCEPTED


# ─── Hint Phase Manager ───────────────────────────────────────────────────────

class HintPhaseManager:
    """
    Orchestrates Phase 2 for one game round.

    External interface:
        manager = HintPhaseManager(config, connections)
        hint_records = await manager.run()

    The WebSocket handler must call:
        await manager.receive_hint(player_id, content)
    whenever a client sends {"type": "hint", ...}.
    """

    def __init__(
        self,
        config:      HintPhaseConfig,
        connections: Dict[str, PlayerConn],
    ) -> None:
        self.config      = config
        self.connections = connections                  # mutable; tracks live players
        self.hint_order  = self._randomise_order()
        self.hints:      List[HintRecord] = []
        self._phase_start = time.monotonic()

        # Async coordination for turn-based hint collection
        self._hint_event:   asyncio.Event               = asyncio.Event()
        self._pending_hint: Optional[tuple[str, str]]   = None  # (player_id, raw_content)

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self) -> List[HintRecord]:
        """
        Drive the full hint phase.
        Returns the list of HintRecord objects (one per player, including skips).
        """
        await self._broadcast_phase_start()

        for position, player_id in enumerate(self.hint_order):
            remaining = self._phase_remaining()
            if remaining <= 0:
                logger.info("[HintPhase] Phase timer expired after %d hints.", len(self.hints))
                break

            conn = self.connections.get(player_id)
            if conn is None:
                logger.warning("[HintPhase] %s disconnected — skipping turn.", player_id)
                continue

            await self._run_turn(player_id, conn, position)

        await self._broadcast_phase_complete()
        return self.hints

    # ── External input ────────────────────────────────────────────────────────

    async def receive_hint(self, player_id: str, raw_content: str) -> None:
        """
        Called by the WebSocket handler whenever ANY player sends a hint message.
        Only wakes the manager if it is currently waiting for that player's turn.
        """
        self._pending_hint = (player_id, raw_content)
        self._hint_event.set()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _randomise_order(self) -> List[str]:
        order = list(self.connections.keys())
        random.shuffle(order)
        logger.info("[HintPhase] Hint order: %s", order)
        return order

    def _phase_remaining(self) -> float:
        return max(0.0, HINT_PHASE_DURATION - (time.monotonic() - self._phase_start))

    # ── Broadcasts ────────────────────────────────────────────────────────────

    async def _broadcast(self, msg: dict) -> None:
        payload = json.dumps(msg)
        dead: List[str] = []
        for pid, conn in list(self.connections.items()):
            try:
                await conn.websocket.send_text(payload)
            except Exception as exc:
                logger.warning("[HintPhase] Send to %s failed: %s", pid, exc)
                dead.append(pid)
        for pid in dead:
            self.connections.pop(pid, None)

    async def _send_to(self, player_id: str, msg: dict) -> None:
        conn = self.connections.get(player_id)
        if conn is None:
            return
        try:
            await conn.websocket.send_text(json.dumps(msg))
        except Exception as exc:
            logger.warning("[HintPhase] Direct send to %s failed: %s", player_id, exc)

    async def _broadcast_phase_start(self) -> None:
        await self._broadcast({
            "event":          "hint_phase_start",
            "round":          self.config.round_id,
            "topic":          self.config.topic,
            "image_grid":     self.config.image_grid,
            "hint_order":     self.hint_order,
            "phase_duration": HINT_PHASE_DURATION,
        })
        logger.info("[HintPhase] Round %d hint phase started.", self.config.round_id)

    async def _broadcast_phase_complete(self) -> None:
        summary = [
            {
                "player_id": h.player_id,
                "role":      h.role,
                "hint":      h.content,
                "status":    h.status,
            }
            for h in self.hints
        ]
        await self._broadcast({
            "event":  "hint_phase_complete",
            "round":  self.config.round_id,
            "hints":  summary,
        })
        logger.info("[HintPhase] Complete. %d hints collected.", len(self.hints))

    # ── Turn logic ────────────────────────────────────────────────────────────

    async def _run_turn(
        self,
        player_id: str,
        conn:      PlayerConn,
        position:  int,
    ) -> None:
        """
        Handle one player's hint turn:
          1. Broadcast whose turn it is.
          2. Send hint_request directly to the active player.
          3. Wait up to PER_HINT_TIMEOUT (capped at phase remaining).
          4. Validate; reject + retry once if sentence-like.
          5. Record and broadcast the result.
        """
        remaining = self._phase_remaining()

        # ── Announce turn ────────────────────────────────────────────────────
        await self._broadcast({
            "event":                "hint_turn",
            "player_id":            player_id,
            "position":             position + 1,
            "total":                len(self.hint_order),
            "remaining_phase_time": remaining,
        })

        # ── Request hint from the active player ──────────────────────────────
        await self._send_to(player_id, {
            "event":        "hint_request",
            "round":        self.config.round_id,
            "topic":        self.config.topic,
            "image_grid":   self.config.image_grid,
            "secret_word":  conn.secret_word,        # None for imposter
            "role":         conn.role,
            "hints_so_far": [
                {"player_id": h.player_id, "hint": h.content}
                for h in self.hints
            ],
            "timeout":      min(PER_HINT_TIMEOUT, remaining),
            "max_chars":    MAX_HINT_CHARS,
        })

        # ── Collect hint (up to 2 attempts: initial + 1 rejection retry) ─────
        deadline      = min(PER_HINT_TIMEOUT, self._phase_remaining())
        attempt_count = 0
        accepted      = False

        while attempt_count < 2 and not accepted:
            raw = await self._wait_for_hint(player_id, deadline)

            if raw is None:
                # Timeout or disconnect
                break

            is_valid, processed, status = validate_hint(raw)

            if status == HintStatus.REJECTED and attempt_count == 0:
                # First rejection → ask for a shorter hint (retry once)
                logger.info("[HintPhase] %s hint rejected (attempt 1): %r", player_id, raw)
                await self._send_to(player_id, {
                    "event":    "hint_rejected",
                    "original": raw,
                    "reason":   "Hint looks like a sentence. Please give a shorter hint (1–4 words).",
                    "retry":    True,
                })
                attempt_count += 1
                # Give remaining time for the retry (no extra time added)
                deadline = self._phase_remaining()
                continue

            if status == HintStatus.REJECTED and attempt_count == 1:
                # Second attempt still sentence-like → truncate and accept anyway
                logger.warning(
                    "[HintPhase] %s hint rejected twice; accepting truncated: %r",
                    player_id, raw,
                )
                processed = raw[:MAX_HINT_CHARS]
                status    = HintStatus.TRUNCATED

            if status == HintStatus.TRUNCATED:
                # Notify player their hint was shortened
                await self._send_to(player_id, {
                    "event":    "hint_truncated",
                    "original": raw,
                    "accepted": processed,
                    "message":  f"Hint shortened to {MAX_HINT_CHARS} characters.",
                })

            # Record and broadcast
            record = HintRecord(
                player_id = player_id,
                role      = conn.role,
                content   = processed,
                status    = status,
                original  = raw if raw != processed else None,
            )
            self.hints.append(record)

            await self._broadcast({
                "event":     "hint_submitted",
                "player_id": player_id,
                "hint":      processed,
                "status":    status,
                "position":  len(self.hints),
            })
            logger.info(
                "[HintPhase] %s (%s) → %r  [%s]",
                player_id, conn.role, processed, status,
            )
            accepted = True

        if not accepted:
            # Player timed out or exhausted retries without submitting
            self.hints.append(HintRecord(
                player_id = player_id,
                role      = conn.role,
                content   = "...",
                status    = HintStatus.SKIPPED,
            ))
            await self._broadcast({
                "event":     "hint_skipped",
                "player_id": player_id,
                "reason":    "timeout",
            })
            logger.info("[HintPhase] %s skipped (timeout/disconnect).", player_id)

    async def _wait_for_hint(
        self,
        expected_player: str,
        timeout: float,
    ) -> Optional[str]:
        """
        Wait until the expected player submits a hint or the timeout fires.
        Ignores hints from other players (they are out of turn).
        Returns the raw hint string, or None on timeout/no submission.
        """
        deadline = time.monotonic() + timeout

        while True:
            remaining_wait = deadline - time.monotonic()
            if remaining_wait <= 0:
                return None

            self._hint_event.clear()
            self._pending_hint = None

            try:
                await asyncio.wait_for(self._hint_event.wait(), timeout=remaining_wait)
            except asyncio.TimeoutError:
                return None

            if self._pending_hint is None:
                continue

            submitter, content = self._pending_hint

            if submitter != expected_player:
                # Out-of-turn submission — ignore and keep waiting
                logger.debug(
                    "[HintPhase] Out-of-turn hint from %s (expected %s) — ignored.",
                    submitter, expected_player,
                )
                continue

            return content


# ─── FastAPI application (standalone test server) ─────────────────────────────

app = FastAPI(title="The Sandbox — Phase 2 Hint Server (standalone)")

_pending_connections: Dict[str, PlayerConn] = {}
_hint_manager: Optional[HintPhaseManager]   = None
_game_started:  bool                        = False


@app.websocket("/ws/{player_id}")
async def ws_endpoint(websocket: WebSocket, player_id: str) -> None:
    global _hint_manager, _game_started

    await websocket.accept()
    logger.info("Player %s connected.", player_id)

    # Assign role: first joiner becomes imposter (deterministic for testing).
    role        = "imposter" if len(_pending_connections) == 0 else "insider"
    secret_word = None if role == "imposter" else MOCK_ROUND["secret_word"]

    conn = PlayerConn(
        player_id   = player_id,
        websocket   = websocket,
        role        = role,
        player_type = "human",  # override via query param in production
        secret_word = secret_word,
    )
    _pending_connections[player_id] = conn

    await websocket.send_text(json.dumps({
        "event":       "joined",
        "player_id":   player_id,
        "role":        role,
        "secret_word": secret_word,
        "waiting_for": REQUIRED_PLAYERS - len(_pending_connections),
    }))

    # Launch the hint phase once all players are connected
    if len(_pending_connections) == REQUIRED_PLAYERS and not _game_started:
        _game_started = True
        config = HintPhaseConfig(
            round_id    = MOCK_ROUND["round_id"],
            topic       = MOCK_ROUND["topic"],
            image_grid  = MOCK_ROUND["image_grid"],
            player_roles       = {pid: c.role        for pid, c in _pending_connections.items()},
            player_secret_words= {pid: c.secret_word for pid, c in _pending_connections.items()},
            player_types       = {pid: c.player_type for pid, c in _pending_connections.items()},
        )
        _hint_manager = HintPhaseManager(config, dict(_pending_connections))
        asyncio.ensure_future(_hint_manager.run())

    # Message loop
    try:
        async for raw in websocket.iter_text():
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON from %s: %r", player_id, raw)
                continue

            if msg.get("type") == "hint" and _hint_manager is not None:
                await _hint_manager.receive_hint(player_id, msg.get("content", ""))

    except WebSocketDisconnect:
        logger.info("Player %s disconnected.", player_id)
        _pending_connections.pop(player_id, None)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-8s  %(message)s",
    )
    uvicorn.run("phase2_hint_server:app", host="0.0.0.0", port=8000, reload=False)
