"""
phase5_reflection_server.py

Phase 5 — Between-Round Reflection Server (standalone)
=======================================================
Manages the 90-second reflection phase between rounds:

  1. Receives player WebSocket connections (human + AI).
  2. On reflection start, broadcasts reflection_start with full round context:
       - Topic, image grid, secret word, all submitted hints with roles
  3. AI agents perform private reflection and broadcast a reflection_submitted.
  4. Players can send free-form chat messages (chat_message) during the window.
  5. Players can send DMs (dm_message) to specific players.
  6. After 90 seconds, broadcasts reflection_complete with all AI reflections.

Protocol (server → client):
    reflection_start      — begins phase; includes full round context + prompt
    reflection_submitted  — an AI agent's reflection (broadcast to all)
    chat_message          — a player's free-form chat (broadcast to all)
    dm_message            — a direct message delivered only to recipient
    reflection_complete   — phase ended; includes all reflections collected

Protocol (client → server):
    {"type": "reflection", "content": "..."}          — AI agent reflection
    {"type": "chat",       "content": "..."}          — free-form chat
    {"type": "dm",         "to": "<player_id>", "content": "..."}  — DM

Run standalone:
    python phase5_reflection_server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

REFLECTION_DURATION = 90   # seconds
REQUIRED_PLAYERS    = 6


# ─── Mock round data (replace with real round results in production) ──────────

MOCK_ROUND_RESULT = {
    "round_number": 1,
    "topic":        "Fruits",
    "secret_word":  "Mango",
    "image_grid":   [
        "Mango", "Apple", "Banana", "Strawberry", "Watermelon",
        "Grape", "Pineapple", "Cherry", "Orange", "Kiwi",
    ],
    "hints": [
        {"player_id": "Player_1", "hint": "tropical",        "role": "insider"},
        {"player_id": "Player_2", "hint": "sweet",           "role": "imposter"},
        {"player_id": "Player_3", "hint": "yellow inside",   "role": "insider"},
        {"player_id": "Player_4", "hint": "aamras vibes",    "role": "insider"},
        {"player_id": "Player_5", "hint": "colorful",        "role": "imposter"},
        {"player_id": "Player_6", "hint": "🌞🏝️",            "role": "insider"},
    ],
}


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class PlayerConn:
    player_id:   str
    websocket:   WebSocket
    player_type: str   # "human" | "ai"


@dataclass
class ReflectionRecord:
    player_id:  str
    content:    str
    timestamp:  float = field(default_factory=time.time)


@dataclass
class ReflectionPhaseConfig:
    round_number: int
    topic:        str
    secret_word:  str
    image_grid:   List[str]
    hints:        List[Dict]   # [{"player_id", "hint", "role"}, ...]


# ─── Reflection Phase Manager ─────────────────────────────────────────────────

class ReflectionPhaseManager:
    """
    Orchestrates Phase 5 for one between-round window.

    External interface:
        manager = ReflectionPhaseManager(config, connections)
        await manager.run()

    The WebSocket handler must call:
        await manager.receive_message(player_id, msg_dict)
    whenever a client sends any message.
    """

    def __init__(
        self,
        config:      ReflectionPhaseConfig,
        connections: Dict[str, PlayerConn],
    ) -> None:
        self.config      = config
        self.connections = connections
        self.reflections: List[ReflectionRecord] = []
        self._phase_start = time.monotonic()

    # ── Public entry ─────────────────────────────────────────────────────────

    async def run(self) -> List[ReflectionRecord]:
        await self._broadcast_reflection_start()
        await asyncio.sleep(REFLECTION_DURATION)
        await self._broadcast_reflection_complete()
        return self.reflections

    # ── External input ────────────────────────────────────────────────────────

    async def receive_message(self, player_id: str, msg: dict) -> None:
        msg_type = msg.get("type", "")

        if msg_type == "reflection":
            await self._on_reflection(player_id, msg.get("content", ""))
        elif msg_type == "chat":
            await self._on_chat(player_id, msg.get("content", ""))
        elif msg_type == "dm":
            await self._on_dm(player_id, msg.get("to", ""), msg.get("content", ""))
        else:
            logger.debug("[ReflectionPhase] Unknown message type %r from %s", msg_type, player_id)

    # ── Broadcasts / sends ────────────────────────────────────────────────────

    async def _broadcast(self, payload: dict) -> None:
        data = json.dumps(payload)
        dead: List[str] = []
        for pid, conn in list(self.connections.items()):
            try:
                await conn.websocket.send_text(data)
            except Exception as exc:
                logger.warning("[ReflectionPhase] Send to %s failed: %s", pid, exc)
                dead.append(pid)
        for pid in dead:
            self.connections.pop(pid, None)

    async def _send_to(self, player_id: str, payload: dict) -> None:
        conn = self.connections.get(player_id)
        if conn is None:
            return
        try:
            await conn.websocket.send_text(json.dumps(payload))
        except Exception as exc:
            logger.warning("[ReflectionPhase] Direct send to %s failed: %s", player_id, exc)

    # ── Phase events ──────────────────────────────────────────────────────────

    async def _broadcast_reflection_start(self) -> None:
        prompt = _build_reflection_prompt(self.config)
        await self._broadcast({
            "event":       "reflection_start",
            "round":       self.config.round_number,
            "topic":       self.config.topic,
            "secret_word": self.config.secret_word,
            "image_grid":  self.config.image_grid,
            "hints":       self.config.hints,
            "prompt":      prompt,
            "duration":    REFLECTION_DURATION,
        })
        logger.info(
            "[ReflectionPhase] Round %d reflection started (%ds window).",
            self.config.round_number, REFLECTION_DURATION,
        )

    async def _broadcast_reflection_complete(self) -> None:
        await self._broadcast({
            "event":       "reflection_complete",
            "round":       self.config.round_number,
            "reflections": [
                {"player_id": r.player_id, "content": r.content}
                for r in self.reflections
            ],
        })
        logger.info(
            "[ReflectionPhase] Complete. %d AI reflections collected.",
            len(self.reflections),
        )

    # ── Message handlers ──────────────────────────────────────────────────────

    async def _on_reflection(self, player_id: str, content: str) -> None:
        if not content.strip():
            return
        record = ReflectionRecord(player_id=player_id, content=content.strip())
        self.reflections.append(record)
        await self._broadcast({
            "event":     "reflection_submitted",
            "player_id": player_id,
            "content":   content.strip(),
        })
        logger.info("[ReflectionPhase] %s submitted reflection (%d chars).", player_id, len(content))

    async def _on_chat(self, player_id: str, content: str) -> None:
        if not content.strip():
            return
        await self._broadcast({
            "event":     "chat_message",
            "player_id": player_id,
            "content":   content.strip(),
        })
        logger.info("[ReflectionPhase] Chat from %s: %r", player_id, content[:60])

    async def _on_dm(self, sender_id: str, recipient_id: str, content: str) -> None:
        if not content.strip() or not recipient_id:
            return
        payload = {
            "event":   "dm_message",
            "from":    sender_id,
            "to":      recipient_id,
            "content": content.strip(),
        }
        # Deliver to recipient only (and echo to sender so they see it sent)
        await self._send_to(recipient_id, payload)
        if recipient_id != sender_id:
            await self._send_to(sender_id, payload)
        logger.info("[ReflectionPhase] DM %s → %s: %r", sender_id, recipient_id, content[:60])


# ─── Reflection prompt builder ────────────────────────────────────────────────

def _build_reflection_prompt(config: ReflectionPhaseConfig) -> str:
    grid_str   = ", ".join(config.image_grid)
    hints_text = "\n".join(
        f"  {h['player_id']}: \"{h['hint']}\"  (role hidden)"
        for h in config.hints
    )
    return (
        f"Reflect on Round {config.round_number}:\n"
        f"- Topic: {config.topic}, Grid: [{grid_str}], Secret word: {config.secret_word}\n"
        f"- Hints submitted:\n{hints_text}\n\n"
        f"Questions to consider:\n"
        f"1. How did each player's hint relate to the grid?\n"
        f"   Did anyone's hint seem like they were guessing from the grid rather than knowing the word?\n"
        f"2. META-GAME: Based on hint style (length, creativity, cultural references, emoji use),\n"
        f"   who seems human vs AI? Explain your reasoning."
    )


# ─── FastAPI application ──────────────────────────────────────────────────────

app = FastAPI(title="The Sandbox — Phase 5 Reflection Server (standalone)")

_pending_connections: Dict[str, PlayerConn] = {}
_reflection_manager: Optional[ReflectionPhaseManager] = None
_phase_started: bool = False


@app.websocket("/ws/{player_id}")
async def ws_endpoint(websocket: WebSocket, player_id: str) -> None:
    global _reflection_manager, _phase_started

    await websocket.accept()
    player_type = websocket.query_params.get("type", "human")
    logger.info("Player %s (%s) connected.", player_id, player_type)

    conn = PlayerConn(player_id=player_id, websocket=websocket, player_type=player_type)
    _pending_connections[player_id] = conn

    await websocket.send_text(json.dumps({
        "event":       "joined",
        "player_id":   player_id,
        "player_type": player_type,
        "waiting_for": REQUIRED_PLAYERS - len(_pending_connections),
    }))

    # Start reflection phase once all players are connected
    if len(_pending_connections) == REQUIRED_PLAYERS and not _phase_started:
        _phase_started = True
        config = ReflectionPhaseConfig(
            round_number = MOCK_ROUND_RESULT["round_number"],
            topic        = MOCK_ROUND_RESULT["topic"],
            secret_word  = MOCK_ROUND_RESULT["secret_word"],
            image_grid   = MOCK_ROUND_RESULT["image_grid"],
            hints        = MOCK_ROUND_RESULT["hints"],
        )
        _reflection_manager = ReflectionPhaseManager(config, dict(_pending_connections))
        asyncio.ensure_future(_reflection_manager.run())

    try:
        async for raw in websocket.iter_text():
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON from %s: %r", player_id, raw[:80])
                continue

            if _reflection_manager is not None:
                await _reflection_manager.receive_message(player_id, msg)

    except WebSocketDisconnect:
        logger.info("Player %s disconnected.", player_id)
        _pending_connections.pop(player_id, None)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-8s  %(message)s",
    )
    uvicorn.run("phase5_reflection_server:app", host="0.0.0.0", port=8001, reload=False)
