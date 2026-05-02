"""WebSocket connection manager for The Sandbox multiplayer game."""

from __future__ import annotations

import logging
import random
import time

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from server.config import RECONNECT_TIMEOUT
from server.models import PlayerInfo, PlayerType, ServerMessage

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for up to 6 players.

    Handles connection lifecycle, message routing, reconnection windows,
    and message queuing for temporarily disconnected players.
    """

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}
        self.player_info: dict[str, PlayerInfo] = {}
        self._message_queue: dict[str, list[str]] = {}
        self._disconnect_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(
        self,
        websocket: WebSocket,
        player_id: str,
        player_info: PlayerInfo,
    ) -> bool:
        """Accept a WebSocket connection and register the player.

        If the player was previously disconnected and is still within the
        reconnect window, any queued messages are flushed immediately.

        Returns True if the connection was established successfully.
        """
        await websocket.accept()
        self.active_connections[player_id] = websocket
        self.player_info[player_id] = player_info
        self.player_info[player_id].is_connected = True

        logger.info("Player %s connected", player_id)

        # Flush queued messages if reconnecting within the timeout window.
        if player_id in self._disconnect_times:
            elapsed = time.monotonic() - self._disconnect_times[player_id]
            if elapsed <= RECONNECT_TIMEOUT:
                await self._flush_queue(player_id, websocket)
            del self._disconnect_times[player_id]

        # Clean up any remaining queue entry (empty or expired).
        self._message_queue.pop(player_id, None)

        return True

    def disconnect(self, player_id: str) -> None:
        """Handle a player disconnection.

        The player is removed from active connections but kept in
        player_info so they can reconnect within the timeout window.
        """
        self.active_connections.pop(player_id, None)

        if player_id in self.player_info:
            self.player_info[player_id].is_connected = False

        self._disconnect_times[player_id] = time.monotonic()
        self._message_queue[player_id] = []

        logger.info("Player %s disconnected", player_id)

    # ------------------------------------------------------------------
    # Message sending
    # ------------------------------------------------------------------

    async def send_personal(self, player_id: str, message: ServerMessage) -> None:
        """Send a message to a single player.

        - Connected players receive the message immediately over their
          WebSocket.
        - Disconnected players within the reconnect window have the
          message queued for delivery upon reconnection.
        - Disconnected players past the reconnect window are silently
          skipped.
        """
        websocket = self.active_connections.get(player_id)
        if websocket is not None:
            try:
                await websocket.send_text(message.model_dump_json())
            except (WebSocketDisconnect, RuntimeError):
                self.disconnect(player_id)
                # Queue the message that failed to send.
                self._message_queue.setdefault(player_id, []).append(
                    message.model_dump_json()
                )
            return

        # Player is not connected — check reconnect window.
        disconnect_time = self._disconnect_times.get(player_id)
        if disconnect_time is not None:
            elapsed = time.monotonic() - disconnect_time
            if elapsed <= RECONNECT_TIMEOUT:
                self._message_queue.setdefault(player_id, []).append(
                    message.model_dump_json()
                )

    async def broadcast(
        self,
        message: ServerMessage,
        exclude: set[str] | None = None,
    ) -> None:
        """Send a message to all known players.

        Connected players receive the message immediately. Disconnected
        players within the reconnect window have the message queued.
        Players listed in *exclude* are skipped entirely.
        """
        exclude = exclude or set()
        for player_id in list(self.player_info):
            if player_id not in exclude:
                await self.send_personal(player_id, message)

    async def send_to_pair(
        self,
        player1_id: str,
        player2_id: str,
        message: ServerMessage,
    ) -> None:
        """Send a message to exactly two players (e.g. for DM routing)."""
        await self.send_personal(player1_id, message)
        await self.send_personal(player2_id, message)

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    async def _flush_queue(self, player_id: str, websocket: WebSocket) -> None:
        """Send all queued messages to a reconnected player and clear the queue."""
        queued = self._message_queue.get(player_id, [])
        for raw_message in queued:
            try:
                await websocket.send_text(raw_message)
            except WebSocketDisconnect:
                self.disconnect(player_id)
                return
        self._message_queue.pop(player_id, None)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_connected_players(self) -> list[str]:
        """Return the player_ids of all currently connected players."""
        return list(self.active_connections)

    def get_game_players(self) -> list[str]:
        """Connected players that are in the game (excludes HOST)."""
        return [
            pid for pid in self.active_connections
            if self.player_info.get(pid) and self.player_info[pid].player_type != PlayerType.HOST
        ]

    def get_host_ids(self) -> list[str]:
        """Connected host player IDs."""
        return [
            pid for pid in self.active_connections
            if self.player_info.get(pid) and self.player_info[pid].player_type == PlayerType.HOST
        ]

    async def send_to_hosts(self, message: ServerMessage) -> None:
        """Send a message to all connected hosts."""
        for host_id in self.get_host_ids():
            await self.send_personal(host_id, message)

    def get_host_roster(self) -> dict[str, dict]:
        """Return {pid: {display_name, player_type, model_name, real_name, connected}}
        for every game player. Host-only — exposes AI vs human identity and the
        typed human name, so must never be broadcast to regular players."""
        roster: dict[str, dict] = {}
        for pid, info in self.player_info.items():
            if info.player_type == PlayerType.HOST:
                continue
            roster[pid] = {
                "display_name": info.display_name,
                "player_type": info.player_type.value,
                "model_name": info.model_name,
                "real_name": info.real_name,
                "connected": pid in self.active_connections,
            }
        return roster

    def get_player_count(self) -> int:
        """Return the total number of registered players (connected + disconnected)."""
        return len(self.player_info)

    def get_game_player_count(self) -> int:
        """Total registered game players (excludes hosts)."""
        return sum(
            1 for info in self.player_info.values()
            if info.player_type != PlayerType.HOST
        )

    def shuffle_display_names(self) -> dict[str, str]:
        """Randomly assign Player_1..N to game players (shuffled)."""
        game_pids = self.get_game_players()
        shuffled = list(game_pids)
        random.shuffle(shuffled)
        roster: dict[str, str] = {}
        for idx, pid in enumerate(shuffled, start=1):
            new_name = f"Player_{idx}"
            self.player_info[pid].display_name = new_name
            roster[pid] = new_name
        return roster

    def is_connected(self, player_id: str) -> bool:
        """Return True if the player has an active WebSocket connection."""
        return player_id in self.active_connections

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> None:
        """Remove players whose disconnect time exceeds RECONNECT_TIMEOUT.

        Cleans up their message queues, player_info entries, and
        disconnect time records.
        """
        now = time.monotonic()
        expired = [
            pid
            for pid, ts in self._disconnect_times.items()
            if now - ts > RECONNECT_TIMEOUT
        ]
        for player_id in expired:
            self._disconnect_times.pop(player_id, None)
            self._message_queue.pop(player_id, None)
            self.player_info.pop(player_id, None)
            logger.info("Cleaned up expired player %s", player_id)
