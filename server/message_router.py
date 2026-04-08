"""MessageRouter — parses incoming ClientMessages and dispatches to handlers."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Awaitable, Callable

from pydantic import ValidationError

from server.models import (
    ActionType,
    ClassificationSubmission,
    ClientMessage,
    GamePhase,
    HintSubmission,
    PlayerClassification,
    ServerMessage,
    SurveyResponse,
    VoteSubmission,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase → allowed actions mapping
# ---------------------------------------------------------------------------

PHASE_ALLOWED_ACTIONS: dict[GamePhase, set[ActionType]] = {
    GamePhase.LOBBY: set(),
    GamePhase.ASSIGNMENT: set(),
    GamePhase.HINT: {ActionType.HINT},
    GamePhase.DISCUSSION: {ActionType.GROUP_MESSAGE, ActionType.DM},
    GamePhase.VOTING: {ActionType.VOTE},
    GamePhase.RESULTS: set(),
    GamePhase.REFLECTION: {ActionType.GROUP_MESSAGE, ActionType.DM},
    GamePhase.META_GAME: {ActionType.CLASSIFICATION},
    GamePhase.SURVEY: {ActionType.SURVEY},
    GamePhase.REVEAL: set(),
    GamePhase.FINISHED: set(),
}

# Patterns that indicate overly verbose / AI-sounding hints.
AI_HINT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)^this\s+(is|fruit|item)"),
    re.compile(r"(?i)^I\s+would\s+describe"),
    re.compile(r"(?i)^(it\s+is|this\s+is)\s+a\s+"),
    re.compile(r"(?i)commonly\s+(found|known|used)"),
]


# ---------------------------------------------------------------------------
# MessageRouter
# ---------------------------------------------------------------------------

class MessageRouter:
    """Stateless router that maps incoming client actions to callback functions.

    Callbacks are wired up by ``main.py`` after construction — the router
    itself holds no references to ConnectionManager, GameStateMachine, or
    GameLogger.
    """

    def __init__(self) -> None:
        # Callbacks — set by main.py during wiring.
        self.on_group_message: (
            Callable[[str, str, int, GamePhase], Awaitable[None]] | None
        ) = None
        self.on_dm: (
            Callable[[str, str, str, int, GamePhase], Awaitable[None]] | None
        ) = None
        self.on_hint: Callable[[str, str, int], Awaitable[None]] | None = None
        self.on_vote: Callable[[str, str, int], Awaitable[None]] | None = None
        self.on_classification: Callable[[str, dict], Awaitable[None]] | None = None
        self.on_survey: Callable[[str, dict], Awaitable[None]] | None = None

        # Optional validator — returns the list of valid player IDs for vote targets.
        self.get_valid_players: Callable[[], list[str]] | None = None

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    async def route(
        self,
        player_id: str,
        raw_message: str,
        current_phase: GamePhase,
        current_round: int,
    ) -> ServerMessage | None:
        """Parse *raw_message* and dispatch to the appropriate handler.

        Returns ``None`` on success (handlers communicate via callbacks) or a
        :class:`ServerMessage` with ``event="error"`` when something goes wrong.
        """
        # 1. Parse JSON into ClientMessage
        try:
            msg = ClientMessage.model_validate_json(raw_message)
        except ValidationError as exc:
            logger.warning("Invalid message from %s: %s", player_id, exc)
            return self._error(f"Invalid message format: {exc.errors()[0]['msg']}")

        # 2. WAIT is a no-op
        if msg.action is ActionType.WAIT:
            return None

        # 3. Phase validation
        allowed = PHASE_ALLOWED_ACTIONS.get(current_phase, set())
        if msg.action not in allowed:
            logger.info(
                "Player %s sent %s during %s (not allowed)",
                player_id,
                msg.action,
                current_phase,
            )
            return self._error(
                f"Action {msg.action} is not allowed during {current_phase} phase"
            )

        # 4. Dispatch to handler
        dispatch: dict[ActionType, Callable[..., Awaitable[ServerMessage | None]]] = {
            ActionType.GROUP_MESSAGE: lambda: self._handle_group_message(
                player_id, msg, current_round, current_phase
            ),
            ActionType.DM: lambda: self._handle_dm(
                player_id, msg, current_round, current_phase
            ),
            ActionType.HINT: lambda: self._handle_hint(
                player_id, msg, current_round
            ),
            ActionType.VOTE: lambda: self._handle_vote(
                player_id, msg, current_round
            ),
            ActionType.CLASSIFICATION: lambda: self._handle_classification(
                player_id, msg
            ),
            ActionType.SURVEY: lambda: self._handle_survey(player_id, msg),
        }

        handler = dispatch.get(msg.action)
        if handler is None:
            return self._error(f"No handler for action {msg.action}")

        return await handler()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_group_message(
        self,
        player_id: str,
        msg: ClientMessage,
        current_round: int,
        current_phase: GamePhase,
    ) -> ServerMessage | None:
        if not msg.content.strip():
            return self._error("Message content cannot be empty")

        if self.on_group_message is not None:
            await self.on_group_message(
                player_id, msg.content, current_round, current_phase
            )
        return None

    async def _handle_dm(
        self,
        player_id: str,
        msg: ClientMessage,
        current_round: int,
        current_phase: GamePhase,
    ) -> ServerMessage | None:
        # target_player is guaranteed present by the ClientMessage validator.
        if not msg.content.strip():
            return self._error("DM content cannot be empty")

        if msg.target_player == player_id:
            logger.warning("Blocked self-DM from %s", player_id)
            return self._error("Cannot DM yourself")

        if self.on_dm is not None:
            await self.on_dm(
                player_id,
                msg.target_player,  # type: ignore[arg-type]
                msg.content,
                current_round,
                current_phase,
            )
        return None

    async def _handle_hint(
        self,
        player_id: str,
        msg: ClientMessage,
        current_round: int,
    ) -> ServerMessage | None:
        # Validate length and word count via HintSubmission
        try:
            submission = HintSubmission(content=msg.content)
        except ValidationError as exc:
            logger.info("Invalid hint from %s: %s", player_id, exc)
            return self._error(f"Invalid hint: {exc.errors()[0]['msg']}")

        # Reject AI-sounding patterns
        for pattern in AI_HINT_PATTERNS:
            if pattern.search(submission.content):
                return self._error(
                    "Hint sounds too verbose — try a shorter, more natural hint"
                )

        if self.on_hint is not None:
            await self.on_hint(player_id, submission.content, current_round)
        return None

    async def _handle_vote(
        self,
        player_id: str,
        msg: ClientMessage,
        current_round: int,
    ) -> ServerMessage | None:
        try:
            vote = VoteSubmission(voter=player_id, voted_for=msg.target_player)  # type: ignore[arg-type]
        except ValidationError as exc:
            logger.info("Invalid vote from %s: %s", player_id, exc)
            return self._error(f"Invalid vote: {exc.errors()[0]['msg']}")

        # Validate that the vote target is a real game player.
        if self.get_valid_players:
            valid = self.get_valid_players()
            if vote.voted_for not in valid:
                logger.warning("Invalid vote target: %s", vote.voted_for)
                return self._error(f"Invalid vote target: {vote.voted_for}")

        if self.on_vote is not None:
            await self.on_vote(player_id, vote.voted_for, current_round)
        return None

    async def _handle_classification(
        self,
        player_id: str,
        msg: ClientMessage,
    ) -> ServerMessage | None:
        try:
            parsed = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError) as exc:
            return self._error(f"Classification content must be valid JSON: {exc}")

        try:
            submission = ClassificationSubmission(
                player_id=player_id, classifications=parsed
            )
        except ValidationError as exc:
            logger.info("Invalid classification from %s: %s", player_id, exc)
            return self._error(f"Invalid classification: {exc.errors()[0]['msg']}")

        if self.on_classification is not None:
            await self.on_classification(player_id, submission.classifications)
        return None

    async def _handle_survey(
        self,
        player_id: str,
        msg: ClientMessage,
    ) -> ServerMessage | None:
        try:
            parsed = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError) as exc:
            return self._error(f"Survey content must be valid JSON: {exc}")

        try:
            submission = SurveyResponse(player_id=player_id, responses=parsed)
        except ValidationError as exc:
            logger.info("Invalid survey from %s: %s", player_id, exc)
            return self._error(f"Invalid survey response: {exc.errors()[0]['msg']}")

        if self.on_survey is not None:
            await self.on_survey(player_id, submission.responses)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error(message: str) -> ServerMessage:
        """Return a ``ServerMessage`` with ``event="error"``."""
        return ServerMessage(event="error", data={"message": message})
