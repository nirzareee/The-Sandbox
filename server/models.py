"""Pydantic data models for The Sandbox game."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from server.config import (
    GRID_SIZE,
    HINT_MAX_CHARS,
    HINT_MAX_WORDS,
    MAX_PLAYERS,
    NUM_ROUNDS,
    PHASE_DURATIONS,
    SCORING,
    TIMER_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GamePhase(StrEnum):
    LOBBY = "LOBBY"
    ASSIGNMENT = "ASSIGNMENT"
    HINT = "HINT"
    DISCUSSION = "DISCUSSION"
    VOTING = "VOTING"
    RESULTS = "RESULTS"
    REFLECTION = "REFLECTION"
    META_GAME = "META_GAME"
    SURVEY = "SURVEY"
    REVEAL = "REVEAL"
    FINISHED = "FINISHED"


class PlayerRole(StrEnum):
    INSIDER = "INSIDER"
    IMPOSTER = "IMPOSTER"


class PlayerType(StrEnum):
    HUMAN = "HUMAN"
    AI = "AI"
    HOST = "HOST"


class ActionType(StrEnum):
    GROUP_MESSAGE = "GROUP_MESSAGE"
    DM = "DM"
    HINT = "HINT"
    VOTE = "VOTE"
    CLASSIFICATION = "CLASSIFICATION"
    SURVEY = "SURVEY"
    WAIT = "WAIT"


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

class PlayerInfo(BaseModel):
    player_id: str
    display_name: str
    player_type: PlayerType
    model_name: Optional[str] = None
    is_connected: bool = True

    @model_validator(mode="after")
    def validate_model_name(self) -> PlayerInfo:
        if self.player_type == PlayerType.AI and not self.model_name:
            raise ValueError("AI players must have a model_name")
        if self.player_type in (PlayerType.HUMAN, PlayerType.HOST) and self.model_name:
            raise ValueError("Human/Host players must not have a model_name")
        return self


class GridItem(BaseModel):
    label: str
    image_path: str = ""


class WordBankEntry(BaseModel):
    round_id: int
    topic: str
    secret_word: str
    grid_items: list[GridItem] = Field(min_length=GRID_SIZE, max_length=GRID_SIZE)

    @model_validator(mode="after")
    def validate_secret_in_grid(self) -> WordBankEntry:
        labels = [item.label for item in self.grid_items]
        if self.secret_word not in labels:
            raise ValueError(
                f"secret_word '{self.secret_word}' must be in grid labels: {labels}"
            )
        return self


class GameConfig(BaseModel):
    session_id: str
    max_players: int = MAX_PLAYERS
    num_rounds: int = NUM_ROUNDS
    phase_durations: dict[str, int] = Field(default_factory=lambda: dict(PHASE_DURATIONS))
    scoring: dict[str, int] = Field(default_factory=lambda: dict(SCORING))
    timer_multiplier: float = TIMER_MULTIPLIER
    player_assignments: dict[str, PlayerRole] = Field(default_factory=dict)


class RoundState(BaseModel):
    round_number: int
    phase: GamePhase = GamePhase.ASSIGNMENT
    imposter_id: str
    topic: str
    secret_word: str
    image_grid: list[GridItem] = Field(min_length=GRID_SIZE, max_length=GRID_SIZE)
    hint_order: list[str] = Field(default_factory=list)
    hints_submitted: dict[str, str] = Field(default_factory=dict)
    votes: dict[str, str] = Field(default_factory=dict)
    scores: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# WebSocket Protocol Models
# ---------------------------------------------------------------------------

class ServerMessage(BaseModel):
    event: str
    data: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ClientMessage(BaseModel):
    action: ActionType
    content: str = ""
    target_player: Optional[str] = None

    @model_validator(mode="after")
    def validate_target_required(self) -> ClientMessage:
        if self.action in (ActionType.DM, ActionType.VOTE) and not self.target_player:
            raise ValueError(f"{self.action} requires a target_player")
        return self


class ChatMessage(BaseModel):
    player_id: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    round_number: int
    phase: GamePhase
    is_dm: bool = False
    target_player: Optional[str] = None

    @model_validator(mode="after")
    def validate_dm_target(self) -> ChatMessage:
        if self.is_dm and not self.target_player:
            raise ValueError("DMs must have a target_player")
        return self


# ---------------------------------------------------------------------------
# Action Models
# ---------------------------------------------------------------------------

class HintSubmission(BaseModel):
    content: str = Field(max_length=HINT_MAX_CHARS)

    @field_validator("content")
    @classmethod
    def validate_hint(cls, v: str) -> str:
        v = v.strip()
        word_count = len(v.split())
        if word_count > HINT_MAX_WORDS:
            raise ValueError(
                f"Hint must be {HINT_MAX_WORDS} words or fewer, got {word_count}"
            )
        return v


class VoteSubmission(BaseModel):
    voter: str
    voted_for: str

    @model_validator(mode="after")
    def validate_no_self_vote(self) -> VoteSubmission:
        if self.voter == self.voted_for:
            raise ValueError("Players cannot vote for themselves")
        return self


class PlayerClassification(BaseModel):
    guess: str = Field(pattern=r"^(human|ai)$")
    confidence: int = Field(ge=1, le=5)
    reasoning: str


class ClassificationSubmission(BaseModel):
    player_id: str
    classifications: dict[str, PlayerClassification]

    @model_validator(mode="after")
    def validate_no_self_classify(self) -> ClassificationSubmission:
        if self.player_id in self.classifications:
            raise ValueError("Players cannot classify themselves")
        return self


class SurveyResponse(BaseModel):
    player_id: str
    responses: dict[str, str | int | list[str]]


# ---------------------------------------------------------------------------
# Result Model
# ---------------------------------------------------------------------------

class RoundResult(BaseModel):
    round_number: int
    imposter_id: str
    imposter_caught: bool
    votes: dict[str, str]
    correct_voters: list[str]
    scores_this_round: dict[str, int]
    cumulative_scores: dict[str, int]
