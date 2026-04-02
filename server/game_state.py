from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from fastapi import WebSocket


class Phase(str, Enum):
    LOBBY = "lobby"
    ASSIGNMENT = "assignment"
    HINT = "hint"
    DISCUSSION = "discussion"
    VOTING = "voting"
    RESULTS = "results"
    REFLECTION = "reflection"
    META_GAME = "meta_game"
    FINISHED = "finished"


@dataclass
class Player:
    player_id: str
    websocket: WebSocket
    is_human: bool
    score: int = 0
    connected: bool = True


@dataclass
class RoundState:
    round_number: int
    imposter_id: str
    topic: str
    secret_word: str
    grid_items: list[dict]   # [{"label": str, "image": str}, ...]
    hints: dict = field(default_factory=dict)   # player_id -> hint string
    votes: dict = field(default_factory=dict)   # player_id -> voted_player_id


class GameSession:
    MAX_PLAYERS = 6
    MAX_ROUNDS = 5

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.players: dict[str, Player] = {}
        self.phase = Phase.LOBBY
        self.current_round = 0
        self.round_state: Optional[RoundState] = None
        self.used_word_indices: set[int] = set()

    def add_player(self, player_id: str, websocket: WebSocket, is_human: bool) -> bool:
        if len(self.players) >= self.MAX_PLAYERS:
            return False
        self.players[player_id] = Player(player_id, websocket, is_human)
        return True

    def is_full(self) -> bool:
        return len(self.players) >= self.MAX_PLAYERS

    def player_ids(self) -> list[str]:
        return list(self.players.keys())

    def connected_players(self) -> dict[str, Player]:
        return {pid: p for pid, p in self.players.items() if p.connected}
