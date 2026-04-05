"""
Game state: phases, player model, round state, scoring engine.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from fastapi import WebSocket


class Phase(str, Enum):
    LOBBY = "lobby"
    ASSIGNMENT = "assignment"
    HINT = "hint"
    HINT_REVIEW = "hint_review"
    DISCUSSION = "discussion"
    VOTING = "voting"
    RESULTS = "results"
    REFLECTION = "reflection"
    META_GAME = "meta_game"
    SURVEY = "survey"
    REVEAL = "reveal"
    FINISHED = "finished"


@dataclass
class Player:
    player_id: str
    websocket: WebSocket
    is_human: bool
    display_name: str = ""
    score: int = 0
    connected: bool = True


@dataclass
class RoundState:
    round_number: int
    imposter_id: str
    topic: str
    secret_word: str
    grid_items: list[dict]
    hints: dict = field(default_factory=dict)
    votes: dict = field(default_factory=dict)


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
        self.past_imposters: list[str] = []
        self.cumulative_scores: dict[str, int] = {}
        self.classifications: dict[str, dict] = {}
        self.survey_responses: dict[str, dict] = {}

    def add_player(self, player_id: str, websocket: WebSocket, is_human: bool) -> bool:
        if len(self.players) >= self.MAX_PLAYERS:
            return False
        self.players[player_id] = Player(player_id, websocket, is_human)
        return True

    def is_full(self) -> bool:
        return len(self.players) >= self.MAX_PLAYERS

    def player_ids(self) -> list[str]:
        return [pid for pid in self.players if pid != "HOST"]

    def connected_players(self) -> dict[str, Player]:
        return {pid: p for pid, p in self.players.items() if p.connected}

    def game_players(self) -> dict[str, Player]:
        """All non-HOST players."""
        return {pid: p for pid, p in self.players.items() if pid != "HOST"}

    # ── Scoring ──────────────────────────────────────────────────────

    def compute_round_scores(self) -> dict:
        rs = self.round_state
        if not rs:
            return {}

        imposter_id = rs.imposter_id
        votes = rs.votes

        tallies: dict[str, int] = {}
        for target in votes.values():
            tallies[target] = tallies.get(target, 0) + 1

        max_votes = max(tallies.values()) if tallies else 0
        top_targets = sorted([t for t, c in tallies.items() if c == max_votes])
        caught = top_targets[0] == imposter_id if top_targets else False

        scores: dict[str, int] = {}
        correct_voters = []

        for pid in self.player_ids():
            scores[pid] = 0
            if pid == imposter_id:
                if not caught:
                    scores[pid] = 3
            else:
                if votes.get(pid) == imposter_id:
                    scores[pid] = 2
                    correct_voters.append(pid)

        for pid, pts in scores.items():
            self.cumulative_scores[pid] = self.cumulative_scores.get(pid, 0) + pts

        return {
            "scores_this_round": scores,
            "cumulative_scores": dict(self.cumulative_scores),
            "imposter_caught": caught,
            "correct_voters": correct_voters,
            "votes": votes,
            "imposter_id": imposter_id,
        }

    def compute_classification_scores(self) -> dict[str, int]:
        scores: dict[str, int] = {}
        for voter_id, guesses in self.classifications.items():
            total = 0
            for target_id, data in guesses.items():
                target = self.players.get(target_id)
                if not target:
                    continue
                actual = "human" if target.is_human else "ai"
                if data.get("guess") == actual:
                    total += 5
            scores[voter_id] = total
        return scores

    def build_confusion_matrix(self) -> dict:
        agg = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for voter_id, guesses in self.classifications.items():
            for target_id, data in guesses.items():
                target = self.players.get(target_id)
                if not target:
                    continue
                actual_ai = not target.is_human
                guess = data.get("guess", "")
                if guess == "ai" and actual_ai:
                    agg["tp"] += 1
                elif guess == "human" and not actual_ai:
                    agg["tn"] += 1
                elif guess == "ai" and not actual_ai:
                    agg["fp"] += 1
                elif guess == "human" and actual_ai:
                    agg["fn"] += 1
        total = sum(agg.values())
        agg["accuracy"] = (agg["tp"] + agg["tn"]) / total if total > 0 else 0.0
        return {"aggregate": agg}

    def build_leaderboard(self) -> list[dict]:
        cls_scores = self.compute_classification_scores()
        combined = {}
        for pid in self.player_ids():
            combined[pid] = self.cumulative_scores.get(pid, 0) + cls_scores.get(pid, 0)

        sorted_players = sorted(combined.items(), key=lambda x: -x[1])
        leaderboard = []
        rank = 1
        for i, (pid, score) in enumerate(sorted_players):
            if i > 0 and score < sorted_players[i - 1][1]:
                rank = i + 1
            leaderboard.append({"player_id": pid, "score": score, "rank": rank})
        return leaderboard
