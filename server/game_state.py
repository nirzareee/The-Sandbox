"""Game state machine for The Sandbox — manages phase transitions and scoring."""

from __future__ import annotations

from collections import Counter

from server.models import (
    GameConfig,
    GamePhase,
    GridItem,
    RoundResult,
    RoundState,
)


class GameStateMachine:
    """Drives game progression through phases and tracks scoring across rounds."""

    VALID_TRANSITIONS: dict[GamePhase, GamePhase] = {
        GamePhase.LOBBY: GamePhase.ASSIGNMENT,
        GamePhase.ASSIGNMENT: GamePhase.HINT,
        GamePhase.HINT: GamePhase.DISCUSSION,
        GamePhase.DISCUSSION: GamePhase.VOTING,
        GamePhase.VOTING: GamePhase.RESULTS,
        GamePhase.RESULTS: GamePhase.REFLECTION,  # overridden to META_GAME after last round
        GamePhase.REFLECTION: GamePhase.ASSIGNMENT,  # next round
        GamePhase.META_GAME: GamePhase.SURVEY,
        GamePhase.SURVEY: GamePhase.REVEAL,
        GamePhase.REVEAL: GamePhase.FINISHED,
    }

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.phase: GamePhase = GamePhase.LOBBY
        self.current_round: int = 0
        self.rounds: list[RoundState] = []
        self.current_round_state: RoundState | None = None
        self.cumulative_scores: dict[str, int] = {}
        self.round_results: list[RoundResult] = []

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def advance_phase(self) -> GamePhase:
        """Transition to the next valid phase and return it.

        Special-cases:
        - RESULTS  -> META_GAME when all rounds are complete, else REFLECTION.
        - REFLECTION -> ASSIGNMENT (start the next round).

        Raises ``ValueError`` if no valid transition exists (e.g. from FINISHED).
        """
        if self.phase == GamePhase.RESULTS:
            if self.current_round >= self.config.num_rounds:
                next_phase = GamePhase.META_GAME
            else:
                next_phase = GamePhase.REFLECTION
        elif self.phase in self.VALID_TRANSITIONS:
            next_phase = self.VALID_TRANSITIONS[self.phase]
        else:
            raise ValueError(
                f"No valid transition from phase {self.phase!r}"
            )

        self.phase = next_phase
        return self.phase

    def get_phase_duration(self) -> float:
        """Return the duration (seconds) for the current phase, scaled by the timer multiplier.

        A return value of ``0`` means the phase has no timer.
        """
        base = self.config.phase_durations.get(self.phase.value, 0)
        return base * self.config.timer_multiplier

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def start_round(
        self,
        imposter_id: str,
        topic: str,
        secret_word: str,
        image_grid: list[GridItem],
        hint_order: list[str],
    ) -> None:
        """Begin a new round, creating its ``RoundState``."""
        self.current_round += 1
        round_state = RoundState(
            round_number=self.current_round,
            imposter_id=imposter_id,
            topic=topic,
            secret_word=secret_word,
            image_grid=image_grid,
            hint_order=hint_order,
        )
        self.current_round_state = round_state
        self.rounds.append(round_state)

        # Ensure every player in hint_order has a cumulative score entry.
        for player_id in hint_order:
            self.cumulative_scores.setdefault(player_id, 0)

    # ------------------------------------------------------------------
    # In-round actions
    # ------------------------------------------------------------------

    def record_hint(self, player_id: str, hint: str) -> None:
        """Store a player's hint for the current round."""
        if self.current_round_state is None:
            raise RuntimeError("No active round — call start_round first")
        self.current_round_state.hints_submitted[player_id] = hint

    def record_vote(self, voter_id: str, voted_for: str) -> None:
        """Store a player's vote for the current round."""
        if self.current_round_state is None:
            raise RuntimeError("No active round — call start_round first")
        self.current_round_state.votes[voter_id] = voted_for

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_round_result(self) -> RoundResult:
        """Tally votes, determine if the imposter was caught, compute scores.

        Tie-breaking: if multiple players share the highest vote count, the one
        whose player-id sorts first alphabetically is treated as the accused.
        """
        if self.current_round_state is None:
            raise RuntimeError("No active round — call start_round first")

        rs = self.current_round_state
        votes = rs.votes
        imposter_id = rs.imposter_id
        scoring = self.config.scoring

        # --- Tally votes ---
        vote_counts: Counter[str] = Counter(votes.values())
        if vote_counts:
            accused_id, max_votes = vote_counts.most_common(1)[0]
        else:
            accused_id = ""

        imposter_caught = accused_id == imposter_id

        # --- Per-player scores for this round ---
        scores_this_round: dict[str, int] = {}
        correct_voters: list[str] = []

        for voter_id, target_id in votes.items():
            if target_id == imposter_id:
                scores_this_round[voter_id] = scores_this_round.get(voter_id, 0) + scoring["correct_vote"]
                correct_voters.append(voter_id)

        if not imposter_caught:
            # Imposter evaded detection.
            scores_this_round[imposter_id] = scores_this_round.get(imposter_id, 0) + scoring["imposter_survives"]

            # The wrongly accused player suffers a penalty (only if they are not the imposter).
            if accused_id and accused_id != imposter_id:
                scores_this_round[accused_id] = scores_this_round.get(accused_id, 0) + scoring["wrongly_accused"]

        # --- Update cumulative scores ---
        for pid, delta in scores_this_round.items():
            self.cumulative_scores[pid] = self.cumulative_scores.get(pid, 0) + delta

        # Also store in the round state for convenience.
        rs.scores = dict(scores_this_round)

        result = RoundResult(
            round_number=rs.round_number,
            imposter_id=imposter_id,
            imposter_caught=imposter_caught,
            votes=dict(votes),
            correct_voters=sorted(correct_voters),
            scores_this_round=scores_this_round,
            cumulative_scores=dict(self.cumulative_scores),
        )
        self.round_results.append(result)
        return result

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_scores(self) -> dict[str, int]:
        """Return a copy of the cumulative score table."""
        return dict(self.cumulative_scores)

    def get_state_snapshot(self) -> dict:
        """Return a serialisable snapshot of the current game state."""
        return {
            "session_id": self.config.session_id,
            "phase": self.phase.value,
            "current_round": self.current_round,
            "current_round_state": (
                self.current_round_state.model_dump() if self.current_round_state else None
            ),
            "cumulative_scores": dict(self.cumulative_scores),
            "total_rounds": self.config.num_rounds,
        }

    def is_game_over(self) -> bool:
        """Return ``True`` when the game has reached the FINISHED phase."""
        return self.phase == GamePhase.FINISHED
