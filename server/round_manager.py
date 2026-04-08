"""Round manager — orchestrates the complete game lifecycle for The Sandbox.

This is the central coordinator: it drives 5 rounds of play (assignment,
hints, discussion, voting, results) followed by post-game phases
(reflection, meta-game classification, survey, and the grand reveal).

The RoundManager does NOT handle WebSocket message routing — that is the
responsibility of MessageRouter / main.py.  Instead it exposes callback
methods (record_classification, record_survey, receive_hint, receive_vote)
that the router invokes when player messages arrive.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone

from server.config import NUM_ROUNDS, HINT_MAX_CHARS, HINT_MAX_WORDS
from server.connection_manager import ConnectionManager
from server.game_state import GameStateMachine
from server.logger import GameLogger
from server.models import (
    ClassificationSubmission,
    GamePhase,
    PlayerRole,
    PlayerType,
    ServerMessage,
)
from server.scoring import ScoringEngine
from server.word_bank import WordBank

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper to build a ServerMessage quickly
# ---------------------------------------------------------------------------

def _msg(event: str, data: dict | None = None) -> ServerMessage:
    return ServerMessage(
        event=event,
        data=data or {},
        timestamp=datetime.now(timezone.utc),
    )


# ===================================================================
# RoundManager
# ===================================================================

class RoundManager:
    """Orchestrates the complete game lifecycle — 5 rounds plus post-game phases."""

    def __init__(
        self,
        game_state: GameStateMachine,
        connection_manager: ConnectionManager,
        logger: GameLogger,
        word_bank: WordBank,
        scoring: ScoringEngine,
    ) -> None:
        self.game_state = game_state
        self.connection_manager = connection_manager
        self.logger = logger
        self.word_bank = word_bank
        self.scoring = scoring

        # Collected during META_GAME phase
        self._classifications: dict[str, ClassificationSubmission] = {}
        # Collected during SURVEY phase
        self._surveys: dict[str, dict] = {}

        # Signalled when all expected submissions arrive (early phase completion)
        self._phase_event: asyncio.Event = asyncio.Event()

        # Per-player vote tracking for early voting completion
        self._votes_received: set[str] = set()

        # Concurrent hint collection
        self._hints_received: dict[str, str] = {}
        self._all_hints_event: asyncio.Event = asyncio.Event()

        # Lock protecting shared mutable state accessed from both the game loop
        # and WebSocket handler callbacks (_hints_received, _votes_received,
        # _classifications, _surveys).
        self._lock = asyncio.Lock()

        # Without-replacement imposter tracking (issue #7)
        self._past_imposters: list[str] = []

    # ===================================================================
    # Master game loop
    # ===================================================================

    async def run_game(self) -> None:
        """Run the full game: 5 rounds, then post-game phases."""
        num_rounds = self.game_state.config.num_rounds
        log.info("Starting game with %d rounds", num_rounds)

        # Transition from LOBBY -> ASSIGNMENT
        self.game_state.advance_phase()
        await self.logger.log_event("game_started", {"num_rounds": num_rounds})

        for round_num in range(1, num_rounds + 1):
            log.info("=== Round %d / %d ===", round_num, num_rounds)
            await self.run_round()

            is_last_round = round_num == num_rounds
            if not is_last_round:
                # RESULTS -> REFLECTION
                self.game_state.advance_phase()
                await self.run_timed_phase(
                    "reflection",
                    time_warnings=[30, 10],
                )
                # REFLECTION -> ASSIGNMENT (for next round)
                self.game_state.advance_phase()

        # --- Post-game phases ---

        # RESULTS -> META_GAME (advance_phase handles the last-round special case)
        self.game_state.advance_phase()
        await self.run_meta_game()

        # META_GAME -> SURVEY
        self.game_state.advance_phase()
        await self.run_survey()

        # SURVEY -> REVEAL
        self.game_state.advance_phase()
        await self.run_grand_reveal()

        # REVEAL -> FINISHED
        self.game_state.advance_phase()
        await self.connection_manager.broadcast(
            _msg("game_finished", {
                "final_scores": self.game_state.get_scores(),
            })
        )
        await self.logger.log_event("game_finished", {
            "final_scores": self.game_state.get_scores(),
        })

        log.info("Game finished.")

    # ===================================================================
    # Single round
    # ===================================================================

    async def run_round(self) -> None:
        """Execute a single round: assignment, hints, discussion, voting, results."""

        connected = self.connection_manager.get_game_players()
        if not connected:
            log.error("No connected players — cannot run round")
            return

        # --- Pick imposter and hint order (without-replacement) ---
        candidates = [p for p in connected if p not in self._past_imposters]
        if not candidates:
            # Everyone has been imposter — reset the cycle
            self._past_imposters.clear()
            candidates = list(connected)
        imposter_id = random.choice(candidates)
        self._past_imposters.append(imposter_id)
        hint_order = random.sample(connected, len(connected))

        # --- Get word bank entry (shuffled grid) ---
        round_number = self.game_state.current_round + 1  # will be incremented by start_round
        entry = self.word_bank.get_entry_for_round(round_number)
        grid_labels = self.word_bank.get_grid_labels(entry)

        # --- Initialize round state ---
        self.game_state.start_round(
            imposter_id=imposter_id,
            topic=entry.topic,
            secret_word=entry.secret_word,
            image_grid=entry.grid_items,
            hint_order=hint_order,
        )

        await self.logger.log_event("round_started", {
            "round_number": self.game_state.current_round,
            "imposter_id": imposter_id,
            "topic": entry.topic,
            "secret_word": entry.secret_word,
            "hint_order": hint_order,
        })

        # --- Broadcast role assignments (different data for imposter vs insiders) ---
        await self._broadcast_round_start(
            imposter_id=imposter_id,
            topic=entry.topic,
            secret_word=entry.secret_word,
            grid_labels=grid_labels,
            hint_order=hint_order,
        )

        # Wait for ASSIGNMENT duration so players can read their role
        assignment_duration = self.game_state.get_phase_duration()
        if assignment_duration > 0:
            await asyncio.sleep(assignment_duration)

        # --- HINT phase ---
        self.game_state.advance_phase()  # ASSIGNMENT -> HINT
        await self.run_hint_phase(hint_order)

        # --- DISCUSSION phase ---
        self.game_state.advance_phase()  # HINT -> DISCUSSION
        await self.run_timed_phase(
            "discussion",
            time_warnings=[60, 30, 10],
        )

        # --- VOTING phase ---
        self.game_state.advance_phase()  # DISCUSSION -> VOTING
        await self.run_voting_phase()

        # --- RESULTS phase ---
        self.game_state.advance_phase()  # VOTING -> RESULTS
        await self.show_results()

    # ===================================================================
    # Round-start broadcast (role-specific)
    # ===================================================================

    async def _broadcast_round_start(
        self,
        imposter_id: str,
        topic: str,
        secret_word: str,
        grid_labels: list[str],
        hint_order: list[str],
    ) -> None:
        """Send round_start to every game player — imposters do NOT learn the secret word."""
        round_number = self.game_state.current_round

        # Send role-specific data to game players only (skip hosts)
        for player_id in self.connection_manager.get_game_players():
            is_imposter = player_id == imposter_id
            role = PlayerRole.IMPOSTER if is_imposter else PlayerRole.INSIDER

            data = {
                "round_number": round_number,
                "role": role.value,
                "topic": topic,
                "secret_word": None if is_imposter else secret_word,
                "image_grid": grid_labels,
                "hint_order": hint_order,
            }
            await self.connection_manager.send_personal(
                player_id, _msg("round_start", data)
            )

        # Send hosts the full picture — they see everything
        host_data = {
            "round_number": round_number,
            "role": "HOST",
            "topic": topic,
            "secret_word": secret_word,
            "image_grid": grid_labels,
            "hint_order": hint_order,
            "imposter_id": imposter_id,
        }
        await self.connection_manager.send_to_hosts(_msg("round_start", host_data))

        log.info(
            "Round %d started — imposter=%s, topic=%s",
            round_number, imposter_id, topic,
        )

    # ===================================================================
    # Hint phase (concurrent — all players submit freely)
    # ===================================================================

    async def run_hint_phase(self, hint_order: list[str]) -> None:
        """Run the hint phase: all players submit hints concurrently."""
        total_duration = self.game_state.get_phase_duration()
        round_number = self.game_state.current_round
        connected = self.connection_manager.get_game_players()

        self._hints_received.clear()
        self._all_hints_event.clear()

        await self.connection_manager.broadcast(
            _msg("hint_phase_start", {
                "round_number": round_number,
                "hint_order": hint_order,
                "total_duration": total_duration,
                "expected_hints": len(connected),
                "max_chars": HINT_MAX_CHARS,
                "max_words": HINT_MAX_WORDS,
            })
        )

        # Wait for all hints or timeout
        try:
            await asyncio.wait_for(
                self._all_hints_event.wait(), timeout=total_duration
            )
        except asyncio.TimeoutError:
            # Auto-submit "..." for anyone who didn't submit
            async with self._lock:
                for pid in connected:
                    if pid not in self._hints_received:
                        self._hints_received[pid] = "..."
                        self.game_state.record_hint(pid, "...")
                        await self.connection_manager.broadcast(
                            _msg("hint_submitted", {
                                "player_id": pid,
                                "hint": "...",
                            })
                        )
                        await self.logger.log_hint(pid, "...", round_number)
                        log.warning(
                            "Player %s timed out during hint phase — auto-submitting '...'",
                            pid,
                        )

        # Reading period — broadcast all_hints_in, then wait
        reading_period = round(10 * self.game_state.config.timer_multiplier)
        await self.connection_manager.broadcast(
            _msg("all_hints_in", {
                "round_number": round_number,
                "reading_period": reading_period,
            })
        )

        if reading_period > 0:
            await asyncio.sleep(reading_period)

        # Save all hints for the round
        hints_data = dict(self.game_state.current_round_state.hints_submitted)  # type: ignore[union-attr]
        await self.logger.save_round_hints(round_number, {
            "round_number": round_number,
            "hints": hints_data,
        })

        await self.connection_manager.broadcast(
            _msg("hint_phase_end", {"round_number": round_number})
        )

    async def receive_hint(self, player_id: str, hint: str) -> None:
        """Called by the message router when a player submits a hint.

        Stores the hint and checks if all players have submitted.
        """
        async with self._lock:
            if player_id in self._hints_received:
                log.warning("Duplicate hint from %s — ignoring", player_id)
                return

            self._hints_received[player_id] = hint
            self.game_state.record_hint(player_id, hint)

            # Broadcast and log asynchronously
            round_number = self.game_state.current_round
            asyncio.create_task(self._process_hint(player_id, hint, round_number))

    async def _process_hint(self, player_id: str, hint: str, round_number: int) -> None:
        """Broadcast a hint submission and check for completion."""
        await self.connection_manager.broadcast(
            _msg("hint_submitted", {
                "player_id": player_id,
                "hint": hint,
            })
        )
        await self.logger.log_hint(player_id, hint, round_number)

        # Check if all game players have submitted (exclude hosts)
        connected = self.connection_manager.get_game_players()
        if all(pid in self._hints_received for pid in connected):
            log.info("All connected players have submitted hints — ending hint phase early")
            self._all_hints_event.set()

    # ===================================================================
    # Generic timed phase (DISCUSSION / REFLECTION)
    # ===================================================================

    async def run_timed_phase(
        self,
        phase_name: str,
        time_warnings: list[int] | None = None,
    ) -> None:
        """Run a generic timed phase with optional countdown warnings."""
        duration = self.game_state.get_phase_duration()
        round_number = self.game_state.current_round

        self._phase_event.clear()

        await self.connection_manager.broadcast(
            _msg(f"{phase_name}_start", {
                "round_number": round_number,
                "duration": duration,
            })
        )

        log.info(
            "Phase '%s' started (round %d, %.0fs)",
            phase_name, round_number, duration,
        )

        if time_warnings:
            # Run phase timer and time warnings concurrently
            timer_task = asyncio.create_task(self._wait_for_phase(duration))
            warning_task = asyncio.create_task(
                self._broadcast_time_warnings(duration, time_warnings)
            )
            await timer_task
            warning_task.cancel()
            try:
                await warning_task
            except asyncio.CancelledError:
                pass
        else:
            await self._wait_for_phase(duration)

        await self.connection_manager.broadcast(
            _msg(f"{phase_name}_end", {
                "round_number": round_number,
            })
        )

        log.info("Phase '%s' ended (round %d)", phase_name, round_number)

    # ===================================================================
    # Voting phase
    # ===================================================================

    async def run_voting_phase(self) -> None:
        """Run the voting phase — wait for all votes or timeout."""
        duration = self.game_state.get_phase_duration()
        round_number = self.game_state.current_round
        connected = self.connection_manager.get_game_players()

        self._phase_event.clear()
        self._votes_received.clear()

        await self.connection_manager.broadcast(
            _msg("voting_start", {
                "round_number": round_number,
                "candidates": connected,
                "duration": duration,
            })
        )

        log.info("Voting started (round %d, %.0fs)", round_number, duration)

        await self._wait_for_phase(duration)

        await self.connection_manager.broadcast(
            _msg("voting_end", {"round_number": round_number})
        )

        # Save votes
        async with self._lock:
            votes = dict(self.game_state.current_round_state.votes)  # type: ignore[union-attr]
        await self.logger.save_round_votes(round_number, {
            "round_number": round_number,
            "votes": votes,
        })

        log.info("Voting ended (round %d) — %d votes cast", round_number, len(votes))

    async def receive_vote(self, voter_id: str, voted_for: str) -> None:
        """Called by the message router when a player casts a vote.

        Records the vote in game state and checks for early completion.
        """
        async with self._lock:
            self.game_state.record_vote(voter_id, voted_for)
            self._votes_received.add(voter_id)

            connected = self.connection_manager.get_game_players()
            if self._votes_received >= set(connected):
                log.info("All connected players have voted — ending voting early")
                self._phase_event.set()

    # ===================================================================
    # Results display
    # ===================================================================

    async def show_results(self) -> None:
        """Compute and broadcast round results, then pause for RESULTS duration."""
        result = self.game_state.compute_round_result()
        round_number = self.game_state.current_round

        result_data = {
            "round_number": result.round_number,
            "imposter_id": result.imposter_id,
            "imposter_caught": result.imposter_caught,
            "votes": result.votes,
            "correct_voters": result.correct_voters,
            "scores_this_round": result.scores_this_round,
            "cumulative_scores": result.cumulative_scores,
        }

        await self.connection_manager.broadcast(
            _msg("round_results", result_data)
        )

        await self.logger.log_event("round_results", result_data)

        log.info(
            "Round %d results — imposter %s %s",
            round_number,
            result.imposter_id,
            "caught!" if result.imposter_caught else "survived!",
        )

        # Hold the results screen for the configured duration
        duration = self.game_state.get_phase_duration()
        if duration > 0:
            await asyncio.sleep(duration)

    # ===================================================================
    # Meta-game (classify human vs AI)
    # ===================================================================

    async def run_meta_game(self) -> None:
        """Run the meta-game classification phase."""
        duration = self.game_state.get_phase_duration()
        connected = self.connection_manager.get_game_players()

        self._phase_event.clear()
        self._classifications.clear()

        players_info = {
            pid: {
                "player_id": pid,
                "display_name": self.connection_manager.player_info[pid].display_name,
            }
            for pid in connected
        }

        await self.connection_manager.broadcast(
            _msg("meta_game_start", {
                "players": players_info,
                "duration": duration,
            })
        )

        log.info("Meta-game phase started (%.0fs)", duration)

        await self._wait_for_phase(duration)

        await self.connection_manager.broadcast(
            _msg("meta_game_end", {
                "submissions_received": len(self._classifications),
            })
        )

        log.info(
            "Meta-game ended — %d/%d classifications received",
            len(self._classifications), len(connected),
        )

    async def record_classification(self, player_id: str, classifications: dict) -> None:
        """Called by the message router when a player submits classifications.

        Stores the submission and checks for early phase completion.
        """
        async with self._lock:
            self._classifications[player_id] = ClassificationSubmission(
                player_id=player_id,
                classifications=classifications,
            )

            connected = self.connection_manager.get_game_players()
            if len(self._classifications) >= len(connected):
                log.info("All connected players have submitted classifications")
                self._phase_event.set()

    # ===================================================================
    # Survey (human players only)
    # ===================================================================

    async def run_survey(self) -> None:
        """Run the post-game survey phase (primarily for human players)."""
        duration = self.game_state.get_phase_duration()

        self._phase_event.clear()
        self._surveys.clear()

        # Identify human players
        human_players = [
            pid
            for pid, info in self.connection_manager.player_info.items()
            if info.player_type == PlayerType.HUMAN and info.is_connected
        ]

        await self.connection_manager.broadcast(
            _msg("survey_start", {
                "duration": duration,
                "human_player_count": len(human_players),
            })
        )

        log.info(
            "Survey phase started (%.0fs, %d human players)",
            duration, len(human_players),
        )

        # If no human players, skip waiting
        if human_players:
            await self._wait_for_phase(duration)
        else:
            log.info("No human players — skipping survey wait")

        await self.connection_manager.broadcast(
            _msg("survey_end", {
                "submissions_received": len(self._surveys),
            })
        )

        log.info(
            "Survey ended — %d/%d surveys received",
            len(self._surveys), len(human_players),
        )

    async def record_survey(self, player_id: str, responses: dict) -> None:
        """Called by the message router when a human player submits the survey.

        Stores the response and checks for early phase completion.
        """
        async with self._lock:
            self._surveys[player_id] = responses

            # Check if all human players have submitted
            human_players = [
                pid
                for pid, info in self.connection_manager.player_info.items()
                if info.player_type == PlayerType.HUMAN and info.is_connected
            ]

            if len(self._surveys) >= len(human_players):
                log.info("All human players have submitted surveys")
                self._phase_event.set()

    # ===================================================================
    # Grand reveal
    # ===================================================================

    async def run_grand_reveal(self) -> None:
        """Compute final scores, build confusion matrix, and broadcast the reveal."""
        duration = self.game_state.get_phase_duration()

        # --- Build ground truth ---
        ground_truth = self.scoring.build_ground_truth(
            self.connection_manager.player_info
        )

        # --- Compute classification scores ---
        classification_scores = self.scoring.compute_classification_scores(
            self._classifications, ground_truth
        )

        # --- Compute final scores (round scores + classification scores) ---
        cumulative_round_scores = self.game_state.get_scores()
        final_scores = self.scoring.compute_final_scores(
            cumulative_round_scores, classification_scores
        )

        # --- Build confusion matrix ---
        confusion_matrix = self.scoring.build_confusion_matrix(
            self._classifications, ground_truth
        )

        # --- Build leaderboard ---
        leaderboard = self.scoring.get_leaderboard(final_scores)

        # --- Build identity reveal data (exclude hosts) ---
        identities = {}
        for pid, info in self.connection_manager.player_info.items():
            if info.player_type == PlayerType.HOST:
                continue
            identities[pid] = {
                "display_name": info.display_name,
                "type": info.player_type.value,
                "model_name": info.model_name,
            }

        # --- Build per-player classification guesses ---
        all_classifications = {}
        for pid, sub in self._classifications.items():
            all_classifications[pid] = {
                target: {
                    "guess": c.guess,
                    "confidence": c.confidence,
                    "reasoning": c.reasoning,
                }
                for target, c in sub.classifications.items()
            }

        reveal_data = {
            "identities": identities,
            "classification_scores": classification_scores,
            "cumulative_round_scores": cumulative_round_scores,
            "final_scores": final_scores,
            "leaderboard": leaderboard,
            "confusion_matrix": confusion_matrix,
            "all_classifications": all_classifications,
        }

        await self.connection_manager.broadcast(
            _msg("grand_reveal", reveal_data)
        )

        log.info("Grand reveal broadcast — final leaderboard: %s", leaderboard)

        # --- Persist results ---
        ground_truth_serializable = {
            pid: ptype.value for pid, ptype in ground_truth.items()
        }
        await self.logger.save_ground_truth(ground_truth_serializable)

        await self.logger.save_meta_game_results({
            "confusion_matrix": confusion_matrix,
            "classification_scores": classification_scores,
            "final_scores": final_scores,
            "leaderboard": leaderboard,
        })

        await self.logger.log_event("grand_reveal", {
            "final_scores": final_scores,
            "leaderboard": leaderboard,
        })

        # Hold the reveal screen for the configured duration
        if duration > 0:
            await asyncio.sleep(duration)

    # ===================================================================
    # Internal helpers
    # ===================================================================

    async def _wait_for_phase(self, duration: float) -> None:
        """Wait for either the _phase_event to be set or *duration* seconds to elapse.

        The _phase_event is cleared before returning so it is ready for the
        next phase.
        """
        try:
            await asyncio.wait_for(self._phase_event.wait(), timeout=duration)
        except asyncio.TimeoutError:
            pass
        finally:
            self._phase_event.clear()

    async def _broadcast_time_warnings(
        self,
        duration: float,
        warnings: list[int],
    ) -> None:
        """Broadcast time_warning events at the specified seconds-remaining thresholds.

        For example, if duration=180 and warnings=[60, 30, 10]:
          - At 120s elapsed (60s remaining): broadcast warning
          - At 150s elapsed (30s remaining): broadcast warning
          - At 170s elapsed (10s remaining): broadcast warning

        This coroutine is designed to run concurrently with the phase timer
        and should be cancelled when the phase ends.
        """
        # Sort warnings descending so we process the largest remaining-time first
        sorted_warnings = sorted(warnings, reverse=True)
        elapsed = 0.0

        for seconds_remaining in sorted_warnings:
            if seconds_remaining >= duration:
                # This warning threshold exceeds the total duration — skip it
                continue

            target_elapsed = duration - seconds_remaining
            sleep_time = target_elapsed - elapsed

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                elapsed = target_elapsed

            await self.connection_manager.broadcast(
                _msg("time_warning", {
                    "seconds_remaining": seconds_remaining,
                })
            )

            log.debug("Time warning: %ds remaining", seconds_remaining)
