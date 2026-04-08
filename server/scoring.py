from __future__ import annotations

from server.config import SCORING
from server.models import ClassificationSubmission, PlayerInfo, PlayerType


class ScoringEngine:
    """Computes classification scores and aggregates final results."""

    def __init__(self, scoring_config: dict[str, int] | None = None) -> None:
        self.scoring = scoring_config if scoring_config is not None else SCORING

    def compute_classification_scores(
        self,
        classifications: dict[str, ClassificationSubmission],
        ground_truth: dict[str, PlayerType],
    ) -> dict[str, int]:
        """Score each player's classification guesses against ground truth.

        For each correct guess (human identified as human, or AI identified
        as AI), the player earns ``correct_classification`` points.
        """
        scores: dict[str, int] = {}
        points = self.scoring["correct_classification"]

        for player_id, submission in classifications.items():
            total = 0
            for target_id, classification in submission.classifications.items():
                guess = classification.guess.lower()
                actual = ground_truth.get(target_id)
                if actual is None:
                    continue
                if (
                    (guess == "human" and actual == PlayerType.HUMAN)
                    or (guess == "ai" and actual == PlayerType.AI)
                ):
                    total += points
            scores[player_id] = total

        return scores

    def compute_final_scores(
        self,
        cumulative_round_scores: dict[str, int],
        classification_scores: dict[str, int],
    ) -> dict[str, int]:
        """Combine round scores and classification scores per player."""
        all_players = set(cumulative_round_scores) | set(classification_scores)
        return {
            pid: cumulative_round_scores.get(pid, 0) + classification_scores.get(pid, 0)
            for pid in all_players
        }

    def build_ground_truth(
        self, player_info: dict[str, PlayerInfo]
    ) -> dict[str, PlayerType]:
        """Extract player_id -> PlayerType mapping from PlayerInfo records (excludes hosts)."""
        return {
            pid: info.player_type
            for pid, info in player_info.items()
            if info.player_type != PlayerType.HOST
        }

    def build_confusion_matrix(
        self,
        classifications: dict[str, ClassificationSubmission],
        ground_truth: dict[str, PlayerType],
    ) -> dict:
        """Build per-player and aggregate confusion matrices.

        Returns a dict with ``per_player`` and ``aggregate`` keys.  Each
        entry contains tp, tn, fp, fn counts and an accuracy ratio.
        """
        per_player: dict[str, dict] = {}

        agg_tp = 0
        agg_tn = 0
        agg_fp = 0
        agg_fn = 0

        for player_id, submission in classifications.items():
            tp = 0
            tn = 0
            fp = 0
            fn = 0

            for target_id, classification in submission.classifications.items():
                guess = classification.guess.lower()
                actual = ground_truth.get(target_id)
                if actual is None:
                    continue

                if actual == PlayerType.AI:
                    if guess == "ai":
                        tp += 1
                    else:
                        fn += 1
                else:  # HUMAN
                    if guess == "human":
                        tn += 1
                    else:
                        fp += 1

            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0

            per_player[player_id] = {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": accuracy,
            }

            agg_tp += tp
            agg_tn += tn
            agg_fp += fp
            agg_fn += fn

        agg_total = agg_tp + agg_tn + agg_fp + agg_fn
        agg_accuracy = (agg_tp + agg_tn) / agg_total if agg_total > 0 else 0.0

        return {
            "per_player": per_player,
            "aggregate": {
                "tp": agg_tp,
                "tn": agg_tn,
                "fp": agg_fp,
                "fn": agg_fn,
                "accuracy": agg_accuracy,
            },
        }

    def get_leaderboard(self, final_scores: dict[str, int]) -> list[dict]:
        """Return a ranked leaderboard sorted by score descending.

        Players with equal scores receive the same rank (dense ranking
        with gaps, i.e. standard competition ranking).
        """
        sorted_entries = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        leaderboard: list[dict] = []
        prev_score: int | None = None
        prev_rank = 0

        for i, (player_id, score) in enumerate(sorted_entries, start=1):
            if score != prev_score:
                rank = i
            else:
                rank = prev_rank
            leaderboard.append({"player_id": player_id, "score": score, "rank": rank})
            prev_score = score
            prev_rank = rank

        return leaderboard
