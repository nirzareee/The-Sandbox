"""Human-like delay simulation for AI agents."""

from __future__ import annotations

import random

import server.config as _cfg


class DelaySimulator:
    """Simulates human-like response delays for AI agents.

    Different action types have different typical delays:
    - Hint: 2-8 seconds
    - Group message: 1-4 seconds
    - DM: 1-3 seconds
    - Vote: 3-10 seconds
    - Classification: 2-15 seconds
    - Discussion turn interval: 4-12 seconds

    All delays are scaled by TIMER_MULTIPLIER (allows 100x speedup in tests).
    """

    DELAY_RANGES: dict[str, tuple[float, float]] = {
        "hint": (2, 8),
        "group_message": (1, 4),
        "dm": (1, 3),
        "vote": (3, 10),
        "classification": (2, 15),
        "discussion_turn": (4, 12),
        "first_discussion_turn": (1.5, 4),
        "burst_turn": (1, 3),
        "reactive_turn": (1, 2.5),
    }

    JITTER_PERCENT = 30  # ±30% randomness

    @staticmethod
    def _compute(action: str) -> float:
        lo, hi = DelaySimulator.DELAY_RANGES[action]
        base = random.uniform(lo, hi)
        jitter = random.uniform(
            1 - DelaySimulator.JITTER_PERCENT / 100,
            1 + DelaySimulator.JITTER_PERCENT / 100,
        )
        return max(0.5, base * jitter) * _cfg.TIMER_MULTIPLIER

    @staticmethod
    def hint() -> float:
        return DelaySimulator._compute("hint")

    @staticmethod
    def group_message() -> float:
        return DelaySimulator._compute("group_message")

    @staticmethod
    def dm() -> float:
        return DelaySimulator._compute("dm")

    @staticmethod
    def vote() -> float:
        return DelaySimulator._compute("vote")

    @staticmethod
    def classification() -> float:
        return DelaySimulator._compute("classification")

    @staticmethod
    def discussion_turn() -> float:
        return DelaySimulator._compute("discussion_turn")

    @staticmethod
    def first_discussion_turn() -> float:
        return DelaySimulator._compute("first_discussion_turn")

    @staticmethod
    def burst_turn() -> float:
        return DelaySimulator._compute("burst_turn")

    @staticmethod
    def reactive_turn() -> float:
        return DelaySimulator._compute("reactive_turn")
