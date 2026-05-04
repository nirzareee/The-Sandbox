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
        "hint": (3, 10),
        "group_message": (2, 6),
        "dm": (2, 5),
        "vote": (4, 12),
        "classification": (3, 18),
        # Discussion timings — pure "deciding-to-speak" gap. typing_time(content)
        # handles per-message typing separately. 2026-05-03: scaled ~1.6× from
        # the original (3, 8) range — observed agents averaging 13s gaps vs
        # humans averaging 27s in real games, so closing the realism gap.
        "discussion_turn": (5, 13),
        "first_discussion_turn": (3, 8),
        "burst_turn": (2.5, 6),
        "reactive_turn": (2.5, 5),
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
    def hint_for_role(is_imposter: bool) -> float:
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

    @staticmethod
    def typing_time(text: str, base_pause: float = 1.6, chars_per_sec: float = 5.0,
                    cap: float = 13.0) -> float:
        """Per-message typing delay scaled to message length.

        Models a natural typist: short pre-typing pause + chars / typing
        speed. Capped to keep messages from going stale by the time they
        land. 2026-05-03: slowed ~1.6× (chars_per_sec 8→5, pause 1→1.6,
        cap 8→13) to better match observed human typing pace.

        Defaults: ~5 chars/sec (~60 wpm), 1.6s pause, 13s hard cap.
        12-char  → ~4.0s   ("hmm not sure")
        30-char  → ~7.6s   ("p3 acting weird tbh i guess")
        60-char  → ~13s cap (would be 13.6s uncapped)
        """
        length = len(text or "")
        base = base_pause + (length / chars_per_sec)
        capped = min(base, cap)
        jitter = random.uniform(
            1 - DelaySimulator.JITTER_PERCENT / 100,
            1 + DelaySimulator.JITTER_PERCENT / 100,
        )
        return max(0.5, capped * jitter) * _cfg.TIMER_MULTIPLIER
