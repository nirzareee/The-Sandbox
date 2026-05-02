"""Groq API key rotation pool."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_MAX_KEYS = 5  # scan GROQ_API_KEY, GROQ_API_KEY_2 … GROQ_API_KEY_5


class GroqKeyPool:
    """Loads all GROQ_API_KEY[_N] from env; provides round-robin + fallback."""

    def __init__(self) -> None:
        self._keys: list[str] = []
        base = os.getenv("GROQ_API_KEY", "")
        if base:
            self._keys.append(base)
        for i in range(2, _MAX_KEYS + 1):
            k = os.getenv(f"GROQ_API_KEY_{i}", "")
            if k:
                self._keys.append(k)
        if not self._keys:
            raise ValueError("No GROQ_API_KEY found in environment")
        log.info("GroqKeyPool: loaded %d key(s)", len(self._keys))

    @property
    def num_keys(self) -> int:
        return len(self._keys)

    def key_for_round(self, round_number: int) -> str:
        """Deterministic round-robin: round 1 -> key 0, round 2 -> key 1, …"""
        idx = (round_number - 1) % len(self._keys)
        return self._keys[idx]

    def next_key(self, current_key: str) -> str | None:
        """Return the next key after *current_key*, or None if only one key."""
        if len(self._keys) < 2:
            return None
        try:
            idx = self._keys.index(current_key)
        except ValueError:
            return self._keys[0]
        candidate = self._keys[(idx + 1) % len(self._keys)]
        return candidate if candidate != current_key else None


# Module-level singleton, created lazily on first use.
_pool: GroqKeyPool | None = None


def get_pool() -> GroqKeyPool:
    """Return the shared key pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = GroqKeyPool()
    return _pool
