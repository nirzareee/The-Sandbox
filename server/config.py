"""Game configuration constants for The Sandbox."""

# Phase durations in seconds
PHASE_DURATIONS: dict[str, int] = {
    "LOBBY": 0,
    "ASSIGNMENT": 10,
    "HINT": 30,
    "DISCUSSION": 90,
    "VOTING": 25,
    "RESULTS": 10,
    "REFLECTION": 20,
    "META_GAME": 60,
    "SURVEY": 180,
    "REVEAL": 15,
}

# Scoring values
SCORING: dict[str, int] = {
    "correct_vote": 10,
    "imposter_survives": 15,
    "wrongly_accused": -5,
    "correct_classification": 5,
}

# Game settings
MAX_PLAYERS: int = 6
NUM_ROUNDS: int = 5
HINT_MAX_CHARS: int = 60
HINT_MAX_WORDS: int = 8
GRID_SIZE: int = 10

# Network settings
RECONNECT_TIMEOUT: int = 60
AI_API_TIMEOUT: int = 15
DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 8000

# Timer multiplier (set < 1.0 for fast testing, e.g. 0.1)
TIMER_MULTIPLIER: float = 1.0
