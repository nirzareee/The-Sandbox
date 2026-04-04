"""
Phase 1: Assignment
- Randomly select imposter from connected players
- Pick an unused word bank entry
- Shuffle grid order (prevent positional exploitation)
- Broadcast round_start to each player with role-appropriate payload
"""

import json
import random
from pathlib import Path

from .game_state import GameSession, RoundState, Phase
from .logger import GameLogger


def load_word_bank(path: str = "data/word_bank.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


async def run_assignment_phase(
    session: GameSession,
    logger: GameLogger,
    word_bank: list[dict],
) -> None:
    session.phase = Phase.ASSIGNMENT
    session.current_round += 1

    # 1. Select imposter randomly from all players
    imposter_id = random.choice(session.player_ids())

    # 2. Pick an unused word bank entry (reset pool if exhausted)
    available = [i for i in range(len(word_bank)) if i not in session.used_word_indices]
    if not available:
        session.used_word_indices.clear()
        available = list(range(len(word_bank)))

    entry_idx = random.choice(available)
    session.used_word_indices.add(entry_idx)
    entry = word_bank[entry_idx]

    # 3. Sample grid items from pool and pick random secret word
    pool: list[dict] = entry["words"]
    grid_size: int = entry.get("grid_size", min(10, len(pool)))
    grid_items: list[dict] = random.sample(pool, min(grid_size, len(pool)))
    secret_word: str = random.choice(grid_items)["label"]
    random.shuffle(grid_items)

    # 4. Store round state
    session.round_state = RoundState(
        round_number=session.current_round,
        imposter_id=imposter_id,
        topic=entry["topic"],
        secret_word=secret_word,
        grid_items=grid_items,
    )

    # 5. Log assignment (server-side only — includes secret word)
    logger.log_round_assignment(
        round_number=session.current_round,
        imposter_id=imposter_id,
        topic=entry["topic"],
        secret_word=secret_word,
        grid_items=grid_items,
    )

    # 6. Broadcast round_start to each player
    image_grid_labels = [item["label"] for item in grid_items]
    image_urls = [item["image"] for item in grid_items]

    for player_id, player in session.connected_players().items():
        is_imposter = player_id == imposter_id
        payload = {
            "event": "round_start",
            "round": session.current_round,
            "total_rounds": session.MAX_ROUNDS,
            "role": "imposter" if is_imposter else "insider",
            "topic": entry["topic"],
            "secret_word": None if is_imposter else secret_word,
            "image_grid": image_grid_labels,
            "image_urls": image_urls,
        }
        await player.websocket.send_json(payload)
