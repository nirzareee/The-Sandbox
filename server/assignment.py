"""
Phase 1: Assignment
- Randomly select imposter (without-replacement across rounds)
- Pick an unused word bank entry
- Shuffle grid order
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
    broadcast_fn=None,
    send_personal_fn=None,
) -> None:
    session.phase = Phase.ASSIGNMENT
    session.current_round += 1

    # 1. Select imposter — avoid repeats until everyone has been imposter
    game_pids = session.player_ids()
    available_imposters = [pid for pid in game_pids if pid not in session.past_imposters]
    if not available_imposters:
        session.past_imposters.clear()
        available_imposters = list(game_pids)
    imposter_id = random.choice(available_imposters)
    session.past_imposters.append(imposter_id)

    # 2. Pick an unused word bank entry
    available = [i for i in range(len(word_bank)) if i not in session.used_word_indices]
    if not available:
        session.used_word_indices.clear()
        available = list(range(len(word_bank)))
    entry_idx = random.choice(available)
    session.used_word_indices.add(entry_idx)
    entry = word_bank[entry_idx]

    # 3. Sample grid items and pick random secret word
    pool: list[dict] = entry["words"]
    grid_size: int = entry.get("grid_size", min(10, len(pool)))
    grid_items: list[dict] = random.sample(pool, min(grid_size, len(pool)))
    secret_word: str = random.choice(grid_items)["label"]
    random.shuffle(grid_items)

    # 4. Build hint order (randomized)
    hint_order = list(game_pids)
    random.shuffle(hint_order)

    # 5. Store round state
    session.round_state = RoundState(
        round_number=session.current_round,
        imposter_id=imposter_id,
        topic=entry["topic"],
        secret_word=secret_word,
        grid_items=grid_items,
    )

    logger.log_round_assignment(
        round_number=session.current_round,
        imposter_id=imposter_id,
        topic=entry["topic"],
        secret_word=secret_word,
        grid_items=grid_items,
    )

    # 6. Send round_start to each player (role-differentiated)
    image_grid_labels = [item["label"] for item in grid_items]
    image_urls = [item["image"] for item in grid_items]

    for player_id, player in session.players.items():
        if not player.connected:
            continue

        is_imposter = player_id == imposter_id
        is_host = player_id == "HOST"

        payload = {
            "event": "round_start",
            "data": {
                "round_number": session.current_round,
                "total_rounds": session.MAX_ROUNDS,
                "role": "HOST" if is_host else ("imposter" if is_imposter else "insider"),
                "topic": entry["topic"],
                "secret_word": entry["topic"] if is_imposter else secret_word,  # imposter gets topic, not word
                "image_grid": image_grid_labels,
                "image_urls": image_urls,
                "hint_order": hint_order,
                # Host gets everything
                "imposter_id": imposter_id if is_host else None,
            },
        }

        # Imposter: secret_word = None
        if is_imposter:
            payload["data"]["secret_word"] = None

        try:
            await player.websocket.send_json(payload)
        except Exception:
            player.connected = False
