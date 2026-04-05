"""
Phase 1 unit tests — run without a live server.
Usage: python test_phase1.py
"""
import asyncio
import json
import random
import sys
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, ".")

from server.game_state import GameSession, Phase
from server.assignment import load_word_bank, run_assignment_phase
from server.logger import GameLogger


# ── helpers ──────────────────────────────────────────────────────────────────

def make_mock_websocket() -> MagicMock:
    ws = MagicMock()
    ws.send_json = AsyncMock()
    ws.connected = True
    return ws


def make_session(num_players: int = 6) -> GameSession:
    session = GameSession("test-session")
    for i in range(1, num_players + 1):
        ws = make_mock_websocket()
        session.add_player(f"Player {i}", ws, is_human=(i <= 2))
    return session


# ── tests ─────────────────────────────────────────────────────────────────────

def test_word_bank_loads():
    bank = load_word_bank()
    assert len(bank) > 0, "Word bank should not be empty"
    for entry in bank:
        assert "topic" in entry
        assert "secret_word" in entry
        assert "grid_items" in entry
        assert len(entry["grid_items"]) == 10, f"Grid must have exactly 10 items, got {len(entry['grid_items'])}"
        labels = [item["label"] for item in entry["grid_items"]]
        assert entry["secret_word"] in labels, f"secret_word '{entry['secret_word']}' must be in grid_items"
    print(f"  word bank OK: {len(bank)} entries")


async def test_assignment_sets_round_state():
    session = make_session()
    logger = GameLogger("test-session", base_dir="/tmp/sandbox_tests")
    bank = load_word_bank()

    await run_assignment_phase(session, logger, bank)

    rs = session.round_state
    assert rs is not None
    assert rs.round_number == 1
    assert rs.imposter_id in session.player_ids()
    assert rs.topic != ""
    assert rs.secret_word != ""
    assert len(rs.grid_items) == 10
    assert session.phase == Phase.ASSIGNMENT
    print(f"  round state OK: imposter={rs.imposter_id}, topic='{rs.topic}', word='{rs.secret_word}'")


async def test_each_player_receives_round_start():
    session = make_session()
    logger = GameLogger("test-session", base_dir="/tmp/sandbox_tests")
    bank = load_word_bank()

    await run_assignment_phase(session, logger, bank)

    imposter_id = session.round_state.imposter_id

    for player_id, player in session.players.items():
        player.websocket.send_json.assert_called_once()
        payload = player.websocket.send_json.call_args[0][0]

        assert payload["event"] == "round_start"
        assert payload["round"] == 1
        assert len(payload["image_grid"]) == 10

        if player_id == imposter_id:
            assert payload["role"] == "imposter"
            assert payload["secret_word"] is None, "Imposter must NOT receive secret word"
        else:
            assert payload["role"] == "insider"
            assert payload["secret_word"] == session.round_state.secret_word

    print(f"  payloads OK: imposter={imposter_id} got secret_word=None, all others got the word")


async def test_grid_is_shuffled():
    """Grid order should vary across rounds (probabilistic — may rarely collide)."""
    session = make_session()
    logger = GameLogger("test-session", base_dir="/tmp/sandbox_tests")
    bank = load_word_bank()

    orders = []
    for _ in range(5):
        # Reset so we can re-run assignment
        session.round_state = None
        session.current_round = 0
        session.used_word_indices.clear()
        for p in session.players.values():
            p.websocket.send_json.reset_mock()

        await run_assignment_phase(session, logger, bank)
        orders.append(tuple(item["label"] for item in session.round_state.grid_items))

    unique_orders = len(set(orders))
    assert unique_orders > 1, "Grid order never changed across 5 runs — shuffle broken?"
    print(f"  shuffle OK: {unique_orders}/5 distinct grid orders observed")


async def test_no_repeated_word_bank_entries():
    """Each entry should only be used once until the pool resets."""
    session = make_session()
    logger = GameLogger("test-session", base_dir="/tmp/sandbox_tests")
    bank = load_word_bank()

    seen_topics = []
    for _ in range(len(bank)):
        session.current_round = 0
        for p in session.players.values():
            p.websocket.send_json.reset_mock()
        await run_assignment_phase(session, logger, bank)
        seen_topics.append(session.round_state.topic)
        # Don't reset used_word_indices — that's the whole point

    assert len(seen_topics) == len(set(seen_topics)), "Same topic was reused before pool exhaustion"
    print(f"  no-repeat OK: all {len(bank)} entries used without repeats")


# ── runner ────────────────────────────────────────────────────────────────────

async def run_all():
    tests = [
        ("word_bank loads & is valid",        lambda: test_word_bank_loads()),
        ("assignment sets round state",        lambda: test_assignment_sets_round_state()),
        ("each player receives round_start",   lambda: test_each_player_receives_round_start()),
        ("grid is shuffled each round",        lambda: test_grid_is_shuffled()),
        ("no repeated word bank entries",      lambda: test_no_repeated_word_bank_entries()),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                await result
            print(f"PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}")
            print(f"      {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    failures = asyncio.run(run_all())
    sys.exit(failures)
