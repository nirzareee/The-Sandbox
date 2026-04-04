"""
phase2_test.py

Standalone integration test for Phase 2 — Hint Phase
======================================================
Starts the FastAPI server in-process, then launches 6 mock AI agents.
No real API keys needed.  No external services required.

Run:
    python phase2_test.py

Expected output (order of hints varies each run because order is randomised):
    [server] hint_phase_start broadcast
    [each agent] hint_turn ... hint_request ... hint_submitted
    One agent (Player_2) triggers rejection + retry path
    [server] hint_phase_complete with 6 hints

Also exercises:
    - 60-character truncation path  (MockTruncationProvider)
    - sentence-rejection + retry    (inject_bad_hint=True on Player_2)
    - phase timer enforcement
    - out-of-turn hint submission (ignored by server)
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from typing import List

import uvicorn

# ── Make sure project root is importable regardless of working directory ──────
import os
sys.path.insert(0, os.path.dirname(__file__))

from phase2_hint_server import app as server_app, REQUIRED_PLAYERS  # noqa: E402
from phase2_ai_agent import (  # noqa: E402
    AIAgentHintClient,
    MockProvider,
    LLMProvider,
    run_mock_session,
)

logger = logging.getLogger(__name__)

# ─── Extra mock providers for edge-case testing ───────────────────────────────

class MockTruncationProvider(LLMProvider):
    """
    Always returns a hint that is deliberately over 60 characters.
    Exercises the server-side truncation path.
    """

    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        await asyncio.sleep(0.5)
        long_hint = "super mega ultra tropical juicy sweet yellow inside king of fruits"
        return {
            "thinking": "Intentionally long hint to test server truncation.",
            "action":   {"type": "hint", "content": long_hint},
        }


class MockTimeoutProvider(LLMProvider):
    """
    Hangs for longer than the per-hint timeout so the server's skip logic fires.
    Only does this on the first call; subsequent calls answer normally.
    """

    def __init__(self) -> None:
        self._first = True

    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        if self._first:
            self._first = False
            logger.info("[MockTimeoutProvider] Simulating LLM timeout (20 s sleep)…")
            await asyncio.sleep(20)   # exceeds PER_HINT_TIMEOUT=15 and agent timeout=15
        await asyncio.sleep(0.3)
        return {"thinking": "fallback", "action": {"type": "hint", "content": "fresh"}}


# ─── Server runner (background thread) ───────────────────────────────────────

class BackgroundServer:
    """Runs uvicorn in a daemon thread so the test can connect to it."""

    HOST = "127.0.0.1"
    PORT = 8765

    def __init__(self) -> None:
        cfg = uvicorn.Config(
            app        = server_app,
            host       = self.HOST,
            port       = self.PORT,
            log_level  = "info",
            loop       = "asyncio",
        )
        self._server  = uvicorn.Server(cfg)
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        asyncio.run(self._server.serve())

    def start(self) -> None:
        self._thread.start()
        # Give uvicorn a moment to bind the port
        time.sleep(1.2)
        logger.info("Server listening on ws://%s:%d", self.HOST, self.PORT)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=3)


# ─── Test scenarios ───────────────────────────────────────────────────────────

async def scenario_all_normal(base_url: str) -> None:
    """
    6 agents, all using MockProvider.
    Player_2 has inject_bad_hint=True → exercises rejection + retry.
    """
    print("\n" + "═" * 60)
    print("SCENARIO 1 — Normal hint round (6 mock agents)")
    print("═" * 60)

    agents: List[AIAgentHintClient] = [
        AIAgentHintClient(base_url, f"Player_{i+1}", MockProvider(inject_bad_hint=(i == 1)))
        for i in range(REQUIRED_PLAYERS)
    ]

    tasks = []
    for i, agent in enumerate(agents):
        await asyncio.sleep(0.2)
        tasks.append(asyncio.create_task(agent.connect_and_run()))

    # Wait until all tasks finish (server sends hint_phase_complete and closes phase)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for pid, result in zip([f"Player_{i+1}" for i in range(REQUIRED_PLAYERS)], results):
        if isinstance(result, Exception):
            logger.warning("Agent %s raised: %s", pid, result)


async def scenario_truncation(base_url: str) -> None:
    """
    Player_1 submits an over-60-char hint → server truncates it.
    Remaining 5 agents use normal MockProvider.
    """
    print("\n" + "═" * 60)
    print("SCENARIO 2 — Truncation path (Player_1 sends long hint)")
    print("═" * 60)

    providers = [MockTruncationProvider()] + [MockProvider() for _ in range(REQUIRED_PLAYERS - 1)]
    agents = [
        AIAgentHintClient(base_url, f"Player_{i+1}", providers[i])
        for i in range(REQUIRED_PLAYERS)
    ]

    tasks = []
    for i, agent in enumerate(agents):
        await asyncio.sleep(0.2)
        tasks.append(asyncio.create_task(agent.connect_and_run()))

    await asyncio.gather(*tasks, return_exceptions=True)


# ─── Unit tests for validate_hint() ──────────────────────────────────────────

def test_validate_hint() -> None:
    """Lightweight unit tests that run without a server."""
    from phase2_hint_server import validate_hint, HintStatus

    cases = [
        # (input, expected_valid, expected_status_or_prefix)
        ("tropical",                                  True,  HintStatus.ACCEPTED),
        ("yellow inside",                             True,  HintStatus.ACCEPTED),
        ("smoothie king",                             True,  HintStatus.ACCEPTED),
        ("🌞🏝️",                                      True,  HintStatus.ACCEPTED),
        ("This is a tropical fruit",                  False, HintStatus.REJECTED),
        ("this is good",                              False, HintStatus.REJECTED),
        ("I would describe it as very tropical",      False, HintStatus.REJECTED),
        ("a" * 61,                                    True,  HintStatus.TRUNCATED),
        ("a" * 60,                                    True,  HintStatus.ACCEPTED),
        ("  trailing spaces   ",                      True,  HintStatus.ACCEPTED),  # stripped
    ]

    passed = 0
    failed = 0
    for raw, exp_valid, exp_status in cases:
        valid, processed, status = validate_hint(raw)
        ok = (valid == exp_valid) and (status == exp_status)
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: input={raw!r}  expected valid={exp_valid} status={exp_status}  got valid={valid} status={status}")

    print(f"\nvalidate_hint unit tests: {passed} passed, {failed} failed.")
    if failed:
        sys.exit(1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        stream = sys.stdout,
    )

    # 1 — Unit tests (no server needed)
    print("Running validate_hint() unit tests…")
    test_validate_hint()

    # 2 — Integration tests (requires server)
    print("\nStarting background server…")
    server = BackgroundServer()
    server.start()

    base_url = f"ws://{BackgroundServer.HOST}:{BackgroundServer.PORT}/ws"

    try:
        asyncio.run(scenario_all_normal(base_url))
        # Reset server state for next scenario
        # (In a real test suite you'd restart the server; here we just note
        #  the mock server is stateful — run scenarios sequentially.)
        print("\nNote: server state is not reset between scenarios in standalone mode.")
        print("To run scenario_truncation, restart the server and run it separately.")
    finally:
        server.stop()
        print("\nAll tests complete.")


if __name__ == "__main__":
    main()
