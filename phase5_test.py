"""
phase5_test.py

Standalone integration test for Phase 5 — Between-Round Reflection
====================================================================
Starts the FastAPI server in-process, then launches 6 mock AI agents.
No real API keys needed.  No external services required.

Run:
    python phase5_test.py

Expected output:
    [server] reflection_start broadcast (with topic, grid, hints, prompt)
    [each agent] waits a random delay, then submits a reflection
    [server] reflection_submitted broadcast for each agent
    One agent also sends a chat message and a DM to exercise those paths
    [server] reflection_complete after 90 seconds with all reflections

Also exercises:
    - Chat message broadcast
    - DM delivery to recipient only
    - Staggered reflection timing (agents don't all reply at once)
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from typing import List

import uvicorn

import os
sys.path.insert(0, os.path.dirname(__file__))

from phase5_reflection_server import app as server_app, REQUIRED_PLAYERS  # noqa: E402
from phase5_reflection_agent import (  # noqa: E402
    AIReflectionAgent,
    MockProvider,
    LLMProvider,
)

logger = logging.getLogger(__name__)


# ─── Extra mock provider for fast testing (overrides the 90s wait) ───────────

class FastMockProvider(MockProvider):
    """
    Same as MockProvider but with minimal sleep so the test completes quickly.
    Use only in tests — real agents should keep realistic delays.
    """

    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        await asyncio.sleep(0.3)
        import random
        grid_analysis = random.choice([
            "Player_1's 'tropical' is highly specific to Mango. Player_2's 'sweet' is too generic — imposter pattern.",
            "Most hints nail Mango. 'colorful' and 'sweet' are suspiciously broad and could fit any grid item.",
        ])
        meta_analysis = random.choice([
            "Player_4's cultural reference 'aamras vibes' reads human. Generic adjectives like 'sweet' read AI.",
            "Emoji-only hints lean human. Short generic words (sweet, colorful) lean AI.",
        ])
        return {
            "thinking": "Fast mock reasoning for testing.",
            "reflection": {
                "grid_analysis": grid_analysis,
                "meta_analysis": meta_analysis,
            },
        }


# ─── Background server ────────────────────────────────────────────────────────

class BackgroundServer:
    HOST = "127.0.0.1"
    PORT = 8766

    def __init__(self) -> None:
        cfg = uvicorn.Config(
            app       = server_app,
            host      = self.HOST,
            port      = self.PORT,
            log_level = "info",
            loop      = "asyncio",
        )
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        asyncio.run(self._server.serve())

    def start(self) -> None:
        self._thread.start()
        time.sleep(1.2)
        logger.info("Server listening on ws://%s:%d", self.HOST, self.PORT)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=3)


# ─── Patched agent for fast testing ──────────────────────────────────────────

class FastTestAgent(AIReflectionAgent):
    """
    Overrides the random stagger delay so the test completes in seconds.
    Player_3 also sends a chat message and a DM after submitting reflection.
    """

    async def _on_reflection_start(self, msg: dict) -> None:
        self.state.round_number = msg.get("round", 0)
        self.state.topic        = msg.get("topic", "")
        self.state.secret_word  = msg.get("secret_word", "")
        self.state.image_grid   = msg.get("image_grid", [])
        self.state.hints        = msg.get("hints", [])

        # Skip the 5–30s stagger; use 0.1–0.5s instead
        import random
        delay = random.uniform(0.1, 0.5)
        await asyncio.sleep(delay)

        reflection_text = await self._generate_reflection()
        await self._submit_reflection(reflection_text)

        # Player_3 sends a chat + DM to exercise those code paths
        if self.player_id == "Player_3":
            await asyncio.sleep(0.2)
            await self.send_chat("Interesting round — those broad hints stood out to me.")
            await asyncio.sleep(0.1)
            await self.send_dm("Player_1", "Nice hint btw, very targeted.")


# ─── Test scenarios ───────────────────────────────────────────────────────────

async def scenario_full_reflection(base_url: str) -> None:
    print("\n" + "═" * 60)
    print("SCENARIO — Full reflection round (6 fast mock agents)")
    print("═" * 60)

    agents: List[FastTestAgent] = [
        FastTestAgent(
            server_url = base_url,
            player_id  = f"Player_{i + 1}",
            provider   = FastMockProvider(),
        )
        for i in range(REQUIRED_PLAYERS)
    ]

    tasks: List[asyncio.Task] = []
    for agent in agents:
        await asyncio.sleep(0.2)
        tasks.append(asyncio.create_task(agent.connect_and_run()))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("Player_%d raised: %s", i + 1, result)


# ─── Patch server reflection duration for fast testing ───────────────────────

def _patch_server_duration(seconds: int = 5) -> None:
    """Lower the 90-second window to `seconds` so the test finishes quickly."""
    import phase5_reflection_server as srv
    srv.REFLECTION_DURATION = seconds
    # Also patch the manager default so in-flight managers use it
    import phase5_reflection_agent as agent_mod
    agent_mod.REFLECTION_DURATION = seconds
    logger.info("Patched REFLECTION_DURATION to %ds for testing.", seconds)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        stream = sys.stdout,
    )

    _patch_server_duration(5)   # 5-second window instead of 90 for fast CI

    print("Starting background server…")
    server = BackgroundServer()
    server.start()

    base_url = f"ws://{BackgroundServer.HOST}:{BackgroundServer.PORT}/ws"

    try:
        asyncio.run(scenario_full_reflection(base_url))
    finally:
        server.stop()
        print("\nAll tests complete.")


if __name__ == "__main__":
    main()
