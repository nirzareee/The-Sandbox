#!/usr/bin/env python3
"""CLI session launcher for The Sandbox.

Usage examples:
    # Full game with mock agents (no API keys needed)
    python launcher.py --mock --humans 0

    # Fast mode for quick testing
    python launcher.py --mock --fast

    # Mixed: 1 human + 5 mock agents
    python launcher.py --humans 1 --mock

    # Real AI agents (requires API keys in .env)
    python launcher.py --humans 2 --agents claude,gpt4o,gemini,llama

    # Groq models: 1 human + 2 groq agents
    python launcher.py --humans 1 --agents groq-scout,groq-llama --players 3
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import random
import signal
import socket
import sys
import webbrowser

import uvicorn
from dotenv import load_dotenv

from server.agent_manager import AGENT_REGISTRY

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("launcher")


# ---------------------------------------------------------------------------
# Agent factory  (uses the single AGENT_REGISTRY from server.agent_manager)
# ---------------------------------------------------------------------------


def _get_env_key(kind: str) -> str | None:
    """Return the environment-variable name required for *kind*, or None."""
    entry = AGENT_REGISTRY.get(kind, {})
    return entry.get("env_key")


def _make_mock_agent(player_id: str, display_name: str, server_url: str):
    from agents.mock_agent import MockAgent
    return MockAgent(player_id=player_id, display_name=display_name, server_url=server_url)


def _make_real_agent(kind: str, player_id: str, display_name: str, server_url: str):
    """Instantiate a real agent by its short name using AGENT_REGISTRY."""
    entry = AGENT_REGISTRY[kind]
    mod = importlib.import_module(entry["module"])
    cls = getattr(mod, entry["class"])
    extra = {}
    if "model" in entry:
        extra["model"] = entry["model"]
    return cls(player_id=player_id, display_name=display_name, server_url=server_url, **extra)


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

async def _wait_for_server(http_url: str, timeout: float = 15.0) -> bool:
    """Poll /health until the server is ready."""
    import urllib.request
    import urllib.error

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            resp = await asyncio.to_thread(
                urllib.request.urlopen, f"{http_url}/health"
            )
            data = json.loads(resp.read())
            if data.get("status") == "ok":
                return True
        except (urllib.error.URLError, OSError):
            pass
        await asyncio.sleep(0.3)
    return False


async def _start_game(http_url: str) -> dict:
    """POST /start and return the response JSON."""
    import urllib.request
    import urllib.error
    req = urllib.request.Request(f"{http_url}/start", method="POST")
    try:
        resp = await asyncio.to_thread(urllib.request.urlopen, req)
        return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            log.warning("Game already started (409) — attaching to existing session")
            return {"session_id": "existing", "status": "already_active"}
        raise


async def _wait_for_humans(http_url: str, expected_total: int, timeout: float = 300.0) -> bool:
    """Wait until the expected number of players have connected."""
    import urllib.request

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            resp = await asyncio.to_thread(
                urllib.request.urlopen, f"{http_url}/health"
            )
            data = json.loads(resp.read())
            if data.get("player_count", 0) >= expected_total:
                return True
        except (Exception,):
            pass
        await asyncio.sleep(1.0)
    return False


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def run_session(args: argparse.Namespace) -> None:
    """Launch server, spawn agents, run game, print results."""

    port = args.port
    bind_host = "0.0.0.0"       # listen on all interfaces (LAN + localhost)
    internal_host = "127.0.0.1" # loopback for agents and health polling
    ws_url = f"ws://{internal_host}:{port}"
    http_url = f"http://{internal_host}:{port}"

    # --- Apply fast mode ---
    if args.fast:
        import server.config as cfg
        cfg.TIMER_MULTIPLIER = 0.3
        log.info("Fast mode: TIMER_MULTIPLIER = 0.3")

    # --- Start uvicorn server ---
    config = uvicorn.Config(
        "server.main:app",
        host=bind_host,
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    log.info("Starting server on %s:%d ...", bind_host, port)

    if not await _wait_for_server(http_url):
        log.error("Server failed to start within 15s")
        server.should_exit = True
        await server_task
        return

    log.info("Server ready.")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
        log.info("WiFi players can join at: http://%s:%d", lan_ip, port)
    except Exception:
        pass

    agent_tasks: list[asyncio.Task] = []

    try:
        # --- Determine player slots ---
        total = args.players
        num_humans = args.humans
        num_agents = total - num_humans

        if num_agents < 0:
            log.error("--humans (%d) exceeds --players (%d)", num_humans, total)
            return

        # --- Host mode (default): open browser, let host control from UI ---
        if not args.mock and not args.agents:
            host_url = f"{http_url}?host=1"
            webbrowser.open(host_url)
            log.info("Host mode: opened browser to %s", host_url)

            # Still open browser tabs for human players if requested
            if num_humans > 0:
                for _ in range(num_humans):
                    webbrowser.open(http_url)
                log.info("Host mode: opened %d human player tab(s)", num_humans)

            # Just keep server alive — host controls everything from UI
            log.info("Waiting for game (host controls setup & start)...")
            await server_task
            return

        # --- Automated mode (--mock or --agents): build agent list ---
        # Server assigns anonymous "Player N" names; agents use their pid internally.
        agents = []

        if args.mock:
            # All AI slots use MockAgent
            for i in range(num_agents):
                pid = f"agent_{i + 1}"
                agent = _make_mock_agent(pid, pid, ws_url)
                agents.append(agent)
                log.info("  MockAgent: %s", pid)
        elif args.agents:
            # Real agents from --agents flag
            kinds = [k.strip().lower() for k in args.agents.split(",")]
            # Pad or truncate to fill agent slots
            while len(kinds) < num_agents:
                kinds.append(kinds[len(kinds) % len(kinds)] if kinds else "claude")
            kinds = kinds[:num_agents]

            for i, kind in enumerate(kinds):
                if kind not in AGENT_REGISTRY:
                    log.error("Unknown agent type: %s (valid: %s)",
                              kind, ", ".join(AGENT_REGISTRY))
                    return
                env_var = _get_env_key(kind)
                if env_var and not os.getenv(env_var):
                    log.error("Missing %s for %s agent", env_var, kind)
                    return

                pid = f"agent_{i + 1}"
                agent = _make_real_agent(kind, pid, pid, ws_url)
                agents.append(agent)
                log.info("  %s agent: %s", kind, pid)

        # --- Handle human players FIRST (so lobby is empty when they open it) ---
        if num_humans > 0:
            log.info("Waiting for %d human player(s) to join via browser...", num_humans)
            frontend_url = http_url
            webbrowser.open(frontend_url)
            log.info("Opened browser to %s", frontend_url)

            if not await _wait_for_humans(http_url, num_humans, timeout=300):
                log.warning("Timed out waiting for humans — starting with current players")

        # --- Connect agents AFTER humans (staggered to look natural) ---
        log.info("Connecting %d agent(s)...", len(agents))
        for agent in agents:
            # Stagger joins so they look like real players arriving
            await asyncio.sleep(random.uniform(2.0, 5.0))
            task = asyncio.create_task(agent.connect())
            agent_tasks.append(task)
            await asyncio.sleep(0.1)

        # --- Start the game ---
        log.info("Starting game...")
        result = await _start_game(http_url)
        session_id = result.get("session_id", "unknown")
        log.info("Game session: %s (status: %s)", session_id, result.get("status"))

        # --- Wait for agents to finish ---
        log.info("Game in progress... waiting for completion.")
        try:
            await asyncio.wait_for(
                asyncio.gather(*agent_tasks, return_exceptions=True),
                timeout=600,
            )
        except asyncio.TimeoutError:
            log.warning("Game timed out after 600s")

        # --- Print summary ---
        print("\n" + "=" * 60)
        print("  GAME COMPLETE")
        print("=" * 60)
        print(f"  Session: {session_id}")

        session_dir = os.path.join("game_sessions", session_id)
        if os.path.isdir(session_dir):
            print(f"  Logs: {session_dir}/")
            # Try to read final scores from game log
            log_path = os.path.join(session_dir, "game_log.jsonl")
            if os.path.isfile(log_path):
                with open(log_path) as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("event") == "game_finished":
                                # Scores may be at top level or nested under "data"
                                scores = (
                                    entry.get("final_scores")
                                    or entry.get("data", {}).get("final_scores")
                                    or {}
                                )
                                if scores:
                                    print("\n  Final Scores:")
                                    for pid, score in sorted(scores.items(), key=lambda x: -x[1]):
                                        print(f"    {pid}: {score}")
                        except json.JSONDecodeError:
                            pass
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        log.info("Interrupted — shutting down...")
    finally:
        # Cancel agent tasks
        for task in agent_tasks:
            if not task.done():
                task.cancel()
        if agent_tasks:
            await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Stop server
        server.should_exit = True
        await server_task
        log.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a full game session of The Sandbox",
    )
    parser.add_argument(
        "--humans", type=int, default=0,
        help="Number of human player slots (default: 0)",
    )
    _valid_agents = ", ".join(sorted(AGENT_REGISTRY.keys()))
    parser.add_argument(
        "--agents", type=str, default=None,
        help=f"Comma-separated agent types: {_valid_agents}",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock agents (no API keys needed)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: TIMER_MULTIPLIER=0.3 for quick testing",
    )
    parser.add_argument(
        "--players", type=int, default=6,
        help="Total player count (default: 6)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Server port (default: 8000)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    log.info("The Sandbox — Session Launcher")
    log.info("  humans=%d, mock=%s, fast=%s, players=%d",
             args.humans, args.mock, args.fast, args.players)

    try:
        asyncio.run(run_session(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
