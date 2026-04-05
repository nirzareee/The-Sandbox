#!/usr/bin/env python3
"""
The Sandbox — Game Launcher

Usage:
    python launcher.py                        # Normal speed, no agents
    python launcher.py --mock --agents 3      # 3 mock bots
    python launcher.py --mock --fast          # 20x speed for testing
    python launcher.py --mock --fast --agents 4  # Fast test with 4 bots

Then:
    Human players:  http://localhost:8000
    Host dashboard: http://localhost:8000?host=1
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Launch The Sandbox")
    parser.add_argument("--mock", action="store_true", help="Use mock agents (no API keys)")
    parser.add_argument("--fast", action="store_true", help="20x speed (TIMER_MULTIPLIER=0.05)")
    parser.add_argument("--agents", type=int, default=0, help="Number of AI agent slots")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    # Set environment vars that server/main.py reads
    if args.fast:
        os.environ["TIMER_MULTIPLIER"] = "0.05"

    print("=" * 50)
    print("  THE SANDBOX — Game Launcher")
    print("=" * 50)
    print(f"  Speed:    {'20x FAST' if args.fast else 'Normal'}")
    print(f"  Agents:   {args.agents} {'mock' if args.mock else 'real'}")
    print(f"  Server:   http://{args.host}:{args.port}")
    print(f"  Players:  http://localhost:{args.port}")
    print(f"  Host:     http://localhost:{args.port}?host=1")
    print("=" * 50)

    # Patch timer multiplier in the server module before uvicorn starts
    import server.main as srv
    if args.fast:
        srv.TIMER_MULTIPLIER = 0.05

    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
