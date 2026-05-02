#!/usr/bin/env python3
"""
The Sandbox — Game Launcher

Usage:
    python launcher.py                              # Normal speed, no agents
    python launcher.py --agents 3                   # 3 mock bots
    python launcher.py --provider claude --agents 3 # 3 Claude bots
    python launcher.py --provider openai --agents 2 # 2 OpenAI bots
    python launcher.py --provider gemini --agents 2 # 2 Gemini bots
    python launcher.py --provider grok   --agents 2 # 2 Grok bots
    python launcher.py --provider llama  --agents 2 # 2 Llama (via Groq) bots
    python launcher.py --provider mock --agents 4 --fast   # Fast test

    Mix providers by using the host dashboard at http://localhost:<port>?host=1

Env vars needed per provider:
    ANTHROPIC_API_KEY  → claude
    OPENAI_API_KEY     → openai
    GEMINI_API_KEY     → gemini
    XAI_API_KEY        → grok
    GROQ_API_KEY       → llama

Human players:  http://localhost:<port>
Host dashboard: http://localhost:<port>?host=1
"""

import argparse
import os
from pathlib import Path


PROVIDER_CHOICES = ["mock", "claude", "openai", "gemini", "grok", "llama"]


def _load_dotenv():
    """Load .env from the project root (no external deps needed)."""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Launch The Sandbox")
    parser.add_argument(
        "--provider", default="mock", choices=PROVIDER_CHOICES,
        help="LLM provider for auto-spawned agents (default: mock)",
    )
    parser.add_argument(
        "--agents", type=int, default=0,
        help="Number of AI agents to auto-spawn when game starts",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="20x speed — TIMER_MULTIPLIER=0.05 (good for testing)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    # Legacy alias
    parser.add_argument("--mock", action="store_true", help="Shorthand for --provider mock")
    args = parser.parse_args()

    if args.mock:
        args.provider = "mock"

    timer_mul = 0.05 if args.fast else 1.0

    print("=" * 55)
    print("  THE SANDBOX — Game Launcher")
    print("=" * 55)
    print(f"  Speed:    {'20x FAST' if args.fast else 'Normal'}")
    print(f"  Agents:   {args.agents} x {args.provider}")
    print(f"  Server:   http://{args.host}:{args.port}")
    print(f"  Players:  http://localhost:{args.port}")
    print(f"  Host:     http://localhost:{args.port}?host=1")
    print("=" * 55)

    # Pass config to server via environment variables (reliable across uvicorn import)
    os.environ["SANDBOX_TIMER_MUL"] = str(timer_mul)
    os.environ["SANDBOX_PORT"] = str(args.port)

    if args.agents > 0:
        provider_names = {
            "mock":   "Mock", "claude": "Claude", "openai": "GPT",
            "gemini": "Gemini", "grok": "Grok", "llama": "Llama",
        }
        base_name = provider_names.get(args.provider, args.provider.title())
        agent_entries = ",".join(
            f"{args.provider}:{base_name} {i + 1}" for i in range(args.agents)
        )
        os.environ["SANDBOX_AGENTS"] = agent_entries
        print(f"  Queued {args.agents} {args.provider} agent(s) — will spawn on /start")
        print("=" * 55)

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
