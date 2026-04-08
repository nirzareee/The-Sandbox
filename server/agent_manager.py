"""Server-side agent spawning and lifecycle management."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os

log = logging.getLogger(__name__)

AGENT_REGISTRY: dict[str, dict] = {
    "mock": {
        "module": "agents.mock_agent",
        "class": "MockAgent",
        "env_key": None,
        "name": "Mock Agent",
    },
    "claude": {
        "module": "agents.claude_agent",
        "class": "ClaudeAgent",
        "env_key": "ANTHROPIC_API_KEY",
        "name": "Claude",
    },
    "gpt4o": {
        "module": "agents.gpt4o_agent",
        "class": "GPT4oAgent",
        "env_key": "OPENAI_API_KEY",
        "name": "GPT-4o",
    },
    "gemini": {
        "module": "agents.gemini_agent",
        "class": "GeminiAgent",
        "env_key": "GOOGLE_API_KEY",
        "name": "Gemini",
    },
    "llama": {
        "module": "agents.llama_agent",
        "class": "LlamaAgent",
        "env_key": "TOGETHER_API_KEY",
        "name": "Llama",
    },
    "kimi": {
        "module": "agents.kimi_agent",
        "class": "KimiAgent",
        "env_key": "KIMI_API_KEY",
        "name": "Kimi K2.5",
    },
    "deepseek": {
        "module": "agents.deepseek_agent",
        "class": "DeepSeekAgent",
        "env_key": "DEEPSEEK_API_KEY",
        "name": "DeepSeek",
    },
    "grok": {
        "module": "agents.grok_agent",
        "class": "GrokAgent",
        "env_key": "GROK_API_KEY",
        "name": "Grok",
    },
    "scout": {
        "module": "agents.groq_agent",
        "class": "GroqAgent",
        "env_key": "GROQ_API_KEY",
        "name": "Llama 4 Scout",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "llama70b": {
        "module": "agents.groq_agent",
        "class": "GroqAgent",
        "env_key": "GROQ_API_KEY",
        "name": "Llama 3.3 70B",
        "model": "llama-3.3-70b-versatile",
    },
    "kimi-k2": {
        "module": "agents.groq_agent",
        "class": "GroqAgent",
        "env_key": "GROQ_API_KEY",
        "name": "Kimi K2",
        "model": "moonshotai/kimi-k2-instruct",
    },
    "qwen3": {
        "module": "agents.groq_agent",
        "class": "GroqAgent",
        "env_key": "GROQ_API_KEY",
        "name": "Qwen 3 32B",
        "model": "qwen/qwen3-32b",
    },
    # Aliases for backward-compatible CLI usage (--agents groq-scout, etc.)
    "groq-scout": {
        "module": "agents.groq_agent",
        "class": "GroqAgent",
        "env_key": "GROQ_API_KEY",
        "name": "Llama 4 Scout",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "groq-llama": {
        "module": "agents.groq_agent",
        "class": "GroqAgent",
        "env_key": "GROQ_API_KEY",
        "name": "Llama 3.3 70B",
        "model": "llama-3.3-70b-versatile",
    },
    "groq-kimi": {
        "module": "agents.groq_agent",
        "class": "GroqAgent",
        "env_key": "GROQ_API_KEY",
        "name": "Kimi K2",
        "model": "moonshotai/kimi-k2-instruct",
    },
    "groq-qwen": {
        "module": "agents.groq_agent",
        "class": "GroqAgent",
        "env_key": "GROQ_API_KEY",
        "name": "Qwen 3 32B",
        "model": "qwen/qwen3-32b",
    },
}


class AgentManager:
    """Manages server-side agent spawning and lifecycle."""

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task] = []

    def get_available_agents(self) -> list[dict]:
        """Return agent types with availability status."""
        result = []
        for key, info in AGENT_REGISTRY.items():
            available = info["env_key"] is None or bool(os.getenv(info["env_key"]))
            # Check if module actually exists (for future agents)
            try:
                importlib.import_module(info["module"])
                implemented = True
            except ImportError:
                implemented = False
            result.append({
                "type": key,
                "name": info["name"],
                "available": available and implemented,
                "implemented": implemented,
                "needs_key": info["env_key"],
            })
        return result

    async def spawn_agents(
        self, agent_configs: list[dict], ws_url: str
    ) -> list[str]:
        """Spawn agent tasks. Returns list of spawned player IDs."""
        spawned: list[str] = []
        offset = len(self._tasks)
        for i, config in enumerate(agent_configs):
            agent_type = config.get("type", "mock")
            reg = AGENT_REGISTRY.get(agent_type)
            if not reg:
                log.warning("Unknown agent type: %s — skipping", agent_type)
                continue

            pid = f"agent_{offset + i + 1}"
            try:
                mod = importlib.import_module(reg["module"])
                cls = getattr(mod, reg["class"])
                kwargs = {"player_id": pid, "display_name": pid, "server_url": ws_url}
                if "model" in reg:
                    kwargs["model"] = reg["model"]
                agent = cls(**kwargs)
            except (ImportError, AttributeError) as exc:
                log.error("Failed to create %s agent: %s", agent_type, exc)
                continue

            task = asyncio.create_task(agent.connect())
            self._tasks.append(task)
            spawned.append(pid)
            log.info("Spawned %s agent as %s", agent_type, pid)
            await asyncio.sleep(0.1)  # stagger connections

        return spawned

    async def shutdown(self) -> None:
        """Cancel all running agent tasks."""
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        log.info("All agent tasks shut down")
