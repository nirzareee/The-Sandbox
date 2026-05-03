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
    # ---- Frontier models ----
    "claude": {
        "module": "agents.claude_agent",
        "class": "ClaudeAgent",
        "env_key": "ANTHROPIC_API_KEY",
        "name": "Claude Opus 4.6",
        "model": "claude-opus-4-6",
    },
    "gpt": {
        "module": "agents.openai_agent",
        "class": "OpenAIAgent",
        "env_key": "OPENAI_API_KEY",
        "name": "GPT-5.5",
        "model": "gpt-5.5",
    },
    "openai": {
        "module": "agents.openai_agent",
        "class": "OpenAIAgent",
        "env_key": "OPENAI_API_KEY",
        "name": "GPT-5.5",
        "model": "gpt-5.5",
    },
    "gpt4o": {
        "module": "agents.openai_agent",
        "class": "OpenAIAgent",
        "env_key": "OPENAI_API_KEY",
        "name": "GPT-5.5",
        "model": "gpt-5.5",
    },
    "gpt-codex": {
        "module": "agents.openai_agent",
        "class": "OpenAIAgent",
        "env_key": "OPENAI_API_KEY",
        "name": "GPT-5.3 Codex",
        "model": "gpt-5.3-codex",
    },
    "gemini": {
        "module": "agents.gemini_agent",
        "class": "GeminiAgent",
        "env_key": "GOOGLE_API_KEY",
        "name": "Gemini 2.0 Flash · free tier OK",
        "model": "gemini-2.0-flash",
    },
    "gemini-2.5": {
        "module": "agents.gemini_agent",
        "class": "GeminiAgent",
        "env_key": "GOOGLE_API_KEY",
        "name": "Gemini 2.5 Flash · billing recommended",
        "model": "gemini-2.5-flash",
    },
    "gemini-pro": {
        "module": "agents.gemini_agent",
        "class": "GeminiAgent",
        "env_key": "GOOGLE_API_KEY",
        "name": "Gemini 3.1 Pro · paid only",
        "model": "gemini-3.1-pro-preview",
    },
    "kimi": {
        "module": "agents.kimi_agent",
        "class": "KimiAgent",
        "env_key": "MOONSHOT_API_KEY",
        "name": "Kimi K2.5",
        "model": "kimi-k2.5",
    },
    "kimi-k2.6": {
        "module": "agents.kimi_agent",
        "class": "KimiAgent",
        "env_key": "MOONSHOT_API_KEY",
        "name": "Kimi K2.6",
        "model": "kimi-k2.6",
    },
    "grok": {
        "module": "agents.grok_agent",
        "class": "GrokAgent",
        "env_key": "XAI_API_KEY",
        "name": "Grok 4.3",
        "model": "grok-4.3",
    },
    "deepseek": {
        "module": "agents.deepseek_agent",
        "class": "DeepSeekAgent",
        "env_key": "DEEPSEEK_API_KEY",
        "name": "DeepSeek V4 Pro",
        "model": "deepseek-v4-pro",
    },
    "deepseek-flash": {
        "module": "agents.deepseek_agent",
        "class": "DeepSeekAgent",
        "env_key": "DEEPSEEK_API_KEY",
        "name": "DeepSeek V4 Flash",
        "model": "deepseek-v4-flash",
    },
    # ---- Together-hosted ----
    "llama": {
        "module": "agents.llama_agent",
        "class": "LlamaAgent",
        "env_key": "TOGETHER_API_KEY",
        "name": "Llama 3.3 70B",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    # ---- Groq-hosted variants (kept for backward compatibility) ----
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
    "qwen3": {
        "module": "agents.groq_agent",
        "class": "GroqAgent",
        "env_key": "GROQ_API_KEY",
        "name": "Qwen 3 32B",
        "model": "qwen/qwen3-32b",
    },
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
        "env_key": "GROQ_API_KEY_2",
        "name": "Llama 3.3 70B",
        "model": "llama-3.3-70b-versatile",
    },
    "groq-llama8b": {
        "module": "agents.groq_agent",
        "class": "GroqAgent",
        "env_key": "GROQ_API_KEY_3",
        "name": "Llama 3.1 8B",
        "model": "llama-3.1-8b-instant",
    },
    "groq-qwen": {
        "module": "agents.groq_agent",
        "class": "GroqAgent",
        "env_key": "GROQ_API_KEY_3",
        "name": "Qwen 3 32B",
        "model": "qwen/qwen3-32b",
    },
}


class AgentManager:
    """Manages server-side agent spawning and lifecycle."""

    def __init__(self) -> None:
        # Keyed by player_id so a single agent can be cancelled.
        self._tasks: dict[str, asyncio.Task] = {}
        self._agents: dict[str, object] = {}
        self._counter: int = 0

    def get_available_agents(self) -> list[dict]:
        """Return agent types with availability status."""
        result = []
        for key, info in AGENT_REGISTRY.items():
            available = info["env_key"] is None or bool(os.getenv(info["env_key"]))
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
        for config in agent_configs:
            agent_type = config.get("type", "mock")
            reg = AGENT_REGISTRY.get(agent_type)
            if not reg:
                log.warning("Unknown agent type: %s — skipping", agent_type)
                continue

            self._counter += 1
            pid = f"agent_{self._counter}"
            try:
                mod = importlib.import_module(reg["module"])
                cls = getattr(mod, reg["class"])
                kwargs = {"player_id": pid, "display_name": pid, "server_url": ws_url}
                if "model" in reg:
                    kwargs["model"] = reg["model"]
                if reg.get("env_key"):
                    kwargs["api_key"] = os.getenv(reg["env_key"], "")
                # Per-agent native reasoning toggle (defaults False)
                if "use_native_reasoning" in config:
                    kwargs["use_native_reasoning"] = bool(config["use_native_reasoning"])
                agent = cls(**kwargs)
            except (ImportError, AttributeError) as exc:
                log.error("Failed to create %s agent: %s", agent_type, exc)
                continue

            task = asyncio.create_task(agent.connect())
            self._tasks[pid] = task
            self._agents[pid] = agent
            spawned.append(pid)
            log.info("Spawned %s agent as %s", agent_type, pid)
            await asyncio.sleep(0.1)

        return spawned

    def list_spawned(self) -> list[str]:
        """Return player_ids of currently-spawned agents (alive or done)."""
        return list(self._tasks.keys())

    async def cancel_agent(self, player_id: str) -> bool:
        """Cancel one agent's task and close its WebSocket. Returns True if found."""
        task = self._tasks.get(player_id)
        agent = self._agents.get(player_id)
        if task is None and agent is None:
            return False
        # Close WebSocket gracefully so server-side cleanup runs.
        if agent is not None:
            ws = getattr(agent, "ws", None)
            if ws is not None:
                try:
                    await ws.close()
                except Exception:  # noqa: BLE001
                    pass
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.pop(player_id, None)
        self._agents.pop(player_id, None)
        log.info("Cancelled agent %s", player_id)
        return True

    async def shutdown(self) -> None:
        """Cancel all running agent tasks."""
        for task in self._tasks.values():
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        self._agents.clear()
        log.info("All agent tasks shut down")
