"""
LLM provider implementations for The Sandbox agents.

All providers share a common interface:
    complete(system, user, max_tokens) -> str       raw text
    complete_json(system, user, max_tokens) -> dict  parsed JSON (strips fences)

OpenAI-compatible APIs (OpenAI, Grok, Groq/Llama) all share OpenAICompatProvider.
Claude and Gemini have their own clients.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
from abc import ABC, abstractmethod


# ── Base ──────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    name: str = "base"
    display_name: str = "Base"
    default_model: str = ""

    @abstractmethod
    async def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        ...

    async def complete_json(self, system: str, user: str, max_tokens: int = 400) -> dict:
        raw = await self.complete(system, user, max_tokens)
        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\n?```$", "", cleaned.strip()).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}


# ── Mock ──────────────────────────────────────────────────────────────────────

class MockProvider(LLMProvider):
    """No API key. Returns plausible canned responses."""
    name = "mock"
    display_name = "Mock Bot"
    default_model = "mock"

    _INSIDER_HINTS = ["tropical", "yellow inside", "classic vibe", "summer staple",
                      "sweet + tart", "king of fruits", "goes with chili", "juicy drip"]
    _IMPOSTER_HINTS = ["pretty good", "colorful", "popular choice", "seasonal", "natural sugar"]
    _DISCUSSION = [
        "ok some of these hints are sus ngl",
        "idk {name}'s hint felt kinda generic",
        "yeah i was thinking the same thing honestly",
        "the hints narrow it down pretty well tbh",
        "im between two people but leaning one way",
        "wait that hint doesn't really fit if you think about it",
        "lol ok yeah that tracks",
    ]

    async def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        await asyncio.sleep(random.uniform(0.3, 1.2))
        is_imposter = "imposter" in user.lower() and "secret_word: none" in user.lower()

        if '"hint"' in user or "HINT PHASE" in user:
            hints = self._IMPOSTER_HINTS if is_imposter else self._INSIDER_HINTS
            return json.dumps({"thinking": "mock", "hint": random.choice(hints)})

        if "DISCUSSION" in user:
            msg = random.choice(self._DISCUSSION)
            return json.dumps({"thinking": "mock", "message": msg})

        if "VOTING" in user:
            # Extract candidates from prompt
            import re
            m = re.search(r"Candidates: (.+)", user)
            candidates = [c.strip() for c in m.group(1).split(",")] if m else []
            vote = random.choice(candidates) if candidates else ""
            return json.dumps({"thinking": "mock", "vote": vote})

        if "META-GAME" in user or "classify" in user.lower():
            import re
            pids = re.findall(r"\b(agent_\d+|player_\d+)\b", user, re.IGNORECASE)
            result = {
                pid: {"guess": random.choice(["human", "ai"]),
                      "confidence": random.randint(1, 4),
                      "reasoning": "just a feeling"}
                for pid in set(pids)
            }
            return json.dumps(result)

        return json.dumps({"reply": random.choice(["yeah for sure", "idk maybe", "interesting"])})


# ── OpenAI-compatible base ────────────────────────────────────────────────────

class OpenAICompatProvider(LLMProvider):
    """Shared base for OpenAI, Grok (xAI), and Groq (Llama)."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("pip install openai")
        self.model = model
        self._client = __import__("openai").AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        from openai import AsyncOpenAI
        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class OpenAIProvider(OpenAICompatProvider):
    name = "openai"
    display_name = "OpenAI GPT"
    default_model = "gpt-4o-mini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(
            model=model or self.default_model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        )


class GrokProvider(OpenAICompatProvider):
    name = "grok"
    display_name = "Grok (xAI)"
    default_model = "grok-3-mini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(
            model=model or self.default_model,
            api_key=api_key or os.environ.get("XAI_API_KEY", ""),
            base_url="https://api.x.ai/v1",
        )


class LlamaProvider(OpenAICompatProvider):
    """Llama via Groq API (fastest inference)."""
    name = "llama"
    display_name = "Llama (Groq)"
    default_model = "llama-3.3-70b-versatile"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(
            model=model or self.default_model,
            api_key=api_key or os.environ.get("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )


# ── Claude ────────────────────────────────────────────────────────────────────

class ClaudeProvider(LLMProvider):
    name = "claude"
    display_name = "Claude (Anthropic)"
    default_model = "claude-haiku-4-5-20251001"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("pip install anthropic")
        self.model = model or self.default_model
        self._client = __import__("anthropic").Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )

    async def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ),
        )
        return response.content[0].text


# ── Gemini ────────────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    name = "gemini"
    display_name = "Gemini (Google)"
    default_model = "gemini-2.0-flash"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("pip install google-generativeai")
        self.model = model or self.default_model
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        genai = __import__("google.generativeai", fromlist=["generativeai"])
        genai.configure(api_key=key)
        self._genai = genai

    async def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        loop = asyncio.get_event_loop()
        model = self._genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
        )

        def _call():
            return model.generate_content(
                user,
                generation_config=self._genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                ),
            )

        response = await loop.run_in_executor(None, _call)
        return response.text


# ── Registry & factory ────────────────────────────────────────────────────────

AVAILABLE_AGENTS = [
    {"type": "mock",   "name": "Mock Bot",           "available": True,  "requires": None},
    {"type": "claude", "name": "Claude (Anthropic)",  "available": bool(os.environ.get("ANTHROPIC_API_KEY")), "requires": "ANTHROPIC_API_KEY"},
    {"type": "openai", "name": "OpenAI GPT",          "available": bool(os.environ.get("OPENAI_API_KEY")),    "requires": "OPENAI_API_KEY"},
    {"type": "gemini", "name": "Gemini (Google)",     "available": bool(os.environ.get("GEMINI_API_KEY")),    "requires": "GEMINI_API_KEY"},
    {"type": "grok",   "name": "Grok (xAI)",          "available": bool(os.environ.get("XAI_API_KEY")),       "requires": "XAI_API_KEY"},
    {"type": "llama",  "name": "Llama (Groq)",         "available": bool(os.environ.get("GROQ_API_KEY")),      "requires": "GROQ_API_KEY"},
]


def get_provider(agent_cfg: dict) -> LLMProvider:
    """Build the right provider from an agent config dict."""
    ptype   = agent_cfg.get("type", "mock")
    model   = agent_cfg.get("model") or None
    api_key = agent_cfg.get("api_key") or None

    mapping = {
        "mock":   lambda: MockProvider(),
        "claude": lambda: ClaudeProvider(model=model, api_key=api_key),
        "openai": lambda: OpenAIProvider(model=model, api_key=api_key),
        "gemini": lambda: GeminiProvider(model=model, api_key=api_key),
        "grok":   lambda: GrokProvider(model=model, api_key=api_key),
        "llama":  lambda: LlamaProvider(model=model, api_key=api_key),
    }

    factory = mapping.get(ptype)
    if not factory:
        raise ValueError(f"Unknown agent type: {ptype!r}. Choose from: {list(mapping)}")
    return factory()
