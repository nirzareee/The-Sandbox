"""Per-model pricing for cost tracking.

Prices are USD per 1M tokens (input, output). Verified May 2026.
Update here when provider pricing changes — every agent reads from this map.
"""

from __future__ import annotations

# (input_$/M, output_$/M)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # ---- Anthropic ----
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),

    # ---- OpenAI ----
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.5-pro": (30.00, 180.00),
    "gpt-5.3-codex": (5.00, 30.00),
    "gpt-4o": (2.50, 10.00),

    # ---- Google ----
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3-pro": (2.00, 12.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),

    # ---- Moonshot Kimi ----
    "kimi-k2.5": (0.60, 2.50),
    "kimi-k2.6": (1.00, 3.50),

    # ---- xAI Grok ----
    "grok-4.3": (1.25, 2.50),
    "grok-4-latest": (1.25, 2.50),
    "grok-4": (1.25, 2.50),

    # ---- DeepSeek ----
    "deepseek-v4-pro": (0.435, 0.87),    # promo through 2026-05-31
    "deepseek-v4-flash": (0.10, 0.40),
    "deepseek-chat": (0.435, 0.87),
    "deepseek-reasoner": (0.435, 0.87),

    # ---- Together-hosted ----
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88, 0.88),
    "meta-llama/Llama-3.1-70B-Instruct-Turbo": (0.88, 0.88),
    "deepseek-ai/DeepSeek-R1": (3.00, 7.00),
    "Qwen/Qwen3.5-397B-A17B": (1.20, 1.20),

    # ---- Groq-hosted ----
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.11, 0.34),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "qwen/qwen3-32b": (0.29, 0.39),

    # ---- Mock (no cost) ----
    "mock-llm": (0.0, 0.0),
}


def cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return total USD cost for a single LLM call.

    Unknown models return 0.0 — the call is logged but cost is unknown.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    in_rate, out_rate = pricing
    return (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate


def has_pricing(model: str) -> bool:
    """True if pricing is known for this model."""
    return model in MODEL_PRICING
