# BaseAgent — Developer Guide

> **File:** `agents/base_agent.py`
> **Purpose:** Abstract base class that every AI agent in The Sandbox game must inherit from.

---

## Overview

`BaseAgent` handles all the plumbing so subclass developers only need to implement **one method**: `call_llm(prompt)`. Everything else — WebSocket connection, game event routing, discussion timing, voting, memory, response parsing — is taken care of here.

```
BaseAgent (ABC)
    └── YourAgent
            └── implements call_llm(prompt) → str
```

---

## Constructor

```python
BaseAgent(
    player_id: str,        # Unique ID for this agent on the server
    display_name: str,     # Human-readable name sent during connection
    model_name: str,       # LLM model identifier (e.g. "gpt-4o")
    server_url: str,       # WebSocket server URL (default: ws://localhost:8000)
    api_timeout: int,      # Seconds before an LLM call times out (default: 15)
)
```

**Key state initialized in `__init__`:**

| Attribute | Type | Purpose |
|---|---|---|
| `self.memory` | `AgentMemory` | Stores round state, hints, messages, reflections |
| `self.current_phase` | `GamePhase \| None` | Tracks which phase the game is in |
| `self.current_round` | `int` | Current round number |
| `self._all_players` | `list[str]` | Raw player_ids of everyone in the game |
| `self._id_to_name` | `dict` | Maps `player_id → "Player 1"` style display name |
| `self._name_to_id` | `dict` | Reverse: `"Player 1" → player_id` |
| `self._incoming_event` | `asyncio.Event` | Fires when the agent is mentioned or DM'd |
| `self._pending_dm_queue` | `list[str]` | Queue of senders who DM'd us, waiting for a reply |
| `self._outbox` | `list[dict]` | Buffered messages to send once connected |

---

## Connection Flow

```
connect()
  └── websockets.connect(url)
        └── _flush_outbox()   ← drain any messages queued before connection
              └── _event_loop()  ← receive messages forever until disconnected
```

- Retries up to 3 times with exponential back-off (1s, 2s).
- The URL includes `type=AI`, `name`, and `model` as query params.

---

## Event Dispatch

Every message from the server is a JSON object with `{ "event": "...", "data": {...} }`.

`_dispatch(msg)` converts the event name to a method call:

```
event = "round_start"  →  calls self._on_round_start(data)
event = "group_message" →  calls self._on_group_message(data)
```

If no matching `_on_<event>` method exists, it is logged and silently ignored.

---

## Game Phases & Event Handlers

The game runs in a fixed sequence of phases each round. Here is what `BaseAgent` does at each phase:

### 1. `game_created`
- Resets memory.
- Stores `session_id` and `num_rounds`.

### 2. `player_joined` / `player_left` / `player_roster`
- Builds/rebuilds the `player_id ↔ display_name` mapping.
- On roster: sets game-start context in memory via `PromptBuilder`.

### 3. `round_start`
- Sets `current_round`, `current_phase = ASSIGNMENT`.
- Calls `memory.start_round(...)` with role, topic, secret word, image grid.

### 4. `hint_phase_start`
- Spawns `_generate_and_submit_hint()` as an async task.
- Builds a hint prompt via `PromptBuilder`, calls the LLM, parses the response, then sends `HINT` action to the server.
- Applies human-like delay via `DelaySimulator`.

### 5. `discussion_start` / `discussion_end`
- Starts `_discussion_behavior(duration)` as a background task.
- On end: cancels the task.

### 6. `voting_start`
- Builds a voting prompt, calls LLM, resolves the target player, sends `VOTE` action.
- Falls back to a random non-self candidate if LLM fails or returns an invalid target.

### 7. `round_results`
- Records vote breakdown into memory.

### 8. `reflection_start`
- Builds a reflection prompt, calls LLM, logs thinking + suspicion updates into memory.
- Calls `memory.compress_round()` to summarize the round for future context.

### 9. `meta_game_start`
- Classifies every other player as `"human"` or `"AI"` with a confidence score.
- Sends a `CLASSIFICATION` action with a JSON payload.

### 10. `game_finished`
- Closes the WebSocket connection.

---

## Discussion Behavior (the core loop)

`_discussion_behavior(duration)` runs for the entire discussion phase. It is a **state-driven loop** that decides when to speak and what to say.

### Timing states

| State | When | Delay |
|---|---|---|
| First turn | First iteration of the loop | Fast opener: 1.5–4s |
| Burst follow-up | Just spoke last turn, no burst yet | Quick follow-up: 1–3s |
| Reactive reply | `_incoming_event` fired (mention or DM) | 1–2.5s |
| Normal turn | Everything else | 4–12s |

> Delays come from `DelaySimulator` to simulate human-like response timing.

### What it can do each turn

The LLM returns a structured action. The loop handles three action types:

- `"group_message"` → sends a public message to all players
- `"dm"` → sends a private message to a specific player
- `"wait"` → does nothing this turn

### Guards

- **Repetitive DM check:** blocks a DM if it is >55% similar (by sequence ratio) to a message already sent to the same target this round.
- **Self-DM check:** ignores any DM targeted at the agent itself.
- **Forbidden word filter:** any `GROUP_MESSAGE` or `DM` containing the round's secret word or image grid words is silently dropped before sending.

---

## Sending Messages (`_send`)

```python
await self._send(action, content, target_player=None)
```

Valid `action` values: `"HINT"`, `"GROUP_MESSAGE"`, `"DM"`, `"VOTE"`, `"CLASSIFICATION"`.

- If the WebSocket is not yet open, the message is buffered in `_outbox` (capped at 50).
- `_flush_outbox()` drains the buffer after connection.

---

## Identity Translation

The server uses raw `player_id` strings internally, but prompts should use anonymous display names like `"Player 1"` so the LLM doesn't leak internal IDs.

| Helper | What it does |
|---|---|
| `_dn(player_id)` | `player_id → "Player 1"` |
| `_pid(name)` | `"Player 1" → player_id` |
| `_resolve_target(target)` | Fuzzy-match an LLM-generated target string (handles `"P1"`, `"player_1"`, `"player 1"`, etc.) back to a real `player_id` |
| `_translate_messages(msgs)` | Replaces `player_id` fields in message dicts with display names |
| `_translate_hints(hints)` | Same for hint dicts |
| `_translate_context(text)` | Replaces all raw IDs in a free-text string with display names |

---

## LLM Interface

### Abstract method (you must implement this)

```python
@abstractmethod
async def call_llm(self, prompt: str) -> str:
    """Call your LLM. Return the raw string response."""
```

This is the **only method you must implement** in your subclass.

### Retry wrapper

`_call_llm_with_retry(prompt, max_retries=2)` wraps `call_llm` with:
- A timeout enforced by `api_timeout` (default 15s).
- Exponential back-off retries (1s, 2s, capped at 5s).

---

## Response Parsing

`_parse_response(raw: str) -> dict` expects the LLM to return JSON, optionally inside a markdown code fence (` ```json ... ``` `).

Expected response shape:

```json
{
  "thinking": "My internal reasoning...",
  "action": {
    "type": "group_message | dm | wait | ...",
    "content": "What I want to say",
    "target_player": "Player 2"
  }
}
```

For voting:
```json
{
  "thinking": "...",
  "action": { "target_player": "Player 3" }
}
```

For classification:
```json
{
  "thinking": "...",
  "action": {
    "classifications": {
      "Player 2": { "guess": "AI", "confidence": 4, "reasoning": "..." },
      "Player 3": { "guess": "human", "confidence": 3, "reasoning": "..." }
    }
  }
}
```

If parsing fails entirely, it falls back to `{"thinking": raw_text, "action": {"type": "wait"}}`.

---

## Thinking & Reflection Persistence

Two methods write internal agent traces to disk for debugging:

- `_flush_thinking(round_num, phase, thinking)` → appends to
  `game_sessions/<session_id>/ai_internals/<player_id>_<model>/thinking_traces.jsonl`

- `_flush_reflection(round_num, reflection)` → appends to
  `game_sessions/<session_id>/ai_internals/<player_id>_<model>/reflections.jsonl`

Both are no-ops if `session_id` is not set or the string is empty.

---

## How to Build a Subclass

```python
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    async def call_llm(self, prompt: str) -> str:
        # Call your LLM API here and return the raw text response.
        response = await my_llm_client.complete(prompt)
        return response.text
```

Then run it:

```python
import asyncio

agent = MyAgent(
    player_id="agent_001",
    display_name="MyBot",
    model_name="my-model-v1",
    server_url="ws://localhost:8000",
)
asyncio.run(agent.connect())
```

---

## Dependencies

| Module | Role |
|---|---|
| `agents.memory.AgentMemory` | Stores round state, messages, hints, reflections, and round summaries |
| `agents.prompt_builder.PromptBuilder` | Builds all LLM prompts for each game phase |
| `agents.delay_simulator.DelaySimulator` | Returns randomized human-like delays for each action type |
| `server.models` | `ActionType`, `GamePhase`, `PlayerRole` enums |
| `websockets` | WebSocket client |
