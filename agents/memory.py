"""Agent game memory and context management."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from server.models import PlayerRole

log = logging.getLogger(__name__)


class AgentMemory:
    """Maintains game state and memory for a single agent across all rounds.

    Tracks:
    - Current round details (full)
    - Past round summaries (~200 tokens each)
    - Meta-game observations (human/AI confidence per player)
    """

    def __init__(self, player_id: str) -> None:
        self.player_id = player_id

        # Current round (full detail)
        self.current_round_state: dict = {}

        # Past rounds (compressed summaries)
        self.round_summaries: dict[int, str] = {}

        # Per-round data
        self.round_hints: dict[int, dict[str, str]] = {}
        self.round_messages: dict[int, list[dict]] = {}
        self.round_dms: dict[int, list[dict]] = {}
        self.round_votes: dict[int, dict[str, str]] = {}
        self.round_results: dict[int, dict] = {}
        self.round_reflections: dict[int, str] = {}

        # Timing tracking (for response-time awareness in prompts)
        self.round_start_times: dict[int, datetime] = {}
        self.round_hint_timestamps: dict[int, dict[str, datetime]] = {}

        # Meta-game tracker
        self.meta_game_tracker: dict[str, dict] = {}

        # Game start context (prepended to every prompt)
        self._game_start_context: str = ""

        # Relationship tracker: {player_id: {"trust": float, "events": [str]}}
        self.relationship_tracker: dict[str, dict] = {}

        # Player ID → display name mapping
        self._name_mapping: dict[str, str] = {}  # player_id → display_name

        # Per-round thinking accumulator (fed back into prompts).
        # Cleared at the start of each round; the previous round's traces
        # are summarized via Groq into self.round_thinking_summaries before
        # the clear so the model retains long-arc reasoning continuity.
        self._round_thinking: list[str] = []
        # Groq-summarized thinking from each completed round, keyed by round number.
        # Fed back into prompts in subsequent rounds so the model recalls its
        # own evolving reasoning across the game.
        self.round_thinking_summaries: dict[int, str] = {}

        # Research logging
        self._thinking_traces: list[dict] = []
        self._reflections: list[dict] = []
        self._api_calls: list[dict] = []
        self._token_counts: list[dict] = []

    def reset(self) -> None:
        """Clear all per-game state for reuse across sessions."""
        self.current_round_state.clear()
        self.round_summaries.clear()
        self.round_hints.clear()
        self.round_messages.clear()
        self.round_dms.clear()
        self.round_votes.clear()
        self.round_results.clear()
        self.round_reflections.clear()
        self.round_start_times.clear()
        self.round_hint_timestamps.clear()
        self.meta_game_tracker.clear()
        self._game_start_context = ""
        self._round_thinking.clear()
        self.round_thinking_summaries.clear()
        self._thinking_traces.clear()
        self._reflections.clear()
        self._api_calls.clear()
        self._token_counts.clear()
        self._name_mapping.clear()

    def set_name_mapping(self, mapping: dict[str, str]) -> None:
        self._name_mapping = dict(mapping)

    def _display(self, pid: str) -> str:
        return self._name_mapping.get(pid, pid)

    # ------------------------------------------------------------------
    # Game start context
    # ------------------------------------------------------------------

    def set_game_start_context(self, ctx: str) -> None:
        """Set context shown at the top of every prompt."""
        self._game_start_context = ctx

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def start_round(
        self,
        round_number: int,
        role: PlayerRole,
        topic: str,
        secret_word: str | None,
        image_grid: list[str],
        hint_order: list[str],
    ) -> None:
        self.current_round_state = {
            "round_number": round_number,
            "role": role,
            "topic": topic,
            "secret_word": secret_word,
            "image_grid": image_grid,
            "hint_order": hint_order,
        }
        self._round_thinking.clear()
        self.round_hints[round_number] = {}
        self.round_messages[round_number] = []
        self.round_dms[round_number] = []
        self.round_votes[round_number] = {}
        self.round_start_times[round_number] = datetime.now(timezone.utc)
        self.round_hint_timestamps[round_number] = {}

    def record_hint(self, player_id: str, hint: str) -> None:
        rn = self.current_round_state.get("round_number", 0)
        self.round_hints.setdefault(rn, {})[player_id] = hint
        self.round_hint_timestamps.setdefault(rn, {})[player_id] = datetime.now(timezone.utc)

    def record_group_message(self, player_id: str, content: str) -> None:
        rn = self.current_round_state.get("round_number", 0)
        self.round_messages.setdefault(rn, []).append({
            "player_id": player_id,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def record_dm(self, sender: str, recipient: str, content: str) -> None:
        rn = self.current_round_state.get("round_number", 0)
        self.round_dms.setdefault(rn, []).append({
            "sender": sender,
            "recipient": recipient,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def record_round_result(self, result_data: dict) -> None:
        rn = self.current_round_state.get("round_number", 0)
        self.round_results[rn] = result_data

    def record_vote(self, voter_id: str, target_id: str) -> None:
        rn = self.current_round_state.get("round_number", 0)
        self.round_votes.setdefault(rn, {})[voter_id] = target_id

    def record_relationship_event(self, player_id: str, event: str, trust_delta: float = 0.0) -> None:
        """Record a social event (alliance, betrayal, cooperation) with another player."""
        if player_id not in self.relationship_tracker:
            self.relationship_tracker[player_id] = {"trust": 0.5, "events": []}
        entry = self.relationship_tracker[player_id]
        entry["trust"] = max(0.0, min(1.0, entry["trust"] + trust_delta))
        entry["events"].append(event)
        # Keep only last 5 events per player
        if len(entry["events"]) > 5:
            entry["events"] = entry["events"][-5:]

    # ------------------------------------------------------------------
    # Memory compression
    # ------------------------------------------------------------------

    async def compress_round(self, round_number: int) -> None:
        """Compress a round's data into a summary.

        Tries LLM-based summarization first (if GROQ_API_KEY_SUMMARY is set).
        Falls back to structured extraction otherwise.
        """
        # Always extract relationship signals regardless of summary method
        self._extract_relationship_signals(round_number)

        # Try LLM-based summary first
        llm_summary = await self._try_llm_summary(round_number)
        if llm_summary:
            self.round_summaries[round_number] = llm_summary
            return

        # Fallback: structured extraction
        self.round_summaries[round_number] = self._structured_summary(round_number)

    async def _try_llm_summary(self, round_number: int) -> str | None:
        """Attempt LLM-based round summarization. Returns None on failure."""
        from agents.summary_client import get_summary_client

        client = get_summary_client()
        if client is None:
            return None

        hints = self.round_hints.get(round_number, {})
        result = self.round_results.get(round_number, {})
        role = self.current_round_state.get("role", "?")
        topic = self.current_round_state.get("topic", "?")
        secret = self.current_round_state.get("secret_word", "UNKNOWN")
        messages = self.round_messages.get(round_number, [])
        dms = self.round_dms.get(round_number, [])
        votes = self.round_votes.get(round_number, {})

        # Translate player IDs to display names for readability
        round_data = {
            "round_number": round_number,
            "topic": topic,
            "secret_word": secret,
            "role": str(role),
            "hints": {self._display(pid): h for pid, h in hints.items()},
            "messages": [
                {"player_id": self._display(m.get("player_id", "?")), "content": m.get("content", "")}
                for m in messages
            ],
            "dms": [
                {
                    "sender": self._display(dm.get("sender", "?")),
                    "recipient": self._display(dm.get("recipient", "?")),
                    "content": dm.get("content", ""),
                }
                for dm in dms
            ],
            "votes": {self._display(v): self._display(t) for v, t in votes.items()},
            "result": result,
        }

        return await client.summarize_round(round_data)

    def _structured_summary(self, round_number: int) -> str:
        """Structured extraction fallback (original compress_round logic)."""
        hints = self.round_hints.get(round_number, {})
        result = self.round_results.get(round_number, {})
        role = self.current_round_state.get("role", "?")
        topic = self.current_round_state.get("topic", "?")
        secret = self.current_round_state.get("secret_word", "UNKNOWN")

        lines = [f"Round {round_number}: {topic} (secret={secret}, my_role={role})"]

        # Hints
        for pid, hint in hints.items():
            lines.append(f"  {self._display(pid)}: '{hint}'")

        # Vote breakdown
        votes = self.round_votes.get(round_number, {})
        if votes:
            tally: dict[str, list[str]] = {}
            for voter, target in votes.items():
                tally.setdefault(self._display(target), []).append(self._display(voter))
            vote_parts = [f"{t}\u2190{','.join(vs)}" for t, vs in tally.items()]
            lines.append(f"  Votes: {'; '.join(vote_parts)}")

        # Result
        imp = result.get("imposter_id", "?")
        caught = result.get("imposter_caught", False)
        lines.append(f"  Imposter: {self._display(imp)} ({'caught' if caught else 'survived'})")

        # Discussion highlights
        highlights = self._extract_discussion_highlights(round_number)
        if highlights:
            lines.append("  Key moments:")
            for h in highlights[:5]:
                lines.append(f"    {self._display(h['player_id'])}: \"{h['content'][:60]}\"")

        # DM summary
        dm_summary = self._compress_dm_threads(round_number)
        if dm_summary:
            lines.append(f"  DMs: {dm_summary}")

        return "\n".join(lines)

    def _extract_discussion_highlights(self, round_number: int) -> list[dict]:
        """Extract notable messages from discussion (accusations, questions, sus calls)."""
        messages = self.round_messages.get(round_number, [])
        keywords = [
            "sus", "vote", "imposter", "weird", "vague", "fake", "lying",
            "cap", "nah", "def", "100%", "idk", "?",
        ]
        highlights = []
        for msg in messages:
            content = msg.get("content", "").lower()
            if any(kw in content for kw in keywords):
                highlights.append(msg)
        return highlights

    def _compress_dm_threads(self, round_number: int) -> str:
        """One-line summary of DM activity this round."""
        dms = self.round_dms.get(round_number, [])
        if not dms:
            return ""
        partners: set[str] = set()
        for dm in dms:
            if dm["sender"] == self.player_id:
                partners.add(self._display(dm["recipient"]))
            else:
                partners.add(self._display(dm["sender"]))
        return f"{len(dms)} DMs with {', '.join(sorted(partners))}"

    def _extract_relationship_signals(self, round_number: int) -> None:
        """Detect alliances, betrayals, and trust events from round data."""
        votes = self.round_votes.get(round_number, {})
        dms = self.round_dms.get(round_number, [])

        # Detect vote coordination from DMs
        dm_partners: set[str] = set()
        for dm in dms:
            partner = dm["recipient"] if dm["sender"] == self.player_id else dm["sender"]
            dm_partners.add(partner)

        # Check if DM partners voted the same target as us
        my_vote = votes.get(self.player_id)
        if my_vote:
            for partner in dm_partners:
                partner_vote = votes.get(partner)
                if partner_vote == my_vote:
                    self.record_relationship_event(partner, f"R{round_number}: voted together", +0.1)
                elif partner_vote and partner_vote != my_vote:
                    self.record_relationship_event(partner, f"R{round_number}: voted differently after DM", -0.15)

        # Track who voted for us (betrayal signal)
        for voter, target in votes.items():
            if target == self.player_id and voter != self.player_id:
                self.record_relationship_event(voter, f"R{round_number}: voted against me", -0.2)

    # ------------------------------------------------------------------
    # Context retrieval for prompts
    # ------------------------------------------------------------------

    def get_context_for_prompt(self, max_chars: int = 8000) -> str:
        """Build text summary of past rounds for LLM prompts."""
        parts: list[str] = []

        if self._game_start_context:
            parts.append(self._game_start_context)
            parts.append("")

        if self.round_summaries:
            parts.append("PAST ROUNDS:")
            for rn in sorted(self.round_summaries):
                parts.append(self.round_summaries[rn])
                # Append truncated reflection if available
                refl = self.round_reflections.get(rn, "")
                if refl:
                    parts.append(f"  My reflection: {refl[:200]}")

        # Cross-round thinking continuity — Groq-summarized reasoning from prior rounds.
        if self.round_thinking_summaries:
            parts.append("\nMY REASONING FROM EARLIER ROUNDS:")
            for rn in sorted(self.round_thinking_summaries):
                parts.append(f"  Round {rn}: {self.round_thinking_summaries[rn]}")

        # Show current standings from most recent round results
        if self.round_results:
            latest_rn = max(self.round_results.keys())
            cumul = self.round_results[latest_rn].get("cumulative_scores", {})
            if cumul:
                parts.append("\nCURRENT STANDINGS:")
                for pid, score in sorted(cumul.items(), key=lambda x: -x[1]):
                    parts.append(f"  {self._display(pid)}: {score} pts")

        if self.meta_game_tracker:
            parts.append("\nMETA-GAME OBSERVATIONS:")
            for pid, obs in self.meta_game_tracker.items():
                try:
                    conf = float(obs.get("human_confidence", 0.5))
                except (TypeError, ValueError):
                    conf = 0.5
                notes = ", ".join(obs.get("notes", [])[-5:])
                parts.append(f"  {self._display(pid)}: human_conf={conf:.1f} ({notes})")

        if self.relationship_tracker:
            parts.append("\nRELATIONSHIP MEMORY:")
            for pid, rel in self.relationship_tracker.items():
                trust = rel.get("trust", 0.5)
                events = rel.get("events", [])
                label = "ally" if trust >= 0.65 else "rival" if trust <= 0.35 else "neutral"
                recent = events[-2:] if events else []
                parts.append(f"  {self._display(pid)}: {label} (trust={trust:.1f}) — {'; '.join(recent)}")

        context = "\n".join(parts)
        if len(context) > max_chars:
            cut = context.rfind('\n', 0, max_chars)
            context = context[:cut] if cut > 0 else context[:max_chars]
        return context

    def get_all_hints_this_round(self) -> dict[str, str]:
        rn = self.current_round_state.get("round_number", 0)
        return self.round_hints.get(rn, {})

    def get_hint_timings_this_round(self) -> dict[str, int]:
        """Return elapsed seconds from round start to each hint submission.

        Returns a dict of {player_id: elapsed_seconds}. Used so agents can
        see how long each player took to hint (fast vs. slow = minor signal).
        """
        rn = self.current_round_state.get("round_number", 0)
        start = self.round_start_times.get(rn)
        stamps = self.round_hint_timestamps.get(rn, {})
        if not start:
            return {}
        return {pid: max(0, int((ts - start).total_seconds())) for pid, ts in stamps.items()}

    def get_past_hints_this_round(self) -> list[tuple[str, str]]:
        """All hints received so far this round (no ordering constraint)."""
        rn = self.current_round_state.get("round_number", 0)
        given = self.round_hints.get(rn, {})
        return [(pid, hint) for pid, hint in given.items() if pid != self.player_id]

    def get_all_discussion_this_round(self) -> list[dict]:
        rn = self.current_round_state.get("round_number", 0)
        msgs = self.round_messages.get(rn, [])
        dms = [{"type": "dm", **d} for d in self.round_dms.get(rn, [])]
        combined = msgs + dms
        return sorted(combined, key=lambda x: x.get("timestamp", ""))

    def get_group_messages_this_round(self) -> list[dict]:
        """Return only group messages for the current round (no DMs)."""
        rn = self.current_round_state.get("round_number", 0)
        return list(self.round_messages.get(rn, []))

    def get_dm_threads_this_round(self) -> dict[str, list[dict]]:
        """Return DMs grouped by conversation partner.

        Keys are the partner's player_id. Each value is a time-sorted
        list of DM dicts (with ``type='dm'`` added).
        """
        rn = self.current_round_state.get("round_number", 0)
        raw_dms = self.round_dms.get(rn, [])
        threads: dict[str, list[dict]] = {}
        for dm in raw_dms:
            partner = dm["recipient"] if dm["sender"] == self.player_id else dm["sender"]
            threads.setdefault(partner, []).append({"type": "dm", **dm})
        # Sort each thread by timestamp
        for msgs in threads.values():
            msgs.sort(key=lambda x: x.get("timestamp", ""))
        return threads

    def get_round_result(self) -> dict | None:
        rn = self.current_round_state.get("round_number", 0)
        return self.round_results.get(rn)

    def get_all_rounds_summary(self) -> str:
        parts = ["COMPLETE GAME SUMMARY:"]
        for rn in sorted(self.round_summaries):
            parts.append(self.round_summaries[rn])
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Research logging
    # ------------------------------------------------------------------

    def log_thinking(self, round_num: int, phase: str, thinking: str) -> None:
        self._thinking_traces.append({
            "round": round_num,
            "phase": phase,
            "thinking": thinking,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if thinking:
            self._round_thinking.append(thinking)

    @staticmethod
    def _summarize(text: str) -> str:
        """Truncate earlier thinking traces to keep context window tight."""
        if len(text) <= 200:
            return text
        cut = text.rfind(".", 0, 200)
        return text[: cut + 1] if cut > 50 else text[:200] + "..."

    def get_recent_thinking(self) -> str:
        """Return all thinking traces from the current round in full.

        No truncation — earlier traces are passed verbatim alongside the
        latest one so the model has its complete within-round reasoning
        history. Cross-round reasoning is fed back separately via the
        Groq-summarized round_thinking_summaries.
        """
        if not self._round_thinking:
            return ""
        lines = ["YOUR EARLIER REASONING THIS ROUND:"]
        for i, trace in enumerate(self._round_thinking, 1):
            lines.append(f"- Turn {i}: {trace}")
        return "\n".join(lines)

    async def compress_round_thinking(
        self, round_number: int, role: str = "?",
    ) -> None:
        """Summarize this round's thinking traces via Groq for cross-round recall.

        Called at the end of a round (after reflection). Uses the in-memory
        ``_round_thinking`` list before the next round clears it. Falls back
        to a simple concatenation of the first sentence of each trace if the
        Groq summary client is not available.
        """
        if not self._round_thinking:
            return
        from agents.summary_client import get_summary_client
        client = get_summary_client()
        summary: str | None = None
        if client is not None:
            try:
                summary = await client.summarize_thinking(
                    round_number=round_number,
                    role=str(role),
                    thinking_traces=list(self._round_thinking),
                )
            except Exception:  # noqa: BLE001 — fall through to fallback
                summary = None
        if not summary:
            # Fallback: keep first sentence of each trace, prefixed by phase index
            parts: list[str] = []
            for i, trace in enumerate(self._round_thinking, 1):
                first = trace.strip().split(". ", 1)[0]
                parts.append(f"[{i}] {first[:200]}")
            summary = " ".join(parts)
        self.round_thinking_summaries[round_number] = summary

    def log_reflection(self, round_num: int, reflection: str) -> None:
        self._reflections.append({
            "round": round_num,
            "reflection": reflection,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if reflection:
            self.round_reflections[round_num] = reflection

    def log_api_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        round_number: int | None = None,
        phase: str | None = None,
    ) -> None:
        from agents.pricing import cost_for
        rn = round_number if round_number is not None else self.current_round_state.get("round_number")
        cost = cost_for(model, prompt_tokens, completion_tokens)
        entry = {
            "model": model,
            "round": rn,
            "phase": phase,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._api_calls.append(entry)
        self._token_counts.append({
            "round": rn,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    def get_cost_summary(self) -> dict:
        """Aggregate API cost by round and total.

        Returns:
            {
              "model": "<last seen model>",
              "total_cost_usd": float,
              "total_prompt_tokens": int,
              "total_completion_tokens": int,
              "total_calls": int,
              "by_round": {round_number: {prompt_tokens, completion_tokens, cost_usd, calls}, ...},
            }
        """
        by_round: dict = {}
        total_cost = 0.0
        total_in = 0
        total_out = 0
        last_model = ""
        for c in self._api_calls:
            rn = c.get("round") or 0
            r = by_round.setdefault(rn, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "calls": 0,
            })
            r["prompt_tokens"] += c["prompt_tokens"]
            r["completion_tokens"] += c["completion_tokens"]
            r["cost_usd"] += c.get("cost_usd", 0.0)
            r["calls"] += 1
            total_cost += c.get("cost_usd", 0.0)
            total_in += c["prompt_tokens"]
            total_out += c["completion_tokens"]
            last_model = c.get("model", last_model)
        return {
            "model": last_model,
            "total_cost_usd": round(total_cost, 6),
            "total_prompt_tokens": total_in,
            "total_completion_tokens": total_out,
            "total_calls": len(self._api_calls),
            "by_round": {
                rn: {
                    "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"],
                    "cost_usd": round(r["cost_usd"], 6),
                    "calls": r["calls"],
                }
                for rn, r in sorted(by_round.items())
            },
        }

    def update_meta_tracker(self, player_id: str, human_confidence: float, note: str) -> None:
        """Update meta-game observation for a player."""
        if player_id not in self.meta_game_tracker:
            self.meta_game_tracker[player_id] = {"human_confidence": 0.5, "notes": []}
        self.meta_game_tracker[player_id]["human_confidence"] = human_confidence
        self.meta_game_tracker[player_id]["notes"].append(note)
