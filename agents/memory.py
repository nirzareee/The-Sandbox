"""Agent game memory and context management."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

from server.models import PlayerRole

log = logging.getLogger(__name__)

_summarizer = LexRankSummarizer()
_SUMMARY_SENTENCES = 2


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

        # Meta-game tracker
        self.meta_game_tracker: dict[str, dict] = {}

        # Game start context (prepended to every prompt)
        self._game_start_context: str = ""

        # Player ID → display name mapping
        self._name_mapping: dict[str, str] = {}  # player_id → display_name

        # Per-round thinking accumulator (fed back into prompts)
        self._round_thinking: list[str] = []

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
        self.meta_game_tracker.clear()
        self._game_start_context = ""
        self._round_thinking.clear()
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

    def record_hint(self, player_id: str, hint: str) -> None:
        rn = self.current_round_state.get("round_number", 0)
        self.round_hints.setdefault(rn, {})[player_id] = hint

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

    # ------------------------------------------------------------------
    # Memory compression
    # ------------------------------------------------------------------

    def compress_round(self, round_number: int) -> None:
        """Compress a round's data into a rich summary (~300 tokens)."""
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

        self.round_summaries[round_number] = "\n".join(lines)

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

        if self.meta_game_tracker:
            parts.append("\nMETA-GAME OBSERVATIONS:")
            for pid, obs in self.meta_game_tracker.items():
                try:
                    conf = float(obs.get("human_confidence", 0.5))
                except (TypeError, ValueError):
                    conf = 0.5
                notes = ", ".join(obs.get("notes", [])[-5:])
                parts.append(f"  {self._display(pid)}: human_conf={conf:.1f} ({notes})")

        context = "\n".join(parts)
        if len(context) > max_chars:
            cut = context.rfind('\n', 0, max_chars)
            context = context[:cut] if cut > 0 else context[:max_chars]
        return context

    def get_all_hints_this_round(self) -> dict[str, str]:
        rn = self.current_round_state.get("round_number", 0)
        return self.round_hints.get(rn, {})

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
        """Extractive summary via LexRank. Falls back to truncation."""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            sentences = _summarizer(parser.document, _SUMMARY_SENTENCES)
            if sentences:
                return " ".join(str(s) for s in sentences)
        except Exception as exc:  # noqa: BLE001
            log.debug("Sumy summarization failed, falling back to truncation: %s", exc)
        return text[:200]

    def get_recent_thinking(self) -> str:
        """Return all thinking this round: older traces summarized, latest in full."""
        if not self._round_thinking:
            return ""
        lines = ["YOUR EARLIER REASONING THIS ROUND:"]
        if len(self._round_thinking) == 1:
            lines.append(f"- {self._round_thinking[0]}")
        else:
            for trace in self._round_thinking[:-1]:
                lines.append(f"- Earlier: {self._summarize(trace)}")
            lines.append(f"- Latest: {self._round_thinking[-1]}")
        return "\n".join(lines)

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
    ) -> None:
        self._api_calls.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._token_counts.append({
            "round": self.current_round_state.get("round_number"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    def update_meta_tracker(self, player_id: str, human_confidence: float, note: str) -> None:
        """Update meta-game observation for a player."""
        if player_id not in self.meta_game_tracker:
            self.meta_game_tracker[player_id] = {"human_confidence": 0.5, "notes": []}
        self.meta_game_tracker[player_id]["human_confidence"] = human_confidence
        self.meta_game_tracker[player_id]["notes"].append(note)
