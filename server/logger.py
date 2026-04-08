from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone

import aiofiles


class GameLogger:
    """Handles all persistent JSONL/JSON logging for a game session."""

    def __init__(self, session_id: str, base_dir: str = "game_sessions") -> None:
        self.session_dir = Path(base_dir) / session_id

        subdirs = [
            self.session_dir,
            self.session_dir / "dm_logs",
            self.session_dir / "hints",
            self.session_dir / "votes",
            self.session_dir / "ai_internals",
            self.session_dir / "human_data",
            self.session_dir / "meta_game_results",
        ]
        for d in subdirs:
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ts(self) -> str:
        """Return the current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def _get_ai_dir(self, player_id: str, model_name: str) -> Path:
        """Build and ensure the ai_internals directory for a given AI player."""
        ai_dir = self.session_dir / "ai_internals" / f"{player_id}_{model_name}"
        os.makedirs(ai_dir, exist_ok=True)
        return ai_dir

    def _get_human_dir(self, player_id: str) -> Path:
        """Build and ensure the human_data directory for a given human player."""
        human_dir = self.session_dir / "human_data" / player_id
        os.makedirs(human_dir, exist_ok=True)
        return human_dir

    async def _append_jsonl(self, filepath: str | Path, data: dict) -> None:
        """Append a single JSON object as one line to a JSONL file."""
        async with aiofiles.open(filepath, mode="a", encoding="utf-8") as f:
            await f.write(json.dumps(data) + "\n")
            await f.flush()

    async def _write_json(self, filepath: str | Path, data: dict | list) -> None:
        """Write a complete JSON file (pretty-printed)."""
        async with aiofiles.open(filepath, mode="w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    async def log_event(self, event: str, data: dict) -> None:
        """Append a timestamped event to the master game_log.jsonl."""
        entry = {"ts": self._ts(), "event": event, **data}
        await self._append_jsonl(self.session_dir / "game_log.jsonl", entry)

    async def log_group_chat(
        self, player_id: str, content: str, round_number: int, phase: str
    ) -> None:
        """Log a group-chat message."""
        entry = {
            "ts": self._ts(),
            "player_id": player_id,
            "content": content,
            "round_number": round_number,
            "phase": phase,
        }
        await self._append_jsonl(self.session_dir / "group_chat.jsonl", entry)
        await self.log_event(
            "group_chat",
            {
                "player_id": player_id,
                "content": content,
                "round_number": round_number,
                "phase": phase,
            },
        )

    async def log_dm(
        self,
        sender_id: str,
        recipient_id: str,
        content: str,
        round_number: int,
        phase: str,
    ) -> None:
        """Log a direct message between two players."""
        key = "_".join(sorted([sender_id, recipient_id]))
        entry = {
            "ts": self._ts(),
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "content": content,
            "round_number": round_number,
            "phase": phase,
        }
        await self._append_jsonl(self.session_dir / "dm_logs" / f"{key}.jsonl", entry)
        await self.log_event(
            "dm",
            {
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "content": content,
                "round_number": round_number,
                "phase": phase,
            },
        )

    async def log_hint(self, player_id: str, hint: str, round_number: int) -> None:
        """Log a hint submission."""
        await self.log_event(
            "hint",
            {"player_id": player_id, "hint": hint, "round_number": round_number},
        )

    async def log_vote(
        self, voter_id: str, voted_for: str, round_number: int
    ) -> None:
        """Log a vote."""
        await self.log_event(
            "vote",
            {
                "voter_id": voter_id,
                "voted_for": voted_for,
                "round_number": round_number,
            },
        )

    async def log_ai_thinking(
        self,
        player_id: str,
        model_name: str,
        round_number: int,
        phase: str,
        thinking: str,
    ) -> None:
        """Log an AI player's thinking trace."""
        ai_dir = self._get_ai_dir(player_id, model_name)
        entry = {
            "ts": self._ts(),
            "round_number": round_number,
            "phase": phase,
            "thinking": thinking,
        }
        await self._append_jsonl(ai_dir / "thinking_traces.jsonl", entry)

    async def log_ai_reflection(
        self,
        player_id: str,
        model_name: str,
        round_number: int,
        reflection: str,
    ) -> None:
        """Log an AI player's reflection."""
        ai_dir = self._get_ai_dir(player_id, model_name)
        entry = {
            "ts": self._ts(),
            "round_number": round_number,
            "reflection": reflection,
        }
        await self._append_jsonl(ai_dir / "reflections.jsonl", entry)

    async def log_ai_api_call(
        self,
        player_id: str,
        model_name: str,
        request_summary: str,
        response_summary: str,
        round_number: int,
        phase: str,
    ) -> None:
        """Log an AI API call summary."""
        ai_dir = self._get_ai_dir(player_id, model_name)
        entry = {
            "ts": self._ts(),
            "round_number": round_number,
            "phase": phase,
            "request_summary": request_summary,
            "response_summary": response_summary,
        }
        await self._append_jsonl(ai_dir / "api_calls.jsonl", entry)

    async def log_ai_tokens(
        self,
        player_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        round_number: int,
        phase: str,
    ) -> None:
        """Log token usage for an AI API call."""
        ai_dir = self._get_ai_dir(player_id, model_name)
        entry = {
            "ts": self._ts(),
            "round_number": round_number,
            "phase": phase,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        await self._append_jsonl(ai_dir / "token_usage.jsonl", entry)

    async def log_human_action(
        self, player_id: str, action: str, data: dict
    ) -> None:
        """Log an action taken by a human player."""
        human_dir = self._get_human_dir(player_id)
        entry = {"ts": self._ts(), "action": action, **data}
        await self._append_jsonl(human_dir / "actions.jsonl", entry)

    # ------------------------------------------------------------------
    # Snapshot / structured-data writers
    # ------------------------------------------------------------------

    async def save_config(self, config: dict) -> None:
        """Write the session configuration."""
        await self._write_json(self.session_dir / "config.json", config)

    async def save_word_bank(self, word_bank: list[dict]) -> None:
        """Write the word bank for this session."""
        await self._write_json(self.session_dir / "word_bank.json", word_bank)

    async def save_round_hints(self, round_number: int, hints_data: dict) -> None:
        """Write aggregated hints for a round."""
        await self._write_json(
            self.session_dir / "hints" / f"round_{round_number}_hints.json",
            hints_data,
        )

    async def save_round_votes(self, round_number: int, votes_data: dict) -> None:
        """Write aggregated votes for a round."""
        await self._write_json(
            self.session_dir / "votes" / f"round_{round_number}_votes.json",
            votes_data,
        )

    async def save_classification(
        self, player_id: str, classification_data: dict
    ) -> None:
        """Save a player's classification and update the aggregate file."""
        # If the player has a human_data directory, write there
        human_dir = self.session_dir / "human_data" / player_id
        if human_dir.exists():
            await self._write_json(
                human_dir / "classification.json", classification_data
            )

        # Always update the aggregate classifications file
        agg_path = self.session_dir / "meta_game_results" / "all_classifications.json"
        existing: dict = {}
        if agg_path.exists():
            async with aiofiles.open(agg_path, mode="r", encoding="utf-8") as f:
                raw = await f.read()
                if raw.strip():
                    existing = json.loads(raw)

        existing[player_id] = classification_data
        await self._write_json(agg_path, existing)

    async def save_survey(self, player_id: str, survey_data: dict) -> None:
        """Save a human player's post-game survey."""
        human_dir = self._get_human_dir(player_id)
        await self._write_json(human_dir / "post_game_survey.json", survey_data)

    async def save_ground_truth(self, ground_truth: dict) -> None:
        """Write the ground-truth roles for the session."""
        await self._write_json(
            self.session_dir / "meta_game_results" / "ground_truth.json",
            ground_truth,
        )

    async def save_meta_game_results(self, results: dict) -> None:
        """Write the meta-game confusion matrix / results."""
        await self._write_json(
            self.session_dir / "meta_game_results" / "confusion_matrix.json",
            results,
        )

    async def save_manifest(self, manifest: dict) -> None:
        """Write the session manifest with player roster and config."""
        await self._write_json(self.session_dir / "session_manifest.json", manifest)
