"""
GameLogger — Structured JSON persistence for research analysis.

Writes to: game_sessions/{session_id}/
├── game_log.jsonl              All server events (append-only)
├── group_chat.jsonl            Public messages chronologically
├── dm_logs/
│   └── {sender}_{target}.jsonl Per-pair private messages
├── hints/
│   └── round_{n}.json          All hints for a given round
├── votes/
│   └── round_{n}.json          All votes + outcome for a given round
├── human_data/
│   ├── {player}_classifications.json
│   └── {player}_survey.json
├── meta_game_results/
│   ├── leaderboard.json
│   ├── confusion_matrix.json
│   ├── all_classifications.json
│   ├── identities.json
│   └── classification_scores.json
├── config.json                 Session configuration snapshot
└── session_manifest.json       Summary metadata for quick analysis
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class GameLogger:
    def __init__(self, session_id: str, base_dir: str = "game_sessions"):
        self.session_id = session_id
        self.start_time = datetime.now(timezone.utc)
        self.session_dir = Path(base_dir) / session_id

        # Create directory structure
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "dm_logs").mkdir(exist_ok=True)
        (self.session_dir / "hints").mkdir(exist_ok=True)
        (self.session_dir / "votes").mkdir(exist_ok=True)
        (self.session_dir / "human_data").mkdir(exist_ok=True)
        (self.session_dir / "meta_game_results").mkdir(exist_ok=True)

        self.game_log_path = self.session_dir / "game_log.jsonl"
        self.group_chat_path = self.session_dir / "group_chat.jsonl"

    def _ts(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_jsonl(self, path: Path, entry: dict):
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _write_json(self, path: Path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Master log ───────────────────────────────────────────────

    def log(self, event: dict):
        entry = {"ts": self._ts(), **event}
        self._write_jsonl(self.game_log_path, entry)

    # ── Assignment ───────────────────────────────────────────────

    def log_round_assignment(self, round_number, imposter_id, topic, secret_word, grid_items):
        self.log({
            "event": "round_assignment",
            "round": round_number,
            "imposter": imposter_id,
            "topic": topic,
            "secret_word": secret_word,
            "grid": [item["label"] for item in grid_items],
        })

    # ── Group chat ───────────────────────────────────────────────

    def log_group_message(self, sender: str, display_name: str, content: str, round_number: int):
        entry = {
            "ts": self._ts(),
            "round": round_number,
            "sender_id": sender,
            "sender_name": display_name,
            "content": content,
        }
        self._write_jsonl(self.group_chat_path, entry)
        self.log({"event": "group_message", "sender": sender, "content": content, "round": round_number})

    # ── DMs ──────────────────────────────────────────────────────

    def log_dm(self, sender: str, target: str, sender_name: str, target_name: str, content: str, round_number: int):
        pair_key = "_".join(sorted([sender, target]))
        dm_path = self.session_dir / "dm_logs" / f"{pair_key}.jsonl"
        entry = {
            "ts": self._ts(),
            "round": round_number,
            "sender_id": sender,
            "sender_name": sender_name,
            "target_id": target,
            "target_name": target_name,
            "content": content,
        }
        self._write_jsonl(dm_path, entry)
        self.log({"event": "dm", "sender": sender, "target": target, "content": content, "round": round_number})

    # ── Hints ────────────────────────────────────────────────────

    def log_hints(self, round_number: int, hints: dict, topic: str, secret_word: str, imposter_id: str, display_names: dict):
        hints_path = self.session_dir / "hints" / f"round_{round_number}.json"
        data = {
            "round": round_number,
            "topic": topic,
            "secret_word": secret_word,
            "imposter_id": imposter_id,
            "hints": [
                {
                    "player_id": pid,
                    "display_name": display_names.get(pid, pid),
                    "hint": hint_text,
                    "is_imposter": pid == imposter_id,
                }
                for pid, hint_text in hints.items()
            ],
        }
        self._write_json(hints_path, data)
        self.log({"event": "hints_saved", "round": round_number, "count": len(hints)})

    # ── Votes ────────────────────────────────────────────────────

    def log_votes(self, round_number: int, result: dict, display_names: dict):
        votes_path = self.session_dir / "votes" / f"round_{round_number}.json"
        data = {
            "round": round_number,
            "imposter_id": result.get("imposter_id"),
            "imposter_caught": result.get("imposter_caught"),
            "votes": {
                voter: {
                    "voter_name": display_names.get(voter, voter),
                    "voted_for": target,
                    "voted_for_name": display_names.get(target, target),
                    "correct": target == result.get("imposter_id"),
                }
                for voter, target in result.get("votes", {}).items()
            },
            "scores_this_round": result.get("scores_this_round", {}),
            "cumulative_scores": result.get("cumulative_scores", {}),
        }
        self._write_json(votes_path, data)
        self.log({"event": "votes_saved", "round": round_number})

    # ── Classifications ──────────────────────────────────────────

    def log_classification(self, player_id: str, display_name: str, classifications: dict):
        path = self.session_dir / "human_data" / f"{player_id}_classifications.json"
        data = {
            "player_id": player_id,
            "display_name": display_name,
            "classifications": classifications,
            "ts": self._ts(),
        }
        self._write_json(path, data)
        self.log({"event": "classification_saved", "player": player_id})

    # ── Survey ───────────────────────────────────────────────────

    def log_survey(self, player_id: str, display_name: str, responses: dict):
        path = self.session_dir / "human_data" / f"{player_id}_survey.json"
        data = {
            "player_id": player_id,
            "display_name": display_name,
            "responses": responses,
            "ts": self._ts(),
        }
        self._write_json(path, data)
        self.log({"event": "survey_saved", "player": player_id})

    # ── Meta-game results ────────────────────────────────────────

    def log_grand_reveal(self, identities, leaderboard, classification_scores, confusion_matrix, all_classifications):
        results_dir = self.session_dir / "meta_game_results"
        self._write_json(results_dir / "leaderboard.json", leaderboard)
        self._write_json(results_dir / "confusion_matrix.json", confusion_matrix)
        self._write_json(results_dir / "all_classifications.json", all_classifications)
        self._write_json(results_dir / "identities.json", identities)
        self._write_json(results_dir / "classification_scores.json", classification_scores)
        self.log({"event": "grand_reveal_saved", "leaderboard_entries": len(leaderboard)})

    # ── Session config + manifest ────────────────────────────────

    def log_config(self, config: dict):
        self._write_json(self.session_dir / "config.json", config)

    def log_session_manifest(self, players: dict, total_rounds: int, final_scores: dict):
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds()

        player_list = []
        for pid, p_info in players.items():
            if pid == "HOST":
                continue
            player_list.append({
                "player_id": pid,
                "display_name": p_info.get("display_name", pid),
                "is_human": p_info.get("is_human", True),
                "final_score": final_scores.get(pid, 0),
            })

        manifest = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 1),
            "total_rounds": total_rounds,
            "num_humans": sum(1 for p in player_list if p["is_human"]),
            "num_ai": sum(1 for p in player_list if not p["is_human"]),
            "players": player_list,
            "final_scores": final_scores,
            "files": {
                "game_log": "game_log.jsonl",
                "group_chat": "group_chat.jsonl",
                "dm_logs": "dm_logs/",
                "hints": "hints/",
                "votes": "votes/",
                "human_data": "human_data/",
                "meta_game_results": "meta_game_results/",
            },
        }
        self._write_json(self.session_dir / "session_manifest.json", manifest)
        self.log({"event": "session_complete", "duration_seconds": round(duration, 1)})
