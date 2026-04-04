import json
from datetime import datetime, timezone
from pathlib import Path


class GameLogger:
    def __init__(self, session_id: str, base_dir: str = "game_sessions"):
        self.session_id = session_id
        self.session_dir = Path(base_dir) / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.game_log_path = self.session_dir / "game_log.jsonl"

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%H:%M:%SZ")

    def log(self, event: dict):
        entry = {"ts": self._ts(), **event}
        with open(self.game_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_round_assignment(
        self,
        round_number: int,
        imposter_id: str,
        topic: str,
        secret_word: str,
        grid_items: list[dict],
    ):
        self.log({
            "event": "round_assignment",
            "round": round_number,
            "imposter": imposter_id,
            "topic": topic,
            "secret_word": secret_word,
            "grid": [item["label"] for item in grid_items],
        })
