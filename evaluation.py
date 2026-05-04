"""
evaluation.py — Post-game analysis script for The Sandbox.

Reads existing game session logs and computes layered metrics.
Does NOT modify any game code.

Usage:
    python evaluation.py game_sessions/session_20260305_001245
    python evaluation.py game_sessions/session_20260305_001245 --output custom_dir

Output: game_sessions/{session_id}/analysis/
    game_metrics.json
    linguistic_features.json
    hint_analysis.json
    social_network.json
    voting_blocs.json
    model_fingerprints.json
    summary_report.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional heavy deps
# ---------------------------------------------------------------------------

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLANG_TOKENS = {
    "u", "ur", "idk", "ngl", "imo", "lol", "nah", "wdym", "tbh", "sus",
    "rn", "fr", "smth", "prob", "yk", "bruh", "lmao", "omg", "nvm", "btw",
    "hbu", "bro", "af", "yo", "fs", "aight", "lowkey", "highkey", "tho",
    "smh", "w", "gg",
}

DEFINITION_PREFIXES = (
    "this is", "it is a", "a type of", "it's a", "these are",
    "kind of", "a kind", "used for", "used to", "known as",
)

ACCUSATION_KEYWORDS = {
    "sus", "suspicious", "imposter", "impostor", "vote", "voting",
    "weird", "vague", "fake", "lying", "lie", "cap", "bluffing", "bluff",
}

EMOJI_RE = re.compile(
    r"[\U00010000-\U0010ffff"
    r"\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF"
    r"\u2600-\u26FF\u2700-\u27BF"
    r"[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]"
    r"]",
    flags=re.UNICODE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    return (numerator / denominator) if denominator > 0 else None


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _stdev(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, skipping bad lines."""
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip malformed lines silently
    except (OSError, IOError):
        pass
    return records


def _load_json(path: str) -> Optional[dict]:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, IOError, json.JSONDecodeError):
        return None


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stripping punctuation."""
    return re.findall(r"[a-z']+", text.lower())


def _has_emoji(text: str) -> bool:
    return bool(EMOJI_RE.search(text))


def _is_definition_like(hint: str) -> bool:
    lower = hint.lower().strip()
    return any(lower.startswith(p) for p in DEFINITION_PREFIXES)


def _contains_slang(text: str) -> bool:
    tokens = set(_tokenize(text))
    return bool(tokens & SLANG_TOKENS)


# ---------------------------------------------------------------------------
# SessionData — loads and indexes all session files
# ---------------------------------------------------------------------------

class SessionData:
    """Loads and normalizes all data from a game session directory."""

    def __init__(self, session_dir: str) -> None:
        self.session_dir = Path(session_dir).resolve()
        self.warnings: list[str] = []

        # ---- manifest ----
        self._load_manifest()

        # ---- config ----
        self._load_config()

        # ---- game log events (categorized) ----
        self.all_events: list[dict] = []
        self.round_start_events: list[dict] = []
        self.hint_events: list[dict] = []
        self.group_chat_events: list[dict] = []
        self.dm_events_from_log: list[dict] = []
        self.vote_events: list[dict] = []
        self.round_results_events: list[dict] = []
        self.game_started_event: Optional[dict] = None
        self.game_finished_event: Optional[dict] = None
        self._load_game_log()

        # ---- group_chat.jsonl (dedicated file) ----
        self.group_chat_messages: list[dict] = []
        self._load_group_chat()

        # ---- dm_logs/*.jsonl ----
        self.dm_messages: list[dict] = []          # flat list
        self.dm_pairs: dict[str, list[dict]] = {}  # pair_key -> messages
        self._load_dms()

        # ---- word_bank.json ----
        self.word_bank: list[dict] = []
        self._load_word_bank()

        # ---- classifications (meta_game_results/) ----
        self.classifications: Optional[dict] = None
        self.ground_truth: Optional[dict] = None
        self.precomputed_confusion: Optional[dict] = None
        self._load_classifications()

        # ---- post-game survey responses ----
        self._load_surveys()

        # ---- derived properties ----
        self.num_rounds_played: int = len(self.round_results_events)
        self.num_rounds_configured: int = self.config.get("num_rounds", 5)
        self.is_complete: bool = self.game_finished_event is not None

        # ---- backfill display_names from recovered roster ----
        # Legacy sessions persisted display_name as "Player" before the
        # shuffle ran. Use the post-shuffle pN map (from manifest, or
        # recovered from thinking traces) to set player labels properly.
        self._backfill_display_names_from_recovered_roster()

    # ---- loaders ----

    def _backfill_display_names_from_recovered_roster(self) -> None:
        """If display_names are placeholders ('Player'), recover the real
        post-shuffle Player_N mapping from agent thinking traces and
        update game_players + all_players entries in place."""
        # Heuristic: a roster has been recovered if every game player's
        # current display_name is a placeholder like 'Player' (no suffix).
        needs_backfill = all(
            (info.get("display_name") in (None, "", "Player"))
            for info in self.game_players.values()
        )
        if not needs_backfill:
            return
        # Lazy import to avoid circular module-load order.
        try:
            from evaluation_metrics import _build_global_pn_map
        except ImportError:
            return
        pn_map = _build_global_pn_map(self)
        if not pn_map:
            return
        for pos, pid in pn_map.items():
            label = f"Player_{pos}"
            if pid in self.game_players:
                self.game_players[pid]["display_name"] = label
            if pid in self.all_players:
                self.all_players[pid]["display_name"] = label
        self.warnings.append(
            f"Recovered Player_N display names from thinking traces "
            f"({len(pn_map)}/{len(self.game_players)} agents)"
        )

    def _load_manifest(self) -> None:
        """Load session_manifest.json. Falls back on prefix-based inference."""
        path = self.session_dir / "session_manifest.json"
        data = _load_json(str(path))
        if data is None:
            self.warnings.append("session_manifest.json missing or unreadable")
            data = {}

        raw_players: dict = data.get("players", {})
        self.session_id: str = data.get("session_id", self.session_dir.name)
        self.started_at: str = data.get("started_at", "")

        # Post-shuffle Player_1..N roster (added 2026-05-04). Older sessions
        # don't have this; fall back to inferring from agent thinking traces
        # in _recover_shuffled_roster_from_traces.
        self.shuffled_roster: dict[str, str] = data.get("shuffled_roster") or {}

        self.all_players: dict[str, dict] = {}
        self.game_players: dict[str, dict] = {}  # excludes HOST

        for pid, info in raw_players.items():
            ptype = info.get("player_type", "")
            # Infer type from prefix if missing
            if not ptype:
                ptype = "HUMAN" if pid.startswith("human_") else "AI"
                self.warnings.append(f"Inferred player_type={ptype} for {pid}")
            # real_name: the human's actual chosen name, or the agent seat ID.
            # Falls back to player_id so labels are never blank.
            real_name = info.get("real_name") or pid
            # Prefer the shuffled-roster name when available (current sessions).
            # Older sessions persisted display_name as a literal "Player"
            # placeholder before the shuffle ran.
            shuffled = self.shuffled_roster.get(pid)
            display = shuffled or info.get("display_name") or pid
            normalized = {
                "player_id": pid,
                "display_name": display,
                "real_name": real_name,
                "player_type": ptype,
                "model_name": info.get("model_name"),
            }
            self.all_players[pid] = normalized
            if ptype != "HOST":
                self.game_players[pid] = normalized

        if not raw_players:
            self.warnings.append("No players found in manifest")

    def _load_config(self) -> None:
        path = self.session_dir / "config.json"
        data = _load_json(str(path))
        if data is None:
            self.warnings.append("config.json missing or unreadable")
            data = {}
        self.config: dict = data

    def _load_game_log(self) -> None:
        path = self.session_dir / "game_log.jsonl"
        records = _load_jsonl(str(path))
        if not records:
            self.warnings.append("game_log.jsonl missing or empty")

        for ev in records:
            self.all_events.append(ev)
            etype = ev.get("event", "")
            if etype == "game_started":
                self.game_started_event = ev
            elif etype == "round_started":
                self.round_start_events.append(ev)
            elif etype == "hint":
                self.hint_events.append(ev)
            elif etype == "group_chat":
                self.group_chat_events.append(ev)
            elif etype == "dm":
                self.dm_events_from_log.append(ev)
            elif etype == "vote":
                self.vote_events.append(ev)
            elif etype == "round_results":
                self.round_results_events.append(ev)
            elif etype in ("game_finished", "game_ended"):
                self.game_finished_event = ev

    def _load_group_chat(self) -> None:
        path = self.session_dir / "group_chat.jsonl"
        records = _load_jsonl(str(path))
        # Deduplicate against game_log events (prefer dedicated file if present)
        if records:
            self.group_chat_messages = records
        else:
            # Fall back to events extracted from game_log
            self.group_chat_messages = self.group_chat_events
            if not self.group_chat_events:
                self.warnings.append("group_chat.jsonl missing and no group_chat events in game_log")

    def _load_dms(self) -> None:
        dm_dir = self.session_dir / "dm_logs"
        if not dm_dir.exists():
            # Fall back to dm events from game_log
            self.dm_messages = self.dm_events_from_log
            if not self.dm_events_from_log:
                self.warnings.append("dm_logs/ directory not found, no DM data")
            return

        for filepath in sorted(dm_dir.glob("*.jsonl")):
            pair_key = filepath.stem  # e.g. "agent_2_agent_3"
            records = _load_jsonl(str(filepath))
            self.dm_pairs[pair_key] = records
            self.dm_messages.extend(records)

        if not self.dm_messages:
            # fall back to game_log DM events
            self.dm_messages = self.dm_events_from_log

    def _load_word_bank(self) -> None:
        path = self.session_dir / "word_bank.json"
        data = _load_json(str(path))
        if isinstance(data, list):
            self.word_bank = data
        elif isinstance(data, dict):
            self.word_bank = [data]

    def _load_surveys(self) -> None:
        """Load every per-human post_game_survey.json into self.surveys.

        Tolerates both the old shape (raw response dict at top level) and
        the new shape (wrapped: {responses, is_partial, real_name, ...}).
        """
        self.surveys: dict[str, dict] = {}
        human_dir = self.session_dir / "human_data"
        if not human_dir.exists():
            return
        for player_dir in human_dir.iterdir():
            if not player_dir.is_dir():
                continue
            survey_path = player_dir / "post_game_survey.json"
            if not survey_path.exists():
                continue
            data = _load_json(str(survey_path))
            if not data:
                continue
            if isinstance(data, dict) and "responses" in data and isinstance(data["responses"], dict):
                # New wrapped shape.
                wrapped = dict(data)
            else:
                # Legacy raw shape — wrap so downstream code has a single shape.
                # Backfill identity from manifest when possible so old sessions
                # look as informative as ones saved with the new logger.
                pid = player_dir.name
                manifest_entry = self.all_players.get(pid, {})
                wrapped = {
                    "submitted_at": None,
                    "is_partial": False,
                    "player_id": pid,
                    "real_name": manifest_entry.get("real_name") or pid,
                    "player_type": manifest_entry.get("player_type") or "HUMAN",
                    "responses": data,
                }
            self.surveys[player_dir.name] = wrapped

    def _load_classifications(self) -> None:
        meta_dir = self.session_dir / "meta_game_results"
        if not meta_dir.exists():
            return

        classifications_path = meta_dir / "all_classifications.json"
        ground_truth_path = meta_dir / "ground_truth.json"
        confusion_path = meta_dir / "confusion_matrix.json"

        self.classifications = _load_json(str(classifications_path))
        self.ground_truth = _load_json(str(ground_truth_path))
        self.precomputed_confusion = _load_json(str(confusion_path))

    # ---- convenience accessors ----

    def player_type(self, pid: str) -> str:
        return self.game_players.get(pid, {}).get("player_type", "UNKNOWN")

    def player_model(self, pid: str) -> Optional[str]:
        return self.game_players.get(pid, {}).get("model_name")

    def player_real_name(self, pid: str) -> str:
        info = self.game_players.get(pid) or self.all_players.get(pid) or {}
        return info.get("real_name") or pid

    def player_display_name(self, pid: str) -> str:
        info = self.game_players.get(pid) or self.all_players.get(pid) or {}
        return info.get("display_name") or pid

    def player_identity(self, pid: str) -> dict:
        """Canonical identity tuple used everywhere in analysis output.

        Always returns the same shape:
          {player_id, real_name, display_name, player_type, model_name, label}
        `label` is a one-line human-readable identifier safe for tables and
        chart legends — e.g. `agent_1 · "agent_1" (AI · DeepSeek-R1)`.
        """
        info = self.game_players.get(pid) or self.all_players.get(pid) or {}
        ptype = info.get("player_type", "UNKNOWN")
        model = info.get("model_name")
        real = info.get("real_name") or pid
        display = info.get("display_name") or pid
        # Short model name for compact labels (drop org prefix)
        short_model = model.split("/")[-1] if model else None
        if ptype == "AI":
            type_part = f"AI · {short_model}" if short_model else "AI"
        elif ptype == "HUMAN":
            type_part = "HUMAN"
        else:
            type_part = ptype
        label = f'{pid} · "{real}" ({type_part})'
        return {
            "player_id": pid,
            "real_name": real,
            "display_name": display,
            "player_type": ptype,
            "model_name": model,
            "label": label,
        }

    def players_directory(self) -> dict:
        """Identity directory included in every analysis JSON.

        Maps player_id -> identity dict. HOST excluded. Single source of
        truth so consumers (dashboard, reports, JSON readers) never have to
        cross-reference manifest fields themselves.
        """
        return {pid: self.player_identity(pid) for pid in self.game_players}

    def round_info(self, round_number: int) -> Optional[dict]:
        for ev in self.round_start_events:
            if ev.get("round_number") == round_number:
                return ev
        return None

    def imposter_for_round(self, round_number: int) -> Optional[str]:
        info = self.round_info(round_number)
        return info.get("imposter_id") if info else None

    @property
    def human_players(self) -> list[str]:
        return [pid for pid, info in self.game_players.items()
                if info["player_type"] == "HUMAN"]

    @property
    def ai_players(self) -> list[str]:
        return [pid for pid, info in self.game_players.items()
                if info["player_type"] == "AI"]


# ---------------------------------------------------------------------------
# Layer 1: Game Metrics
# ---------------------------------------------------------------------------

def compute_game_metrics(session: SessionData) -> dict:
    """Layer 1: per-round results, detection rates, vote accuracy."""

    # ---- Per-round results ----
    rounds = []
    for ev in session.round_results_events:
        rn = ev.get("round_number")
        imp_id = ev.get("imposter_id")
        round_entry = {
            "round_number": rn,
            "topic": None,
            "secret_word": None,
            "imposter_id": imp_id,
            "imposter_player_type": session.player_type(imp_id) if imp_id else None,
            "imposter_model": session.player_model(imp_id) if imp_id else None,
            "imposter_caught": ev.get("imposter_caught"),
            "votes": ev.get("votes", {}),
            "correct_voters": ev.get("correct_voters", []),
            "scores_this_round": ev.get("scores_this_round", {}),
            "cumulative_scores": ev.get("cumulative_scores", {}),
        }
        # Enrich with round_started data
        rinfo = session.round_info(rn)
        if rinfo:
            round_entry["topic"] = rinfo.get("topic")
            round_entry["secret_word"] = rinfo.get("secret_word")
        rounds.append(round_entry)

    # ---- Imposter detection rates ----
    total_rounds = len(rounds)
    caught_rounds = [r for r in rounds if r["imposter_caught"]]
    overall_rate = _safe_div(len(caught_rounds), total_rounds)

    # By imposter type
    human_imp_rounds = [r for r in rounds if r["imposter_player_type"] == "HUMAN"]
    ai_imp_rounds = [r for r in rounds if r["imposter_player_type"] == "AI"]
    detection_by_imposter_type = {
        "HUMAN": _safe_div(
            sum(1 for r in human_imp_rounds if r["imposter_caught"]),
            len(human_imp_rounds)
        ),
        "AI": _safe_div(
            sum(1 for r in ai_imp_rounds if r["imposter_caught"]),
            len(ai_imp_rounds)
        ),
    }

    # By imposter model
    model_rounds: dict[str, list] = defaultdict(list)
    for r in rounds:
        model = r["imposter_model"] or "HUMAN"
        model_rounds[model].append(r)
    detection_by_model = {
        model: _safe_div(
            sum(1 for r in rlist if r["imposter_caught"]),
            len(rlist)
        )
        for model, rlist in model_rounds.items()
    }

    # ---- Vote accuracy per player ----
    # Rounds where the player WAS the imposter are skipped — the imposter
    # cannot vote for themselves, so any vote they cast is structurally
    # never "correct," and counting them dilutes the metric.
    vote_accuracy: dict[str, dict] = {}
    for pid in session.game_players:
        correct = 0
        total = 0
        rounds_as_imposter = 0
        for r in rounds:
            if r.get("imposter_id") == pid:
                rounds_as_imposter += 1
                continue
            voted_for = r["votes"].get(pid)
            if voted_for is not None:
                total += 1
                if voted_for == r["imposter_id"]:
                    correct += 1
        vote_accuracy[pid] = {
            "correct_votes": correct,
            "total_votes": total,
            "rounds_as_imposter": rounds_as_imposter,
            "accuracy": _safe_div(correct, total),
            "player_type": session.player_type(pid),
            "model_name": session.player_model(pid),
        }

    # ---- Average vote accuracy by player type ----
    human_acc = [v["accuracy"] for v in vote_accuracy.values()
                 if v["player_type"] == "HUMAN" and v["accuracy"] is not None]
    ai_acc = [v["accuracy"] for v in vote_accuracy.values()
              if v["player_type"] == "AI" and v["accuracy"] is not None]
    accuracy_by_type = {
        "HUMAN": _mean(human_acc),
        "AI": _mean(ai_acc),
    }

    # ---- Detection over rounds (learning curve) ----
    detection_curve: list[dict] = []
    human_correct_cumul = 0
    human_total_cumul = 0
    ai_correct_cumul = 0
    ai_total_cumul = 0

    for r in rounds:
        rn = r["round_number"]
        imp_id = r["imposter_id"]
        for pid, voted_for in r["votes"].items():
            # Skip the imposter's own vote — structurally never "correct"
            # since they cannot vote for themselves. Including it drags
            # the cumulative detection rate down by ~1/N every round.
            if pid == imp_id:
                continue
            ptype = session.player_type(pid)
            correct = 1 if voted_for == imp_id else 0
            if ptype == "HUMAN":
                human_correct_cumul += correct
                human_total_cumul += 1
            elif ptype == "AI":
                ai_correct_cumul += correct
                ai_total_cumul += 1
        detection_curve.append({
            "round_number": rn,
            "imposter_caught": r["imposter_caught"],
            "human_cumul_accuracy": _safe_div(human_correct_cumul, human_total_cumul),
            "ai_cumul_accuracy": _safe_div(ai_correct_cumul, ai_total_cumul),
        })

    # ---- Classification confusion matrix ----
    confusion_matrix = None
    if session.precomputed_confusion:
        confusion_matrix = session.precomputed_confusion
    elif session.classifications and session.ground_truth:
        confusion_matrix = _compute_confusion_matrix(
            session.classifications, session.ground_truth
        )

    # ---- Final scores ----
    final_scores: dict = {}
    if rounds:
        final_scores = rounds[-1].get("cumulative_scores", {})

    # Enrich each round with imposter identity + per-vote identity tuples
    for r in rounds:
        imp_id = r["imposter_id"]
        if imp_id:
            r["imposter_identity"] = session.player_identity(imp_id)
        votes_with_identity = {}
        for voter_id, target_id in (r.get("votes") or {}).items():
            votes_with_identity[voter_id] = {
                "voter": session.player_identity(voter_id),
                "target": session.player_identity(target_id) if target_id else None,
                "correct": (target_id == imp_id) if (target_id and imp_id) else None,
            }
        r["votes_detailed"] = votes_with_identity

    # Attach identity to per-player vote accuracy
    for pid, stats in vote_accuracy.items():
        stats["identity"] = session.player_identity(pid)

    return {
        "session_id": session.session_id,
        "num_rounds_configured": session.num_rounds_configured,
        "num_rounds_played": session.num_rounds_played,
        "is_complete": session.is_complete,
        "players_directory": session.players_directory(),
        "rounds": rounds,
        "overall_detection_rate": overall_rate,
        "detection_by_imposter_type": detection_by_imposter_type,
        "detection_by_model": detection_by_model,
        "vote_accuracy_per_player": vote_accuracy,
        "accuracy_by_player_type": accuracy_by_type,
        "detection_over_rounds": detection_curve,
        "confusion_matrix": confusion_matrix,
        "final_scores": final_scores,
    }


def _compute_confusion_matrix(
    classifications: dict, ground_truth: dict
) -> dict:
    """Replicate scoring.py build_confusion_matrix logic for standalone use."""
    per_player: dict = {}
    agg = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for player_id, player_classifications in classifications.items():
        tp = tn = fp = fn = 0
        for target_id, classification in player_classifications.items():
            guess = (classification.get("guess") or "").lower()
            actual = (ground_truth.get(target_id) or "").upper()
            if not actual:
                continue
            if actual == "AI":
                if guess == "ai":
                    tp += 1
                else:
                    fn += 1
            elif actual == "HUMAN":
                if guess == "human":
                    tn += 1
                else:
                    fp += 1
        total = tp + tn + fp + fn
        per_player[player_id] = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": _safe_div(tp + tn, total),
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, tp + fn),
        }
        for k in ("tp", "tn", "fp", "fn"):
            agg[k] += per_player[player_id][k]

    agg_total = agg["tp"] + agg["tn"] + agg["fp"] + agg["fn"]
    return {
        "per_player": per_player,
        "aggregate": {
            **agg,
            "accuracy": _safe_div(agg["tp"] + agg["tn"], agg_total),
            "precision": _safe_div(agg["tp"], agg["tp"] + agg["fp"]),
            "recall": _safe_div(agg["tp"], agg["tp"] + agg["fn"]),
        },
    }


# ---------------------------------------------------------------------------
# Layer 2: Linguistic Features
# ---------------------------------------------------------------------------

def compute_linguistic_features(session: SessionData) -> dict:
    """Layer 2: per-player message statistics and style features."""

    # Gather all messages per player
    player_group_msgs: dict[str, list[str]] = defaultdict(list)
    player_dm_msgs: dict[str, list[str]] = defaultdict(list)

    for msg in session.group_chat_messages:
        pid = msg.get("player_id")
        content = msg.get("content", "")
        if pid and pid in session.game_players and content:
            player_group_msgs[pid].append(content)

    for msg in session.dm_messages:
        pid = msg.get("sender_id")
        content = msg.get("content", "")
        if pid and pid in session.game_players and content:
            player_dm_msgs[pid].append(content)

    features: dict[str, dict] = {}

    for pid in session.game_players:
        group_msgs = player_group_msgs[pid]
        dm_msgs = player_dm_msgs[pid]
        all_msgs = group_msgs + dm_msgs

        if not all_msgs:
            features[pid] = _empty_linguistic_features(
                session.player_type(pid),
                session.player_model(pid),
                session.player_identity(pid),
            )
            continue

        # Message counts
        msg_count = len(all_msgs)
        group_msg_count = len(group_msgs)
        dm_msg_count = len(dm_msgs)

        # Length stats
        char_lengths = [len(m) for m in all_msgs]
        words_per_msg = [len(m.split()) for m in all_msgs]
        all_words = []
        for m in all_msgs:
            all_words.extend(_tokenize(m))

        avg_msg_length_chars = _mean(char_lengths)
        std_msg_length_chars = _stdev(char_lengths)
        avg_msg_length_words = _mean(words_per_msg)
        avg_word_length = _mean([len(w) for w in all_words]) if all_words else None

        # Vocabulary diversity (type-token ratio)
        vocab_diversity = _safe_div(len(set(all_words)), len(all_words))

        # First word diversity
        first_words = []
        for m in all_msgs:
            tokens = _tokenize(m)
            if tokens:
                first_words.append(tokens[0])
        first_word_diversity = _safe_div(len(set(first_words)), len(first_words))

        # Slang rate
        slang_msgs = sum(1 for m in all_msgs if _contains_slang(m))
        slang_rate = _safe_div(slang_msgs, msg_count)

        # Question / exclamation rates
        question_rate = _safe_div(
            sum(1 for m in all_msgs if "?" in m), msg_count
        )
        exclamation_rate = _safe_div(
            sum(1 for m in all_msgs if "!" in m), msg_count
        )

        # Capitalization rate (uppercase chars / total alpha chars)
        total_alpha = sum(c.isalpha() for m in all_msgs for c in m)
        total_upper = sum(c.isupper() for m in all_msgs for c in m)
        capitalization_rate = _safe_div(total_upper, total_alpha)

        # Period and comma rates (per message)
        period_rate = _safe_div(
            sum(m.count(".") for m in all_msgs), msg_count
        )
        comma_rate = _safe_div(
            sum(m.count(",") for m in all_msgs), msg_count
        )

        # Emoji rate
        emoji_rate = _safe_div(
            sum(1 for m in all_msgs if _has_emoji(m)), msg_count
        )

        features[pid] = {
            "player_id": pid,
            "real_name": session.player_real_name(pid),
            "player_type": session.player_type(pid),
            "model_name": session.player_model(pid),
            "identity": session.player_identity(pid),
            "msg_count": msg_count,
            "group_msg_count": group_msg_count,
            "dm_msg_count": dm_msg_count,
            "avg_msg_length_chars": avg_msg_length_chars,
            "std_msg_length_chars": std_msg_length_chars,
            "avg_msg_length_words": avg_msg_length_words,
            "avg_word_length": avg_word_length,
            "vocab_diversity": vocab_diversity,
            "first_word_diversity": first_word_diversity,
            "slang_rate": slang_rate,
            "question_rate": question_rate,
            "exclamation_rate": exclamation_rate,
            "capitalization_rate": capitalization_rate,
            "period_rate": period_rate,
            "comma_rate": comma_rate,
            "emoji_rate": emoji_rate,
        }

    # Group-level summaries
    human_feats = [v for v in features.values() if v["player_type"] == "HUMAN"]
    ai_feats = [v for v in features.values() if v["player_type"] == "AI"]

    numeric_keys = [
        "avg_msg_length_chars", "avg_msg_length_words", "vocab_diversity",
        "slang_rate", "question_rate", "exclamation_rate", "capitalization_rate",
        "period_rate", "comma_rate", "emoji_rate", "first_word_diversity",
    ]

    def _group_avg(player_list: list[dict], key: str) -> Optional[float]:
        vals = [p[key] for p in player_list if p.get(key) is not None]
        return _mean(vals)

    human_summary = {k: _group_avg(human_feats, k) for k in numeric_keys}
    ai_summary = {k: _group_avg(ai_feats, k) for k in numeric_keys}

    # Per-model summaries
    model_summaries: dict[str, dict] = {}
    model_groups: dict[str, list] = defaultdict(list)
    for pdata in ai_feats:
        model = pdata.get("model_name") or "unknown"
        model_groups[model].append(pdata)
    for model, plist in model_groups.items():
        model_summaries[model] = {k: _group_avg(plist, k) for k in numeric_keys}

    return {
        "players_directory": session.players_directory(),
        "players": features,
        "group_summary": {
            "HUMAN": human_summary,
            "AI": ai_summary,
        },
        "per_model_summary": model_summaries,
    }


def _empty_linguistic_features(
    player_type: str, model_name: Optional[str], identity: Optional[dict] = None
) -> dict:
    return {
        "player_type": player_type,
        "model_name": model_name,
        "identity": identity,
        "msg_count": 0,
        "group_msg_count": 0,
        "dm_msg_count": 0,
        "avg_msg_length_chars": None,
        "std_msg_length_chars": None,
        "avg_msg_length_words": None,
        "avg_word_length": None,
        "vocab_diversity": None,
        "first_word_diversity": None,
        "slang_rate": None,
        "question_rate": None,
        "exclamation_rate": None,
        "capitalization_rate": None,
        "period_rate": None,
        "comma_rate": None,
        "emoji_rate": None,
    }


def compute_hint_analysis(session: SessionData) -> dict:
    """Layer 2 (hints): per-hint features and grouped stats."""

    # Build per-round imposter lookup
    round_imposters = {
        ev.get("round_number"): ev.get("imposter_id")
        for ev in session.round_start_events
    }

    per_hint_records: list[dict] = []
    per_player_hints: dict[str, list[dict]] = defaultdict(list)

    for ev in session.hint_events:
        pid = ev.get("player_id")
        hint_text = ev.get("hint", "")
        rn = ev.get("round_number")

        if not pid or pid not in session.game_players:
            continue

        imp_id = round_imposters.get(rn)
        words = hint_text.split()

        record = {
            "player_id": pid,
            "real_name": session.player_real_name(pid),
            "player_type": session.player_type(pid),
            "model_name": session.player_model(pid),
            "identity": session.player_identity(pid),
            "round_number": rn,
            "hint": hint_text,
            "hint_length_words": len(words),
            "hint_length_chars": len(hint_text),
            "has_slang": _contains_slang(hint_text),
            "is_definition_like": _is_definition_like(hint_text),
            "hint_was_imposter": (pid == imp_id),
        }
        per_hint_records.append(record)
        per_player_hints[pid].append(record)

    # Per-player aggregates
    player_hint_stats: dict[str, dict] = {}
    for pid in session.game_players:
        hints = per_player_hints[pid]
        if not hints:
            player_hint_stats[pid] = {
                "identity": session.player_identity(pid),
                "player_type": session.player_type(pid),
                "model_name": session.player_model(pid),
                "hint_count": 0,
            }
            continue
        player_hint_stats[pid] = {
            "identity": session.player_identity(pid),
            "player_type": session.player_type(pid),
            "model_name": session.player_model(pid),
            "hint_count": len(hints),
            "avg_hint_length_words": _mean([h["hint_length_words"] for h in hints]),
            "avg_hint_length_chars": _mean([h["hint_length_chars"] for h in hints]),
            "slang_hint_rate": _safe_div(
                sum(1 for h in hints if h["has_slang"]), len(hints)
            ),
            "definition_like_rate": _safe_div(
                sum(1 for h in hints if h["is_definition_like"]), len(hints)
            ),
            "hints_as_imposter": sum(1 for h in hints if h["hint_was_imposter"]),
        }

    # Group summaries
    def _hint_group_avg(ptype_or_model: str, key: str, by_model: bool = False) -> Optional[float]:
        vals = []
        for pid, stats in player_hint_stats.items():
            if stats.get("hint_count", 0) == 0:
                continue
            match = (
                stats.get("model_name") == ptype_or_model
                if by_model
                else stats.get("player_type") == ptype_or_model
            )
            if match and stats.get(key) is not None:
                vals.append(stats[key])
        return _mean(vals)

    summary_keys = ["avg_hint_length_words", "avg_hint_length_chars",
                    "slang_hint_rate", "definition_like_rate"]
    group_summary = {
        "HUMAN": {k: _hint_group_avg("HUMAN", k) for k in summary_keys},
        "AI": {k: _hint_group_avg("AI", k) for k in summary_keys},
    }

    model_hint_summary: dict[str, dict] = {}
    all_models = {
        stats["model_name"]
        for stats in player_hint_stats.values()
        if stats.get("model_name")
    }
    for model in all_models:
        model_hint_summary[model] = {k: _hint_group_avg(model, k, by_model=True) for k in summary_keys}

    return {
        "players_directory": session.players_directory(),
        "per_hint": per_hint_records,
        "per_player": player_hint_stats,
        "group_summary": group_summary,
        "per_model_summary": model_hint_summary,
    }


# ---------------------------------------------------------------------------
# Layer 3: Social Network
# ---------------------------------------------------------------------------

def compute_social_network(session: SessionData) -> dict:
    """Layer 3: DM network, accusation graph."""

    # ---- DM network ----
    # Build thread-level data: who sent the first message to whom
    # A "thread" between A and B = sorted pair
    thread_initiators: dict[str, str] = {}  # pair_key -> initiator player_id
    thread_messages: dict[str, list[dict]] = defaultdict(list)

    all_dms = sorted(session.dm_messages, key=lambda m: m.get("ts", ""))

    for msg in all_dms:
        sender = msg.get("sender_id", "")
        recipient = msg.get("recipient_id", "")
        if not sender or not recipient:
            continue
        pair_key = "_".join(sorted([sender, recipient]))
        thread_messages[pair_key].append(msg)
        if pair_key not in thread_initiators:
            thread_initiators[pair_key] = sender

    # Per-player DM aggregates
    player_dm_stats: dict[str, dict] = {}
    for pid in session.game_players:
        threads_initiated = sum(
            1 for pair_key, init in thread_initiators.items()
            if init == pid and pid in pair_key
        )
        threads_received = sum(
            1 for pair_key, init in thread_initiators.items()
            if init != pid and pid in pair_key.split("_", 1)[0] + "_" + pair_key.split("_", 1)[-1]
            and pid in pair_key
        )
        # More precise: count pairs where this player appears but didn't initiate
        threads_received = sum(
            1 for pair_key, init in thread_initiators.items()
            if pid in pair_key.split("_") and init != pid
        )

        total_sent = sum(
            1 for msg in all_dms if msg.get("sender_id") == pid
        )
        total_received = sum(
            1 for msg in all_dms if msg.get("recipient_id") == pid
        )

        # Reply rate: for threads where pid received first msg, did they reply?
        replied = 0
        received_threads = [
            pk for pk, init in thread_initiators.items()
            if pk.replace(pid, "").lstrip("_").rstrip("_") and pid in pk.split("_") and init != pid
        ]
        for pk in received_threads:
            msgs = thread_messages[pk]
            pid_replied = any(m.get("sender_id") == pid for m in msgs[1:])
            if pid_replied:
                replied += 1
        reply_rate = _safe_div(replied, len(received_threads))

        group_count = sum(
            1 for m in session.group_chat_messages if m.get("player_id") == pid
        )
        group_to_dm_ratio = _safe_div(group_count, group_count + total_sent)

        player_dm_stats[pid] = {
            "player_id": pid,
            "real_name": session.player_real_name(pid),
            "player_type": session.player_type(pid),
            "model_name": session.player_model(pid),
            "identity": session.player_identity(pid),
            "dm_threads_initiated": threads_initiated,
            "dm_threads_received": threads_received,
            "dm_reply_rate": reply_rate,
            "total_dms_sent": total_sent,
            "total_dms_received": total_received,
            "group_to_dm_ratio": group_to_dm_ratio,
        }

    # DM pair details
    dm_pair_details: dict[str, dict] = {}
    for pair_key, msgs in thread_messages.items():
        parts = pair_key.split("_")
        # Re-derive the two player IDs from initiator and the pair key
        initiator = thread_initiators[pair_key]
        other = None
        # Pair key is sorted join; get other by removing initiator tokens
        # Handle multi-token player IDs like "human_hfn6" by checking
        all_pids = sorted(session.game_players.keys())
        pair_players = [p for p in all_pids if p in pair_key]
        if len(pair_players) == 2:
            a, b = pair_players[0], pair_players[1]
        else:
            continue  # ambiguous, skip

        dm_pair_details[pair_key] = {
            "players": [a, b],
            "initiator": initiator,
            "total_messages": len(msgs),
            f"{a}_to_{b}": sum(1 for m in msgs if m.get("sender_id") == a),
            f"{b}_to_{a}": sum(1 for m in msgs if m.get("sender_id") == b),
            "rounds_active": sorted(
                set(m.get("round_number") for m in msgs if m.get("round_number"))
            ),
        }

    # ---- Accusation detection ----
    accusation_graph: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    player_name_patterns = {}
    for pid in session.game_players:
        display = session.game_players[pid].get("display_name", pid)
        # Match player_id or display name (case-insensitive)
        player_name_patterns[pid] = re.compile(
            r"\b(" + re.escape(pid) + r"|" + re.escape(display) + r")\b",
            re.IGNORECASE,
        )

    for msg in session.group_chat_messages:
        sender = msg.get("player_id")
        content = msg.get("content", "")
        if not sender or sender not in session.game_players:
            continue
        content_lower = content.lower()
        words = set(re.findall(r"\b\w+\b", content_lower))
        if not (words & ACCUSATION_KEYWORDS):
            continue
        # Check if this message references another player
        for target_pid, pattern in player_name_patterns.items():
            if target_pid == sender:
                continue
            if pattern.search(content):
                accusation_graph[sender][target_pid] += 1

    # Convert defaultdicts to plain dicts for JSON serialization
    accusation_graph_plain = {
        k: dict(v) for k, v in accusation_graph.items()
    }

    return {
        "players_directory": session.players_directory(),
        "dm_network": {
            "per_player": player_dm_stats,
            "pairs": dm_pair_details,
        },
        "accusation_graph": accusation_graph_plain,
    }


def compute_voting_blocs(session: SessionData) -> dict:
    """Layer 3: voting agreement analysis between player pairs."""

    players = list(session.game_players.keys())
    rounds = session.round_results_events

    if not rounds or len(players) < 2:
        return {
            "pairs": {},
            "category_averages": {},
            "note": "Insufficient data for voting bloc analysis",
        }

    # Build per-pair agreement data
    pair_agreement: dict[str, dict] = {}

    for i, pa in enumerate(players):
        for pb in players[i + 1:]:
            agreed = 0
            both_voted = 0
            agreed_rounds = []

            for r in rounds:
                votes = r.get("votes", {})
                va = votes.get(pa)
                vb = votes.get(pb)
                if va is not None and vb is not None:
                    both_voted += 1
                    if va == vb:
                        agreed += 1
                        agreed_rounds.append(r["round_number"])

            # Categorize pair
            ta = session.player_type(pa)
            tb = session.player_type(pb)
            ma = session.player_model(pa)
            mb = session.player_model(pb)

            if ta == "AI" and tb == "AI":
                if ma and mb and ma == mb:
                    category = "same_model"
                else:
                    category = "cross_model"
            elif ta == tb:  # both human (unlikely but possible)
                category = "same_type"
            else:
                category = "human_ai"

            pair_key = f"{pa}__{pb}"
            pair_agreement[pair_key] = {
                "players": [pa, pb],
                "identities": [
                    session.player_identity(pa),
                    session.player_identity(pb),
                ],
                "player_types": [ta, tb],
                "models": [ma, mb],
                "category": category,
                "rounds_both_voted": both_voted,
                "agreed_count": agreed,
                "agreement_rate": _safe_div(agreed, both_voted),
                "agreed_rounds": agreed_rounds,
            }

    # Category averages
    category_rates: dict[str, list[float]] = defaultdict(list)
    for pair_data in pair_agreement.values():
        rate = pair_data["agreement_rate"]
        cat = pair_data["category"]
        if rate is not None:
            category_rates[cat].append(rate)

    category_averages = {
        cat: _mean(rates) for cat, rates in category_rates.items()
    }

    # Identify strong voting blocs (pairs with agreement_rate >= 0.67 over 2+ rounds)
    strong_pairs = [
        v for v in pair_agreement.values()
        if (v["agreement_rate"] or 0) >= 0.67 and v["rounds_both_voted"] >= 2
    ]

    return {
        "players_directory": session.players_directory(),
        "pairs": pair_agreement,
        "category_averages": category_averages,
        "strong_voting_pairs": strong_pairs,
        "interpretation": (
            "same_model > cross_model agreement suggests implicit model alliances"
            if (
                category_averages.get("same_model", 0) or 0
            ) > (category_averages.get("cross_model", 0) or 0)
            else "No strong same-model alliance signal detected"
        ),
    }


# ---------------------------------------------------------------------------
# Layer 5: Model Fingerprinting
# ---------------------------------------------------------------------------

# Features the AI system prompt directly prescribes (lowercase openings, use
# specific slang tokens, keep messages 3-15 words, drop into emojis, vary
# openers). Because all AI agents receive the same styling instructions, these
# reflect instruction-following, not model identity — so they must NOT be used
# for fingerprinting. Reported separately as "prompt compliance" descriptors.
COMPLIANCE_FEATURES = [
    "avg_msg_length_chars",
    "avg_msg_length_words",
    "slang_rate",
    "capitalization_rate",
    "emoji_rate",
    "first_word_diversity",
]

# Features the system prompt does not directly coach. These are safer as
# model-identity signals because they emerge from the model's own distribution
# rather than from prompt compliance.
NATURAL_FEATURES = [
    "msg_count",
    "vocab_diversity",
    "question_rate",
    "exclamation_rate",
    "period_rate",
    "comma_rate",
]

# Fingerprinting uses only natural (non-prompted) features. See
# research_integrity_backlog.md for the rationale behind this split.
FINGERPRINT_FEATURES = NATURAL_FEATURES


def compute_model_fingerprints(
    session: SessionData,
    ling_features: dict,
) -> dict:
    """Layer 5: model fingerprinting from linguistic features."""

    player_features = ling_features.get("players", {})

    # Build feature matrix
    player_ids = [
        pid for pid in session.game_players
        if player_features.get(pid, {}).get("msg_count", 0) > 0
    ]

    if len(player_ids) < 2:
        return {
            "players_directory": session.players_directory(),
            "note": "Too few players with messages for fingerprinting analysis",
            "feature_matrix": {},
        }

    # Build raw feature vectors (replace None with 0)
    feature_matrix_raw: dict[str, list[float]] = {}
    for pid in player_ids:
        pfeats = player_features[pid]
        vec = [
            float(pfeats.get(f) or 0.0)
            for f in FINGERPRINT_FEATURES
        ]
        feature_matrix_raw[pid] = vec

    # Include hint features if available
    result: dict = {
        "players_directory": session.players_directory(),
        "players": player_ids,
        "feature_names": FINGERPRINT_FEATURES,
        "compliance_features_excluded": COMPLIANCE_FEATURES,
        "methodology_note": (
            "Fingerprint uses only NATURAL_FEATURES (features the AI system "
            "prompt does not directly prescribe). COMPLIANCE_FEATURES such as "
            "slang_rate, msg length, capitalization, and emoji_rate are "
            "excluded because all AI agents receive identical style "
            "instructions — those features measure instruction-following, "
            "not model identity."
        ),
        "feature_matrix": {
            pid: {k: v for k, v in zip(FINGERPRINT_FEATURES, vec)}
            for pid, vec in feature_matrix_raw.items()
        },
        "labels": {
            pid: {
                "identity": session.player_identity(pid),
                "player_type": session.player_type(pid),
                "model_name": session.player_model(pid),
            }
            for pid in player_ids
        },
    }

    # ---- Pairwise cosine similarity (numpy or pure Python fallback) ----
    result["pairwise_similarity"] = _compute_pairwise_similarity(
        player_ids, feature_matrix_raw
    )

    if not HAS_NUMPY:
        result["note"] = (
            "numpy not installed — advanced analysis (t-SNE, RandomForest) skipped. "
            "Run: pip install numpy scikit-learn"
        )
        return result

    # ---- Standardize features ----
    import numpy as np
    mat = np.array([feature_matrix_raw[pid] for pid in player_ids], dtype=float)

    if HAS_SKLEARN:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        try:
            mat_scaled = scaler.fit_transform(mat)
        except Exception:
            mat_scaled = mat
    else:
        # Manual z-score standardization
        means = mat.mean(axis=0)
        stds = mat.std(axis=0)
        stds[stds == 0] = 1.0
        mat_scaled = (mat - means) / stds

    # ---- t-SNE ----
    tsne_coords = None
    if HAS_SKLEARN and len(player_ids) >= 3:
        try:
            from sklearn.manifold import TSNE
            perplexity = min(5, len(player_ids) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            coords_2d = tsne.fit_transform(mat_scaled)
            tsne_coords = {
                pid: {
                    "x": float(coords_2d[i, 0]),
                    "y": float(coords_2d[i, 1]),
                    "identity": session.player_identity(pid),
                    "player_type": session.player_type(pid),
                    "model_name": session.player_model(pid),
                }
                for i, pid in enumerate(player_ids)
            }
        except Exception as e:
            tsne_coords = {"error": str(e)}

    result["tsne_coordinates"] = tsne_coords

    # ---- Random Forest classifier ----
    rf_results = None
    if HAS_SKLEARN and len(player_ids) >= 5:
        try:
            from sklearn.ensemble import RandomForestClassifier
            labels_arr = [session.player_type(pid) for pid in player_ids]
            unique_labels = list(set(labels_arr))
            if len(unique_labels) < 2:
                rf_results = {"note": "Only one class present, skipping classifier"}
            else:
                # Leave-one-out cross-validation
                from sklearn.model_selection import LeaveOneOut
                loo = LeaveOneOut()
                preds = []
                actuals = []
                label_to_int = {l: i for i, l in enumerate(unique_labels)}
                y = [label_to_int[l] for l in labels_arr]

                for train_idx, test_idx in loo.split(mat_scaled):
                    X_train = mat_scaled[train_idx]
                    y_train = [y[i] for i in train_idx]
                    X_test = mat_scaled[test_idx]
                    if len(set(y_train)) < 2:
                        continue
                    clf = RandomForestClassifier(n_estimators=50, random_state=42)
                    clf.fit(X_train, y_train)
                    pred = clf.predict(X_test)[0]
                    preds.append(unique_labels[pred])
                    actuals.append(labels_arr[test_idx[0]])

                loo_accuracy = _safe_div(
                    sum(p == a for p, a in zip(preds, actuals)), len(preds)
                )

                # Feature importances from a full model
                clf_full = RandomForestClassifier(n_estimators=100, random_state=42)
                clf_full.fit(mat_scaled, y)
                importances = clf_full.feature_importances_
                feature_importance = sorted(
                    zip(FINGERPRINT_FEATURES, importances.tolist()),
                    key=lambda x: x[1],
                    reverse=True,
                )

                rf_results = {
                    "loo_accuracy": loo_accuracy,
                    "n_players": len(player_ids),
                    "top_features": feature_importance[:10],
                    "predictions": list(zip(player_ids, preds, actuals)),
                }
        except Exception as e:
            rf_results = {"error": str(e), "traceback": traceback.format_exc()}
    elif HAS_SKLEARN:
        rf_results = {
            "note": f"Only {len(player_ids)} players with messages; need ≥5 for classifier"
        }

    result["random_forest"] = rf_results
    return result


def _compute_pairwise_similarity(
    player_ids: list[str],
    feature_matrix: dict[str, list[float]],
) -> dict:
    """Cosine similarity between all player pairs (numpy or pure Python)."""
    pairs = {}

    def dot(a: list, b: list) -> float:
        return sum(x * y for x, y in zip(a, b))

    def norm(v: list) -> float:
        return math.sqrt(sum(x * x for x in v))

    for i, pa in enumerate(player_ids):
        for pb in player_ids[i + 1:]:
            va = feature_matrix[pa]
            vb = feature_matrix[pb]
            na, nb = norm(va), norm(vb)
            if na == 0 or nb == 0:
                sim = None
            else:
                sim = dot(va, vb) / (na * nb)
            pairs[f"{pa}__{pb}"] = {
                "players": [pa, pb],
                "cosine_similarity": sim,
            }

    return pairs


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------

def write_summary_report(
    session: SessionData,
    game_metrics: dict,
    ling_features: dict,
    hint_analysis: dict,
    social_network: dict,
    voting_blocs: dict,
    model_fingerprints: dict,
) -> str:
    """Generate a human-readable summary_report.txt content."""

    lines = []

    def h(title: str) -> None:
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"  {title}")
        lines.append("=" * 60)

    def s(title: str) -> None:
        lines.append("")
        lines.append(f"--- {title} ---")

    def p(text: str) -> None:
        lines.append(text)

    def label(pid: Optional[str]) -> str:
        """Canonical inline label: `pid · "real_name" (TYPE · model)`."""
        if not pid:
            return "?"
        return session.player_identity(pid)["label"]

    # Header
    lines.append("THE SANDBOX — POST-GAME ANALYSIS REPORT")
    lines.append("=" * 60)

    # Session info
    s("SESSION INFO")
    p(f"Session ID : {session.session_id}")
    started = session.started_at[:19].replace("T", " ") if session.started_at else "unknown"
    p(f"Started    : {started}")
    p(f"Status     : {'Complete' if session.is_complete else f'Incomplete ({session.num_rounds_played}/{session.num_rounds_configured} rounds)'}")
    p(f"Players    : {len(session.game_players)} "
      f"({len(session.human_players)} human, {len(session.ai_players)} AI)")
    if session.ai_players:
        models = sorted(set(
            session.player_model(p) for p in session.ai_players
            if session.player_model(p)
        ))
        p(f"AI Models  : {', '.join(models)}")

    # Identity directory — definitive who-is-who, surfaced first so every
    # downstream label in this report can be cross-referenced unambiguously.
    s("IDENTITY DIRECTORY")
    p(f"  {'Player ID':<14}  {'Real Name':<20}  {'Type':<6}  Model")
    p("  " + "-" * 70)
    for pid in sorted(session.game_players):
        ident = session.player_identity(pid)
        real = ident["real_name"]
        ptype = ident["player_type"]
        model = ident["model_name"] or ""
        p(f"  {pid:<14}  {real:<20}  {ptype:<6}  {model}")

    # Game results
    h("GAME RESULTS")
    total = game_metrics["num_rounds_played"]
    overall = game_metrics["overall_detection_rate"]
    p(f"Rounds played         : {total} of {game_metrics['num_rounds_configured']} configured")
    p(f"Overall detection rate: {_pct(overall)}")

    by_type = game_metrics["detection_by_imposter_type"]
    p(f"  When human was imposter : {_pct(by_type.get('HUMAN'))}")
    p(f"  When AI was imposter    : {_pct(by_type.get('AI'))}")

    s("PER-ROUND BREAKDOWN")
    for r in game_metrics["rounds"]:
        imp_id = r["imposter_id"]
        result_str = "CAUGHT" if r["imposter_caught"] else "ESCAPED"
        p(f"  Round {r['round_number']}: [{r.get('topic','?')}] "
          f"Imposter = {label(imp_id)} → {result_str}")
        correct = r.get("correct_voters", [])
        if correct:
            p(f"    Correct voters: {', '.join(label(v) for v in correct)}")

    s("VOTE ACCURACY PER PLAYER")
    p("  (rounds where the player was imposter are excluded — they cannot self-vote)")
    vap = game_metrics["vote_accuracy_per_player"]
    for pid, stats in sorted(vap.items(), key=lambda x: (x[1].get("accuracy") or -1), reverse=True):
        acc = stats.get("accuracy")
        imp_n = stats.get("rounds_as_imposter", 0)
        suffix = f"  [imposter {imp_n}r]" if imp_n else ""
        p(f"  {label(pid)}: {stats['correct_votes']}/{stats['total_votes']} correct ({_pct(acc)}){suffix}")

    by_type_acc = game_metrics["accuracy_by_player_type"]
    p(f"\nHuman avg vote accuracy : {_pct(by_type_acc.get('HUMAN'))}")
    p(f"AI avg vote accuracy    : {_pct(by_type_acc.get('AI'))}")

    # Linguistic insights
    h("LINGUISTIC FEATURES")
    group_sum = ling_features.get("group_summary", {})
    human_ling = group_sum.get("HUMAN", {})
    ai_ling = group_sum.get("AI", {})

    s("HUMAN vs AI AVERAGES")
    keys_labels = [
        ("avg_msg_length_chars", "Avg message length (chars)"),
        ("avg_msg_length_words", "Avg message length (words)"),
        ("vocab_diversity",      "Vocabulary diversity (TTR)"),
        ("slang_rate",           "Slang rate"),
        ("question_rate",        "Question rate"),
        ("exclamation_rate",     "Exclamation rate"),
        ("capitalization_rate",  "Capitalization rate"),
        ("first_word_diversity", "First-word diversity"),
    ]
    col_w = 32
    p(f"{'Metric':<{col_w}}  {'HUMAN':>10}  {'AI':>10}")
    p("-" * (col_w + 24))
    for key, lbl in keys_labels:
        hv = human_ling.get(key)
        av = ai_ling.get(key)
        p(f"  {lbl:<{col_w - 2}}  {_fmt(hv):>10}  {_fmt(av):>10}")

    s("PER-PLAYER MESSAGES")
    players_ling = ling_features.get("players", {})
    for pid, pdata in sorted(players_ling.items(),
                             key=lambda x: x[1].get("msg_count", 0), reverse=True):
        count = pdata.get("msg_count", 0)
        vd = pdata.get("vocab_diversity")
        sr = pdata.get("slang_rate")
        p(f"  {label(pid)}: {count} msgs, "
          f"vocab_div={_fmt(vd)}, slang={_fmt(sr)}")

    # Hint insights
    s("HINT ANALYSIS")
    hint_group = hint_analysis.get("group_summary", {})
    hh = hint_group.get("HUMAN", {})
    ah = hint_group.get("AI", {})
    p(f"{'Metric':<{col_w}}  {'HUMAN':>10}  {'AI':>10}")
    p("-" * (col_w + 24))
    hint_keys = [
        ("avg_hint_length_words", "Avg hint length (words)"),
        ("avg_hint_length_chars", "Avg hint length (chars)"),
        ("slang_hint_rate",       "Slang in hints rate"),
        ("definition_like_rate",  "Definition-like hint rate"),
    ]
    for key, lbl in hint_keys:
        hv = hh.get(key)
        av = ah.get(key)
        p(f"  {lbl:<{col_w - 2}}  {_fmt(hv):>10}  {_fmt(av):>10}")

    # Social network
    h("SOCIAL NETWORK")
    dm_per_player = social_network.get("dm_network", {}).get("per_player", {})
    accusation_graph = social_network.get("accusation_graph", {})

    s("DM ACTIVITY")
    dm_sorted = sorted(
        dm_per_player.items(),
        key=lambda x: x[1].get("total_dms_sent", 0),
        reverse=True,
    )
    for pid, stats in dm_sorted:
        sent = stats.get("total_dms_sent", 0)
        recv = stats.get("total_dms_received", 0)
        init = stats.get("dm_threads_initiated", 0)
        p(f"  {label(pid)}: {sent} sent, {recv} received, {init} threads initiated")

    if accusation_graph:
        s("ACCUSATION NETWORK")
        for accuser, targets in sorted(accusation_graph.items()):
            for target, count in sorted(targets.items(), key=lambda x: x[1], reverse=True):
                p(f"  {label(accuser)} → {label(target)}: {count} accusation(s)")

    # Voting blocs
    h("VOTING BLOCS")
    cat_avgs = voting_blocs.get("category_averages", {})
    if cat_avgs:
        p("Average voting agreement by pair category:")
        for cat, avg in sorted(cat_avgs.items(), key=lambda x: x[1] or 0, reverse=True):
            p(f"  {cat:<20}: {_pct(avg)}")
        interp = voting_blocs.get("interpretation", "")
        if interp:
            p(f"\nInterpretation: {interp}")

    strong = voting_blocs.get("strong_voting_pairs", [])
    if strong:
        s("STRONG VOTING PAIRS (≥67% agreement, ≥2 rounds)")
        for pair_data in sorted(strong, key=lambda x: x["agreement_rate"] or 0, reverse=True):
            a, b = pair_data["players"]
            rate = pair_data["agreement_rate"]
            rounds_n = pair_data["rounds_both_voted"]
            cat = pair_data["category"]
            p(f"  {label(a)} ↔ {label(b)}: {_pct(rate)} agreement over {rounds_n} rounds [{cat}]")

    # Model fingerprinting
    h("MODEL FINGERPRINTS")
    if "note" in model_fingerprints:
        p(model_fingerprints["note"])
    else:
        sims = model_fingerprints.get("pairwise_similarity", {})
        if sims:
            s("PAIRWISE COSINE SIMILARITY")
            sorted_sims = sorted(
                sims.items(),
                key=lambda x: x[1].get("cosine_similarity") or 0,
                reverse=True,
            )
            for pair_key, sim_data in sorted_sims[:10]:
                a, b = sim_data["players"]
                sim = sim_data.get("cosine_similarity")
                p(f"  {label(a)} ↔ {label(b)}: cosine = {_fmt(sim)}")

        rf = model_fingerprints.get("random_forest")
        if rf and isinstance(rf, dict):
            if "note" in rf:
                p(f"\nClassifier: {rf['note']}")
            elif "loo_accuracy" in rf:
                s("RANDOM FOREST CLASSIFIER (leave-one-out)")
                p(f"LOO accuracy: {_pct(rf.get('loo_accuracy'))}")
                p(f"Top features:")
                for feat, imp in (rf.get("top_features") or [])[:5]:
                    p(f"  {feat:<30}: {imp:.4f}")

        tsne = model_fingerprints.get("tsne_coordinates")
        if tsne and "error" not in tsne:
            p(f"\nt-SNE 2D coordinates computed for {len(tsne)} players (see model_fingerprints.json)")

    # Pointer to richer outputs
    h("MORE IN THE OTHER OUTPUTS")
    p("  This text report covers the original layered metrics. The richer")
    p("  research outputs sit alongside it in the same analysis directory:")
    p("    conversational_dynamics.json   — latency / burstiness / replies")
    p("    deep_stylometry.json           — function-words, hedges, self-refs")
    p("    topic_adherence.json           — hint relevance to topic + grid")
    p("    imposter_signals.json          — per-round imposter behavior delta")
    p("    suspicion_received.json        — votes / mentions / accusations in")
    p("    per_round_evolution.json       — feature time series per player")
    p("    hint_similarity.json           — pairwise hint Jaccard per round")
    p("    classification_breakdown.json  — meta-game per-target detail")
    p("    message_clustering.json        — TF-IDF clusters + similarity")
    p("    information_theoretic.json     — entropy + vocabulary growth")
    p("    communication_graph.json       — D3-ready nodes + multi-typed edges")
    p("    dashboard.html                 — open in a browser to visualize")

    # Warnings
    if session.warnings:
        h("WARNINGS / DATA ISSUES")
        for w in session.warnings:
            p(f"  ! {w}")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Generated by evaluation.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    return "\n".join(lines)


def _pct(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _fmt(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val:.3f}"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

# Set by main() before run_evaluation reads them. Module-level rather than
# function args so a long signature doesn't propagate through every helper.
_SKIP_DASHBOARD = False
_OPEN_DASHBOARD = False


def run_evaluation(session_dir: str, output_dir: Optional[str] = None) -> None:
    session_path = Path(session_dir)
    if not session_path.exists():
        print(f"[evaluation] ERROR: Session directory not found: {session_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[evaluation] Loading session data from {session_dir}...")

    session = SessionData(str(session_path))

    n_human = len(session.human_players)
    n_ai = len(session.ai_players)
    print(
        f"[evaluation] Loaded: {len(session.game_players)} players "
        f"({n_human} human, {n_ai} AI), "
        f"{session.num_rounds_played} round(s) played"
    )

    if session.warnings:
        for w in session.warnings:
            print(f"[evaluation] WARNING: {w}")

    # Output directory
    out_dir = Path(output_dir) if output_dir else (session_path / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    # Layer 1
    print("[evaluation] Computing game metrics (Layer 1)...")
    try:
        results["game_metrics"] = compute_game_metrics(session)
        _write_json(out_dir / "game_metrics.json", results["game_metrics"])
        print(f"[evaluation] Wrote {out_dir / 'game_metrics.json'}")
    except Exception as e:
        results["game_metrics"] = {"error": str(e)}
        _write_json(out_dir / "game_metrics.json", results["game_metrics"])
        print(f"[evaluation] ERROR in Layer 1: {e}")

    # Layer 2 — linguistic features
    print("[evaluation] Computing linguistic features (Layer 2)...")
    try:
        results["linguistic_features"] = compute_linguistic_features(session)
        _write_json(out_dir / "linguistic_features.json", results["linguistic_features"])
        print(f"[evaluation] Wrote {out_dir / 'linguistic_features.json'}")
    except Exception as e:
        results["linguistic_features"] = {"error": str(e), "players": {}}
        _write_json(out_dir / "linguistic_features.json", results["linguistic_features"])
        print(f"[evaluation] ERROR in Layer 2 (linguistic): {e}")

    # Layer 2 — hint analysis
    print("[evaluation] Computing hint analysis (Layer 2)...")
    try:
        results["hint_analysis"] = compute_hint_analysis(session)
        _write_json(out_dir / "hint_analysis.json", results["hint_analysis"])
        print(f"[evaluation] Wrote {out_dir / 'hint_analysis.json'}")
    except Exception as e:
        results["hint_analysis"] = {"error": str(e)}
        _write_json(out_dir / "hint_analysis.json", results["hint_analysis"])
        print(f"[evaluation] ERROR in Layer 2 (hints): {e}")

    # Layer 3 — social network
    print("[evaluation] Computing social network (Layer 3)...")
    try:
        results["social_network"] = compute_social_network(session)
        _write_json(out_dir / "social_network.json", results["social_network"])
        print(f"[evaluation] Wrote {out_dir / 'social_network.json'}")
    except Exception as e:
        results["social_network"] = {"error": str(e)}
        _write_json(out_dir / "social_network.json", results["social_network"])
        print(f"[evaluation] ERROR in Layer 3 (social network): {e}")

    # Layer 3 — voting blocs
    print("[evaluation] Computing voting blocs (Layer 3)...")
    try:
        results["voting_blocs"] = compute_voting_blocs(session)
        _write_json(out_dir / "voting_blocs.json", results["voting_blocs"])
        print(f"[evaluation] Wrote {out_dir / 'voting_blocs.json'}")
    except Exception as e:
        results["voting_blocs"] = {"error": str(e)}
        _write_json(out_dir / "voting_blocs.json", results["voting_blocs"])
        print(f"[evaluation] ERROR in Layer 3 (voting blocs): {e}")

    # ---- Extended research metrics (Phase 2 additions) ----
    try:
        import evaluation_metrics as _em
        extra_specs = [
            ("conversational_dynamics",   _em.compute_conversational_dynamics),
            ("deep_stylometry",           _em.compute_deep_stylometry),
            ("topic_adherence",           _em.compute_topic_adherence),
            ("imposter_signals",          _em.compute_imposter_signals),
            ("suspicion_received",        _em.compute_suspicion_received),
            ("per_round_evolution",       _em.compute_per_round_evolution),
            ("hint_similarity",           _em.compute_hint_similarity),
            ("classification_breakdown",  _em.compute_classification_breakdown),
            ("message_clustering",        _em.compute_message_clustering),
            ("information_theoretic",     _em.compute_information_theoretic),
            ("communication_graph",       _em.compute_communication_graph),
            ("survey_responses",          _em.compute_survey_responses),
        ]
        for key, fn in extra_specs:
            print(f"[evaluation] Computing {key}...")
            try:
                results[key] = fn(session)
                _write_json(out_dir / f"{key}.json", results[key])
                print(f"[evaluation] Wrote {out_dir / (key + '.json')}")
            except Exception as e:
                results[key] = {"error": str(e), "traceback": traceback.format_exc()}
                _write_json(out_dir / f"{key}.json", results[key])
                print(f"[evaluation] ERROR in {key}: {e}")
    except ImportError as e:
        print(f"[evaluation] WARNING: evaluation_metrics module not loaded: {e}")

    # Layer 5 — model fingerprints
    print("[evaluation] Computing model fingerprints (Layer 5)...")
    if not HAS_NUMPY:
        print("[evaluation] numpy not available — advanced fingerprinting skipped")
    if not HAS_SKLEARN:
        print("[evaluation] scikit-learn not available — classifier/t-SNE skipped")
    try:
        results["model_fingerprints"] = compute_model_fingerprints(
            session, results.get("linguistic_features", {})
        )
        _write_json(out_dir / "model_fingerprints.json", results["model_fingerprints"])
        print(f"[evaluation] Wrote {out_dir / 'model_fingerprints.json'}")
    except Exception as e:
        results["model_fingerprints"] = {"error": str(e)}
        _write_json(out_dir / "model_fingerprints.json", results["model_fingerprints"])
        print(f"[evaluation] ERROR in Layer 5: {e}")

    # Summary report
    print("[evaluation] Writing summary report...")
    try:
        report_text = write_summary_report(
            session,
            results.get("game_metrics", {}),
            results.get("linguistic_features", {}),
            results.get("hint_analysis", {}),
            results.get("social_network", {}),
            results.get("voting_blocs", {}),
            results.get("model_fingerprints", {}),
        )
        report_path = out_dir / "summary_report.txt"
        report_path.write_text(report_text, encoding="utf-8")
        print(f"[evaluation] Wrote {report_path}")
    except Exception as e:
        print(f"[evaluation] ERROR writing summary report: {e}")

    # ---- HTML dashboard ----
    if not _SKIP_DASHBOARD:
        print("[evaluation] Rendering HTML dashboard...")
        try:
            import evaluation_dashboard as _ed
            dash_path = out_dir / "dashboard.html"
            _ed.write_dashboard(session, results, dash_path)
            print(f"[evaluation] Wrote {dash_path}")
            if _OPEN_DASHBOARD:
                import webbrowser
                webbrowser.open(dash_path.resolve().as_uri())
        except Exception as e:
            print(f"[evaluation] ERROR rendering dashboard: {e}")
            traceback.print_exc()

    # Done
    files_written = len(list(out_dir.glob("*")))
    print(f"\n[evaluation] Done! {files_written} file(s) in {out_dir}")
    print(f"[evaluation] Session: {session.session_id} — "
          f"{session.num_rounds_played}/{session.num_rounds_configured} rounds")
    if session.is_complete:
        print("[evaluation] Game was complete.")
    else:
        print("[evaluation] Game was incomplete (session ended early).")
    if not _SKIP_DASHBOARD:
        dash = (out_dir / "dashboard.html").resolve()
        if dash.exists():
            print(f"[evaluation] Open the dashboard: {dash.as_uri()}")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-game analysis for The Sandbox social deduction game.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation.py game_sessions/session_20260305_001245
  python evaluation.py game_sessions/session_20260306_145000
  python evaluation.py game_sessions/session_20260305_190148 --output /tmp/my_analysis
        """,
    )
    parser.add_argument(
        "session_dir",
        help="Path to the game session directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Custom output directory (default: {session_dir}/analysis/)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip generating dashboard.html",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open dashboard.html in the default browser when done",
    )
    args = parser.parse_args()
    global _SKIP_DASHBOARD, _OPEN_DASHBOARD
    _SKIP_DASHBOARD = args.no_dashboard
    _OPEN_DASHBOARD = args.open
    run_evaluation(args.session_dir, args.output)


if __name__ == "__main__":
    main()
