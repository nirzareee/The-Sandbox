"""
evaluation_metrics.py — Extended research metrics for The Sandbox.

These functions are called from evaluation.py's run_evaluation() orchestrator
to produce the additional analysis JSONs requested for the research project:

    conversational_dynamics.json   — latency, burstiness, reply patterns
    deep_stylometry.json           — function-word vectors, hedge words, etc.
    topic_adherence.json           — hint relevance to topic + grid items
    imposter_signals.json          — per-round imposter behavior signatures
    suspicion_received.json        — votes/accusations/mentions targeting each player
    per_round_evolution.json       — per-player feature time series across rounds
    hint_similarity.json           — pairwise hint-text similarity per round
    classification_breakdown.json  — meta-game per-target classification matrix
    message_clustering.json        — TF-IDF clustering of player message corpora
    information_theoretic.json     — entropy + vocabulary growth
    communication_graph.json       — nodes + multi-typed directed edges

Every output JSON includes a `players_directory` block (identity for each
player) so consumers never need to cross-reference a separate manifest.

All functions degrade gracefully when optional deps (numpy, scipy, sklearn)
are missing — pure-Python fallbacks always exist.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Optional heavy deps
# ---------------------------------------------------------------------------

try:
    import numpy as _np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy.cluster.hierarchy import linkage as _linkage, fcluster as _fcluster
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Lexicons (kept small + explicit so behavior is reproducible)
# ---------------------------------------------------------------------------

# Top English function words. Used as a stylometric fingerprint vector since
# function-word distributions are notoriously stable per author/model and
# resist topical confounds. Source: standard stylometric literature.
FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "it",
    "for", "with", "as", "on", "by", "this", "but", "be", "are", "have",
    "from", "or", "not", "an", "they", "we", "you", "i", "he", "she",
]

# Hedge / uncertainty markers. High rate = imposter or cautious player.
HEDGE_WORDS = {
    "maybe", "perhaps", "probably", "kinda", "sorta", "sort", "kind",
    "guess", "think", "thinking", "thought", "feels", "seems", "seem",
    "might", "could", "possibly", "supposedly", "apparently", "ish",
    "lowkey", "highkey", "tbh", "honestly", "idk", "unsure", "dunno",
    "prolly", "prob",
}

# First-person singular pronouns.
SELF_REFERENCES = {"i", "me", "my", "mine", "myself", "im", "ive", "ill", "id"}

# Common contractions (presence often correlates with informal/human style
# AFTER the AI has been prompted to use slang — useful as a secondary signal).
CONTRACTION_RE = re.compile(r"\b\w+'\w+\b", re.IGNORECASE)

# Vague generic-language tokens — common imposter crutches when they don't
# know the secret word.
VAGUE_TOKENS = {
    "thing", "stuff", "something", "anything", "everything", "nothing",
    "kind", "type", "sort", "general", "various", "common", "popular",
    "famous", "known", "regular", "normal", "basic", "everyday",
}

# Words that mark a player calling out / suggesting a vote on someone.
ACCUSATION_KEYWORDS = {
    "sus", "suspicious", "imposter", "impostor", "vote", "voting",
    "weird", "vague", "fake", "lying", "lie", "cap", "bluffing", "bluff",
}


# ---------------------------------------------------------------------------
# Generic helpers (intentionally duplicated from evaluation.py to keep this
# module standalone — small, stable functions, low maintenance cost)
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    return (numerator / denominator) if denominator > 0 else None


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _stdev(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", (text or "").lower())


def _sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text or "")
    return [p.strip() for p in parts if p.strip()]


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # Python 3.11+ handles "+00:00" suffix natively; replace Z just in case.
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _build_round_pn_map(session) -> dict[int, dict[int, str]]:
    """Build per-round positional name map: round_number -> {1: pid, 2: pid, ...}.

    Players reference each other in chat as `p1`, `p2`, `p3`, etc. The
    mapping is per-round, derived from each round's `hint_order` field
    (the order players submit hints in). Without this map, mention and
    accusation detection misses every shorthand reference.
    """
    out: dict[int, dict[int, str]] = {}
    for ev in session.round_start_events:
        rn = ev.get("round_number")
        order = ev.get("hint_order") or []
        if rn:
            out[rn] = {i + 1: pid for i, pid in enumerate(order)}
    return out


_PN_RE = re.compile(r"\bp(\d+)\b", re.IGNORECASE)


def _resolve_mentions(
    session, content: str, round_number: Optional[int],
    round_pn_map: dict[int, dict[int, str]],
    static_patterns: dict[str, "re.Pattern"],
) -> set[str]:
    """Return the set of player_ids referenced in `content`.

    Combines two strategies:
    1. Static patterns: matches player_id, real_name, display_name as words.
    2. Per-round pN resolution: maps `p1`, `p2`, etc. via this round's hint_order.
    """
    targets: set[str] = set()
    # Static name/id patterns
    for pid, pat in static_patterns.items():
        if pat.search(content):
            targets.add(pid)
    # pN shorthand
    if round_number and round_number in round_pn_map:
        pn_map = round_pn_map[round_number]
        for m in _PN_RE.finditer(content):
            pos = int(m.group(1))
            target = pn_map.get(pos)
            if target and target in session.game_players:
                targets.add(target)
    return targets


def _build_static_name_patterns(session) -> dict[str, "re.Pattern"]:
    """Compile per-player word patterns matching player_id and real_name."""
    out: dict[str, "re.Pattern"] = {}
    for pid in session.game_players:
        ident = session.player_identity(pid)
        parts = {pid, ident["real_name"]}
        # display_name is "Player" in current logs — useless as a key, skip.
        parts = {p for p in parts if p and len(p) >= 2}
        escaped = sorted({re.escape(p) for p in parts}, key=len, reverse=True)
        if not escaped:
            continue
        out[pid] = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
    return out


def _entropy(counts: Counter) -> Optional[float]:
    """Shannon entropy in bits over a frequency distribution."""
    total = sum(counts.values())
    if total == 0:
        return None
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# 1. Conversational dynamics
# ---------------------------------------------------------------------------

def compute_conversational_dynamics(session) -> dict:
    """Per-player response latency, burstiness, who-replies-to-whom.

    Latency is measured against the previous message in the same channel
    (group chat OR a specific DM thread). Burstiness is the coefficient of
    variation of inter-message gaps (>1 means bursty, <1 means metronome).
    """
    pdir = session.players_directory()

    # ---- Per-player latency in group chat ----
    group_msgs = sorted(
        [m for m in session.group_chat_messages if m.get("ts")],
        key=lambda m: m.get("ts", ""),
    )

    per_player_latencies: dict[str, list[float]] = defaultdict(list)
    per_player_intervals: dict[str, list[float]] = defaultdict(list)
    last_msg_ts: Optional[datetime] = None
    last_per_player_ts: dict[str, datetime] = {}

    for msg in group_msgs:
        pid = msg.get("player_id")
        ts = _parse_ts(msg.get("ts", ""))
        if not pid or not ts or pid not in session.game_players:
            continue
        if last_msg_ts is not None:
            delta = (ts - last_msg_ts).total_seconds()
            if 0 <= delta <= 600:  # ignore anomalies (cross-phase gaps)
                per_player_latencies[pid].append(delta)
        if pid in last_per_player_ts:
            interval = (ts - last_per_player_ts[pid]).total_seconds()
            if 0 <= interval <= 600:
                per_player_intervals[pid].append(interval)
        last_per_player_ts[pid] = ts
        last_msg_ts = ts

    # ---- Per-thread DM latency ----
    dm_threads: dict[str, list[dict]] = defaultdict(list)
    for m in session.dm_messages:
        s = m.get("sender_id")
        r = m.get("recipient_id")
        if not s or not r:
            continue
        key = "|".join(sorted([s, r]))
        dm_threads[key].append(m)

    per_player_dm_replies: dict[str, list[float]] = defaultdict(list)
    for key, msgs in dm_threads.items():
        msgs_sorted = sorted(msgs, key=lambda m: m.get("ts", ""))
        for i in range(1, len(msgs_sorted)):
            prev = msgs_sorted[i - 1]
            cur = msgs_sorted[i]
            if prev.get("sender_id") == cur.get("sender_id"):
                continue  # same person continuing — not a "reply"
            prev_ts = _parse_ts(prev.get("ts", ""))
            cur_ts = _parse_ts(cur.get("ts", ""))
            if not prev_ts or not cur_ts:
                continue
            delta = (cur_ts - prev_ts).total_seconds()
            if 0 <= delta <= 600:
                per_player_dm_replies[cur.get("sender_id", "")].append(delta)

    # ---- Reply-to-whom matrix (group chat) ----
    # For every group message, the "previous message" sender is a candidate
    # for who this player is replying to. This is approximate but useful.
    reply_to: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prev_sender: Optional[str] = None
    for msg in group_msgs:
        pid = msg.get("player_id")
        if not pid or pid not in session.game_players:
            continue
        if prev_sender and prev_sender != pid:
            reply_to[pid][prev_sender] += 1
        prev_sender = pid

    # ---- Aggregate per player ----
    per_player: dict[str, dict] = {}
    for pid in session.game_players:
        intervals = per_player_intervals[pid]
        gap_mean = _mean(intervals)
        gap_std = _stdev(intervals)
        burstiness = _safe_div(gap_std, gap_mean) if gap_mean else None

        latencies = per_player_latencies[pid]
        dm_reply_latencies = per_player_dm_replies[pid]

        per_player[pid] = {
            "identity": session.player_identity(pid),
            "group_msg_count": sum(
                1 for m in group_msgs if m.get("player_id") == pid
            ),
            "median_latency_after_others_sec": _median(latencies),
            "mean_latency_after_others_sec": _mean(latencies),
            "median_self_interval_sec": _median(intervals),
            "burstiness": burstiness,
            "median_dm_reply_latency_sec": _median(dm_reply_latencies),
            "mean_dm_reply_latency_sec": _mean(dm_reply_latencies),
            "dm_replies_made": len(dm_reply_latencies),
            "replies_to_others": dict(reply_to.get(pid, {})),
        }

    # ---- Group-level summaries ----
    def _agg(key: str, ptype: str) -> Optional[float]:
        vals = [
            v[key] for pid, v in per_player.items()
            if session.player_type(pid) == ptype and v.get(key) is not None
        ]
        return _mean(vals)

    group_summary = {}
    for ptype in ("HUMAN", "AI"):
        group_summary[ptype] = {
            "median_latency_after_others_sec": _agg("median_latency_after_others_sec", ptype),
            "burstiness": _agg("burstiness", ptype),
            "median_dm_reply_latency_sec": _agg("median_dm_reply_latency_sec", ptype),
        }

    return {
        "players_directory": pdir,
        "per_player": per_player,
        "group_summary": group_summary,
        "notes": (
            "Latency = seconds between this player's message and the previous "
            "message in the same channel (group or DM thread). Bursty values "
            "(>1.0) indicate clusters of fast messages followed by silence."
        ),
    }


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


# ---------------------------------------------------------------------------
# 2. Deep stylometry
# ---------------------------------------------------------------------------

def compute_deep_stylometry(session) -> dict:
    """Function-word vector + hedge / self-ref / sentence statistics per player."""
    pdir = session.players_directory()

    # Aggregate text per player (group + DM)
    per_player_text: dict[str, list[str]] = defaultdict(list)
    for m in session.group_chat_messages:
        pid = m.get("player_id")
        if pid in session.game_players:
            per_player_text[pid].append(m.get("content", ""))
    for m in session.dm_messages:
        pid = m.get("sender_id")
        if pid in session.game_players:
            per_player_text[pid].append(m.get("content", ""))

    per_player: dict[str, dict] = {}

    for pid in session.game_players:
        msgs = per_player_text.get(pid, [])
        all_text = " ".join(msgs)
        toks = _tokens(all_text)
        total = len(toks)

        # Function-word frequency vector (normalized).
        fw_counts = Counter(t for t in toks if t in FUNCTION_WORDS)
        fw_vector = {
            w: _safe_div(fw_counts.get(w, 0), total) or 0.0
            for w in FUNCTION_WORDS
        }

        # Hedge words
        hedge_count = sum(1 for t in toks if t in HEDGE_WORDS)
        # Self-references
        self_count = sum(1 for t in toks if t in SELF_REFERENCES)
        # Vague tokens
        vague_count = sum(1 for t in toks if t in VAGUE_TOKENS)
        # Contractions
        contraction_count = sum(
            len(CONTRACTION_RE.findall(m or "")) for m in msgs
        )

        # Sentence-level stats
        sent_lengths = []
        for m in msgs:
            for s in _sentences(m):
                sent_lengths.append(len(s.split()))
        n_sentences = len(sent_lengths)

        per_player[pid] = {
            "identity": session.player_identity(pid),
            "total_tokens": total,
            "total_messages": len(msgs),
            "function_word_vector": fw_vector,
            "hedge_word_rate": _safe_div(hedge_count, total),
            "hedge_word_count": hedge_count,
            "self_reference_rate": _safe_div(self_count, total),
            "self_reference_count": self_count,
            "vague_token_rate": _safe_div(vague_count, total),
            "vague_token_count": vague_count,
            "contraction_rate_per_msg": _safe_div(contraction_count, len(msgs)),
            "contraction_count": contraction_count,
            "sentences_per_msg": _safe_div(n_sentences, len(msgs)),
            "avg_sentence_length_words": _mean(sent_lengths),
            "std_sentence_length_words": _stdev(sent_lengths),
        }

    # Group summaries (mean of per-player values, by type and by model)
    def _summarize(filter_fn) -> dict:
        relevant = [v for pid, v in per_player.items() if filter_fn(pid)]
        if not relevant:
            return {}
        scalar_keys = [
            "hedge_word_rate", "self_reference_rate", "vague_token_rate",
            "contraction_rate_per_msg", "sentences_per_msg",
            "avg_sentence_length_words", "std_sentence_length_words",
        ]
        return {
            k: _mean([v[k] for v in relevant if v.get(k) is not None])
            for k in scalar_keys
        }

    group_summary = {
        "HUMAN": _summarize(lambda pid: session.player_type(pid) == "HUMAN"),
        "AI": _summarize(lambda pid: session.player_type(pid) == "AI"),
    }

    per_model_summary: dict[str, dict] = {}
    models = {session.player_model(pid) for pid in session.ai_players}
    for model in models:
        if not model:
            continue
        per_model_summary[model] = _summarize(
            lambda pid, m=model: session.player_model(pid) == m
        )

    return {
        "players_directory": pdir,
        "per_player": per_player,
        "group_summary": group_summary,
        "per_model_summary": per_model_summary,
        "function_word_inventory": FUNCTION_WORDS,
    }


# ---------------------------------------------------------------------------
# 3. Topic adherence
# ---------------------------------------------------------------------------

def compute_topic_adherence(session) -> dict:
    """How well each hint matches its round's topic + grid items.

    Score = fraction of hint tokens that appear in the topic name OR any
    grid-item name. Insiders should usually score higher than imposters.
    """
    pdir = session.players_directory()
    rounds_meta = {
        ev.get("round_number"): ev for ev in session.round_start_events
    }

    # Index word_bank by topic so we can look up grid items by name. The
    # round_started event only carries topic + secret_word, not the grid.
    word_bank_by_topic: dict[str, list] = {}
    for entry in session.word_bank:
        t = (entry.get("topic") or "").lower()
        if t:
            word_bank_by_topic[t] = entry.get("grid_items") or []

    per_hint: list[dict] = []
    per_player_scores: dict[str, list[float]] = defaultdict(list)

    for ev in session.hint_events:
        pid = ev.get("player_id")
        rn = ev.get("round_number")
        text = ev.get("hint", "")
        if not pid or pid not in session.game_players or not rn:
            continue
        rmeta = rounds_meta.get(rn) or {}
        topic = (rmeta.get("topic") or "").lower()
        # Pull grid from event (rare) OR look up via word_bank by topic.
        grid = (
            rmeta.get("image_grid")
            or rmeta.get("grid_items")
            or word_bank_by_topic.get(topic, [])
        )
        grid_words = set()
        for item in grid:
            name = item if isinstance(item, str) else (
                item.get("label") or item.get("name") or ""
            )
            for t in _tokens(name):
                grid_words.add(t)
        for t in _tokens(topic):
            grid_words.add(t)
        # Add the secret word too — insiders typically allude to it directly.
        for t in _tokens(rmeta.get("secret_word") or ""):
            grid_words.add(t)

        toks = _tokens(text)
        if not toks:
            score = None
        else:
            hits = sum(1 for t in toks if t in grid_words)
            score = hits / len(toks)
        is_imposter = (rmeta.get("imposter_id") == pid)
        per_hint.append({
            "round_number": rn,
            "topic": rmeta.get("topic"),
            "secret_word": rmeta.get("secret_word"),
            "player_id": pid,
            "identity": session.player_identity(pid),
            "is_imposter": is_imposter,
            "hint": text,
            "adherence_score": score,
        })
        if score is not None:
            per_player_scores[pid].append(score)

    per_player_summary = {
        pid: {
            "identity": session.player_identity(pid),
            "mean_adherence": _mean(per_player_scores[pid]),
            "median_adherence": _median(per_player_scores[pid]),
            "n_hints": len(per_player_scores[pid]),
        }
        for pid in session.game_players
    }

    # Imposter vs insider: do imposters' adherence scores look different?
    imposter_scores = [h["adherence_score"] for h in per_hint
                       if h["is_imposter"] and h["adherence_score"] is not None]
    insider_scores = [h["adherence_score"] for h in per_hint
                      if not h["is_imposter"] and h["adherence_score"] is not None]

    return {
        "players_directory": pdir,
        "per_hint": per_hint,
        "per_player_summary": per_player_summary,
        "imposter_vs_insider": {
            "imposter_mean_adherence": _mean(imposter_scores),
            "insider_mean_adherence": _mean(insider_scores),
            "n_imposter_hints": len(imposter_scores),
            "n_insider_hints": len(insider_scores),
        },
        "notes": (
            "Adherence = fraction of hint tokens that literally match a "
            "topic / grid-item / secret-word token. Players intentionally "
            "give oblique hints ('hoops' for Basketball, 'flips' for "
            "Gymnastics), so scores are usually low even for valid hints. "
            "Use this as a relative comparison signal, not an absolute one."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Imposter signals
# ---------------------------------------------------------------------------

def compute_imposter_signals(session) -> dict:
    """Per-imposter behavioral signature each round, vs same-round insiders.

    Imposters tend to: hedge more, use vaguer tokens, send fewer messages
    early in discussion, ask more questions to fish for info.
    """
    pdir = session.players_directory()
    per_round: list[dict] = []

    for r_ev in session.round_results_events:
        rn = r_ev.get("round_number")
        imp = r_ev.get("imposter_id")
        if not rn or not imp:
            continue

        # Collect text per player for this round (group + DM)
        per_player_text: dict[str, list[str]] = defaultdict(list)
        for m in session.group_chat_messages:
            if m.get("round_number") == rn and m.get("player_id") in session.game_players:
                per_player_text[m["player_id"]].append(m.get("content", ""))
        for m in session.dm_messages:
            if m.get("round_number") == rn and m.get("sender_id") in session.game_players:
                per_player_text[m["sender_id"]].append(m.get("content", ""))

        def _stat(text_list: list[str]) -> dict:
            all_text = " ".join(text_list)
            toks = _tokens(all_text)
            total = len(toks) or 1
            return {
                "msg_count": len(text_list),
                "total_tokens": total,
                "hedge_rate": _safe_div(
                    sum(1 for t in toks if t in HEDGE_WORDS), total
                ),
                "vague_rate": _safe_div(
                    sum(1 for t in toks if t in VAGUE_TOKENS), total
                ),
                "question_rate": _safe_div(
                    sum(1 for m in text_list if "?" in (m or "")), len(text_list)
                ),
                "avg_msg_words": _mean([len(_tokens(m)) for m in text_list]),
            }

        imp_stats = _stat(per_player_text.get(imp, []))
        insider_aggregate = []
        for pid in session.game_players:
            if pid == imp:
                continue
            insider_aggregate.append(_stat(per_player_text.get(pid, [])))

        # Aggregate insiders
        insider_means = {}
        for k in ("hedge_rate", "vague_rate", "question_rate", "avg_msg_words", "msg_count"):
            vals = [d[k] for d in insider_aggregate if d.get(k) is not None]
            insider_means[k] = _mean(vals)

        per_round.append({
            "round_number": rn,
            "imposter_identity": session.player_identity(imp),
            "imposter_caught": r_ev.get("imposter_caught"),
            "imposter_stats": imp_stats,
            "insider_means": insider_means,
            "deltas": {
                k: (
                    (imp_stats[k] - insider_means[k])
                    if imp_stats.get(k) is not None and insider_means.get(k) is not None
                    else None
                )
                for k in ("hedge_rate", "vague_rate", "question_rate", "avg_msg_words", "msg_count")
            },
        })

    return {
        "players_directory": pdir,
        "per_round": per_round,
        "notes": (
            "deltas = imposter_stat - mean(insiders_in_same_round). Positive "
            "hedge/vague/question deltas suggest the imposter was visibly "
            "fishing or stalling. Used as a research signal, not a verdict."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Suspicion received
# ---------------------------------------------------------------------------

def compute_suspicion_received(session) -> dict:
    """For each player: how many votes / mentions / accusations targeted them."""
    pdir = session.players_directory()

    # Vote-target counts
    vote_targets: dict[str, int] = defaultdict(int)
    for r_ev in session.round_results_events:
        for voter, target in (r_ev.get("votes") or {}).items():
            if target in session.game_players:
                vote_targets[target] += 1

    static_patterns = _build_static_name_patterns(session)
    pn_map = _build_round_pn_map(session)

    mention_counts: dict[str, int] = defaultdict(int)
    accusation_received: dict[str, int] = defaultdict(int)

    for m in session.group_chat_messages:
        sender = m.get("player_id")
        content = m.get("content") or ""
        rn = m.get("round_number")
        words_lower = set(re.findall(r"\b\w+\b", content.lower()))
        is_accusation = bool(words_lower & ACCUSATION_KEYWORDS)
        targets = _resolve_mentions(session, content, rn, pn_map, static_patterns)
        for target_pid in targets:
            if target_pid == sender:
                continue
            mention_counts[target_pid] += 1
            if is_accusation:
                accusation_received[target_pid] += 1

    per_player = {}
    for pid in session.game_players:
        per_player[pid] = {
            "identity": session.player_identity(pid),
            "votes_received_total": vote_targets.get(pid, 0),
            "mentions_in_group_chat": mention_counts.get(pid, 0),
            "accusations_received": accusation_received.get(pid, 0),
        }

    return {
        "players_directory": pdir,
        "per_player": per_player,
    }


# ---------------------------------------------------------------------------
# 6. Per-round evolution (time series)
# ---------------------------------------------------------------------------

def compute_per_round_evolution(session) -> dict:
    """Per-player feature vectors per round so we can see learning/adaptation."""
    pdir = session.players_directory()

    rounds = sorted({
        m.get("round_number") for m in session.group_chat_messages
        if m.get("round_number")
    })

    per_player_series: dict[str, list[dict]] = {pid: [] for pid in session.game_players}

    for rn in rounds:
        # Per-player text this round
        text_by_pid: dict[str, list[str]] = defaultdict(list)
        for m in session.group_chat_messages:
            if m.get("round_number") == rn and m.get("player_id") in session.game_players:
                text_by_pid[m["player_id"]].append(m.get("content", ""))
        for m in session.dm_messages:
            if m.get("round_number") == rn and m.get("sender_id") in session.game_players:
                text_by_pid[m["sender_id"]].append(m.get("content", ""))

        for pid in session.game_players:
            msgs = text_by_pid.get(pid, [])
            toks = []
            for m in msgs:
                toks.extend(_tokens(m))
            total = len(toks) or 1
            per_player_series[pid].append({
                "round_number": rn,
                "msg_count": len(msgs),
                "total_tokens": len(toks),
                "avg_msg_words": _mean([len(_tokens(m)) for m in msgs]),
                "hedge_rate": _safe_div(
                    sum(1 for t in toks if t in HEDGE_WORDS), total
                ),
                "self_ref_rate": _safe_div(
                    sum(1 for t in toks if t in SELF_REFERENCES), total
                ),
                "vague_rate": _safe_div(
                    sum(1 for t in toks if t in VAGUE_TOKENS), total
                ),
                "question_rate": _safe_div(
                    sum(1 for m in msgs if "?" in (m or "")), len(msgs)
                ),
            })

    return {
        "players_directory": pdir,
        "rounds": rounds,
        "per_player": {
            pid: {
                "identity": session.player_identity(pid),
                "series": per_player_series[pid],
            }
            for pid in session.game_players
        },
    }


# ---------------------------------------------------------------------------
# 7. Hint similarity
# ---------------------------------------------------------------------------

def compute_hint_similarity(session) -> dict:
    """Pairwise Jaccard similarity between hints in the same round."""
    pdir = session.players_directory()

    hints_by_round: dict[int, list[dict]] = defaultdict(list)
    for ev in session.hint_events:
        rn = ev.get("round_number")
        pid = ev.get("player_id")
        text = ev.get("hint", "")
        if rn and pid in session.game_players:
            hints_by_round[rn].append({
                "player_id": pid,
                "hint": text,
                "tokens": set(_tokens(text)),
            })

    per_round: dict[int, dict] = {}
    for rn, hints in hints_by_round.items():
        pairs = []
        for i, ha in enumerate(hints):
            for hb in hints[i + 1:]:
                inter = ha["tokens"] & hb["tokens"]
                union = ha["tokens"] | hb["tokens"]
                jaccard = (len(inter) / len(union)) if union else None
                pairs.append({
                    "player_a": ha["player_id"],
                    "player_b": hb["player_id"],
                    "identity_a": session.player_identity(ha["player_id"]),
                    "identity_b": session.player_identity(hb["player_id"]),
                    "hint_a": ha["hint"],
                    "hint_b": hb["hint"],
                    "jaccard_similarity": jaccard,
                    "shared_tokens": sorted(inter),
                })
        per_round[rn] = {"pairs": pairs}

    return {
        "players_directory": pdir,
        "per_round": per_round,
    }


# ---------------------------------------------------------------------------
# 8. Classification breakdown (meta-game)
# ---------------------------------------------------------------------------

def compute_classification_breakdown(session) -> dict:
    """Per-target: who classified them as what, with confidence + miss type."""
    pdir = session.players_directory()
    classifications = session.classifications or {}
    ground_truth = session.ground_truth or {}

    # If ground_truth missing, derive from manifest
    if not ground_truth:
        ground_truth = {
            pid: ("AI" if info["player_type"] == "AI" else "HUMAN")
            for pid, info in session.game_players.items()
        }

    per_target: dict[str, dict] = {}
    for target_pid in session.game_players:
        actual = (ground_truth.get(target_pid) or "").upper()
        votes_in: list[dict] = []
        for voter_pid, voter_classifications in classifications.items():
            if voter_pid == target_pid:
                continue
            entry = (voter_classifications or {}).get(target_pid)
            if not entry:
                continue
            guess = (entry.get("guess") or "").upper()
            correct = (guess == actual) if actual else None
            # Miss type: from the voter's perspective
            miss_type = None
            if actual and guess:
                if actual == "AI" and guess == "HUMAN":
                    miss_type = "false_negative"  # missed an AI
                elif actual == "HUMAN" and guess == "AI":
                    miss_type = "false_positive"  # accused a human
            votes_in.append({
                "voter_id": voter_pid,
                "voter_identity": session.player_identity(voter_pid),
                "guess": guess.lower() if guess else None,
                "confidence": entry.get("confidence"),
                "reasoning": entry.get("reasoning"),
                "correct": correct,
                "miss_type": miss_type,
            })
        n = len(votes_in)
        n_correct = sum(1 for v in votes_in if v["correct"])
        n_called_ai = sum(1 for v in votes_in if v["guess"] == "ai")
        per_target[target_pid] = {
            "identity": session.player_identity(target_pid),
            "actual_type": actual,
            "n_votes_in": n,
            "n_correct": n_correct,
            "fraction_called_ai": _safe_div(n_called_ai, n),
            "votes_in": votes_in,
        }

    # Per-voter accuracy summary
    per_voter: dict[str, dict] = {}
    for voter_pid, voter_classifications in classifications.items():
        if voter_pid not in session.game_players:
            continue
        entries = []
        n_correct = 0
        n_total = 0
        for target_pid, entry in (voter_classifications or {}).items():
            actual = (ground_truth.get(target_pid) or "").upper()
            guess = (entry.get("guess") or "").upper()
            if actual and guess:
                n_total += 1
                if guess == actual:
                    n_correct += 1
            entries.append({
                "target_id": target_pid,
                "target_identity": session.player_identity(target_pid),
                "guess": guess.lower() if guess else None,
                "confidence": entry.get("confidence"),
                "correct": (guess == actual) if actual else None,
            })
        per_voter[voter_pid] = {
            "identity": session.player_identity(voter_pid),
            "accuracy": _safe_div(n_correct, n_total),
            "n_total": n_total,
            "n_correct": n_correct,
            "entries": entries,
        }

    return {
        "players_directory": pdir,
        "per_target": per_target,
        "per_voter": per_voter,
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# 9. Message clustering (the user's "clustering" ask)
# ---------------------------------------------------------------------------

def compute_message_clustering(session) -> dict:
    """TF-IDF + pairwise cosine + hierarchical clustering of player corpora.

    Each player's full message history (group + DM) becomes one document.
    We build a TF-IDF matrix over the player set, compute pairwise cosine
    similarity, and produce hierarchical-cluster assignments at k=2 and k=3.
    Pure-Python fallbacks make this work even without numpy/scipy.
    """
    pdir = session.players_directory()

    # Build per-player corpus
    corpus: dict[str, list[str]] = defaultdict(list)
    for m in session.group_chat_messages:
        pid = m.get("player_id")
        if pid in session.game_players:
            corpus[pid].extend(_tokens(m.get("content", "")))
    for m in session.dm_messages:
        pid = m.get("sender_id")
        if pid in session.game_players:
            corpus[pid].extend(_tokens(m.get("content", "")))

    player_ids = [pid for pid in session.game_players if corpus[pid]]
    if len(player_ids) < 2:
        return {
            "players_directory": pdir,
            "note": "Too few players with messages for clustering.",
            "n_players": len(player_ids),
        }

    # Vocabulary: keep tokens appearing in >=2 messages across the corpus,
    # cap at top 500 by document frequency for stability.
    df: Counter = Counter()
    for pid in player_ids:
        df.update(set(corpus[pid]))
    vocab = [w for w, c in df.most_common(500) if c >= 1]
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    n_docs = len(player_ids)

    # TF
    tf: dict[str, dict[int, float]] = {}
    for pid in player_ids:
        counts = Counter(corpus[pid])
        total = sum(counts.values()) or 1
        tf[pid] = {vocab_idx[w]: counts[w] / total for w in counts if w in vocab_idx}

    # IDF
    idf = {
        vocab_idx[w]: math.log(n_docs / (1 + df[w])) + 1.0
        for w in vocab
    }

    # TF-IDF vectors (sparse dict form)
    tfidf: dict[str, dict[int, float]] = {}
    for pid in player_ids:
        tfidf[pid] = {i: tf[pid][i] * idf[i] for i in tf[pid]}

    # Pairwise cosine similarity
    def _cos(a: dict, b: dict) -> float:
        # dot
        if len(a) > len(b):
            a, b = b, a
        dot = sum(v * b.get(i, 0.0) for i, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    similarity_matrix: dict[str, dict[str, float]] = {}
    for pa in player_ids:
        similarity_matrix[pa] = {}
        for pb in player_ids:
            if pa == pb:
                similarity_matrix[pa][pb] = 1.0
            elif pb in similarity_matrix and pa in similarity_matrix[pb]:
                similarity_matrix[pa][pb] = similarity_matrix[pb][pa]
            else:
                similarity_matrix[pa][pb] = _cos(tfidf[pa], tfidf[pb])

    # Hierarchical clustering (used only for k=2/k=3 group assignments —
    # the dashboard renders a 2D scatter via `scatter_2d` below).
    cluster_assignments: dict[str, dict] = {}

    n = len(player_ids)
    if HAS_SCIPY and HAS_NUMPY:
        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_matrix[player_ids[i]][player_ids[j]]
                condensed.append(max(0.0, 1.0 - sim))
        try:
            Z = _linkage(condensed, method="average")
            for k in (2, 3):
                if n >= k:
                    labels = _fcluster(Z, t=k, criterion="maxclust")
                    cluster_assignments[f"k={k}"] = {
                        player_ids[i]: int(labels[i]) for i in range(n)
                    }
        except Exception as e:  # pragma: no cover
            cluster_assignments["error"] = str(e)
    else:
        for k in (2, 3):
            if n < k:
                continue
            cluster_assignments[f"k={k}"] = _greedy_cluster(
                player_ids, similarity_matrix, k
            )

    # 2D layout via classical MDS on the cosine-distance matrix. Pure
    # numpy when available; falls back to a deterministic spring layout
    # otherwise. Each point's (x, y) means: closer ≈ more similar
    # speaking style.
    scatter_2d = _embed_2d(player_ids, similarity_matrix)

    # Group by cluster (k=2 typically separates HUMAN-style from AI-style)
    cluster_summaries = {}
    for k_label, mapping in cluster_assignments.items():
        if not isinstance(mapping, dict):
            continue
        groups: dict[int, list[str]] = defaultdict(list)
        for pid, c in mapping.items():
            groups[c].append(pid)
        cluster_summaries[k_label] = {
            str(c): {
                "members": [session.player_identity(pid) for pid in members],
                "n_humans": sum(1 for pid in members if session.player_type(pid) == "HUMAN"),
                "n_ais": sum(1 for pid in members if session.player_type(pid) == "AI"),
                "models": sorted({
                    session.player_model(pid) for pid in members
                    if session.player_model(pid)
                }),
            }
            for c, members in groups.items()
        }

    return {
        "players_directory": pdir,
        "players": player_ids,
        "vocabulary_size": len(vocab),
        "n_docs": n_docs,
        "similarity_matrix": similarity_matrix,
        "cluster_assignments": cluster_assignments,
        "cluster_summaries": cluster_summaries,
        "scatter_2d": scatter_2d,
        "method": (
            "scipy.cluster.hierarchy.linkage(method='average') + classical MDS"
            if HAS_SCIPY and HAS_NUMPY
            else "pure-Python greedy single-linkage + spring layout"
        ),
    }


def _greedy_cluster(
    items: list[str], sim_matrix: dict[str, dict[str, float]], k: int
) -> dict[str, int]:
    """Greedy single-linkage clustering down to k clusters. O(n^3) — fine for ≤20 players."""
    clusters: list[set[str]] = [{x} for x in items]

    while len(clusters) > k:
        best = (-1.0, 0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # max similarity between any two members across the pair
                sim = max(
                    sim_matrix[a][b]
                    for a in clusters[i] for b in clusters[j]
                )
                if sim > best[0]:
                    best = (sim, i, j)
        _, i, j = best
        clusters[i] = clusters[i] | clusters[j]
        clusters.pop(j)

    out = {}
    for cluster_idx, members in enumerate(clusters, start=1):
        for pid in members:
            out[pid] = cluster_idx
    return out


def _embed_2d(
    player_ids: list[str], sim_matrix: dict[str, dict[str, float]]
) -> dict | None:
    """Project players into a 2D plane where Euclidean distance ≈ cosine
    distance from `sim_matrix`. Uses classical MDS via numpy SVD when
    available, otherwise a deterministic Fruchterman-Reingold-style
    spring layout.

    Returns:
        {"coords": {pid: [x, y]}, "method": "mds"|"spring", "stress": ?}
        or None if there are <2 players.
    """
    n = len(player_ids)
    if n < 2:
        return None

    # Build distance matrix in player_ids order
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = 0.0
            else:
                sim = sim_matrix.get(player_ids[i], {}).get(player_ids[j], 0.0)
                D[i][j] = max(0.0, 1.0 - sim)

    if HAS_NUMPY:
        try:
            import numpy as np
            Dm = np.asarray(D, dtype=float)
            # Classical MDS: B = -1/2 J D^2 J ; coords = U[:, :2] * sqrt(eig[:2])
            D2 = Dm ** 2
            J = np.eye(n) - np.ones((n, n)) / n
            B = -0.5 * J @ D2 @ J
            # Symmetrize to mitigate float drift
            B = (B + B.T) / 2.0
            evals, evecs = np.linalg.eigh(B)
            # eigh returns ascending; flip to descending
            order = np.argsort(-evals)
            evals = evals[order]
            evecs = evecs[:, order]
            top = max(0, int((evals[:2] > 1e-12).sum()))
            if top == 0:
                # All players speak too uniformly to separate — fall through
                raise ValueError("degenerate distance matrix")
            coords = np.zeros((n, 2))
            for k in range(min(2, top)):
                coords[:, k] = evecs[:, k] * np.sqrt(max(evals[k], 0.0))
            # Stress: sqrt(sum (d_ij - ||x_i - x_j||)^2 / sum d_ij^2)
            num = 0.0
            den = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    fitted = float(np.linalg.norm(coords[i] - coords[j]))
                    num += (D[i][j] - fitted) ** 2
                    den += D[i][j] ** 2
            stress = (num / den) ** 0.5 if den > 0 else None
            return {
                "coords": {
                    player_ids[i]: [float(coords[i, 0]), float(coords[i, 1])]
                    for i in range(n)
                },
                "method": "classical_mds",
                "stress": stress,
            }
        except Exception:
            pass  # fall back to spring

    # Pure-Python deterministic spring layout. Stable seed → stable plot.
    import math
    import random
    rng = random.Random(0xC0FFEE)
    pos = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)]
    k_const = 1.0
    for _ in range(120):
        disp = [[0.0, 0.0] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.hypot(dx, dy) or 1e-6
                # Repulsive
                f_rep = k_const * k_const / dist
                # Attractive towards target distance (cosine distance)
                f_att = (dist - D[i][j]) * 0.5
                f = f_rep / dist - f_att / dist
                disp[i][0] += dx * f
                disp[i][1] += dy * f
        for i in range(n):
            mag = math.hypot(disp[i][0], disp[i][1]) or 1.0
            step = min(mag, 0.1)
            pos[i][0] += disp[i][0] / mag * step
            pos[i][1] += disp[i][1] / mag * step
    return {
        "coords": {player_ids[i]: [pos[i][0], pos[i][1]] for i in range(n)},
        "method": "spring_layout",
        "stress": None,
    }


# ---------------------------------------------------------------------------
# 10. Information-theoretic
# ---------------------------------------------------------------------------

def compute_information_theoretic(session) -> dict:
    """Per-player word entropy + char-bigram entropy + vocabulary growth."""
    pdir = session.players_directory()
    per_player: dict[str, dict] = {}

    for pid in session.game_players:
        msgs = []
        for m in session.group_chat_messages:
            if m.get("player_id") == pid:
                msgs.append(m.get("content", ""))
        for m in session.dm_messages:
            if m.get("sender_id") == pid:
                msgs.append(m.get("content", ""))

        all_text = " ".join(msgs)
        toks = _tokens(all_text)
        word_entropy = _entropy(Counter(toks))

        # char bigrams
        cleaned = re.sub(r"\s+", " ", all_text.lower())
        bigrams = [cleaned[i:i + 2] for i in range(len(cleaned) - 1)]
        char_bigram_entropy = _entropy(Counter(bigrams))

        # Vocabulary growth (cumulative unique tokens after each message)
        seen: set[str] = set()
        growth = []
        for m in msgs:
            for t in _tokens(m):
                seen.add(t)
            growth.append(len(seen))

        per_player[pid] = {
            "identity": session.player_identity(pid),
            "n_messages": len(msgs),
            "n_tokens": len(toks),
            "word_entropy_bits": word_entropy,
            "char_bigram_entropy_bits": char_bigram_entropy,
            "vocabulary_growth_curve": growth,
            "final_vocab_size": len(seen),
        }

    return {
        "players_directory": pdir,
        "per_player": per_player,
    }


# ---------------------------------------------------------------------------
# 10b. Post-game survey
# ---------------------------------------------------------------------------

# MUST mirror the SURVEY_QUESTIONS array in
# frontend-src/src/components/Survey.jsx — if that file changes, update this
# constant. Keeping a duplicate (instead of serving it from the server) avoids
# adding a runtime fetch dependency to the evaluator.
SURVEY_SCHEMA = [
    {"id": "q1",  "text": "How enjoyable was the game?", "type": "likert",  "required": True},
    {"id": "q2",  "text": "How difficult was it to identify the imposter?", "type": "likert",  "required": True},
    {"id": "q3",  "text": "How difficult was it to distinguish humans from AI?", "type": "likert",  "required": True},
    {"id": "q4",  "text": "How natural did the AI players' conversation feel?", "type": "likert",  "required": True},
    {"id": "q5",  "text": "Did you change your strategy after identifying potential AI players?", "type": "yesno", "required": True},
    {"id": "q6",  "text": "What strategies did you use to identify AI players?", "type": "checkbox", "depends_on_yes": "q5",
     "options": ["language_patterns", "response_timing", "hint_quality", "conversation_style", "voting_behavior", "other"]},
    {"id": "q7",  "text": "Which AI player was most convincing? Why?", "type": "textarea"},
    {"id": "q8",  "text": "Which AI player was least convincing? Why?", "type": "textarea"},
    {"id": "q9",  "text": "What gave away that a player was AI?", "type": "textarea"},
    {"id": "q10", "text": "Would you play again?", "type": "likert",  "required": True},
    {"id": "q11", "text": "Any additional feedback?", "type": "textarea"},
]


def compute_survey_responses(session) -> dict:
    """Aggregate post-game survey data across all human respondents.

    Joins each saved survey with the question schema and player identity,
    rolls up Likert distributions per question, lists open-ended text, and
    surfaces metadata (partial vs final, who submitted, which questions were
    actively skipped).
    """
    pdir = session.players_directory()
    surveys = getattr(session, "surveys", {}) or {}

    per_respondent: list[dict] = []
    for pid, wrapped in surveys.items():
        responses = wrapped.get("responses") or {}
        identity = session.player_identity(pid) if pid in session.game_players else {
            "player_id": pid,
            "real_name": wrapped.get("real_name") or pid,
            "player_type": wrapped.get("player_type") or "HUMAN",
            "model_name": None,
        }
        per_q = []
        for q in SURVEY_SCHEMA:
            raw = responses.get(q["id"])
            applicable = True
            if q.get("depends_on_yes"):
                gate = responses.get(q["depends_on_yes"])
                if gate != "yes":
                    applicable = False
            answered = raw is not None and raw != "" and raw != []
            per_q.append({
                "id": q["id"],
                "text": q["text"],
                "type": q["type"],
                "required": q.get("required", False),
                "applicable": applicable,
                "answered": answered,
                "answer": raw,
                "options": q.get("options"),
            })
        per_respondent.append({
            "identity": identity,
            "submitted_at": wrapped.get("submitted_at"),
            "is_partial": wrapped.get("is_partial", False),
            "questions": per_q,
        })

    # Aggregate distributions across respondents
    aggregate: dict[str, dict] = {}
    for q in SURVEY_SCHEMA:
        bucket: dict[str, int] = defaultdict(int)
        textual: list[dict] = []
        n_applicable = 0
        n_answered = 0
        for resp in per_respondent:
            entry = next((e for e in resp["questions"] if e["id"] == q["id"]), None)
            if not entry or not entry["applicable"]:
                continue
            n_applicable += 1
            if not entry["answered"]:
                continue
            n_answered += 1
            ans = entry["answer"]
            if q["type"] in ("likert", "yesno"):
                bucket[str(ans)] += 1
            elif q["type"] == "checkbox":
                if isinstance(ans, list):
                    for opt in ans:
                        bucket[str(opt)] += 1
            elif q["type"] == "textarea":
                textual.append({
                    "respondent": resp["identity"]["real_name"],
                    "respondent_id": resp["identity"]["player_id"],
                    "answer": ans,
                })
        aggregate[q["id"]] = {
            "text": q["text"],
            "type": q["type"],
            "n_applicable": n_applicable,
            "n_answered": n_answered,
            "completion_rate": _safe_div(n_answered, n_applicable),
            "distribution": dict(bucket) if q["type"] != "textarea" else None,
            "open_ended_responses": textual if q["type"] == "textarea" else None,
        }

    return {
        "players_directory": pdir,
        "schema": SURVEY_SCHEMA,
        "per_respondent": per_respondent,
        "aggregate_per_question": aggregate,
        "n_respondents": len(per_respondent),
        "n_partial": sum(1 for r in per_respondent if r["is_partial"]),
    }


# ---------------------------------------------------------------------------
# 11. Communication graph
# ---------------------------------------------------------------------------

def compute_communication_graph(session) -> dict:
    """Multi-typed directed graph: DMs, mentions, accusations, vote targets.

    Output is suitable for a D3 force-directed visualization in the dashboard.
    Each node carries identity; each edge has a `type` and a `weight`.
    """
    pdir = session.players_directory()

    nodes = []
    msg_count_per_pid: dict[str, int] = defaultdict(int)
    for m in session.group_chat_messages:
        if m.get("player_id") in session.game_players:
            msg_count_per_pid[m["player_id"]] += 1
    for m in session.dm_messages:
        if m.get("sender_id") in session.game_players:
            msg_count_per_pid[m["sender_id"]] += 1

    for pid in session.game_players:
        ident = session.player_identity(pid)
        nodes.append({
            **ident,
            "msg_count": msg_count_per_pid.get(pid, 0),
        })

    # Edges
    edges: list[dict] = []

    # DM edges (directed)
    dm_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for m in session.dm_messages:
        s = m.get("sender_id")
        r = m.get("recipient_id")
        if s in session.game_players and r in session.game_players:
            dm_pair_counts[(s, r)] += 1
    for (s, r), cnt in dm_pair_counts.items():
        edges.append({
            "source": s,
            "target": r,
            "type": "dm",
            "weight": cnt,
        })

    # Mention edges from group chat (sender mentions target by id/name/pN)
    static_patterns = _build_static_name_patterns(session)
    pn_map = _build_round_pn_map(session)

    mention_counts: dict[tuple[str, str], int] = defaultdict(int)
    accusation_counts: dict[tuple[str, str], int] = defaultdict(int)
    for m in session.group_chat_messages:
        sender = m.get("player_id")
        if sender not in session.game_players:
            continue
        content = m.get("content") or ""
        rn = m.get("round_number")
        words_lower = set(re.findall(r"\b\w+\b", content.lower()))
        is_accusation = bool(words_lower & ACCUSATION_KEYWORDS)
        targets = _resolve_mentions(session, content, rn, pn_map, static_patterns)
        for target_pid in targets:
            if target_pid == sender:
                continue
            mention_counts[(sender, target_pid)] += 1
            if is_accusation:
                accusation_counts[(sender, target_pid)] += 1

    for (s, r), cnt in mention_counts.items():
        edges.append({
            "source": s,
            "target": r,
            "type": "mention",
            "weight": cnt,
        })
    for (s, r), cnt in accusation_counts.items():
        edges.append({
            "source": s,
            "target": r,
            "type": "accusation",
            "weight": cnt,
        })

    # Vote-target edges (one per voter per round)
    vote_counts: dict[tuple[str, str], int] = defaultdict(int)
    for r_ev in session.round_results_events:
        for voter, target in (r_ev.get("votes") or {}).items():
            if voter in session.game_players and target in session.game_players:
                vote_counts[(voter, target)] += 1
    for (s, r), cnt in vote_counts.items():
        edges.append({
            "source": s,
            "target": r,
            "type": "vote",
            "weight": cnt,
        })

    # Centrality (degree-based, manual — works without networkx)
    in_degree: dict[str, int] = defaultdict(int)
    out_degree: dict[str, int] = defaultdict(int)
    weighted_in: dict[str, int] = defaultdict(int)
    weighted_out: dict[str, int] = defaultdict(int)
    for e in edges:
        out_degree[e["source"]] += 1
        in_degree[e["target"]] += 1
        weighted_out[e["source"]] += e["weight"]
        weighted_in[e["target"]] += e["weight"]

    centrality = {
        pid: {
            "identity": session.player_identity(pid),
            "in_degree": in_degree.get(pid, 0),
            "out_degree": out_degree.get(pid, 0),
            "weighted_in": weighted_in.get(pid, 0),
            "weighted_out": weighted_out.get(pid, 0),
        }
        for pid in session.game_players
    }

    return {
        "players_directory": pdir,
        "nodes": nodes,
        "edges": edges,
        "centrality": centrality,
        "edge_types": ["dm", "mention", "accusation", "vote"],
    }
