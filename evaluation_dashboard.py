"""
evaluation_dashboard.py — HTML dashboard renderer for The Sandbox analysis.

Generates a single self-contained `dashboard.html` with all analysis data
embedded inline as JSON, plus rendering JS using Chart.js v4 and D3 v7
loaded from CDN. Open the file in any modern browser; no server needed.

Sections rendered:
  1. Overview            — identity directory, session info, final scores
  2. Game Results        — per-round timeline, detection rates by type/model
  3. Linguistic          — HUMAN vs AI radar, per-player feature table
  4. Communication Graph — D3 force-directed network (DM/mention/accusation/vote)
  5. Voting Patterns     — voting-agreement heatmap, strong pairs
  6. Clustering          — TF-IDF cosine heatmap, dendrogram, cluster summary
  7. Hints               — per-round hints with imposter highlighted
  8. Classification      — meta-game confusion matrix + per-voter detail
  9. Per-Player          — message timeline + feature vector for any player
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_dashboard(session, results: dict) -> str:
    """Return a complete HTML document string for the dashboard."""

    # Embed raw chat so the per-player deep-dive can render actual message
    # text. We only need a thin projection (sender + content + ts + round + phase).
    raw_group = [
        {
            "ts": m.get("ts"),
            "player_id": m.get("player_id"),
            "content": m.get("content"),
            "round_number": m.get("round_number"),
            "phase": m.get("phase"),
        }
        for m in session.group_chat_messages
    ]
    raw_dms = [
        {
            "ts": m.get("ts"),
            "sender_id": m.get("sender_id"),
            "recipient_id": m.get("recipient_id"),
            "content": m.get("content"),
            "round_number": m.get("round_number"),
            "phase": m.get("phase"),
        }
        for m in session.dm_messages
    ]

    payload = {
        "session_id": session.session_id,
        "started_at": session.started_at,
        "is_complete": session.is_complete,
        "n_rounds_played": session.num_rounds_played,
        "n_rounds_configured": session.num_rounds_configured,
        "players_directory": session.players_directory(),
        "results": results,
        "raw_group_chat": raw_group,
        "raw_dms": raw_dms,
    }

    # Embed payload safely: split any literal "</" sequence so it can never
    # break out of the <script> tag — and json.dumps escapes the rest.
    payload_json = json.dumps(payload, default=str).replace("</", "<\\/")

    return _HTML_TEMPLATE.replace("__PAYLOAD__", payload_json).replace(
        "__SESSION_ID__", session.session_id
    )


def write_dashboard(session, results: dict, out_path: Path) -> Path:
    out_path.write_text(render_dashboard(session, results), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# HTML / CSS / JS template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>The Sandbox — Analysis Dashboard (__SESSION_ID__)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7.8.5/dist/d3.min.js"></script>
<style>
  :root {
    --bg: #0f1115;
    --panel: #181b22;
    --panel-2: #1f242d;
    --border: #2a313d;
    --text: #e6e9ef;
    --muted: #98a2b3;
    --accent: #7aa2f7;
    --human: #4ade80;
    --ai: #f59e0b;
    --imposter: #ef4444;
    --good: #22c55e;
    --bad: #ef4444;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: var(--bg); color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 14px; }
  header { position: sticky; top: 0; z-index: 10;
    background: rgba(15, 17, 21, 0.96); border-bottom: 1px solid var(--border);
    padding: 14px 24px; backdrop-filter: blur(8px); }
  header h1 { margin: 0 0 6px 0; font-size: 18px; color: var(--text); }
  header .meta { color: var(--muted); font-size: 12px; }
  nav { display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 24px;
    background: var(--panel); border-bottom: 1px solid var(--border);
    position: sticky; top: 56px; z-index: 9; }
  nav button { background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px;
    cursor: pointer; font-size: 13px; }
  nav button:hover { border-color: var(--accent); }
  nav button.active { background: var(--accent); color: #0b0d12; border-color: var(--accent); }
  main { padding: 24px; max-width: 1400px; margin: 0 auto; }
  section { background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px; margin-bottom: 20px; }
  section h2 { margin-top: 0; font-size: 16px; color: var(--accent); }
  section h3 { font-size: 14px; color: var(--text); margin-top: 24px; }
  .grid { display: grid; gap: 16px; }
  .grid-2 { grid-template-columns: 1fr 1fr; }
  .grid-3 { grid-template-columns: repeat(3, 1fr); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 10px; text-align: left;
    border-bottom: 1px solid var(--border); }
  th { background: var(--panel-2); color: var(--muted);
    font-weight: 600; font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.04em; }
  tr:hover td { background: rgba(122, 162, 247, 0.06); }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600; }
  .pill-human { background: rgba(74, 222, 128, 0.18); color: var(--human); }
  .pill-ai    { background: rgba(245, 158, 11, 0.18); color: var(--ai); }
  .pill-imp   { background: rgba(239, 68, 68, 0.20); color: var(--imposter); }
  .pill-good  { background: rgba(34, 197, 94, 0.18); color: var(--good); }
  .pill-bad   { background: rgba(239, 68, 68, 0.18); color: var(--bad); }
  .muted { color: var(--muted); }
  .mono  { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
  .canvas-wrap { background: var(--panel-2); padding: 12px; border-radius: 8px;
    border: 1px solid var(--border); }
  .heatmap-cell { display: inline-block; width: 36px; height: 36px;
    text-align: center; line-height: 36px; font-size: 11px;
    border: 1px solid var(--border); }
  .legend { display: flex; gap: 14px; font-size: 12px; color: var(--muted);
    margin: 8px 0; flex-wrap: wrap; }
  .legend span::before { content: ""; display: inline-block; width: 10px;
    height: 10px; margin-right: 6px; border-radius: 50%; vertical-align: middle; }
  .toggle { display: inline-flex; gap: 6px; align-items: center; margin-right: 14px;
    color: var(--muted); font-size: 12px; cursor: pointer; user-select: none; }
  .toggle input { accent-color: var(--accent); }
  select, input[type=text] { background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); padding: 6px 10px; border-radius: 6px;
    font-size: 13px; }
  .msg-row { padding: 6px 8px; border-left: 3px solid var(--border);
    margin-bottom: 4px; background: var(--panel-2); border-radius: 0 4px 4px 0; }
  .msg-row.human { border-left-color: var(--human); }
  .msg-row.ai    { border-left-color: var(--ai); }
  .msg-meta { color: var(--muted); font-size: 11px; margin-bottom: 2px; }
  svg.network { width: 100%; height: 560px; background: var(--panel-2);
    border: 1px solid var(--border); border-radius: 8px; display: block; }
  svg.dendrogram { width: 100%; height: 320px; background: var(--panel-2);
    border: 1px solid var(--border); border-radius: 8px; display: block; }
  .badge-num { font-weight: 600; color: var(--text); }
  .kpi { display: flex; flex-direction: column; align-items: center;
    padding: 14px; background: var(--panel-2); border-radius: 8px;
    border: 1px solid var(--border); }
  .kpi .v { font-size: 22px; font-weight: 700; color: var(--text); }
  .kpi .l { color: var(--muted); font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.04em; margin-top: 4px; }
  details summary { cursor: pointer; padding: 6px 0; color: var(--accent); }
  details[open] summary { margin-bottom: 8px; }
  .small { font-size: 12px; }
  hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }
</style>
</head>
<body>
<header>
  <h1>The Sandbox — Analysis Dashboard</h1>
  <div class="meta" id="header-meta"></div>
</header>
<nav id="navbar"></nav>
<main id="main"></main>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";

// ============================================================================
// Data + helpers
// ============================================================================
const DATA = JSON.parse(document.getElementById("payload").textContent);
const R = DATA.results || {};
const DIR = DATA.players_directory || {};
const PIDS = Object.keys(DIR);

function ident(pid) { return DIR[pid] || {player_id: pid, real_name: pid, player_type: "?", model_name: null, label: pid}; }
function isHuman(pid) { return ident(pid).player_type === "HUMAN"; }
function isAI(pid)    { return ident(pid).player_type === "AI"; }
function pillFor(pid) {
  const t = ident(pid).player_type;
  const cls = t === "HUMAN" ? "pill-human" : t === "AI" ? "pill-ai" : "";
  return `<span class="pill ${cls}">${t}</span>`;
}
function colorFor(pid) {
  return isHuman(pid) ? getCSS("--human") : getCSS("--ai");
}
function getCSS(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function fmtPct(v) { return v == null ? "—" : (v * 100).toFixed(1) + "%"; }
function fmtNum(v, p=3) { return v == null ? "—" : Number(v).toFixed(p); }
function fmtInt(v) { return v == null ? "—" : Math.round(v); }
function escapeHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]
  );
}
function labelInline(pid) {
  const i = ident(pid);
  const m = i.model_name ? " · " + i.model_name.split("/").pop() : "";
  return `<span class="mono">${escapeHtml(i.player_id)}</span> · "<b>${escapeHtml(i.real_name)}</b>" ${pillFor(pid)}<span class="muted small">${escapeHtml(m)}</span>`;
}

// ============================================================================
// Header + nav
// ============================================================================
document.getElementById("header-meta").innerHTML =
  `Session <span class="mono">${escapeHtml(DATA.session_id)}</span> · ` +
  `started ${escapeHtml((DATA.started_at || "").slice(0, 19).replace("T", " "))} · ` +
  `${DATA.n_rounds_played}/${DATA.n_rounds_configured} rounds · ` +
  (DATA.is_complete ? "<span class=\"pill pill-good\">complete</span>" : "<span class=\"pill pill-bad\">incomplete</span>");

const SECTIONS = [
  ["overview",     "Overview"],
  ["game",         "Game Results"],
  ["linguistic",   "Linguistic"],
  ["network",      "Communication Graph"],
  ["voting",       "Voting Patterns"],
  ["clustering",   "Clustering"],
  ["hints",        "Hints"],
  ["classify",     "Classification"],
  ["survey",       "Survey"],
  ["dive",         "Per-Player Deep Dive"],
];
const nav = document.getElementById("navbar");
SECTIONS.forEach(([id, label], i) => {
  const b = document.createElement("button");
  b.textContent = label;
  b.dataset.target = id;
  b.onclick = () => showSection(id);
  if (i === 0) b.classList.add("active");
  nav.appendChild(b);
});
// Section-specific drawers run only after the section becomes visible. Drawing
// into a `display:none` parent makes Chart.js misread canvas pixel dimensions
// and makes any layout that reads `clientWidth` collapse to zero.
const SECTION_DRAWERS = {
  overview:   drawOverviewCharts,
  game:       drawGameCharts,
  linguistic: drawLinguisticCharts,
  network:    drawNetwork,
  voting:     drawVotingHeatmap,
  clustering: () => { drawClusterHeatmap(); drawDendrogram(); drawTSNE(); },
  hints:      () => {},
  classify:   () => {},
  survey:     () => {},
  dive:       setupDive,
};
const drawn = new Set();

function showSection(id) {
  document.querySelectorAll("#navbar button").forEach(b =>
    b.classList.toggle("active", b.dataset.target === id)
  );
  document.querySelectorAll("main > section").forEach(s =>
    s.style.display = s.id === id ? "" : "none"
  );
  if (!drawn.has(id) && SECTION_DRAWERS[id]) {
    drawn.add(id);
    // Wait one frame so the browser has applied the visibility change before
    // we measure / instantiate any layout-sensitive widgets.
    requestAnimationFrame(() => SECTION_DRAWERS[id]());
  }
}

// ============================================================================
// Section: Overview
// ============================================================================
function renderOverview() {
  const finals = (R.game_metrics && R.game_metrics.final_scores) || {};
  const sortedFinals = Object.entries(finals).sort((a, b) => b[1] - a[1]);

  const idRows = PIDS.map(pid => {
    const i = ident(pid);
    return `<tr>
      <td class="mono">${escapeHtml(i.player_id)}</td>
      <td><b>${escapeHtml(i.real_name)}</b></td>
      <td>${pillFor(pid)}</td>
      <td class="muted small">${escapeHtml(i.model_name || "—")}</td>
      <td class="badge-num">${finals[pid] != null ? finals[pid] : "—"}</td>
    </tr>`;
  }).join("");

  const overall = R.game_metrics && R.game_metrics.overall_detection_rate;

  const kpis = `
    <div class="grid grid-3">
      <div class="kpi"><div class="v">${PIDS.length}</div><div class="l">Players</div></div>
      <div class="kpi"><div class="v">${PIDS.filter(isHuman).length} / ${PIDS.filter(isAI).length}</div><div class="l">Human / AI</div></div>
      <div class="kpi"><div class="v">${fmtPct(overall)}</div><div class="l">Imposter Detection Rate</div></div>
    </div>`;

  return `
    <section id="overview">
      <h2>Overview</h2>
      ${kpis}
      <h3>Identity Directory</h3>
      <div class="muted small">Every analysis output cross-references this table by <span class="mono">player_id</span>.</div>
      <table>
        <thead><tr><th>Player ID</th><th>Real Name</th><th>Type</th><th>Model</th><th>Final Score</th></tr></thead>
        <tbody>${idRows}</tbody>
      </table>
      <h3>Final Scores</h3>
      <div class="canvas-wrap"><canvas id="chart-finals" height="220"></canvas></div>
    </section>`;
}

function drawOverviewCharts() {
  const finals = (R.game_metrics && R.game_metrics.final_scores) || {};
  const labels = PIDS.map(pid => `${ident(pid).real_name}`);
  const values = PIDS.map(pid => finals[pid] || 0);
  const colors = PIDS.map(pid => colorFor(pid));
  new Chart(document.getElementById("chart-finals"), {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: colors }] },
    options: { plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: "#98a2b3" } }, y: { ticks: { color: "#98a2b3" } } } },
  });
}

// ============================================================================
// Section: Game Results
// ============================================================================
function renderGame() {
  const gm = R.game_metrics || {};
  const rounds = gm.rounds || [];
  const detTypeRows = Object.entries(gm.detection_by_imposter_type || {})
    .map(([k, v]) => `<tr><td>${k}</td><td>${fmtPct(v)}</td></tr>`).join("");
  const detModelRows = Object.entries(gm.detection_by_model || {})
    .map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${fmtPct(v)}</td></tr>`).join("");

  const roundRows = rounds.map(r => {
    const imp = r.imposter_id ? labelInline(r.imposter_id) : "?";
    const out = r.imposter_caught
      ? `<span class="pill pill-good">CAUGHT</span>`
      : `<span class="pill pill-bad">ESCAPED</span>`;
    const correct = (r.correct_voters || []).map(v => `<span class="mono">${escapeHtml(v)}</span>`).join(", ");
    return `<tr>
      <td>${r.round_number}</td>
      <td>${escapeHtml(r.topic || "?")}</td>
      <td><span class="mono">${escapeHtml(r.secret_word || "")}</span></td>
      <td>${imp}</td>
      <td>${out}</td>
      <td class="muted small">${correct || "—"}</td>
    </tr>`;
  }).join("");

  const vap = gm.vote_accuracy_per_player || {};
  const voteRows = Object.entries(vap)
    .sort((a,b) => (b[1].accuracy || 0) - (a[1].accuracy || 0))
    .map(([pid, s]) => `<tr>
      <td>${labelInline(pid)}</td>
      <td>${s.correct_votes}/${s.total_votes}</td>
      <td>${fmtPct(s.accuracy)}</td>
    </tr>`).join("");

  return `
    <section id="game" style="display:none">
      <h2>Game Results</h2>
      <h3>Per-Round Breakdown</h3>
      <table>
        <thead><tr><th>#</th><th>Topic</th><th>Secret</th><th>Imposter</th><th>Outcome</th><th>Correct Voters</th></tr></thead>
        <tbody>${roundRows}</tbody>
      </table>

      <div class="grid grid-2" style="margin-top:20px">
        <div>
          <h3>Detection by Imposter Type</h3>
          <table><thead><tr><th>Type</th><th>Detection Rate</th></tr></thead><tbody>${detTypeRows}</tbody></table>
        </div>
        <div>
          <h3>Detection by Model</h3>
          <table><thead><tr><th>Model</th><th>Detection Rate</th></tr></thead><tbody>${detModelRows}</tbody></table>
        </div>
      </div>

      <h3>Vote Accuracy per Player</h3>
      <table><thead><tr><th>Player</th><th>Correct/Total</th><th>Accuracy</th></tr></thead><tbody>${voteRows}</tbody></table>

      <h3>Cumulative Detection Accuracy Over Rounds</h3>
      <div class="canvas-wrap"><canvas id="chart-detection-curve" height="220"></canvas></div>
    </section>`;
}

function drawGameCharts() {
  const curve = (R.game_metrics && R.game_metrics.detection_over_rounds) || [];
  const labels = curve.map(r => "R" + r.round_number);
  new Chart(document.getElementById("chart-detection-curve"), {
    type: "line",
    data: { labels,
      datasets: [
        { label: "Human voters", data: curve.map(r => (r.human_cumul_accuracy || 0) * 100),
          borderColor: getCSS("--human"), backgroundColor: getCSS("--human") + "33", tension: 0.3 },
        { label: "AI voters",    data: curve.map(r => (r.ai_cumul_accuracy || 0) * 100),
          borderColor: getCSS("--ai"),    backgroundColor: getCSS("--ai") + "33",    tension: 0.3 },
      ]},
    options: { plugins: { legend: { labels: { color: "#e6e9ef" } } },
      scales: { x: { ticks: { color: "#98a2b3" } }, y: { ticks: { color: "#98a2b3" }, max: 100 } } },
  });
}

// ============================================================================
// Section: Linguistic
// ============================================================================
function renderLinguistic() {
  const lf = R.linguistic_features || {};
  const grp = lf.group_summary || {};
  const players = lf.players || {};
  const rows = Object.entries(players).sort((a,b) => (b[1].msg_count||0) - (a[1].msg_count||0))
    .map(([pid, p]) => `<tr>
      <td>${labelInline(pid)}</td>
      <td>${p.msg_count || 0}</td>
      <td>${fmtNum(p.avg_msg_length_words)}</td>
      <td>${fmtNum(p.vocab_diversity)}</td>
      <td>${fmtNum(p.slang_rate)}</td>
      <td>${fmtNum(p.question_rate)}</td>
      <td>${fmtNum(p.first_word_diversity)}</td>
    </tr>`).join("");

  // Deep stylometry table
  const deep = (R.deep_stylometry && R.deep_stylometry.per_player) || {};
  const deepRows = Object.entries(deep).map(([pid, p]) => `<tr>
    <td>${labelInline(pid)}</td>
    <td>${fmtNum(p.hedge_word_rate)}</td>
    <td>${fmtNum(p.self_reference_rate)}</td>
    <td>${fmtNum(p.vague_token_rate)}</td>
    <td>${fmtNum(p.contraction_rate_per_msg)}</td>
    <td>${fmtNum(p.sentences_per_msg)}</td>
    <td>${fmtNum(p.avg_sentence_length_words)}</td>
  </tr>`).join("");

  return `
    <section id="linguistic" style="display:none">
      <h2>Linguistic Features</h2>
      <h3>HUMAN vs AI — averaged style profile</h3>
      <div class="canvas-wrap"><canvas id="chart-radar" height="280"></canvas></div>

      <h3>Per-Player Surface Features</h3>
      <table>
        <thead><tr><th>Player</th><th>Msgs</th><th>Avg Words</th><th>Vocab Div</th><th>Slang</th><th>Question</th><th>1st-word Div</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>

      <h3>Deep Stylometry — hedge / self-ref / vague / contractions / sentences</h3>
      <div class="muted small">Function-word distribution + lexical signals that resist topical confounds.</div>
      <table>
        <thead><tr><th>Player</th><th>Hedge rate</th><th>Self-ref</th><th>Vague</th><th>Contractions/msg</th><th>Sent/msg</th><th>Avg sent words</th></tr></thead>
        <tbody>${deepRows}</tbody>
      </table>
    </section>`;
}

function drawLinguisticCharts() {
  const grp = (R.linguistic_features && R.linguistic_features.group_summary) || {};
  const labels = ["Avg msg words", "Vocab diversity", "Slang", "Questions", "Exclamations", "Capitals", "1st-word div"];
  const keys = ["avg_msg_length_words", "vocab_diversity", "slang_rate", "question_rate", "exclamation_rate", "capitalization_rate", "first_word_diversity"];
  function vec(g) { return keys.map(k => g[k] == null ? 0 : g[k]); }
  // Normalize each axis independently (max across H/A) so radar isn't dominated by msg-length.
  const H = vec(grp.HUMAN || {}), A = vec(grp.AI || {});
  const norms = keys.map((_, i) => Math.max(H[i], A[i]) || 1);
  const Hn = H.map((v, i) => v / norms[i]);
  const An = A.map((v, i) => v / norms[i]);
  new Chart(document.getElementById("chart-radar"), {
    type: "radar",
    data: { labels, datasets: [
      { label: "HUMAN avg", data: Hn, backgroundColor: getCSS("--human") + "33", borderColor: getCSS("--human") },
      { label: "AI avg",    data: An, backgroundColor: getCSS("--ai") + "33",    borderColor: getCSS("--ai") },
    ]},
    options: {
      plugins: { legend: { labels: { color: "#e6e9ef" } } },
      scales: { r: { suggestedMin: 0, suggestedMax: 1,
        angleLines: { color: "#2a313d" }, grid: { color: "#2a313d" },
        pointLabels: { color: "#e6e9ef", font: { size: 11 } },
        ticks: { color: "#98a2b3", backdropColor: "transparent" } } },
    },
  });
}

// ============================================================================
// Section: Communication Graph
// ============================================================================
function renderNetwork() {
  return `
    <section id="network" style="display:none">
      <h2>Communication Graph</h2>
      <div class="muted small">Force-directed network. Nodes are players (sized by message volume). Edges are directed; toggle types below.</div>
      <div class="legend">
        <span style="color:var(--human)">HUMAN</span>
        <span style="color:var(--ai)">AI</span>
      </div>
      <div style="margin: 8px 0">
        <label class="toggle"><input type="checkbox" data-edge="dm" checked> DMs</label>
        <label class="toggle"><input type="checkbox" data-edge="mention" checked> Mentions</label>
        <label class="toggle"><input type="checkbox" data-edge="accusation" checked> Accusations</label>
        <label class="toggle"><input type="checkbox" data-edge="vote" checked> Vote targets</label>
      </div>
      <svg class="network" id="svg-network" viewBox="0 0 1000 560" preserveAspectRatio="xMidYMid meet"></svg>

      <h3>Centrality</h3>
      <table>
        <thead><tr><th>Player</th><th>Out-deg (uniq)</th><th>In-deg (uniq)</th><th>Weighted out (msgs)</th><th>Weighted in (msgs)</th></tr></thead>
        <tbody id="centrality-rows"></tbody>
      </table>
    </section>`;
}

function drawNetwork() {
  const cg = R.communication_graph || {};
  const nodes = (cg.nodes || []).map(n => ({ ...n, id: n.player_id }));
  let edges = (cg.edges || []).map(e => ({ ...e }));

  const svg = d3.select("#svg-network");
  // Use the SVG's own viewBox as the virtual coordinate system. clientWidth
  // returns 0 when the section is hidden at boot, which used to collapse the
  // graph into the top-left corner; viewBox is layout-independent.
  const vb = (svg.attr("viewBox") || "0 0 1000 560").split(/\s+/).map(Number);
  const W = vb[2], H = vb[3];
  svg.selectAll("*").remove();

  const edgeColor = { dm: "#7aa2f7", mention: "#9aa5b1", accusation: "#ef4444", vote: "#f59e0b" };
  const sim = d3.forceSimulation(nodes)
    .force("link",   d3.forceLink([]).id(d => d.id).distance(140))
    .force("charge", d3.forceManyBody().strength(-260))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collide", d3.forceCollide(36));

  // Arrowheads per edge type
  const defs = svg.append("defs");
  ["dm", "mention", "accusation", "vote"].forEach(t => {
    defs.append("marker").attr("id", "arrow-" + t)
      .attr("viewBox", "0 -5 10 10").attr("refX", 24).attr("refY", 0)
      .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
      .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", edgeColor[t]);
  });

  let linkSel = svg.append("g").selectAll("line");
  let nodeSel = svg.append("g").selectAll("circle");
  let labelSel = svg.append("g").selectAll("text");

  function update(activeTypes) {
    const filteredEdges = edges.filter(e => activeTypes.has(e.type));
    sim.force("link").links(filteredEdges);
    linkSel = linkSel.data(filteredEdges, d => d.source.id + "->" + d.target.id + "/" + d.type);
    linkSel.exit().remove();
    linkSel = linkSel.enter().append("line")
      .attr("stroke", d => edgeColor[d.type] || "#888")
      .attr("stroke-opacity", 0.65)
      .attr("stroke-width", d => Math.min(8, 1 + Math.log(d.weight + 1) * 1.6))
      .attr("marker-end", d => "url(#arrow-" + d.type + ")")
      .merge(linkSel);

    nodeSel = nodeSel.data(nodes, d => d.id);
    nodeSel.exit().remove();
    nodeSel = nodeSel.enter().append("circle")
      .attr("r", d => Math.max(10, Math.min(28, 8 + Math.sqrt(d.msg_count || 1) * 2)))
      .attr("fill", d => d.player_type === "HUMAN" ? getCSS("--human") : getCSS("--ai"))
      .attr("stroke", "#0f1115").attr("stroke-width", 2)
      .call(drag(sim))
      .merge(nodeSel);
    nodeSel.append("title").text(d => `${d.real_name} (${d.player_type}${d.model_name ? ", " + d.model_name : ""})\nmsgs: ${d.msg_count}`);

    labelSel = labelSel.data(nodes, d => d.id);
    labelSel.exit().remove();
    labelSel = labelSel.enter().append("text")
      .attr("text-anchor", "middle").attr("dy", -22)
      .attr("fill", "#e6e9ef").attr("font-size", 11)
      .text(d => d.real_name)
      .merge(labelSel);

    sim.alpha(0.7).restart();
  }

  sim.on("tick", () => {
    linkSel
      .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    nodeSel.attr("cx", d => d.x).attr("cy", d => d.y);
    labelSel.attr("x", d => d.x).attr("y", d => d.y);
  });

  function drag(simulation) {
    return d3.drag()
      .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag",  (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on("end",   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; });
  }

  function activeTypes() {
    const set = new Set();
    document.querySelectorAll("#network input[type=checkbox]").forEach(cb => {
      if (cb.checked) set.add(cb.dataset.edge);
    });
    return set;
  }
  document.querySelectorAll("#network input[type=checkbox]").forEach(cb => {
    cb.addEventListener("change", () => update(activeTypes()));
  });
  update(activeTypes());

  // Centrality table
  const c = cg.centrality || {};
  const rows = Object.entries(c)
    .sort((a, b) => (b[1].weighted_out || 0) - (a[1].weighted_out || 0))
    .map(([pid, v]) => `<tr>
      <td>${labelInline(pid)}</td>
      <td>${v.out_degree}</td><td>${v.in_degree}</td>
      <td>${v.weighted_out}</td><td>${v.weighted_in}</td>
    </tr>`).join("");
  document.getElementById("centrality-rows").innerHTML = rows;
}

// ============================================================================
// Section: Voting Patterns
// ============================================================================
function renderVoting() {
  const vb = R.voting_blocs || {};
  const cat = vb.category_averages || {};
  const catRows = Object.entries(cat).sort((a,b) => (b[1]||0) - (a[1]||0))
    .map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${fmtPct(v)}</td></tr>`).join("");

  const strong = vb.strong_voting_pairs || [];
  const strongRows = strong.map(p => {
    const [a, b] = p.players;
    return `<tr>
      <td>${labelInline(a)} ↔ ${labelInline(b)}</td>
      <td>${fmtPct(p.agreement_rate)}</td>
      <td>${p.rounds_both_voted}</td>
      <td>${escapeHtml(p.category)}</td>
    </tr>`;
  }).join("");

  // Heatmap of pairwise agreement
  const pairs = vb.pairs || {};
  const sr = R.suspicion_received && R.suspicion_received.per_player || {};
  const susRows = Object.entries(sr)
    .sort((a,b) => (b[1].accusations_received||0) - (a[1].accusations_received||0))
    .map(([pid, v]) => `<tr>
      <td>${labelInline(pid)}</td>
      <td>${v.votes_received_total}</td>
      <td>${v.mentions_in_group_chat}</td>
      <td>${v.accusations_received}</td>
    </tr>`).join("");

  return `
    <section id="voting" style="display:none">
      <h2>Voting Patterns</h2>
      <h3>Voting Agreement Heatmap</h3>
      <div class="muted small">Cell = % of rounds where row voted the same as column.</div>
      <div id="vote-heatmap"></div>

      <h3>Average Agreement by Pair Category</h3>
      <table><thead><tr><th>Category</th><th>Avg Agreement</th></tr></thead><tbody>${catRows}</tbody></table>

      <h3>Strong Voting Pairs (≥67% agreement, ≥2 rounds)</h3>
      ${strong.length ? `<table>
        <thead><tr><th>Pair</th><th>Agreement</th><th>Rounds</th><th>Category</th></tr></thead>
        <tbody>${strongRows}</tbody></table>` : `<div class="muted">No strong pairs detected.</div>`}

      <h3>Suspicion Received (votes / mentions / accusations directed at each player)</h3>
      <table>
        <thead><tr><th>Player</th><th>Votes received</th><th>Group mentions</th><th>Accusations</th></tr></thead>
        <tbody>${susRows}</tbody>
      </table>
    </section>`;
}

function drawVotingHeatmap() {
  const vb = R.voting_blocs || {};
  const pairs = vb.pairs || {};
  // Build symmetric matrix
  const matrix = {};
  PIDS.forEach(a => { matrix[a] = {}; PIDS.forEach(b => matrix[a][b] = a === b ? 1.0 : null); });
  Object.values(pairs).forEach(p => {
    const [a, b] = p.players;
    matrix[a][b] = matrix[b][a] = p.agreement_rate;
  });
  const container = document.getElementById("vote-heatmap");
  let html = `<table style="border-collapse:separate; border-spacing:2px;"><thead><tr><th></th>${PIDS.map(p => `<th class="small">${escapeHtml(ident(p).real_name)}</th>`).join("")}</tr></thead><tbody>`;
  PIDS.forEach(a => {
    html += `<tr><th class="small" style="text-align:right">${escapeHtml(ident(a).real_name)}</th>`;
    PIDS.forEach(b => {
      const v = matrix[a][b];
      const intensity = v == null ? 0 : v;
      const bg = `rgba(122, 162, 247, ${0.15 + intensity * 0.7})`;
      const txt = v == null ? "—" : (v * 100).toFixed(0) + "%";
      html += `<td class="heatmap-cell" style="background:${bg}; color:#0f1115; font-weight:600">${txt}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  container.innerHTML = html;
}

// ============================================================================
// Section: Clustering (the "who is similar to whom" ask)
// ============================================================================
function renderClustering() {
  const mc = R.message_clustering || {};
  if (mc.note) {
    return `<section id="clustering" style="display:none"><h2>Clustering</h2><div class="muted">${escapeHtml(mc.note)}</div></section>`;
  }

  // Cluster summary
  const summaries = mc.cluster_summaries || {};
  let sumHtml = "";
  for (const [k, groups] of Object.entries(summaries)) {
    sumHtml += `<h3>Cluster assignments at ${escapeHtml(k)}</h3><div class="grid grid-2">`;
    for (const [c, info] of Object.entries(groups)) {
      const members = info.members.map(m => `${escapeHtml(m.real_name)} <span class="muted small">(${m.player_type}${m.model_name ? ", " + m.model_name : ""})</span>`).join("<br>");
      sumHtml += `<div class="kpi" style="align-items:flex-start">
        <div class="l">Cluster ${c} — ${info.n_humans} human, ${info.n_ais} AI</div>
        <div style="margin-top:8px">${members}</div>
      </div>`;
    }
    sumHtml += `</div>`;
  }

  return `
    <section id="clustering" style="display:none">
      <h2>Clustering — message-corpus similarity</h2>
      <div class="muted small">Per-player TF-IDF over ${mc.n_docs || 0} corpora and ${mc.vocabulary_size || 0}-token vocabulary. Method: ${escapeHtml(mc.method || "?")}.</div>

      <h3>Pairwise cosine similarity heatmap</h3>
      <div id="cluster-heatmap"></div>

      ${mc.dendrogram ? `<h3>Dendrogram</h3><svg class="dendrogram" id="svg-dendro" viewBox="0 0 1000 320" preserveAspectRatio="xMidYMid meet"></svg>` : ""}

      ${sumHtml}

      ${R.model_fingerprints && R.model_fingerprints.tsne_coordinates ? `
        <h3>Stylometric fingerprint — t-SNE projection</h3>
        <div class="canvas-wrap"><canvas id="chart-tsne" height="280"></canvas></div>
      ` : ""}
    </section>`;
}

function drawClusterHeatmap() {
  const mc = R.message_clustering || {};
  if (!mc.similarity_matrix) return;
  const players = mc.players || PIDS;
  const sim = mc.similarity_matrix;
  let html = `<table style="border-collapse:separate; border-spacing:2px"><thead><tr><th></th>${players.map(p => `<th class="small">${escapeHtml(ident(p).real_name)}</th>`).join("")}</tr></thead><tbody>`;
  players.forEach(a => {
    html += `<tr><th class="small" style="text-align:right">${escapeHtml(ident(a).real_name)}</th>`;
    players.forEach(b => {
      const v = (sim[a] && sim[a][b]) || 0;
      const bg = `rgba(122, 162, 247, ${0.10 + v * 0.8})`;
      html += `<td class="heatmap-cell" style="background:${bg}; color:#0f1115; font-weight:600">${v.toFixed(2)}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  document.getElementById("cluster-heatmap").innerHTML = html;
}

function drawDendrogram() {
  const dg = R.message_clustering && R.message_clustering.dendrogram;
  if (!dg) return;
  const svg = d3.select("#svg-dendro");
  // Same fix as the network: use viewBox dims, not clientWidth.
  const vb = (svg.attr("viewBox") || "0 0 1000 320").split(/\s+/).map(Number);
  const W = vb[2], H = vb[3];
  svg.selectAll("*").remove();
  const margin = { top: 16, right: 16, bottom: 60, left: 16 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;
  const g = svg.append("g").attr("transform", `translate(${margin.left}, ${margin.top})`);

  const leaves = dg.leaf_player_ids;
  const n = leaves.length;
  const Z = dg.linkage_matrix; // each row: [a, b, dist, n]

  const x = d3.scaleBand().domain(d3.range(n)).range([0, innerW]).padding(0.2);
  const maxDist = d3.max(Z, d => d[2]) || 1;
  const y = d3.scaleLinear().domain([0, maxDist]).range([innerH, 0]);

  // Compute coords for each cluster id (0..n-1 are leaves; n..n+k-1 are merges)
  const cx = {}, cy = {};
  for (let i = 0; i < n; i++) { cx[i] = x(i) + x.bandwidth() / 2; cy[i] = innerH; }
  Z.forEach((row, i) => {
    const a = Math.round(row[0]), b = Math.round(row[1]), d = row[2];
    const ax = cx[a], bx = cx[b], ay = cy[a], by = cy[b];
    const my = y(d);
    g.append("path")
      .attr("d", `M${ax},${ay} L${ax},${my} L${bx},${my} L${bx},${by}`)
      .attr("fill", "none").attr("stroke", "#7aa2f7").attr("stroke-width", 1.5);
    cx[n + i] = (ax + bx) / 2; cy[n + i] = my;
  });
  g.append("g").attr("transform", `translate(0,${innerH})`)
    .selectAll("text").data(leaves).enter().append("text")
    .attr("x", (_, i) => x(i) + x.bandwidth() / 2).attr("y", 14)
    .attr("text-anchor", "middle").attr("fill", "#e6e9ef").attr("font-size", 11)
    .text(pid => ident(pid).real_name);
}

function drawTSNE() {
  const ts = R.model_fingerprints && R.model_fingerprints.tsne_coordinates;
  if (!ts || ts.error) return;
  const points = Object.entries(ts).map(([pid, p]) => ({ pid, ...p }));
  new Chart(document.getElementById("chart-tsne"), {
    type: "scatter",
    data: { datasets: [{
      data: points.map(p => ({ x: p.x, y: p.y, pid: p.pid, name: ident(p.pid).real_name })),
      backgroundColor: points.map(p => p.player_type === "HUMAN" ? getCSS("--human") : getCSS("--ai")),
      pointRadius: 9,
    }]},
    options: { plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: (ctx) => ctx.raw.name + " (" + ctx.raw.pid + ")" } } },
      scales: { x: { ticks: { color: "#98a2b3" }, grid: { color: "#2a313d" } },
                y: { ticks: { color: "#98a2b3" }, grid: { color: "#2a313d" } } },
    },
  });
}

// ============================================================================
// Section: Hints
// ============================================================================
function renderHints() {
  const ha = R.hint_analysis || {};
  const ta = R.topic_adherence || {};
  const adherenceByHint = {};
  (ta.per_hint || []).forEach(h => { adherenceByHint[h.round_number + "/" + h.player_id] = h.adherence_score; });
  const perHint = ha.per_hint || [];
  const byRound = {};
  perHint.forEach(h => { (byRound[h.round_number] = byRound[h.round_number] || []).push(h); });
  let html = "";
  Object.keys(byRound).sort((a,b) => Number(a) - Number(b)).forEach(rn => {
    const rows = byRound[rn].map(h => {
      const adh = adherenceByHint[rn + "/" + h.player_id];
      const impTag = h.hint_was_imposter ? `<span class="pill pill-imp">IMPOSTER</span>` : "";
      return `<tr>
        <td>${labelInline(h.player_id)} ${impTag}</td>
        <td><span class="mono">${escapeHtml(h.hint)}</span></td>
        <td>${h.hint_length_words}w</td>
        <td>${fmtNum(adh)}</td>
      </tr>`;
    }).join("");
    html += `<h3>Round ${rn}</h3>
      <table><thead><tr><th>Player</th><th>Hint</th><th>Length</th><th>Topic adherence</th></tr></thead><tbody>${rows}</tbody></table>`;
  });

  const hs = R.hint_similarity && R.hint_similarity.per_round;
  let simHtml = "";
  if (hs) {
    Object.entries(hs).forEach(([rn, info]) => {
      if (!info.pairs.length) return;
      const top = info.pairs.slice().sort((a,b) => (b.jaccard_similarity||0) - (a.jaccard_similarity||0)).slice(0, 5);
      const rows = top.map(p => `<tr>
        <td>${escapeHtml(ident(p.player_a).real_name)} ↔ ${escapeHtml(ident(p.player_b).real_name)}</td>
        <td>${fmtNum(p.jaccard_similarity)}</td>
        <td class="mono small">${escapeHtml(p.hint_a)} | ${escapeHtml(p.hint_b)}</td>
      </tr>`).join("");
      simHtml += `<h4>Round ${rn} — top hint similarities</h4>
        <table><thead><tr><th>Pair</th><th>Jaccard</th><th>Hints</th></tr></thead><tbody>${rows}</tbody></table>`;
    });
  }

  return `
    <section id="hints" style="display:none">
      <h2>Hints</h2>
      ${html}
      ${simHtml ? `<h3>Hint Similarity (per round)</h3>${simHtml}` : ""}
    </section>`;
}

// ============================================================================
// Section: Classification
// ============================================================================
function renderClassify() {
  const cb = R.classification_breakdown || {};
  if (!cb.per_target) {
    return `<section id="classify" style="display:none"><h2>Classification</h2><div class="muted">No meta-game classifications available.</div></section>`;
  }
  const targetRows = Object.entries(cb.per_target).map(([pid, t]) => {
    const ai = (t.fraction_called_ai || 0) * 100;
    return `<tr>
      <td>${labelInline(pid)}</td>
      <td>${escapeHtml(t.actual_type)}</td>
      <td>${t.n_votes_in}</td>
      <td>${t.n_correct}</td>
      <td><div style="background:linear-gradient(90deg, var(--ai) ${ai}%, var(--human) ${ai}%); height:14px; border-radius:3px"></div>
          <div class="muted small">${ai.toFixed(0)}% called AI</div></td>
    </tr>`;
  }).join("");

  const voterRows = Object.entries(cb.per_voter || {}).map(([pid, v]) => `<tr>
    <td>${labelInline(pid)}</td>
    <td>${v.n_correct}/${v.n_total}</td>
    <td>${fmtPct(v.accuracy)}</td>
  </tr>`).join("");

  // Per-voter detail
  let detailHtml = "";
  for (const [pid, v] of Object.entries(cb.per_voter || {})) {
    const rows = (v.entries || []).map(e => {
      const correct = e.correct == null ? "" : (e.correct ? `<span class="pill pill-good">✓</span>` : `<span class="pill pill-bad">✗</span>`);
      return `<tr>
        <td>${labelInline(e.target_id)}</td>
        <td>${e.guess || "—"}</td>
        <td>${e.confidence != null ? e.confidence : "—"}</td>
        <td>${correct}</td>
      </tr>`;
    }).join("");
    detailHtml += `<details><summary>${labelInline(pid)} — ${v.n_correct}/${v.n_total} correct</summary>
      <table><thead><tr><th>Target</th><th>Guess</th><th>Confidence</th><th>Correct?</th></tr></thead><tbody>${rows}</tbody></table>
    </details>`;
  }

  return `
    <section id="classify" style="display:none">
      <h2>Meta-Game Classification</h2>
      <h3>Per-Target — how many people called each player AI?</h3>
      <table>
        <thead><tr><th>Target</th><th>Actual</th><th>Voters</th><th>Correct</th><th>Distribution (orange = called AI)</th></tr></thead>
        <tbody>${targetRows}</tbody>
      </table>

      <h3>Per-Voter Accuracy</h3>
      <table>
        <thead><tr><th>Voter</th><th>Correct/Total</th><th>Accuracy</th></tr></thead>
        <tbody>${voterRows}</tbody>
      </table>

      <h3>Per-Voter Detail</h3>
      ${detailHtml}
    </section>`;
}

// ============================================================================
// Section: Survey
// ============================================================================
function renderSurvey() {
  const sr = R.survey_responses || {};
  const respondents = sr.per_respondent || [];
  const agg = sr.aggregate_per_question || {};
  const schema = sr.schema || [];

  if (!respondents.length) {
    return `<section id="survey" style="display:none">
      <h2>Post-Game Survey</h2>
      <div class="muted">No survey responses recorded for this session.</div>
    </section>`;
  }

  const meta = `<div class="muted small">
    ${respondents.length} respondent${respondents.length === 1 ? "" : "s"} ·
    ${sr.n_partial} partial submission${sr.n_partial === 1 ? "" : "s"}
  </div>`;

  // Per-respondent cards
  const respondentCards = respondents.map(r => {
    const ident = r.identity;
    const partial = r.is_partial ? `<span class="pill pill-bad">PARTIAL</span>` : `<span class="pill pill-good">FINAL</span>`;
    const submittedAt = r.submitted_at ? r.submitted_at.replace("T", " ").slice(0, 19) : "—";
    const qRows = (r.questions || []).map(q => {
      let answer;
      if (!q.applicable) answer = `<span class="muted small">N/A (q5≠yes)</span>`;
      else if (!q.answered) answer = `<span class="muted small">— skipped</span>`;
      else if (q.type === "checkbox") answer = `<span class="mono">${escapeHtml(JSON.stringify(q.answer))}</span>`;
      else if (q.type === "textarea") answer = `<div style="white-space:pre-wrap; padding:6px 0">${escapeHtml(q.answer)}</div>`;
      else answer = `<span class="mono">${escapeHtml(String(q.answer))}</span>`;
      return `<tr>
        <td class="mono small">${q.id}</td>
        <td>${escapeHtml(q.text)}</td>
        <td>${answer}</td>
      </tr>`;
    }).join("");
    return `<details ${respondents.length === 1 ? "open" : ""}>
      <summary>${labelInline(ident.player_id)} ${partial}
        <span class="muted small">submitted ${escapeHtml(submittedAt)}</span></summary>
      <table>
        <thead><tr><th style="width:40px">#</th><th>Question</th><th>Answer</th></tr></thead>
        <tbody>${qRows}</tbody>
      </table>
    </details>`;
  }).join("");

  // Aggregate per question — likert/yesno/checkbox bars + textarea blocks
  const aggBlocks = schema.map(q => {
    const a = agg[q.id] || {};
    const completion = `<span class="muted small">${a.n_answered || 0}/${a.n_applicable || 0} answered</span>`;
    if (q.type === "textarea") {
      const items = (a.open_ended_responses || []);
      if (!items.length) return `<div class="kpi" style="align-items:flex-start">
        <div class="l">${escapeHtml(q.text)} ${completion}</div>
        <div class="muted small">No open-ended responses.</div>
      </div>`;
      const list = items.map(x => `<div class="msg-row human">
        <div class="msg-meta">${escapeHtml(x.respondent)}</div>
        <div style="white-space:pre-wrap">${escapeHtml(x.answer)}</div>
      </div>`).join("");
      return `<div class="kpi" style="align-items:flex-start">
        <div class="l">${escapeHtml(q.text)} ${completion}</div>
        <div style="margin-top:8px; width:100%">${list}</div>
      </div>`;
    }
    // Distribution bars
    const dist = a.distribution || {};
    let bins;
    if (q.type === "likert") bins = ["1", "2", "3", "4", "5"];
    else if (q.type === "yesno") bins = ["yes", "no"];
    else bins = (q.options || []).map(o => typeof o === "string" ? o : o.value);
    const max = Math.max(1, ...bins.map(b => dist[b] || 0));
    const bars = bins.map(b => {
      const v = dist[b] || 0;
      const w = (v / max) * 100;
      return `<div style="display:flex; align-items:center; gap:8px; margin:2px 0">
        <div class="mono small" style="width:120px">${escapeHtml(b)}</div>
        <div style="background:var(--accent); height:14px; width:${w}%; border-radius:3px; min-width:2px"></div>
        <div class="mono small">${v}</div>
      </div>`;
    }).join("");
    return `<div class="kpi" style="align-items:flex-start">
      <div class="l">${escapeHtml(q.text)} ${completion}</div>
      <div style="margin-top:8px; width:100%">${bars}</div>
    </div>`;
  }).join("");

  return `
    <section id="survey" style="display:none">
      <h2>Post-Game Survey</h2>
      ${meta}

      <h3>Aggregate distribution per question</h3>
      <div class="grid grid-2">${aggBlocks}</div>

      <h3>Per-respondent detail</h3>
      ${respondentCards}
    </section>`;
}

// ============================================================================
// Section: Per-Player Deep Dive
// ============================================================================
function renderDive() {
  const opts = PIDS.map(pid => `<option value="${pid}">${escapeHtml(ident(pid).real_name)} — ${pid}</option>`).join("");
  return `
    <section id="dive" style="display:none">
      <h2>Per-Player Deep Dive</h2>
      <select id="dive-select">${opts}</select>
      <div id="dive-body" style="margin-top:16px"></div>
    </section>`;
}

function setupDive() {
  const sel = document.getElementById("dive-select");
  if (!sel) return;
  sel.addEventListener("change", () => drawDive(sel.value));
  drawDive(sel.value);
}

function drawDive(pid) {
  const ling = (R.linguistic_features && R.linguistic_features.players[pid]) || {};
  const deep = (R.deep_stylometry && R.deep_stylometry.per_player[pid]) || {};
  const conv = (R.conversational_dynamics && R.conversational_dynamics.per_player[pid]) || {};
  const susp = (R.suspicion_received && R.suspicion_received.per_player[pid]) || {};
  const series = ((R.per_round_evolution && R.per_round_evolution.per_player[pid]) || {}).series || [];

  // Combined message timeline (group + sent DMs), chronological
  const groupMsgs = (DATA.raw_group_chat || []).filter(m => m.player_id === pid)
    .map(m => ({ ts: m.ts, kind: "GROUP", round: m.round_number, content: m.content, target: null }));
  const sentDms = (DATA.raw_dms || []).filter(m => m.sender_id === pid)
    .map(m => ({ ts: m.ts, kind: "DM →", round: m.round_number, content: m.content, target: m.recipient_id }));
  const allMsgs = groupMsgs.concat(sentDms).sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));

  // DM partners (sent + received)
  const dmCounts = {};
  (DATA.raw_dms || []).forEach(m => {
    if (m.sender_id === pid && m.recipient_id) {
      const key = m.recipient_id;
      dmCounts[key] = dmCounts[key] || { sent: 0, recv: 0 };
      dmCounts[key].sent += 1;
    }
    if (m.recipient_id === pid && m.sender_id) {
      const key = m.sender_id;
      dmCounts[key] = dmCounts[key] || { sent: 0, recv: 0 };
      dmCounts[key].recv += 1;
    }
  });

  const cls = isHuman(pid) ? "human" : "ai";

  let html = `<div class="grid grid-3">
    <div class="kpi"><div class="v">${ling.msg_count || 0}</div><div class="l">Messages</div></div>
    <div class="kpi"><div class="v">${conv.median_latency_after_others_sec ? conv.median_latency_after_others_sec.toFixed(1) + "s" : "—"}</div><div class="l">Median reply latency</div></div>
    <div class="kpi"><div class="v">${susp.accusations_received || 0}</div><div class="l">Accusations received</div></div>
  </div>

  <h3>Identity</h3>
  <table><tbody>
    <tr><td>Player ID</td><td class="mono">${escapeHtml(pid)}</td></tr>
    <tr><td>Real name</td><td><b>${escapeHtml(ident(pid).real_name)}</b></td></tr>
    <tr><td>Type</td><td>${pillFor(pid)}</td></tr>
    <tr><td>Model</td><td class="mono">${escapeHtml(ident(pid).model_name || "—")}</td></tr>
  </tbody></table>

  <h3>Style profile</h3>
  <table>
    <tbody>
      <tr><td>Avg msg length (words)</td><td>${fmtNum(ling.avg_msg_length_words)}</td></tr>
      <tr><td>Vocab diversity (TTR)</td><td>${fmtNum(ling.vocab_diversity)}</td></tr>
      <tr><td>Slang rate</td><td>${fmtNum(ling.slang_rate)}</td></tr>
      <tr><td>Hedge rate</td><td>${fmtNum(deep.hedge_word_rate)}</td></tr>
      <tr><td>Self-reference rate</td><td>${fmtNum(deep.self_reference_rate)}</td></tr>
      <tr><td>Sentences per msg</td><td>${fmtNum(deep.sentences_per_msg)}</td></tr>
      <tr><td>Burstiness</td><td>${fmtNum(conv.burstiness)}</td></tr>
    </tbody>
  </table>

  <h3>Round-by-round evolution</h3>
  <div class="canvas-wrap"><canvas id="chart-dive-evolution" height="220"></canvas></div>

  <h3>DM partners</h3>
  <table><thead><tr><th>Partner</th><th>Sent to</th><th>Received from</th></tr></thead><tbody>
    ${Object.entries(dmCounts).sort((a,b) => (b[1].sent + b[1].recv) - (a[1].sent + a[1].recv))
      .map(([t, c]) => `<tr><td>${labelInline(t)}</td><td>${c.sent}</td><td>${c.recv}</td></tr>`).join("")}
  </tbody></table>

  <h3>Replies-to-others (group chat heuristic)</h3>
  <table><thead><tr><th>Reply target</th><th>Count</th></tr></thead><tbody>
    ${Object.entries(conv.replies_to_others || {}).sort((a,b) => b[1] - a[1]).map(([t, c]) =>
      `<tr><td>${labelInline(t)}</td><td>${c}</td></tr>`).join("")}
  </tbody></table>

  <h3>Message timeline (${allMsgs.length} messages)</h3>
  <details ${allMsgs.length < 30 ? "open" : ""}><summary>Show all</summary>
    <div style="max-height:560px; overflow-y:auto; padding:8px; background:var(--panel-2); border-radius:6px; border:1px solid var(--border)">
      ${allMsgs.map(m => `<div class="msg-row ${cls}">
        <div class="msg-meta">${escapeHtml((m.ts || "").slice(11, 19))} · R${m.round || "?"} · ${m.kind}${m.target ? " " + escapeHtml(ident(m.target).real_name) : ""}</div>
        <div>${escapeHtml(m.content || "")}</div>
      </div>`).join("")}
    </div>
  </details>`;

  document.getElementById("dive-body").innerHTML = html;

  if (series.length) {
    const labels = series.map(s => "R" + s.round_number);
    new Chart(document.getElementById("chart-dive-evolution"), {
      type: "line",
      data: { labels, datasets: [
        { label: "msg_count", data: series.map(s => s.msg_count || 0), borderColor: getCSS("--accent"), tension: 0.3 },
        { label: "hedge_rate", data: series.map(s => (s.hedge_rate || 0) * 100), borderColor: getCSS("--ai"), tension: 0.3 },
        { label: "question_rate", data: series.map(s => (s.question_rate || 0) * 100), borderColor: getCSS("--human"), tension: 0.3 },
      ]},
      options: { plugins: { legend: { labels: { color: "#e6e9ef" } } },
        scales: { x: { ticks: { color: "#98a2b3" } }, y: { ticks: { color: "#98a2b3" } } } },
    });
  }
}

// ============================================================================
// Boot
// ============================================================================
const main = document.getElementById("main");
main.innerHTML = renderOverview() + renderGame() + renderLinguistic() + renderNetwork() +
  renderVoting() + renderClustering() + renderHints() + renderClassify() + renderSurvey() + renderDive();

// Boot only the Overview section. Other sections lazy-init the first time
// their tab is opened (see SECTION_DRAWERS / showSection above) so that no
// chart or SVG is ever drawn into a `display:none` parent.
showSection("overview");
</script>
</body>
</html>
"""
