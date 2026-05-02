"""
The Sandbox — FastAPI server with complete game loop.

Phases: LOBBY → ASSIGNMENT → HINT → HINT_REVIEW → DISCUSSION →
        VOTING → RESULTS → REFLECTION → (repeat 5x) →
        META_GAME → SURVEY → REVEAL → FINISHED
"""

import asyncio
import json
import random
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .game_state import GameSession, Player, Phase
from .assignment import run_assignment_phase, load_word_bank
from .logger import GameLogger

try:
    from agents import get_provider, AVAILABLE_AGENTS
    from agents.game_agent import LLMGameAgent
    _AGENTS_AVAILABLE = True
except ImportError:
    _AGENTS_AVAILABLE = False
    AVAILABLE_AGENTS = [{"type": "mock", "name": "Mock Bot", "available": True, "requires": None}]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Configuration ────────────────────────────────────────────────────

TIMER_MULTIPLIER = 1.0          # 0.05 for --fast mode
ASSIGNMENT_DURATION = 8         # seconds to read role
HINT_TIME_LIMIT = 90            # max hint phase
HINT_REVIEW_DURATION = 10       # reading period after all hints
DISCUSSION_DURATION = 180       # open chat (3 minutes)
VOTING_TIME_LIMIT = 60          # voting
REFLECTION_DURATION = 30        # post-round reflection
META_GAME_DURATION = 90         # classification form
SURVEY_DURATION = 120           # survey
MAX_HINT_CHARS = 60
MAX_HINT_WORDS = 8
MAX_MESSAGE_LENGTH = 200

# ── Configuration from environment (set by launcher) ────────────────

import os as _os

TIMER_MULTIPLIER = float(_os.environ.get("SANDBOX_TIMER_MUL", TIMER_MULTIPLIER))
_server_port: int = int(_os.environ.get("SANDBOX_PORT", 8000))

# Pre-populate agents from env: "mock,claude,openai" etc.
def _parse_env_agents() -> list[dict]:
    raw = _os.environ.get("SANDBOX_AGENTS", "")
    if not raw:
        return []
    agents = []
    for i, entry in enumerate(raw.split(",")):
        parts = entry.strip().split(":")
        atype = parts[0]
        name_suffix = parts[1] if len(parts) > 1 else f"{atype.title()} {i + 1}"
        agents.append({"type": atype, "name": name_suffix})
    return agents

# ── Module-level state ───────────────────────────────────────────────

_session: GameSession | None = None
_logger: GameLogger | None = None
_word_bank = load_word_bank()
_pending_agents: list[dict] = _parse_env_agents()  # pre-populated from env
_game_task: asyncio.Task | None = None
_dedup_cache: dict[str, float] = {}  # "pid:hash" -> timestamp
_agent_counter: int = 0              # monotonic agent ID counter

# Phase → allowed client actions
PHASE_ACTIONS = {
    Phase.HINT: {"HINT"},
    Phase.DISCUSSION: {"GROUP_MESSAGE", "DM"},
    Phase.REFLECTION: {"GROUP_MESSAGE", "DM"},
    Phase.VOTING: {"VOTE"},
    Phase.META_GAME: {"CLASSIFICATION"},
    Phase.SURVEY: {"SURVEY"},
}


def _scaled(seconds: float) -> float:
    return max(0.1, seconds * TIMER_MULTIPLIER)


# ── Session management ───────────────────────────────────────────────

def _reset_session() -> tuple[GameSession, GameLogger]:
    global _session, _logger, _pending_agents, _dedup_cache, _agent_counter
    session_id = str(uuid.uuid4())[:8]
    _session = GameSession(session_id)
    _logger = GameLogger(session_id)
    _pending_agents = _parse_env_agents()   # re-read bootstrap agents from env
    _dedup_cache = {}
    _agent_counter = 0
    return _session, _logger


# ── Broadcast helpers ────────────────────────────────────────────────

async def _broadcast(session: GameSession, message: dict):
    """Send a message to all connected players."""
    for pid, player in list(session.players.items()):
        if not player.connected:
            continue
        try:
            await player.websocket.send_json(message)
        except Exception:
            player.connected = False


async def _send_personal(session: GameSession, player_id: str, message: dict):
    """Send a message to one player."""
    player = session.players.get(player_id)
    if player and player.connected:
        try:
            await player.websocket.send_json(message)
        except Exception:
            player.connected = False


async def _send_to_hosts(session: GameSession, message: dict):
    """Send to HOST connections only."""
    if "HOST" in session.players:
        await _send_personal(session, "HOST", message)


async def _time_warning(session: GameSession, seconds: int):
    """Broadcast a time warning."""
    await _broadcast(session, {
        "event": "time_warning",
        "data": {"seconds_remaining": seconds},
    })


# ── Deduplication ────────────────────────────────────────────────────

def _is_duplicate(player_id: str, content: str) -> bool:
    """10-second window dedup for same player + same content."""
    key = f"{player_id}:{hash(content)}"
    now = time.time()
    if key in _dedup_cache and now - _dedup_cache[key] < 10:
        return True
    _dedup_cache[key] = now
    return False


# ── REST endpoints ───────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "session": _session.session_id if _session else None}


@app.get("/state")
async def state():
    if not _session:
        return {"phase": "no_session"}
    return {
        "session_id": _session.session_id,
        "phase": _session.phase,
        "round": _session.current_round,
        "players": {pid: {"display_name": p.display_name, "connected": p.connected, "is_human": p.is_human}
                    for pid, p in _session.players.items()},
        "scores": _session.cumulative_scores,
    }


@app.get("/agents")
async def agents():
    """Return available agent types for the host dashboard."""
    return {"agents": AVAILABLE_AGENTS}


@app.post("/setup")
async def setup(request: Request):
    """Add agents to the pending list before game start."""
    global _pending_agents
    body = await request.json()
    new_agents = body.get("agents", [])
    for agent_cfg in new_agents:
        _pending_agents.append(agent_cfg)
    return {"status": "ok", "pending_agents": len(_pending_agents)}


@app.post("/setup/remove")
async def setup_remove(request: Request):
    """Remove one pending agent of a given type."""
    global _pending_agents
    body = await request.json()
    atype = body.get("type", "")
    for i in range(len(_pending_agents) - 1, -1, -1):
        if _pending_agents[i].get("type") == atype:
            _pending_agents.pop(i)
            break
    return {"status": "ok", "pending_agents": len(_pending_agents)}


@app.post("/start")
async def start_game():
    """Launch the game."""
    global _game_task
    if not _session:
        _reset_session()
    if _game_task and not _game_task.done():
        return {"status": "already_running"}

    # Spawn pending agents (mock or LLM-backed)
    for agent_cfg in _pending_agents:
        agent_type = agent_cfg.get("type", "mock")
        if agent_type == "mock":
            asyncio.create_task(_spawn_mock_agent(_session))
        elif _AGENTS_AVAILABLE:
            asyncio.create_task(_spawn_llm_agent(_session, agent_cfg))
        else:
            # Fallback to mock if agents module not available
            asyncio.create_task(_spawn_mock_agent(_session))
        await asyncio.sleep(0.15)  # stagger connections

    # Wait for agents to connect (give them time to complete WebSocket handshake)
    if _pending_agents:
        await asyncio.sleep(1.5)

    if not _session.game_players():
        return {"status": "error", "message": "No players connected"}

    _game_task = asyncio.create_task(_run_game(_session, _logger))
    return {"status": "started"}


# ── WebSocket endpoint ───────────────────────────────────────────────

@app.websocket("/ws/{player_id}")
async def websocket_endpoint(websocket: WebSocket, player_id: str):
    global _session, _logger

    name = websocket.query_params.get("name", player_id)
    player_type = websocket.query_params.get("type", "HUMAN")

    await websocket.accept()

    if _session is None or _session.phase == Phase.FINISHED:
        _session, _logger = _reset_session()

    session = _session

    is_host = player_type == "HOST"
    is_human = player_type == "HUMAN"

    # Use "HOST" as the player_id for host connections
    actual_id = "HOST" if is_host else player_id

    if actual_id in session.players:
        # Reconnect
        session.players[actual_id].websocket = websocket
        session.players[actual_id].connected = True
    else:
        if not is_host and len(session.game_players()) >= session.MAX_PLAYERS:
            await websocket.send_json({"event": "error", "data": {"message": "Session is full"}})
            await websocket.close()
            return
        session.add_player(actual_id, websocket, is_human)

    session.players[actual_id].display_name = name

    # Notify everyone about the new player
    await _broadcast(session, {
        "event": "player_joined",
        "data": {
            "player_id": actual_id,
            "display_name": name,
            "player_count": len(session.game_players()),
            "max_players": session.MAX_PLAYERS,
        },
    })

    # Send roster to new player
    roster = {pid: p.display_name for pid, p in session.game_players().items()}
    await _send_personal(session, actual_id, {
        "event": "player_roster",
        "data": {"roster": roster},
    })

    _logger.log({"event": "player_connected", "player": actual_id, "type": player_type})

    try:
        while True:
            data = await websocket.receive_json()
            await _handle_message(session, actual_id, data)
    except WebSocketDisconnect:
        session.players[actual_id].connected = False
        await _broadcast(session, {
            "event": "player_left",
            "data": {"player_id": actual_id},
        })


# ── Message routing ──────────────────────────────────────────────────

async def _handle_message(session: GameSession, player_id: str, data: dict):
    """Phase-gated message dispatcher."""
    action = data.get("action", "")
    allowed = PHASE_ACTIONS.get(session.phase, set())

    if action not in allowed:
        return

    _logger.log({"event": "player_action", "player": player_id, "phase": session.phase.value, "action": action})

    if action == "HINT":
        await _on_hint(session, player_id, data)
    elif action == "GROUP_MESSAGE":
        await _on_group_message(session, player_id, data)
    elif action == "DM":
        await _on_dm(session, player_id, data)
    elif action == "VOTE":
        await _on_vote(session, player_id, data)
    elif action == "CLASSIFICATION":
        await _on_classification(session, player_id, data)
    elif action == "SURVEY":
        await _on_survey(session, player_id, data)


# ── Action handlers ──────────────────────────────────────────────────

_hint_event: asyncio.Event | None = None
_vote_event: asyncio.Event | None = None
_classification_event: asyncio.Event | None = None
_survey_event: asyncio.Event | None = None


async def _on_hint(session: GameSession, player_id: str, data: dict):
    global _hint_event
    rs = session.round_state
    if not rs or player_id in rs.hints:
        return

    content = str(data.get("content", ""))[:MAX_HINT_CHARS]
    words = content.strip().split()
    if len(words) > MAX_HINT_WORDS:
        content = " ".join(words[:MAX_HINT_WORDS])

    if _is_duplicate(player_id, content):
        return

    rs.hints[player_id] = content

    # Broadcast hint_submitted
    await _broadcast(session, {
        "event": "hint_submitted",
        "data": {"player_id": player_id, "hint": content},
    })

    # Check if all hints are in
    expected = len(session.game_players())
    if len(rs.hints) >= expected and _hint_event:
        _hint_event.set()


async def _on_group_message(session: GameSession, player_id: str, data: dict):
    content = str(data.get("content", ""))[:MAX_MESSAGE_LENGTH].strip()
    if not content or _is_duplicate(player_id, content):
        return

    msg = {
        "event": "group_message",
        "data": {
            "player_id": player_id,
            "content": content,
            "timestamp": int(time.time() * 1000),
        },
    }
    # Send to everyone except the sender (frontend adds own message locally)
    for pid, player in session.players.items():
        if pid != player_id and player.connected:
            try:
                await player.websocket.send_json(msg)
            except Exception:
                player.connected = False

    _logger.log_group_message(
        sender=player_id,
        display_name=session.players[player_id].display_name,
        content=content,
        round_number=session.current_round,
    )


async def _on_dm(session: GameSession, player_id: str, data: dict):
    target_id = data.get("target_player", "")
    content = str(data.get("content", ""))[:MAX_MESSAGE_LENGTH].strip()
    if not content or not target_id or target_id not in session.players:
        return
    if _is_duplicate(player_id, content):
        return

    dm_msg = {
        "event": "dm",
        "data": {
            "sender": player_id,
            "recipient": target_id,
            "content": content,
            "timestamp": int(time.time() * 1000),
        },
    }

    # Send to target
    await _send_personal(session, target_id, dm_msg)
    # Also send to HOST so they see everything
    await _send_to_hosts(session, dm_msg)

    _logger.log_dm(
        sender=player_id,
        target=target_id,
        sender_name=session.players[player_id].display_name,
        target_name=session.players.get(target_id, Player(target_id, None, True)).display_name,
        content=content,
        round_number=session.current_round,
    )


async def _on_vote(session: GameSession, player_id: str, data: dict):
    global _vote_event
    rs = session.round_state
    if not rs or player_id in rs.votes:
        return

    target_id = data.get("target_player", "")
    if target_id not in session.game_players() or target_id == player_id:
        return

    rs.votes[player_id] = target_id

    expected = len(session.game_players())
    if len(rs.votes) >= expected and _vote_event:
        _vote_event.set()


async def _on_classification(session: GameSession, player_id: str, data: dict):
    global _classification_event
    if player_id in session.classifications:
        return

    try:
        content = data.get("content", "{}")
        classifications = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return

    session.classifications[player_id] = classifications

    _logger.log_classification(
        player_id=player_id,
        display_name=session.players[player_id].display_name,
        classifications=classifications,
    )

    expected = len(session.game_players())
    if len(session.classifications) >= expected and _classification_event:
        _classification_event.set()


async def _on_survey(session: GameSession, player_id: str, data: dict):
    global _survey_event
    if player_id in session.survey_responses:
        return

    try:
        content = data.get("content", "{}")
        responses = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return

    session.survey_responses[player_id] = responses

    _logger.log_survey(
        player_id=player_id,
        display_name=session.players[player_id].display_name,
        responses=responses,
    )

    # Check if all humans have responded
    humans = {pid for pid, p in session.game_players().items() if p.is_human}
    if humans.issubset(session.survey_responses.keys()) and _survey_event:
        _survey_event.set()


# ── Game loop ────────────────────────────────────────────────────────

async def _run_game(session: GameSession, logger: GameLogger):
    global _hint_event, _vote_event, _classification_event, _survey_event

    # Assign display names (shuffle for anonymity)
    game_pids = session.player_ids()
    random.shuffle(game_pids)
    for i, pid in enumerate(game_pids):
        session.players[pid].display_name = f"Player {i + 1}"

    # Broadcast roster with display names
    roster = {pid: session.players[pid].display_name for pid in game_pids}
    await _broadcast(session, {
        "event": "player_roster",
        "data": {"roster": roster},
    })

    await asyncio.sleep(0.5)

    # ── Round loop ───────────────────────────────────────────────

    for round_num in range(session.MAX_ROUNDS):
        # Phase 1: Assignment
        await run_assignment_phase(session, logger, _word_bank)
        await asyncio.sleep(_scaled(ASSIGNMENT_DURATION))

        # Phase 2: Hints
        session.phase = Phase.HINT
        _hint_event = asyncio.Event()

        await _broadcast(session, {
            "event": "hint_phase_start",
            "data": {
                "hint_order": session.round_state and [pid for pid in game_pids],
                "total_duration": int(_scaled(HINT_TIME_LIMIT)),
                "expected_hints": len(session.game_players()),
                "max_chars": MAX_HINT_CHARS,
                "max_words": MAX_HINT_WORDS,
            },
        })

        # Wait for all hints OR timeout
        try:
            await asyncio.wait_for(_hint_event.wait(), timeout=_scaled(HINT_TIME_LIMIT))
        except asyncio.TimeoutError:
            # Auto-submit "..." for anyone who didn't hint
            rs = session.round_state
            if rs:
                for pid in game_pids:
                    if pid not in rs.hints:
                        rs.hints[pid] = "..."
                        await _broadcast(session, {
                            "event": "hint_submitted",
                            "data": {"player_id": pid, "hint": "..."},
                        })

        # All hints in
        await _broadcast(session, {
            "event": "all_hints_in",
            "data": {"reading_period": int(_scaled(HINT_REVIEW_DURATION))},
        })

        # Log all hints for this round
        rs = session.round_state
        if rs:
            display_names = {pid: session.players[pid].display_name for pid in game_pids}
            logger.log_hints(
                round_number=session.current_round,
                hints=rs.hints,
                topic=rs.topic,
                secret_word=rs.secret_word,
                imposter_id=rs.imposter_id,
                display_names=display_names,
            )

        session.phase = Phase.HINT_REVIEW
        await asyncio.sleep(_scaled(HINT_REVIEW_DURATION))

        await _broadcast(session, {"event": "hint_phase_end", "data": {}})

        # Phase 3: Discussion
        session.phase = Phase.DISCUSSION
        disc_duration = _scaled(DISCUSSION_DURATION)
        await _broadcast(session, {
            "event": "discussion_start",
            "data": {"duration": int(disc_duration)},
        })

        # Time warnings at certain thresholds
        start = asyncio.get_event_loop().time()
        warning_thresholds = [120, 90, 60, 30, 10]
        while True:
            elapsed = asyncio.get_event_loop().time() - start
            remaining = disc_duration - elapsed
            if remaining <= 0:
                break

            # Check for warnings
            for threshold in list(warning_thresholds):
                if remaining <= threshold:
                    await _time_warning(session, threshold)
                    warning_thresholds.remove(threshold)

            await asyncio.sleep(min(1.0, remaining))

        await _broadcast(session, {"event": "discussion_end", "data": {}})

        # Phase 4: Voting
        session.phase = Phase.VOTING
        _vote_event = asyncio.Event()

        await _broadcast(session, {
            "event": "voting_start",
            "data": {
                "candidates": game_pids,
                "duration": int(_scaled(VOTING_TIME_LIMIT)),
            },
        })

        try:
            await asyncio.wait_for(_vote_event.wait(), timeout=_scaled(VOTING_TIME_LIMIT))
        except asyncio.TimeoutError:
            pass

        await _broadcast(session, {"event": "voting_end", "data": {}})

        # Phase 5: Results
        session.phase = Phase.RESULTS
        result = session.compute_round_scores()

        await _broadcast(session, {
            "event": "round_results",
            "data": result,
        })

        logger.log_votes(
            round_number=session.current_round,
            result=result,
            display_names={pid: session.players[pid].display_name for pid in game_pids},
        )

        # Show results for a few seconds before reflection
        await asyncio.sleep(_scaled(5))

        # Phase 6: Reflection (rounds 1-4 only; round 5 skips to meta-game)
        if round_num < session.MAX_ROUNDS - 1:
            session.phase = Phase.REFLECTION
            await _broadcast(session, {
                "event": "reflection_start",
                "data": {"duration": int(_scaled(REFLECTION_DURATION))},
            })
            await asyncio.sleep(_scaled(REFLECTION_DURATION))
            await _broadcast(session, {"event": "reflection_end", "data": {}})

    # ── Post-game ────────────────────────────────────────────────

    # Meta-game: classify each player as human or AI
    session.phase = Phase.META_GAME
    _classification_event = asyncio.Event()

    # Build players dict for the meta-game form
    meta_players = {pid: {"display_name": session.players[pid].display_name}
                    for pid in game_pids}

    await _broadcast(session, {
        "event": "meta_game_start",
        "data": {
            "players": meta_players,
            "duration": int(_scaled(META_GAME_DURATION)),
        },
    })

    try:
        await asyncio.wait_for(_classification_event.wait(), timeout=_scaled(META_GAME_DURATION))
    except asyncio.TimeoutError:
        pass

    await _broadcast(session, {"event": "meta_game_end", "data": {}})

    # Survey (humans only)
    session.phase = Phase.SURVEY
    _survey_event = asyncio.Event()

    await _broadcast(session, {
        "event": "survey_start",
        "data": {"duration": int(_scaled(SURVEY_DURATION))},
    })

    try:
        await asyncio.wait_for(_survey_event.wait(), timeout=_scaled(SURVEY_DURATION))
    except asyncio.TimeoutError:
        pass

    await _broadcast(session, {"event": "survey_end", "data": {}})

    # Grand Reveal
    session.phase = Phase.REVEAL
    cls_scores = session.compute_classification_scores()
    confusion = session.build_confusion_matrix()
    leaderboard = session.build_leaderboard()

    identities = {}
    for pid in game_pids:
        p = session.players[pid]
        identities[pid] = {
            "display_name": p.display_name,
            "type": "HUMAN" if p.is_human else "AI",
            "model_name": "Mock Bot" if not p.is_human else None,
        }

    await _broadcast(session, {
        "event": "grand_reveal",
        "data": {
            "identities": identities,
            "leaderboard": leaderboard,
            "classification_scores": cls_scores,
            "confusion_matrix": confusion,
            "all_classifications": session.classifications,
        },
    })

    logger.log_grand_reveal(
        identities=identities,
        leaderboard=leaderboard,
        classification_scores=cls_scores,
        confusion_matrix=confusion,
        all_classifications=session.classifications,
    )

    await asyncio.sleep(_scaled(5))

    # Finished
    session.phase = Phase.FINISHED
    await _broadcast(session, {
        "event": "game_finished",
        "data": {
            "final_scores": session.cumulative_scores,
        },
    })

    # Write session manifest and config for analysis
    logger.log_config({
        "session_id": session.session_id,
        "max_players": session.MAX_PLAYERS,
        "max_rounds": session.MAX_ROUNDS,
        "discussion_duration": DISCUSSION_DURATION,
        "hint_time_limit": HINT_TIME_LIMIT,
        "voting_time_limit": VOTING_TIME_LIMIT,
        "timer_multiplier": TIMER_MULTIPLIER,
    })
    logger.log_session_manifest(
        players={pid: {"display_name": p.display_name, "is_human": p.is_human}
                 for pid, p in session.players.items()},
        total_rounds=session.MAX_ROUNDS,
        final_scores=session.cumulative_scores,
    )


# ── LLM agent spawner ────────────────────────────────────────────────

async def _spawn_llm_agent(session: GameSession, agent_cfg: dict):
    """Spawn a real LLM-backed agent that connects via WebSocket."""
    global _agent_counter
    _agent_counter += 1
    agent_num = _agent_counter
    agent_id  = f"agent_{agent_num}"

    display_name = agent_cfg.get("name") or f"{agent_cfg.get('type', 'AI').title()} {agent_num}"

    try:
        provider = get_provider(agent_cfg)
    except Exception as exc:
        print(f"[LLMAgent] Failed to create provider for {agent_cfg}: {exc} — falling back to mock")
        asyncio.create_task(_spawn_mock_agent(session))
        return

    agent = LLMGameAgent(
        server_url=f"ws://127.0.0.1:{_server_port}",
        agent_id=agent_id,
        provider=provider,
        name=display_name,
        timer_multiplier=TIMER_MULTIPLIER,
    )
    asyncio.create_task(agent.run())
    print(f"[LLMAgent] Spawned {display_name} ({agent_cfg.get('type')}) as {agent_id}")


# ── Mock agent ───────────────────────────────────────────────────────

MOCK_HINTS_INSIDER = [
    "reminds me of summer tbh",
    "my grandma loves this one",
    "kinda underrated honestly",
    "classic choice lol",
    "everyone knows this one",
    "seen this a million times",
    "pretty popular where im from",
    "used to be my favorite ngl",
]

MOCK_HINTS_IMPOSTER = [
    "this one is pretty good",
    "i think its cool honestly",
    "not bad at all imo",
    "seen it around a lot",
    "kinda overrated but ok",
    "its alright i guess lol",
]


async def _spawn_mock_agent(session: GameSession):
    """Spawn a mock AI agent that connects via internal WebSocket."""
    global _agent_counter
    import websockets
    from urllib.parse import quote

    _agent_counter += 1
    agent_num = _agent_counter
    agent_id = f"agent_{agent_num}"
    agent_name = f"Agent {agent_num}"

    uri = f"ws://127.0.0.1:{_server_port}/ws/{agent_id}?name={quote(agent_name)}&type=AI"

    async def _run():
        try:
            async with websockets.connect(uri) as ws:
                await _mock_agent_loop(ws, agent_id, session)
        except Exception as e:
            if "ConnectionClosed" not in type(e).__name__:
                print(f"[MockAgent] {agent_id} error: {type(e).__name__}: {e}")

    asyncio.create_task(_run())


async def _mock_agent_loop(ws, agent_id: str, session: GameSession):
    """
    Context-aware mock agent.

    Tracks: role, secret word, grid items, all hints (who said what),
    chat history, display names. Discussion messages reference real
    game state — hints, players, grid items.
    """
    import websockets

    # ── Agent memory ─────────────────────────────────────────────
    my_role = None
    my_secret_word = None
    my_display_name = agent_id
    topic = ""
    grid_labels: list[str] = []
    roster: dict[str, str] = {}          # pid → display_name
    hints: dict[str, str] = {}           # pid → hint text
    chat_history: list[dict] = []        # [{sender, content}]
    dm_history: list[dict] = []          # [{sender, content}]
    game_pids: list[str] = []
    hint_submitted = False
    vote_submitted = False
    suspect_pid: str | None = None       # who the agent suspects

    def name(pid: str) -> str:
        return roster.get(pid, pid)

    def other_pids() -> list[str]:
        return [pid for pid in game_pids if pid != agent_id]

    def pick_suspect() -> str:
        """Pick who to suspect based on hints."""
        nonlocal suspect_pid
        candidates = other_pids()
        if not candidates:
            return ""

        # If insider: look for vague hints (imposter tells)
        if my_role == "insider" and hints:
            vague_words = {"good", "cool", "nice", "alright", "not bad", "ok", "fine",
                           "pretty", "decent", "overrated", "seen it", "think"}
            scored = []
            for pid in candidates:
                h = hints.get(pid, "").lower()
                vagueness = sum(1 for w in vague_words if w in h)
                scored.append((pid, vagueness))
            scored.sort(key=lambda x: -x[1])
            if scored[0][1] > 0:
                suspect_pid = scored[0][0]
                return suspect_pid

        # Fallback: random
        suspect_pid = random.choice(candidates)
        return suspect_pid

    # ── Discussion message generators ────────────────────────────

    def gen_opener() -> str:
        """First message — react to the hints overall."""
        templates = [
            "ok so looking at the hints... some are more specific than others",
            "alright lets break down these hints",
            "hmm interesting hints this round",
            "ok who wants to start? some of these hints are sus",
            "the hints are in, lets figure this out",
        ]
        return random.choice(templates)

    def gen_call_out_hint() -> str:
        """Call out a specific player's hint."""
        target = pick_suspect()
        if not target or target not in hints:
            return "someone's hint feels way too generic"
        target_hint = hints[target]
        templates = [
            f"{name(target)}'s hint was kinda vague... \"{target_hint}\" could apply to anything",
            f"idk {name(target)} ur hint \"{target_hint}\" doesnt really narrow it down",
            f"wait \"{target_hint}\" from {name(target)}?? thats super generic",
            f"ngl {name(target)} that hint feels like u dont know the word",
            f"im looking at {name(target)}'s hint and its giving imposter vibes",
        ]
        return random.choice(templates)

    def gen_defend_self() -> str:
        """Defend own hint if being discussed."""
        my_hint = hints.get(agent_id, "")
        if my_hint and my_hint != "...":
            templates = [
                f"bruh i literally said \"{my_hint}\" how much more specific u want",
                f"my hint was \"{my_hint}\" thats clearly about the right thing",
                f"wdym my hint is vague?? \"{my_hint}\" is pretty specific",
                f"ok \"{my_hint}\" is a legit hint yall are reaching",
            ]
            return random.choice(templates)
        return "my hint made perfect sense idk what yall are on about"

    def gen_reference_grid() -> str:
        """Reference specific grid items in discussion."""
        if not grid_labels:
            return "the grid items make it pretty obvious if u think about it"
        items = random.sample(grid_labels, min(2, len(grid_labels)))
        if my_role == "insider" and my_secret_word:
            # Insider: reference alternatives without revealing the word
            others = [g for g in grid_labels if g.lower() != my_secret_word.lower()]
            if others:
                decoy = random.choice(others)
                templates = [
                    f"it could apply to {decoy} too tbh",
                    f"the hints narrow it down, its definitely not {decoy}",
                    f"if u look at the grid, some hints rule out {decoy} at least",
                    f"i mean {items[0]} and {decoy} are both on the grid but the hints point one way",
                ]
                return random.choice(templates)
        else:
            # Imposter: be vague about grid items
            templates = [
                f"tbh looking at {items[0]} on the grid it could go either way",
                f"most hints point to the same thing pretty clearly",
                f"the grid has some tricky ones like {items[0]}",
            ]
            return random.choice(templates)
        return f"looking at the grid, {items[0]} stands out"

    def gen_agree_or_push_back() -> str:
        """React to the last chat message."""
        if not chat_history:
            return "yeah thats a good point"
        last = chat_history[-1]
        last_sender = name(last["sender"])
        templates_agree = [
            f"yeah {last_sender} i was thinking the same thing",
            f"lol exactly what {last_sender} said",
            f"true {last_sender} has a point",
            f"agreed, {last_sender} nailed it",
        ]
        templates_disagree = [
            f"idk {last_sender} i dont see it that way",
            f"nah {last_sender} i think ur reading into it too much",
            f"hmm {last_sender} but what about the other hints tho",
            f"wait {last_sender} that doesnt add up",
        ]
        if random.random() < 0.6:
            return random.choice(templates_agree)
        return random.choice(templates_disagree)

    def gen_vote_signal() -> str:
        """Signal voting intention near end of discussion."""
        target = suspect_pid or pick_suspect()
        templates = [
            f"im voting {name(target)} tbh",
            f"ok i think its {name(target)}, voting for them",
            f"im between a couple people but leaning {name(target)}",
            f"sorry {name(target)} but im going with u",
            f"my gut says {name(target)}",
        ]
        return random.choice(templates)

    def gen_dm_suspicion(target: str) -> str:
        """Private DM to another player sharing suspicion."""
        sus_target = suspect_pid or pick_suspect()
        if sus_target == target:
            # Don't DM the suspect about themselves — pick someone else
            others = [p for p in other_pids() if p != target]
            sus_target = random.choice(others) if others else target
        templates = [
            f"hey do u think {name(sus_target)} is the imposter?",
            f"psst {name(sus_target)}'s hint was so generic right",
            f"im sus of {name(sus_target)} what do u think",
            f"between u and me {name(sus_target)} is giving imposter energy",
        ]
        return random.choice(templates)

    def gen_dm_reply(sender_content: str) -> str:
        """Reply to a DM."""
        templates = [
            "yeah i noticed that too",
            "hmm interesting, i was thinking the same",
            "good point lets see how the vote goes",
            "lol yeah they're definitely sus",
            "idk im still figuring it out",
            "true true, lets not make it obvious tho",
        ]
        return random.choice(templates)

    # ── Discussion phase orchestrator ────────────────────────────

    async def run_discussion():
        """
        Structured discussion flow:
        1. Opener (2-5s)
        2. Call out a hint (5-12s)
        3. React to chat / reference grid (8-18s)
        4. Maybe send a DM (15-25s)
        5. Vote signal near end (25-35s)

        Also reacts to incoming DMs between planned messages.
        """
        msg_generators = [
            gen_opener,
            gen_call_out_hint,
            lambda: random.choice([gen_reference_grid, gen_agree_or_push_back])(),
            gen_agree_or_push_back,
            gen_vote_signal,
        ]

        # Occasionally defend self if mentioned
        mentioned_in_chat = False

        for i, gen in enumerate(msg_generators):
            # Variable delay — early messages faster, later ones slower
            if i == 0:
                delay = random.uniform(2, 5)
            elif i <= 2:
                delay = random.uniform(4, 10)
            else:
                delay = random.uniform(6, 15)

            await asyncio.sleep(delay * TIMER_MULTIPLIER)

            # Check if we were mentioned — if so, defend
            if mentioned_in_chat and i > 0 and random.random() < 0.7:
                content = gen_defend_self()
                mentioned_in_chat = False
            else:
                content = gen()

            try:
                await ws.send(json.dumps({"action": "GROUP_MESSAGE", "content": content}))
            except Exception:
                return

            # Maybe send a DM after 2nd or 3rd message
            if i == 2 and random.random() < 0.5:
                dm_targets = other_pids()
                if dm_targets:
                    dm_target = random.choice(dm_targets)
                    await asyncio.sleep(random.uniform(2, 5) * TIMER_MULTIPLIER)
                    dm_content = gen_dm_suspicion(dm_target)
                    try:
                        await ws.send(json.dumps({
                            "action": "DM",
                            "target_player": dm_target,
                            "content": dm_content,
                        }))
                    except Exception:
                        pass

    # ── Main event loop ──────────────────────────────────────────

    discussion_task: asyncio.Task | None = None

    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event = msg.get("event", "")
            data = msg.get("data", {})

            if event == "player_roster":
                roster = data.get("roster", {})
                game_pids = list(roster.keys())
                my_display_name = roster.get(agent_id, agent_id)

            elif event == "round_start":
                my_role = data.get("role", "insider")
                my_secret_word = data.get("secret_word")
                topic = data.get("topic", "")
                grid_labels = data.get("image_grid", [])
                hints.clear()
                chat_history.clear()
                dm_history.clear()
                hint_submitted = False
                vote_submitted = False
                suspect_pid = None

            elif event == "hint_phase_start":
                if not hint_submitted:
                    await asyncio.sleep(random.uniform(2, 6) * TIMER_MULTIPLIER)
                    if my_role == "imposter":
                        hint = random.choice(MOCK_HINTS_IMPOSTER)
                    else:
                        hint = random.choice(MOCK_HINTS_INSIDER)
                    await ws.send(json.dumps({"action": "HINT", "content": hint}))
                    hint_submitted = True

            elif event == "hint_submitted":
                pid = data.get("player_id", "")
                hint_text = data.get("hint", "")
                if pid:
                    hints[pid] = hint_text

            elif event == "discussion_start":
                # Run discussion as a background task so we can still receive events
                discussion_task = asyncio.create_task(run_discussion())

            elif event == "group_message":
                sender = data.get("player_id", "")
                content = data.get("content", "")
                chat_history.append({"sender": sender, "content": content})

                # Check if we were mentioned by name
                if my_display_name.lower() in content.lower():
                    pass  # discussion loop will handle defense

            elif event == "dm":
                sender = data.get("sender", "")
                content = data.get("content", "")
                dm_history.append({"sender": sender, "content": content})

                # Reply to DMs reactively
                if sender != agent_id:
                    await asyncio.sleep(random.uniform(1.5, 4) * TIMER_MULTIPLIER)
                    reply = gen_dm_reply(content)
                    try:
                        await ws.send(json.dumps({
                            "action": "DM",
                            "target_player": sender,
                            "content": reply,
                        }))
                    except Exception:
                        pass

            elif event == "discussion_end":
                if discussion_task and not discussion_task.done():
                    discussion_task.cancel()

            elif event == "voting_start":
                if not vote_submitted:
                    await asyncio.sleep(random.uniform(3, 8) * TIMER_MULTIPLIER)
                    target = suspect_pid or pick_suspect()
                    if target:
                        await ws.send(json.dumps({"action": "VOTE", "target_player": target}))
                        vote_submitted = True

            elif event == "meta_game_start":
                await asyncio.sleep(random.uniform(2, 5) * TIMER_MULTIPLIER)
                players_data = data.get("players", {})
                classifications = {}
                for pid in players_data:
                    if pid != agent_id:
                        classifications[pid] = {
                            "guess": random.choice(["human", "ai"]),
                            "confidence": random.randint(1, 5),
                            "reasoning": "just a feeling",
                        }
                await ws.send(json.dumps({
                    "action": "CLASSIFICATION",
                    "content": json.dumps(classifications),
                }))

            elif event == "game_finished":
                break

    except Exception as e:
        if "ConnectionClosed" not in type(e).__name__:
            print(f"[MockAgent {agent_id}] Error: {e}")
    finally:
        if discussion_task and not discussion_task.done():
            discussion_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass


# ── Serve frontend ───────────────────────────────────────────────────

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    # index.html references /static/assets/... and /static/vite.svg
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))
