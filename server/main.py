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
DISCUSSION_DURATION = 120       # open chat
VOTING_TIME_LIMIT = 60          # voting
REFLECTION_DURATION = 30        # post-round reflection
META_GAME_DURATION = 90         # classification form
SURVEY_DURATION = 120           # survey
MAX_HINT_CHARS = 60
MAX_HINT_WORDS = 8
MAX_MESSAGE_LENGTH = 200

# ── Module-level state ───────────────────────────────────────────────

_session: GameSession | None = None
_logger: GameLogger | None = None
_word_bank = load_word_bank()
_pending_agents: list[dict] = []     # agents to spawn on game start
_game_task: asyncio.Task | None = None
_dedup_cache: dict[str, float] = {}  # "pid:hash" -> timestamp

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
    global _session, _logger, _pending_agents, _dedup_cache
    session_id = str(uuid.uuid4())[:8]
    _session = GameSession(session_id)
    _logger = GameLogger(session_id)
    _pending_agents = []
    _dedup_cache = {}
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
    return {
        "agents": [
            {"type": "mock", "name": "Mock Bot", "available": True},
        ]
    }


@app.post("/setup")
async def setup(request: Request):
    """Add agents to the pending list before game start."""
    global _pending_agents
    body = await request.json()
    new_agents = body.get("agents", [])
    for agent_cfg in new_agents:
        _pending_agents.append(agent_cfg)
    return {"status": "ok", "pending_agents": len(_pending_agents)}


@app.post("/start")
async def start_game():
    """Launch the game."""
    global _game_task
    if not _session:
        _reset_session()
    if _game_task and not _game_task.done():
        return {"status": "already_running"}

    # Spawn pending mock agents
    for agent_cfg in _pending_agents:
        agent_type = agent_cfg.get("type", "mock")
        if agent_type == "mock":
            asyncio.create_task(_spawn_mock_agent(_session))
            await asyncio.sleep(0.1)  # stagger

    # Wait a moment for agents to connect
    if _pending_agents:
        await asyncio.sleep(0.5)

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

    _logger.log({"event": "group_message", "sender": player_id, "content": content})


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

    _logger.log({"event": "dm", "sender": player_id, "target": target_id, "content": content})


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
        warning_thresholds = [60, 30, 10]
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

        logger.log({"event": "round_results", "round": session.current_round, **result})

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

    logger.log({
        "event": "grand_reveal",
        "leaderboard": leaderboard,
        "classification_scores": cls_scores,
        "confusion_matrix": confusion,
    })

    await asyncio.sleep(_scaled(5))

    # Finished
    session.phase = Phase.FINISHED
    await _broadcast(session, {
        "event": "game_finished",
        "data": {
            "final_scores": session.cumulative_scores,
        },
    })


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

MOCK_MESSAGES = [
    "hmm that hint was kinda weird",
    "idk anyone else find that sus?",
    "lol yeah i was thinking the same",
    "wait what do you guys think",
    "im not sure who to vote for",
    "that's a good point actually",
    "ok so who we voting for",
    "nah i think they're fine",
    "the hints all seem normal to me",
    "someone is definitely lying tho",
    "i have a feeling about this",
    "lets figure this out guys",
]


async def _spawn_mock_agent(session: GameSession):
    """Spawn a mock AI agent that connects via internal WebSocket."""
    import websockets

    # Find the next available agent ID
    existing = [pid for pid in session.players if pid.startswith("agent_")]
    agent_num = len(existing) + 1
    agent_id = f"agent_{agent_num}"
    agent_name = f"Agent {agent_num}"

    # Connect via localhost WebSocket
    try:
        uri = f"ws://127.0.0.1:8000/ws/{agent_id}?name={agent_name}&type=AI"
        ws = await websockets.connect(uri)
        asyncio.create_task(_mock_agent_loop(ws, agent_id, session))
    except Exception as e:
        print(f"[MockAgent] Failed to spawn {agent_id}: {e}")


async def _mock_agent_loop(ws, agent_id: str, session: GameSession):
    """Mock agent event loop — listens for server events and responds."""
    import websockets

    my_role = None
    my_secret_word = None
    game_pids = []
    hint_submitted = False
    vote_submitted = False

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
                game_pids = [pid for pid in roster if pid != agent_id and pid != "HOST"]

            elif event == "round_start":
                my_role = data.get("role", "insider")
                my_secret_word = data.get("secret_word")
                hint_submitted = False
                vote_submitted = False

            elif event == "hint_phase_start":
                if not hint_submitted:
                    await asyncio.sleep(random.uniform(2, 6) * TIMER_MULTIPLIER)
                    if my_role == "imposter":
                        hint = random.choice(MOCK_HINTS_IMPOSTER)
                    else:
                        hint = random.choice(MOCK_HINTS_INSIDER)
                    await ws.send(json.dumps({"action": "HINT", "content": hint}))
                    hint_submitted = True

            elif event == "discussion_start":
                # Send 2-4 messages during discussion
                num_msgs = random.randint(2, 4)
                for i in range(num_msgs):
                    await asyncio.sleep(random.uniform(3, 10) * TIMER_MULTIPLIER)
                    content = random.choice(MOCK_MESSAGES)
                    await ws.send(json.dumps({"action": "GROUP_MESSAGE", "content": content}))

            elif event == "voting_start":
                if not vote_submitted and game_pids:
                    await asyncio.sleep(random.uniform(3, 8) * TIMER_MULTIPLIER)
                    target = random.choice(game_pids)
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

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        print(f"[MockAgent {agent_id}] Error: {e}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ── Serve frontend ───────────────────────────────────────────────────

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dir / "assets")), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))
