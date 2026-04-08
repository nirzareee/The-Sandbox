"""FastAPI application for The Sandbox game server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time as _time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.agent_manager import AgentManager
from server.config import DEFAULT_HOST, DEFAULT_PORT
from server.connection_manager import ConnectionManager
from server.game_state import GameStateMachine
from server.logger import GameLogger
from server.message_router import MessageRouter
from server.models import (
    GameConfig,
    GamePhase,
    PlayerInfo,
    PlayerType,
    ServerMessage,
)
from server.round_manager import RoundManager
from server.scoring import ScoringEngine
from server.word_bank import WordBank

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state — initialized by lifespan or POST /start
# ---------------------------------------------------------------------------

connection_manager = ConnectionManager()
agent_manager = AgentManager()
router = MessageRouter()

_recent_msg_hashes: dict[tuple, float] = {}
_DEDUP_WINDOW = 10.0


def _is_duplicate_message(player_id: str, content: str) -> bool:
    now = _time.time()
    # Clean expired
    expired = [k for k, ts in _recent_msg_hashes.items() if now - ts > _DEDUP_WINDOW]
    for k in expired:
        del _recent_msg_hashes[k]
    key = (player_id, content.lower().strip())
    if key in _recent_msg_hashes:
        return True
    _recent_msg_hashes[key] = now
    return False


# These are set once a game session is created (POST /start).
game_state: GameStateMachine | None = None
game_logger: GameLogger | None = None
round_manager: RoundManager | None = None
_game_task: asyncio.Task | None = None
_start_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Router callback wiring
# ---------------------------------------------------------------------------

async def _on_group_message(
    player_id: str, content: str, round_number: int, phase: GamePhase
) -> None:
    if _is_duplicate_message(player_id, content):
        log.info("Blocked duplicate group message from %s", player_id)
        return
    msg = ServerMessage(
        event="group_message",
        data={"player_id": player_id, "content": content, "round": round_number},
    )
    await connection_manager.broadcast(msg)
    if game_logger:
        await game_logger.log_group_chat(player_id, content, round_number, phase.value)


async def _on_dm(
    player_id: str,
    target_player: str,
    content: str,
    round_number: int,
    phase: GamePhase,
) -> None:
    msg = ServerMessage(
        event="dm",
        data={
            "sender": player_id,
            "recipient": target_player,
            "content": content,
            "round": round_number,
        },
    )
    await connection_manager.send_to_pair(player_id, target_player, msg)
    # Forward all DMs to hosts so they see everything
    await connection_manager.send_to_hosts(msg)
    if game_logger:
        await game_logger.log_dm(
            player_id, target_player, content, round_number, phase.value
        )


async def _on_hint(player_id: str, content: str, round_number: int) -> None:
    if round_manager:
        await round_manager.receive_hint(player_id, content)


async def _on_vote(player_id: str, voted_for: str, round_number: int) -> None:
    if round_manager:
        await round_manager.receive_vote(player_id, voted_for)
    if game_logger:
        await game_logger.log_vote(player_id, voted_for, round_number)


async def _on_classification(player_id: str, classifications: dict) -> None:
    if round_manager:
        await round_manager.record_classification(player_id, classifications)
    if game_logger:
        # Convert Pydantic models to plain dicts for JSON serialization.
        serializable = {
            k: v.model_dump() if hasattr(v, "model_dump") else v
            for k, v in classifications.items()
        }
        await game_logger.save_classification(player_id, serializable)


async def _on_survey(player_id: str, responses: dict) -> None:
    if round_manager:
        await round_manager.record_survey(player_id, responses)
    if game_logger:
        await game_logger.save_survey(player_id, responses)


router.on_group_message = _on_group_message
router.on_dm = _on_dm
router.on_hint = _on_hint
router.on_vote = _on_vote
router.on_classification = _on_classification
router.on_survey = _on_survey
router.get_valid_players = connection_manager.get_game_players


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("The Sandbox server starting up")
    yield
    await agent_manager.shutdown()
    log.info("The Sandbox server shutting down")


app = FastAPI(title="The Sandbox", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files (if directory exists).
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """Serve the frontend entry point."""
    index_path = os.path.join(_frontend_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "The Sandbox — frontend not built yet"})


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "players_connected": connection_manager.get_game_players(),
        "player_count": connection_manager.get_game_player_count(),
        "game_active": game_state is not None and not game_state.is_game_over(),
    })


@app.get("/state")
async def state():
    if game_state is None:
        return JSONResponse({"error": "No game session active"}, status_code=404)
    return JSONResponse(game_state.get_state_snapshot())


@app.get("/agents")
async def list_agents():
    """Return available agent types for the host UI."""
    return JSONResponse({"agents": agent_manager.get_available_agents()})


@app.post("/setup")
async def setup_agents(request: Request):
    """Spawn agents based on host's selection.

    Can be called multiple times to add agents incrementally.
    Rejects if a game is already active.
    """
    # Don't spawn agents if game is already running
    if game_state is not None and not game_state.is_game_over():
        return JSONResponse(
            {"error": "Game already active — cannot spawn agents"},
            status_code=409,
        )

    body = await request.json()
    agents = body.get("agents", [])

    # Use 127.0.0.1 for agent connections — 0.0.0.0 is a bind address, not connectable
    port = int(os.getenv("PORT", str(DEFAULT_PORT)))
    ws_url = f"ws://127.0.0.1:{port}"

    spawned = await agent_manager.spawn_agents(agents, ws_url)
    return JSONResponse({"status": "ok", "spawned": spawned})


@app.post("/start")
async def start_game(request: Request):
    """Create a new game session and launch the game loop.

    Accepts optional JSON body with ``session_id`` and overrides.
    """
    global game_state, game_logger, round_manager, _game_task

    async with _start_lock:
        if game_state is not None and not game_state.is_game_over():
            return JSONResponse(
                {"error": "A game session is already active"},
                status_code=409,
            )

        # Parse optional JSON body.
        config: dict = {}
        body = await request.body()
        if body:
            try:
                config = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                pass

        default_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_id = config.get(
            "session_id",
            os.getenv("SESSION_ID", default_id),
        )
        game_config = GameConfig(session_id=session_id)

        game_state = GameStateMachine(game_config)
        game_logger = GameLogger(session_id)
        await game_logger.save_config(game_config.model_dump())

        # Load word bank.
        word_bank = WordBank()
        word_bank.load()
        await game_logger.save_word_bank(
            [e.model_dump() for e in word_bank.entries]
        )

        # Create round manager and scoring engine.
        scoring = ScoringEngine(game_config.scoring)
        round_manager = RoundManager(
            game_state=game_state,
            connection_manager=connection_manager,
            logger=game_logger,
            word_bank=word_bank,
            scoring=scoring,
        )

        log.info("Game session '%s' created", session_id)

        # Notify connected players that the game is ready.
        await connection_manager.broadcast(
            ServerMessage(event="game_created", data={
                "session_id": session_id,
                "num_rounds": game_config.num_rounds,
            })
        )

        # Save session manifest with player roster and config.
        await game_logger.save_manifest({
            "session_id": session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "players": {
                pid: info.model_dump()
                for pid, info in connection_manager.player_info.items()
            },
            "config": game_config.model_dump(),
        })

        # Shuffle display names to anonymize connection order
        roster = connection_manager.shuffle_display_names()
        await connection_manager.broadcast(
            ServerMessage(event="player_roster", data={"roster": roster})
        )

        # Launch the game loop as a background task.
        _game_task = asyncio.create_task(_run_game_loop())

        return JSONResponse({"session_id": session_id, "status": "created"})


async def _run_game_loop() -> None:
    """Wrapper that runs the game loop and logs any errors."""
    try:
        if round_manager:
            await round_manager.run_game()
    except Exception:
        log.exception("Game loop crashed")
        try:
            await connection_manager.broadcast(
                ServerMessage(event="error", data={"message": "Game encountered an error. Please restart."})
            )
        except Exception:
            pass


@app.get("/debug/state")
async def debug_state():
    return JSONResponse({
        "phase": game_state.phase.value if game_state else "NO_GAME",
        "current_round": game_state.current_round if game_state else 0,
        "is_game_over": game_state.is_game_over() if game_state else False,
        "cumulative_scores": game_state.cumulative_scores if game_state else {},
        "players": connection_manager.get_game_players(),
    })


@app.get("/debug/connections")
async def debug_connections():
    info = {}
    for pid, pi in connection_manager.player_info.items():
        info[pid] = {
            "display_name": pi.display_name,
            "player_type": pi.player_type.value,
            "is_connected": pi.is_connected,
            "model_name": pi.model_name,
        }
    return JSONResponse(info)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/{player_id}")
async def websocket_endpoint(websocket: WebSocket, player_id: str):
    # Extract player info from query params (or use defaults).
    params = websocket.query_params
    player_type_str = params.get("type", "HUMAN").upper()
    model_name = params.get("model", None)

    try:
        player_type = PlayerType(player_type_str)
    except ValueError:
        await websocket.close(code=4000, reason=f"Invalid player type: {player_type_str}")
        return

    is_host = player_type == PlayerType.HOST

    # Assign anonymous "Player N" name to maintain anonymity during gameplay.
    # Reconnecting players keep their existing name. Hosts get "Host".
    if player_id in connection_manager.player_info:
        display_name = connection_manager.player_info[player_id].display_name
    elif is_host:
        display_name = "Host"
    else:
        # Only count game players for numbering (exclude hosts)
        display_name = "Player"

    try:
        player_info = PlayerInfo(
            player_id=player_id,
            display_name=display_name,
            player_type=player_type,
            model_name=model_name,
        )
    except ValueError as exc:
        await websocket.close(code=4001, reason=str(exc))
        return

    await connection_manager.connect(websocket, player_id, player_info)

    if is_host:
        # Host sees all existing game players but is invisible to them.
        for existing_pid, existing_info in connection_manager.player_info.items():
            if existing_pid != player_id and existing_info.player_type != PlayerType.HOST:
                await connection_manager.send_personal(
                    player_id,
                    ServerMessage(
                        event="player_joined",
                        data={
                            "player_id": existing_pid,
                            "display_name": existing_info.display_name,
                            "player_count": connection_manager.get_game_player_count(),
                            "connected": connection_manager.get_game_players(),
                        },
                    ),
                )
        # Do NOT broadcast player_joined for the host.
    else:
        # Send existing game players to the newly connected game player
        # (skip hosts — they are invisible).
        for existing_pid, existing_info in connection_manager.player_info.items():
            if existing_pid != player_id and existing_info.player_type != PlayerType.HOST:
                await connection_manager.send_personal(
                    player_id,
                    ServerMessage(
                        event="player_joined",
                        data={
                            "player_id": existing_pid,
                            "display_name": existing_info.display_name,
                            "player_count": connection_manager.get_game_player_count(),
                            "connected": connection_manager.get_game_players(),
                        },
                    ),
                )

        # Announce to all players (including hosts) that a game player joined.
        await connection_manager.broadcast(
            ServerMessage(
                event="player_joined",
                data={
                    "player_id": player_id,
                    "display_name": display_name,
                    "player_count": connection_manager.get_game_player_count(),
                    "connected": connection_manager.get_game_players(),
                },
            )
        )

    try:
        while True:
            raw = await websocket.receive_text()

            # Hosts are read-only — ignore any messages they send.
            if is_host:
                continue

            # Determine current game phase and round.
            if game_state is not None:
                current_phase = game_state.phase
                current_round = game_state.current_round
            else:
                current_phase = GamePhase.LOBBY
                current_round = 0

            error_msg = await router.route(
                player_id, raw, current_phase, current_round
            )
            if error_msg is not None:
                await connection_manager.send_personal(player_id, error_msg)

    except WebSocketDisconnect:
        connection_manager.disconnect(player_id)
        if not is_host:
            # Only broadcast player_left for game players (host is invisible).
            await connection_manager.broadcast(
                ServerMessage(
                    event="player_left",
                    data={
                        "player_id": player_id,
                        "player_count": connection_manager.get_game_player_count(),
                        "connected": connection_manager.get_game_players(),
                    },
                )
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", DEFAULT_HOST)
    port = int(os.getenv("PORT", str(DEFAULT_PORT)))
    uvicorn.run("server.main:app", host=host, port=port, reload=True)
