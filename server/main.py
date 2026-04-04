import asyncio
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .game_state import GameSession, Phase
from .assignment import run_assignment_phase, load_word_bank
from .logger import GameLogger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# v1: one session at a time
_session: GameSession | None = None
_logger: GameLogger | None = None
_next_player_num: int = 1
_word_bank = load_word_bank()


def _reset_session() -> tuple[GameSession, GameLogger]:
    global _session, _logger, _next_player_num
    session_id = str(uuid.uuid4())[:8]
    _session = GameSession(session_id)
    _logger = GameLogger(session_id)
    _next_player_num = 1
    return _session, _logger


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global _session, _logger, _next_player_num

    await websocket.accept()

    # Start a new session if none exists or current is already in-progress
    if _session is None or _session.phase != Phase.LOBBY:
        _session, _logger = _reset_session()

    session = _session
    logger = _logger

    # Assign player ID
    player_id = f"Player {_next_player_num}"
    _next_player_num += 1

    # Handshake: client sends {"player_type": "human"} or {"player_type": "ai"}
    try:
        handshake = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        is_human = handshake.get("player_type", "human") == "human"
    except Exception:
        is_human = True

    if not session.add_player(player_id, websocket, is_human):
        await websocket.send_json({"event": "error", "message": "Session is full"})
        await websocket.close()
        return

    await websocket.send_json({
        "event": "joined",
        "player_id": player_id,
        "player_count": len(session.players),
        "max_players": session.MAX_PLAYERS,
    })

    await _broadcast(session, {
        "event": "lobby_update",
        "player_count": len(session.players),
        "max_players": session.MAX_PLAYERS,
    })

    if session.is_full():
        asyncio.create_task(_run_game(session, logger))

    try:
        while True:
            data = await websocket.receive_json()
            await _handle_message(session, player_id, data, logger)
    except WebSocketDisconnect:
        session.players[player_id].connected = False
        await _broadcast(session, {"event": "player_disconnected", "player_id": player_id})


async def _run_game(session: GameSession, logger: GameLogger):
    await _broadcast(session, {
        "event": "game_start",
        "total_rounds": session.MAX_ROUNDS,
        "players": session.player_ids(),
    })

    for _ in range(session.MAX_ROUNDS):
        # Phase 1: Assignment (this file)
        await run_assignment_phase(session, logger, _word_bank)

        # Phase 2: Hint — TODO: @Nirzaree
        # Phase 3: Discussion — TODO
        # Phase 4: Voting — TODO
        # Phase 5: Results + Reflection — TODO: @조용은 + @Nirzaree
        await asyncio.sleep(1)  # placeholder until Phase 2 is wired in

    session.phase = Phase.META_GAME
    await _broadcast(session, {"event": "meta_game_start"})

    session.phase = Phase.FINISHED


async def _handle_message(
    session: GameSession, player_id: str, data: dict, logger: GameLogger
):
    """Dispatch incoming player messages — phase-specific handlers added here as phases are built."""
    logger.log({
        "event": "player_action",
        "player": player_id,
        "phase": session.phase,
        "data": data,
    })
    # Phase 2+ handlers will be registered here


async def _broadcast(session: GameSession, message: dict):
    for player_id, player in list(session.players.items()):
        if not player.connected:
            continue
        try:
            await player.websocket.send_json(message)
        except Exception:
            session.players[player_id].connected = False
