"""
Frontier Gambit — WebSocket Game Server
Built with FastAPI + asyncio

Architecture overview:
- One WebSocket connection per player (persistent, low-bandwidth)
- Flashpoint events are server → broadcast (1 write, N reads)
- Score updates batch-flush every 30 sec (not per-action)
- Horde detection is O(1) per activation via time-windowed counter
- All animations/VFX run client-side — server sends state, not pixels
- Estimated load at peak Flashpoint: ~equivalent to a normal KE weekend
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from game_engine import (
    Bracket, FlashpointType, GameEngine, Player, ScoringEngine
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("frontier")


# ─── STATE ────────────────────────────────────────────────────────────────────

engine: GameEngine = GameEngine(creature_data_path="../data/creatures.json")

# Phase 3 runs for 48 hours from when it's started
PHASE_3_DURATION_HOURS = 48
phase_end: Optional[datetime] = None


# ─── CONNECTION MANAGER ───────────────────────────────────────────────────────

class ConnectionManager:
    """
    Manages all active WebSocket connections.

    Separation of concerns:
    - broadcast()          → all connected players  (Flashpoints, server events)
    - broadcast_alliance() → one alliance's members (Horde warnings, Recon)
    - send()               → one specific player    (personal effects, scores)

    This keeps traffic minimal:
    - A Flashpoint event is 1 write on the server, received by N clients
    - A Horde event goes only to the target alliance (~20–100 players)
    - Score updates go only to the affected player
    """

    def __init__(self):
        self._sockets: Dict[str, WebSocket] = {}

    async def connect(self, player_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._sockets[player_id] = ws
        if player_id in engine.players:
            engine.players[player_id].is_online = True
        log.info(f"Player {player_id} connected. Online: {len(self._sockets)}")

    async def disconnect(self, player_id: str) -> None:
        self._sockets.pop(player_id, None)
        if player_id in engine.players:
            engine.players[player_id].is_online = False
        log.info(f"Player {player_id} disconnected. Online: {len(self._sockets)}")

    async def broadcast(self, message: dict) -> None:
        """Send event to all connected players."""
        data = json.dumps(message)
        dead: list = []
        for pid, ws in self._sockets.items():
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(pid)
        for pid in dead:
            await self.disconnect(pid)

    async def broadcast_alliance(self, alliance_id: str, message: dict) -> None:
        """Send event to all online members of one alliance."""
        data    = json.dumps(message)
        members = engine.alliances.get(alliance_id, set())
        for pid in members:
            ws = self._sockets.get(pid)
            if ws:
                try:
                    await ws.send_text(data)
                except Exception:
                    pass

    async def send(self, player_id: str, message: dict) -> None:
        """Send event to one specific player."""
        ws = self._sockets.get(player_id)
        if ws:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                await self.disconnect(player_id)


mgr = ConnectionManager()


# ─── BACKGROUND TASKS ─────────────────────────────────────────────────────────

async def flashpoint_scheduler() -> None:
    """
    Runs continuously during Phase 3.
    Uses FlashpointEngine to determine random timing.
    Broadcasts events to all players — client handles animation.
    """
    global phase_end
    while phase_end is None:
        await asyncio.sleep(5)

    log.info(f"Flashpoint scheduler active. Phase ends: {phase_end.isoformat()}")
    await engine.flashpoints.run(
        broadcast=mgr.broadcast,
        phase_end=phase_end,
    )
    log.info("Flashpoint scheduler finished — Phase 3 complete.")


async def score_flush_task() -> None:
    """
    Writes pending score deltas to the database every 30 seconds.

    Why 30 seconds?
    - Per-action writes at 10k players = potentially 50k+ writes/sec at Flashpoint peaks
    - 30-second batching reduces that to ~300 writes/sec maximum
    - Score display on client uses optimistic local updates, so players see
      instant feedback without waiting for the flush
    """
    async def write_to_db(batch: list) -> None:
        # Production: bulk INSERT into score_events table
        # Example: await db.executemany("INSERT INTO score_events ...", batch)
        log.debug(f"Score flush: {len(batch)} events")

    while True:
        await asyncio.sleep(30)
        flushed = await engine.scoring.flush(write_to_db)
        if flushed:
            log.debug(f"Flushed {flushed} score events")


# ─── APP LIFESPAN ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background tasks on boot
    asyncio.create_task(flashpoint_scheduler(), name="flashpoint-scheduler")
    asyncio.create_task(score_flush_task(),     name="score-flush")
    log.info("Frontier Gambit server started.")
    yield
    log.info("Frontier Gambit server shutting down.")


app = FastAPI(title="Frontier Gambit Game Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── HTTP ENDPOINTS ───────────────────────────────────────────────────────────

class RegisterBody(BaseModel):
    player_id: str
    power:     int
    alliance_id: str

@app.post("/player/register")
async def register_player(body: RegisterBody):
    if body.player_id in engine.players:
        raise HTTPException(400, "Player already registered")
    p = engine.register_player(body.player_id, body.power, body.alliance_id)
    return {"status": "ok", "bracket": p.bracket.value}


class PhaseBody(BaseModel):
    phase_duration_hours: int = 48

@app.post("/admin/start-phase3")
async def start_phase3(body: PhaseBody):
    global phase_end
    phase_end = datetime.utcnow() + timedelta(hours=body.phase_duration_hours)
    log.info(f"Phase 3 started. Ends at {phase_end.isoformat()}")
    await mgr.broadcast({"event": "phase_start", "data": {
        "phase": 3,
        "phase_end": phase_end.isoformat(),
    }})
    return {"status": "ok", "phase_end": phase_end.isoformat()}


@app.get("/leaderboard/{bracket}")
async def leaderboard(bracket: str):
    try:
        b = Bracket(bracket)
    except ValueError:
        raise HTTPException(400, f"Unknown bracket: {bracket}")
    standings = ScoringEngine.bracket_standings(engine.players, b)
    return {"bracket": bracket, "standings": standings[:20]}


@app.get("/leaderboard/alliance/all")
async def alliance_leaderboard():
    return {"standings": ScoringEngine.alliance_standings(engine.players)[:10]}


@app.get("/state/{player_id}")
async def player_state(player_id: str):
    p = engine.players.get(player_id)
    if not p:
        raise HTTPException(404, "Player not found")
    return {
        "player_id":            p.id,
        "bracket":              p.bracket.value,
        "gold_dust":            p.gold_dust,
        "bounty_tokens":        p.bounty_tokens,
        "final_score":          p.final_score,
        "gather_trips":         p.gather_trips_completed,
        "gather_multiplier":    p.gather_multiplier,
        "creature_caches":      p.creature_caches,
        "outlaw_marks":         p.outlaw_marks,
        "horde_active_on_us":   engine.horde.is_horde_active(p.alliance_id),
        "current_flashpoint":   engine.flashpoints.current.to_dict()
                                if engine.flashpoints.current else None,
    }


# ─── WEBSOCKET ────────────────────────────────────────────────────────────────

@app.websocket("/ws/{player_id}")
async def websocket_endpoint(ws: WebSocket, player_id: str):
    await mgr.connect(player_id, ws)
    player = engine.players.get(player_id)

    try:
        # Send current state on connect
        if player:
            await mgr.send(player_id, {
                "event": "state_sync",
                "data": {
                    "gold_dust":     player.gold_dust,
                    "bounty_tokens": player.bounty_tokens,
                    "creature_caches": player.creature_caches,
                    "current_flashpoint": engine.flashpoints.current.to_dict()
                                          if engine.flashpoints.current else None,
                }
            })

        while True:
            raw = await ws.receive_text()
            await handle_message(player_id, json.loads(raw))

    except WebSocketDisconnect:
        await mgr.disconnect(player_id)


async def handle_message(player_id: str, msg: dict) -> None:
    """
    Route incoming WebSocket messages to the appropriate handler.

    Message format: { "action": "...", "data": {...} }
    """
    action = msg.get("action")
    data   = msg.get("data", {})
    player = engine.players.get(player_id)

    if not player:
        await mgr.send(player_id, {"event": "error", "data": {"msg": "Not registered"}})
        return

    # ── Gather complete ───────────────────────────────────────────────────────
    if action == "gather_complete":
        result = engine.on_gather_complete(player_id, data["tile_grade"])
        await mgr.send(player_id, {"event": "gather_result", "data": result})

    # ── Zero (combat) ─────────────────────────────────────────────────────────
    elif action == "zero":
        target_id = data.get("target_id")
        if not target_id or target_id not in engine.players:
            await mgr.send(player_id, {"event": "error", "data": {"msg": "Invalid target"}})
            return
        result = engine.on_zero(player_id, target_id)
        await mgr.send(player_id, {"event": "combat_result", "data": result})
        if result.get("outlaw_mark_applied"):
            # Server-side mark — no announcement, but target's alliance sees
            # the mark on the map on their next state sync
            pass

    # ── Activate creature ─────────────────────────────────────────────────────
    elif action == "activate_creature":
        creature_id = data.get("creature_id")
        target_id   = data.get("target_id")
        result      = engine.creatures.activate(
            creature_id, player, target_id,
            horde_detector=engine.horde
        )
        await mgr.send(player_id, {"event": "creature_result", "data": result})

        # Horde: notify target alliance + all potential attackers
        if result.get("status") == "HORDE_TRIGGERED":
            target_alliance = result["target_alliance"]
            await mgr.broadcast_alliance(target_alliance, {
                "event": "horde_incoming",
                "data": {
                    "warning_seconds": engine.horde.WARNING_AHEAD,
                    "effects": result["effects"],
                }
            })
            # Also notify attacker's alliance to coordinate attacks
            await mgr.broadcast_alliance(player.alliance_id, {
                "event": "horde_triggered",
                "data": result,
            })

    # ── Trade creature ────────────────────────────────────────────────────────
    elif action == "trade_creature":
        creature_id  = data.get("creature_id")
        recipient_id = data.get("recipient_id")
        recipient    = engine.players.get(recipient_id)
        if (not recipient or creature_id not in player.creature_caches
                or recipient.alliance_id != player.alliance_id
                or len(recipient.creature_caches) >= 3):
            await mgr.send(player_id, {"event": "error", "data": {"msg": "Trade failed"}})
            return
        player.creature_caches.remove(creature_id)
        recipient.creature_caches.append(creature_id)
        await mgr.send(player_id,    {"event": "trade_sent",     "data": {"creature_id": creature_id}})
        await mgr.send(recipient_id, {"event": "trade_received", "data": {"creature_id": creature_id,
                                                                           "from": player_id}})

    # ── Bounty task complete ──────────────────────────────────────────────────
    elif action == "bounty_complete":
        base_reward = data.get("base_reward", 100)
        blitz_active = (
            engine.flashpoints.current is not None
            and engine.flashpoints.current.type == FlashpointType.BOUNTY_BLITZ
        )
        bt = engine.scoring.score_bounty_task(player, base_reward, blitz_active)
        await mgr.send(player_id, {"event": "bounty_result", "data": {
            "bounty_tokens_earned": bt,
            "blitz_multiplier": 3.0 if blitz_active else 1.0,
            "total_bounty_tokens": player.bounty_tokens,
        }})

    # ── Heartbeat ─────────────────────────────────────────────────────────────
    elif action == "ping":
        await mgr.send(player_id, {"event": "pong", "data": {
            "server_time": datetime.utcnow().isoformat(),
            "current_flashpoint": engine.flashpoints.current.to_dict()
                                   if engine.flashpoints.current else None,
        }})

    else:
        await mgr.send(player_id, {"event": "error", "data": {"msg": f"Unknown action: {action}"}})
