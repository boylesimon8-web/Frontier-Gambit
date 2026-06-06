"""
Frontier Gambit — Core Game Engine
Handles: Flashpoints, Horde detection, Scoring, Creature activations
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple


# ─── ENUMS ───────────────────────────────────────────────────────────────────

class FlashpointType(str, Enum):
    GOLD_CONVOY    = "gold_convoy"
    STANDOFF       = "standoff"
    BOUNTY_BLITZ   = "bounty_blitz"
    SHERIFFS_CHASE = "sheriffs_chase"
    LAST_STAND     = "last_stand"
    NIGHT_RIDE     = "night_ride"
    RECKONING      = "reckoning"   # fires exactly once, always last


class Bracket(str, Enum):
    PIONEER    = "pioneer"
    RANGER     = "ranger"
    GUNSLINGER = "gunslinger"
    WARLORD    = "warlord"

BRACKET_ORDER = [Bracket.PIONEER, Bracket.RANGER, Bracket.GUNSLINGER, Bracket.WARLORD]
BRACKET_THRESHOLDS = [(0, 10_000_000), (10_000_001, 50_000_000),
                      (50_000_001, 200_000_000), (200_000_001, None)]


# ─── DATA CLASSES ─────────────────────────────────────────────────────────────

@dataclass
class Flashpoint:
    type: FlashpointType
    start_time: datetime
    end_time: datetime
    location: Optional[Tuple[int, int]] = None

    @property
    def active(self) -> bool:
        return self.start_time <= datetime.utcnow() <= self.end_time

    @property
    def seconds_remaining(self) -> int:
        return max(0, int((self.end_time - datetime.utcnow()).total_seconds()))

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "end_time": self.end_time.isoformat(),
            "seconds_remaining": self.seconds_remaining,
            "location": self.location,
        }


@dataclass
class Player:
    id: str
    power: int
    alliance_id: str
    gold_dust: int = 0
    bounty_tokens: int = 0
    gather_trips_completed: int = 0
    creature_caches: List[str] = field(default_factory=list)
    outlaw_marks: int = 0
    flashpoints_joined: int = 0
    is_online: bool = False

    @property
    def bracket(self) -> Bracket:
        for (lo, hi), b in zip(BRACKET_THRESHOLDS, BRACKET_ORDER):
            if hi is None or lo <= self.power <= hi:
                return b
        return Bracket.PIONEER

    @property
    def final_score(self) -> int:
        return self.gold_dust + int(self.bounty_tokens * 1.5)

    @property
    def gather_multiplier(self) -> float:
        if self.bracket == Bracket.PIONEER and self.gather_trips_completed >= 8:
            return 1.5
        return 1.0


# ─── FLASHPOINT ENGINE ────────────────────────────────────────────────────────

class FlashpointEngine:
    """
    Randomly schedules Flashpoints across Phase 3 (48 hours).

    Design intent:
    - Gap between events: 2–6 hours (unpredictable scheduling)
    - Warning lead time: 12–45 minutes (not always enough to plan)
    - Duration: 20–45 minutes per event (randomised per type)
    - The Reckoning fires exactly once in the final 6-hour window
    - Night Ride only fires between 22:00–06:00 server time
    - Same type never repeats back-to-back
    """

    DURATIONS: Dict[FlashpointType, Tuple[int, int]] = {
        FlashpointType.GOLD_CONVOY:    (1200, 2700),
        FlashpointType.STANDOFF:       (1200, 2700),
        FlashpointType.BOUNTY_BLITZ:   (1200, 1800),
        FlashpointType.SHERIFFS_CHASE: (1200, 2700),
        FlashpointType.LAST_STAND:     (1200, 2700),
        FlashpointType.NIGHT_RIDE:     (1800, 2700),
        FlashpointType.RECKONING:      (2700, 2700),
    }

    GAP_RANGE     = (7_200,  21_600)  # 2–6 hours
    WARNING_RANGE = (720,    2_700)   # 12–45 min before start

    def __init__(self):
        self.current:         Optional[Flashpoint] = None
        self.history:         List[FlashpointType] = []
        self.reckoning_fired: bool = False
        self._next_start:     Optional[datetime] = None
        self._next_warning:   Optional[datetime] = None

    def _pick_type(self) -> FlashpointType:
        hour = datetime.utcnow().hour
        pool = [t for t in FlashpointType if t != FlashpointType.RECKONING]
        if not (hour >= 22 or hour < 6):
            pool = [t for t in pool if t != FlashpointType.NIGHT_RIDE]
        if self.history:
            pool = [t for t in pool if t != self.history[-1]] or pool
        return random.choice(pool)

    def _schedule_next(self, after: Optional[datetime] = None) -> None:
        base = after or datetime.utcnow()
        gap  = random.randint(*self.GAP_RANGE)
        self._next_start   = base + timedelta(seconds=gap)
        lead               = random.randint(*self.WARNING_RANGE)
        self._next_warning = self._next_start - timedelta(seconds=lead)

    def _build(self, fp_type: FlashpointType) -> Flashpoint:
        lo, hi  = self.DURATIONS[fp_type]
        dur     = random.randint(lo, hi)
        now     = datetime.utcnow()
        return Flashpoint(type=fp_type, start_time=now, end_time=now + timedelta(seconds=dur))

    async def run(self, broadcast: Callable, phase_end: datetime) -> None:
        """
        Main scheduler loop — runs as a background asyncio task.
        Calls broadcast(event_dict) for warning, start, and end events.
        All client animations triggered client-side from these events.
        Zero rendering cost on the server.
        """
        self._schedule_next()

        while datetime.utcnow() < phase_end:
            now = datetime.utcnow()

            # Check if The Reckoning should fire (final 6-hour window)
            reckoning_window = phase_end - timedelta(hours=6)
            if (not self.reckoning_fired
                    and now >= reckoning_window
                    and self.current is None):
                self.reckoning_fired = True
                fp = self._build(FlashpointType.RECKONING)
                self.current = fp
                self.history.append(fp.type)
                await broadcast({"event": "flashpoint_start", "data": fp.to_dict()})
                await asyncio.sleep(fp.duration_seconds)
                await broadcast({"event": "flashpoint_end", "data": {"type": fp.type.value}})
                self.current = None
                break  # Reckoning is the final event

            # Warning
            if (self._next_warning and now >= self._next_warning
                    and self.current is None):
                await broadcast({"event": "storm_warning",
                                 "data": {"message": "Storm's coming..."}})
                self._next_warning = None

            # Start next flashpoint
            if (self._next_start and now >= self._next_start
                    and self.current is None):
                fp_type = self._pick_type()
                fp      = self._build(fp_type)
                self.current = fp
                self.history.append(fp_type)

                await broadcast({"event": "flashpoint_start", "data": fp.to_dict()})
                await asyncio.sleep(fp.duration_seconds)
                await broadcast({"event": "flashpoint_end",
                                 "data": {"type": fp_type.value}})
                self.current = None
                self._schedule_next(after=datetime.utcnow())

            await asyncio.sleep(15)  # poll every 15 seconds


# ─── HORDE DETECTOR ───────────────────────────────────────────────────────────

class HordeDetector:
    """
    Time-windowed Mosquito Horde detection.

    In production this would be a Redis sorted set per target alliance:
        ZADD  horde:{alliance_id}  {timestamp}  {player_id}
        ZRANGEBYSCORE horde:{alliance_id}  (now-600)  now  →  count
        EXPIRE horde:{alliance_id}  600

    This in-memory version is equivalent for single-server use.
    At scale: one lightweight Redis op per swarm activation.
    Server load is O(1) per activation — not proportional to player count.
    """

    THRESHOLD      = 5
    WINDOW_SECONDS = 600   # 10-minute coordination window
    WARNING_AHEAD  = 180   # smoke signal fires 3 min before horde hits
    HORDE_DURATION = 1200  # 20-minute horde window

    def __init__(self):
        # target_alliance_id -> [(timestamp, player_id)]
        self._pending: Dict[str, List[Tuple[datetime, str]]] = {}
        self._active:  Dict[str, datetime] = {}  # alliance_id -> expiry

    def _clean(self, target: str) -> None:
        cutoff = datetime.utcnow() - timedelta(seconds=self.WINDOW_SECONDS)
        self._pending[target] = [
            (ts, pid) for ts, pid in self._pending.get(target, []) if ts > cutoff
        ]

    def activate_swarm(self, player_id: str, target_alliance_id: str) -> dict:
        """
        Register one Mosquito Swarm activation.
        Returns the effect result — caller broadcasts this to relevant players.
        """
        target = target_alliance_id
        self._pending.setdefault(target, [])
        self._clean(target)
        self._pending[target].append((datetime.utcnow(), player_id))

        count = len(self._pending[target])

        if count >= self.THRESHOLD:
            expiry = datetime.utcnow() + timedelta(seconds=self.HORDE_DURATION)
            self._active[target] = expiry
            self._pending[target] = []
            return {
                "status": "HORDE_TRIGGERED",
                "target_alliance": target,
                "contributors": count,
                "horde_expires": expiry.isoformat(),
                "effects": {
                    "defense_reduction":      0.35,
                    "gather_speed_reduction": 0.20,
                    "bounty_token_bonus":     0.15,
                },
            }
        elif count == self.THRESHOLD - 1:
            return {
                "status": "HORDE_IMMINENT",
                "target_alliance": target,
                "count": count,
                "needed": 1,
                "warning_fires_in_seconds": self.WARNING_AHEAD,
            }
        else:
            return {
                "status": "SOLO_HIT",
                "target_alliance": target,
                "count": count,
                "effects": {"defense_reduction": 0.08},
            }

    def is_horde_active(self, alliance_id: str) -> bool:
        expiry = self._active.get(alliance_id)
        if expiry and datetime.utcnow() < expiry:
            return True
        self._active.pop(alliance_id, None)
        return False


# ─── SCORING ENGINE ───────────────────────────────────────────────────────────

class ScoringEngine:
    """
    Calculates and batch-flushes scores.

    Key decisions:
    - Small player gather multiplier activates only after 8 completed trips
      (they earn it, not gifted)
    - Punching down 2+ brackets = Outlaw Mark + zero reward
    - Bracket standings calculated per division — Pioneer vs Pioneer only
    - Scores flush to DB every 30 seconds (not per-action) to keep write
      throughput at ~2 writes/player/minute under full load
    - At 10,000 concurrent players: ~333 writes/second at flush time,
      compared to ~50,000/sec if scored per-action
    """

    GATHER_BASE  = {1: 10, 2: 25, 3: 60}
    BOUNTY_FINAL = 1.5  # Bounty Tokens × 1.5 added to final score

    OUTLAW_TRIGGER_TIER_GAP = 2   # hitting this many brackets below = outlaw

    def __init__(self):
        self._pending: List[dict] = []

    def _bracket_index(self, bracket: Bracket) -> int:
        return BRACKET_ORDER.index(bracket)

    def _tier_gap(self, attacker: Player, target: Player) -> int:
        return self._bracket_index(attacker.bracket) - self._bracket_index(target.bracket)

    # ── Gather ────────────────────────────────────────────────────────────────

    def score_gather(self, player: Player, tile_grade: int) -> int:
        base  = self.GATHER_BASE[tile_grade]
        pts   = int(base * player.gather_multiplier)
        player.gold_dust             += pts
        player.gather_trips_completed += 1
        self._pending.append({
            "player_id": player.id, "type": "gold_dust",
            "amount": pts, "ts": datetime.utcnow().isoformat()
        })
        return pts

    # ── Combat ────────────────────────────────────────────────────────────────

    def score_combat(self, attacker: Player, target: Player,
                     target_outlaw_marked: bool = False) -> dict:
        gap = self._tier_gap(attacker, target)

        if gap >= self.OUTLAW_TRIGGER_TIER_GAP:
            attacker.outlaw_marks += 1
            return {
                "gold_dust": 0, "bounty_tokens": 0,
                "outlaw_mark_applied": True,
                "reason": f"Cannot earn points hitting {self.OUTLAW_TRIGGER_TIER_GAP}+ brackets below"
            }

        gd = 100
        bt = 150
        if gap < 0:           # punching up
            gd, bt = 175, 250
        if target_outlaw_marked:
            gd, bt = 200, 300

        # Apply flashpoint Bounty Blitz multiplier (passed in by caller)
        attacker.gold_dust     += gd
        attacker.bounty_tokens += bt
        self._pending.extend([
            {"player_id": attacker.id, "type": "gold_dust",     "amount": gd, "ts": datetime.utcnow().isoformat()},
            {"player_id": attacker.id, "type": "bounty_tokens", "amount": bt, "ts": datetime.utcnow().isoformat()},
        ])
        return {"gold_dust": gd, "bounty_tokens": bt, "outlaw_mark_applied": False}

    # ── Bounty Board ──────────────────────────────────────────────────────────

    def score_bounty_task(self, player: Player, base_reward: int,
                          blitz_active: bool = False) -> int:
        multiplier = 3.0 if blitz_active else 1.0
        bt = int(base_reward * multiplier)
        player.bounty_tokens += bt
        self._pending.append({
            "player_id": player.id, "type": "bounty_tokens",
            "amount": bt, "ts": datetime.utcnow().isoformat()
        })
        return bt

    # ── Final score ───────────────────────────────────────────────────────────

    @staticmethod
    def final_score(player: Player) -> int:
        return player.gold_dust + int(player.bounty_tokens * 1.5)

    # ── Leaderboard ───────────────────────────────────────────────────────────

    @staticmethod
    def bracket_standings(players: Dict[str, Player], bracket: Bracket) -> List[dict]:
        in_bracket = [p for p in players.values() if p.bracket == bracket]
        ranked = sorted(in_bracket, key=lambda p: ScoringEngine.final_score(p), reverse=True)
        return [
            {
                "rank": i + 1,
                "player_id": p.id,
                "final_score": ScoringEngine.final_score(p),
                "gold_dust": p.gold_dust,
                "bounty_tokens": p.bounty_tokens,
            }
            for i, p in enumerate(ranked)
        ]

    @staticmethod
    def alliance_standings(players: Dict[str, Player]) -> List[dict]:
        alliances: Dict[str, List[Player]] = {}
        for p in players.values():
            alliances.setdefault(p.alliance_id, []).append(p)
        results = []
        for aid, members in alliances.items():
            total = sum(ScoringEngine.final_score(m) for m in members)
            per_member = total / len(members) if members else 0
            results.append({
                "alliance_id": aid,
                "total_score": total,
                "per_member_score": round(per_member),
                "member_count": len(members),
            })
        return sorted(results, key=lambda r: r["per_member_score"], reverse=True)

    # ── Flush ─────────────────────────────────────────────────────────────────

    async def flush(self, write_fn: Callable) -> int:
        """
        Batch-write all pending score deltas.
        Called every 30 seconds by the background task — not per-action.
        write_fn receives a list of dicts for bulk INSERT/UPDATE.
        """
        if not self._pending:
            return 0
        batch, self._pending = self._pending, []
        await write_fn(batch)
        return len(batch)


# ─── CREATURE SYSTEM ──────────────────────────────────────────────────────────

class CreatureSystem:
    """
    Handles creature cache drops and activation effects.
    All animation/VFX triggered client-side from the returned event dict.
    Server only tracks state — zero rendering cost.
    """

    def __init__(self, creature_data_path: str = "data/creatures.json"):
        with open(creature_data_path) as f:
            raw = json.load(f)
        self._creatures = {c["id"]: c for c in raw["creatures"]}
        self._fusions   = {tuple(sorted(fx["requires"])): fx for fx in raw["fusions"]}

    def roll_drop(self, tile_grade: int, player: Player) -> Optional[str]:
        """
        Roll for a cache drop after a completed gather.
        Higher grade tiles have higher drop chances.
        Returns creature id or None.
        """
        base_chance = {1: 0.08, 2: 0.14, 3: 0.22}[tile_grade]
        if random.random() > base_chance:
            return None

        pool = [
            c for c in self._creatures.values()
            if tile_grade in c["drop_grades"]
        ]
        weights = [c["drop_weight"] for c in pool]
        chosen  = random.choices(pool, weights=weights, k=1)[0]
        return chosen["id"]

    def activate(self, creature_id: str, player: Player,
                 target_id: Optional[str] = None,
                 horde_detector: Optional[HordeDetector] = None) -> dict:
        c = self._creatures.get(creature_id)
        if not c:
            return {"error": "unknown_creature"}
        if creature_id not in player.creature_caches:
            return {"error": "not_in_inventory"}

        player.creature_caches.remove(creature_id)

        if creature_id == "mosquito_swarm" and horde_detector and target_id:
            return horde_detector.activate_swarm(player.id, target_id)

        if creature_id == "jackalope":
            pool    = c["effects"]["random_pool"]
            weights = [o["weight"] for o in pool]
            chosen  = random.choices(pool, weights=weights, k=1)[0]
            return {
                "creature": "jackalope",
                "effect": chosen["id"],
                "description": chosen["desc"],
                "reveal_delay_seconds": c["effects"]["reveal_delay_seconds"],
            }

        return {
            "creature": creature_id,
            "effects": c["effects"],
            "target": target_id,
            "duration_seconds": c.get("duration_seconds"),
        }

    def check_fusion(self, creature_a: str, creature_b: str,
                     same_target: bool = True) -> Optional[dict]:
        key = tuple(sorted([creature_a, creature_b]))
        fusion = self._fusions.get(key)
        if fusion and same_target:
            return fusion
        return None


# ─── OUTLAW SYSTEM ────────────────────────────────────────────────────────────

class OutlawSystem:
    """
    Tracks Outlaw Marks and manages Phase 3 reveal mechanics.
    Outlaws are players who hit 2+ brackets below them.
    """

    BOUNTY_BONUS_GD = 200
    BOUNTY_BONUS_BT = 300
    GOLD_DROP_PCT   = 0.25  # Reckoning: top 3 drop 25% of gold if zeroed

    def __init__(self):
        self._marks:     Dict[str, int]      = {}  # player_id -> mark count
        self._survivors: Set[str]            = set()
        self._deputies:  Set[str]            = set()

    def add_mark(self, player_id: str) -> int:
        self._marks[player_id] = self._marks.get(player_id, 0) + 1
        return self._marks[player_id]

    def is_marked(self, player_id: str) -> bool:
        return self._marks.get(player_id, 0) > 0

    def clear_mark(self, player_id: str) -> None:
        self._marks.pop(player_id, None)

    def top_outlaws(self, players: Dict[str, Player], n: int = 3) -> List[Player]:
        marked = [players[pid] for pid in self._marks if pid in players]
        return sorted(marked, key=lambda p: p.final_score, reverse=True)[:n]

    def grant_deputy(self, player_id: str) -> None:
        self._deputies.add(player_id)

    def is_deputy(self, player_id: str) -> bool:
        return player_id in self._deputies


# ─── MAIN GAME ENGINE ─────────────────────────────────────────────────────────

class GameEngine:
    """
    Orchestrates all subsystems.
    One instance per server process (shared across all WebSocket connections).
    All player state is in-memory; flush tasks write to DB asynchronously.
    """

    def __init__(self, creature_data_path: str = "data/creatures.json"):
        self.flashpoints  = FlashpointEngine()
        self.horde        = HordeDetector()
        self.scoring      = ScoringEngine()
        self.creatures    = CreatureSystem(creature_data_path)
        self.outlaws      = OutlawSystem()
        self.players:     Dict[str, Player] = {}
        self.alliances:   Dict[str, Set[str]] = {}  # alliance_id -> {player_ids}

    def register_player(self, player_id: str, power: int, alliance_id: str) -> Player:
        p = Player(id=player_id, power=power, alliance_id=alliance_id)
        self.players[player_id] = p
        self.alliances.setdefault(alliance_id, set()).add(player_id)
        return p

    def on_gather_complete(self, player_id: str, tile_grade: int) -> dict:
        player = self.players[player_id]
        pts    = self.scoring.score_gather(player, tile_grade)
        drop   = self.creatures.roll_drop(tile_grade, player)
        if drop and len(player.creature_caches) < 3:
            player.creature_caches.append(drop)
        return {
            "gold_dust_earned": pts,
            "total_gold_dust": player.gold_dust,
            "cache_drop": drop,
            "gather_multiplier": player.gather_multiplier,
            "trips_completed": player.gather_trips_completed,
        }

    def on_zero(self, attacker_id: str, target_id: str) -> dict:
        attacker = self.players[attacker_id]
        target   = self.players[target_id]
        marked   = self.outlaws.is_marked(target_id)
        result   = self.scoring.score_combat(attacker, target, marked)

        if result.get("outlaw_mark_applied"):
            self.outlaws.add_mark(attacker_id)
        elif marked:
            self.outlaws.clear_mark(target_id)
            self.outlaws.grant_deputy(attacker_id)

        return result
