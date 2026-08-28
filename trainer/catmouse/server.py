"""The live link between Python and the recording app.

One WebSocket, three modes:

  PLAY   a saved checkpoint plays the shared arena set, one episode at a time, at a
         pace you control. This is what the Academy screen walks through.
  TRAIN  a school actually trains, on camera, streaming both its algorithm internals
         and a live episode sampled out of the batch it is learning from.
  FINAL  the cross-play champion cat against the cross-play champion mouse, five
         rounds on an arena neither has seen.
  RACE   all three schools on the SAME room at the same time, sharing the same spawns,
         the same hearing noise and the same sampling draws — so any difference on
         screen is the policy and nothing else.

**Everything sent is journalled** to `runs/journals/<stamp>.jsonl`. A live stream is
the riskiest thing to record against — a stall or a crash kills the take — so every
frame is written as it goes out and `--replay <file>` plays a journal back at the same
pace, frame for frame. A take you liked can be re-shot exactly.

    python trainer/scripts/serve.py --run runs/latest
    python trainer/scripts/serve.py --replay runs/journals/2026-08-27T15-04-11.jsonl
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import arena
from .league import EXAMINER_SKILL
from .nets import FlatActor
from .school import Budget
from .scripted import ScriptedPair
from .tournament import LABELS, SCHOOLS, load_run
from .vec import MapSet, VecEnv

CHECKPOINTS = ("untrained", "half", "trained")


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


class Journal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w")
        self.t0 = time.perf_counter()

    def write(self, msg: dict) -> None:
        self.f.write(json.dumps({"at": round(time.perf_counter() - self.t0, 4), **msg},
                                default=_json) + "\n")
        self.f.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.f.close()


class Session:
    """One recording session. Owns the environment, the policies and the clock."""

    def __init__(self, run_dir: Path, device: torch.device, journal: Journal | None = None):
        self.run_dir = run_dir
        self.device = device
        self.journal = journal
        # How many holes this run was trained with. A room with one hole plays a
        # completely different game from a room with two, so the server takes it from
        # the run rather than assuming.
        cfg = self._load_json("config.json") or {}
        self.nests = arena.parse_nests(cfg.get("nests", arena.DEFAULT_NESTS))
        self.maps = MapSet(arena.TRAIN_SEEDS, arena.spread(self.nests, len(arena.TRAIN_SEEDS)))
        self.final_maps = MapSet(arena.FINAL_SEEDS, arena.spread(self.nests, len(arena.FINAL_SEEDS)))
        self.policies = {ck: load_run(run_dir, ck) for ck in CHECKPOINTS}
        self.tournament = self._load_json("tournament.json")
        self.progression = self._load_json("progression.json")
        self.budgets = self._load_json("budgets.json")
        self.highlights = self._load_json("highlights.json")

        self.mode = "idle"
        self.speed = 4.0
        self.playing = True
        self.rng = np.random.default_rng(0)
        self.env: VecEnv | None = None
        self.actors: dict = {}
        self.bot: ScriptedPair | None = None
        self.level = 0
        self.level_order: list[int] = []
        self.results: list[str] = []
        self.ctx: dict = {}
        self._map_sent = None
        self._train_task = None
        self._train_events: asyncio.Queue = asyncio.Queue(maxsize=512)
        self.train_school = None
        self._shadow_at = 0
        self.race: list[str] = []
        self._race_done: list = []

    def _load_json(self, name: str):
        p = self.run_dir / name
        return json.loads(p.read_text()) if p.exists() else None

    # ---------- catalogue ----------

    def hello(self) -> dict:
        have = {ck: sorted(self.policies[ck].keys()) for ck in CHECKPOINTS}
        return {
            "type": "hello",
            "schools": [s for s in SCHOOLS if s in self.policies["trained"]],
            "labels": LABELS,
            "checkpoints": list(CHECKPOINTS),
            "available": have,
            "levels": [{"seed": int(t.seed), "optimal": int(t.optimal), "nests": int(t.n_nests),
                        "trapsOnRoute": int(t.n_traps_on_route)} for t in self.maps.tables],
            "finalSeeds": list(arena.FINAL_SEEDS),
            "finalNests": [int(t.n_nests) for t in self.final_maps.tables],
            "nests": self.nests,
            "examinerSkill": EXAMINER_SKILL,
            "tournament": self.tournament,
            "progression": self.progression,
            "budgets": self.budgets,
            "highlights": self.highlights,
            "runDir": str(self.run_dir),
        }

    # ---------- PLAY ----------

    def start_play(self, school: str, checkpoint: str, opponent: str = "self",
                   levels: list[int] | None = None) -> dict:
        pol = self.policies[checkpoint].get(school)
        if pol is None:
            return {"type": "error", "message": f"no {school} @ {checkpoint} in this run"}
        self.mode = "play"
        self.ctx = {"school": school, "checkpoint": checkpoint, "opponent": opponent}
        self.level_order = levels if levels is not None else list(range(len(self.maps)))
        self.level = 0
        self.results = []
        self.env = VecEnv(self.maps, 1, seed=1234)
        self.bot = ScriptedPair(self.env, EXAMINER_SKILL, seed=99)
        self.actors = {
            "cat": FlatActor(pol["cat"], self.device),
            "mouse": FlatActor(pol["mouse"], self.device),
        }
        if opponent == "examiner-mouse":
            self.actors["mouse"] = None
        elif opponent == "examiner-cat":
            self.actors["cat"] = None
        self._begin_episode()
        return {"type": "state", **self.state()}

    def start_final(self, rounds: int = 5) -> dict:
        t = self.tournament
        if not t:
            return {"type": "error", "message": "no tournament.json — run the tournament first"}
        ck, mk = t["champion"]["cat"], t["champion"]["mouse"]
        self.mode = "final"
        self.ctx = {"catSchool": ck, "mouseSchool": mk, "rounds": rounds,
                    "wins": {"cat": 0, "mouse": 0, "draw": 0}}
        self.level_order = list(range(min(rounds, len(self.final_maps))))
        self.level = 0
        self.results = []
        self.env = VecEnv(self.final_maps, 1, seed=0xF1A1)
        self.bot = ScriptedPair(self.env, EXAMINER_SKILL, seed=7)
        self.actors = {
            "cat": FlatActor(self.policies["trained"][ck]["cat"], self.device),
            "mouse": FlatActor(self.policies["trained"][mk]["mouse"], self.device),
        }
        self._begin_episode()
        return {"type": "state", **self.state()}

    # ---------- RACE ----------

    def start_race(self, checkpoint: str = "trained", levels: list[int] | None = None) -> dict:
        """Three schools, one room, three panes.

        The three environments share a map, a spawn, a hearing-noise stream
        (`noise_tile = 1` tiles one draw across all three) and one set of sampling
        uniforms. That is the whole claim of the screen: every difference you can see is
        the policy, because nothing else was allowed to differ.
        """
        have = [s for s in SCHOOLS if s in self.policies[checkpoint]]
        if len(have) < 2:
            return {"type": "error", "message": f"need at least two schools at {checkpoint}"}
        self.mode = "race"
        self.race = have
        self.ctx = {"schools": have, "checkpoint": checkpoint,
                    "wins": {s: {"catch": 0, "escape": 0, "draw": 0} for s in have}}
        self.level_order = levels if levels is not None else list(range(len(self.maps)))
        self.level = 0
        self.results = []
        n = len(have)
        self.env = VecEnv(self.maps, n, seed=0x2ACE)
        self.env.noise_tile = 1
        self.bot = None
        self.actors = {
            "cat": FlatActor(np.stack([self.policies[checkpoint][s]["cat"] for s in have]),
                             self.device, assign=np.arange(n)),
            "mouse": FlatActor(np.stack([self.policies[checkpoint][s]["mouse"] for s in have]),
                               self.device, assign=np.arange(n)),
        }
        self._race_done = [None] * n
        self._begin_episode()
        return {"type": "state", **self.state()}

    def race_tick(self) -> list[dict]:
        e = self.env
        n = len(self.race)
        msgs = []
        if not e.done.all():
            oc, om = e.observe("cat"), e.observe("mouse")
            u = np.repeat(self.rng.random(1), n)      # one draw, shared by all three
            e.step(self.actors["cat"].act(oc, self.rng, u=u),
                   self.actors["mouse"].act(om, self.rng, u=u))
        lanes = []
        for i, school in enumerate(self.race):
            lanes.append({"school": school, **e.render(i)})
        payload = {"type": "race", "mode": "race", "lanes": lanes,
                   "level": self.level, "wins": self.ctx["wins"],
                   "schools": self.race, "checkpoint": self.ctx["checkpoint"]}
        idx = int(e.map_idx[0])
        if self._map_sent != idx:
            self._map_sent = idx
            payload["map"] = e.map_payload(0)
        msgs.append(payload)
        for i, school in enumerate(self.race):
            if e.done[i] and self._race_done[i] is None:
                r = ["", "catch", "escape", "timeout"][int(e.result[i])]
                self._race_done[i] = r
                # The tally is keyed by outcome, and a timeout IS the draw column.
                self.ctx["wins"][school]["draw" if r == "timeout" else r] += 1
                msgs.append({"type": "laneEnd", "school": school, "result": r,
                             "steps": int(e.step_n[i]), "level": self.level})
        return msgs

    def _begin_episode(self) -> None:
        assert self.env is not None
        idx = self.level_order[self.level % len(self.level_order)]
        self.env.reset(map_idx=np.array([idx]))
        if self.bot:
            self.bot.reset()
        self._map_sent = None
        if self.mode == "race":
            self._race_done = [None] * self.env.n
        self._refresh_shadow()

    def _refresh_shadow(self) -> None:
        """Re-read the policies at the start of each shadow episode, so the arena shows
        this minute's brain rather than the one training started with."""
        sch = self.train_school
        if sch is None:
            return
        try:
            self.actors = {r: FlatActor(sch.params(r), self.device) for r in ("cat", "mouse")}
        except Exception:
            pass                    # mid-update; the previous episode's actors will do

    def _actions(self):
        e = self.env
        oc, om = e.observe("cat"), e.observe("mouse")
        probs = {}
        if self.actors.get("cat") is not None:
            a_c = self.actors["cat"].act(oc, self.rng)
            probs["cat"] = self.actors["cat"].probs(oc)[0]
        else:
            a_c = self.bot.cat_act()
        if self.actors.get("mouse") is not None:
            a_m = self.actors["mouse"].act(om, self.rng)
            probs["mouse"] = self.actors["mouse"].probs(om)[0]
        else:
            a_m = self.bot.mouse_act()
        return a_c, a_m, probs

    def tick(self) -> list[dict]:
        """Advance one simulation step and return the messages to send."""
        e = self.env
        if e is None:
            return []
        msgs = []
        if not e.done[0]:
            a_c, a_m, probs = self._actions()
            e.step(a_c, a_m)
            frame = e.render(0, probs)
        else:
            frame = e.render(0)
        payload = {"type": "frame", "mode": self.mode, **self.ctx_public(), **frame,
                   "level": self.level, "levelIndex": self.level_order[self.level % len(self.level_order)]}
        idx = int(e.map_idx[0])
        if self._map_sent != idx:
            self._map_sent = idx
            payload["map"] = e.map_payload(0)
        msgs.append(payload)
        if e.done[0] and not self.results[self.level:self.level + 1]:
            r = ["", "catch", "escape", "timeout"][int(e.result[0])]
            self.results.append(r)
            if self.mode == "final":
                key = "cat" if r == "catch" else ("mouse" if r == "escape" else "draw")
                self.ctx["wins"][key] += 1
            msgs.append({"type": "episodeEnd", "result": r, "level": self.level,
                         "steps": int(e.step_n[0]), **self.ctx_public()})
        return msgs

    def advance(self) -> dict | None:
        """Move to the next arena, or finish the run."""
        if self.mode == "race":
            if self.level + 1 >= len(self.level_order):
                self.mode = "done"
                return {"type": "runEnd", **self.state()}
            self.level += 1
            self._begin_episode()
            return None
        if self.mode == "train":
            # Training has no end of run — the shadow keeps cycling the arenas with
            # whatever the optimiser has produced by now.
            self.level = (self.level + 1) % max(1, len(self.level_order))
            self.results = []
            self._begin_episode()
            return None
        if self.level + 1 >= len(self.level_order):
            self.mode = "done"
            return {"type": "runEnd", **self.state()}
        self.level += 1
        self._begin_episode()
        return None

    def ctx_public(self) -> dict:
        return {k: v for k, v in self.ctx.items() if k != "wins"} | (
            {"wins": self.ctx["wins"]} if "wins" in self.ctx else {})

    def state(self) -> dict:
        n = len(self.level_order) or 1
        catch = self.results.count("catch")
        escape = self.results.count("escape")
        return {
            "mode": self.mode, "playing": self.playing, "speed": self.speed,
            "level": self.level, "levels": n, "results": self.results,
            "catchRate": catch / max(1, len(self.results)),
            "escapeRate": escape / max(1, len(self.results)),
            **self.ctx_public(),
        }

    # ---------- TRAIN ----------

    def start_train(self, school: str, minutes: float, seed: int = 11) -> dict:
        if self._train_task and not self._train_task.done():
            return {"type": "error", "message": "training already running"}
        self.mode = "train"
        self.ctx = {"school": school, "minutes": minutes}
        self.train_school = None
        # A shadow episode, replayed from the policy as it currently stands. Training
        # itself runs thousands of episodes a second across a batch; showing one of them
        # raw would be a blur. This plays a single episode at a watchable pace with
        # whatever the optimiser has produced by now, so the arena visibly improves
        # while the panel shows why.
        self.env = VecEnv(self.maps, 1, seed=4242)
        self.bot = ScriptedPair(self.env, EXAMINER_SKILL, seed=13)
        self.level_order = list(range(len(self.maps)))
        self.level = 0
        self.results = []
        self.actors = {"cat": None, "mouse": None}
        self._begin_episode()
        loop = asyncio.get_running_loop()
        self._train_task = loop.run_in_executor(
            None, self._train_blocking, school, minutes, seed, loop)
        return {"type": "state", **self.state()}

    def _train_blocking(self, school: str, minutes: float, seed: int, loop) -> None:
        """Runs in a worker thread; telemetry is handed back through the event queue."""
        from .cmaes import CMAESSchool
        from .ga import GASchool
        from .ppo import PPOSchool
        cls = {"ppo": PPOSchool, "ga": GASchool, "cmaes": CMAESSchool}[school]

        def on_event(ev: dict) -> None:
            msg = {"type": "train", **ev}
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(self._train_events.put(msg), loop)

        s = cls(self.maps, MapSet(arena.EVAL_SEEDS[:8],
                                  arena.spread(self.nests, 8)), self.device,
                Budget(seconds=minutes * 60), seed=seed, on_event=on_event)
        s.setup()
        self.train_school = s
        s.train(eval_every=0.02)
        self.train_school = None
        on_event({"kind": "trainDone", "school": school})


async def replay(path: Path, send) -> None:
    """Play a journal back at its recorded pace — the safety net for a re-shoot."""
    t0 = time.perf_counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        msg = json.loads(line)
        at = msg.pop("at", 0.0)
        wait = at - (time.perf_counter() - t0)
        if wait > 0:
            await asyncio.sleep(wait)
        await send(msg)
