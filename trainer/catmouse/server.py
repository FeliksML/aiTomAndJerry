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
import os
import signal
import subprocess
import sys
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

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
TRAIN_PY = ROOT / "trainer" / "scripts" / "train.py"
SCORE_STEPS = (("tournament_run.py", ["--reps", "40"]), ("highlights.py", ["--episodes", "400"]))

# BEST is written by the trainer alongside the three Academy checkpoints: the policy
# each role was actually strongest at, which is not always the one it finished on. It is
# absent from older runs, and `load_run` simply returns nothing for it in that case.
CHECKPOINTS = ("untrained", "half", "trained", "best")


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
        self.device = device
        self.journal = journal
        self.load_run_dir(run_dir)

        # The offline trainer, driven from the app. One supervisor process at a time; the
        # per-school telemetry is tailed out of the JSONL each child already writes, so
        # a run started from the app and a run started from a terminal produce byte-for-
        # byte the same artefacts.
        self.runner: subprocess.Popen | None = None
        self.runner_meta: dict = {}
        self.runner_schools: dict = {}
        self.scorer: subprocess.Popen | None = None
        self._runner_task = None
        self._scorer_task = None

        self.mode = "idle"
        self.speed = 4.0
        self.playing = True
        self.rng = np.random.default_rng(0)
        self.env: VecEnv | None = None
        self.actors: dict = {}
        self.bot: ScriptedPair | None = None
        self.level = 0
        self.level_order: list[int] = []
        self.level_seeds: list[int] | None = None
        self.results: list[str] = []
        self.ctx: dict = {}
        self._map_sent = None
        self._train_task = None
        self._train_events: asyncio.Queue = asyncio.Queue(maxsize=512)
        self.train_school = None
        self._shadow_at = 0
        self.race: list[str] = []
        self._race_done: list = []

    def load_run_dir(self, run_dir: Path) -> None:
        """Point the session at a run. Called at start-up and again whenever the app
        switches runs, so a run trained on camera can be watched without a restart."""
        self.run_dir = run_dir
        # How many holes this run was trained with. A room with one hole plays a
        # completely different game from a room with two, so the server takes it from
        # the run rather than assuming.
        cfg = self._load_json("config.json") or {}
        self.config = cfg
        self.nests = arena.parse_nests(cfg.get("nests", arena.DEFAULT_NESTS))
        self.maps = MapSet(arena.TRAIN_SEEDS, arena.spread(self.nests, len(arena.TRAIN_SEEDS)))
        self.final_maps = MapSet(arena.FINAL_SEEDS, arena.spread(self.nests, len(arena.FINAL_SEEDS)))
        self.policies = {ck: load_run(run_dir, ck) for ck in CHECKPOINTS}
        self.tournament = self._load_json("tournament.json")
        self.progression = self._load_json("progression.json")
        self.budgets = self._load_json("budgets.json")
        self.highlights = self._load_json("highlights.json")

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
            "runTag": self.run_dir.name,
            "runs": self.list_runs(),
            "config": self.config,
            "training": self.runner_state(),
        }

    def list_runs(self) -> list[dict]:
        """Every run on disk, newest first, with enough for the app to say what it is."""
        out = []
        if not RUNS.exists():
            return out
        for d in RUNS.iterdir():
            if not d.is_dir() or d.name == "journals":
                continue
            schools = sorted(x.name for x in d.iterdir()
                             if x.is_dir() and (x / "checkpoints.npz").exists())
            if not schools and not (d / "config.json").exists():
                continue
            cfg = {}
            with contextlib.suppress(Exception):
                cfg = json.loads((d / "config.json").read_text())
            out.append({
                "tag": d.name, "schools": schools,
                "budget": cfg.get("budget"), "startedAt": cfg.get("startedAt"),
                "scored": (d / "tournament.json").exists(),
                "current": d == self.run_dir,
                "mtime": d.stat().st_mtime,
            })
        return sorted(out, key=lambda r: r["mtime"], reverse=True)

    # ---------- the offline trainer, driven from the app ----------

    def runner_state(self) -> dict:
        live = self.train_school
        return {
            "full": self.runner is not None and self.runner.poll() is None,
            "live": bool(self._train_task and not self._train_task.done()),
            "liveSchool": live.key if live is not None else None,
            "scoring": self.scorer is not None and self.scorer.poll() is None,
            **self.runner_meta,
        }

    def start_train_all(self, loop, steps=None, minutes=None, envs=None, seed=7,
                        tag="v5", nests="2", device="auto") -> dict:
        """Run the real trainer — three schools, three processes — and stream it back.

        This is exactly `train.py` as a terminal would launch it, not a re-implementation:
        the app should not be able to produce a run that differs from a hand-typed one.
        Its telemetry is read from the `events.jsonl` each child already writes.
        """
        if self.runner is not None and self.runner.poll() is None:
            return {"type": "error", "message": "a full run is already going"}
        if self._train_task and not self._train_task.done():
            return {"type": "error", "message": "a live run is going — stop it first"}
        if not steps and not minutes:
            return {"type": "error", "message": "a run needs a budget: steps, minutes, or both"}
        tag = "".join(c for c in str(tag) if c.isalnum() or c in "-_.") or "v5"
        out = RUNS / tag
        cmd = [sys.executable, str(TRAIN_PY), "--tag", tag, "--seed", str(int(seed)),
               "--nests", str(nests), "--device", device]
        if steps:
            cmd += ["--steps", str(steps)]
        if minutes:
            cmd += ["--minutes", str(minutes)]
        if envs:
            cmd += ["--envs", str(int(envs))]
        # Its own process group, so stopping the run reaches all three children rather
        # than only the supervisor.
        self.runner = subprocess.Popen(cmd, cwd=str(ROOT), start_new_session=True)
        self.runner_meta = {"tag": tag, "steps": steps, "minutes": minutes,
                            "envs": envs, "seed": seed, "nests": nests,
                            "startedAt": time.strftime("%H:%M:%S")}
        self.runner_schools = {}
        self._runner_task = loop.create_task(self._follow_run(out, tag))
        return {"type": "trainAllStarted", **self.runner_meta,
                "schools": list(SCHOOLS), "out": str(out)}

    async def _follow_run(self, out: Path, tag: str) -> None:
        """Tail each school's JSONL and forward every event, then report the exit."""
        handles: dict[str, object] = {}
        try:
            while True:
                for school in SCHOOLS:
                    f = handles.get(school)
                    if f is None:
                        path = out / school / "events.jsonl"
                        if not path.exists():
                            continue
                        f = handles[school] = path.open()
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        with contextlib.suppress(Exception):
                            ev = json.loads(line)
                            ev.setdefault("school", school)
                            self.runner_schools[school] = ev
                            await self._train_events.put({"type": "train", "run": tag, **ev})
                done = self.runner is None or self.runner.poll() is not None
                if done and all(s in handles for s in SCHOOLS):
                    # One last sweep, so the closing events are not lost to the poll gap.
                    await asyncio.sleep(0.4)
                    for school, f in handles.items():
                        for line in f:
                            with contextlib.suppress(Exception):
                                ev = json.loads(line)
                                ev.setdefault("school", school)
                                await self._train_events.put({"type": "train", "run": tag, **ev})
                    break
                if done:
                    break
                await asyncio.sleep(0.35)
        finally:
            for f in handles.values():
                with contextlib.suppress(Exception):
                    f.close()
            code = self.runner.poll() if self.runner else None
            await self._train_events.put({"type": "trainAllDone", "run": tag, "code": code,
                                          "out": str(out)})

    def stop_train_all(self) -> dict:
        """Ask, do not kill. The children finish the iteration they are on and save."""
        if self.runner is None or self.runner.poll() is not None:
            return {"type": "error", "message": "no full run is going"}
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(self.runner.pid), signal.SIGTERM)
        return {"type": "trainStopping", "scope": "all"}

    # ---------- scoring, also from the app ----------

    def start_score(self, loop, tag: str | None = None, checkpoint: str = "trained") -> dict:
        if self.scorer is not None and self.scorer.poll() is None:
            return {"type": "error", "message": "already scoring"}
        if self.runner is not None and self.runner.poll() is None:
            return {"type": "error", "message": "wait for the run to finish before scoring it"}
        run = RUNS / (tag or self.run_dir.name)
        if not run.exists():
            return {"type": "error", "message": f"no run at {run}"}
        self._scorer_task = loop.create_task(self._run_score(run, checkpoint))
        return {"type": "scoreStarted", "run": run.name, "checkpoint": checkpoint}

    async def _run_score(self, run: Path, checkpoint: str) -> None:
        """The tournament and the highlight scan, in order, with their output relayed.

        Both take minutes, and both used to be terminal-only — which meant a run trained
        on camera had no leaderboard and no grand final until somebody typed something.
        """
        ok = True
        for script, extra in SCORE_STEPS:
            args = [sys.executable, str(ROOT / "trainer" / "scripts" / script),
                    "--run", str(run)] + extra
            if script == "tournament_run.py":
                args += ["--checkpoint", checkpoint]
            await self._train_events.put({"type": "scoreStep", "step": script, "run": run.name})
            proc = await asyncio.create_subprocess_exec(
                *args, cwd=str(ROOT), stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
            self.scorer = proc
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    await self._train_events.put({"type": "scoreLine", "line": line})
            code = await proc.wait()
            if code != 0:
                ok = False
                await self._train_events.put(
                    {"type": "scoreLine", "line": f"{script} exited {code}"})
                break
        self.scorer = None
        # Re-read whatever the scan just wrote, so the leaderboard and the grand final
        # are live without a reload.
        with contextlib.suppress(Exception):
            self.load_run_dir(run)
        await self._train_events.put({"type": "scoreDone", "run": run.name, "ok": ok})

    # ---------- PLAY ----------

    def start_play(self, school: str, checkpoint: str, opponent: str = "self",
                   levels: list[int] | None = None, mouse_school: str | None = None,
                   seeds: list[int] | None = None) -> dict:
        """One school's pair, or a named cross-school pairing, over a list of arenas.

        `mouse_school` lets the cat come from one school and the mouse from another —
        the highlight reel needs it, because the scan that scored those episodes ran the
        champion cat against the champion *mouse*, which is usually a different school.

        `seeds` gives one RNG seed per entry in `levels`. With it, an episode depends on
        nothing but (arena, seed): the environment stream and the action-sampling stream
        are both re-seeded at the start of that episode, so the same pair of numbers
        replays the same episode frame for frame, alone or in any order. Without it the
        streams simply run on, which is what a straight twelve-arena run wants.
        """
        pol = self.policies[checkpoint].get(school)
        if pol is None:
            return {"type": "error", "message": f"no {school} @ {checkpoint} in this run"}
        mpol = pol
        if mouse_school and mouse_school != school:
            mpol = self.policies[checkpoint].get(mouse_school)
            if mpol is None:
                return {"type": "error",
                        "message": f"no {mouse_school} @ {checkpoint} in this run"}
        self.mode = "play"
        self.ctx = {"school": school, "checkpoint": checkpoint, "opponent": opponent,
                    "mouseSchool": mouse_school or school}
        self.level_order = levels if levels is not None else list(range(len(self.maps)))
        self.level_seeds = list(seeds) if seeds else None
        self.level = 0
        self.results = []
        self.env = VecEnv(self.maps, 1, seed=1234)
        self.bot = ScriptedPair(self.env, EXAMINER_SKILL, seed=99)
        self.actors = {
            "cat": FlatActor(pol["cat"], self.device),
            "mouse": FlatActor(mpol["mouse"], self.device),
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
        self.level_seeds = None
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
        self.level_seeds = None
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
        if self.level_seeds:
            # Both streams, because both feed the episode: `env.rng` draws the hearing
            # noise and `self.rng` draws the action samples. Re-seeding them here is what
            # makes (arena, seed) a complete description of an episode — otherwise the
            # streams carry over from whatever played before, and the same arena replays
            # differently depending on how long the previous episode ran.
            sd = int(self.level_seeds[self.level % len(self.level_seeds)])
            self.env.rng = np.random.default_rng(sd)
            self.rng = np.random.default_rng(sd ^ 0x5EED)
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
        # In train mode the shadow arena must not step until there is a policy to step
        # it with. `start_train` returns before the optimiser thread exists (setup alone
        # is ~2.5 s for PPO), and `_refresh_shadow` only runs at episode boundaries, so
        # the whole first episode used to fall through to the scripted Examiner —
        # a skilled chase at t=0, under a caption promising this run's own policy, with
        # the play getting visibly *worse* once the real network arrived.
        if self.mode == "train" and (self.actors.get("cat") is None
                                     or self.actors.get("mouse") is None):
            self._refresh_shadow()
            if self.actors.get("cat") is None or self.actors.get("mouse") is None:
                return [{"type": "trainWait", "mode": self.mode, **self.ctx_public()}]
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

    def start_train(self, school: str, minutes: float | None, seed: int = 11,
                    steps: int | None = None) -> dict:
        """`minutes`, `steps`, or both — both means whichever runs out first."""
        if self._train_task and not self._train_task.done():
            return {"type": "error", "message": "training already running"}
        if not minutes and not steps:
            return {"type": "error", "message": "a live run needs a budget: minutes, steps, or both"}
        self.mode = "train"
        self.level_seeds = None
        self.ctx = {"school": school, "minutes": minutes, "steps": steps,
                    "budget": Budget(seconds=None if minutes is None else minutes * 60,
                                     steps=steps).describe()}
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
            None, self._train_blocking, school, minutes, seed, loop, steps)
        return {"type": "state", **self.state()}

    def _train_blocking(self, school: str, minutes: float | None, seed: int, loop,
                        steps: int | None = None) -> None:
        """Runs in a worker thread; telemetry is handed back through the event queue."""
        from .cmaes import CMAESSchool
        from .ga import GASchool
        from .ppo import PPOSchool
        cls = {"ppo": PPOSchool, "ga": GASchool, "cmaes": CMAESSchool}[school]

        def on_event(ev: dict) -> None:
            msg = {"type": "train", **ev}
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(self._train_events.put(msg), loop)

        # A live take is a real run — at a step budget it can be hours of work — so it
        # writes its checkpoints like any other. Under `live/<stamp>/` rather than
        # `<school>/`, because the run directory being served belongs to the offline
        # trainer and a take must never overwrite what the leaderboard is reading.
        out = self.run_dir / "live" / time.strftime("%Y-%m-%dT%H-%M-%S")
        s = cls(self.maps, MapSet(arena.EVAL_SEEDS[:8],
                                  arena.spread(self.nests, 8)), self.device,
                Budget(seconds=None if minutes is None else minutes * 60, steps=steps),
                seed=seed, on_event=on_event, out_dir=out)
        s.setup()
        self.train_school = s
        s.train(eval_every=0.02)
        # The live run picked a best cat and a best mouse of its own. Hand them to the
        # session so `play best` right after a take plays what was just watched being
        # trained, rather than the last run scored off disk.
        with contextlib.suppress(Exception):
            self.policies.setdefault("best", {})[school] = {
                r: np.array(v, np.float32, copy=True)
                for r, v in s.checkpoints["best"].items()}
        self.train_school = None
        on_event({"kind": "trainDone", "school": school, "savedTo": str(out),
                  "best": {r: v.get("pick") for r, v in (s.best_report or {}).items()}})


async def replay(path: Path, send) -> None:
    """Play a journal back at its recorded pace — the safety net for a re-shoot.

    Every message is flagged `replay`, because a replay is not a live trainer and the app
    should not claim it is. Nothing here listens: pause, speed and skip have no clock to
    act on, so the screen says so rather than letting a key press look ignored.
    """
    t0 = time.perf_counter()
    await send({"type": "replay", "source": path.name})
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        msg = json.loads(line)
        at = msg.pop("at", 0.0)
        msg["replay"] = True
        wait = at - (time.perf_counter() - t0)
        if wait > 0:
            await asyncio.sleep(wait)
        await send(msg)
    await send({"type": "replayEnd", "source": path.name})
