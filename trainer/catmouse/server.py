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
from . import env
from .league import EXAMINER_SKILL
from .nets import FlatActor, init_flat
from .school import Budget, load_timeline
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
        # Bumped by RESET TO ZERO. A live run cannot be killed outright -- the stop is
        # cooperative, so the optimiser keeps going until it next reads its budget -- and
        # the run outlives the wipe by up to a whole iteration. This is how the run finds
        # out that the session it belongs to no longer exists.
        self.epoch = 0
        # `ctx` is whatever the arena is currently playing, and `start_play` overwrites
        # it. This is the training run's own copy, so a detour into a checkpoint can be
        # walked back without the HUD losing the budget it was reporting.
        self.train_ctx: dict | None = None
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
        self._loop = None
        self._train_events: asyncio.Queue = asyncio.Queue(maxsize=512)
        self.train_school = None
        # Which frame of which school's timeline the arena is playing. None means
        # "whatever the optimiser has right now", which is the live shadow. The school is
        # kept with the index because `frames()` otherwise answers for whichever school
        # happens to be training, and an index validated against one reel would then load
        # weights out of another.
        self.pinned: int | None = None
        self.pinned_school: str | None = None
        self._shadow_at = 0
        self.race: list[str] = []
        self._race_done: list = []
        self.zeroed = False

    def busy_reason(self) -> str | None:
        """Why a run-level action must wait. Both of these rebind `self.maps` and
        `self.policies` underneath a School that is mid-run, and the live take would then
        write its result into whichever run happened to be loaded when it finished."""
        if self._train_task is not None and not self._train_task.done():
            return "a live run is going — stop it first"
        if self.runner is not None and self.runner.poll() is None:
            return "a full run is going — stop it first"
        return None

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
        # One scrub timeline per school, read off disk here and replaced in memory by a
        # live run as it goes, so the slider works on a finished run and on one that is
        # still training without the screen knowing the difference.
        self.timelines = {s: self._reel_for(run_dir, s) for s in SCHOOLS}
        # A pin addresses a frame of the run being left behind.
        self.pinned = None
        self.pinned_school = None
        self.tournament = self._load_json("tournament.json")
        self.progression = self._load_json("progression.json")
        self.budgets = self._load_json("budgets.json")
        self.highlights = self._load_json("highlights.json")

    @staticmethod
    def _reel_for(run_dir: Path, school: str) -> list[dict]:
        """The scrub reel for one school: the offline run's own, else the newest take.

        A take trained from the academy writes under `live/<stamp>/`, so after a restart
        the run being served had no reel at all and the slider the author had just been
        dragging was simply gone. A reel carries its own weights, so playing the last
        take's frames is self-contained and true — it is that take's brains, at that
        take's steps.
        """
        own = load_timeline(run_dir / school / "timeline.npz")
        if own:
            return own
        takes = sorted((run_dir / "live").glob(f"*/{school}/timeline.npz")) if (run_dir / "live").exists() else []
        return load_timeline(takes[-1]) if takes else []

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
            "academies": self.academies(),
            "shapingDefaults": dict(VecEnv.SHAPING_TRAINING),
            "rewards": self.reward_rules(),
            "zeroed": self.zeroed,
            "runDir": str(self.run_dir),
            "runTag": self.run_dir.name,
            "runs": self.list_runs(),
            "config": self.config,
            "training": self.runner_state(),
            "pinnedAt": self.pinned,
            **self.pin_info(),
        }

    def academies(self) -> dict:
        """Per school: the knobs its screen offers and the values they start at.

        Sent rather than hardcoded in the app, so a knob added to a config dataclass
        appears on the screen instead of quietly not existing."""
        from .cmaes import CMAESSchool, CMAConfig
        from .ga import GASchool, GAConfig
        from .ppo import PPOConfig, PPOSchool
        out = {}
        for cls, cfg in ((PPOSchool, PPOConfig()), (GASchool, GAConfig()),
                         (CMAESSchool, CMAConfig())):
            out[cls.key] = {
                "label": cls.label,
                "tunables": [dict(t, value=getattr(cfg, t["key"])) for t in cls.TUNABLES],
                # `frames()`, not `self.timelines` — a run in progress keeps its reel on
                # the School object, and a window that joins mid-run needs those frames
                # or its slider starts empty and then indexes into holes.
                "timeline": [
                    {k: v for k, v in f.items() if k not in ("cat", "mouse")}
                    for f in self.frames(cls.key)],
            }
        return out

    def reward_rules(self) -> list[dict]:
        """The terminal rewards, shown but not editable.

        These are the rules of the game and the scoreboard counts them directly, so a
        school trained under different ones is not playing the same sport as the others.
        The shaping above them is the part that is safe to tune, and the screen says
        which is which rather than presenting nine numbers as equally yours to move.
        """
        return [
            {"key": "R_CAT_CATCH", "label": "Tom catches her", "value": env.R_CAT_CATCH},
            {"key": "R_CAT_ESCAPED", "label": "Tom lets her home", "value": env.R_CAT_ESCAPED},
            {"key": "R_MOUSE_NEST", "label": "Jerry reaches a hole", "value": env.R_MOUSE_NEST},
            {"key": "R_MOUSE_CAUGHT", "label": "Jerry is caught", "value": env.R_MOUSE_CAUGHT},
            {"key": "R_TRAP", "label": "either one snaps a trap", "value": env.R_TRAP},
            {"key": "R_CAT_STEP", "label": "cost of a step", "value": env.R_CAT_STEP},
        ]
    # ---------- RESET ----------

    def reset_to_zero(self, seed: int = 0) -> dict:
        """Throw the run away and start from an untrained brain, in memory only.

        Every school's weights become a fresh draw from `init_flat` — the same
        distribution every school starts training from, so nothing here is a special
        "demo" policy. The measured artefacts go with them: the tournament, the
        checkpoint progression, the budgets and the highlight reel all describe policies
        that no longer exist, so continuing to show them would be the exact class of lie
        the rest of this app is built to avoid. The leaderboard, the grand final and the
        verdict each already have an honest empty state, and this is what puts them in it.

        **Nothing on disk is touched.** `runs/<tag>` still holds the trained checkpoints;
        restarting the server brings the whole run back.
        """
        if self.train_school is not None:
            # Cooperative stop: the training loop breaks as soon as its budget reads
            # full, which it checks once an iteration.
            self.train_school.budget.seconds = 1e-9
            self.train_school.budget.steps = 1
            # And let go of it. The budget above is mutated on the object itself, which
            # the worker thread holds, so the stop still lands -- but `frames()` serves
            # `train_school.timeline` in preference to everything else, so keeping the
            # reference meant `hello` went on handing out the reel of the run this wipe
            # just discarded. A window that reconnected got the graph back.
            self.train_school = None
        # The reels of runs that no longer exist. `timelines` is what keeps a finished
        # run scrubbable, and after this none of these policies are the ones that were
        # measured, so none of these reels describe anything on this session.
        self.timelines = {}
        rng = np.random.default_rng(seed)
        schools = [s for s in SCHOOLS if s in self.policies["trained"]] or list(SCHOOLS)
        for ck in CHECKPOINTS:
            self.policies[ck] = {
                s: {r: init_flat(1, rng)[0] for r in ("cat", "mouse")} for s in schools
            }
        self.tournament = None
        self.progression = None
        self.budgets = None
        self.highlights = None
        self.zeroed = True
        # Everything below is now the truth, and the run still finishing in the worker
        # thread must not undo it. See `_train_body`, which checks this before it writes
        # its best policies and its timeline back into the session.
        self.epoch += 1
        self.mode = "idle"
        self.train_ctx = None
        self.env = None
        self.actors = {}
        self.bot = None
        self.results = []
        self.level = 0
        self.level_seeds = None
        self._map_sent = None
        return self.hello()

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

    def training_msg(self) -> dict:
        """The one message that answers "is anything running".

        This fact used to travel only inside `hello`, which is sent on connect, after a run
        switch and after a scoring scan — never on the two events that change it. So a run
        could start and finish without any window being told, and every screen but one had
        no way to know.
        """
        return {"type": "training", **self.runner_state()}

    def push(self, msg: dict) -> None:
        """Queue a message for the broadcast pump from anywhere, including a worker
        thread. The pump drains this queue every tick."""
        loop = self._loop
        if loop is None:
            return
        # Always the threadsafe call. Deciding first whether we are on the loop's thread
        # meant asking for the current event loop from a worker thread, which raises —
        # and the suppression then swallowed the push itself, so the message announcing
        # that a run had FINISHED was the one message that never went out.
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._train_events.put(msg), loop)

    def runner_state(self) -> dict:
        # `train_school` is only published once the School object exists, which is a beat
        # after the take is launched — so the very first announcement said "live, school
        # unknown" and every reader that tested both fields concluded nothing was running.
        # `ctx["school"]` is set synchronously by `start_train` and is the honest answer
        # for that beat.
        live = self.train_school
        running = bool(self._train_task and not self._train_task.done())
        return {
            "full": self.runner is not None and self.runner.poll() is None,
            "live": running,
            "liveSchool": (live.key if live is not None
                           else (self.ctx.get("school") if running else None)),
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
        self._loop = loop
        self._runner_task = loop.create_task(self._follow_run(out, tag))
        self.push(self.training_msg())
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
            await self._train_events.put(self.training_msg())

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
        busy = self.busy_reason()
        if busy:
            return {"type": "error", "message": busy}
        run = RUNS / (tag or self.run_dir.name)
        if not run.exists():
            return {"type": "error", "message": f"no run at {run}"}
        self._loop = loop
        self._scorer_task = loop.create_task(self._run_score(run, checkpoint))
        self.push(self.training_msg())
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
        await self._train_events.put(self.training_msg())

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
        self.unpin()
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
        self.unpin()
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
        self.unpin()
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

    def start_pop_race(self, role: str = "cat") -> dict:
        """The best and the worst of one generation, in the same room, at the same time.

        The grid says a hundred and five of these are replaced; it cannot say WHY, because
        the arena only ever plays the winner. This puts six of them in twelve identical
        rooms -- same map, same spawns, same hearing noise, one shared draw of the action
        uniforms -- against the same opponent. Every difference on screen is the genome,
        because nothing else was allowed to differ. That is the same claim SIDE BY SIDE
        makes about the three schools, made about six members of one family.

        Racing the population is only possible while a run is live: the losers exist for
        one generation and are then written over, so they are kept as they are scored
        rather than reconstructed afterwards. A finished run cannot show this.
        """
        sch = self.train_school
        if sch is None:
            return {"type": "error", "message":
                    "the population race needs a live run — the brains that lost are not "
                    "kept once it ends"}
        show = getattr(sch, "showcase", {}).get(role)
        if not show:
            return {"type": "error", "message":
                    f"{getattr(sch, 'key', 'this school')} does not breed a population — "
                    "the race is for the evolutionary schools"}
        other = "mouse" if role == "cat" else "cat"
        rows = show["rows"]
        n = int(rows.shape[0])
        keys = ["g%d" % i for i in range(n)]
        self.mode = "race"
        self.unpin()
        self.level_seeds = None
        self.race = keys
        self.ctx = {
            "schools": keys, "checkpoint": "live",
            "wins": {k: {"catch": 0, "escape": 0, "draw": 0} for k in keys},
            "raceKind": "population", "raceRole": role, "raceSchool": getattr(sch, "key", None),
            "raceLanes": [{"key": keys[i], "rank": show["rank"][i], "fitness": show["fit"][i],
                           "genome": show["idx"][i]} for i in range(n)],
            "raceGen": show["gen"], "racePop": show["size"],
        }
        self.level_order = list(range(len(self.maps)))
        self.level = 0
        self.results = []
        self.env = VecEnv(self.maps, n, seed=0x2ACE)
        self.env.noise_tile = 1
        self.bot = None
        # One opponent, copied into every lane. Racing six cats against six different
        # mice would make the mouse the variable as well, and then a lane that lost would
        # not tell you whose fault it was.
        opp = np.stack([sch.best[other]] * n)
        self.actors = {
            role: FlatActor(np.stack([rows[i] for i in range(n)]), self.device,
                            assign=np.arange(n)),
            other: FlatActor(opp, self.device, assign=np.arange(n)),
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
        # A population race has no schools to look up, so what each lane IS has to travel
        # with the frames. Absent for the three-school race, which reads them from ORDER.
        for extra in ("raceKind", "raceRole", "raceSchool", "raceLanes", "raceGen", "racePop"):
            if extra in self.ctx:
                payload[extra] = self.ctx[extra]
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
        this minute's brain rather than the one training started with — unless a frame of
        the timeline is pinned, in which case it shows exactly that brain and holds it
        while the optimiser carries on behind.

        TRAIN MODE ONLY. `_begin_episode` calls this at the top of every episode, and
        `start_play`, `start_final` and `start_race` all end by beginning an episode — so
        without this guard, clicking UNTRAINED while a run was training silently played
        the optimiser's *current* weights under a caption naming the untrained ones, and
        the grand final played the trainee instead of the champion.
        """
        if self.mode != "train":
            return
        sch = self.train_school
        frame = self.frame_at(self.pinned, self.pinned_school) if self.pinned is not None else None
        if frame is not None:
            self.actors = {r: FlatActor(frame[r], self.device) for r in ("cat", "mouse")}
            return
        if sch is None:
            # The run has ended but the shadow keeps cycling arenas. Falling through here
            # left LIVE as a no-op that lit the chip and changed nothing on screen, so
            # come back to the last brain the run produced.
            last = self.frames(self.pinned_school)
            if last:
                self.actors = {r: FlatActor(last[-1][r], self.device) for r in ("cat", "mouse")}
            return
        try:
            self.actors = {r: FlatActor(sch.params(r), self.device) for r in ("cat", "mouse")}
        except Exception:
            pass                    # mid-update; the previous episode's actors will do

    # ---------- the scrub timeline ----------

    def frames(self, school: str | None = None) -> list[dict]:
        """The reel for a school: the live run's own, or the last one, or disk.

        A run in progress does not blank the reel the moment it starts. Its own timeline
        is empty until the first evaluation lands a couple of seconds later, and until
        then the previous reel is still the true answer to "what can I scrub" — otherwise
        pressing train wiped the checkpoint that was on screen mid-sentence.
        """
        sch = self.train_school
        key = school or (sch.key if sch is not None else self.ctx.get("school"))
        if sch is not None and sch.key == key and sch.timeline:
            return sch.timeline
        return self.timelines.get(key, [])

    def frame_at(self, i: int | None, school: str | None = None):
        fr = self.frames(school)
        if not fr or i is None:
            return None
        return fr[max(0, min(len(fr) - 1, int(i)))]

    def unpin(self) -> None:
        """Leave the reel. Every mode that takes the arena over calls this, because a pin
        that outlives its screen quietly replaces the policy the new screen names."""
        self.pinned = None
        self.pinned_school = None

    def shadow(self) -> dict:
        """Back to the shadow arena of the run that is already going.

        The one-way door this closes: `start_play` is what a checkpoint pill sends, and
        it moves the session out of `train`. That alone was enough to end scrubbing for
        the rest of the run — `pin` is refused outside train mode, `_refresh_shadow`
        returns early there, and `start_train` is the only other thing that sets the mode
        back but refuses while a run is going ("training already running"). So looking at
        UNTRAINED for two seconds during a forty-minute run cost the reel for the whole
        forty minutes, with the run still visibly training behind it.

        This is the way back. It rebuilds exactly what `start_train` builds for the
        arena — a single watchable episode over the twelve rooms — restores the run's own
        context, and starts an episode; `_begin_episode` calls `_refresh_shadow`, which
        now passes its guard and re-reads the optimiser's current weights. The optimiser
        itself is never touched: no task is started, stopped or restarted here.
        """
        if not (self._train_task and not self._train_task.done()):
            return {"type": "error", "message":
                    "nothing is training — the shadow arena is what a live run plays into"}
        self.mode = "train"
        self.level_seeds = None
        if self.train_ctx is not None:
            self.ctx = dict(self.train_ctx)
        self.env = VecEnv(self.maps, 1, seed=4242)
        self.bot = ScriptedPair(self.env, EXAMINER_SKILL, seed=13)
        self.level_order = list(range(len(self.maps)))
        self.level = 0
        self.results = []
        self.actors = {"cat": None, "mouse": None}
        self._begin_episode()
        return {"type": "state", **self.state()}

    def pin(self, i, school: str | None = None) -> dict:
        """Play the arena on the weights of one frame — or go back to live with None.

        This is the whole point of keeping every evaluation's brain: pick a step and the
        same rooms are played by the policy the run had at that step, so "they start
        stupid and learn to hunt" is something the screen shows rather than claims.
        """
        if i is None:
            self.unpin()
            self._refresh_shadow()
            return {"type": "pinned", "at": None, **self.pin_info()}
        key = school or (self.train_school.key if self.train_school is not None
                         else self.ctx.get("school"))
        if not key:
            return {"type": "error", "message": "pin needs a school — none is on screen yet"}
        fr = self.frames(key)
        if not fr:
            return {"type": "error", "message": f"{key} has no scrub timeline in this run yet"}
        # The reel only drives the arena in train mode — `_refresh_shadow` returns early
        # anywhere else. Recording the pin and answering "pinned" over an arena that went
        # on playing something else was the worst kind of lie: a control that reports
        # success and changes nothing.
        if self.mode != "train":
            return {"type": "error", "message":
                    "the reel plays into the training arena — press train, or open this "
                    "school's academy, to scrub it"}
        self.pinned = max(0, min(len(fr) - 1, int(i)))
        self.pinned_school = key
        self._refresh_shadow()
        return {"type": "pinned", "at": self.pinned, **self.pin_info()}

    def pin_info(self) -> dict:
        f = self.frame_at(self.pinned, self.pinned_school)
        return {"pinnedSchool": self.pinned_school,
                "frame": None if f is None else
                {k: v for k, v in f.items() if k not in ("cat", "mouse")},
                "frames": len(self.frames(self.pinned_school))}

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
                    steps: int | None = None, shaping: dict | None = None,
                    hyper: dict | None = None) -> dict:
        """One academy, on its own budget, with its own shaping and its own knobs.

        `minutes`, `steps`, or both — both means whichever runs out first.
        """
        if self._train_task and not self._train_task.done():
            return {"type": "error", "message": "training already running"}
        if self.runner is not None and self.runner.poll() is None:
            return {"type": "error", "message": "a full run is going — stop it first"}
        if not minutes and not steps:
            return {"type": "error", "message": "a live run needs a budget: minutes, steps, or both"}
        self.mode = "train"
        self.level_seeds = None
        # A new run means a new timeline, but it does not exist yet — see `frames()`. The
        # pin does have to go: it addresses a frame of the run being replaced.
        self.unpin()
        self.ctx = {"school": school, "minutes": minutes, "steps": steps,
                    "shaping": shaping, "hyper": hyper,
                    "budget": Budget(seconds=None if minutes is None else minutes * 60,
                                     steps=steps).describe()}
        # Kept whole, because `shadow()` has to put it back exactly: `ctx` rides on every
        # frame and every state, so a rebuilt approximation would change the budget, the
        # shaping and the knobs the HUD reports mid-run.
        self.train_ctx = dict(self.ctx)
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
        self._loop = loop
        self._train_task = loop.run_in_executor(
            None, self._train_blocking, school, minutes, seed, loop, steps, shaping, hyper)
        self._train_task.add_done_callback(lambda _f: self.push(self.training_msg()))
        # Say so immediately, to every window, on every screen.
        self.push(self.training_msg())
        return {"type": "state", **self.state()}

    def _train_blocking(self, *a, **kw) -> None:
        """Runs in a worker thread. Nothing awaits the future it returns, so an exception
        in here used to be invisible: the arena froze on the last actors, the HUD held its
        last numbers, and STOP cheerfully answered "the run still saves its checkpoints"
        about a thread that had been dead for ten minutes."""
        school = a[0] if a else kw.get("school", "?")
        loop = a[3] if len(a) > 3 else kw.get("loop")
        try:
            self._train_body(*a, **kw)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            msg = f"{type(exc).__name__}: {exc}"
            for ev in ({"type": "error", "message": f"the {school} run stopped: {msg}"},
                       {"type": "train", "kind": "trainDone", "school": school,
                        "failed": True, "message": msg}):
                with contextlib.suppress(Exception):
                    asyncio.run_coroutine_threadsafe(self._train_events.put(ev), loop)
        finally:
            self.train_school = None
            # The announcement is NOT made here: this still runs inside the worker, and
            # the executor future is not `done()` until the function returns — so a
            # message sent from here reports the run as still live. `start_train` hangs a
            # done-callback on the future instead, which fires on the loop, after.
            pass

    def _train_body(self, school: str, minutes: float | None, seed: int, loop,
                    steps: int | None = None, shaping: dict | None = None,
                    hyper: dict | None = None) -> None:
        """Telemetry is handed back through the event queue."""
        from .cmaes import CMAESSchool
        from .ga import GASchool
        from .ppo import PPOSchool
        cls = {"ppo": PPOSchool, "ga": GASchool, "cmaes": CMAESSchool}[school]

        # The generation this run belongs to. RESET TO ZERO bumps it, and everything
        # below asks whether it is still the current one before it says or writes
        # anything -- see `Session.epoch`.
        epoch = self.epoch

        def alive() -> bool:
            return self.epoch == epoch

        def emit(ev: dict) -> None:
            msg = {"type": "train", **ev}
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(self._train_events.put(msg), loop)

        def on_event(ev: dict) -> None:
            # A run that has been reset away has to stop talking. Its evaluations went on
            # arriving for another iteration after the wipe, and each one was appended to
            # the reel the wipe had just emptied -- the screen rebuilding a run it had
            # been told no longer existed.
            if not alive():
                return
            emit(ev)

        # A live take is a real run — at a step budget it can be hours of work — so it
        # writes its checkpoints like any other. Under `live/<stamp>/` rather than
        # `<school>/`, because the run directory being served belongs to the offline
        # trainer and a take must never overwrite what the leaderboard is reading.
        out = self.run_dir / "live" / time.strftime("%Y-%m-%dT%H-%M-%S")
        s = cls(self.maps, MapSet(arena.EVAL_SEEDS[:8],
                                  arena.spread(self.nests, 8)), self.device,
                Budget(seconds=None if minutes is None else minutes * 60, steps=steps),
                seed=seed, on_event=on_event, out_dir=out, shaping=shaping, hyper=hyper)
        # `train()` calls `setup()` itself. Building it here as well made every live take
        # construct two complete vector environments and two optimisers — at the academy
        # slider's top setting that is a real doubling of peak memory — and put a policy
        # in the arena that was then thrown away. Publish the school first instead, so the
        # shadow has something to ask for the moment setup finishes.
        self.train_school = s
        s.train(eval_every=0.02)
        # Everything from here writes the run back INTO the session, and RESET TO ZERO may
        # have emptied that session while the optimiser was still finishing its iteration.
        # Doing it anyway put trained weights back under `best` and rebuilt the reel, next
        # to a `zeroed` flag still reading True and school cards showing only UNTRAINED --
        # a session claiming to hold nothing while holding the run it had just thrown out.
        if not alive():
            # Announced, not swallowed. The wipe does not reach the disk, and `train()`
            # has been writing checkpoints the whole way, so the work is still there and
            # the one useful thing to say is where.
            emit({"kind": "trainDone", "school": school, "discarded": True,
                  "savedTo": str(out), "frames": 0, "best": {},
                  "message": "reset to zero threw this run away while it was training"})
            return
        # The live run picked a best cat and a best mouse of its own. Hand them to the
        # session so `play best` right after a take plays what was just watched being
        # trained, rather than the last run scored off disk.
        with contextlib.suppress(Exception):
            self.policies.setdefault("best", {})[school] = {
                r: np.array(v, np.float32, copy=True)
                for r, v in s.checkpoints["best"].items()}
        # Keep the timeline reachable after the school object goes: the slider is the
        # point of the run and it must still work once the run has stopped.
        self.timelines[school] = list(s.timeline)
        on_event({"kind": "trainDone", "school": school, "savedTo": str(out),
                  "frames": len(s.timeline),
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
