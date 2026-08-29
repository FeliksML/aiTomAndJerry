"""What every school has in common: a budget, three checkpoints, and one scoreboard.

A "school" trains one cat and one mouse with one algorithm. The three schools differ
only in `iteration()`; everything that decides whether the comparison is fair lives
here so no algorithm can accidentally get a different deal:

  * the same arenas to train on and the same held-out arenas to be scored on
  * the same starting weight distribution
  * the same Examiner, at the same skill, on the same seeds
  * the same three checkpoints — UNTRAINED (before a single update), HALF-TRAINED
    (halfway through the budget) and TRAINED (at the end) — which are exactly the
    three the Academy screen walks through
  * both budget clocks recorded on every log line, so "equal wall-clock" and
    "equal environment steps" can both be read off one run afterwards

Telemetry is pushed through `on_event`, which the trainer writes to JSONL and the
live server forwards to the browser. The algorithm-specific payloads are what the
on-screen explainers draw, so they are produced during training rather than faked.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import arena
from .vec import MapSet

CHECKPOINT_NAMES = ("untrained", "half", "trained")


@dataclass
class Budget:
    """Two clocks, either of which can end the run.

    Wall-clock is the primary axis for the fairness protocol — three schools, identical
    minutes, same machine. A step budget is the other way to ask the question: give all
    three the same number of environment steps and let the wall-clock fall where it may.

    With both set the run ends at whichever comes first, so `--minutes 60 --steps 500M`
    reads as "500M steps, but never more than an hour on camera".
    """
    seconds: float | None = None
    steps: int | None = None

    def fraction(self, elapsed: float, steps: int) -> float:
        fr = 0.0
        if self.seconds:
            fr = max(fr, elapsed / self.seconds)
        if self.steps:
            fr = max(fr, steps / self.steps)
        return min(1.0, fr)

    def eta(self, elapsed: float, steps: int, sps: float) -> float | None:
        """Seconds left on whichever clock runs out first. None if nothing can be said —
        no budget at all, or a step budget with no rate measured yet."""
        left = []
        if self.seconds:
            left.append(max(0.0, self.seconds - elapsed))
        if self.steps and sps > 0:
            left.append(max(0.0, (self.steps - steps) / sps))
        return min(left) if left else None

    def describe(self) -> str:
        parts = []
        if self.steps:
            parts.append(human_steps(self.steps) + " steps")
        if self.seconds:
            parts.append(f"{self.seconds / 60:.0f} min")
        return " or ".join(parts) if parts else "unbounded"


def human_steps(n: float) -> str:
    """1_500_000 -> '1.5M'. Used on screen, in the console and in the run's own JSON, so
    the number the author typed is the number that comes back."""
    n = float(n)
    for cut, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= cut:
            v = n / cut
            return f"{v:.0f}{suffix}" if v >= 100 or v == int(v) else f"{v:.1f}{suffix}"
    return str(int(n))


def parse_steps(text) -> int | None:
    """Accept what a person actually types: 500M, 1.5b, 2e8, 500_000_000, 500000000."""
    if text is None:
        return None
    t = str(text).strip().replace("_", "").replace(",", "")
    if not t:
        return None
    mult = {"k": 1e3, "m": 1e6, "b": 1e9, "g": 1e9}.get(t[-1].lower())
    if mult:
        t = t[:-1]
    else:
        mult = 1.0
    return int(round(float(t) * mult))


@dataclass
class RunState:
    steps: int = 0
    iters: int = 0
    started: float = field(default_factory=time.perf_counter)
    # A sliding window over the last few seconds. The average since t=0 is the wrong
    # number to put on screen: setup, the first evaluation and any thermal throttling
    # are baked into it forever, so the rate keeps drifting long after the run settled.
    _marks: list = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def mark(self, window: float = 12.0) -> float:
        """Record where we are and return steps/second over the recent window."""
        now = time.perf_counter()
        self._marks.append((now, self.steps))
        while len(self._marks) > 2 and now - self._marks[0][0] > window:
            self._marks.pop(0)
        t0, s0 = self._marks[0]
        dt = now - t0
        return (self.steps - s0) / dt if dt > 0.5 else self.steps / max(1e-9, self.elapsed)


class School:
    key = "base"
    label = "Base"

    def __init__(self, maps: MapSet, eval_maps: MapSet, device: torch.device,
                 budget: Budget, seed: int = 0, on_event=None, out_dir: Path | None = None):
        self.maps = maps
        self.eval_maps = eval_maps
        self.device = device
        self.budget = budget
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.on_event = on_event or (lambda ev: None)
        self.out = out_dir
        self.run = RunState()
        self.checkpoints: dict[str, dict[str, np.ndarray]] = {}
        self.history: list[dict] = []
        self._next_eval = 0.0
        # The best policy each role reached at any point in the run, which is not
        # necessarily the one it ended on — self-play is not monotone, and a cat can be
        # walked backwards by a mouse that got good after it did.
        #
        # Named `peak`, not `best`: the population schools already own a `self.best`,
        # which holds their current champion *weight vectors* and is what `params()`
        # returns. A high-water-mark dict under the same name silently replaced those
        # arrays the first time a run scored itself.
        self.peak: dict[str, dict] = {}
        self.best_report: dict = {}
        self._sps = 0.0
        self._last_progress = 0.0
        self._train_wall = 0.0
        self._stop = False

    # ---------- subclass contract ----------

    def setup(self) -> None:
        """Build the initial policies. Must leave `params('cat')` / `params('mouse')` valid."""
        raise NotImplementedError

    def iteration(self) -> dict:
        """Advance training once. Return the algorithm's telemetry for this step and
        add the environment steps consumed to `self.run.steps`."""
        raise NotImplementedError

    def params(self, role: str) -> np.ndarray:
        """The current best flat weight vector for a role."""
        raise NotImplementedError

    # ---------- shared machinery ----------

    def emit(self, kind: str, **payload) -> dict:
        """Every event carries where the run is, on both clocks.

        The app used to be able to say *what* an optimiser was doing and nothing at all
        about *how far in* it was — no iteration, no step count against a target, no
        rate. One place to add it is here, so no event type can forget."""
        elapsed = self.run.elapsed
        eta = self.budget.eta(elapsed, self.run.steps, self._sps)
        ev = {
            "kind": kind, "school": self.key, "iter": self.run.iters,
            "steps": self.run.steps, "wall": round(elapsed, 3),
            "frac": round(self.budget.fraction(elapsed, self.run.steps), 4),
            "targetSteps": self.budget.steps, "targetSeconds": self.budget.seconds,
            "sps": round(self._sps), "eta": None if eta is None else round(eta, 1),
            **payload,
        }
        self.on_event(ev)
        return ev

    def snapshot(self, name: str) -> None:
        self.checkpoints[name] = {
            "cat": np.array(self.params("cat"), np.float32, copy=True),
            "mouse": np.array(self.params("mouse"), np.float32, copy=True),
        }
        self.emit("checkpoint", name=name)

    def score(self, tag: str, reps: int = 6) -> dict:
        """Both roles against the Examiner on the held-out arenas."""
        c = arena.examiner_score(self.eval_maps, self.params("cat"), "cat",
                                 self.device, seed=self.seed + 17, reps=reps)
        m = arena.examiner_score(self.eval_maps, self.params("mouse"), "mouse",
                                 self.device, seed=self.seed + 23, reps=reps)
        row = {
            "tag": tag,
            "catExam": c.catch_rate, "catExamCi": c.rate_ci("catch")[1:],
            "mouseExam": m.escape_rate, "mouseExamCi": m.rate_ci("escape")[1:],
            "catTraps": c.trap_hits, "mouseTraps": m.trap_hits,
            "catSteps": c.mean_steps, "mouseSteps": m.mean_steps,
        }
        self.history.append({**row, "steps": self.run.steps, "wall": self.run.elapsed})
        self.emit("eval", **row)
        self._consider_best(row)
        return row

    ROLE_KEYS = {"cat": ("catExam", "catExamCi"), "mouse": ("mouseExam", "mouseExamCi")}

    def _consider_best(self, row: dict) -> None:
        """Keep the high-water mark for each role, judged on the CI *lower bound*.

        A run scores itself fifty-odd times on six repeats per arena. Taking the best
        point estimate out of fifty noisy measurements is a lottery the noise usually
        wins — the "best" cat would mostly be the one that drew the easiest evaluation.
        Ranking on the lower end of the interval asks for evidence instead: a policy has
        to be good enough that even its pessimistic reading beats the incumbent's.
        """
        for role, (key, ci_key) in self.ROLE_KEYS.items():
            lo = float(row[ci_key][0])
            cur = self.peak.get(role)
            if cur is not None and lo <= cur["lo"]:
                continue
            self.peak[role] = {
                "role": role, "tag": row["tag"], "rate": float(row[key]), "lo": lo,
                "hi": float(row[ci_key][1]), "steps": self.run.steps,
                "iter": self.run.iters, "wall": round(self.run.elapsed, 2),
                "flat": np.array(self.params(role), np.float32, copy=True),
            }
            self.emit("best", role=role, rate=float(row[key]), lo=lo, tag=row["tag"])

    def _finalise_best(self, reps: int = 12) -> None:
        """Peak versus finish, re-measured on a seed neither of them was picked on.

        The high-water mark was chosen by looking at the same numbers many times, so it
        is biased upwards by construction. Re-scoring both candidates once, on a fresh
        seed and twice the repeats, is what turns "the best score we saw" into "the
        policy that is actually better", and both readings are written down so the
        choice can be argued with.
        """
        picks, report = {}, {}
        for role in ("cat", "mouse"):
            which = "catch" if role == "cat" else "escape"
            final = np.array(self.params(role), np.float32, copy=True)
            cands = {"final": final}
            b = self.peak.get(role)
            if b is not None and b["steps"] != self.run.steps:
                cands["peak"] = b["flat"]
            runoff = {}
            for name, flat in cands.items():
                o = arena.examiner_score(self.eval_maps, flat, role, self.device,
                                         seed=self.seed + 101, reps=reps)
                ci = o.rate_ci(which)
                runoff[name] = {"rate": ci[0], "lo": ci[1], "hi": ci[2]}
            pick = max(runoff, key=lambda k: runoff[k]["lo"])
            picks[role] = cands[pick]
            report[role] = {
                "pick": pick, "runoff": runoff, "reps": reps,
                "peakSeenAt": None if b is None else {
                    "tag": b["tag"], "steps": b["steps"], "iter": b["iter"],
                    "wall": b["wall"], "rate": b["rate"], "lo": b["lo"], "hi": b["hi"]},
                "finalAt": {"steps": self.run.steps, "iter": self.run.iters,
                            "wall": round(self.run.elapsed, 2)},
            }
            # `fromSteps`, not `steps`: every event already carries `steps` meaning "how
            # far this run has got", and overloading it here to mean "where the winning
            # policy came from" made the live progress row jump back to zero whenever
            # the peak happened to be an early one.
            self.emit("bestFinal", role=role, pick=pick,
                      rate=runoff[pick]["rate"], lo=runoff[pick]["lo"],
                      fromSteps=(report[role]["peakSeenAt"]["steps"] if pick == "peak"
                                 else self.run.steps))
        self.checkpoints["best"] = picks
        self.best_report = report

    def request_stop(self) -> None:
        """End the run at the next iteration boundary.

        Not a kill: a long budget is worth being able to cut short — you have seen
        enough, or the take is over — but the run still snapshots, still holds the
        peak-versus-finish run-off, and still writes its checkpoints. Stopping a
        500M-step run at 300M should leave a usable school behind, not a dead process
        and an empty directory."""
        self._stop = True

    def train(self, eval_every: float = 0.05, progress_every: float = 1.0) -> dict:
        """Run to the budget, snapshotting at 0%, 50% and 100%."""
        self.setup()
        self.run = RunState()
        self.emit("start", budget=self.budget.describe())
        self.snapshot("untrained")
        self.score("untrained")
        half_done = False
        self._next_eval = eval_every
        self._last_progress = 0.0

        while True:
            frac = self.budget.fraction(self.run.elapsed, self.run.steps)
            if frac >= 1.0 or self._stop:
                break
            tel = self.iteration()
            self.run.iters += 1
            if tel:
                self.emit("algo", **tel)
            # A heartbeat on its own clock. `algo` fires once per iteration, and an
            # iteration is seconds long for the population schools and can be tens of
            # seconds at a large batch — far too coarse to drive a progress bar.
            if self.run.elapsed - self._last_progress >= progress_every:
                self._last_progress = self.run.elapsed
                self._sps = self.run.mark()
                self.emit("progress")
            frac = self.budget.fraction(self.run.elapsed, self.run.steps)
            if not half_done and frac >= 0.5:
                half_done = True
                self.snapshot("half")
                self.score("half")
            if frac >= self._next_eval:
                self._next_eval = frac + eval_every
                self.score(f"{frac:.0%}")

        # The wall-clock the *training* took. `save()` runs after the final scoring and
        # the peak/finish run-off, and folding those into the rate would understate it.
        self._train_wall = self.run.elapsed
        if not half_done:                     # a budget too small to reach 50%
            self.snapshot("half")
        self.snapshot("trained")
        final = self.score("trained", reps=12)
        self._finalise_best()
        self.save()
        return final

    def save(self) -> None:
        if not self.out:
            return
        d = self.out / self.key
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "checkpoints.npz",
            **{f"{name}_{role}": v[role]
               for name, v in self.checkpoints.items() for role in ("cat", "mouse")},
        )
        (d / "history.json").write_text(json.dumps({
            "school": self.key, "label": self.label, "seed": self.seed,
            "budget": {"seconds": self.budget.seconds, "steps": self.budget.steps},
            "envSteps": self.run.steps, "wall": self.run.elapsed,
            "iters": self.run.iters, "stepsPerSec": round(self.run.steps / max(1e-9, self._train_wall)),
            "history": self.history,
        }, indent=2))
        if self.best_report:
            (d / "best.json").write_text(json.dumps(
                {"school": self.key, "label": self.label, **self.best_report}, indent=2))


def load_checkpoints(path: Path) -> dict[str, dict[str, np.ndarray]]:
    z = np.load(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for k in z.files:
        name, role = k.rsplit("_", 1)
        out.setdefault(name, {})[role] = z[k]
    return out
