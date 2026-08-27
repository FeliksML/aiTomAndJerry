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
    """Wall-clock is the primary axis; steps is available as a hard cap for a
    deliberately sample-matched run."""
    seconds: float | None = None
    steps: int | None = None

    def fraction(self, elapsed: float, steps: int) -> float:
        fr = 0.0
        if self.seconds:
            fr = max(fr, elapsed / self.seconds)
        if self.steps:
            fr = max(fr, steps / self.steps)
        return min(1.0, fr)


@dataclass
class RunState:
    steps: int = 0
    iters: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started


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
        ev = {
            "kind": kind, "school": self.key, "iter": self.run.iters,
            "steps": self.run.steps, "wall": round(self.run.elapsed, 3),
            "frac": round(self.budget.fraction(self.run.elapsed, self.run.steps), 4),
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
        return row

    def train(self, eval_every: float = 0.05) -> dict:
        """Run to the budget, snapshotting at 0%, 50% and 100%."""
        self.setup()
        self.run = RunState()
        self.snapshot("untrained")
        self.score("untrained")
        half_done = False
        self._next_eval = eval_every

        while True:
            frac = self.budget.fraction(self.run.elapsed, self.run.steps)
            if frac >= 1.0:
                break
            tel = self.iteration()
            self.run.iters += 1
            if tel:
                self.emit("algo", **tel)
            frac = self.budget.fraction(self.run.elapsed, self.run.steps)
            if not half_done and frac >= 0.5:
                half_done = True
                self.snapshot("half")
                self.score("half")
            if frac >= self._next_eval:
                self._next_eval = frac + eval_every
                self.score(f"{frac:.0%}")

        if not half_done:                     # a budget too small to reach 50%
            self.snapshot("half")
        self.snapshot("trained")
        final = self.score("trained", reps=12)
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
            "iters": self.run.iters, "history": self.history,
        }, indent=2))


def load_checkpoints(path: Path) -> dict[str, dict[str, np.ndarray]]:
    z = np.load(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for k in z.files:
        name, role = k.rsplit("_", 1)
        out.setdefault(name, {})[role] = z[k]
    return out
