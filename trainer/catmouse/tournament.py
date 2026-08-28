"""Who is actually the best Tom, and the best Jerry.

Three separate numbers, answering three different questions, and only the first one
decides the championship:

  **cross-play** — every cat plays every mouse, including the two it has never met, on
      arenas none of them trained on. A cat's score is its mean catch rate across the
      two mice it did *not* grow up with. This is the headline, and the exclusion is
      what makes it one: leaving self-play in would let a school lift its own score by
      having raised a weak sparring partner — exactly the effect the whole tournament
      exists to remove. The diagonal is still reported, as `home`, and still drawn on
      the leaderboard; it simply does not vote.

  **anchor** — every policy plays the scripted Examiner at skill 0.60, a difficulty
      absent from every school's training ladder. Same opponent for all six, so it puts
      the three schools on one absolute axis. It is the same opponent *family* the
      schools trained against, which is why it supports the story rather than decides it.

  **home** — each school's own cat against its own mouse. Reported because it is what
      the viewer watched during training, and because the gap between this and
      cross-play is the interesting part: a school can look dominant at home purely by
      having raised a weak opponent.

Every rate carries a Wilson 95% interval. With eight arenas and a few dozen repeats a
three-point gap is noise, and the scoreboard says so instead of crowning someone on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import arena
from .league import EXAMINER_SKILL
from .nets import FLAT_DIM
from .school import load_checkpoints
from .vec import MapSet

SCHOOLS = ("ppo", "ga", "cmaes")
LABELS = {"ppo": "PPO", "ga": "GA", "cmaes": "CMA-ES"}


@dataclass
class Entry:
    school: str
    role: str
    flat: np.ndarray


def load_run(run_dir: Path, checkpoint: str = "trained") -> dict[str, dict[str, np.ndarray]]:
    """{school: {role: flat}} for one checkpoint across every school present."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for s in SCHOOLS:
        p = run_dir / s / "checkpoints.npz"
        if not p.exists():
            continue
        cps = load_checkpoints(p)
        if checkpoint not in cps:
            continue
        got = cps[checkpoint]["cat"].shape[-1]
        if got != FLAT_DIM:
            print(f"  skipping {s}: {got}-weight policy, this build expects {FLAT_DIM} "
                  f"(trained against a different observation)")
            continue
        out[s] = cps[checkpoint]
    return out


def run_tournament(run_dir: Path, device: torch.device, reps: int = 40,
                   checkpoint: str = "trained", seed: int = 4242,
                   nests=arena.DEFAULT_NESTS) -> dict:
    pols = load_run(run_dir, checkpoint)
    if not pols:
        raise SystemExit(f"no checkpoints under {run_dir}")
    present = [s for s in SCHOOLS if s in pols]
    maps = MapSet(arena.FINAL_SEEDS, nests)   # never trained on, never evaluated on before

    cross: dict[str, dict[str, dict]] = {}
    for ck in present:
        cross[ck] = {}
        for mk in present:
            o = arena.head_to_head(maps, pols[ck]["cat"], pols[mk]["mouse"], device,
                                   seed=seed + hash((ck, mk)) % 1000, reps=reps)
            cross[ck][mk] = o.as_dict()

    anchor: dict[str, dict] = {}
    for s in present:
        c = arena.examiner_score(maps, pols[s]["cat"], "cat", device,
                                 seed=seed + 11, reps=reps, skill=EXAMINER_SKILL)
        m = arena.examiner_score(maps, pols[s]["mouse"], "mouse", device,
                                 seed=seed + 13, reps=reps, skill=EXAMINER_SKILL)
        anchor[s] = {"cat": c.as_dict(), "mouse": m.as_dict()}

    # Cross-play marginals, OFF-DIAGONAL ONLY: a cat is judged on the mice it did not
    # grow up with. Averaging its own mouse back in is the one thing that would let a
    # weak sparring partner flatter the score, and on the two-hole run it decides the
    # answer — with the diagonal in, GA's mouse leads on 64% against GA's own cat, the
    # worst in the field; with it out, PPO's mouse leads and the intervals separate.
    cat_score, mouse_score = {}, {}
    for s in present:
        others = [x for x in present if x != s] or present   # a lone school scores at home
        cw = sum(cross[s][mk]["catch"] * cross[s][mk]["n"] for mk in others)
        cn = sum(cross[s][mk]["n"] for mk in others)
        mw = sum(cross[ck][s]["escape"] * cross[ck][s]["n"] for ck in others)
        mn = sum(cross[ck][s]["n"] for ck in others)
        p, lo, hi = arena.wilson(int(round(cw)), cn)
        q, mlo, mhi = arena.wilson(int(round(mw)), mn)
        cat_score[s] = {"rate": p, "lo": lo, "hi": hi, "n": cn}
        mouse_score[s] = {"rate": q, "lo": mlo, "hi": mhi, "n": mn}

    best_cat = max(present, key=lambda s: cat_score[s]["rate"])
    best_mouse = max(present, key=lambda s: mouse_score[s]["rate"])

    # An honest verdict needs to admit when the top two overlap.
    def contested(score: dict, winner: str) -> list[str]:
        return [s for s in present
                if s != winner and score[s]["hi"] >= score[winner]["lo"]]

    return {
        "checkpoint": checkpoint,
        "schools": present,
        "labels": LABELS,
        "arenas": list(arena.FINAL_SEEDS),
        "episodesPerPair": len(maps) * reps,
        "cross": cross,
        "anchor": anchor,
        "home": {s: cross[s][s] for s in present},
        "catScore": cat_score,
        "mouseScore": mouse_score,
        "champion": {"cat": best_cat, "mouse": best_mouse},
        "contested": {"cat": contested(cat_score, best_cat),
                      "mouse": contested(mouse_score, best_mouse)},
        "examinerSkill": EXAMINER_SKILL,
        "nests": nests,
    }


def progression(run_dir: Path, device: torch.device, reps: int = 16,
                nests=arena.DEFAULT_NESTS) -> dict:
    """The same anchor score at all three checkpoints — the rising bars on screen."""
    maps = MapSet(arena.FINAL_SEEDS, nests)
    out: dict[str, dict] = {}
    for name in ("untrained", "half", "trained"):
        pols = load_run(run_dir, name)
        out[name] = {}
        for s, p in pols.items():
            c = arena.examiner_score(maps, p["cat"], "cat", device, seed=77, reps=reps,
                                     skill=EXAMINER_SKILL)
            m = arena.examiner_score(maps, p["mouse"], "mouse", device, seed=79, reps=reps,
                                     skill=EXAMINER_SKILL)
            out[name][s] = {"cat": c.catch_rate, "mouse": m.escape_rate,
                            "catTraps": c.trap_hits, "mouseTraps": m.trap_hits}
    return out


def budgets(run_dir: Path) -> dict:
    """Both clocks, per school — the equal-samples vs equal-compute story."""
    out = {}
    for s in SCHOOLS:
        p = run_dir / s / "history.json"
        if p.exists():
            h = json.loads(p.read_text())
            out[s] = {"envSteps": h["envSteps"], "wall": h["wall"], "iters": h["iters"],
                      "history": h["history"]}
    return out
