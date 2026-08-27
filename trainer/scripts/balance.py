"""Is the game itself fair before anybody learns anything?

Runs the scripted controller against itself across a skill sweep and reports how
episodes end. env-spec.json's sanity targets: a cat under 20% or draws over 20%
means the *rules* are wrong, not the learner. Draws are the number to watch — they
are what a nest stand-off looks like.

    python trainer/scripts/balance.py [episodes_per_cell]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catmouse import vec as V  # noqa: E402
from catmouse.scripted import ScriptedPair  # noqa: E402
from catmouse import env as S  # noqa: E402

TRAIN_SEEDS = [20260826 + i * 911 for i in range(12)]


def play(maps: V.MapSet, cat_skill: float, mouse_skill: float, n: int, seed: int = 0):
    e = V.VecEnv(maps, n, seed=seed)
    e.reset(map_idx=np.arange(n) % len(maps))
    cat_bot = ScriptedPair(e, cat_skill, seed=seed + 1)
    mouse_bot = ScriptedPair(e, mouse_skill, seed=seed + 2)
    steps = 0
    for _ in range(S.MAX_STEPS + 1):
        if e.done.all():
            break
        e.step(cat_bot.cat_act(), mouse_bot.mouse_act())
        steps += 1
    r = e.result
    return {
        "catch": float((r == 1).mean()), "escape": float((r == 2).mean()),
        "draw": float((r == 3).mean()), "steps": float(e.step_n.mean()),
        "traps": float(e.trap_hits.mean()), "wall": steps,
    }


def main(n: int = 480) -> None:
    print(f"compiling {len(TRAIN_SEEDS)} arenas …")
    maps = V.MapSet(TRAIN_SEEDS)

    print("\nscripted vs scripted, matched skill")
    print(f"{'skill':>6} {'catch':>7} {'escape':>7} {'draw':>7} {'steps':>7} {'traps/ep':>9}")
    t0 = time.time()
    for sk in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
        r = play(maps, sk, sk, n, seed=int(sk * 1000))
        flag = "  <-- stand-off" if r["draw"] > 0.20 and sk >= 0.5 else ""
        print(f"{sk:6.2f} {r['catch']:7.1%} {r['escape']:7.1%} {r['draw']:7.1%} "
              f"{r['steps']:7.1f} {r['traps']:9.2f}{flag}")

    print("\nthe Examiner (0.60) against every skill, per role")
    print(f"{'opponent':>9} {'cat vs Examiner-mouse':>23} {'mouse vs Examiner-cat':>23}")
    for sk in (0.05, 0.35, 0.65, 0.95):
        a = play(maps, sk, 0.60, n, seed=11)
        b = play(maps, 0.60, sk, n, seed=12)
        print(f"{sk:9.2f} {a['catch']:23.1%} {b['escape']:23.1%}")
    print(f"\n{len(TRAIN_SEEDS)} arenas, {n} episodes per cell, {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 480)
