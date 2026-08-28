"""Find the episodes worth putting in the video.

    python trainer/scripts/highlights.py --run runs/v2 --episodes 400

Plays the champion cat against the champion mouse across many arenas and seeds, scores
every episode for drama, and writes `highlights.json` into the run. The app can then
replay exactly those arenas and seeds — the same episode, frame for frame, because the
environment is deterministic given (arena, seed).

What counts as drama, and why each one reads on camera:

  nail-biter    she reaches the hole with him inside two cells. The cone is on her the
                whole way in.
  heartbreak    he catches her within a few cells of home, after she has walked the
                whole room.
  the snap      a trap fires and the episode turns on it — someone frozen while the
                other closes.
  the long one  a chase that keeps nearly ending: several separate moments at distance
                one without a catch.
  the shutout   he camps and she never gets past. Boring to lose, interesting to watch
                once.

Each entry carries the numbers behind its score, so nothing has to be taken on trust
when you are deciding what to cut.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trainer"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v2")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--arenas", default="train", choices=("train", "final"))
    ap.add_argument("--nests", default=None)
    a = ap.parse_args()

    from catmouse import arena as A
    from catmouse import env as S
    from catmouse import nets, vec
    from catmouse.tournament import LABELS, load_run

    run_dir = Path(a.run) if Path(a.run).is_absolute() else ROOT / a.run
    dev = nets.pick_device(a.device)
    pols = load_run(run_dir, "trained")
    if not pols:
        raise SystemExit(f"no trained checkpoints under {run_dir}")

    t_path = run_dir / "tournament.json"
    if t_path.exists():
        t = json.loads(t_path.read_text())
        ck, mk = t["champion"]["cat"], t["champion"]["mouse"]
    else:
        ck = mk = sorted(pols)[0]
    print(f"champions: {LABELS.get(ck, ck)} cat vs {LABELS.get(mk, mk)} mouse", flush=True)

    seeds = A.TRAIN_SEEDS if a.arenas == "train" else A.FINAL_SEEDS
    maps = vec.MapSet(seeds, A.spread(A.parse_nests(a.nests), len(seeds)))
    n = a.episodes
    e = vec.VecEnv(maps, n, seed=20260827)
    map_idx = np.arange(n) % len(maps)
    e.reset(map_idx=map_idx)

    cat = nets.FlatActor(pols[ck]["cat"], dev)
    mouse = nets.FlatActor(pols[mk]["mouse"], dev)
    rng_c = np.random.default_rng(1)
    rng_m = np.random.default_rng(2)

    W = vec.W
    start_home = maps.nest_field[map_idx, e.mouse].astype(np.float32)
    min_gap = np.full(n, 99.0)
    min_home = start_home.copy()
    brushes = np.zeros(n, np.int32)      # separate moments at Manhattan distance 1
    touching = np.zeros(n, bool)
    cat_frozen_steps = np.zeros(n, np.int32)
    mouse_frozen_steps = np.zeros(n, np.int32)
    gap_at_snap = np.full(n, 99.0)
    snapped = np.zeros(n, bool)

    for _ in range(S.MAX_STEPS + 1):
        if e.done.all():
            break
        live = ~e.done
        oc, om = e.observe("cat"), e.observe("mouse")
        prev_cf, prev_mf = e.cat_frozen.copy(), e.mouse_frozen.copy()
        e.step(cat.act(oc, rng_c), mouse.act(om, rng_m))
        gap = np.abs(e.cat % W - e.mouse % W) + np.abs(e.cat // W - e.mouse // W)
        home = maps.nest_field[map_idx, e.mouse].astype(np.float32)
        min_gap = np.where(live, np.minimum(min_gap, gap), min_gap)
        min_home = np.where(live, np.minimum(min_home, home), min_home)
        # A "brush" is a fresh approach to distance 1, not one long stretch of it.
        close = live & (gap <= 1) & ~e.done
        brushes += (close & ~touching)
        touching = close
        cat_frozen_steps += live & (e.cat_frozen > 0)
        mouse_frozen_steps += live & (e.mouse_frozen > 0)
        fresh_snap = live & (((prev_cf == 0) & (e.cat_frozen > 0)) | ((prev_mf == 0) & (e.mouse_frozen > 0)))
        gap_at_snap = np.where(fresh_snap & ~snapped, gap, gap_at_snap)
        snapped |= fresh_snap

    res = e.result
    steps = e.step_n
    rows = []
    for i in range(n):
        kind, score, why = None, 0.0, ""
        if res[i] == 2 and min_gap[i] <= 2:
            kind = "nail-biter"
            score = 100 - 10 * min_gap[i] + 2 * brushes[i]
            why = f"got home with him {int(min_gap[i])} cell(s) away"
        elif res[i] == 1 and min_home[i] <= max(3, 0.15 * start_home[i]):
            kind = "heartbreak"
            score = 95 - 6 * min_home[i] + 2 * brushes[i]
            why = f"caught {int(min_home[i])} cell(s) from the hole"
        elif snapped[i] and gap_at_snap[i] <= 6 and res[i] != 3:
            kind = "the snap"
            score = 80 - 4 * gap_at_snap[i] + 1.5 * brushes[i]
            why = f"trap fired with {int(gap_at_snap[i])} cells between them"
        elif brushes[i] >= 3:
            kind = "the long one"
            score = 55 + 6 * brushes[i]
            why = f"{int(brushes[i])} separate near-misses over {int(steps[i])} steps"
        elif res[i] == 3 and min_home[i] <= 4:
            kind = "the shutout"
            score = 45 + (5 - min_home[i]) * 4
            why = f"she got within {int(min_home[i])} of home and ran out of time"
        if kind:
            rows.append({
                "kind": kind, "score": round(float(score), 1), "why": why,
                "arena": int(map_idx[i]), "seed": int(seeds[map_idx[i]]),
                "result": ["", "catch", "escape", "timeout"][int(res[i])],
                "steps": int(steps[i]), "minGap": int(min_gap[i]),
                "minHome": int(min_home[i]), "brushes": int(brushes[i]),
                "trapSnap": bool(snapped[i]),
            })

    rows.sort(key=lambda r: -r["score"])
    # One per arena, so a highlight reel is not six takes of the same room.
    seen, picked = set(), []
    for r in rows:
        if r["arena"] in seen:
            continue
        seen.add(r["arena"])
        picked.append(r)
        if len(picked) >= a.top:
            break

    out = {"catSchool": ck, "mouseSchool": mk, "arenaSet": a.arenas,
           "episodes": n, "found": len(rows), "highlights": picked}
    (run_dir / "highlights.json").write_text(json.dumps(out, indent=2))

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    print(f"\n{len(rows)} dramatic episodes out of {n}   " +
          "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"\ntop {len(picked)}, one per arena:")
    for r in picked:
        print(f"  {r['score']:5.1f}  {r['kind']:<12} arena {r['arena']:2d}  "
              f"{r['result']:<7} {r['steps']:3d} steps  — {r['why']}")
    print(f"\nwritten to {run_dir}/highlights.json")


if __name__ == "__main__":
    main()
