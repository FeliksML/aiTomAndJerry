"""Decide the championship and write it where the app can read it.

    python trainer/scripts/tournament_run.py --run runs/v1

Writes into the run directory:
    tournament.json   the cross-play grid, the anchor column, the champions, and which
                      margins are too close to call
    progression.json  the anchor score at all three checkpoints, per school
    budgets.json      environment steps and wall-clock per school, so the equal-samples
                      and equal-compute readings can both be taken from one run

Nothing here trains. It only measures, on arenas that appear in no training set and in
no evaluation used during training.

By default each school enters the policy it FINISHED on. `--checkpoint best` enters the
strongest policy it reached instead — the one `best.json` records, chosen by a run-off
between the run's high-water mark and its finish on a seed neither was picked on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trainer"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v1")
    ap.add_argument("--reps", type=int, default=40, help="episodes per arena, per pairing")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--nests", default=None,
                    help="hole count(s); defaults to what the run's config.json records")
    ap.add_argument("--checkpoint", default="trained", choices=("trained", "best"),
                    help="which policy each school enters: the one it finished on "
                         "(default), or the strongest it reached — see best.json")
    a = ap.parse_args()

    from catmouse import arena, nets, tournament

    run_dir = Path(a.run) if Path(a.run).is_absolute() else ROOT / a.run
    dev = nets.pick_device(a.device)
    print(f"scoring {run_dir}  (device {dev}, {a.reps} reps per arena)", flush=True)

    t0 = time.time()
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    # The run's own config is the authority; --nests is for older runs whose config
    # predates the option and would otherwise be re-scored at today's default.
    nests = arena.spread(arena.parse_nests(a.nests if a.nests is not None
                                           else cfg.get("nests")), len(arena.FINAL_SEEDS))
    print(f"  holes per room: {sorted(set(nests))}", flush=True)
    print(f"  entering each school's {a.checkpoint.upper()} policy", flush=True)
    t = tournament.run_tournament(run_dir, dev, reps=a.reps, nests=nests,
                                  checkpoint=a.checkpoint)
    t["checkpoint"] = a.checkpoint
    (run_dir / "tournament.json").write_text(json.dumps(t, indent=2))
    print(f"  cross-play + anchor: {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    # Same reps as the tournament, so the TRAINED row is literally the vs EXAMINER
    # column the leaderboard shows rather than a second, smaller estimate of it.
    prog = tournament.progression(run_dir, dev, reps=a.reps, nests=nests)
    (run_dir / "progression.json").write_text(json.dumps(prog, indent=2))
    print(f"  checkpoint progression: {time.time() - t1:.0f}s", flush=True)

    (run_dir / "budgets.json").write_text(json.dumps(tournament.budgets(run_dir), indent=2))

    lab = t["labels"]
    W1, W2 = 13, 16
    print("\ncross-play — SCORE is the mean against the OTHER TWO schools; self-play does not vote")
    print("".ljust(W1) + "".join((lab[m] + " mouse").rjust(W2) for m in t["schools"])
          + "SCORE".rjust(W2) + "vs EXAMINER".rjust(14))
    for c in t["schools"]:
        cells = "".join(f"{t['cross'][c][m]['catch']:.1%}".rjust(W2) for m in t["schools"])
        sc = t["catScore"][c]
        print((lab[c] + " cat").rjust(W1) + cells
              + f"{sc['rate']:.1%} ±{(sc['hi'] - sc['lo']) / 2:.1%}".rjust(W2)
              + f"{t['anchor'][c]['cat']['catch']:.1%}".rjust(14))
    for m in t["schools"]:
        cells = "".join(f"{t['cross'][c][m]['escape']:.1%}".rjust(W2) for c in t["schools"])
        sc = t["mouseScore"][m]
        print((lab[m] + " mouse").rjust(W1) + cells
              + f"{sc['rate']:.1%} ±{(sc['hi'] - sc['lo']) / 2:.1%}".rjust(W2)
              + f"{t['anchor'][m]['mouse']['escape']:.1%}".rjust(14))

    print(f"\nBEST TOM   {lab[t['champion']['cat']]}"
          + (f"   (too close to call vs {', '.join(lab[s] for s in t['contested']['cat'])})"
             if t["contested"]["cat"] else "   — clear of the field"))
    print(f"BEST JERRY {lab[t['champion']['mouse']]}"
          + (f"   (too close to call vs {', '.join(lab[s] for s in t['contested']['mouse'])})"
             if t["contested"]["mouse"] else "   — clear of the field"))

    b = tournament.budgets(run_dir)
    if b:
        print("\nbudgets — the same wall-clock buys very different numbers of updates")
        for s, v in b.items():
            print(f"  {lab.get(s, s):>7}  {v['envSteps'] / 1e6:8.1f}M env steps"
                  f"  {v['wall'] / 60:6.1f} min  {v['iters']:6d} updates")

    print(f"\nwritten to {run_dir}/tournament.json")


if __name__ == "__main__":
    main()
