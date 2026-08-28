"""Are the held-out arenas actually held out?

"Evaluated on rooms nobody trained on" is the claim the whole leaderboard rests on, and
it is one seed collision away from being false: `gen_map` reseeds on a failed attempt by
adding 104729, so two distant seeds *could* in principle land on the same room. This
fingerprints every arena in all three sets and asserts they are disjoint.

Hole count matters: the generator branches on it, so proving the one-hole rooms are
disjoint says nothing about the two-hole rooms the leaderboard was actually measured on.
Every count in `--nests` is checked, and the default covers both shipped configurations.

    python trainer/scripts/check_arenas.py
    python trainer/scripts/check_arenas.py --nests 2
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catmouse import arena as A  # noqa: E402
from catmouse import env as S  # noqa: E402


def fingerprint(seed: int, nests: int) -> str:
    m = S.gen_map(seed, nests)
    key = bytes(m.grid) + repr((m.nests, m.cat_spawn, m.mouse_spawn, m.traps)).encode()
    return hashlib.sha1(key).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nests", default="1,2", help="hole counts to check (default both)")
    a = ap.parse_args()
    counts = [int(x) for x in str(a.nests).split(",") if x.strip()]

    sets = {"train": A.TRAIN_SEEDS, "eval": A.EVAL_SEEDS, "final": A.FINAL_SEEDS}
    failed = False
    for nests in counts:
        seen: dict[str, tuple[str, int]] = {}
        clashes = []
        for name, seeds in sets.items():
            for s in seeds:
                h = fingerprint(s, nests)
                if h in seen and seen[h][0] != name:
                    clashes.append((seen[h], (name, s)))
                seen.setdefault(h, (name, s))

        print(f"{nests} hole(s)   " + "  ".join(f"{k}: {len(v)}" for k, v in sets.items())
              + f"   distinct rooms: {len(seen)}")
        for x, y in clashes:
            print(f"  CLASH  {x[0]} seed {x[1]}  ==  {y[0]} seed {y[1]}")
        failed = failed or bool(clashes)

    if failed:
        print("\nFAIL — an arena appears in more than one set.")
        return 1
    print("\nPASS — training, evaluation and tournament arenas are disjoint, "
          f"at {' and '.join(str(c) for c in counts)} hole(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
