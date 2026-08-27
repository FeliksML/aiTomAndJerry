"""Are the held-out arenas actually held out?

"Evaluated on rooms nobody trained on" is the claim the whole leaderboard rests on, and
it is one seed collision away from being false: `gen_map` reseeds on a failed attempt by
adding 104729, so two distant seeds *could* in principle land on the same room. This
fingerprints every arena in all three sets and asserts they are disjoint.

    python trainer/scripts/check_arenas.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catmouse import arena as A  # noqa: E402
from catmouse import env as S  # noqa: E402


def fingerprint(seed: int) -> str:
    m = S.gen_map(seed)
    key = bytes(m.grid) + repr((m.nest, m.cat_spawn, m.mouse_spawn, m.traps)).encode()
    return hashlib.sha1(key).hexdigest()[:12]


def main() -> int:
    sets = {"train": A.TRAIN_SEEDS, "eval": A.EVAL_SEEDS, "final": A.FINAL_SEEDS}
    seen: dict[str, tuple[str, int]] = {}
    clashes = []
    for name, seeds in sets.items():
        for s in seeds:
            h = fingerprint(s)
            if h in seen and seen[h][0] != name:
                clashes.append((seen[h], (name, s)))
            seen.setdefault(h, (name, s))

    print("  ".join(f"{k}: {len(v)}" for k, v in sets.items())
          + f"   distinct rooms: {len(seen)}")
    if clashes:
        print("\nFAIL — these arenas appear in more than one set:")
        for a, b in clashes:
            print(f"  {a[0]} seed {a[1]}  ==  {b[0]} seed {b[1]}")
        return 1
    print("\nPASS — training, evaluation and tournament arenas are disjoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
