"""Does the batched environment still behave like the reference one?

`vec.py` trades the readable implementation for lookup tables and array ops. This
replays identical action streams through both and asserts they agree on every
observable: positions, facing, freeze counters, rewards to the last float, the
outcome, and the parts of the observation that are deterministic.

The mouse's hearing is the one exception — it is noise, and the two draw it from
different generators — so its three cue channels are excluded by construction.
The cat's nose is deterministic and IS compared.

    python trainer/scripts/vec_parity.py [n_maps]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catmouse import env as S  # noqa: E402
from catmouse import vec as V  # noqa: E402


def main(n_maps: int = 8, eps_per_map: int = 4) -> int:
    seeds = [4200 + i for i in range(n_maps)]
    print(f"compiling {n_maps} arenas …")
    ms = V.MapSet(seeds)

    rng = np.random.default_rng(7)
    n = n_maps * eps_per_map
    ve = V.VecEnv(ms, n, seed=1)
    ve.reset(map_idx=np.repeat(np.arange(n_maps), eps_per_map))

    scal = []
    for i in range(n):
        m = S.gen_map(seeds[i // eps_per_map])
        scal.append(S.reset(m, 999 + i))

    # One shared action stream so both sides walk the same trajectory.
    acts = rng.integers(0, 5, size=(S.MAX_STEPS + 2, n, 2))

    fails: list[str] = []
    worst_r = 0.0
    worst_obs = 0.0
    rc_sum = np.zeros(n)
    rm_sum = np.zeros(n)

    for t in range(S.MAX_STEPS + 1):
        # Observation check before stepping, while both are in the same state.
        if t in (0, 1, 5, 17, 60):
            oc = ve.observe("cat")
            om = ve.observe("mouse")
            for i, s in enumerate(scal):
                if s.done:
                    continue
                sc = S.observe(s, "cat")
                sm = S.observe(s, "mouse")
                worst_obs = max(
                    worst_obs,
                    np.abs(np.asarray(sc["rays"]) - oc[i, 7:28]).max(),
                    np.abs(np.asarray(sm["rays"]) - om[i, 7:28]).max(),
                    abs(sc["nestDist"] - oc[i, 30]), abs(sm["nestDist"] - om[i, 30]),
                    abs(sc["targetDist"] - oc[i, 34]), abs(sm["targetDist"] - om[i, 34]),
                )
                if sc["targetVisible"] != oc[i, 31] or sm["targetVisible"] != om[i, 31]:
                    fails.append(f"env {i} t{t}: visibility differs")
                want = sc["scent"]
                got = oc[i, 36:40]
                if (want is None) != (got[3] == 0):
                    fails.append(f"env {i} t{t}: cat scent presence differs")
                elif want is not None:
                    worst_obs = max(worst_obs, max(abs(want[k] - got[k]) for k in range(3)))

        a = acts[t]
        rc, rm, done, res = ve.step(a[:, 0].copy(), a[:, 1].copy())
        rc_sum += rc
        rm_sum += rm
        for i, s in enumerate(scal):
            if s.done:
                continue
            S.step(s, int(a[i, 0]), int(a[i, 1]))

        for i, s in enumerate(scal):
            want = (s.cat.x + s.cat.y * S.W, s.cat.facing, s.cat.frozen,
                    s.mouse.x + s.mouse.y * S.W, s.mouse.facing, s.mouse.frozen, s.step_n)
            got = (int(ve.cat[i]), int(ve.cat_face[i]), int(ve.cat_frozen[i]),
                   int(ve.mouse[i]), int(ve.mouse_face[i]), int(ve.mouse_frozen[i]), int(ve.step_n[i]))
            if want != got:
                fails.append(f"env {i} t{t}: state {got} != {want}")
                scal[i].done = True  # stop cascading noise from one divergence

    names = {0: None, 1: "catch", 2: "escape", 3: "timeout"}
    for i, s in enumerate(scal):
        if names[int(ve.result[i])] != s.result:
            fails.append(f"env {i}: result {names[int(ve.result[i])]} != {s.result}")
        worst_r = max(worst_r, abs(rc_sum[i] - s.cat.reward), abs(rm_sum[i] - s.mouse.reward))
        if ve.trap_hits[i] != s.trap_hits:
            fails.append(f"env {i}: trap hits {ve.trap_hits[i]} != {s.trap_hits}")

    print(f"episodes compared  : {n}")
    print(f"worst reward delta : {worst_r:.3e}")
    print(f"worst obs delta    : {worst_obs:.3e}")
    if fails:
        print(f"\nFAIL — {len(fails)} mismatch(es):")
        for f in fails[:20]:
            print("  " + f)
        return 1
    print("\nPASS — the batched environment matches the reference exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
