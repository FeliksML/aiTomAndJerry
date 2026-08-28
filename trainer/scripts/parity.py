"""Parity gate: does env.py reproduce env.js exactly?

Map generation must be bit-identical — that is the contract in env-spec.json, and
the visualiser paints arenas from a seed while the trainer trains on them. Episode
dynamics are integer too, so trajectories must match step for step. Only the ray
casts touch trig, where V8 and CPython may disagree by an ULP; those are compared
with a tolerance and the worst deviation is reported.

    node trainer/scripts/dump_js.js 1 200 2 > runs/parity-js.json
    python trainer/scripts/parity.py runs/parity-js.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catmouse import env as E  # noqa: E402
from catmouse.jsmath import JsRng, floor_mul  # noqa: E402


def hash_grid(g) -> str:
    h = 0x811C9DC5
    for v in g:
        h ^= v
        h = (h * 0x01000193) & 0xFFFFFFFF
    return format(h, "x")


def main(path: str) -> int:
    blob = json.loads(Path(path).read_text())
    ref, n_nests = blob["maps"], blob["nests"]
    fails: list[str] = []
    worst_ray = 0.0

    for r in ref:
        seed = r["seed"]
        m = E.gen_map(seed, n_nests)
        tag = f"seed {seed}"

        if hash_grid(m.grid) != r["gridHash"]:
            fails.append(f"{tag}: grid hash {hash_grid(m.grid)} != {r['gridHash']}")
            continue
        for name, got, want in (
            ("nests", [list(n) for n in m.nests], r["nests"]),
            ("catSpawn", list(m.cat_spawn), r["catSpawn"]),
            ("mouseSpawn", list(m.mouse_spawn), r["mouseSpawn"]),
            ("traps", [list(t) for t in m.traps], r["traps"]),
            ("optimal", m.optimal, r["optimal"]),
            ("routeLen", len(m.route), r["routeLen"]),
            ("blocks", len(m.blocks), r["blocks"]),
            ("pillars", len(m.pillars), r["pillars"]),
        ):
            if got != want:
                fails.append(f"{tag}: {name} {got} != {want}")

        ar = JsRng(seed ^ 0xA5A5A5)
        s = E.reset(m, seed)
        traj = []
        g = 0
        while not s.done and g < 400:
            g += 1
            if s.step_n == 3 and r["obsSample"]:
                oc = E.observe(s, "cat")
                om = E.observe(s, "mouse")
                for a, b in ((oc["rays"], r["obsSample"]["catRays"]),
                             (om["rays"], r["obsSample"]["mouseRays"])):
                    worst_ray = max(worst_ray, max(abs(x - y) for x, y in zip(a, b)))
                if oc["targetVisible"] != r["obsSample"]["catVis"]:
                    fails.append(f"{tag}: cat targetVisible differs")
                if om["targetVisible"] != r["obsSample"]["mouseVis"]:
                    fails.append(f"{tag}: mouse targetVisible differs")
            ca = floor_mul(ar(), 5)
            ma = floor_mul(ar(), 5)
            E.step(s, ca, ma)
            traj.append([s.cat.x, s.cat.y, s.cat.facing, s.cat.frozen,
                         s.mouse.x, s.mouse.y, s.mouse.facing, s.mouse.frozen])

        if traj != r["traj"]:
            first = next((i for i, (a, b) in enumerate(zip(traj, r["traj"])) if a != b), None)
            fails.append(f"{tag}: trajectory diverges at step {first} "
                         f"(len {len(traj)} vs {len(r['traj'])})")
        if s.result != r["result"] or s.step_n != r["steps"]:
            fails.append(f"{tag}: outcome {s.result}@{s.step_n} != {r['result']}@{r['steps']}")
        for name, got, want in (("catReward", s.cat.reward, r["catReward"]),
                                ("mouseReward", s.mouse.reward, r["mouseReward"])):
            if abs(got - want) > 1e-9:
                fails.append(f"{tag}: {name} {got} != {want}")

    print(f"maps compared      : {len(ref)}  ({n_nests} hole(s) each)")
    print(f"worst ray deviation: {worst_ray:.3e}  (trig ULP noise; 0 means bit-identical)")
    if fails:
        print(f"\nFAIL — {len(fails)} mismatch(es):")
        for f in fails[:25]:
            print("  " + f)
        if len(fails) > 25:
            print(f"  … and {len(fails) - 25} more")
        return 1
    print("\nPASS — Python and JS agree on every map and every step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/parity-js.json"))
