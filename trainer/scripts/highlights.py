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


PLAY_ENV_SEED = 1234        # the seed serve.py's PLAY mode builds its VecEnv with
ACTION_SEED_XOR = 0x5EED    # and how it derives the action stream from an episode seed


def measure(e, nest_field, map_idx, act):
    """Play a batch to the end, returning the numbers the drama rules are made of."""
    from catmouse import env as S
    from catmouse import vec

    W = vec.W
    n = e.n
    start_home = nest_field[map_idx, e.mouse].astype(np.float32)
    min_gap = np.full(n, 99.0)
    min_home = start_home.copy()
    brushes = np.zeros(n, np.int32)      # separate moments at Manhattan distance 1
    touching = np.zeros(n, bool)
    gap_at_snap = np.full(n, 99.0)
    snapped = np.zeros(n, bool)

    for _ in range(S.MAX_STEPS + 1):
        if e.done.all():
            break
        live = ~e.done
        prev_cf, prev_mf = e.cat_frozen.copy(), e.mouse_frozen.copy()
        a_c, a_m = act(e)
        e.step(a_c, a_m)
        gap = np.abs(e.cat % W - e.mouse % W) + np.abs(e.cat // W - e.mouse // W)
        home = nest_field[map_idx, e.mouse].astype(np.float32)
        min_gap = np.where(live, np.minimum(min_gap, gap), min_gap)
        min_home = np.where(live, np.minimum(min_home, home), min_home)
        # A "brush" is a fresh approach to distance 1, not one long stretch of it.
        close = live & (gap <= 1) & ~e.done
        brushes += (close & ~touching)
        touching = close
        fresh_snap = live & (((prev_cf == 0) & (e.cat_frozen > 0))
                             | ((prev_mf == 0) & (e.mouse_frozen > 0)))
        gap_at_snap = np.where(fresh_snap & ~snapped, gap, gap_at_snap)
        snapped |= fresh_snap

    return {"res": e.result, "steps": e.step_n, "startHome": start_home,
            "minGap": min_gap, "minHome": min_home, "brushes": brushes,
            "snapped": snapped, "gapAtSnap": gap_at_snap}


def classify(m, i):
    """The drama rules, in one place — so the scan and the verification cannot drift."""
    res = int(m["res"][i])
    min_gap, min_home = m["minGap"][i], m["minHome"][i]
    brushes, steps, start_home = m["brushes"][i], m["steps"][i], m["startHome"][i]
    if res == 2 and min_gap <= 2:
        return ("nail-biter", 100 - 10 * min_gap + 2 * brushes,
                f"got home with him {int(min_gap)} cell(s) away")
    if res == 1 and min_home <= max(3, 0.15 * start_home):
        return ("heartbreak", 95 - 6 * min_home + 2 * brushes,
                f"caught {int(min_home)} cell(s) from the hole")
    if m["snapped"][i] and m["gapAtSnap"][i] <= 6 and res != 3:
        return ("the snap", 80 - 4 * m["gapAtSnap"][i] + 1.5 * brushes,
                f"trap fired with {int(m['gapAtSnap'][i])} cells between them")
    if brushes >= 3:
        return ("the long one", 55 + 6 * brushes,
                f"{int(brushes)} separate near-misses over {int(steps)} steps")
    if res == 3 and min_home <= 4:
        return ("the shutout", 45 + (5 - min_home) * 4,
                f"she got within {int(min_home)} of home and ran out of time")
    return (None, 0.0, "")


def replay_solo(maps, arena_idx, seed, cat, mouse):
    """The exact episode serve.py plays for {cmd:"play", levels:[arena], seeds:[seed]}.

    Same VecEnv construction, the same per-episode re-seeding of both streams, and the
    same order of draws — cat first, then mouse, out of one generator.
    """
    from catmouse import vec

    e = vec.VecEnv(maps, 1, seed=PLAY_ENV_SEED)
    idx = np.array([arena_idx])
    e.reset(map_idx=idx)
    e.rng = np.random.default_rng(seed)
    rng = np.random.default_rng(seed ^ ACTION_SEED_XOR)

    def act(env):
        oc, om = env.observe("cat"), env.observe("mouse")
        return cat.act(oc, rng), mouse.act(om, rng)

    return measure(e, maps.nest_field, idx, act), e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v2")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--arenas", default="train", choices=("train", "final"))
    ap.add_argument("--nests", default=None)
    ap.add_argument("--cat", default=None, help="cat's school (default: the cross-play champion)")
    ap.add_argument("--mouse", default=None, help="mouse's school (default: the cross-play champion)")
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
    ck, mk = a.cat or ck, a.mouse or mk
    for who, s_ in (("cat", ck), ("mouse", mk)):
        if s_ not in pols:
            raise SystemExit(f"no {s_} {who} under {run_dir}")
    print(f"pairing: {LABELS.get(ck, ck)} cat vs {LABELS.get(mk, mk)} mouse"
          + ("" if (a.cat or a.mouse) else "   (the cross-play champions)"), flush=True)

    seeds = A.TRAIN_SEEDS if a.arenas == "train" else A.FINAL_SEEDS
    maps = vec.MapSet(seeds, A.spread(A.parse_nests(a.nests), len(seeds)))

    cat = nets.FlatActor(pols[ck]["cat"], dev)
    mouse = nets.FlatActor(pols[mk]["mouse"], dev)

    # The scan IS the replay. An earlier version played hundreds of episodes in one
    # batch, which is faster but not reproducible: a lane's action samples and hearing
    # noise came out of a shared stream at a position no single-episode replay can reach,
    # so the app played a different episode from the one that had been scored — sometimes
    # a different *outcome*, under the caption of the scored one. Every candidate here is
    # instead played exactly the way serve.py will play it, keyed on (arena, seed), so
    # what gets written down is by construction what gets watched. 480 of them cost 3 s.
    per_arena = max(1, a.episodes // len(maps))
    n = per_arena * len(maps)
    print(f"scanning {n} episodes — {per_arena} seeds on each of {len(maps)} arenas, "
          f"each one played the way the app will play it", flush=True)

    rows = []
    for ai in range(len(maps)):
        base = int(seeds[ai])
        for k in range(per_arena):
            sd = (base + k * 7919) & 0x7FFFFFFF      # a stride, so arenas cannot collide
            m, _ = replay_solo(maps, ai, sd, cat, mouse)
            kind, score, why = classify(m, 0)
            if not kind:
                continue
            rows.append({
                "sig": (ai, int(m["res"][0]), int(m["steps"][0]), int(m["minGap"][0]),
                        int(m["minHome"][0]), int(m["brushes"][0])),
                "kind": kind, "score": round(float(score), 1), "why": why,
                "arena": ai, "seed": sd,
                "result": ["", "catch", "escape", "timeout"][int(m["res"][0])],
                "steps": int(m["steps"][0]), "minGap": int(m["minGap"][0]),
                "minHome": int(m["minHome"][0]), "brushes": int(m["brushes"][0]),
                "trapSnap": bool(m["snapped"][0]),
            })

    rows.sort(key=lambda r: -r["score"])
    # A saturated policy plays the same room the same way whatever the seed, so counting
    # seeds would report 149 dramatic episodes where there are four. Count the episodes
    # that are actually different, and say how many rooms produced any.
    distinct = {r["sig"] for r in rows}
    rooms = {r["arena"] for r in rows}
    # One per arena, so a highlight reel is not six takes of the same room.
    seen, picked = set(), []
    for r in rows:
        if r["arena"] in seen:
            continue
        seen.add(r["arena"])
        picked.append({k: v for k, v in r.items() if k != "sig"})
        if len(picked) >= a.top:
            break

    out = {"catSchool": ck, "mouseSchool": mk, "arenaSet": a.arenas,
           "seedsTried": n, "arenas": len(maps), "episodes": len(distinct),
           "found": len(distinct), "rooms": len(rooms), "highlights": picked}
    (run_dir / "highlights.json").write_text(json.dumps(out, indent=2))

    kind_of = {r["sig"]: r["kind"] for r in rows}
    counts: dict[str, int] = {}
    for sig in distinct:
        counts[kind_of[sig]] = counts.get(kind_of[sig], 0) + 1
    print(f"\n{len(distinct)} distinct dramatic episodes in {len(rooms)} of {len(maps)} rooms "
          f"({n} seeds tried)   " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"\ntop {len(picked)}, one per arena:")
    for r in picked:
        print(f"  {r['score']:5.1f}  {r['kind']:<12} arena {r['arena']:2d}  "
              f"{r['result']:<7} {r['steps']:3d} steps  — {r['why']}")
    print(f"\nwritten to {run_dir}/highlights.json")


if __name__ == "__main__":
    main()
