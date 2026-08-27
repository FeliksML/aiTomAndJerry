"""Batched environment: thousands of episodes stepping at once.

`env.py` is the definition of truth but it is far too slow to train against — a
single step does a full BFS for the cat's nose and casts 21 rays for each cone.
None of that depends on anything but the arena, though, and the arenas are fixed
and few. So every arena is compiled once into lookup tables:

    move_to   [cell, action]          -> destination cell, wall check already folded in
    dist      [cell, cell]            -> all-pairs BFS distance (the cat's nose)
    sees      [cell, facing, cell]    -> range + cone angle + occlusion, one bit
    los       [cell, cell]            -> facing-free line of sight
    rays      [cell, facing, 21]      -> the vision cone the policy actually reads
    nest_field[cell]                  -> BFS distance home
    next_home [cell]                  -> one greedy step toward the nest

after which a step is pure array indexing and the whole batch moves together.
A 27x19 grid is 513 cells, so the biggest table is 513*4*513 bits per arena.

`scripts/vec_parity.py` replays scalar episodes through this class and asserts
they agree step for step, so the speed does not quietly cost correctness.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass

import numpy as np

from . import env as S

W, H = S.W, S.H
NCELL = W * H
NACT = 5

# Observation layout, shared by every algorithm and both roles — fairness starts here.
#   0     x / W
#   1     y / H
#   2-5   facing, one-hot
#   6     frozen fraction
#   7-27  21 vision rays
#   28-29 sin, cos of the bearing to the nest
#   30    BFS distance to the nest, normalised
#   31    target visible
#   32-33 sin, cos of the bearing to the target (zero when unseen)
#   34    target distance (1.0 when unseen)
#   35    step fraction
#   36-38 the role's private cue: cat = scent (x, y, strength), mouse = hearing (x, y, confidence)
#   39    cue valid
OBS_DIM = 40
IX_RAYS = 7
IX_CUE = 36

CACHE = pathlib.Path(__file__).resolve().parents[2] / "runs" / "maptables"


@dataclass
class Tables:
    """One arena, compiled."""
    seed: int
    grid: np.ndarray        # (NCELL,) uint8 cell codes
    move_to: np.ndarray     # (NCELL, 5) int16
    dist: np.ndarray        # (NCELL, NCELL) int16, -1 unreachable
    sees: np.ndarray        # (NCELL, 4, NCELL) bool
    los: np.ndarray         # (NCELL, NCELL) bool, facing-free line of sight
    next_home: np.ndarray   # (NCELL,) int16, greedy step toward the nest
    rays: np.ndarray        # (NCELL, 4, 21) float32
    nest_field: np.ndarray  # (NCELL,) int16
    nest_cell: int
    cat_spawn: int
    mouse_spawn: int
    nest_sin: np.ndarray    # (NCELL,) float32
    nest_cos: np.ndarray    # (NCELL,) float32
    optimal: int
    n_traps_on_route: int


def _compile(m: S.Map) -> Tables:
    g = np.frombuffer(bytes(m.grid), dtype=np.uint8).copy()
    xs = np.arange(NCELL) % W
    ys = np.arange(NCELL) // W

    move_to = np.zeros((NCELL, NACT), np.int16)
    move_to[:, 0] = np.arange(NCELL)
    for a, (dx, dy) in enumerate(S.DIRS):
        nx, ny = xs + dx, ys + dy
        ok = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
        dest = np.where(ok, ny * W + nx, np.arange(NCELL))
        blocked = g[dest] == S.WALL
        move_to[:, a + 1] = np.where(ok & ~blocked, dest, np.arange(NCELL))

    dist = np.full((NCELL, NCELL), -1, np.int16)
    for c in range(NCELL):
        if g[c] == S.WALL:
            continue
        dist[c] = np.asarray(S.bfs(m.grid, c % W, c // W), np.int16)

    rays = np.zeros((NCELL, 4, 21), np.float32)
    sees = np.zeros((NCELL, 4, NCELL), bool)
    los = np.zeros((NCELL, NCELL), bool)
    for c in range(NCELL):
        if g[c] == S.WALL:
            continue
        cx, cy = c % W, c // W
        for f in range(4):
            _, rr = S.cast_cone(m.grid, cx, cy, f)
            rays[c, f] = rr
        for t in range(NCELL):
            if g[t] == S.WALL:
                continue
            tx, ty = t % W, t // W
            if t >= c:
                v = S.line_of_sight(m.grid, cx, cy, tx, ty)
                los[c, t] = v
                los[t, c] = v
            for f in range(4):
                sees[c, f, t] = S.sees_target(m.grid, cx, cy, f, tx, ty)

    # One greedy step down the nest field, tie-broken N,E,S,W exactly as env.js does.
    # Chaining it k times is the `pathAhead` the scripted cat uses to cut her off.
    nf = np.asarray(m.nest_field, np.int32)
    next_home = np.arange(NCELL, dtype=np.int16)
    for c in range(NCELL):
        if g[c] == S.WALL or nf[c] <= 0:
            continue
        bd, best = nf[c], c
        for dx, dy in S.DIRS:
            qx, qy = c % W + dx, c // W + dy
            if not S.passable(m.grid, qx, qy):
                continue
            qd = nf[qy * W + qx]
            if 0 <= qd < bd:
                bd, best = qd, qy * W + qx
        next_home[c] = best

    nest = m.nest[1] * W + m.nest[0]
    nsin = np.zeros(NCELL, np.float32)
    ncos = np.zeros(NCELL, np.float32)
    b = np.arctan2(m.nest[1] - ys, m.nest[0] - xs)
    nsin[:] = np.sin(b)
    ncos[:] = np.cos(b)

    on_route = {(p[0], p[1]) for p in m.route}
    return Tables(
        seed=m.seed, grid=g, move_to=move_to, dist=dist, sees=sees, los=los,
        next_home=next_home, rays=rays,
        nest_field=np.asarray(m.nest_field, np.int16), nest_cell=nest,
        cat_spawn=m.cat_spawn[1] * W + m.cat_spawn[0],
        mouse_spawn=m.mouse_spawn[1] * W + m.mouse_spawn[0],
        nest_sin=nsin, nest_cos=ncos, optimal=m.optimal,
        n_traps_on_route=sum(1 for t in m.traps if (t[0], t[1]) in on_route),
    )


def tables_for(seed: int, use_cache: bool = True) -> Tables:
    """Compile (or load) one arena. Compiling costs ~1s; arenas are reused constantly."""
    key = hashlib.sha1(f"v2:{seed}".encode()).hexdigest()[:16]
    p = CACHE / f"{key}.npz"
    if use_cache and p.exists():
        z = np.load(p, allow_pickle=False)
        return Tables(
            seed=int(z["seed"]), grid=z["grid"], move_to=z["move_to"], dist=z["dist"],
            sees=np.unpackbits(z["sees"], count=NCELL * 4 * NCELL).astype(bool).reshape(NCELL, 4, NCELL),
            los=np.unpackbits(z["los"], count=NCELL * NCELL).astype(bool).reshape(NCELL, NCELL),
            next_home=z["next_home"], rays=z["rays"], nest_field=z["nest_field"], nest_cell=int(z["nest_cell"]),
            cat_spawn=int(z["cat_spawn"]), mouse_spawn=int(z["mouse_spawn"]),
            nest_sin=z["nest_sin"], nest_cos=z["nest_cos"], optimal=int(z["optimal"]),
            n_traps_on_route=int(z["n_traps_on_route"]),
        )
    t = _compile(S.gen_map(seed))
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, seed=t.seed, grid=t.grid, move_to=t.move_to, dist=t.dist,
            sees=np.packbits(t.sees.reshape(-1)), los=np.packbits(t.los.reshape(-1)),
            next_home=t.next_home, rays=t.rays, nest_field=t.nest_field,
            nest_cell=t.nest_cell, cat_spawn=t.cat_spawn, mouse_spawn=t.mouse_spawn,
            nest_sin=t.nest_sin, nest_cos=t.nest_cos, optimal=t.optimal,
            n_traps_on_route=t.n_traps_on_route,
        )
    return t


class MapSet:
    """A stack of compiled arenas, addressable by index inside a batch."""

    def __init__(self, seeds):
        self.seeds = list(seeds)
        ts = [tables_for(s) for s in self.seeds]
        self.tables = ts
        self.grid = np.stack([t.grid for t in ts])
        self.move_to = np.stack([t.move_to for t in ts])
        self.dist = np.stack([t.dist for t in ts])
        self.sees = np.stack([t.sees for t in ts])
        self.los = np.stack([t.los for t in ts])
        self.next_home = np.stack([t.next_home for t in ts])
        self.rays = np.stack([t.rays for t in ts])
        self.nest_field = np.stack([t.nest_field for t in ts])
        self.nest_sin = np.stack([t.nest_sin for t in ts])
        self.nest_cos = np.stack([t.nest_cos for t in ts])
        self.nest_cell = np.array([t.nest_cell for t in ts], np.int32)
        self.cat_spawn = np.array([t.cat_spawn for t in ts], np.int32)
        self.mouse_spawn = np.array([t.mouse_spawn for t in ts], np.int32)
        self.optimal = np.array([t.optimal for t in ts], np.int32)

    def __len__(self) -> int:
        return len(self.seeds)


class VecEnv:
    """B independent episodes, stepped together.

    Semantics follow `env.py` exactly, including the order that matters: the mouse
    reaching the hole is checked BEFORE a simultaneous catch, and a frozen agent
    burns one freeze step instead of moving.
    """

    def __init__(self, maps: MapSet, n: int, seed: int = 0):
        self.M = maps
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.map_idx = np.zeros(n, np.int32)
        self.cat = np.zeros(n, np.int32)
        self.mouse = np.zeros(n, np.int32)
        self.cat_face = np.zeros(n, np.int32)
        self.mouse_face = np.zeros(n, np.int32)
        self.cat_frozen = np.zeros(n, np.int32)
        self.mouse_frozen = np.zeros(n, np.int32)
        self.step_n = np.zeros(n, np.int32)
        self.scent = np.zeros((n, NCELL), np.float32)
        self.heard = np.zeros((n, 3), np.float32)   # x, y, confidence
        self.heard_on = np.zeros(n, bool)
        self.done = np.zeros(n, bool)
        self.result = np.zeros(n, np.int8)          # 0 running, 1 catch, 2 escape, 3 timeout
        self.trap_hits = np.zeros(n, np.int32)
        self.cat_trapped = np.zeros(n, np.int32)
        self.mouse_trapped = np.zeros(n, np.int32)
        self.saw_mouse = np.zeros(n, bool)
        self.saw_cat = np.zeros(n, bool)
        self._ar = np.arange(n)

    # ---------- lifecycle ----------

    def reset(self, map_idx=None) -> None:
        if map_idx is None:
            map_idx = self.rng.integers(0, len(self.M), self.n)
        self.map_idx[:] = map_idx
        self.reset_where(np.ones(self.n, bool))

    def reset_where(self, mask: np.ndarray, map_idx=None) -> None:
        """Restart just the flagged episodes, optionally on a new arena each."""
        if map_idx is not None:
            self.map_idx[mask] = map_idx
        mi = self.map_idx[mask]
        self.cat[mask] = self.M.cat_spawn[mi]
        self.mouse[mask] = self.M.mouse_spawn[mi]
        self.cat_face[mask] = 1
        self.mouse_face[mask] = 3
        self.cat_frozen[mask] = 0
        self.mouse_frozen[mask] = 0
        self.step_n[mask] = 0
        self.scent[mask] = 0.0
        self.heard[mask] = 0.0
        self.heard_on[mask] = False
        self.done[mask] = False
        self.result[mask] = 0
        self.trap_hits[mask] = 0
        self.cat_trapped[mask] = 0
        self.mouse_trapped[mask] = 0
        self.saw_mouse[mask] = False
        self.saw_cat[mask] = False

    # ---------- observation ----------

    def scent_cue(self):
        """The cat's nose: the strongest scent cell within BFS range 6, or nothing.

        Returns (cell, strength, valid). The scripted cat walks to the cell, so it is
        exposed rather than folded straight into the observation vector.
        """
        d = self.M.dist[self.map_idx, self.cat]              # (n, NCELL)
        ok = (d >= 0) & (d <= S.SCENT_RANGE)
        v = np.where(ok, self.scent, 0.0)
        v = np.where(v >= 0.08, v, 0.0)
        best = v.argmax(1)
        strength = v[self._ar, best]
        return best.astype(np.int32), strength, strength > 0

    def _cue_cat(self):
        best, strength, valid = self.scent_cue()
        cue = np.zeros((self.n, 4), np.float32)
        cue[:, 0] = np.where(valid, (best % W) / W, 0)
        cue[:, 1] = np.where(valid, (best // W) / H, 0)
        cue[:, 2] = np.where(valid, strength, 0)
        cue[:, 3] = valid
        return cue

    def _cue_mouse(self):
        """The mouse's ears: a noisy fix on the cat, decaying while he is silent."""
        cue = np.zeros((self.n, 4), np.float32)
        cue[:, 0] = np.where(self.heard_on, self.heard[:, 0] / W, 0)
        cue[:, 1] = np.where(self.heard_on, self.heard[:, 1] / H, 0)
        cue[:, 2] = np.where(self.heard_on, self.heard[:, 2], 0)
        cue[:, 3] = self.heard_on
        return cue

    def observe(self, role: str) -> np.ndarray:
        me = self.cat if role == "cat" else self.mouse
        other = self.mouse if role == "cat" else self.cat
        face = self.cat_face if role == "cat" else self.mouse_face
        frozen = self.cat_frozen if role == "cat" else self.mouse_frozen

        o = np.zeros((self.n, OBS_DIM), np.float32)
        mx, my = me % W, me // W
        ox, oy = other % W, other // W
        o[:, 0] = mx / W
        o[:, 1] = my / H
        o[self._ar, 2 + face] = 1.0
        o[:, 6] = frozen / S.FREEZE_STEPS
        o[:, IX_RAYS:IX_RAYS + 21] = self.M.rays[self.map_idx, me, face]
        o[:, 28] = self.M.nest_sin[self.map_idx, me]
        o[:, 29] = self.M.nest_cos[self.map_idx, me]
        o[:, 30] = self.M.nest_field[self.map_idx, me] / (W + H)

        vis = self.M.sees[self.map_idx, me, face, other]
        bear = np.arctan2(oy - my, ox - mx)
        o[:, 31] = vis
        o[:, 32] = np.where(vis, np.sin(bear), 0)
        o[:, 33] = np.where(vis, np.cos(bear), 0)
        o[:, 34] = np.where(vis, np.hypot(ox - mx, oy - my) / S.VISION_RANGE, 1.0)
        o[:, 35] = self.step_n / S.MAX_STEPS
        o[:, IX_CUE:IX_CUE + 4] = self._cue_cat() if role == "cat" else self._cue_mouse()
        return o

    # ---------- dynamics ----------

    def step(self, cat_a: np.ndarray, mouse_a: np.ndarray):
        """One simultaneous step. Returns (cat_reward, mouse_reward, done, result).

        Episodes that already finished are inert: they hold their final state and
        result until the caller resets them, so a batch where some episodes end
        early stays correct without the caller having to mask anything.
        """
        M, mi = self.M, self.map_idx
        live = ~self.done
        pcat, pmouse = self.cat.copy(), self.mouse.copy()
        prev_gap = np.hypot(pcat % W - pmouse % W, pcat // W - pmouse // W)
        prev_home = M.nest_field[mi, pmouse].astype(np.float32)

        rc = np.zeros(self.n, np.float32)
        rm = np.zeros(self.n, np.float32)

        # A frozen agent burns a freeze step instead of acting.
        cf = live & (self.cat_frozen > 0)
        mf = live & (self.mouse_frozen > 0)
        self.cat_frozen[cf] -= 1
        self.mouse_frozen[mf] -= 1
        rc[cf] += S.R_TRAP * 0.2
        rm[mf] += S.R_TRAP * 0.2

        act_c = live & ~cf
        act_m = live & ~mf
        cat_dest = M.move_to[mi, self.cat, cat_a]
        mouse_dest = M.move_to[mi, self.mouse, mouse_a]
        self.cat = np.where(act_c, cat_dest, self.cat).astype(np.int32)
        self.mouse = np.where(act_m, mouse_dest, self.mouse).astype(np.int32)
        moved_c = act_c & (cat_a > 0)
        moved_m = act_m & (mouse_a > 0)
        self.cat_face = np.where(moved_c, cat_a - 1, self.cat_face).astype(np.int32)
        self.mouse_face = np.where(moved_m, mouse_a - 1, self.mouse_face).astype(np.int32)

        snap_c = act_c & (M.grid[mi, self.cat] == S.TRAP)
        snap_m = act_m & (M.grid[mi, self.mouse] == S.TRAP)
        self.cat_frozen[snap_c] = S.FREEZE_STEPS
        self.mouse_frozen[snap_m] = S.FREEZE_STEPS
        self.cat_trapped += snap_c
        self.mouse_trapped += snap_m
        rc[snap_c] += S.R_TRAP
        rm[snap_m] += S.R_TRAP
        self.trap_hits += (snap_c | snap_m)

        self.scent *= np.where(live, S.SCENT_DECAY, 1.0)[:, None]
        self.scent[self._ar[live], self.mouse[live]] = 1.0

        # Hearing: his footsteps carry through walls, accuracy falls off with distance.
        cx, cy = self.cat % W, self.cat // W
        mx, my = self.mouse % W, self.mouse // W
        d_cat = np.hypot(cx - mx, cy - my)
        audible = live & (self.cat_frozen == 0) & (d_cat <= S.HEARING_RANGE)
        noise = S.HEARING_BASE_NOISE + S.HEARING_DIST_NOISE * d_cat
        jx = (self.rng.random(self.n) - 0.5) * noise * 6
        jy = (self.rng.random(self.n) - 0.5) * noise * 6
        fresh = np.stack([
            np.clip(cx + jx, 0, W - 1),
            np.clip(cy + jy, 0, H - 1),
            np.maximum(0.08, 1 - d_cat / S.HEARING_RANGE),
        ], 1).astype(np.float32)
        faded = self.heard.copy()
        faded[:, 2] *= np.where(live, 0.82, 1.0)
        self.heard = np.where(audible[:, None], fresh, faded)
        self.heard_on = np.where(audible, True, self.heard_on & (faded[:, 2] >= 0.1))

        self.saw_mouse = M.sees[mi, self.cat, self.cat_face, self.mouse]
        self.saw_cat = M.sees[mi, self.mouse, self.mouse_face, self.cat]

        new_gap = np.hypot(cx - mx, cy - my)
        new_home = M.nest_field[mi, self.mouse].astype(np.float32)
        rc += np.where(self.saw_mouse, S.R_CAT_APPROACH * (prev_gap - new_gap), 0)
        rm += S.R_MOUSE_APPROACH * (prev_home - new_home)
        rc += S.R_CAT_STEP
        rm += S.R_MOUSE_STEP

        self.step_n += live
        man = np.abs(cx - mx) + np.abs(cy - my)
        swapped = (self.cat != pcat) & (self.cat == pmouse) & (self.mouse == pcat)
        caught = (man <= 1) | swapped
        escaped = self.mouse == M.nest_cell[mi]
        timeout = self.step_n >= S.MAX_STEPS

        # Reaching the hole beats a simultaneous catch — she is inside.
        res = np.where(escaped, 2, np.where(caught, 1, np.where(timeout, 3, 0))).astype(np.int8)
        rc += np.where(res == 1, S.R_CAT_CATCH, 0) + np.where(res == 2, S.R_CAT_ESCAPED, 0)
        rm += np.where(res == 2, S.R_MOUSE_NEST, 0) + np.where(res == 1, S.R_MOUSE_CAUGHT, 0)

        rc = np.where(live, rc, 0.0)
        rm = np.where(live, rm, 0.0)
        self.result = np.where(live, res, self.result).astype(np.int8)
        self.done = self.result != 0
        return rc, rm, self.done.copy(), self.result.copy()
