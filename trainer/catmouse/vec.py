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
    nest_field[cell]                  -> BFS distance to the NEAREST hole
    next_home [cell]                  -> one greedy step toward the nearest hole

after which a step is pure array indexing and the whole batch moves together.
A 27x19 grid is 513 cells, so the biggest table is 513*4*513 bits per arena.

`scripts/vec_parity.py` replays scalar episodes through this class and asserts
they agree step for step, so the speed does not quietly cost correctness.
"""

from __future__ import annotations

import functools
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
#   28    BFS distance to the NEAREST hole, normalised
#   29    step fraction
#   30    target visible
#   31-32 sin, cos of the bearing to the target (zero when unseen)
#   33    target distance (1.0 when unseen)
#   34-36 the role's private cue: cat = scent (x, y, strength), mouse = hearing (x, y, confidence)
#   37    cue valid
#   38-49 one slot per hole, nearest first: sin, cos of its bearing, its distance, valid
#
# The holes get a slot each rather than a single "distance home" because with more than
# one of them the choice is the whole game: she picks which to run for, he has to decide
# which to cover. A policy that could only sense the nearest could do neither.
# Slots are always MAX_NESTS wide and zero-filled, so one trained network can play a
# room with one hole or three without being reshaped.
MAX_NESTS = S.MAX_NESTS
OBS_DIM = 38 + 4 * MAX_NESTS
IX_RAYS = 7
IX_CUE = 34
IX_NEST = 38

CACHE = pathlib.Path(__file__).resolve().parents[2] / "runs" / "maptables"


@dataclass
class Tables:
    """One arena, compiled."""
    seed: int
    grid: np.ndarray        # (NCELL,) uint8 cell codes
    move_to: np.ndarray     # (NCELL, 5) int16
    dist: np.ndarray        # (NCELL, NCELL) int16, -1 unreachable
    sees_cat: np.ndarray    # (NCELL, 4, NCELL) bool — the cat's cone
    sees_mouse: np.ndarray  # (NCELL, 4, NCELL) bool — the mouse's, which may differ
    los: np.ndarray         # (NCELL, NCELL) bool, facing-free line of sight
    next_home: np.ndarray   # (NCELL,) int16, greedy step toward the nest
    rays_cat: np.ndarray    # (NCELL, 4, 21) float32
    rays_mouse: np.ndarray  # (NCELL, 4, 21) float32
    nest_field: np.ndarray  # (NCELL,) int16 — distance to the NEAREST hole
    nest_cells: np.ndarray  # (MAX_NESTS,) int32, -1 padded
    n_nests: int
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

    # One set per role. `los` below stays single: line of sight is angle-free, so it is
    # the same question whoever asks it.
    rays = {r: np.zeros((NCELL, 4, 21), np.float32) for r in ("cat", "mouse")}
    sees = {r: np.zeros((NCELL, 4, NCELL), bool) for r in ("cat", "mouse")}
    los = np.zeros((NCELL, NCELL), bool)
    for c in range(NCELL):
        if g[c] == S.WALL:
            continue
        cx, cy = c % W, c // W
        for f in range(4):
            for role in ("cat", "mouse"):
                rays[role][c, f] = S.cast_rays(m.grid, cx, cy, f, role)
        for t in range(NCELL):
            if g[t] == S.WALL:
                continue
            tx, ty = t % W, t // W
            if t >= c:
                v = S.line_of_sight(m.grid, cx, cy, tx, ty)
                los[c, t] = v
                los[t, c] = v
            for f in range(4):
                for role in ("cat", "mouse"):
                    sees[role][c, f, t] = S.sees_target(m.grid, cx, cy, f, tx, ty, role)

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

    cells = np.full(MAX_NESTS, -1, np.int32)
    for k, n in enumerate(m.nests[:MAX_NESTS]):
        cells[k] = n[1] * W + n[0]
    # Bearing from every cell to every hole, precomputed: (MAX_NESTS, NCELL).
    nsin = np.zeros((MAX_NESTS, NCELL), np.float32)
    ncos = np.zeros((MAX_NESTS, NCELL), np.float32)
    for k, n in enumerate(m.nests[:MAX_NESTS]):
        b = np.arctan2(n[1] - ys, n[0] - xs)
        nsin[k] = np.sin(b)
        ncos[k] = np.cos(b)

    on_route = {(p[0], p[1]) for p in m.route}
    return Tables(
        seed=m.seed, grid=g, move_to=move_to, dist=dist, los=los,
        sees_cat=sees["cat"], sees_mouse=sees["mouse"],
        rays_cat=rays["cat"], rays_mouse=rays["mouse"],
        next_home=next_home,
        nest_field=np.asarray(m.nest_field, np.int16),
        nest_cells=cells, n_nests=len(m.nests),
        cat_spawn=m.cat_spawn[1] * W + m.cat_spawn[0],
        mouse_spawn=m.mouse_spawn[1] * W + m.mouse_spawn[0],
        nest_sin=nsin, nest_cos=ncos, optimal=m.optimal,
        n_traps_on_route=sum(1 for t in m.traps if (t[0], t[1]) in on_route),
    )


@functools.lru_cache(maxsize=1)
def _env_fingerprint() -> str:
    """A hash of env.py itself.

    Listing the constants by hand is not enough, and this project has the scar to prove
    it: the key was `v3:{seed}:{nests}`, the hole-placement rule was changed, and every
    one of the 54 cached arenas went on serving the OLD rooms. Two 500M-step runs trained
    against maps their author believed had been replaced, every gate passed, and the only
    symptom was a policy that scored differently once the cache finally missed.

    The rules live in code, not only in constants, so the fingerprint is the code. Editing
    env.py costs a recompile of the arenas in use (~2s each, once); it can no longer cost
    a silently wrong experiment.
    """
    src = (pathlib.Path(S.__file__)).read_bytes()
    return hashlib.sha1(src).hexdigest()[:12]


def cache_key(seed: int, nests: int) -> str:
    """Everything the compiled tables depend on, in the key."""
    return hashlib.sha1(
        f"v4:{seed}:{nests}:{_env_fingerprint()}".encode()).hexdigest()[:16]


def tables_for(seed: int, nests: int = 1, use_cache: bool = True) -> Tables:
    """Compile (or load) one arena. Compiling costs ~2s; arenas are reused constantly."""
    p = CACHE / f"{cache_key(seed, nests)}.npz"
    if use_cache and p.exists():
        z = np.load(p, allow_pickle=False)
        bits = lambda k, shape: np.unpackbits(z[k], count=int(np.prod(shape))).astype(bool).reshape(shape)
        return Tables(
            seed=int(z["seed"]), grid=z["grid"], move_to=z["move_to"], dist=z["dist"],
            # Both role fields renamed on purpose: had one kept the old name `sees`, a
            # stale file would load silently for that role and only that role.
            sees_cat=bits("sees_cat", (NCELL, 4, NCELL)),
            sees_mouse=bits("sees_mouse", (NCELL, 4, NCELL)),
            los=bits("los", (NCELL, NCELL)),
            next_home=z["next_home"], rays_cat=z["rays_cat"], rays_mouse=z["rays_mouse"],
            nest_field=z["nest_field"],
            nest_cells=z["nest_cells"], n_nests=int(z["n_nests"]),
            cat_spawn=int(z["cat_spawn"]), mouse_spawn=int(z["mouse_spawn"]),
            nest_sin=z["nest_sin"], nest_cos=z["nest_cos"], optimal=int(z["optimal"]),
            n_traps_on_route=int(z["n_traps_on_route"]),
        )
    t = _compile(S.gen_map(seed, nests))
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, seed=t.seed, grid=t.grid, move_to=t.move_to, dist=t.dist,
            sees_cat=np.packbits(t.sees_cat.reshape(-1)),
            sees_mouse=np.packbits(t.sees_mouse.reshape(-1)),
            los=np.packbits(t.los.reshape(-1)),
            next_home=t.next_home, rays_cat=t.rays_cat, rays_mouse=t.rays_mouse,
            nest_field=t.nest_field,
            nest_cells=t.nest_cells, n_nests=t.n_nests,
            cat_spawn=t.cat_spawn, mouse_spawn=t.mouse_spawn,
            nest_sin=t.nest_sin, nest_cos=t.nest_cos, optimal=t.optimal,
            n_traps_on_route=t.n_traps_on_route,
        )
    return t


class MapSet:
    """A stack of compiled arenas, addressable by index inside a batch."""

    def __init__(self, seeds, nests=1):
        """`nests` is a count, or one count per seed — a level set may deliberately mix
        one-, two- and three-hole rooms so a policy learns to handle any of them."""
        self.seeds = list(seeds)
        if isinstance(nests, int):
            nests = [nests] * len(self.seeds)
        self.nests = [max(1, min(MAX_NESTS, int(n))) for n in nests]
        ts = [tables_for(sd, nn) for sd, nn in zip(self.seeds, self.nests)]
        self.tables = ts
        self.grid = np.stack([t.grid for t in ts])
        self.move_to = np.stack([t.move_to for t in ts])
        self.dist = np.stack([t.dist for t in ts])
        self.sees_cat = np.stack([t.sees_cat for t in ts])
        self.sees_mouse = np.stack([t.sees_mouse for t in ts])
        self.los = np.stack([t.los for t in ts])
        self.next_home = np.stack([t.next_home for t in ts])
        self.rays_cat = np.stack([t.rays_cat for t in ts])
        self.rays_mouse = np.stack([t.rays_mouse for t in ts])
        self.nest_field = np.stack([t.nest_field for t in ts])
        self.nest_sin = np.stack([t.nest_sin for t in ts])
        self.nest_cos = np.stack([t.nest_cos for t in ts])
        self.nest_cells = np.stack([t.nest_cells for t in ts])     # (M, MAX_NESTS)
        self.n_nests = np.array([t.n_nests for t in ts], np.int32)
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

    #: Reward shaping. The defaults reproduce env.py exactly, which is what the parity
    #: test checks. `training_shaping()` swaps in the stronger terms used to LEARN.
    #:
    #: Why they need to be stronger: with the spec's coefficients the mouse's best
    #: available strategy from a cold start is to hide until the step limit. A timeout
    #: scores 0; being caught scores -1; and running the whole way home only earns
    #: +0.25 of shaping, which does not cover the risk. Measured over 400 CMA-ES
    #: generations, the mouse duly learned to hide and never escaped once.
    #:
    #: The fix is potential-based shaping (Ng, Harada & Russell): the extra reward is a
    #: difference of a potential over states, so it cannot create a strategy that was
    #: not already optimal — it only makes the existing one findable. Scoring is
    #: untouched: catch, escape and timeout rates are counted exactly as the spec says.
    SHAPING_DEFAULT = {"mouseApproach": 0.010, "catApproachVisible": 0.012, "catApproachAlways": 0.0}
    SHAPING_TRAINING = {"mouseApproach": 0.050, "catApproachVisible": 0.012, "catApproachAlways": 0.020}

    def __init__(self, maps: MapSet, n: int, seed: int = 0):
        self.M = maps
        self.n = n
        self.shaping = dict(self.SHAPING_DEFAULT)
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
        # x, y, confidence, radius. The radius is the width of the fix she took, so it
        # is stored with the fix rather than recomputed: once he goes quiet she still
        # believes the last thing she heard, and how vague it was cannot keep changing.
        self.heard = np.zeros((n, 4), np.float32)
        self.heard_on = np.zeros(n, bool)
        self.done = np.zeros(n, bool)
        self.result = np.zeros(n, np.int8)          # 0 running, 1 catch, 2 escape, 3 timeout
        self.trap_hits = np.zeros(n, np.int32)
        self.cat_trapped = np.zeros(n, np.int32)
        self.mouse_trapped = np.zeros(n, np.int32)
        self.saw_mouse = np.zeros(n, bool)
        self.saw_cat = np.zeros(n, bool)
        self._ar = np.arange(n)
        # Common random numbers. Population methods lay a batch out as
        # env = genome * episodes + slot; setting noise_tile = episodes makes every
        # genome meet the SAME hearing noise in the same slot, so a fitness gap is a
        # difference in policy rather than a difference in luck.
        self.noise_tile: int | None = None
        # Environment steps actually simulated. A finished episode costs nothing, so
        # only live environments count — this is the budget the three schools share.
        self.env_steps = 0

    # ---------- lifecycle ----------

    def training_shaping(self, overrides: dict | None = None) -> "VecEnv":
        """Use the learning-strength shaping. Never used when scoring.

        `overrides` lets a school be trained with its own pull toward the mouse or the
        hole. That is a decision about *how to teach*, not about what counts as winning:
        catch, escape and timeout are still counted exactly as the spec says, on an
        environment that never sees these numbers. Two schools taught with different
        shaping remain comparable on the scoreboard for that reason.
        """
        self.shaping = dict(self.SHAPING_TRAINING)
        for k, v in (overrides or {}).items():
            if k in self.shaping and v is not None:
                self.shaping[k] = float(v)
        return self

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
        M, mi = self.M, self.map_idx

        o = np.zeros((self.n, OBS_DIM), np.float32)
        mx, my = me % W, me // W
        ox, oy = other % W, other // W
        o[:, 0] = mx / W
        o[:, 1] = my / H
        o[self._ar, 2 + face] = 1.0
        o[:, 6] = frozen / S.FREEZE_STEPS
        o[:, IX_RAYS:IX_RAYS + 21] = (M.rays_cat if role == "cat" else M.rays_mouse)[mi, me, face]
        o[:, 28] = M.nest_field[mi, me] / (W + H)
        o[:, 29] = self.step_n / S.MAX_STEPS

        vis = (M.sees_cat if role == "cat" else M.sees_mouse)[mi, me, face, other]
        bear = np.arctan2(oy - my, ox - mx)
        o[:, 30] = vis
        o[:, 31] = np.where(vis, np.sin(bear), 0)
        o[:, 32] = np.where(vis, np.cos(bear), 0)
        o[:, 33] = np.where(vis, np.hypot(ox - mx, oy - my) / S.VISION_RANGE, 1.0)
        o[:, IX_CUE:IX_CUE + 4] = self._cue_cat() if role == "cat" else self._cue_mouse()

        # One slot per hole, nearest first. Sorting by walking distance rather than by
        # index keeps the layout meaningful: slot 0 is always "the one I could reach
        # soonest", whichever hole that happens to be on this map from where I stand.
        cells = M.nest_cells[mi]                                   # (n, MAX_NESTS)
        live = cells >= 0
        safe = np.where(live, cells, 0)
        d = M.dist[mi[:, None], safe, me[:, None]].astype(np.float32)   # (n, MAX_NESTS)
        d = np.where(live & (d >= 0), d, 1e6)
        order = np.argsort(d, axis=1, kind="stable")
        d_sorted = np.take_along_axis(d, order, 1)
        ok = np.take_along_axis(live, order, 1) & (d_sorted < 1e5)
        for k in range(MAX_NESTS):
            base = IX_NEST + 4 * k
            # `order[:, k]` is which hole ranked k-th here; the bearing tables are
            # indexed by that hole's slot, not by the cell it sits on.
            hole = order[:, k]
            o[:, base + 0] = np.where(ok[:, k], M.nest_sin[mi, hole, me], 0)
            o[:, base + 1] = np.where(ok[:, k], M.nest_cos[mi, hole, me], 0)
            o[:, base + 2] = np.where(ok[:, k], d_sorted[:, k] / (W + H), 0)
            o[:, base + 3] = ok[:, k]
        return o

    def _draw(self) -> np.ndarray:
        """One uniform per environment — or one per episode slot, tiled, when common
        random numbers are on (see `noise_tile`)."""
        k = self.noise_tile
        if not k:
            return self.rng.random(self.n)
        return np.tile(self.rng.random(k), self.n // k)

    # ---------- rendering ----------

    def map_payload(self, i: int = 0) -> dict:
        """The arena, in the shape the visualiser draws. Sent only when it changes."""
        t = self.M.tables[int(self.map_idx[i])]
        g = t.grid
        return {
            "seed": int(t.seed), "w": W, "h": H,
            "grid": [int(v) for v in g],
            "nests": [[int(c % W), int(c // W)] for c in t.nest_cells if c >= 0],
            "nest": [int(t.nest_cells[0] % W), int(t.nest_cells[0] // W)],
            "traps": [[int(c % W), int(c // W)] for c in np.flatnonzero(g == S.TRAP)],
            "optimal": int(t.optimal), "trapsOnRoute": int(t.n_traps_on_route),
        }

    def render(self, i: int = 0, probs: dict | None = None) -> dict:
        """One environment as a frame. Cones are cast here rather than stored: the
        policy only ever needed the 21 ray lengths, but the screen needs the polygon."""
        t = self.M.tables[int(self.map_idx[i])]
        g = t.grid
        cx, cy = int(self.cat[i] % W), int(self.cat[i] // W)
        mx, my = int(self.mouse[i] % W), int(self.mouse[i] // W)
        cc, _ = S.cast_cone(g, cx, cy, int(self.cat_face[i]), "cat")
        cm, _ = S.cast_cone(g, mx, my, int(self.mouse_face[i]), "mouse")
        sc = self.scent[i]
        hot = np.flatnonzero(sc > 0.08)
        out = {
            "step": int(self.step_n[i]),
            "result": [None, "catch", "escape", "timeout"][int(self.result[i])],
            "cat": {
                "x": cx, "y": cy, "facing": int(self.cat_face[i]),
                "frozen": int(self.cat_frozen[i]),
                "cone": [[round(float(a), 3), round(float(b), 3)] for a, b in cc],
                "sees": bool(self.saw_mouse[i]), "mode": self.cat_mode(i),
            },
            "mouse": {
                "x": mx, "y": my, "facing": int(self.mouse_face[i]),
                "frozen": int(self.mouse_frozen[i]),
                "cone": [[round(float(a), 3), round(float(b), 3)] for a, b in cm],
                "sees": bool(self.saw_cat[i]), "mode": self.mouse_mode(i),
                "heard": ({"x": round(float(self.heard[i, 0]), 2),
                           "y": round(float(self.heard[i, 1]), 2),
                           "conf": round(float(self.heard[i, 2]), 3),
                           # The radius the fix was taken with, not one recomputed from
                           # where he actually is now. Recomputing it made the ring track
                           # his live distance while he was frozen or out of earshot —
                           # drawing information she does not have.
                           "radius": round(float(self.heard[i, 3]), 3)}
                          if self.heard_on[i] else None),
            },
            "scent": [[int(c % W), int(c // W), round(float(sc[c]), 3)] for c in hot],
            "nestDist": int(t.nest_field[int(self.mouse[i])]),
        }
        if probs:
            for role in ("cat", "mouse"):
                if role in probs:
                    out[role]["probs"] = [round(float(v), 4) for v in probs[role]]
        return out

    def cat_mode(self, i: int) -> str:
        """What the cat can currently sense, in the UI's vocabulary. Derived from the
        environment, not from the policy — a network has no mode, so claiming to read
        its intent would be a lie. The action probabilities carry the real intent."""
        if self.cat_frozen[i] > 0:
            return "trapped"
        if self.saw_mouse[i]:
            return "chase"
        d = self.M.dist[self.map_idx[i], self.cat[i]]
        ok = (d >= 0) & (d <= S.SCENT_RANGE) & (self.scent[i] >= 0.08)
        if ok.any():
            return "scent"
        if self.M.nest_field[self.map_idx[i], self.cat[i]] <= 3:
            return "camp"
        return "patrol"

    def mouse_mode(self, i: int) -> str:
        if self.mouse_frozen[i] > 0:
            return "trapped"
        if self.saw_cat[i]:
            return "evade"
        if self.heard_on[i]:
            return "listen"
        return "dash"

    # ---------- dynamics ----------

    def step(self, cat_a: np.ndarray, mouse_a: np.ndarray):
        """One simultaneous step. Returns (cat_reward, mouse_reward, done, result).

        Episodes that already finished are inert: they hold their final state and
        result until the caller resets them, so a batch where some episodes end
        early stays correct without the caller having to mask anything.
        """
        M, mi = self.M, self.map_idx
        live = ~self.done
        self.env_steps += int(live.sum())
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
        jx = (self._draw() - 0.5) * noise * 6
        jy = (self._draw() - 0.5) * noise * 6
        fresh = np.stack([
            np.clip(cx + jx, 0, W - 1),
            np.clip(cy + jy, 0, H - 1),
            np.maximum(0.08, 1 - d_cat / S.HEARING_RANGE),
            1 + noise * 7,                       # env.py stores this on the Heard record
        ], 1).astype(np.float32)
        faded = self.heard.copy()
        faded[:, 2] *= np.where(live, 0.82, 1.0)
        self.heard = np.where(audible[:, None], fresh, faded)
        self.heard_on = np.where(audible, True, self.heard_on & (faded[:, 2] >= 0.1))

        # Not only an observation: `saw_mouse` also gates the cat's approach shaping a
        # few lines below, so a wider cat cone pays her shaping over a wider arc too.
        self.saw_mouse = M.sees_cat[mi, self.cat, self.cat_face, self.mouse]
        self.saw_cat = M.sees_mouse[mi, self.mouse, self.mouse_face, self.cat]

        new_gap = np.hypot(cx - mx, cy - my)
        new_home = M.nest_field[mi, self.mouse].astype(np.float32)
        sh = self.shaping
        rc += np.where(self.saw_mouse, sh["catApproachVisible"] * (prev_gap - new_gap), 0)
        rm += sh["mouseApproach"] * (prev_home - new_home)
        if sh["catApproachAlways"]:
            # Walk-distance, not straight-line: closing across a wall is not progress.
            # Only the reward sees this; the policy still gets the same observation.
            prev_walk = M.dist[mi, pcat, pmouse].astype(np.float32)
            new_walk = M.dist[mi, self.cat, self.mouse].astype(np.float32)
            ok = (prev_walk >= 0) & (new_walk >= 0)
            rc += np.where(ok, sh["catApproachAlways"] * (prev_walk - new_walk), 0)
        rc += S.R_CAT_STEP
        rm += S.R_MOUSE_STEP

        self.step_n += live
        man = np.abs(cx - mx) + np.abs(cy - my)
        swapped = (self.cat != pcat) & (self.cat == pmouse) & (self.mouse == pcat)
        caught = (man <= 1) | swapped
        # Any hole will do — that is the point of having more than one.
        escaped = (M.nest_cells[mi] == self.mouse[:, None]).any(1)
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
