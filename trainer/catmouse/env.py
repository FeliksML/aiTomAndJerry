"""CatMousePursuit-v1 — the reference environment, ported 1:1 from viz/env.js.

The contract in env-spec.json says the same seed must yield the same map in both
languages, so `gen_map` mirrors env.js statement for statement, including the order
candidate lists are built in and the exact sequence of PRNG draws. Anything that
looks redundant here is load-bearing: reordering a filter changes the map.

Map generation is integer-only and therefore bit-exact against JS. The dynamics use
trig (`cos`, `sin`, `atan2`), where V8 and CPython can differ by an ULP; that can
only matter for a ray that lands exactly on a wall boundary, and `scripts/parity.py`
measures how often that happens (answer: it doesn't).

This is the readable, scalar implementation — the one that defines truth. Training
throughput comes from `vec.py`, which precomputes this environment into lookup
tables and is checked against this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .jsmath import JsRng, floor_mul

W = 27
H = 19
EMPTY, WALL, TRAP, NEST = 0, 1, 2, 3
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # N E S W
ACTIONS = ("stay", "north", "east", "south", "west")

# One cone per role. They were a single constant, and the measurement that motivated the
# split says why that mattered: widening the SHARED cone makes the cat WORSE (at equal
# scripted skill, 37% catches at 50 degrees falls to 22% at 180), because sight is worth
# more to the evader than to the pursuer — she routes around what she can see coming.
# Deliberately no `VISION_HALF_ANGLE_DEG` fallback: a call site that forgot to say which
# role it means must fail loudly rather than quietly keep the old 50.
VISION_HALF_ANGLE_CAT_DEG = 50.0
VISION_HALF_ANGLE_MOUSE_DEG = 50.0


def half_angle(role: str) -> float:
    """Half-angle in RADIANS for one role. Anything that casts a cone must say whose."""
    if role == "cat":
        return VISION_HALF_ANGLE_CAT_DEG * math.pi / 180.0
    if role == "mouse":
        return VISION_HALF_ANGLE_MOUSE_DEG * math.pi / 180.0
    raise ValueError(f"vision is per role; got {role!r}")
# How far either side of a wall corner the silhouette rays are aimed. At full range
# 1e-4 rad puts them 0.0009 cells apart — past the corner, far closer than a pixel.
CORNER_EPS = 1e-4
VISION_RANGE = 8.5
VISION_RAYS = 21
HEARING_RANGE = 12.0
HEARING_BASE_NOISE = 0.10
HEARING_DIST_NOISE = 0.055
SCENT_RANGE = 6
SCENT_DECAY = 0.93
FREEZE_STEPS = 5
MAX_STEPS = 180

# How many holes a room may have, and how far apart they must be. One hole makes
# camping it a dominant strategy; two give the mouse a choice the cat cannot cover.
MAX_NESTS = 3
MIN_NEST_SEP = 10

R_CAT_CATCH = 1.0
R_CAT_ESCAPED = -1.0
R_CAT_STEP = -0.002
R_CAT_APPROACH = 0.012
R_MOUSE_NEST = 1.0
R_MOUSE_CAUGHT = -1.0
R_MOUSE_STEP = -0.002
R_MOUSE_APPROACH = 0.010
R_TRAP = -0.05


def idx(x: int, y: int) -> int:
    return y * W + x


def in_b(x: int, y: int) -> bool:
    return 0 <= x < W and 0 <= y < H


def passable(g, x: int, y: int) -> bool:
    return in_b(x, y) and g[idx(x, y)] != WALL


def bfs(g, sx: int, sy: int) -> list[int]:
    """Distance field from (sx, sy). -1 where unreachable. Neighbour order N,E,S,W
    matches env.js so tie-breaking downstream (greedy path descent) agrees."""
    d = [-1] * (W * H)
    if not passable(g, sx, sy):
        return d
    q = [sx, sy]
    head = 0
    d[idx(sx, sy)] = 0
    while head < len(q):
        x = q[head]
        y = q[head + 1]
        head += 2
        nd = d[idx(x, y)] + 1
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if not passable(g, nx, ny):
                continue
            if d[idx(nx, ny)] != -1:
                continue
            d[idx(nx, ny)] = nd
            q.append(nx)
            q.append(ny)
    return d


def bfs_multi(g, sources) -> list[int]:
    """Distance to the NEAREST of several sources. With more than one hole the mouse's
    "distance home" is the distance to whichever hole is closest, and every shaping,
    observation and route calculation downstream then works unchanged."""
    d = [-1] * (W * H)
    q: list[int] = []
    for sx, sy in sources:
        if not passable(g, sx, sy) or d[idx(sx, sy)] == 0:
            continue
        d[idx(sx, sy)] = 0
        q.append(sx)
        q.append(sy)
    head = 0
    while head < len(q):
        x, y = q[head], q[head + 1]
        head += 2
        nd = d[idx(x, y)] + 1
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if not passable(g, nx, ny):
                continue
            if d[idx(nx, ny)] != -1:
                continue
            d[idx(nx, ny)] = nd
            q.append(nx)
            q.append(ny)
    return d


@dataclass
class Map:
    seed: int
    grid: bytearray
    blocks: list[dict]
    pillars: list[tuple[int, int]]
    traps: list[tuple[int, int]]
    nest: tuple[int, int]
    nests: list[tuple[int, int]]
    cat_spawn: tuple[int, int]
    mouse_spawn: tuple[int, int]
    route: list[tuple[int, int]]
    nest_field: list[int]
    optimal: int
    w: int = W
    h: int = H


def gen_map(seed: int, nest_count: int = 1) -> Map:
    """Seeded arena generation. Port of env.js genMap — see that file's comments for
    why each constraint exists (two approaches per hole, traps on the walked route,
    holes kept apart so one cat cannot cover two)."""
    seed &= 0xFFFFFFFF
    nest_count = max(1, min(MAX_NESTS, nest_count))
    for attempt in range(80):
        r = JsRng((seed + attempt * 7919) & 0xFFFFFFFF)
        g = bytearray(W * H)
        for x in range(W):
            g[idx(x, 0)] = WALL
            g[idx(x, H - 1)] = WALL
        for y in range(H):
            g[idx(0, y)] = WALL
            g[idx(W - 1, y)] = WALL

        blocks: list[dict] = []
        tries = 0
        while len(blocks) < 7 and tries < 400:
            tries += 1
            bw = 2 + floor_mul(r(), 4)
            bh = 2 + floor_mul(r(), 3)
            bx = 2 + floor_mul(r(), W - 4 - bw)
            by = 2 + floor_mul(r(), H - 4 - bh)
            ok = True
            for b in blocks:
                if (bx - 2 < b["x"] + b["w"] and bx + bw + 2 > b["x"]
                        and by - 2 < b["y"] + b["h"] and by + bh + 2 > b["y"]):
                    ok = False
                    break
            if not ok:
                continue
            blocks.append({"x": bx, "y": by, "w": bw, "h": bh})
            for y in range(by, by + bh):
                for x in range(bx, bx + bw):
                    g[idx(x, y)] = WALL

        pil = 0
        pt = 0
        pillars: list[tuple[int, int]] = []
        while pil < 10 and pt < 400:
            pt += 1
            px = 2 + floor_mul(r(), W - 4)
            py = 2 + floor_mul(r(), H - 4)
            if g[idx(px, py)] != EMPTY:
                continue
            near = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if g[idx(px + dx, py + dy)] == WALL:
                        near += 1
            if near > 1:
                continue
            g[idx(px, py)] = WALL
            pillars.append((px, py))
            pil += 1

        free: list[tuple[int, int]] = []
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if g[idx(x, y)] != WALL:
                    free.append((x, y))
        if len(free) < 200:
            continue
        probe = bfs(g, free[0][0], free[0][1])
        reach = sum(1 for c in free if probe[idx(c[0], c[1])] >= 0)
        if reach < len(free) * 0.94:
            continue

        # Each hole needs two INDEPENDENT approaches: walling any single neighbour
        # must leave it reachable. A one-corridor pocket can be sealed by a camping
        # cat and then every episode dies on the step limit.
        #
        # With more than one hole the mouse gets a CHOICE, which is the point — one
        # hole makes camping it a dominant strategy and the chase collapses into a
        # stakeout. Holes are therefore also kept MIN_NEST_SEP apart, so a cat standing
        # between two of them cannot cover both.
        nest_cands = []
        for c in free:
            if probe[idx(c[0], c[1])] < 0:
                continue
            open_n = sum(1 for dx, dy in DIRS if passable(g, c[0] + dx, c[1] + dy))
            # Anywhere on the floor that is not a dead end. Holes used to be pinned to
            # the two side walls (x <= 4 or x >= W-5), which meant the middle of every
            # room was hole-free and the mouse always ran to one edge or the other. The
            # trek is not shortened by letting them sit inland — the spawn is drawn from
            # the slice farthest from any hole, so it moves with them; measured over the
            # level set the optimal route barely changed (19.4 -> 19.1 steps at two
            # holes) while the middle of the room stopped being dead space.
            if open_n >= 2:
                nest_cands.append(c)

        anchor = None
        for c in free:
            if probe[idx(c[0], c[1])] >= 0:
                anchor = c
                break

        nests: list[tuple[int, int]] = []
        for _slot in range(nest_count):
            picked = None
            for _ in range(24):
                if picked is not None or not nest_cands:
                    break
                cand = nest_cands.pop(floor_mul(r(), len(nest_cands)))
                if cand[0] == anchor[0] and cand[1] == anchor[1]:
                    continue
                cand_field = bfs(g, cand[0], cand[1])
                far_enough = True
                for other in nests:
                    dd = cand_field[idx(other[0], other[1])]
                    if dd < 0 or dd < MIN_NEST_SEP:
                        far_enough = False
                        break
                if not far_enough:
                    continue
                ok2 = True
                for dx, dy in DIRS:
                    if not ok2:
                        break
                    nb = (cand[0] + dx, cand[1] + dy)
                    if not passable(g, nb[0], nb[1]):
                        continue
                    if nb[0] == anchor[0] and nb[1] == anchor[1]:
                        continue
                    saved = g[idx(nb[0], nb[1])]
                    g[idx(nb[0], nb[1])] = WALL
                    probe2 = bfs(g, anchor[0], anchor[1])
                    if probe2[idx(cand[0], cand[1])] < 0:
                        ok2 = False
                    g[idx(nb[0], nb[1])] = saved
                if ok2:
                    picked = cand
            if picked is None:
                break
            nests.append(picked)
        if len(nests) < nest_count:
            continue
        for n in nests:
            g[idx(n[0], n[1])] = NEST
        nest = nests[0]

        nest_field = bfs_multi(g, nests)

        # How far "far from home" is depends on how many holes there are: two holes on
        # opposite walls halve the longest possible trek, so a fixed 21-32 range simply
        # cannot be met. The spawn is instead drawn from the farthest slice of the room,
        # measured against this map's own maximum. Integer arithmetic, so both languages
        # agree exactly.
        dmax = 0
        for c in free:
            fd = nest_field[idx(c[0], c[1])]
            if fd > dmax:
                dmax = fd
        if dmax < 10:
            continue
        m_min = dmax * 72 // 100
        m_cands = [c for c in free
                   if nest_field[idx(c[0], c[1])] >= m_min and g[idx(c[0], c[1])] == EMPTY]
        if not m_cands:
            continue
        mouse = m_cands[floor_mul(r(), len(m_cands))]
        mouse_field = bfs(g, mouse[0], mouse[1])

        c_max = max(10, dmax * 85 // 100)
        c_cands = []
        for c in free:
            dm = mouse_field[idx(c[0], c[1])]
            dn = nest_field[idx(c[0], c[1])]
            if dm >= 10 and 4 <= dn <= c_max and g[idx(c[0], c[1])] == EMPTY:
                c_cands.append(c)
        if not c_cands:
            continue
        cat = c_cands[floor_mul(r(), len(c_cands))]

        # Her shortest route home. Hazards have to sit ON it.
        path: list[tuple[int, int]] = []
        px, py = mouse
        pg = 0
        while nest_field[idx(px, py)] > 0 and pg < 400:
            pg += 1
            path.append((px, py))
            bd = nest_field[idx(px, py)]
            bx, by = px, py
            for dx, dy in DIRS:
                qx, qy = px + dx, py + dy
                if not passable(g, qx, qy):
                    continue
                qd = nest_field[idx(qx, qy)]
                if 0 <= qd < bd:
                    bd, bx, by = qd, qx, qy
            if bx == px and by == py:
                break
            px, py = bx, by

        traps: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()

        def place_trap(c) -> bool:
            if c is None or not in_b(c[0], c[1]):
                return False
            if c in used:
                return False
            if g[idx(c[0], c[1])] != EMPTY:
                return False
            if c[0] == cat[0] and c[1] == cat[1]:
                return False
            if c[0] == mouse[0] and c[1] == mouse[1]:
                return False
            if nest_field[idx(c[0], c[1])] < 2:  # keep the nest mouth clear
                return False
            used.add(c)
            g[idx(c[0], c[1])] = TRAP
            traps.append(c)
            return True

        for fr in (0.24, 0.5, 0.76):
            if len(traps) >= 3:
                break
            at = min(len(path) - 2, max(3, round_js(len(path) * fr)))
            if 0 <= at < len(path):
                place_trap(path[at])

        pi = round_js(len(path) * 0.4)
        while pi < len(path) - 1 and len(traps) < 5:
            for dx, dy in DIRS:
                if len(traps) >= 5:
                    break
                place_trap((path[pi][0] + dx, path[pi][1] + dy))
            pi += 3

        t_cands = []
        for c in free:
            if g[idx(c[0], c[1])] != EMPTY:
                continue
            dm = mouse_field[idx(c[0], c[1])]
            dn = nest_field[idx(c[0], c[1])]
            if dm >= 3 and dn >= 2:
                t_cands.append(c)
        while len(traps) < 6 and t_cands:
            place_trap(t_cands.pop(floor_mul(r(), len(t_cands))))
        if len(traps) < 4:
            continue  # reseed rather than ship a hazard-free map

        return Map(
            seed=seed, grid=g, blocks=blocks, pillars=pillars, traps=traps,
            nest=nest, nests=nests, cat_spawn=cat, mouse_spawn=mouse, route=path,
            nest_field=nest_field, optimal=nest_field[idx(mouse[0], mouse[1])],
        )
    return gen_map((seed + 104729) & 0xFFFFFFFF, nest_count)


def round_js(v: float) -> int:
    """JS Math.round: half rounds toward +Infinity (Python's round() is banker's)."""
    return math.floor(v + 0.5)


# ---------- sensors ----------

def cast_ray(grid, ox: float, oy: float, dx: float, dy: float, rng: float):
    """One ray, walked cell boundary to cell boundary, to the first wall face it meets
    or to `rng` if it meets none. Returns (exact distance, 1 if it hit).

    The old caster advanced in 0.18 steps and, on contact, backed off a whole step, so
    every ray stopped somewhere within the last 0.18 cells before the wall and by a
    different amount each: one flat wall came back with 0.17 cells of wobble along it.
    It also tested one step BEYOND the range, letting an unobstructed ray reach 8.64 of
    a nominal 8.5 (readings of 1.016 on a [0,1] input).

    A ray crossing exactly through a lattice corner passes only if at least one of the
    two cells sharing that corner is open: light never squeezes through a zero-width
    diagonal gap, but it does graze the tip of a single block.

    Agents stand on cell centres, so an exact 45-degree sightline is common, not a
    measure-zero curiosity, and the two halves of that rule are the two ways to be wrong.
    Grazing the tip lets sight through the odd corner-to-corner slit: 1 pair in 4,788
    sampled. Refusing it instead punches a blind line through the middle of a region the
    cone correctly draws as lit, which is eight times more common and is the failure a
    viewer actually notices — the mouse standing in the light, unseen.
    """
    cx, cy = math.floor(ox), math.floor(oy)
    if not passable(grid, cx, cy):
        return 0.0, 1
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    dtx = math.inf if dx == 0 else abs(1.0 / dx)
    dty = math.inf if dy == 0 else abs(1.0 / dy)
    tx = math.inf if dx == 0 else (((cx + 1) if dx > 0 else cx) - ox) / dx
    ty = math.inf if dy == 0 else (((cy + 1) if dy > 0 else cy) - oy) / dy
    for _ in range(2 * (W + H)):
        if tx < ty:
            t = tx
            tx += dtx
            cx += sx
        elif ty < tx:
            t = ty
            ty += dty
            cy += sy
        else:
            t = tx
            if t >= rng:
                break
            if not passable(grid, cx + sx, cy) and not passable(grid, cx, cy + sy):
                return t, 1
            tx += dtx
            ty += dty
            cx += sx
            cy += sy
        if t >= rng:
            break
        if not passable(grid, cx, cy):
            return t, 1
    return rng, 0


def cone_corners(grid, ox: float, oy: float, base: float, half: float, rng: float):
    """The lattice points inside the cone where a wall silhouette pivots.

    Twenty-one rays over 100 degrees is one every five, which at full range samples the
    world every 0.74 cells, so the fan almost never lands ON the corner a shadow turns
    around; the polygon then bridged a near hit and a far miss with one straight edge,
    and that edge is the spike. A ray a hair either side of every corner puts a real
    vertex where the silhouette actually turns.
    """
    out = []
    x0 = max(0, math.ceil(ox - rng))
    x1 = min(W, math.floor(ox + rng))
    y0 = max(0, math.ceil(oy - rng))
    y1 = min(H, math.floor(oy + rng))
    for py in range(y0, y1 + 1):
        for px in range(x0, x1 + 1):
            dx, dy = px - ox, py - oy
            dd = dx * dx + dy * dy
            if dd < 1e-12 or dd > rng * rng:
                continue
            da = math.atan2(dy, dx) - base
            da = math.atan2(math.sin(da), math.cos(da))
            if da < -half or da > half:
                continue
            # only a corner with solid on one side and open on the other casts an edge
            k = ((0 if passable(grid, px - 1, py - 1) else 1)
                 + (0 if passable(grid, px, py - 1) else 1)
                 + (0 if passable(grid, px - 1, py) else 1)
                 + (0 if passable(grid, px, py) else 1))
            if k == 0 or k == 4:
                continue
            out.append(da)
    return out


def cast_rays(grid, ax: float, ay: float, facing: int, role: str):
    """The VISION_RAYS evenly spaced readings a policy consumes, in fan order, in [0,1].

    This is the hot path — vec.py bakes it for every cell and facing — so it skips the
    corner sweep that only the drawn polygon needs.
    """
    half = half_angle(role)
    base = facing * math.pi / 2 - math.pi / 2  # facing 0 = N
    out = []
    for i in range(VISION_RAYS):
        a = base - half + (2 * half) * (i / (VISION_RAYS - 1))
        t, _ = cast_ray(grid, ax + 0.5, ay + 0.5, math.cos(a), math.sin(a), VISION_RANGE)
        out.append(t / VISION_RANGE)
    return out


def cast_cone(grid, ax: float, ay: float, facing: int, role: str):
    """Vision cone with wall occlusion. Returns (polygon, ray distances in [0,1]).

    The distances are exactly `cast_rays` — the observation vector's shape is a
    contract. The polygon carries the extra corner vertices on top of them.
    """
    half = half_angle(role)
    base = facing * math.pi / 2 - math.pi / 2  # facing 0 = N
    ox, oy = ax + 0.5, ay + 0.5
    reads = []
    verts = []

    for i in range(VISION_RAYS):
        a = base - half + (2 * half) * (i / (VISION_RAYS - 1))
        t, _ = cast_ray(grid, ox, oy, math.cos(a), math.sin(a), VISION_RANGE)
        reads.append(t / VISION_RANGE)
        verts.append((a, t))

    for c in cone_corners(grid, ox, oy, base, half, VISION_RANGE):
        for s in (-1, 1):
            da = c + s * CORNER_EPS
            if da <= -half or da >= half:
                continue  # the rim rays already stand here
            a = base + da
            t, _ = cast_ray(grid, ox, oy, math.cos(a), math.sin(a), VISION_RANGE)
            verts.append((a, t))

    # every angle lies inside base +- half, a span under a half turn, so raw order is
    # fan order and no wrap handling is needed
    verts.sort(key=lambda v: v[0])
    poly = [(ox, oy)] + [(ox + math.cos(a) * t, oy + math.sin(a) * t) for a, t in verts]
    return poly, reads


def line_of_sight(grid, x0: int, y0: int, x1: int, y1: int) -> bool:
    """Centre to centre sight, walked with the very ray the cone is drawn from, so the
    cone on screen and the targetVisible bit can no longer disagree. The old test
    interpolated three samples per cell and stepped over walls it clipped diagonally."""
    dx, dy = x1 - x0, y1 - y0
    d = math.sqrt(dx * dx + dy * dy)
    if d == 0:
        return True
    return cast_ray(grid, x0 + 0.5, y0 + 0.5, dx / d, dy / d, d)[1] == 0


def sees_target(grid, ax: int, ay: int, facing: int, tx: int, ty: int, role: str) -> bool:
    d = math.hypot(tx - ax, ty - ay)
    if d > VISION_RANGE:
        return False
    base = facing * math.pi / 2 - math.pi / 2
    a = math.atan2(ty - ay, tx - ax)
    diff = math.atan2(math.sin(a - base), math.cos(a - base))
    if abs(diff) > half_angle(role):
        return False
    return line_of_sight(grid, ax, ay, tx, ty)


# ---------- episode ----------

@dataclass
class Actor:
    x: int
    y: int
    facing: int
    frozen: int = 0
    trapped: int = 0
    reward: float = 0.0


@dataclass
class Heard:
    x: float
    y: float
    conf: float
    radius: float


@dataclass
class State:
    map: Map
    seed: int
    rand: JsRng
    cat: Actor
    mouse: Actor
    scent: list[float]
    step_n: int = 0
    done: bool = False
    result: Optional[str] = None
    heard: Optional[Heard] = None
    saw_mouse: bool = False
    saw_cat: bool = False
    trap_hits: int = 0

    @property
    def grid(self):
        return self.map.grid


def reset(m: Map, seed: int) -> State:
    seed &= 0xFFFFFFFF
    return State(
        map=m, seed=seed, rand=JsRng(seed ^ 0x9E3779B9),
        cat=Actor(m.cat_spawn[0], m.cat_spawn[1], 1),
        mouse=Actor(m.mouse_spawn[0], m.mouse_spawn[1], 3),
        scent=[0.0] * (W * H),
    )


def observe(s: State, role: str) -> dict:
    """The observation handed to a policy. Keys mirror env.js observe()."""
    g = s.grid
    me = s.cat if role == "cat" else s.mouse
    other = s.mouse if role == "cat" else s.cat
    rays = cast_rays(g, me.x, me.y, me.facing, role)
    visible = sees_target(g, me.x, me.y, me.facing, other.x, other.y, role)
    nest = s.map.nest
    obs = {
        "role": role,
        "pos": (me.x / W, me.y / H),
        "facing": me.facing,
        "frozen": me.frozen / FREEZE_STEPS,
        "rays": rays,
        "nestBearing": math.atan2(nest[1] - me.y, nest[0] - me.x),
        "nestDist": s.map.nest_field[idx(me.x, me.y)] / (W + H),
        "targetVisible": 1 if visible else 0,
        "targetBearing": math.atan2(other.y - me.y, other.x - me.x) if visible else 0.0,
        "targetDist": (math.hypot(other.x - me.x, other.y - me.y) / VISION_RANGE) if visible else 1.0,
        "stepFrac": s.step_n / MAX_STEPS,
    }
    if role == "mouse":
        obs["heard"] = ((s.heard.x / W, s.heard.y / H, s.heard.conf) if s.heard else None)
    else:
        best = None
        field = bfs(g, me.x, me.y)
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                v = s.scent[idx(x, y)]
                if v < 0.08:
                    continue
                d = field[idx(x, y)]
                if d < 0 or d > SCENT_RANGE:
                    continue
                if best is None or v > best[2]:
                    best = (x, y, v)
        obs["scent"] = ((best[0] / W, best[1] / H, best[2]) if best else None)
        obs["_catField"] = field
    return obs


def _try_move(g, a: Actor, action: int):
    if action == 0:
        return a.x, a.y, a.facing
    dx, dy = DIRS[action - 1]
    nx, ny = a.x + dx, a.y + dy
    if not passable(g, nx, ny):
        return a.x, a.y, action - 1
    return nx, ny, action - 1


def step(s: State, cat_action: int, mouse_action: int) -> dict:
    """Simultaneous resolution. Returns which agents newly snapped a trap."""
    g = s.grid
    ev = {"trapCat": False, "trapMouse": False}
    prev_cat_to_mouse = math.hypot(s.cat.x - s.mouse.x, s.cat.y - s.mouse.y)
    prev_mouse_to_nest = s.map.nest_field[idx(s.mouse.x, s.mouse.y)]
    pcx, pcy = s.cat.x, s.cat.y
    pmx, pmy = s.mouse.x, s.mouse.y

    for role, a, action in (("cat", s.cat, cat_action), ("mouse", s.mouse, mouse_action)):
        if a.frozen > 0:
            a.frozen -= 1
            a.reward += R_TRAP * 0.2
            continue
        a.x, a.y, a.facing = _try_move(g, a, action)
        if g[idx(a.x, a.y)] == TRAP:
            a.frozen = FREEZE_STEPS
            a.trapped += 1
            a.reward += R_TRAP
            ev["trapCat" if role == "cat" else "trapMouse"] = True

    for i in range(len(s.scent)):
        s.scent[i] *= SCENT_DECAY
    s.scent[idx(s.mouse.x, s.mouse.y)] = 1.0

    d_cat = math.hypot(s.cat.x - s.mouse.x, s.cat.y - s.mouse.y)
    if s.cat.frozen == 0 and d_cat <= HEARING_RANGE:
        noise = HEARING_BASE_NOISE + HEARING_DIST_NOISE * d_cat
        jx = (s.rand() - 0.5) * noise * 6
        jy = (s.rand() - 0.5) * noise * 6
        s.heard = Heard(
            x=min(W - 1, max(0, s.cat.x + jx)),
            y=min(H - 1, max(0, s.cat.y + jy)),
            conf=max(0.08, 1 - d_cat / HEARING_RANGE),
            radius=1 + noise * 7,
        )
    elif s.heard is not None:
        s.heard.conf *= 0.82
        if s.heard.conf < 0.1:
            s.heard = None

    s.saw_mouse = sees_target(g, s.cat.x, s.cat.y, s.cat.facing, s.mouse.x, s.mouse.y, "cat")
    s.saw_cat = sees_target(g, s.mouse.x, s.mouse.y, s.mouse.facing, s.cat.x, s.cat.y, "mouse")

    new_cat_to_mouse = math.hypot(s.cat.x - s.mouse.x, s.cat.y - s.mouse.y)
    new_mouse_to_nest = s.map.nest_field[idx(s.mouse.x, s.mouse.y)]
    if s.saw_mouse:
        s.cat.reward += R_CAT_APPROACH * (prev_cat_to_mouse - new_cat_to_mouse)
    s.mouse.reward += R_MOUSE_APPROACH * (prev_mouse_to_nest - new_mouse_to_nest)
    s.cat.reward += R_CAT_STEP
    s.mouse.reward += R_MOUSE_STEP

    s.step_n += 1
    man = abs(s.cat.x - s.mouse.x) + abs(s.cat.y - s.mouse.y)
    swapped = (not (s.cat.x == pcx and s.cat.y == pcy)) and \
        s.cat.x == pmx and s.cat.y == pmy and s.mouse.x == pcx and s.mouse.y == pcy
    caught = man <= 1 or swapped
    escaped = any(s.mouse.x == n[0] and s.mouse.y == n[1] for n in s.map.nests)

    # Reaching the hole beats a simultaneous catch — she is inside.
    if escaped:
        s.done, s.result = True, "escape"
        s.mouse.reward += R_MOUSE_NEST
        s.cat.reward += R_CAT_ESCAPED
    elif caught:
        s.done, s.result = True, "catch"
        s.cat.reward += R_CAT_CATCH
        s.mouse.reward += R_MOUSE_CAUGHT
    elif s.step_n >= MAX_STEPS:
        s.done, s.result = True, "timeout"
    if ev["trapCat"] or ev["trapMouse"]:
        s.trap_hits += 1
    return ev
