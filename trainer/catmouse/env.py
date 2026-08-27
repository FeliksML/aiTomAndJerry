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

VISION_HALF_ANGLE_DEG = 50.0
VISION_RANGE = 8.5
VISION_RAYS = 21
HEARING_RANGE = 12.0
HEARING_BASE_NOISE = 0.10
HEARING_DIST_NOISE = 0.055
SCENT_RANGE = 6
SCENT_DECAY = 0.93
FREEZE_STEPS = 5
MAX_STEPS = 180

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


@dataclass
class Map:
    seed: int
    grid: bytearray
    blocks: list[dict]
    pillars: list[tuple[int, int]]
    traps: list[tuple[int, int]]
    nest: tuple[int, int]
    cat_spawn: tuple[int, int]
    mouse_spawn: tuple[int, int]
    route: list[tuple[int, int]]
    nest_field: list[int]
    optimal: int
    w: int = W
    h: int = H


def gen_map(seed: int) -> Map:
    """Seeded arena generation. Port of env.js genMap — see that file's comments for
    why each constraint exists (two nest approaches, traps on the walked route, etc.)."""
    seed &= 0xFFFFFFFF
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

        # The nest needs two INDEPENDENT approaches: walling any single neighbour
        # must leave it reachable. A one-corridor pocket can be sealed by a camping
        # cat and then every episode dies on the step limit.
        nest_cands = []
        for c in free:
            if probe[idx(c[0], c[1])] < 0:
                continue
            open_n = sum(1 for dx, dy in DIRS if passable(g, c[0] + dx, c[1] + dy))
            if open_n >= 2 and (c[0] <= 4 or c[0] >= W - 5):
                nest_cands.append(c)

        anchor = None
        for c in free:
            if probe[idx(c[0], c[1])] >= 0:
                anchor = c
                break

        nest = None
        for _ in range(24):
            if nest is not None or not nest_cands:
                break
            cand = nest_cands.pop(floor_mul(r(), len(nest_cands)))
            if cand[0] == anchor[0] and cand[1] == anchor[1]:
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
                nest = cand
        if nest is None:
            continue
        g[idx(nest[0], nest[1])] = NEST

        nest_field = bfs(g, nest[0], nest[1])

        m_cands = [c for c in free
                   if 21 <= nest_field[idx(c[0], c[1])] <= 32 and g[idx(c[0], c[1])] == EMPTY]
        if not m_cands:
            continue
        mouse = m_cands[floor_mul(r(), len(m_cands))]
        mouse_field = bfs(g, mouse[0], mouse[1])

        c_cands = []
        for c in free:
            dm = mouse_field[idx(c[0], c[1])]
            dn = nest_field[idx(c[0], c[1])]
            if dm >= 10 and 5 <= dn <= 17 and g[idx(c[0], c[1])] == EMPTY:
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
            nest=nest, cat_spawn=cat, mouse_spawn=mouse, route=path,
            nest_field=nest_field, optimal=nest_field[idx(mouse[0], mouse[1])],
        )
    return gen_map((seed + 104729) & 0xFFFFFFFF)


def round_js(v: float) -> int:
    """JS Math.round: half rounds toward +Infinity (Python's round() is banker's)."""
    return math.floor(v + 0.5)


# ---------- sensors ----------

def cast_cone(grid, ax: int, ay: int, facing: int):
    """Vision cone with wall occlusion. Returns (polygon, ray distances in [0,1])."""
    half = VISION_HALF_ANGLE_DEG * math.pi / 180.0
    base = facing * math.pi / 2 - math.pi / 2  # facing 0 = N
    poly = [(ax + 0.5, ay + 0.5)]
    reads = []
    for i in range(VISION_RAYS):
        a = base - half + (2 * half) * (i / (VISION_RAYS - 1))
        dx, dy = math.cos(a), math.sin(a)
        t = 0.0
        step = 0.18
        while t < VISION_RANGE:
            t += step
            cx = math.floor(ax + 0.5 + dx * t)
            cy = math.floor(ay + 0.5 + dy * t)
            if not in_b(cx, cy) or grid[idx(cx, cy)] == WALL:
                t -= step
                break
        poly.append((ax + 0.5 + dx * t, ay + 0.5 + dy * t))
        reads.append(t / VISION_RANGE)
    return poly, reads


def line_of_sight(grid, x0: int, y0: int, x1: int, y1: int) -> bool:
    dx, dy = x1 - x0, y1 - y0
    n = max(abs(dx), abs(dy)) * 3
    if n == 0:
        return True
    for i in range(1, n):
        cx = math.floor(x0 + 0.5 + dx * (i / n))
        cy = math.floor(y0 + 0.5 + dy * (i / n))
        if not in_b(cx, cy) or grid[idx(cx, cy)] == WALL:
            return False
    return True


def sees_target(grid, ax: int, ay: int, facing: int, tx: int, ty: int) -> bool:
    d = math.hypot(tx - ax, ty - ay)
    if d > VISION_RANGE:
        return False
    base = facing * math.pi / 2 - math.pi / 2
    a = math.atan2(ty - ay, tx - ax)
    diff = math.atan2(math.sin(a - base), math.cos(a - base))
    if abs(diff) > VISION_HALF_ANGLE_DEG * math.pi / 180.0:
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
    _, rays = cast_cone(g, me.x, me.y, me.facing)
    visible = sees_target(g, me.x, me.y, me.facing, other.x, other.y)
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

    s.saw_mouse = sees_target(g, s.cat.x, s.cat.y, s.cat.facing, s.mouse.x, s.mouse.y)
    s.saw_cat = sees_target(g, s.mouse.x, s.mouse.y, s.mouse.facing, s.cat.x, s.cat.y)

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
    escaped = (s.mouse.x == s.map.nest[0] and s.mouse.y == s.map.nest[1])

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
