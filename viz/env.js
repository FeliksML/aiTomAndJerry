/* Cat & Mouse Lab — reference environment.
 * Single source of truth for map generation, sensors, dynamics, rewards.
 * Port this file to Python 1:1 (same seeds => same maps) when wiring real training.
 * Exposes window.CatMouseEnv and module.exports.
 */
(function (global) {
  'use strict';

  var W = 27, H = 19;
  var EMPTY = 0, WALL = 1, TRAP = 2, NEST = 3;
  var DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]];          // N E S W
  var ACTIONS = ['stay', 'north', 'east', 'south', 'west']; // action space (5, discrete)

  // How many holes a room may have, and how far apart they must be. One hole makes
  // camping it a dominant strategy; two give the mouse a choice the cat cannot cover.
  var MAX_NESTS = 3;
  var MIN_NEST_SEP = 10;

  var CFG = {
    vision: { halfAngleDeg: 50, range: 8.5, rays: 21 },
    hearing: { range: 12, baseNoise: 0.10, distNoise: 0.055 }, // mouse only
    scent: { range: 6, decay: 0.93 },                          // cat only
    freezeSteps: 5,
    maxSteps: 180,
    reward: {
      catCatch: 1, catEscaped: -1, catStep: -0.002, catApproach: 0.012,
      mouseNest: 1, mouseCaught: -1, mouseStep: -0.002, mouseApproach: 0.010,
      trap: -0.05
    }
  };

  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function idx(x, y) { return y * W + x; }
  function inB(x, y) { return x >= 0 && x < W && y >= 0 && y < H; }
  function passable(g, x, y) { return inB(x, y) && g[idx(x, y)] !== WALL; }

  function bfs(g, sx, sy) {
    var d = new Int16Array(W * H).fill(-1);
    if (!passable(g, sx, sy)) return d;
    var q = [sx | 0, sy | 0], head = 0;
    d[idx(sx, sy)] = 0;
    while (head < q.length) {
      var x = q[head++], y = q[head++], nd = d[idx(x, y)] + 1;
      for (var i = 0; i < 4; i++) {
        var nx = x + DIRS[i][0], ny = y + DIRS[i][1];
        if (!passable(g, nx, ny)) continue;
        if (d[idx(nx, ny)] !== -1) continue;
        d[idx(nx, ny)] = nd; q.push(nx, ny);
      }
    }
    return d;
  }

  // Distance to the NEAREST of several sources. With more than one hole the mouse's
  // "distance home" is the distance to whichever hole is closest, and every shaping,
  // observation and route calculation downstream then works unchanged.
  function bfsMulti(g, sources) {
    var d = new Int16Array(W * H).fill(-1);
    var q = [], head = 0, i;
    for (i = 0; i < sources.length; i++) {
      var sx = sources[i][0], sy = sources[i][1];
      if (!passable(g, sx, sy) || d[idx(sx, sy)] === 0) continue;
      d[idx(sx, sy)] = 0; q.push(sx, sy);
    }
    while (head < q.length) {
      var x = q[head++], y = q[head++], nd = d[idx(x, y)] + 1;
      for (i = 0; i < 4; i++) {
        var nx = x + DIRS[i][0], ny = y + DIRS[i][1];
        if (!passable(g, nx, ny)) continue;
        if (d[idx(nx, ny)] !== -1) continue;
        d[idx(nx, ny)] = nd; q.push(nx, ny);
      }
    }
    return d;
  }

  /* ---------- map generation ---------- */

  function genMap(seed, nestCount) {
    nestCount = Math.max(1, Math.min(MAX_NESTS, nestCount || 1));
    for (var attempt = 0; attempt < 80; attempt++) {
      var r = rng((seed >>> 0) + attempt * 7919);
      var g = new Uint8Array(W * H);
      var x, y, i;
      for (x = 0; x < W; x++) { g[idx(x, 0)] = WALL; g[idx(x, H - 1)] = WALL; }
      for (y = 0; y < H; y++) { g[idx(0, y)] = WALL; g[idx(W - 1, y)] = WALL; }

      var blocks = [], tries = 0;
      while (blocks.length < 7 && tries++ < 400) {
        var bw = 2 + Math.floor(r() * 4), bh = 2 + Math.floor(r() * 3);
        var bx = 2 + Math.floor(r() * (W - 4 - bw)), by = 2 + Math.floor(r() * (H - 4 - bh));
        var ok = true;
        for (i = 0; i < blocks.length; i++) {
          var b = blocks[i];
          if (bx - 2 < b.x + b.w && bx + bw + 2 > b.x && by - 2 < b.y + b.h && by + bh + 2 > b.y) { ok = false; break; }
        }
        if (!ok) continue;
        blocks.push({ x: bx, y: by, w: bw, h: bh });
        for (y = by; y < by + bh; y++) for (x = bx; x < bx + bw; x++) g[idx(x, y)] = WALL;
      }

      var pil = 0, pt = 0, pillars = [];
      while (pil < 10 && pt++ < 400) {
        var px = 2 + Math.floor(r() * (W - 4)), py = 2 + Math.floor(r() * (H - 4));
        if (g[idx(px, py)] !== EMPTY) continue;
        var near = 0;
        for (var dy = -1; dy <= 1; dy++) for (var dx = -1; dx <= 1; dx++) if (g[idx(px + dx, py + dy)] === WALL) near++;
        if (near > 1) continue;
        g[idx(px, py)] = WALL; pillars.push([px, py]); pil++;
      }

      // connectivity: one open region must hold nearly every free cell
      var free = [];
      for (y = 1; y < H - 1; y++) for (x = 1; x < W - 1; x++) if (g[idx(x, y)] !== WALL) free.push([x, y]);
      if (free.length < 200) continue;
      var probe = bfs(g, free[0][0], free[0][1]);
      var reach = 0;
      for (i = 0; i < free.length; i++) if (probe[idx(free[i][0], free[i][1])] >= 0) reach++;
      if (reach < free.length * 0.94) continue;

      // Each hole needs TWO INDEPENDENT approaches, not just two open neighbours:
      // with a capture radius of 1 a single camping cat can seal a one-corridor
      // pocket completely, and every episode then dies on the step limit.
      //
      // With more than one hole the mouse gets a CHOICE, which is the point — one hole
      // makes camping it a dominant strategy and the chase collapses into a stakeout.
      // Holes are therefore also kept MIN_NEST_SEP apart, so a cat standing between
      // two of them cannot cover both.
      var nestCands = free.filter(function (c) {
        if (probe[idx(c[0], c[1])] < 0) return false;
        var open = 0;
        for (var k = 0; k < 4; k++) if (passable(g, c[0] + DIRS[k][0], c[1] + DIRS[k][1])) open++;
        return open >= 2 && (c[0] <= 4 || c[0] >= W - 5);
      });
      var anchor = null, ni, nk, nb, saved, probe2;
      for (ni = 0; ni < free.length && !anchor; ni++) if (probe[idx(free[ni][0], free[ni][1])] >= 0) anchor = free[ni];

      var nests = [], wanted = nestCount;
      for (var slot = 0; slot < wanted; slot++) {
        var picked = null;
        for (ni = 0; ni < 24 && nestCands.length && !picked; ni++) {
          var cand = nestCands.splice(Math.floor(r() * nestCands.length), 1)[0];
          if (cand[0] === anchor[0] && cand[1] === anchor[1]) continue;
          var candField = bfs(g, cand[0], cand[1]);
          var farEnough = true;
          for (nk = 0; nk < nests.length; nk++) {
            var dd = candField[idx(nests[nk][0], nests[nk][1])];
            if (dd < 0 || dd < MIN_NEST_SEP) { farEnough = false; break; }
          }
          if (!farEnough) continue;
          var ok2 = true;
          for (nk = 0; nk < 4 && ok2; nk++) {
            nb = [cand[0] + DIRS[nk][0], cand[1] + DIRS[nk][1]];
            if (!passable(g, nb[0], nb[1])) continue;
            if (nb[0] === anchor[0] && nb[1] === anchor[1]) continue;
            saved = g[idx(nb[0], nb[1])];
            g[idx(nb[0], nb[1])] = WALL;
            probe2 = bfs(g, anchor[0], anchor[1]);
            if (probe2[idx(cand[0], cand[1])] < 0) ok2 = false;
            g[idx(nb[0], nb[1])] = saved;
          }
          if (ok2) picked = cand;
        }
        if (!picked) break;
        nests.push(picked);
      }
      if (nests.length < wanted) continue;
      for (ni = 0; ni < nests.length; ni++) g[idx(nests[ni][0], nests[ni][1])] = NEST;
      var nest = nests[0];

      var nestField = bfsMulti(g, nests);

      // How far "far from home" is depends on how many holes there are: two holes on
      // opposite walls halve the longest possible trek, so a fixed 21-32 range simply
      // cannot be met. The spawn is instead drawn from the farthest slice of the room,
      // measured against this map's own maximum. Integer arithmetic, so both languages
      // agree exactly.
      var dmax = 0;
      for (i = 0; i < free.length; i++) {
        var fd = nestField[idx(free[i][0], free[i][1])];
        if (fd > dmax) dmax = fd;
      }
      if (dmax < 10) continue;
      var mMin = Math.floor(dmax * 72 / 100);
      var mCands = free.filter(function (c) {
        return nestField[idx(c[0], c[1])] >= mMin && g[idx(c[0], c[1])] === EMPTY;
      });
      if (!mCands.length) continue;
      var mouse = mCands[Math.floor(r() * mCands.length)];
      var mouseField = bfs(g, mouse[0], mouse[1]);

      // cat: far from the mouse, but able to reach a hole first if it wants to camp one
      var cMax = Math.max(10, Math.floor(dmax * 85 / 100));
      var cCands = free.filter(function (c) {
        var dm = mouseField[idx(c[0], c[1])], dn = nestField[idx(c[0], c[1])];
        return dm >= 10 && dn >= 4 && dn <= cMax && g[idx(c[0], c[1])] === EMPTY;
      });
      if (!cCands.length) continue;
      var cat = cCands[Math.floor(r() * cCands.length)];

      // The mouse's shortest route to the nest. Hazards have to sit ON it — a trap
      // parked in an unvisited corner is never learned and never seen.
      var path = [], px = mouse[0], py = mouse[1], pg = 0, qx, qy, qd, bd, bx, by;
      while (nestField[idx(px, py)] > 0 && pg++ < 400) {
        path.push([px, py]);
        bd = nestField[idx(px, py)]; bx = px; by = py;
        for (i = 0; i < 4; i++) {
          qx = px + DIRS[i][0]; qy = py + DIRS[i][1];
          if (!passable(g, qx, qy)) continue;
          qd = nestField[idx(qx, qy)];
          if (qd >= 0 && qd < bd) { bd = qd; bx = qx; by = qy; }
        }
        if (bx === px && by === py) break;
        px = bx; py = by;
      }

      var traps = [], used = {};
      var placeTrap = function (c) {
        if (!c || !inB(c[0], c[1])) return false;
        var k = c[0] + ',' + c[1];
        if (used[k]) return false;
        if (g[idx(c[0], c[1])] !== EMPTY) return false;
        if (c[0] === cat[0] && c[1] === cat[1]) return false;
        if (c[0] === mouse[0] && c[1] === mouse[1]) return false;
        if (nestField[idx(c[0], c[1])] < 2) return false;   // keep the nest mouth clear
        used[k] = 1; g[idx(c[0], c[1])] = TRAP; traps.push([c[0], c[1]]);
        return true;
      };

      // three spread along the route — quarter, middle, three-quarter — never bunched
      // at the spawn, so the mouse is tested the whole way rather than in the first
      // three steps
      var fr = [0.24, 0.5, 0.76], at;
      for (i = 0; i < fr.length && traps.length < 3; i++) {
        at = Math.min(path.length - 2, Math.max(3, Math.round(path.length * fr[i])));
        if (path[at]) placeTrap(path[at]);
      }
      // two flanking the second half, so detouring is not automatically safe either
      for (var pi = Math.round(path.length * 0.4); pi < path.length - 1 && traps.length < 5; pi += 3) {
        for (i = 0; i < 4 && traps.length < 5; i++) {
          placeTrap([path[pi][0] + DIRS[i][0], path[pi][1] + DIRS[i][1]]);
        }
      }
      // remainder scattered, including the cat's likely approach to the nest
      var tCands = free.filter(function (c) {
        if (g[idx(c[0], c[1])] !== EMPTY) return false;
        var dm = mouseField[idx(c[0], c[1])], dn = nestField[idx(c[0], c[1])];
        return dm >= 3 && dn >= 2;
      });
      while (traps.length < 6 && tCands.length) placeTrap(tCands.splice(Math.floor(r() * tCands.length), 1)[0]);
      if (traps.length < 4) continue;   // reseed rather than ship a hazard-free map

      return {
        seed: seed, w: W, h: H, grid: g, blocks: blocks, pillars: pillars,
        traps: traps, nest: nest, nests: nests, catSpawn: cat, mouseSpawn: mouse,
        route: path, nestField: nestField, optimal: nestField[idx(mouse[0], mouse[1])]
      };
    }
    return genMap((seed >>> 0) + 104729, nestCount);
  }

  /* ---------- sensors ---------- */

  // Vision cone with wall occlusion. Returns render polygon + per-ray readings.
  function castCone(grid, ax, ay, facing, opt) {
    var cfg = opt || CFG.vision;
    var half = (cfg.halfAngleDeg || 50) * Math.PI / 180;
    var range = cfg.range, n = cfg.rays;
    var base = facing * Math.PI / 2 - Math.PI / 2; // facing 0=N
    var poly = [[ax + 0.5, ay + 0.5]], reads = [];
    for (var i = 0; i < n; i++) {
      var a = base - half + (2 * half) * (i / (n - 1));
      var dx = Math.sin(a + Math.PI / 2) * 0, dy = 0;
      dx = Math.cos(a); dy = Math.sin(a);
      var t = 0, hit = 0, step = 0.18;
      while (t < range) {
        t += step;
        var px = ax + 0.5 + dx * t, py = ay + 0.5 + dy * t;
        var cx = Math.floor(px), cy = Math.floor(py);
        if (!inB(cx, cy) || grid[idx(cx, cy)] === WALL) { t -= step; hit = 1; break; }
      }
      poly.push([ax + 0.5 + dx * t, ay + 0.5 + dy * t]);
      reads.push({ angle: a, dist: t / range, blocked: hit });
    }
    return { poly: poly, reads: reads, half: half, base: base, range: range };
  }

  function lineOfSight(grid, x0, y0, x1, y1) {
    var dx = x1 - x0, dy = y1 - y0, n = Math.max(Math.abs(dx), Math.abs(dy)) * 3;
    if (n === 0) return true;
    for (var i = 1; i < n; i++) {
      var px = x0 + 0.5 + dx * (i / n), py = y0 + 0.5 + dy * (i / n);
      var cx = Math.floor(px), cy = Math.floor(py);
      if (!inB(cx, cy) || grid[idx(cx, cy)] === WALL) return false;
    }
    return true;
  }

  function seesTarget(grid, ax, ay, facing, tx, ty, opt) {
    var cfg = opt || CFG.vision;
    var d = Math.hypot(tx - ax, ty - ay);
    if (d > cfg.range) return false;
    var base = facing * Math.PI / 2 - Math.PI / 2;
    var a = Math.atan2(ty - ay, tx - ax);
    var diff = Math.atan2(Math.sin(a - base), Math.cos(a - base));
    if (Math.abs(diff) > cfg.halfAngleDeg * Math.PI / 180) return false;
    return lineOfSight(grid, ax, ay, tx, ty);
  }

  /* ---------- episode ---------- */

  function reset(map, seed) {
    return {
      map: map, grid: map.grid, seed: seed >>> 0, rand: rng((seed >>> 0) ^ 0x9e3779b9),
      step: 0, done: false, result: null,
      cat: { x: map.catSpawn[0], y: map.catSpawn[1], facing: 1, frozen: 0, trapped: 0, reward: 0 },
      mouse: { x: map.mouseSpawn[0], y: map.mouseSpawn[1], facing: 3, frozen: 0, trapped: 0, reward: 0 },
      scent: new Float32Array(W * H),
      heard: null, sawMouse: false, sawCat: false
    };
  }

  // Observation vectors handed to a policy. Mirror these keys in Python.
  function observe(s, role) {
    var g = s.grid, me = role === 'cat' ? s.cat : s.mouse, other = role === 'cat' ? s.mouse : s.cat;
    var cone = castCone(g, me.x, me.y, me.facing);
    var visible = seesTarget(g, me.x, me.y, me.facing, other.x, other.y);
    var nest = s.map.nest;
    var obs = {
      role: role,
      pos: [me.x / W, me.y / H],
      facing: me.facing,
      frozen: me.frozen / CFG.freezeSteps,
      rays: cone.reads.map(function (r) { return r.dist; }),
      nestBearing: Math.atan2(nest[1] - me.y, nest[0] - me.x),
      nestDist: s.map.nestField[idx(me.x, me.y)] / (W + H),
      targetVisible: visible ? 1 : 0,
      targetBearing: visible ? Math.atan2(other.y - me.y, other.x - me.x) : 0,
      targetDist: visible ? Math.hypot(other.x - me.x, other.y - me.y) / CFG.vision.range : 1,
      stepFrac: s.step / CFG.maxSteps,
      cone: cone
    };
    if (role === 'mouse') {
      obs.heard = s.heard ? [s.heard.x / W, s.heard.y / H, s.heard.conf] : null;
    } else {
      var best = null, field = bfs(g, me.x, me.y);
      for (var y = 1; y < H - 1; y++) for (var x = 1; x < W - 1; x++) {
        var v = s.scent[idx(x, y)];
        if (v < 0.08) continue;
        var d = field[idx(x, y)];
        if (d < 0 || d > CFG.scent.range) continue;
        if (!best || v > best.v) best = { x: x, y: y, v: v, d: d };
      }
      obs.scent = best ? [best.x / W, best.y / H, best.v] : null;
      obs._catField = field;
    }
    return obs;
  }

  function tryMove(g, a, action) {
    if (action === 0) return { x: a.x, y: a.y, facing: a.facing };
    var d = DIRS[action - 1], nx = a.x + d[0], ny = a.y + d[1];
    if (!passable(g, nx, ny)) return { x: a.x, y: a.y, facing: action - 1 };
    return { x: nx, y: ny, facing: action - 1 };
  }

  // actions: {cat: 0..4, mouse: 0..4}. Simultaneous resolution.
  function step(s, actions) {
    var g = s.grid, R = CFG.reward, ev = { trapCat: false, trapMouse: false };
    var prevCatToMouse = Math.hypot(s.cat.x - s.mouse.x, s.cat.y - s.mouse.y);
    var prevMouseToNest = s.map.nestField[idx(s.mouse.x, s.mouse.y)];
    var pcx = s.cat.x, pcy = s.cat.y, pmx = s.mouse.x, pmy = s.mouse.y;

    ['cat', 'mouse'].forEach(function (role) {
      var a = s[role];
      if (a.frozen > 0) { a.frozen--; a.reward += R.trap * 0.2; return; }
      var m = tryMove(g, a, actions[role] | 0);
      a.x = m.x; a.y = m.y; a.facing = m.facing;
      if (g[idx(a.x, a.y)] === TRAP) {
        a.frozen = CFG.freezeSteps; a.trapped++; a.reward += R.trap;
        ev[role === 'cat' ? 'trapCat' : 'trapMouse'] = true;
      }
    });

    // mouse scent emission + decay
    for (var i = 0; i < s.scent.length; i++) s.scent[i] *= CFG.scent.decay;
    s.scent[idx(s.mouse.x, s.mouse.y)] = 1;

    // mouse hearing: cat's footsteps carry through walls, accuracy falls off
    var dCat = Math.hypot(s.cat.x - s.mouse.x, s.cat.y - s.mouse.y);
    if (s.cat.frozen === 0 && dCat <= CFG.hearing.range) {
      var noise = CFG.hearing.baseNoise + CFG.hearing.distNoise * dCat;
      var jx = (s.rand() - 0.5) * noise * 6, jy = (s.rand() - 0.5) * noise * 6;
      s.heard = {
        x: Math.min(W - 1, Math.max(0, s.cat.x + jx)),
        y: Math.min(H - 1, Math.max(0, s.cat.y + jy)),
        conf: Math.max(0.08, 1 - dCat / CFG.hearing.range), radius: 1 + noise * 7
      };
    } else if (s.heard) {
      s.heard.conf *= 0.82;
      if (s.heard.conf < 0.1) s.heard = null;
    }

    s.sawMouse = seesTarget(g, s.cat.x, s.cat.y, s.cat.facing, s.mouse.x, s.mouse.y);
    s.sawCat = seesTarget(g, s.mouse.x, s.mouse.y, s.mouse.facing, s.cat.x, s.cat.y);

    var newCatToMouse = Math.hypot(s.cat.x - s.mouse.x, s.cat.y - s.mouse.y);
    var newMouseToNest = s.map.nestField[idx(s.mouse.x, s.mouse.y)];
    if (s.sawMouse) s.cat.reward += R.catApproach * (prevCatToMouse - newCatToMouse);
    s.mouse.reward += R.mouseApproach * (prevMouseToNest - newMouseToNest);
    s.cat.reward += R.catStep; s.mouse.reward += R.mouseStep;

    s.step++;
    // Capture radius 1. With simultaneous moves at equal speed a same-cell-only rule
    // is nearly unsatisfiable — the two just swap places forever — so contact means
    // same cell, one cell away, or having swapped through each other.
    var man = Math.abs(s.cat.x - s.mouse.x) + Math.abs(s.cat.y - s.mouse.y);
    var swapped = (s.cat.x === pcx && s.cat.y === pcy) === false &&
      s.cat.x === pmx && s.cat.y === pmy && s.mouse.x === pcx && s.mouse.y === pcy;
    var caught = man <= 1 || swapped;
    var escaped = s.map.nests.some(function (n) { return s.mouse.x === n[0] && s.mouse.y === n[1]; });

    // Reaching the hole beats a simultaneous catch: she is inside. Without this
    // precedence, sitting on the hole is an absolute strategy and episodes stall
    // out on the step limit instead of resolving.
    if (escaped) { s.done = true; s.result = 'escape'; s.mouse.reward += R.mouseNest; s.cat.reward += R.catEscaped; }
    else if (caught) { s.done = true; s.result = 'catch'; s.cat.reward += R.catCatch; s.mouse.reward += R.mouseCaught; }
    else if (s.step >= CFG.maxSteps) { s.done = true; s.result = 'timeout'; }
    return ev;
  }

  var API = {
    W: W, H: H, EMPTY: EMPTY, WALL: WALL, TRAP: TRAP, NEST: NEST,
    MAX_NESTS: MAX_NESTS, MIN_NEST_SEP: MIN_NEST_SEP, bfsMulti: bfsMulti,
    DIRS: DIRS, ACTIONS: ACTIONS, CFG: CFG,
    rng: rng, idx: idx, inB: inB, passable: passable, bfs: bfs,
    genMap: genMap, reset: reset, step: step, observe: observe,
    castCone: castCone, lineOfSight: lineOfSight, seesTarget: seesTarget
  };
  global.CatMouseEnv = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
})(typeof window !== 'undefined' ? window : globalThis);
