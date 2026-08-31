/* Cat & Mouse Lab — placeholder policies.
 * THIS IS THE FILE REAL TRAINING REPLACES.
 * Each (algorithm, role) pair is a `skill` scalar in [0,1] driving a scored
 * greedy controller: skill 0 ≈ random walk, skill 1 ≈ competent hunter/evader.
 * Swap `makePolicy` for a real network forward pass, or feed frames through
 * window.CATMOUSE_ADAPTER (see README.md) and delete this file entirely.
 */
(function (global) {
  'use strict';
  var E = global.CatMouseEnv;

  var ALGOS = {
    ppo: {
      key: 'ppo', short: 'PPO', full: 'Proximal Policy Optimization',
      line: 'gradient · clipped trust region',
      color: '#4ea8ff', light: '#a6d3ff', deep: '#0a2544',
      // steady, low-variance, monotone
      curve: function (e, off) { return 1 / (1 + Math.exp(-(e - 20 + off) / 6.2)); },
      jitter: 0.035, batch: 8
    },
    ga: {
      key: 'ga', short: 'GA', full: 'Genetic Algorithm',
      line: 'population 48 · elitism · mutation 0.08',
      color: '#3ddc84', light: '#9af0be', deep: '#08331f',
      // plateaus then generational leaps
      curve: function (e, off) {
        var gen = Math.floor((e + off) / 6);
        return 1 / (1 + Math.exp(-(gen - 3.4) / 1.15));
      },
      jitter: 0.06, batch: 48
    },
    cmaes: {
      key: 'cmaes', short: 'CMA-ES', full: 'Covariance Matrix Adaptation',
      line: 'σ-adaptation · λ=16 · rank-μ update',
      color: '#a97cff', light: '#cdb2ff', deep: '#26134d',
      // fast early climb, noisy exploration bursts
      curve: function (e, off) {
        var t = Math.max(0, e + off);
        return (1 - Math.exp(-t / 9.5)) * (0.93 + 0.07 * Math.sin(t / 3.1));
      },
      jitter: 0.085, batch: 16
    }
  };

  // Per-role talent offsets so the leaderboard is not a three-way tie.
  var ROLE_BIAS = {
    ppo: { cat: 0, mouse: -2 },
    ga: { cat: -4, mouse: 5 },
    cmaes: { cat: 3, mouse: -3 }
  };

  function skillFor(algoKey, role, episode) {
    var a = ALGOS[algoKey];
    var raw = a.curve(episode, ROLE_BIAS[algoKey][role] || 0);
    var noise = (Math.sin(episode * 12.9898 + (role === 'cat' ? 1.7 : 4.3)) * 43758.5453) % 1;
    return Math.max(0.04, Math.min(0.985, raw * 0.94 + 0.05 + noise * a.jitter));
  }

  /* ---------- scored greedy controller ---------- */

  function candidates(s, a) {
    var out = [{ act: 0, x: a.x, y: a.y }];
    for (var i = 0; i < 4; i++) {
      var nx = a.x + E.DIRS[i][0], ny = a.y + E.DIRS[i][1];
      if (E.passable(s.grid, nx, ny)) out.push({ act: i + 1, x: nx, y: ny });
    }
    return out;
  }

  function pathAhead(field, x, y, k) {
    var cx = x, cy = y;
    for (var s = 0; s < k; s++) {
      var best = field[E.idx(cx, cy)], bx = cx, by = cy;
      for (var i = 0; i < 4; i++) {
        var nx = cx + E.DIRS[i][0], ny = cy + E.DIRS[i][1];
        if (!E.inB(nx, ny)) continue;
        var d = field[E.idx(nx, ny)];
        if (d >= 0 && d < best) { best = d; bx = nx; by = ny; }
      }
      if (bx === cx && by === cy) break;
      cx = bx; cy = by;
    }
    return [cx, cy];
  }

  /* Trap caution is LEARNED, not built in: near zero for the first clumsy episodes,
     full weight only once the policy is competent. And an agent with the chase in
     its teeth tunnel-visions straight into one — which is where the snap moments
     come from once both sides are good. */
  function trapWeight(skill, tunnel) {
    var w = 26 * Math.min(1, Math.max(0, (skill - 0.2) / 0.62));
    return tunnel ? w * 0.06 : w;
  }

  function catAct(s, obs, skill, memo) {
    var g = s.grid, cat = s.cat, mouse = s.mouse, nest = s.map.nest;
    var cands = candidates(s, cat), rnd = s.rand;
    var trapW = trapWeight(skill, obs.targetVisible && obs.targetDist * E.CFG.vision.range <= 2.2);
    var target = null, mode = 'patrol';

    if (obs.targetVisible) {
      // lead the mouse: intercept where its nest-path is heading
      var lead = Math.round(3 * skill);
      target = lead > 0 ? pathAhead(s.map.nestField, mouse.x, mouse.y, lead) : [mouse.x, mouse.y];
      mode = lead > 1 ? 'intercept' : 'chase';
      memo.lastSeen = [mouse.x, mouse.y]; memo.lastSeenAge = 0;
    } else if (obs.scent && skill > 0.25) {
      target = [obs.scent[0] * E.W | 0, obs.scent[1] * E.H | 0]; mode = 'scent';
    } else if (memo.lastSeen && memo.lastSeenAge < 14 && skill > 0.15) {
      target = memo.lastSeen; mode = 'memory';
    } else if (skill > 0.48 && ((memo.campClock = (memo.campClock || 0) + 1) % 54) < 26) {
      target = [nest[0], nest[1]]; mode = 'camp';         // learned: guard the exit
    } else {
      if (!memo.wander || memo.wanderAge > 18 || rnd() < 0.06) {
        memo.wander = [1 + (rnd() * (E.W - 2)) | 0, 1 + (rnd() * (E.H - 2)) | 0];
        memo.wanderAge = 0;
      }
      target = memo.wander; mode = 'patrol';
    }
    memo.lastSeenAge = (memo.lastSeenAge || 0) + 1;
    memo.wanderAge = (memo.wanderAge || 0) + 1;

    var field = E.bfs(g, target[0], target[1]);
    var best = null;
    for (var i = 0; i < cands.length; i++) {
      var c = cands[i], sc = 0;
      var d = field[E.idx(c.x, c.y)];
      sc += d >= 0 ? -d * (0.35 + 0.65 * skill) : -40;
      if (g[E.idx(c.x, c.y)] === E.TRAP) sc -= trapW;
      if (mode === 'camp') {
        var dn = s.map.nestField[E.idx(c.x, c.y)];
        if (dn >= 0 && dn <= 3) sc += 1.5 * skill;        // sit in ambush range
      }
      if (c.act === 0) sc -= mode === 'camp' ? 0.2 : 1.4;
      sc += (rnd() - 0.5) * 17 * Math.pow(1 - skill, 1.5);
      if (!best || sc > best.sc) best = { act: c.act, sc: sc };
    }
    memo.mode = mode;
    return best.act;
  }

  function mouseAct(s, obs, skill, memo) {
    var g = s.grid, cat = s.cat, mouse = s.mouse;
    var cands = candidates(s, mouse), rnd = s.rand;
    var trapW = trapWeight(skill, obs.targetVisible && obs.targetDist * E.CFG.vision.range <= 3);

    // what the mouse believes about the cat: sight beats hearing beats memory
    var belief = null, beliefConf = 0;
    if (obs.targetVisible) { belief = [cat.x, cat.y]; beliefConf = 1; memo.lastCat = belief; memo.lastCatAge = 0; }
    else if (obs.heard) { belief = [obs.heard[0] * E.W, obs.heard[1] * E.H]; beliefConf = obs.heard[2]; }
    else if (memo.lastCat && memo.lastCatAge < 10) { belief = memo.lastCat; beliefConf = 0.4; }
    memo.lastCatAge = (memo.lastCatAge || 0) + 1;

    var threat = belief ? E.bfs(g, Math.round(belief[0]), Math.round(belief[1])) : null;
    var nestField = s.map.nestField;
    // The cat's cone, drawn from the cat's own angle.
    var catCone = obs.targetVisible ? E.castCone(g, cat.x, cat.y, cat.facing, 'cat') : null;

    var here = nestField[E.idx(mouse.x, mouse.y)];
    var commit = here >= 0 && here <= 7 ? 0.2 : 1;   // this close to home, take the risk

    var best = null;
    for (var i = 0; i < cands.length; i++) {
      var c = cands[i], sc = 0;
      var dn = nestField[E.idx(c.x, c.y)];
      sc += dn >= 0 ? -dn * (0.5 + 0.5 * skill) : -40;

      if (threat) {
        var dc = threat[E.idx(c.x, c.y)];
        if (dc >= 0) {
          var danger = Math.max(0, 9 - dc);
          sc -= danger * danger * 0.34 * skill * beliefConf * commit;
          // learned: don't run into the corridor the cat guards
          if (dc <= 2) sc -= 20 * skill * commit;
        }
      }
      if (g[E.idx(c.x, c.y)] === E.TRAP) sc -= trapW;

      // learned: break line of sight
      if (catCone && skill > 0.3) {
        if (!E.lineOfSight(g, cat.x, cat.y, c.x, c.y)) sc += 6 * skill;
      }
      // learned: freeze behind cover instead of bolting into the open
      if (c.act === 0) {
        var hidden = threat && !E.lineOfSight(g, Math.round(belief[0]), Math.round(belief[1]), mouse.x, mouse.y);
        sc += hidden && beliefConf > 0.5 && skill > 0.6 ? 1.5 * skill : -1.4;
      }
      sc += (rnd() - 0.5) * 17 * Math.pow(1 - skill, 1.5);
      if (!best || sc > best.sc) best = { act: c.act, sc: sc };
    }
    memo.mode = obs.targetVisible ? 'evade' : (obs.heard ? 'listen' : 'dash');
    return best.act;
  }

  function makePolicy(algoKey, role) {
    var memo = {};
    return {
      algo: algoKey, role: role, memo: memo,
      reset: function () { memo = this.memo = {}; },
      act: function (s, obs, episode) {
        var skill = skillFor(algoKey, role, episode);
        var a = role === 'cat' ? catAct(s, obs, skill, memo) : mouseAct(s, obs, skill, memo);
        return { action: a, skill: skill, mode: memo.mode };
      }
    };
  }

  var API = { ALGOS: ALGOS, ROLE_BIAS: ROLE_BIAS, skillFor: skillFor, makePolicy: makePolicy };
  global.CatMouseAgents = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
})(typeof window !== 'undefined' ? window : globalThis);
