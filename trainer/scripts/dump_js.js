/* Reference dump from the JS environment, for the Python parity check.
 * Emits one JSON object per seed: the generated map plus a fully deterministic
 * rollout driven by a shared PRNG, so Python can reproduce both halves exactly.
 *
 *   node trainer/scripts/dump_js.js <firstSeed> <count> [nests] > runs/parity-js.json
 */
const path = require('path');
const E = require(path.join(__dirname, '..', '..', 'viz', 'env.js'));

const first = parseInt(process.argv[2] || '1', 10);
const count = parseInt(process.argv[3] || '200', 10);
const nests = parseInt(process.argv[4] || '1', 10);

function hashGrid(g) {
  // FNV-1a over the cell codes — a short, order-sensitive fingerprint.
  let h = 0x811c9dc5;
  for (let i = 0; i < g.length; i++) { h ^= g[i]; h = Math.imul(h, 0x01000193); }
  return (h >>> 0).toString(16);
}

const out = [];
for (let k = 0; k < count; k++) {
  const seed = (first + k) >>> 0;
  const m = E.genMap(seed, nests);

  // Deterministic action stream, identical on both sides.
  const ar = E.rng((seed ^ 0xa5a5a5) >>> 0);
  const s = E.reset(m, seed);
  const traj = [];
  let obsSample = null;
  let g = 0;
  while (!s.done && g++ < 400) {
    if (s.step === 3) {
      const oc = E.observe(s, 'cat'), om = E.observe(s, 'mouse');
      obsSample = {
        catRays: oc.rays.map(v => +v.toFixed(9)),
        mouseRays: om.rays.map(v => +v.toFixed(9)),
        catScent: oc.scent, mouseHeard: om.heard,
        catVis: oc.targetVisible, mouseVis: om.targetVisible,
        catNestDist: oc.nestDist, mouseNestDist: om.nestDist
      };
    }
    const ca = Math.floor(ar() * 5), ma = Math.floor(ar() * 5);
    E.step(s, { cat: ca, mouse: ma });
    traj.push([s.cat.x, s.cat.y, s.cat.facing, s.cat.frozen,
               s.mouse.x, s.mouse.y, s.mouse.facing, s.mouse.frozen]);
  }

  out.push({
    seed,
    gridHash: hashGrid(m.grid),
    nests: m.nests, catSpawn: m.catSpawn, mouseSpawn: m.mouseSpawn,
    traps: m.traps, optimal: m.optimal, routeLen: m.route.length,
    blocks: m.blocks.length, pillars: m.pillars.length,
    steps: s.step, result: s.result,
    catReward: +s.cat.reward.toFixed(9), mouseReward: +s.mouse.reward.toFixed(9),
    traj, obsSample
  });
}
process.stdout.write(JSON.stringify({ nests, maps: out }));
