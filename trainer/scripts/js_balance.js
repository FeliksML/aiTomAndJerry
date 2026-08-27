/* The same balance sweep, run through the ORIGINAL JS controller.
 *
 * If Python and JS report the same outcome mix, the Python port is faithful and any
 * problem is in the game's rules. If they disagree, the port is wrong. Either way
 * this is the check that tells them apart.
 *
 *   node trainer/scripts/js_balance.js [episodesPerCell]
 */
const path = require('path');
const root = path.join(__dirname, '..', '..', 'viz');
const E = require(path.join(root, 'env.js'));
global.CatMouseEnv = E;
const A = require(path.join(root, 'agents.js'));

const SEEDS = Array.from({ length: 12 }, (_, i) => 20260826 + i * 911);
const N = parseInt(process.argv[2] || '480', 10);

// agents.js exposes competence only through an episode counter, so find the episode
// that lands closest to each target skill — exactly what the Academy's epFor does.
function epFor(key, role, target) {
  let best = 0, bd = 9;
  for (let e = 0; e <= 400; e++) {
    const d = Math.abs(A.skillFor(key, role, e) - target);
    if (d < bd) { bd = d; best = e; }
  }
  return best;
}

const maps = SEEDS.map(s => E.genMap(s));

function play(catSkill, mouseSkill, n, salt) {
  const ec = epFor('ppo', 'cat', catSkill), em = epFor('ppo', 'mouse', mouseSkill);
  let catch_ = 0, escape = 0, draw = 0, steps = 0, traps = 0;
  for (let i = 0; i < n; i++) {
    const m = maps[i % maps.length];
    const pol = { cat: A.makePolicy('ppo', 'cat'), mouse: A.makePolicy('ppo', 'mouse') };
    const s = E.reset(m, (i * 7919 + salt) >>> 0);
    let g = 0;
    while (!s.done && g++ < 400) {
      const ev = E.step(s, {
        cat: pol.cat.act(s, E.observe(s, 'cat'), ec).action,
        mouse: pol.mouse.act(s, E.observe(s, 'mouse'), em).action
      });
      if (ev.trapCat || ev.trapMouse) traps++;
    }
    if (s.result === 'catch') catch_++; else if (s.result === 'escape') escape++; else draw++;
    steps += s.step;
  }
  return { catch: catch_ / n, escape: escape / n, draw: draw / n, steps: steps / n, traps: traps / n };
}

const pc = v => (v * 100).toFixed(1).padStart(6) + '%';
console.log('\nJS scripted vs scripted, matched skill');
console.log(' skill   catch  escape    draw   steps  traps/ep');
for (const sk of [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]) {
  const r = play(sk, sk, N, (sk * 1000) | 0);
  console.log(`${sk.toFixed(2).padStart(6)} ${pc(r.catch)} ${pc(r.escape)} ${pc(r.draw)} ` +
    `${r.steps.toFixed(1).padStart(7)} ${r.traps.toFixed(2).padStart(9)}` +
    (r.draw > 0.20 ? '  <-- draws too high' : ''));
}
