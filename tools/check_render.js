/* Does the renderer ever draw something inside a wall?
 *
 *   node tools/check_render.js [journal.jsonl]
 *
 * The simulation is provably clean — nobody ever occupies a wall cell. But the screen
 * does not show simulation state, it shows an interpolation of it, and that is where
 * both wall bugs lived:
 *
 *   1. the bodies were tweened between the last two frames, and on the first frame of a
 *      new episode those are the old death position and the new spawn, 30-odd cells
 *      apart — so the pair slid across the room straight through the blocks;
 *   2. the vision cone was ray-cast against the walls at the destination cell and then
 *      TRANSLATED to the tweened body position, which took an exact shape off its
 *      anchor and spilled light through cover. It is now cast at the tweened position.
 *
 * This replays a recorded journal through the real painter and fails if either returns.
 * It samples each tween at several points, because the worst offence is at the middle.
 */
'use strict';
global.window = global;
require('../app/js/env.js');
require('../app/js/paint.js');

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const E = global.CatMouseEnv;
const P = global.Paint;

const ROOT = path.join(__dirname, '..');
const file = process.argv[2] ||
  execSync('ls -S runs/journals/*.jsonl 2>/dev/null | head -1', { cwd: ROOT }).toString().trim();
if (!file) {
  console.error('no journal to check — record one first, or pass a path');
  process.exit(2);
}

const ALPHAS = [0.25, 0.5, 0.75, 1];
const CS = 44;

/* Two coordinate spaces, and mixing them silently invents failures.
 *  - a BODY position is a cell index, possibly fractional mid-tween: cell = round(v)
 *  - a CONE vertex is in centre-space (the cast starts at cell + 0.5): cell = floor(v),
 *    which is exactly what castCone itself does when it walks a ray. */
function wallAtCell(grid, x, y) {
  const cx = Math.round(x), cy = Math.round(y);
  return E.inB(cx, cy) && grid[E.idx(cx, cy)] === E.WALL;
}

function wallAtPoint(grid, x, y) {
  const cx = Math.floor(x), cy = Math.floor(y);
  return E.inB(cx, cy) && grid[E.idx(cx, cy)] === E.WALL;
}

/* How far past a wall face a ray reaches, in cells.
 *
 * A binary "does it touch a wall" test is the wrong question: the caster walks in 0.18
 * steps and backs off one, so an exact cone still grazes wall faces by a few hundredths
 * of a cell — measured, 0.034 at worst. What matters is whether the light visibly
 * covers a block, so the gate measures depth and allows a graze. For reference, the
 * translate bug reached 0.5 cells and was obvious on screen. */
const GRAZE = 0.12;      // cells; about 5px at the arena's 44px cell

function rayDepthIntoWall(grid, ax, ay, bx, by) {
  let worst = 0;
  const n = 48;
  for (let i = 1; i < n; i++) {
    const t = i / n;
    const x = ax + (bx - ax) * t, y = ay + (by - ay) * t;
    if (!wallAtPoint(grid, x, y)) continue;
    const fx = x - Math.floor(x), fy = y - Math.floor(y);
    worst = Math.max(worst, Math.min(Math.min(fx, 1 - fx), Math.min(fy, 1 - fy)));
  }
  return worst;
}

let frames = 0, bodyBad = 0, coneBad = 0, pairs = 0, coneWorst = 0;
let local = null, prev = null;
const worst = [];

for (const line of fs.readFileSync(path.resolve(ROOT, file), 'utf8').split('\n')) {
  if (!line.trim()) continue;
  let m;
  try { m = JSON.parse(line); } catch (e) { continue; }
  if (m.type !== 'frame') continue;
  frames++;
  if (m.map) local = E.genMap(m.map.seed >>> 0, (m.map.nests || []).length || 1);
  if (!local) { prev = m; continue; }
  if (prev) {
    pairs++;
    const joined = P.continuous(prev, m);
    for (const a of ALPHAS) {
      const svg = P.fxSvg({ frame: m, prev: prev, alpha: a, cs: CS, map: local, key: 'chk', now: 0 });

      // bodies: the same rule the painter uses, so the test cannot drift from it
      const lp = (p0, p1) => (joined ? p0 + (p1 - p0) * a : p1);
      for (const who of ['cat', 'mouse']) {
        const x = lp(prev[who].x, m[who].x), y = lp(prev[who].y, m[who].y);
        if (wallAtCell(local.grid, x, y)) {
          bodyBad++;
          if (worst.length < 5) worst.push(`body ${who} at ${x.toFixed(2)},${y.toFixed(2)} step ${prev.step}->${m.step}`);
        }
      }

      // cones: every ray must reach its endpoint without crossing a wall
      // Read the capture group rather than slicing the match — an off-by-one there
      // silently shifts every apex a tenth of a cell and the gate reports 97k phantom
      // failures with apexes sitting inside the border wall.
      for (const mm of svg.matchAll(/<polygon points="([^"]+)"/g)) {
        const v = mm[1].trim().split(' ')
          .map(q => q.split(',').map(Number).map(n => n / CS));
        if (v.length < 3) continue;
        const [ax, ay] = v[0];
        let deepest = 0;
        for (let i = 1; i < v.length; i++) {
          deepest = Math.max(deepest, rayDepthIntoWall(local.grid, ax, ay, v[i][0], v[i][1]));
        }
        coneWorst = Math.max(coneWorst, deepest);
        if (deepest > GRAZE) {
          coneBad++;
          if (worst.length < 5) {
            worst.push(`cone ${deepest.toFixed(2)} cells into a wall, apex ${ax.toFixed(2)},${ay.toFixed(2)}, step ${prev.step}->${m.step}`);
          }
        }
      }
    }
  }
  prev = m;
}

console.log(`journal              : ${file}`);
console.log(`frames / tween pairs : ${frames.toLocaleString()} / ${pairs.toLocaleString()}`);
console.log(`bodies drawn in wall : ${bodyBad}`);
console.log(`cones past the ${GRAZE} graze : ${coneBad}   (deepest ${coneWorst.toFixed(3)} cells)`);
if (bodyBad || coneBad) {
  console.log('\nFAIL — the renderer is drawing through solid geometry:');
  worst.forEach(w => console.log('  ' + w));
  process.exit(1);
}
console.log('\nPASS — nothing is ever drawn inside or through a wall.');
