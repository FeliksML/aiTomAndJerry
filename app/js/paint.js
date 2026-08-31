/* The visual grammar. Ported from the Claude Design artboard, unchanged in substance.
 *
 * The rules that make the screen readable, and that must survive any edit:
 *
 *   Algorithm identity  = emblem + one accent object on the character. The cat wears
 *                         the accent as a collar and inner ears, the mouse as a scarf.
 *   Role identity       = species and cone colour, NEVER the fur. The cat is always
 *                         slate and large, the mouse always tan and small; cat cones
 *                         are warm, mouse cones cool. Predator and prey therefore read
 *                         instantly, and swapping algorithms never changes who is who.
 *   Neutral hazards     = gold, and the two gold objects never share a silhouette.
 *                         Trap: a toothed ring around an amber plate, snapping shut to
 *                         a vertical lens. Hole: an arch. Circle vs lens vs arch — three
 *                         unmistakable shapes at 22px.
 *   Second skin         = both characters also exist as sprite sheets built from the
 *                         source renders; `fxSvg({sprites: true})` swaps the vector pair
 *                         for them. The identity rule is why spriteSvg exists at all — a
 *                         painted cat cannot wear an accent collar, so the accent moves
 *                         to a ring on the floor beneath it, and the alert tick and frost
 *                         ring are drawn over the sprite exactly as before.
 *   State without text  = alert is wide eyes, small pupils and a tick above the head;
 *                         frozen is X eyes plus a dashed frost ring, with the trap drawn
 *                         shut for as long as the hold lasts. Facing is pupil offset and
 *                         a horizontal flip — the characters never rotate.
 */
(function (global) {
  'use strict';
  var E = global.CatMouseEnv;

  /* How big the two painted props are drawn, in map cells, and where in the cell the
     thing they stand on sits. Sizing is by the OBJECT, not by its canvas: the prop sheet
     carries each one's own height as a share of its frame, so an open trap — half again
     as wide as it is tall — and a tall arch can share one atlas without either being
     drawn to the wrong scale. The numbers themselves are set against the vector pair they
     replace, which is the only comparison that matters: a hazard has to read at the size
     the player already learned it at. */
  var HOLE_CELLS = 1.05, HOLE_FOOT = 0.30;
  var TRAP_CELLS = 0.72, TRAP_FOOT = 0.20;

  function rgba(hex, a) {
    var h = (hex || '#8fb6e6').replace('#', '');
    return 'rgba(' + parseInt(h.slice(0, 2), 16) + ',' + parseInt(h.slice(2, 4), 16) + ','
      + parseInt(h.slice(4, 6), 16) + ',' + a + ')';
  }

  /* ---------- the arena ---------- */

  function mapSvg(map, CS, opts) {
    var W = E.W, H = E.H, g = map.grid, w = W * CS, h = H * CS, p = [];
    var f = function (v) { return (+v).toFixed(1); };
    p.push('<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '" style="display:block">');
    p.push('<rect width="' + w + '" height="' + h + '" fill="#0b0f16"/>');
    var tile = '', x, y;
    for (y = 1; y < H - 1; y++) for (x = 1; x < W - 1; x++) {
      if (g[E.idx(x, y)] === E.WALL) continue;
      if ((x + y) % 2) continue;
      tile += 'M' + (x * CS) + ' ' + (y * CS) + 'h' + CS + 'v' + CS + 'h-' + CS + 'Z';
    }
    p.push('<path d="' + tile + '" fill="rgba(150,185,235,.022)"/>');
    var dots = '';
    for (y = 1; y < H - 1; y++) for (x = 1; x < W - 1; x++) {
      if (g[E.idx(x, y)] === E.WALL) continue;
      dots += 'M' + f(x * CS + CS / 2) + ' ' + f(y * CS + CS / 2) + 'h0.5';
    }
    p.push('<path d="' + dots + '" stroke="rgba(130,170,225,.16)" stroke-width="' + f(CS / 14) + '" stroke-linecap="round"/>');

    var b = CS;
    p.push('<g fill="#111a2a">');
    p.push('<rect x="0" y="0" width="' + w + '" height="' + b + '"/>');
    p.push('<rect x="0" y="' + (h - b) + '" width="' + w + '" height="' + b + '"/>');
    p.push('<rect x="0" y="' + b + '" width="' + b + '" height="' + (h - 2 * b) + '"/>');
    p.push('<rect x="' + (w - b) + '" y="' + b + '" width="' + b + '" height="' + (h - 2 * b) + '"/>');
    p.push('</g>');
    p.push('<rect x="' + (b - 0.5) + '" y="' + (b - 0.5) + '" width="' + (w - 2 * b + 1) + '" height="' + (h - 2 * b + 1) + '" fill="none" stroke="rgba(135,175,230,.2)" stroke-width="1.2"/>');

    var r = Math.max(3, CS * 0.24);
    (map.blocks || []).forEach(function (bl) {
      var bx = bl.x * CS + 1, by = bl.y * CS + 1, bw = bl.w * CS - 2, bh = bl.h * CS - 2;
      p.push('<rect x="' + f(bx) + '" y="' + f(by) + '" width="' + f(bw) + '" height="' + f(bh) + '" rx="' + f(r) + '" fill="#18243a" stroke="rgba(138,180,235,.26)" stroke-width="1.2"/>');
      p.push('<rect x="' + f(bx + CS * 0.22) + '" y="' + f(by + CS * 0.22) + '" width="' + f(bw - CS * 0.44) + '" height="' + f(bh - CS * 0.44) + '" rx="' + f(r * 0.6) + '" fill="none" stroke="rgba(120,160,215,.1)" stroke-width="1"/>');
    });
    // A pillar blocks its whole cell, and the vision cone is cast against that whole
    // cell — so drawing a 0.68-cell disc left the light stopping in what looked like open
    // floor beside it, and made every corridor read wider than it is. The cell footprint
    // is drawn first, at the same weight as the cover blocks, with the disc on top.
    (map.pillars || []).forEach(function (pp) {
      var px = pp[0] * CS + 1, py = pp[1] * CS + 1, ps = CS - 2;
      var cx = (pp[0] + 0.5) * CS, cy = (pp[1] + 0.5) * CS;
      p.push('<rect x="' + f(px) + '" y="' + f(py) + '" width="' + f(ps) + '" height="' + f(ps)
        + '" rx="' + f(Math.max(3, CS * 0.24)) + '" fill="#18243a" stroke="rgba(138,180,235,.26)" stroke-width="1.2"/>');
      p.push('<circle cx="' + f(cx) + '" cy="' + f(cy) + '" r="' + f(CS * 0.34) + '" fill="#1b2740" stroke="rgba(138,180,235,.28)" stroke-width="1.1"/>');
      p.push('<circle cx="' + f(cx) + '" cy="' + f(cy) + '" r="' + f(CS * 0.17) + '" fill="none" stroke="rgba(175,215,255,.16)" stroke-width="1"/>');
    });

    // Every hole, not just the first. With two of them the room is a different game —
    // the mouse chooses and the cat cannot cover both — so both have to be on screen.
    var holes = map.nests || [map.nest];
    // With the painted hole in play this layer keeps only the warm pool on the floor and
    // hands the arch itself to fxSvg. The map layer is cached until the map changes, and
    // the arch cannot live there any more: at the end of an escape it has to be replaced,
    // for nine tenths of a second, by the four frames of a mouse diving into it.
    var painted = !!(opts && opts.sprites) && CS >= 20
      && global.PropSprite && global.PropSprite.ready() && global.PropSprite.has('hole');
    for (var hi = 0; hi < holes.length; hi++) {
      var n = holes[hi], nx = (n[0] + 0.5) * CS, ny = (n[1] + 0.5) * CS;
      var hw = CS * 0.62, hh = CS * 0.72;
      p.push('<ellipse cx="' + f(nx) + '" cy="' + f(ny) + '" rx="' + f(CS * 0.95) + '" ry="' + f(CS * 0.85) + '" fill="rgba(255,209,102,' + (painted ? '.10' : '.06') + ')"/>');
      if (painted) continue;
      p.push('<path d="M' + f(nx - hw / 2) + ' ' + f(ny + hh / 2) + 'L' + f(nx - hw / 2) + ' ' + f(ny - hh * 0.05) + 'A' + f(hw / 2) + ' ' + f(hw / 2) + ' 0 0 1 ' + f(nx + hw / 2) + ' ' + f(ny - hh * 0.05) + 'L' + f(nx + hw / 2) + ' ' + f(ny + hh / 2) + 'Z" fill="#05080c" stroke="#ffd166" stroke-width="' + f(Math.max(1.6, CS * 0.085)) + '"/>');
      p.push('<path d="M' + f(nx - hw * 0.28) + ' ' + f(ny + hh / 2) + 'L' + f(nx - hw * 0.28) + ' ' + f(ny + hh * 0.08) + 'A' + f(hw * 0.28) + ' ' + f(hw * 0.28) + ' 0 0 1 ' + f(nx + hw * 0.28) + ' ' + f(ny + hh * 0.08) + 'L' + f(nx + hw * 0.28) + ' ' + f(ny + hh / 2) + 'Z" fill="rgba(255,209,102,.16)"/>');
    }

    if (CS < 20) {
      (map.traps || []).forEach(function (t) {
        var cx = (t[0] + 0.5) * CS, cy = (t[1] + 0.5) * CS;
        p.push('<circle cx="' + f(cx) + '" cy="' + f(cy) + '" r="' + f(CS * 0.3) + '" fill="none" stroke="#b9cade" stroke-width="' + f(CS * 0.09) + '"/>');
        p.push('<circle cx="' + f(cx) + '" cy="' + f(cy) + '" r="' + f(CS * 0.13) + '" fill="#f2b544"/>');
      });
      if (map.catSpawn) {
        p.push('<circle cx="' + f((map.catSpawn[0] + 0.5) * CS) + '" cy="' + f((map.catSpawn[1] + 0.5) * CS) + '" r="' + f(CS * 0.34) + '" fill="rgba(255,138,92,.85)"/>');
        p.push('<circle cx="' + f((map.mouseSpawn[0] + 0.5) * CS) + '" cy="' + f((map.mouseSpawn[1] + 0.5) * CS) + '" r="' + f(CS * 0.26) + '" fill="rgba(110,226,255,.85)"/>');
      }
    }
    p.push('</svg>');
    return p.join('');
  }

  /* ---------- hazards ---------- */

  function trapSvg(cx, cy, S, snapped, frozen) {
    var n = function (v) { return (S * v).toFixed(2); };
    var steel = snapped ? '#f2f9ff' : '#b9cade', dark = '#0f151d';
    var o = ['<g transform="translate(' + cx.toFixed(1) + ',' + cy.toFixed(1) + ')">'];
    o.push('<circle cx="0" cy="0" r="' + n(0.64) + '" fill="rgba(242,181,68,' + (snapped ? 0.22 : 0.09) + ')"/>');
    o.push('<ellipse cx="0" cy="' + n(0.4) + '" rx="' + n(0.42) + '" ry="' + n(0.1) + '" fill="rgba(0,0,0,.42)"/>');
    o.push('<path d="M' + n(0.36) + ' ' + n(0.26) + 'L' + n(0.6) + ' ' + n(0.42) + '" stroke="#6b7d92" stroke-width="' + n(0.05) + '" stroke-linecap="round"/>');
    o.push('<circle cx="' + n(0.62) + '" cy="' + n(0.44) + '" r="' + n(0.07) + '" fill="none" stroke="#6b7d92" stroke-width="' + n(0.04) + '"/>');
    o.push('<circle cx="' + n(-0.42) + '" cy="0" r="' + n(0.1) + '" fill="' + dark + '" stroke="' + steel + '" stroke-width="' + n(0.045) + '"/>');
    o.push('<circle cx="' + n(0.42) + '" cy="0" r="' + n(0.1) + '" fill="' + dark + '" stroke="' + steel + '" stroke-width="' + n(0.045) + '"/>');
    if (!snapped) {
      o.push('<circle cx="0" cy="0" r="' + n(0.34) + '" fill="rgba(12,18,26,.55)"/>');
      o.push('<path d="M0 ' + n(-0.38) + 'A' + n(0.38) + ' ' + n(0.38) + ' 0 0 0 0 ' + n(0.38) + '" fill="none" stroke="' + steel + '" stroke-width="' + n(0.1) + '" stroke-linecap="round"/>');
      o.push('<path d="M0 ' + n(-0.38) + 'A' + n(0.38) + ' ' + n(0.38) + ' 0 0 1 0 ' + n(0.38) + '" fill="none" stroke="' + steel + '" stroke-width="' + n(0.1) + '" stroke-linecap="round"/>');
      var teeth = '', pairs = [[122, 238], [-58, 58]];
      for (var i = 0; i < pairs.length; i++) {
        for (var k = 0; k < 4; k++) {
          var ang = (pairs[i][0] + (pairs[i][1] - pairs[i][0]) * (k + 0.5) / 4) * Math.PI / 180;
          teeth += 'M' + n(Math.cos(ang) * 0.32) + ' ' + n(Math.sin(ang) * 0.32) +
            'L' + n(Math.cos(ang) * 0.19) + ' ' + n(Math.sin(ang) * 0.19);
        }
      }
      o.push('<path d="' + teeth + '" stroke="' + steel + '" stroke-width="' + n(0.045) + '" stroke-linecap="round" opacity=".9"/>');
      o.push('<circle cx="0" cy="0" r="' + n(0.15) + '" fill="#f2b544" stroke="' + dark + '" stroke-width="' + n(0.035) + '"/>');
    } else {
      var t = Math.max(0, Math.min(1, frozen / 5));
      o.push('<path d="M' + n(-0.42) + ' 0H' + n(-0.06) + 'M' + n(0.42) + ' 0H' + n(0.06) + '" stroke="' + steel + '" stroke-width="' + n(0.075) + '" stroke-linecap="round"/>');
      o.push('<path d="M0 ' + n(-0.4) + 'Q' + n(-0.15) + ' 0 0 ' + n(0.4) + '" fill="none" stroke="' + steel + '" stroke-width="' + n(0.11) + '" stroke-linecap="round"/>');
      o.push('<path d="M0 ' + n(-0.4) + 'Q' + n(0.15) + ' 0 0 ' + n(0.4) + '" fill="none" stroke="' + steel + '" stroke-width="' + n(0.11) + '" stroke-linecap="round"/>');
      o.push('<path d="M' + n(-0.1) + ' ' + n(-0.2) + 'h' + n(0.2) + 'M' + n(-0.12) + ' 0h' + n(0.24) + 'M' + n(-0.1) + ' ' + n(0.2) + 'h' + n(0.2) + '" stroke="' + dark + '" stroke-width="' + n(0.04) + '" stroke-linecap="round"/>');
      o.push('<path d="M' + n(-0.56) + ' ' + n(-0.34) + 'l' + n(-0.16) + ' ' + n(-0.12) + 'M' + n(0.56) + ' ' + n(-0.34) + 'l' + n(0.16) + ' ' + n(-0.12) + 'M0 ' + n(-0.52) + 'v' + n(-0.18) + '" stroke="rgba(255,240,200,' + (0.85 * t).toFixed(2) + ')" stroke-width="' + n(0.055) + '" stroke-linecap="round"/>');
      o.push('<circle cx="0" cy="0" r="' + n(0.46 + 0.36 * (1 - t)) + '" fill="none" stroke="rgba(255,236,190,' + (0.6 * t).toFixed(2) + ')" stroke-width="' + n(0.055) + '"/>');
    }
    o.push('</g>');
    return o.join('');
  }

  /* ---------- characters ---------- */

  function eyesSvg(S, cfg) {
    var n = function (v) { return (S * v).toFixed(2); };
    var o = [], sides = [-1, 1];
    for (var i = 0; i < sides.length; i++) {
      var ex = sides[i] * cfg.dx;
      if (cfg.frozen) {
        var q = cfg.rx * 0.85;
        o.push('<path d="M' + n(ex - q) + ' ' + n(-q) + 'L' + n(ex + q) + ' ' + n(q) + 'M' + n(ex + q) + ' ' + n(-q) + 'L' + n(ex - q) + ' ' + n(q) + '" stroke="#1b232e" stroke-width="' + n(0.032) + '" stroke-linecap="round"/>');
        continue;
      }
      if (cfg.dark) {
        o.push('<ellipse cx="' + n(ex) + '" cy="0" rx="' + n(cfg.rx) + '" ry="' + n(cfg.ry) + '" fill="#141c26"/>');
        o.push('<circle cx="' + n(ex - cfg.rx * 0.3) + '" cy="' + n(-cfg.ry * 0.34) + '" r="' + n(cfg.rx * 0.34) + '" fill="rgba(255,255,255,.9)"/>');
      } else {
        o.push('<ellipse cx="' + n(ex) + '" cy="0" rx="' + n(cfg.rx) + '" ry="' + n(cfg.alert ? cfg.ry * 1.12 : cfg.ry) + '" fill="#fdfefe"/>');
        o.push('<circle cx="' + n(ex + cfg.ox) + '" cy="' + n(cfg.oy) + '" r="' + n(cfg.alert ? cfg.pr * 0.78 : cfg.pr) + '" fill="#141c26"/>');
        o.push('<circle cx="' + n(ex + cfg.ox - cfg.pr * 0.34) + '" cy="' + n(cfg.oy - cfg.pr * 0.4) + '" r="' + n(cfg.pr * 0.3) + '" fill="rgba(255,255,255,.85)"/>');
      }
    }
    return o.join('');
  }

  function frostSvg(S) {
    var n = function (v) { return (S * v).toFixed(2); };
    return '<circle cx="0" cy="' + n(-0.1) + '" r="' + n(0.62) + '" fill="none" stroke="rgba(168,232,255,.75)" stroke-width="' + n(0.05) + '" stroke-dasharray="' + n(0.14) + ' ' + n(0.1) + '"/>' +
      '<path d="M' + n(-0.7) + ' ' + n(-0.62) + 'l' + n(0.1) + ' ' + n(0.1) + 'M' + n(-0.6) + ' ' + n(-0.62) + 'l' + n(-0.1) + ' ' + n(0.1) + 'M' + n(0.62) + ' ' + n(0.2) + 'l' + n(0.1) + ' ' + n(0.1) + 'M' + n(0.72) + ' ' + n(0.2) + 'l' + n(-0.1) + ' ' + n(0.1) + '" stroke="rgba(200,244,255,.7)" stroke-width="' + n(0.035) + '" stroke-linecap="round"/>';
  }

  function catSvg(cx, cy, S, accent, st) {
    var n = function (v) { return (S * v).toFixed(2); };
    var f = st.facing, flip = f === 3 ? -1 : 1, back = f === 0;
    var fur = '#7e90ad', dk = '#55647e', lt = '#9fb1cb', cream = '#f2e8d5';
    accent = accent || '#8fb6e6';
    var o = [];
    o.push('<g transform="translate(' + ((cx + 0.5) * S).toFixed(1) + ',' + ((cy + 0.5) * S).toFixed(1) + ')">');
    o.push('<ellipse cx="0" cy="' + n(0.44) + '" rx="' + n(0.38) + '" ry="' + n(0.13) + '" fill="rgba(0,0,0,.5)"/>');
    o.push('<g transform="scale(' + flip + ',1)">');
    o.push('<path d="M' + n(-0.22) + ' ' + n(0.3) + 'C' + n(-0.62) + ' ' + n(0.34) + ' ' + n(-0.72) + ' ' + n(-0.06) + ' ' + n(-0.5) + ' ' + n(-0.16) + '" fill="none" stroke="' + dk + '" stroke-width="' + n(0.13) + '" stroke-linecap="round"/>');
    o.push('<path d="M' + n(-0.56) + ' ' + n(-0.14) + 'C' + n(-0.66) + ' ' + n(-0.16) + ' ' + n(-0.66) + ' ' + n(-0.02) + ' ' + n(-0.56) + ' ' + n(-0.02) + '" fill="none" stroke="' + cream + '" stroke-width="' + n(0.11) + '" stroke-linecap="round"/>');
    o.push('<path d="M' + n(-0.3) + ' ' + n(0.42) + 'C' + n(-0.44) + ' ' + n(0.12) + ' ' + n(-0.36) + ' ' + n(-0.14) + ' 0 ' + n(-0.18) + 'C' + n(0.36) + ' ' + n(-0.14) + ' ' + n(0.44) + ' ' + n(0.12) + ' ' + n(0.3) + ' ' + n(0.42) + 'Z" fill="' + fur + '" stroke="' + dk + '" stroke-width="' + n(0.035) + '"/>');
    o.push('<ellipse cx="0" cy="' + n(0.22) + '" rx="' + n(0.19) + '" ry="' + n(0.19) + '" fill="' + cream + '" opacity="' + (back ? 0 : 0.95) + '"/>');
    o.push('<ellipse cx="' + n(-0.17) + '" cy="' + n(0.42) + '" rx="' + n(0.1) + '" ry="' + n(0.07) + '" fill="' + lt + '"/>');
    o.push('<ellipse cx="' + n(0.17) + '" cy="' + n(0.42) + '" rx="' + n(0.1) + '" ry="' + n(0.07) + '" fill="' + lt + '"/>');
    o.push('<path d="M' + n(-0.2) + ' ' + n(-0.12) + 'Q0 ' + n(0.0) + ' ' + n(0.2) + ' ' + n(-0.12) + '" fill="none" stroke="' + accent + '" stroke-width="' + n(0.085) + '" stroke-linecap="round"/>');
    o.push('<circle cx="0" cy="' + n(-0.03) + '" r="' + n(0.05) + '" fill="' + accent + '" stroke="rgba(255,255,255,.5)" stroke-width="' + n(0.018) + '"/>');
    o.push('<g transform="translate(0,' + n(-0.42) + ')">');
    o.push('<path d="M' + n(-0.3) + ' ' + n(-0.12) + 'L' + n(-0.42) + ' ' + n(-0.52) + 'L' + n(-0.06) + ' ' + n(-0.3) + 'Z" fill="' + fur + '" stroke="' + dk + '" stroke-width="' + n(0.03) + '"/>');
    o.push('<path d="M' + n(0.3) + ' ' + n(-0.12) + 'L' + n(0.42) + ' ' + n(-0.52) + 'L' + n(0.06) + ' ' + n(-0.3) + 'Z" fill="' + fur + '" stroke="' + dk + '" stroke-width="' + n(0.03) + '"/>');
    o.push('<path d="M' + n(-0.27) + ' ' + n(-0.18) + 'L' + n(-0.35) + ' ' + n(-0.44) + 'L' + n(-0.13) + ' ' + n(-0.29) + 'Z" fill="' + accent + '" opacity=".75"/>');
    o.push('<path d="M' + n(0.27) + ' ' + n(-0.18) + 'L' + n(0.35) + ' ' + n(-0.44) + 'L' + n(0.13) + ' ' + n(-0.29) + 'Z" fill="' + accent + '" opacity=".75"/>');
    o.push('<ellipse cx="0" cy="0" rx="' + n(0.36) + '" ry="' + n(0.32) + '" fill="' + fur + '" stroke="' + dk + '" stroke-width="' + n(0.035) + '"/>');
    o.push('<ellipse cx="' + n(-0.24) + '" cy="' + n(0.02) + '" rx="' + n(0.11) + '" ry="' + n(0.09) + '" fill="' + lt + '" opacity=".5"/>');
    o.push('<ellipse cx="' + n(0.24) + '" cy="' + n(0.02) + '" rx="' + n(0.11) + '" ry="' + n(0.09) + '" fill="' + lt + '" opacity=".5"/>');
    if (!back) {
      o.push('<ellipse cx="0" cy="' + n(0.13) + '" rx="' + n(0.19) + '" ry="' + n(0.13) + '" fill="' + cream + '"/>');
      o.push('<g transform="translate(0,' + n(-0.05) + ')">' + eyesSvg(S, {
        dx: 0.14, rx: 0.115, ry: 0.13, pr: 0.055,
        ox: (E.DIRS[f] || [0, 1])[0] * 0.035, oy: (E.DIRS[f] || [0, 1])[1] * 0.03,
        dark: false, frozen: st.frozen > 0, alert: st.sees
      }) + '</g>');
      var brow = st.sees ? 0.055 : 0.02;
      o.push('<path d="M' + n(-0.23) + ' ' + n(-0.21 + brow) + 'L' + n(-0.08) + ' ' + n(-0.25) + 'M' + n(0.23) + ' ' + n(-0.21 + brow) + 'L' + n(0.08) + ' ' + n(-0.25) + '" stroke="' + dk + '" stroke-width="' + n(0.035) + '" stroke-linecap="round"/>');
      o.push('<path d="M0 ' + n(0.07) + 'l' + n(-0.05) + ' ' + n(0.05) + 'l' + n(0.05) + ' ' + n(0.04) + 'l' + n(0.05) + ' ' + n(-0.04) + 'Z" fill="#f2a3ad"/>');
      o.push('<path d="M0 ' + n(0.16) + 'q' + n(-0.06) + ' ' + n(0.05) + ' ' + n(-0.1) + ' 0M0 ' + n(0.16) + 'q' + n(0.06) + ' ' + n(0.05) + ' ' + n(0.1) + ' 0" fill="none" stroke="' + dk + '" stroke-width="' + n(0.028) + '" stroke-linecap="round"/>');
      o.push('<path d="M' + n(-0.19) + ' ' + n(0.1) + 'H' + n(-0.42) + 'M' + n(-0.19) + ' ' + n(0.16) + 'l' + n(-0.22) + ' ' + n(0.06) + 'M' + n(0.19) + ' ' + n(0.1) + 'H' + n(0.42) + 'M' + n(0.19) + ' ' + n(0.16) + 'l' + n(0.22) + ' ' + n(0.06) + '" stroke="rgba(240,248,255,.55)" stroke-width="' + n(0.022) + '" stroke-linecap="round"/>');
    } else {
      o.push('<path d="M' + n(-0.14) + ' ' + n(0.02) + 'q' + n(0.14) + ' ' + n(0.1) + ' ' + n(0.28) + ' 0" fill="none" stroke="' + dk + '" stroke-width="' + n(0.03) + '" stroke-linecap="round" opacity=".5"/>');
    }
    o.push('</g></g>');
    if (st.frozen > 0) o.push(frostSvg(S));
    if (st.sees) o.push('<g transform="translate(0,' + n(-1.02) + ')"><path d="M0 0v' + n(0.16) + '" stroke="#ff8a5c" stroke-width="' + n(0.1) + '" stroke-linecap="round"/><circle cx="0" cy="' + n(0.29) + '" r="' + n(0.055) + '" fill="#ff8a5c"/></g>');
    o.push('</g>');
    return o.join('');
  }

  function mouseSvg(cx, cy, S, accent, st) {
    var n = function (v) { return (S * v).toFixed(2); };
    var f = st.facing, flip = f === 3 ? -1 : 1, back = f === 0;
    var fur = '#d09b6a', dk = '#a16c3a', lt = '#e8c39a', cream = '#f8ecd9', pink = '#f2a3ad';
    accent = accent || '#8fb6e6';
    var o = [];
    o.push('<g transform="translate(' + ((cx + 0.5) * S).toFixed(1) + ',' + ((cy + 0.5) * S).toFixed(1) + ')">');
    o.push('<ellipse cx="0" cy="' + n(0.34) + '" rx="' + n(0.26) + '" ry="' + n(0.09) + '" fill="rgba(0,0,0,.45)"/>');
    o.push('<g transform="scale(' + flip + ',1)">');
    o.push('<path d="M' + n(-0.16) + ' ' + n(0.24) + 'C' + n(-0.5) + ' ' + n(0.3) + ' ' + n(-0.58) + ' ' + n(-0.04) + ' ' + n(-0.38) + ' ' + n(-0.12) + '" fill="none" stroke="' + pink + '" stroke-width="' + n(0.06) + '" stroke-linecap="round" opacity=".85"/>');
    o.push('<path d="M' + n(-0.21) + ' ' + n(0.32) + 'C' + n(-0.31) + ' ' + n(0.08) + ' ' + n(-0.25) + ' ' + n(-0.1) + ' 0 ' + n(-0.13) + 'C' + n(0.25) + ' ' + n(-0.1) + ' ' + n(0.31) + ' ' + n(0.08) + ' ' + n(0.21) + ' ' + n(0.32) + 'Z" fill="' + fur + '" stroke="' + dk + '" stroke-width="' + n(0.03) + '"/>');
    o.push('<ellipse cx="0" cy="' + n(0.17) + '" rx="' + n(0.13) + '" ry="' + n(0.14) + '" fill="' + cream + '" opacity="' + (back ? 0 : 0.95) + '"/>');
    o.push('<ellipse cx="' + n(-0.12) + '" cy="' + n(0.33) + '" rx="' + n(0.075) + '" ry="' + n(0.05) + '" fill="' + pink + '"/>');
    o.push('<ellipse cx="' + n(0.12) + '" cy="' + n(0.33) + '" rx="' + n(0.075) + '" ry="' + n(0.05) + '" fill="' + pink + '"/>');
    o.push('<path d="M' + n(-0.15) + ' ' + n(-0.08) + 'Q0 ' + n(0.02) + ' ' + n(0.15) + ' ' + n(-0.08) + '" fill="none" stroke="' + accent + '" stroke-width="' + n(0.075) + '" stroke-linecap="round"/>');
    o.push('<path d="M' + n(0.13) + ' ' + n(-0.05) + 'l' + n(0.12) + ' ' + n(0.12) + 'l' + n(-0.07) + ' ' + n(0.03) + 'Z" fill="' + accent + '"/>');
    o.push('<g transform="translate(0,' + n(-0.3) + ')">');
    o.push('<circle cx="' + n(-0.24) + '" cy="' + n(-0.14) + '" r="' + n(0.17) + '" fill="' + fur + '" stroke="' + dk + '" stroke-width="' + n(0.028) + '"/>');
    o.push('<circle cx="' + n(0.24) + '" cy="' + n(-0.14) + '" r="' + n(0.17) + '" fill="' + fur + '" stroke="' + dk + '" stroke-width="' + n(0.028) + '"/>');
    o.push('<circle cx="' + n(-0.24) + '" cy="' + n(-0.14) + '" r="' + n(0.1) + '" fill="' + pink + '" opacity=".8"/>');
    o.push('<circle cx="' + n(0.24) + '" cy="' + n(-0.14) + '" r="' + n(0.1) + '" fill="' + pink + '" opacity=".8"/>');
    o.push('<ellipse cx="0" cy="0" rx="' + n(0.26) + '" ry="' + n(0.24) + '" fill="' + fur + '" stroke="' + dk + '" stroke-width="' + n(0.03) + '"/>');
    o.push('<ellipse cx="' + n(-0.17) + '" cy="' + n(0.04) + '" rx="' + n(0.08) + '" ry="' + n(0.065) + '" fill="' + lt + '" opacity=".55"/>');
    o.push('<ellipse cx="' + n(0.17) + '" cy="' + n(0.04) + '" rx="' + n(0.08) + '" ry="' + n(0.065) + '" fill="' + lt + '" opacity=".55"/>');
    if (!back) {
      o.push('<ellipse cx="0" cy="' + n(0.11) + '" rx="' + n(0.145) + '" ry="' + n(0.105) + '" fill="' + cream + '"/>');
      o.push('<g transform="translate(0,' + n(-0.04) + ')">' + eyesSvg(S, {
        dx: 0.105, rx: 0.078, ry: 0.092, pr: 0.04,
        ox: (E.DIRS[f] || [0, 1])[0] * 0.02, oy: (E.DIRS[f] || [0, 1])[1] * 0.02,
        dark: true, frozen: st.frozen > 0, alert: st.sees
      }) + '</g>');
      o.push('<circle cx="0" cy="' + n(0.13) + '" r="' + n(0.042) + '" fill="' + pink + '"/>');
      o.push('<path d="M' + n(-0.15) + ' ' + n(0.09) + 'H' + n(-0.33) + 'M' + n(-0.15) + ' ' + n(0.15) + 'l' + n(-0.17) + ' ' + n(0.05) + 'M' + n(0.15) + ' ' + n(0.09) + 'H' + n(0.33) + 'M' + n(0.15) + ' ' + n(0.15) + 'l' + n(0.17) + ' ' + n(0.05) + '" stroke="rgba(255,250,242,.6)" stroke-width="' + n(0.02) + '" stroke-linecap="round"/>');
    }
    o.push('</g></g>');
    if (st.frozen > 0) o.push(frostSvg(S));
    if (st.sees) o.push('<g transform="translate(0,' + n(-0.86) + ')"><path d="M0 0v' + n(0.15) + '" stroke="#6ee2ff" stroke-width="' + n(0.09) + '" stroke-linecap="round"/><circle cx="0" cy="' + n(0.27) + '" r="' + n(0.05) + '" fill="#6ee2ff"/></g>');
    o.push('</g>');
    return o.join('');
  }

  /* Either character, drawn from its sheet instead of from paths.
     Placed by its ground anchor so the feet sit on the cell the game thinks it is in,
     rather than by a corner of the frame — a sprite is much taller than one cell and
     centring it would float the character above its own position. */
  function spriteSvg(sheet, cx, cy, S, accent, st, opts) {
    if (!sheet || !sheet.ready()) return null;
    var n = function (v) { return (S * v).toFixed(2); };
    // `cells` is how tall the CHARACTER should be, in map cells — not how tall the frame
    // is. The trapped sheet is padded more generously than the walk sheet, so sizing by
    // the frame would shrink the character the moment it stepped in a trap.
    var cells = (opts && opts.cells) || 1.38;
    var size = S * cells / (sheet.meta().charHeight || 0.8);
    var foot = opts && opts.footOffset !== undefined ? opts.footOffset : 0.34;
    var ax = (cx + 0.5) * S, ay = (cy + 0.5 + foot) * S;
    var dir = sheet.fromFacing(st.facing);
    var frame = opts && opts.frame !== null && opts.frame !== undefined ? opts.frame
              : st.frozen > 0 ? sheet.idleFrame(dir)
              : sheet.frameAt(dir, !!(opts && opts.moving), (opts && opts.now) || 0,
                              opts && opts.fps);
    // `bare` is for the frames that are not a character standing on the floor. The escape
    // sheet draws an ARCH, and a body shadow and an accent ring under an arch read as a
    // second object on the tile rather than as the mouse's marker.
    var bare = !!(opts && opts.bare);
    var o = ['<g>'];
    if (!bare) {
      o.push('<ellipse cx="' + ax.toFixed(1) + '" cy="' + ay.toFixed(1) + '" rx="' + n(0.4)
        + '" ry="' + n(0.14) + '" fill="rgba(0,0,0,.5)"/>');
      o.push('<ellipse cx="' + ax.toFixed(1) + '" cy="' + ay.toFixed(1) + '" rx="' + n(0.46)
        + '" ry="' + n(0.17) + '" fill="none" stroke="' + rgba(accent, st.sees ? 0.85 : 0.5)
        + '" stroke-width="' + n(0.055) + '"/>');
    }
    o.push(sheet.svgAt(dir, frame, ax, ay, size));
    o.push('<g transform="translate(' + ((cx + 0.5) * S).toFixed(1) + ','
      + ((cy + 0.5) * S).toFixed(1) + ')">');
    // The frost ring is the vector skin's way of saying "held". The trapped sprite says it
    // far more plainly — the character is visibly struggling in a trap — so drawing both
    // would be saying it twice.
    if (st.frozen > 0 && !(opts && opts.held) && !bare) o.push(frostSvg(S));
    if (st.sees && !bare) o.push('<g transform="translate(0,' + n(-1.35) + ')"><path d="M0 0v' + n(0.16)
      + '" stroke="#ff8a5c" stroke-width="' + n(0.1) + '" stroke-linecap="round"/><circle cx="0" cy="'
      + n(0.29) + '" r="' + n(0.055) + '" fill="#ff8a5c"/></g>');
    o.push('</g></g>');
    return o.join('');
  }

  /* ---------- the live layer over the arena ---------- */

  /* Both agents move at most one cell per step, so anything larger between two frames
     is not motion — it is a new episode, and the pair have respawned somewhere else.
     Interpolating across that slides them diagonally across the room and straight
     through the walls in between, which is exactly what "they sometimes walk into
     walls" turned out to be. Measured on one recorded session: 198 such frame pairs,
     jumping 28 to 43 cells each.

     The check lives here rather than in the caller because this is the only place that
     interpolates; anything that forgets to pass a sane `prev` is still safe. */
  function continuous(prev, cur) {
    if (!prev || !cur) return false;
    return Math.abs(prev.cat.x - cur.cat.x) <= 1 && Math.abs(prev.cat.y - cur.cat.y) <= 1
      && Math.abs(prev.mouse.x - cur.mouse.x) <= 1 && Math.abs(prev.mouse.y - cur.mouse.y) <= 1;
  }

  function fxSvg(opts) {
    var v = opts.frame;
    var pv = continuous(opts.prev, v) ? opts.prev : v;
    var a = continuous(opts.prev, v) ? opts.alpha : 1;
    var CS = opts.cs;
    var map = opts.map, key = opts.key || 'x', now = opts.now || 0;
    if (!v || !map) return '';
    var showV = opts.showVision !== false, showH = opts.showHearing !== false,
      showS = opts.showScent !== false;
    var w = E.W * CS, h = E.H * CS, p = [];
    var f = function (x) { return (+x).toFixed(1); };
    var lp = function (x0, x1) { return x0 + (x1 - x0) * a; };
    var cx = lp(pv.cat.x, v.cat.x), cy = lp(pv.cat.y, v.cat.y);
    var mx = lp(pv.mouse.x, v.mouse.x), my = lp(pv.mouse.y, v.mouse.y);
    var range = (E.CFG.vision.range + 0.6) * CS;
    p.push('<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '" style="display:block">');

    if (showV) {
      var grad = function (id, col, cxp, cyp, boost) {
        return '<radialGradient id="' + id + '" gradientUnits="userSpaceOnUse" cx="' + f(cxp) + '" cy="' + f(cyp) + '" r="' + f(range) + '">' +
          '<stop offset="0" stop-color="' + col + '" stop-opacity="' + (0.34 * boost).toFixed(2) + '"/>' +
          '<stop offset="0.5" stop-color="' + col + '" stop-opacity="' + (0.15 * boost).toFixed(2) + '"/>' +
          '<stop offset="1" stop-color="' + col + '" stop-opacity="0"/></radialGradient>';
      };
      p.push('<defs>' +
        grad('acm-' + key, '#6ee2ff', (mx + 0.5) * CS, (my + 0.5) * CS, v.mouse.sees ? 1.7 : 1) +
        grad('ack-' + key, '#ff7a54', (cx + 0.5) * CS, (cy + 0.5) * CS, v.cat.sees ? 1.9 : 1) +
        '</defs>');
    }

    var busy = {};
    [v.cat, v.mouse].forEach(function (ag) {
      if (ag && ag.frozen > 0) busy[Math.round(ag.x) + ',' + Math.round(ag.y)] = ag.frozen;
    });
    var W = global.WalkSprite;
    var spriteTrap = {};
    [['tom', v.cat], ['jerry', v.mouse]].forEach(function (e) {
      var ch = opts.sprites && W ? W[e[0]] : null;
      if (e[1].frozen > 0 && ch && ch.trapped && ch.trapped.ready()) {
        spriteTrap[Math.round(e[1].x) + ',' + Math.round(e[1].y)] = 1;
      }
    });
    /* The painted props, and the one cell they must stand out of the way for.
       Both hazards moved out of the cached map layer and into this one: the trap has
       states, and the hole is animated over at the end of an escape, and neither can be
       said in a layer that is only redrawn when the map itself changes. */
    var PR = global.PropSprite;
    var props = opts.sprites && PR && PR.ready() && CS >= 20 ? PR : null;
    var ending = opts.ending && (opts.ending.result === 'catch' || opts.ending.result === 'escape')
      ? opts.ending : null;
    var endSheet = null, endWho = null, endFrame = 0;
    if (opts.sprites && W && ending) {
      var cand = ending.result === 'catch' ? W.tom['catch'] : W.jerry.escape;
      if (cand && cand.ready()) {
        endSheet = cand;
        endWho = ending.result === 'catch' ? 'cat' : 'mouse';
        endFrame = cand.frameForElapsed(ending.ms || 0, opts.spriteFps);
      }
    }
    if (props) {
      var holes2 = map.nests || [map.nest];
      var holeSize = PR.sizeFor('hole', HOLE_CELLS, CS);
      for (var hj = 0; hj < holes2.length; hj++) {
        var hc = holes2[hj];
        // The hole she went down draws its own arch, four frames of it, so the standing
        // one underneath would be a second arch a few pixels off the first.
        if (endWho === 'mouse' && Math.round(v.mouse.x) === hc[0] && Math.round(v.mouse.y) === hc[1]) continue;
        p.push(PR.svgAt('hole', (hc[0] + 0.5) * CS, (hc[1] + 0.5 + HOLE_FOOT) * CS, holeSize));
      }
    }
    (map.traps || []).forEach(function (t) {
      // A trapped sprite is drawn holding its own trap, so the one underneath it would be
      // a second trap in the same cell. Every other trap on the map still draws.
      if (spriteTrap[t[0] + ',' + t[1]]) return;
      var fr = busy[t[0] + ',' + t[1]] || 0;
      if (props) {
        // The gold pool the vector trap carried, kept. It is not decoration: the painted
        // trap is dark wood and mid-grey steel on a near-black floor, and without the
        // pool underneath it reads as a smudge rather than as the hazard the whole map is
        // built around. Gold is also the arena's word for "neutral danger", and dropping
        // it would have left the hole speaking a language the trap no longer spoke.
        p.push('<ellipse cx="' + f((t[0] + 0.5) * CS) + '" cy="' + f((t[1] + 0.62) * CS)
          + '" rx="' + f(CS * 0.6) + '" ry="' + f(CS * 0.36) + '" fill="rgba(242,181,68,'
          + (fr > 0 ? '.24' : '.13') + ')"/>');
        p.push(PR.svgAt(fr > 0 ? 'trapShut' : 'trapSet', (t[0] + 0.5) * CS,
                        (t[1] + 0.5 + TRAP_FOOT) * CS, PR.sizeFor('trap', TRAP_CELLS, CS)));
        return;
      }
      p.push(trapSvg((t[0] + 0.5) * CS, (t[1] + 0.5) * CS, CS, fr > 0, fr));
    });

    if (showS && v.scent && v.scent.length) {
      var g = '<g>';
      for (var i = 0; i < v.scent.length; i++) {
        var c = v.scent[i], rr = CS * (0.07 + 0.14 * c[2]);
        g += '<circle cx="' + f((c[0] + 0.5) * CS) + '" cy="' + f((c[1] + 0.5) * CS) + '" r="' + f(rr) + '" fill="rgba(126,232,255,' + (0.04 + 0.19 * c[2]).toFixed(3) + ')"/>';
      }
      p.push(g + '</g>');
    }

    var poly = function (pts) {
      return pts.map(function (q) { return f(q[0] * CS) + ',' + f(q[1] * CS); }).join(' ');
    };
    /* Cast the cone AT the tweened position rather than moving a finished one there.
     *
     * Two wrong ways were tried first. Translating the destination cone to the tweened
     * body takes a shape that was ray-cast against the walls at one cell and hangs it
     * off another, so the light spills straight through cover. Blending the two frames'
     * cones vertex by vertex is better but still bulges up to half a cell into a wall,
     * because a blend of two star-shaped polygons about two different centres is not
     * star-shaped about the centre in between.
     *
     * env.js is right here in the browser and castCone takes fractional coordinates, so
     * the honest answer is simply to cast it properly. Measured penetration drops from
     * 0.498 cells to 0 — the caster walks cell boundaries and stops on the wall face,
     * which is where a ray is supposed to stop. About seventy rays a cone at 0.011 ms,
     * two cones a frame, against a 16.7 ms budget.
     */
    var coneAt = function (px, py, st, fallback) {
      if (!map.grid || !E.castCone) return fallback;
      return E.castCone(map.grid, px, py, st.facing).poly;
    };
    if (showV) {
      var mCone = coneAt(mx, my, v.mouse, v.mouse.cone);
      var cCone = coneAt(cx, cy, v.cat, v.cat.cone);
      if (mCone) p.push('<polygon points="' + poly(mCone) + '" fill="url(#acm-' + key + ')" stroke="rgba(150,235,255,' + (v.mouse.sees ? 0.5 : 0.22) + ')" stroke-width="1"/>');
      if (cCone) p.push('<polygon points="' + poly(cCone) + '" fill="url(#ack-' + key + ')" stroke="rgba(255,155,115,' + (v.cat.sees ? 0.66 : 0.28) + ')" stroke-width="1"/>');
    }

    if (showH && v.mouse.heard) {
      var hd = v.mouse.heard, hx = (hd.x + 0.5) * CS, hy = (hd.y + 0.5) * CS;
      var R = Math.max(CS * 0.9, (hd.radius || 2) * CS * 0.7);
      for (var k = 0; k < 3; k++) {
        var ph = ((now / 950) + k / 3) % 1;
        p.push('<circle cx="' + f(hx) + '" cy="' + f(hy) + '" r="' + f(R * ph) + '" fill="none" stroke="rgba(196,240,255,' + (0.36 * (1 - ph) * hd.conf).toFixed(3) + ')" stroke-width="1.4"/>');
      }
      p.push('<circle cx="' + f(hx) + '" cy="' + f(hy) + '" r="' + f(R) + '" fill="none" stroke="rgba(196,240,255,' + (0.2 * hd.conf).toFixed(3) + ')" stroke-width="1" stroke-dasharray="' + f(CS * 0.2) + ' ' + f(CS * 0.22) + '"/>');
      p.push('<line x1="' + f((mx + 0.5) * CS) + '" y1="' + f((my + 0.5) * CS) + '" x2="' + f(hx) + '" y2="' + f(hy) + '" stroke="rgba(196,240,255,' + (0.15 * hd.conf).toFixed(3) + ')" stroke-width="1" stroke-dasharray="2 4"/>');
      var q2 = CS * 0.13;
      p.push('<path d="M' + f(hx - q2) + ' ' + f(hy - q2) + 'L' + f(hx + q2) + ' ' + f(hy + q2) + 'M' + f(hx + q2) + ' ' + f(hy - q2) + 'L' + f(hx - q2) + ' ' + f(hy + q2) + '" stroke="rgba(196,240,255,' + (0.55 * hd.conf).toFixed(3) + ')" stroke-width="1.5" stroke-linecap="round"/>');
    }

    /* Which sheet a character is drawn from is decided by the game state, not by a timer:
       a held agent uses its trapped sheet, and its frame comes from the freeze countdown,
       so the snap lands on the very step the jaw closed and cannot drift out of sync. */
    var skin = function (who, fallback, x, y, accent, ag, moving, cells, force) {
      var ch = opts.sprites && W ? W[who] : null;
      if (!ch) return fallback(x, y, CS, accent, ag);
      var held = ag.frozen > 0 && ch.trapped && ch.trapped.ready();
      var sheet = force ? force.sheet : (held ? ch.trapped : ch.walk);
      if (!sheet || !sheet.ready()) return fallback(x, y, CS, accent, ag);
      var s = spriteSvg(sheet, x, y, CS, accent, ag, {
        now: now, moving: moving, fps: opts.spriteFps, held: held,
        cells: force ? force.cells : cells,
        bare: !!(force && force.bare),
        footOffset: force && force.footOffset !== undefined ? force.footOffset : undefined,
        frame: force ? force.frame : (held ? sheet.frameForHold(ag.frozen, opts.holdSteps || 5) : null)
      });
      return s === null ? fallback(x, y, CS, accent, ag) : s;
    };
    // 1.38 and 0.95 cells are the vector pair's own measured heights, so swapping skins
    // changes the drawing and nothing else — the cat stays exactly as large as the cat.
    var catCells = opts.catCells || 1.38;
    if (endWho === 'cat') {
      // One drawing, two characters. Jerry is in the cat's fist, so drawing him from his
      // own sheet as well would put a second mouse on the floor beside himself.
      p.push(skin('tom', catSvg, cx, cy, opts.catAccent, v.cat, false, catCells,
                  { sheet: endSheet, frame: endFrame, cells: catCells }));
    } else if (endWho === 'mouse') {
      // She is inside the hole; the arch is hers for the length of the hold. The cat is
      // still drawn — the shot is him arriving a moment too late.
      p.push(skin('jerry', mouseSvg, mx, my, opts.mouseAccent, v.mouse, false, HOLE_CELLS,
                  { sheet: endSheet, frame: endFrame, cells: HOLE_CELLS,
                    bare: true, footOffset: HOLE_FOOT }));
      p.push(skin('tom', catSvg, cx, cy, opts.catAccent, v.cat, opts.catMoving, catCells));
    } else {
      p.push(skin('jerry', mouseSvg, mx, my, opts.mouseAccent, v.mouse,
                  opts.mouseMoving, opts.mouseCells || 0.95));
      p.push(skin('tom', catSvg, cx, cy, opts.catAccent, v.cat, opts.catMoving, catCells));
    }
    p.push('</svg>');
    return p.join('');
  }

  /* ---------- emblems ---------- */

  function emblem(algo, stroke) {
    var s = stroke || '#8fb6e6';
    var head = '<svg viewBox="0 0 40 40" width="100%" height="100%" style="display:block;overflow:visible">';
    var body = {
      ppo: '<g fill="none" stroke="' + s + '" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">' +
        '<circle cx="20" cy="20" r="15.2" opacity="0.85"></circle>' +
        '<path d="M9.4 26.6 L16.6 19.6 L22.4 23.2 L30.8 12.4" stroke-width="2.8"></path>' +
        '<path d="M26.8 12.4 L30.8 12.4 L30.8 16.4"></path>' +
        '<path d="M7.2 12.6 L14.6 12.6" opacity="0.5"></path>' +
        '<path d="M25.4 30.2 L32.8 30.2" opacity="0.5"></path></g>',
      ga: '<g fill="none" stroke="' + s + '" stroke-width="2.4" stroke-linecap="round">' +
        '<path d="M11 5 C 28 13, 12 19, 29 27 C 22 32, 16 33, 11 35" opacity="0.9"></path>' +
        '<path d="M29 5 C 12 13, 28 19, 11 27 C 18 32, 24 33, 29 35" opacity="0.9"></path>' +
        '<path d="M13.6 10.2 L26.4 10.2" stroke-width="2" opacity="0.75"></path>' +
        '<path d="M14.8 19 L25.2 19" stroke-width="2" opacity="0.75"></path>' +
        '<path d="M13.6 27.8 L26.4 27.8" stroke-width="2" opacity="0.75"></path></g>',
      cmaes: '<g fill="none" stroke="' + s + '" stroke-width="2.4" stroke-linecap="round">' +
        '<ellipse cx="20" cy="20" rx="16.4" ry="8.6" transform="rotate(-32 20 20)" opacity="0.9"></ellipse>' +
        '<ellipse cx="20" cy="20" rx="8.6" ry="4.4" transform="rotate(-32 20 20)" stroke-width="1.9" opacity="0.6"></ellipse>' +
        '<circle cx="20" cy="20" r="2.1" fill="' + s + '" stroke="none"></circle>' +
        '<circle cx="30.4" cy="12.6" r="1.5" fill="' + s + '" stroke="none" opacity="0.7"></circle>' +
        '<circle cx="10.4" cy="26.4" r="1.5" fill="' + s + '" stroke="none" opacity="0.7"></circle>' +
        '<circle cx="25.6" cy="26.8" r="1.3" fill="' + s + '" stroke="none" opacity="0.45"></circle></g>',
      // The redacted stand-in: a sealed disc. Deliberately a shape of its own, so a
      // hidden school never looks like a broken one.
      sealed: '<g fill="none" stroke="' + s + '" stroke-width="2.4" stroke-linecap="round">' +
        '<circle cx="20" cy="20" r="15.2" opacity="0.55"></circle>' +
        '<path d="M13 20h14" opacity="0.8"></path>' +
        '<path d="M20 13v14" opacity="0.35" stroke-dasharray="2 3"></path></g>'
    }[algo] || '';
    return head + body + '</svg>';
  }

  /* The card portraits on the leaderboard and the grand final. They draw the same sprite
     the arena draws, standing still and facing the camera, so a single frame does not put
     two different Toms on screen at once — the rendered one in the arena and a cartoon
     vector on the card beside it.

     Cropped by the sheet's own metadata rather than by hand: `charHeight` is how much of
     the frame the character fills and `anchor` is where its feet are, so the crop follows
     the art if the sheets are ever rebuilt. The vector pair stays as the fallback for the
     moment before the sheets have loaded, and for a build without them. */
  function portrait(role, accent, size) {
    var S = size || 130, st = { facing: 2, frozen: 0, sees: false };
    var WS = global.WalkSprite;
    var ch = WS && (role === 'cat' ? WS.tom : WS.jerry);
    var sh = ch && ch.walk;
    if (sh && sh.ready()) {
      var m = sh.meta() || {};
      var a = m.anchor || { x: 0.5, y: 0.9 };
      var F = 100 / (m.charHeight || 0.8);       // frame size that makes the body 100 tall
      var fx2 = F * a.x, fy2 = F * a.y;          // the feet, inside that frame
      var pad = 7, vbW = 100, vbH = 100 + pad * 2;
      var n = function (v) { return (+v).toFixed(2); };
      return '<svg width="100%" height="100%" viewBox="' + n(fx2 - vbW / 2) + ' '
        + n(fy2 - 100 - pad) + ' ' + n(vbW) + ' ' + n(vbH) + '" style="display:block">'
        // The same ground shadow and accent ring the arena puts under them, so the card
        // reads as the school's colour rather than as a sticker.
        + '<ellipse cx="' + n(fx2) + '" cy="' + n(fy2) + '" rx="25" ry="7" fill="rgba(0,0,0,.45)"/>'
        + '<ellipse cx="' + n(fx2) + '" cy="' + n(fy2) + '" rx="29" ry="9" fill="none" stroke="'
        + rgba(accent, 0.6) + '" stroke-width="2"/>'
        + sh.svgAt('down', sh.idleFrame('down'), fx2, fy2, F) + '</svg>';
    }
    if (role === 'cat') {
      return '<svg width="100%" height="100%" viewBox="' + (-0.34 * S) + ' ' + (-0.56 * S) + ' '
        + (1.68 * S) + ' ' + (1.72 * S) + '" style="display:block">' + catSvg(0, 0, S, accent, st) + '</svg>';
    }
    return '<svg width="100%" height="100%" viewBox="' + (-0.16 * S) + ' ' + (-0.34 * S) + ' '
      + (1.32 * S) + ' ' + (1.35 * S) + '" style="display:block">' + mouseSvg(0, 0, S, accent, st) + '</svg>';
  }

  global.Paint = {
    rgba: rgba, mapSvg: mapSvg, fxSvg: fxSvg, trapSvg: trapSvg, continuous: continuous,
    catSvg: catSvg, mouseSvg: mouseSvg, spriteSvg: spriteSvg,
    emblem: emblem, portrait: portrait
  };
})(window);
