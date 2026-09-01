/* How each algorithm works, drawn from its own training telemetry.
 *
 * Nothing here is a mock-up. Every number arrives from the running optimiser:
 *
 *   PPO      the five action probabilities on a fixed probe batch, and the importance
 *            ratio histogram against the clip band. The bars nudge; the histogram
 *            piles up inside the band. That IS the clipped objective — the update is
 *            not allowed to move far, and you can watch it not moving far.
 *   GA       forty-eight genomes as DNA strips, ranked by fitness, elites lit. Lines
 *            join the two parents of a child. The strips are a fixed random projection
 *            of the weights, so a child visibly resembles its parents instead of
 *            looking like fresh noise.
 *   CMA-ES   the sampled brains projected onto the two principal directions of the
 *            real sample cloud, with the empirical covariance ellipse and the step the
 *            mean took. The ellipse stretches along whatever keeps working and
 *            contracts as sigma falls.
 *
 * Values ease toward their targets rather than snapping, because telemetry arrives a
 * couple of times a second and a snapping chart reads as broken on video.
 */
(function (global) {
  'use strict';

  var ACTIONS = ['STAY', 'N', 'E', 'S', 'W'];

  function ease(cur, target, k) { return cur + (target - cur) * k; }

  function easeArr(cur, target, k) {
    if (!cur || cur.length !== target.length) return target.slice();
    return target.map(function (v, i) { return ease(cur[i], v, k); });
  }

  function fmt(v, d) { return (v === undefined || v === null) ? '—' : (+v).toFixed(d === undefined ? 3 : d); }

  /* Panels fade on the wall clock rather than per animation frame: a per-frame decay is a
     different duration on a 60Hz and a 120Hz panel, and the recording is neither. */
  function now() {
    return (typeof performance !== 'undefined' && performance.now) ? performance.now() : 0;
  }

  /* Both roles train at once and both stream telemetry; each panel draws one of them.
     Which one is not a detail the viewer can infer from the picture, so it is written on
     it — on its own row, because right-aligning it against a column header is how it
     ended up printed on top of one. `role` is set by the app when it forwards telemetry. */
  function roleTag(role, x, y) {
    var r = String(role || 'cat').toUpperCase();
    return '<text x="' + x + '" y="' + y + '" fill="#5b6b85" font-size="9.5" letter-spacing="1.3"'
      + ' font-family="JetBrains Mono,monospace">TRAINING THE ' + r
      + ' — the ' + (r === 'CAT' ? 'mouse' : 'cat') + ' is learning in the same run</text>';
  }

  /* A column heading and the one line that says what the picture underneath actually is.
     Two rows, always: a subtitle squeezed onto the title's row is a collision waiting for
     a longer word. */
  function heading(x, y, title, sub) {
    return '<text x="' + x + '" y="' + y + '" fill="#8fa4c4" font-size="11" letter-spacing="1.4">'
      + title + '</text>'
      + '<text x="' + x + '" y="' + (y + 14) + '" fill="#61708a" font-size="9.5">' + sub + '</text>';
  }

  /* Every panel draws this until its optimiser has actually said something. An empty
     box on camera invites "is it broken?"; an invented number is worse. */
  /* Two different silences. A school that is not training has nothing to say and says so.
     A school that IS training has not said anything YET — the first generation has to
     finish before there is a number — and telling that author to "press t" three seconds
     after they pressed t is the one moment this box produces the "is it broken?" it exists
     to prevent. The true sentence already lived one card away, in the dim block nobody
     frames the shot on. */
  var STARTING = {
    ppo: 'building the first batch — every environment has to fill before the first update lands',
    ga: 'playing generation 1 — every brain in the crowd has to finish before there is anything to draw',
    cmaes: 'playing generation 1 — the whole sample has to be scored before the cloud has a shape'
  };

  function waiting(w, h, opt) {
    var live = opt && opt.starting;
    var line = live
      ? (STARTING[opt.school] || 'the optimiser is starting — the first numbers are seconds away')
      : 'no telemetry yet — press t to train this school on camera';
    var bar = live
      ? '<rect x="' + (w / 2 - 90) + '" y="' + (h / 2 + 14) + '" width="180" height="3" rx="1.5"'
        + ' fill="rgba(130,160,200,.14)"/>'
        + '<rect x="' + (w / 2 - 90) + '" y="' + (h / 2 + 14) + '" width="54" height="3" rx="1.5" fill="#4e6a94"/>'
      : '';
    return '<svg viewBox="0 0 ' + w + ' ' + h + '" width="100%" height="100%" style="display:block">'
      + '<text x="' + (w / 2) + '" y="' + (h / 2) + '" fill="#3d4a60" font-size="13" text-anchor="middle"'
      + ' font-family="JetBrains Mono,monospace">' + line + '</text>' + bar + '</svg>';
  }

  /* ---------------- PPO ---------------- */

  function PpoPanel() {
    this.probs = null;
    this.hist = null;
    this.stats = {};
    this.range = [0.45, 1.55];
    this.band = [0.8, 1.2];
    this.trail = [];
  }

  PpoPanel.prototype.update = function (t) {
    if (!t) return;
    this.target = t.actionProbs || this.target;
    this.histTarget = t.ratioHist || this.histTarget;
    if (t.ratioRange) this.range = t.ratioRange;
    if (t.clipBand) this.band = t.clipBand;
    this.role = t.role || this.role;
    this.stats = {
      clipped: t.clippedFrac, kl: t.approxKl, entropy: t.entropy,
      value: t.valueLoss, ev: t.explainedVar, year: t.year, ladder: t.ladderWin
    };
    if (t.entropy !== undefined) {
      this.trail.push(t.entropy);
      if (this.trail.length > 90) this.trail.shift();
    }
  };

  PpoPanel.prototype.draw = function (w, h, accent, opt) {
    // Nothing until the optimiser has spoken, the same as GaPanel and CmaPanel. A
    // fallback of [0.2 x 5] would draw a *measurement* — five bars reading "20", a
    // perfectly uniform policy — out of the absence of one, and the clip band would be
    // drawn at the JS constant rather than at the trainer's epsilon.
    if (!this.target) return waiting(w, h, opt);
    this.probs = easeArr(this.probs, this.target, 0.18);
    this.hist = easeArr(this.hist, this.histTarget || new Array(32).fill(0), 0.22);
    var p = [], f = function (v) { return (+v).toFixed(1); };
    p.push('<svg viewBox="0 0 ' + w + ' ' + h + '" width="100%" height="100%" style="display:block">');

    // Row 0 is the role, on its own. Rows 1-2 are each column's title and its subtitle.
    // Everything below starts under all of them, so no two strings share a baseline.
    p.push(roleTag(this.role, 16, 14));
    var bx = 16, by = 62, bw = w * 0.36, bh = h - 128;   // leaves the footer its own two rows
    p.push(heading(bx, 36, 'WHAT IT WOULD DO',
                   'the five moves, scored on one fixed batch of situations'));
    var slot = bw / ACTIONS.length;
    for (var i = 0; i < ACTIONS.length; i++) {
      var v = Math.max(0, Math.min(1, this.probs[i] || 0));
      var barH = Math.max(2, v * bh);
      var x = bx + i * slot + slot * 0.16, ww = slot * 0.68;
      p.push('<rect x="' + f(x) + '" y="' + f(by) + '" width="' + f(ww) + '" height="' + f(bh) + '" rx="3" fill="rgba(255,255,255,.035)"/>');
      p.push('<rect x="' + f(x) + '" y="' + f(by + bh - barH) + '" width="' + f(ww) + '" height="' + f(barH) + '" rx="3" fill="' + accent + '" opacity="' + (0.45 + 0.5 * v).toFixed(2) + '"/>');
      p.push('<text x="' + f(x + ww / 2) + '" y="' + f(by + bh + 15) + '" fill="#7d90ad" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">' + ACTIONS[i] + '</text>');
      p.push('<text x="' + f(x + ww / 2) + '" y="' + f(by + bh - barH - 5) + '" fill="#c9d8ee" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">' + Math.round(v * 100) + '</text>');
    }

    // Ratio histogram against the clip band
    var hx = bx + bw + 34, hw2 = w - hx - 16, hy = 62, hh = bh * 0.60;
    p.push(heading(hx, 36, 'HOW FAR THE UPDATE TRIED TO MOVE',
                   'one bar per sampled move · the shaded band is where it is allowed to go'));
    var lo = this.range[0], hi = this.range[1];
    var sx = function (r) { return hx + (r - lo) / (hi - lo) * hw2; };
    var b0 = sx(this.band[0]), b1 = sx(this.band[1]);
    p.push('<rect x="' + f(b0) + '" y="' + f(hy) + '" width="' + f(b1 - b0) + '" height="' + f(hh) + '" fill="' + accent + '" opacity=".10"/>');
    p.push('<line x1="' + f(b0) + '" y1="' + f(hy) + '" x2="' + f(b0) + '" y2="' + f(hy + hh) + '" stroke="' + accent + '" stroke-width="1" stroke-dasharray="3 3" opacity=".7"/>');
    p.push('<line x1="' + f(b1) + '" y1="' + f(hy) + '" x2="' + f(b1) + '" y2="' + f(hy + hh) + '" stroke="' + accent + '" stroke-width="1" stroke-dasharray="3 3" opacity=".7"/>');
    p.push('<line x1="' + f(sx(1)) + '" y1="' + f(hy - 4) + '" x2="' + f(sx(1)) + '" y2="' + f(hy + hh) + '" stroke="rgba(230,240,255,.45)" stroke-width="1"/>');
    var maxH = Math.max.apply(null, this.hist.concat([1]));
    var cw = hw2 / this.hist.length;
    for (var k = 0; k < this.hist.length; k++) {
      var vh = (this.hist[k] / maxH) * (hh - 6);
      var cx = hx + k * cw;
      var r = lo + (k + 0.5) / this.hist.length * (hi - lo);
      var inside = r >= this.band[0] && r <= this.band[1];
      p.push('<rect x="' + f(cx + 0.5) + '" y="' + f(hy + hh - vh) + '" width="' + f(Math.max(1, cw - 1)) + '" height="' + f(vh) + '" fill="' + (inside ? accent : '#5b6b85') + '" opacity="' + (inside ? 0.85 : 0.5) + '"/>');
    }
    p.push('<text x="' + f(sx(1)) + '" y="' + f(hy + hh + 14) + '" fill="#7d90ad" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">no change</text>');
    p.push('<text x="' + f(b0) + '" y="' + f(hy + hh + 14) + '" fill="#7d90ad" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">clip</text>');
    p.push('<text x="' + f(b1) + '" y="' + f(hy + hh + 14) + '" fill="#7d90ad" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">clip</text>');

    // Entropy trail — how much exploration is left
    // Absolute axis, 0 to ln 5. Fitting the trail to its own min and max turned 0.3% of
    // noise into a full-height collapse, which is the shape of the story this panel is
    // meant to tell and therefore the last thing it should draw when it has not happened.
    var ty = hy + hh + 48, th = h - ty - 46;
    var EMAX = Math.log(5);
    if (this.trail.length > 2 && th > 10) {
      var d = '';
      for (var j = 0; j < this.trail.length; j++) {
        var px = hx + j / (this.trail.length - 1) * hw2;
        var py = ty + th - Math.max(0, Math.min(1, this.trail[j] / EMAX)) * th;
        d += (j ? 'L' : 'M') + f(px) + ' ' + f(py);
      }
      p.push('<line x1="' + f(hx) + '" y1="' + f(ty) + '" x2="' + f(hx + hw2) + '" y2="' + f(ty) + '" stroke="rgba(130,160,200,.14)" stroke-width="1"/>');
      p.push('<line x1="' + f(hx) + '" y1="' + f(ty + th) + '" x2="' + f(hx + hw2) + '" y2="' + f(ty + th) + '" stroke="rgba(130,160,200,.14)" stroke-width="1"/>');
      p.push('<path d="' + d + '" fill="none" stroke="' + accent + '" stroke-width="1.6" opacity=".8"/>');
      // One string, left-aligned. The value used to be right-aligned on the same
      // baseline, which collided with the label the moment the label grew.
      p.push('<text x="' + hx + '" y="' + f(ty - 8) + '" fill="#7d90ad" font-size="10">'
        + 'ENTROPY <tspan fill="#c9d8ee" font-family="JetBrains Mono,monospace">'
        + fmt(this.trail[this.trail.length - 1], 3) + '</tspan>'
        + ' · how much doubt is left · axis fixed, 0 to ln 5 = 1.609</text>');
    }

    // The footer used to read "clipped 1% KL 0.0031 value-fit 38% year 1" — four numbers
    // with nothing saying what any of them means. Each now carries its own plain label.
    var s = this.stats;
    var cells = [
      ['CLIPPED', s.clipped !== undefined ? Math.round(s.clipped * 100) + '%' : '—',
       'of moves the leash held back'],
      ['KL', fmt(s.kl, 4), 'how far the policy shifted'],
      ['VALUE-FIT', s.ev !== undefined ? Math.round(s.ev * 100) + '%' : '—',
       'how well it predicts its own score'],
      ['YEAR', s.year ? String(s.year) : '—', 'difficulty of the opponents']
    ];
    var cwid = (w - 32) / cells.length;
    for (var ci = 0; ci < cells.length; ci++) {
      var cxp = bx + ci * cwid;
      p.push('<text x="' + f(cxp) + '" y="' + (h - 22) + '" fill="#61708a" font-size="9" letter-spacing="1.2" font-family="JetBrains Mono,monospace">' + cells[ci][0] + '</text>');
      p.push('<text x="' + f(cxp + 62) + '" y="' + (h - 22) + '" fill="#e8eef9" font-size="12" font-family="JetBrains Mono,monospace">' + cells[ci][1] + '</text>');
      p.push('<text x="' + f(cxp) + '" y="' + (h - 8) + '" fill="#4f6280" font-size="9">' + cells[ci][2] + '</text>');
    }
    p.push('</svg>');
    return p.join('');
  };

  /* ---------------- GA ----------------
   *
   * Selection is a story about individuals: this one bred, that one did not. The panel
   * could not tell it, because the grid was ranked by fitness every generation -- a
   * genome changed slot whenever its score moved, so nobody could be followed for more
   * than one beat and nobody was ever seen to die.
   *
   * The grid is laid out by genome index instead, which is stable by construction:
   * `tell()` writes the survivors into the first `elite` slots and the children after
   * them. Position stops moving, so colour and brightness can carry the story --
   * ancestry and who is about to be replaced.
   */

  /* Circular mean, so a child sits BETWEEN its parents on the wheel: hue 350 bred with
     hue 10 has to meet at 0, not at the numeric average of 180 -- which is the colour of
     neither parent and would invent a lineage nobody has. */
  function hueBlend(a, b) {
    var x = Math.cos(a * Math.PI / 180) + Math.cos(b * Math.PI / 180);
    var y = Math.sin(a * Math.PI / 180) + Math.sin(b * Math.PI / 180);
    if (Math.abs(x) < 1e-9 && Math.abs(y) < 1e-9) return a;
    return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
  }
  function hueCss(hu, l, sat) {
    return 'hsl(' + (hu || 0).toFixed(0) + ',' + (sat === undefined ? 60 : sat) + '%,' + (l || 58) + '%)';
  }

  function GaPanel() {
    this.fp = null;
    this.fit = null;
    this.elites = [];
    this.pairs = [];
    this.mut = [];
    this.stats = {};
    this.gen = 0;
    this.flash = 0;
    // Identity of the population currently on screen, one entry per slot.
    this.hue = null;
    this.founder = null;
    // The previous message's survivors and pairings -- what BUILT the population this
    // message reports, and therefore what the identities have to be carried across.
    this.prev = null;
    // Founder counts per generation, oldest first: the dynasty band.
    this.dynasty = [];
  }

  /* Carry identity across one generation. `tell()` builds the next population as
     `nxt[:elite] = pop[elites]` followed by one child per entry of `pairs`, so slot j
     below `elite` IS last generation's `elites[j]`, and every slot after it is the child
     of a known pair. Nothing here is inferred. */
  GaPanel.prototype.advance = function (n, elites, pairs, fit) {
    if (!this.hue || !elites || !pairs) return;
    var hue = new Array(n), founder = new Array(n), e = elites.length;
    for (var i = 0; i < n; i++) {
      var src = i < e ? elites[i] : -1;
      if (src >= 0 && this.hue[src] !== undefined) {
        hue[i] = this.hue[src]; founder[i] = this.founder[src];
        continue;
      }
      var pr = pairs[i - e];
      if (!pr || this.hue[pr[0]] === undefined || this.hue[pr[1]] === undefined) {
        hue[i] = this.hue[i] === undefined ? i * 360 / n : this.hue[i];
        founder[i] = this.founder[i] === undefined ? i : this.founder[i];
        continue;
      }
      hue[i] = hueBlend(this.hue[pr[0]], this.hue[pr[1]]);
      // Counted under the fitter parent's line. Once one founder is on both sides of
      // most pairings, that is the line that has taken the population over, and the
      // band below should say so rather than splitting the credit evenly.
      founder[i] = (fit && fit[pr[1]] > fit[pr[0]]) ? this.founder[pr[1]] : this.founder[pr[0]];
    }
    this.hue = hue; this.founder = founder;
  };

  GaPanel.prototype.update = function (t) {
    if (t && t.role) this.role = t.role;
    if (!t) return;
    this.fpTarget = t.fingerprints || this.fpTarget;
    this.fitTarget = t.fitness || this.fitTarget;
    this.elites = t.elites || this.elites;
    this.pairs = t.pairs || this.pairs;
    this.mut = t.mutation || this.mut;
    this.order = t.order || this.order;
    this.stats = {
      sigma: t.sigma, diversity: t.diversity, beat: t.childBeatParents,
      best: t.fitBest, mean: t.fitMean, year: t.year, ladder: t.ladderWin
    };
    this.gen = (t.gen !== undefined) ? t.gen : this.gen + 1;
    // Wall clock, not frames. The decay was 0.045 per animation frame, so the crossover
    // curves lived 22 frames — 0.37s at 60Hz, 0.19s on a 120Hz panel — against a
    // generation that lands about every 1.1s. Breeding read as a strobe rather than an
    // event, and there was no moment at which "those two, right there" was still true
    // while the sentence finished.
    this.flashAt = now();
    var n = (this.fitTarget || []).length;
    if (!n) return;
    if (!this.hue || this.hue.length !== n) {
      // Founders: one hue each, evenly around the wheel. Every colour after this one is
      // descended from these, which is the whole point of the band.
      this.hue = []; this.founder = [];
      for (var i = 0; i < n; i++) { this.hue.push(i * 360 / n); this.founder.push(i); }
      this.dynasty = [];
      // Lineage is tracked from telemetry, and telemetry starts arriving when this
      // window connects. Joining at generation 300 means these "founders" are simply
      // whoever was alive at generation 300 -- calling them the founders of the run
      // would be a lie the band cannot back up, so the caption says which it is.
      this.baseGen = this.gen;
    } else if (this.prev) {
      this.advance(n, this.prev.elites, this.prev.pairs, this.prev.fit);
    }
    this.prev = { elites: this.elites, pairs: this.pairs, fit: this.fitTarget };

    var counts = {};
    for (var j = 0; j < n; j++) counts[this.founder[j]] = (counts[this.founder[j]] || 0) + 1;
    this.dynasty.push(counts);
    // One generation per pixel column at most; older ones fall off the left.
    if (this.dynasty.length > 240) this.dynasty.shift();
  };

  GaPanel.prototype.draw = function (w, h, accent, opt) {
    var fpT = this.fpTarget, fitT = this.fitTarget;
    if (!fpT || !fitT) return waiting(w, h, opt);
    this.fit = easeArr(this.fit, fitT, 0.2);
    // Two seconds, so a pairing is still on screen when the sentence about it ends.
    this.flash = Math.max(0, 1 - (now() - (this.flashAt || 0)) / 2000);
    var p = [], f = function (v) { return (+v).toFixed(1); };
    var n = fpT.length;
    // Twelve columns was written for a population of forty-eight. The academy's slider
    // goes far past that -- at 120 it gave ten rows of 56x28 letterboxes -- so the grid
    // is shaped from the box it actually has, which keeps the cells roughly square at
    // any population size.
    var gwTmp = w - 32, ghTmp = h - 138;
    var cols = Math.max(6, Math.min(24, Math.round(Math.sqrt(n * gwTmp / Math.max(1, ghTmp)))));
    var rows = Math.ceil(n / cols);
    var nE = (this.elites || []).length;
    var self = this;
    var hueOf = function (i) { return self.hue && self.hue[i] !== undefined ? self.hue[i] : 0; };

    p.push('<svg viewBox="0 0 ' + w + ' ' + h + '" width="100%" height="100%" style="display:block">');
    p.push(roleTag(this.role, 16, 14));
    p.push(heading(16, 36, n + ' BRAINS · ' + nE + ' SURVIVE, ' + (n - nE) + ' ARE REPLACED',
                   'colour is ancestry — a child is its two parents mixed · dim ones do not breed'));

    var gx = 16, gy = 62, gw = w - 32, gh = h - 138;
    var cw = gw / cols, ch = gh / rows;
    var pos = {};
    var lo = Math.min.apply(null, this.fit), hi = Math.max.apply(null, this.fit);
    var span = Math.max(1e-6, hi - lo);
    // The genome the arena is actually playing. `tell()` keeps `best = pop[order[0]]`,
    // so this is not a guess about which cat is on screen -- it is the same index.
    var inArena = this.order && this.order.length ? this.order[0] : -1;

    for (var i = 0; i < n; i++) {
      var col = i % cols, row = (i / cols) | 0;
      var x = gx + col * cw, y = gy + row * ch;
      pos[i] = { x: x + cw / 2, y: y + ch / 2 };
      var isElite = this.elites.indexOf(i) >= 0;
      var hu = hueOf(i);
      var stroke = isElite ? hueCss(hu, 66) : 'rgba(130,160,200,.16)';
      var strip = fpT[i] || [];
      p.push('<g transform="translate(' + f(x + 2) + ',' + f(y + 2) + ')" opacity="'
        + (isElite ? 1 : 0.5) + '">');
      p.push('<rect x="0" y="0" width="' + f(cw - 4) + '" height="' + f(ch - 8) + '" rx="3" fill="'
        + (isElite ? hueCss(hu, 22, 45) : 'rgba(255,255,255,.03)') + '" stroke="' + stroke
        + '" stroke-width="' + (isElite ? 1.5 : 0.8) + '"'
        + (isElite ? ' opacity="' + (0.75 + 0.25 * this.flash).toFixed(2) + '"' : '') + '/>');
      // The genome strip is only drawn while a bar is wide enough to be a bar. Forty
      // coordinates across a 56px cell is one pixel each: a texture that looks like
      // detail and carries none, over the one signal that does read at this size --
      // whose descendant this is. Below the threshold the cell becomes that colour.
      var bw2 = (cw - 10) / Math.max(1, strip.length);
      if (bw2 >= 2.2) {
        for (var b = 0; b < strip.length; b++) {
          var v = strip[b];
          var mag = Math.min(1, Math.abs(v));
          var hgt = Math.max(1, mag * (ch - 22));
          p.push('<rect x="' + f(3 + b * bw2) + '" y="' + f((ch - 14 - hgt) / 2 + 2) + '" width="'
            + f(Math.max(0.8, bw2 - 0.4)) + '" height="' + f(hgt)
            + '" fill="' + (v >= 0 ? hueCss(hu, 62) : hueCss(hu, 34, 30))
            + '" opacity="' + (0.3 + 0.6 * mag).toFixed(2) + '"/>');
        }
      } else {
        var frTile = (this.fit[i] - lo) / span;
        p.push('<rect x="3" y="3" width="' + f(cw - 10) + '" height="' + f(ch - 16)
          + '" rx="2" fill="' + hueCss(hu, 54) + '" opacity="' + (0.3 + 0.55 * frTile).toFixed(2) + '"/>');
      }
      var fr = (this.fit[i] - lo) / span;
      p.push('<rect x="3" y="' + f(ch - 11) + '" width="' + f((cw - 10) * fr)
        + '" height="2.5" rx="1.2" fill="' + hueCss(hu, isElite ? 64 : 40) + '"/>');
      // The condemned are struck through rather than merely dimmed. Forty of these are
      // gone the moment this generation ends, and that is the event the screen exists
      // to show -- dimming alone read as "less important", not as "about to die".
      if (!isElite) {
        p.push('<line x1="3" y1="' + f(ch - 9) + '" x2="' + f(cw - 7) + '" y2="4" stroke="rgba(255,138,92,.22)" stroke-width="1"/>');
      }
      p.push('</g>');
    }

    // Which strip is the cat you are watching. Without it the grid and the arena are two
    // unrelated pictures on one screen.
    if (pos[inArena]) {
      var a0 = pos[inArena];
      p.push('<rect x="' + f(a0.x - cw / 2 + 1) + '" y="' + f(a0.y - ch / 2 + 1) + '" width="' + f(cw - 2)
        + '" height="' + f(ch - 6) + '" rx="4" fill="none" stroke="' + accent + '" stroke-width="1.6"/>');
      p.push('<text x="' + f(a0.x) + '" y="' + f(a0.y - ch / 2 - 2) + '" fill="' + accent
        + '" font-size="8" letter-spacing="1.1" text-anchor="middle"'
        + ' font-family="JetBrains Mono,monospace">IN THE ARENA</text>');
    }

    // A few of this generation's pairings. The child is drawn between its parents rather
    // than in a slot: it belongs to the NEXT population and has no slot here yet.
    if (this.pairs.length) {
      var shown = Math.min(6, this.pairs.length);
      for (var q = 0; q < shown; q++) {
        var pa = pos[this.pairs[q][0]], pb = pos[this.pairs[q][1]];
        if (!pa || !pb) continue;
        var midx = (pa.x + pb.x) / 2, midy = (pa.y + pb.y) / 2 - 10;
        var op = (0.55 * this.flash).toFixed(2);
        var kid = hueBlend(hueOf(this.pairs[q][0]), hueOf(this.pairs[q][1]));
        p.push('<path d="M' + f(pa.x) + ' ' + f(pa.y) + 'Q' + f(midx) + ' ' + f(midy - 8) + ' '
          + f(midx) + ' ' + f(midy) + '" fill="none" stroke="' + hueCss(hueOf(this.pairs[q][0]), 60)
          + '" stroke-width="1.1" opacity="' + op + '"/>');
        p.push('<path d="M' + f(pb.x) + ' ' + f(pb.y) + 'Q' + f(midx) + ' ' + f(midy - 8) + ' '
          + f(midx) + ' ' + f(midy) + '" fill="none" stroke="' + hueCss(hueOf(this.pairs[q][1]), 60)
          + '" stroke-width="1.1" opacity="' + op + '"/>');
        p.push('<circle cx="' + f(midx) + '" cy="' + f(midy) + '" r="3.2" fill="' + hueCss(kid, 62)
          + '" opacity="' + (0.9 * this.flash).toFixed(2) + '"/>');
      }
    }

    // The dynasty band: what share of the population descends from each founder, one
    // column per generation. This is the picture of evolution the grid cannot give,
    // because the grid only ever shows one generation at a time.
    var by = h - 66, bh = 22, bw = w - 32, lines = 0;
    if (this.dynasty.length) {
      var last = this.dynasty[this.dynasty.length - 1];
      lines = Object.keys(last).length;
      var colW = bw / this.dynasty.length;
      for (var g = 0; g < this.dynasty.length; g++) {
        var cnt = this.dynasty[g], acc = 0;
        var keys = Object.keys(cnt).sort(function (m2, n2) { return +m2 - +n2; });
        for (var k = 0; k < keys.length; k++) {
          var share = cnt[keys[k]] / n, segH = share * bh;
          p.push('<rect x="' + f(16 + g * colW) + '" y="' + f(by + acc) + '" width="'
            + f(Math.max(0.7, colW + 0.3)) + '" height="' + f(Math.max(0.6, segH))
            + '" fill="' + hueCss(+keys[k] * 360 / n, 55) + '" opacity=".9"/>');
          acc += segH;
        }
      }
      var fromStart = !this.baseGen;
      p.push('<text x="' + (w - 16) + '" y="' + (by - 4) + '" fill="#61708a" font-size="9"'
        + ' text-anchor="end" font-family="JetBrains Mono,monospace">'
        + lines + ' of ' + n + ' line' + (lines === 1 ? '' : 's') + ' still alive</text>');
      p.push('<text x="16" y="' + (by - 4) + '" fill="#61708a" font-size="9" letter-spacing="1.1">'
        + (fromStart ? 'DYNASTIES · one column per generation'
                     : 'LINES SINCE GENERATION ' + this.baseGen + ' · one column per generation')
        + '</text>');
    }

    var st = this.stats;
    p.push('<text x="16" y="' + (h - 26) + '" fill="#c9d8ee" font-size="11" font-family="JetBrains Mono,monospace">generation '
      + this.gen + '   sigma ' + fmt(st.sigma, 4) + '   diversity ' + fmt(st.diversity, 4) + '</text>');
    p.push('<text x="16" y="' + (h - 9) + '" fill="#7d90ad" font-size="10.5">'
      + 'Bottom bar = fitness. Lit ones survive and breed; struck-through ones are replaced. '
      + 'The band is who everyone is descended from.</text>');
    p.push('</svg>');
    return p.join('');
  };

  /* ---------------- CMA-ES ---------------- */

  function CmaPanel() {
    this.samples = null;
    this.ghosts = [];
    this.ellipse = { rx: 1, ry: 1, angle: 0 };
    this.step = [0, 0];
    this.stats = {};
    this.gen = 0;
    this.scale = 1;
  }

  CmaPanel.prototype.update = function (t) {
    if (!t || !t.samples) return;
    if (this.samples) {
      this.ghosts.push(this.samples);
      if (this.ghosts.length > 4) this.ghosts.shift();
    }
    this.samples = t.samples;
    this.selected = t.selected || (t.samples.length / 2) | 0;
    this.ellipseT = t.ellipse || this.ellipseT;
    this.stepT = t.meanStep || [0, 0];
    this.role = t.role || this.role;
    this.stats = {
      sigma: t.sigma, condition: t.condition, ps: t.psNorm,
      best: t.fitBest, mean: t.fitMean, year: t.year, ladder: t.ladderWin
    };
    this.gen = (t.gen !== undefined) ? t.gen : this.gen + 1;
    // Autoscale from the cloud so the ellipse fills the frame as sigma shrinks —
    // otherwise the whole picture collapses to a dot within a minute.
    var m = 0;
    for (var i = 0; i < t.samples.length; i++) {
      m = Math.max(m, Math.abs(t.samples[i][0]), Math.abs(t.samples[i][1]));
    }
    this.scaleT = Math.max(1e-6, m * 1.25);
    // The frame zooms in as the cloud contracts, which is what keeps the picture legible
    // — and also what hides the contraction, since the ellipse fills the same space at
    // any sigma. Remember the first scale so the zoom itself can be shown.
    if (this.scale0 === undefined) this.scale0 = this.scaleT;
  };

  CmaPanel.prototype.draw = function (w, h, accent, opt) {
    if (!this.samples) return waiting(w, h, opt);
    this.scale = ease(this.scale, this.scaleT || 1, 0.08);
    if (this.ellipseT) {
      this.ellipse.rx = ease(this.ellipse.rx, this.ellipseT.rx, 0.15);
      this.ellipse.ry = ease(this.ellipse.ry, this.ellipseT.ry, 0.15);
      // Angles wrap; ease along the short way round or the ellipse spins on a flip.
      var da = this.ellipseT.angle - this.ellipse.angle;
      while (da > Math.PI / 2) da -= Math.PI;
      while (da < -Math.PI / 2) da += Math.PI;
      this.ellipse.angle += da * 0.15;
    }
    this.step = easeArr(this.step, this.stepT || [0, 0], 0.15);

    var p = [], f = function (v) { return (+v).toFixed(1); };
    var cx = w * 0.36, cy = h * 0.5 + 16, R = Math.min(w * 0.32, h * 0.36);
    var k = R / this.scale;
    var X = function (v) { return cx + v * k; };
    var Y = function (v) { return cy - v * k; };

    p.push('<svg viewBox="0 0 ' + w + ' ' + h + '" width="100%" height="100%" style="display:block">');
    p.push(roleTag(this.role, 16, 14));
    p.push(heading(16, 36, 'THE SEARCH CLOUD, AND THE SHAPE IT HAS LEARNED',
                   'each dot is one sampled brain, on the two directions it varies most'));

    for (var r = 1; r <= 3; r++) {
      p.push('<circle cx="' + f(cx) + '" cy="' + f(cy) + '" r="' + f(R * r / 3) + '" fill="none" stroke="rgba(130,160,200,.10)" stroke-width="1"/>');
    }
    // The ring carries its real value, and the zoom says how far the frame has closed in
    // since the first generation. Without them the cloud looks the same size forever.
    p.push('<text x="' + f(cx) + '" y="' + f(cy - R - 6) + '" text-anchor="middle" fill="#7d90ad" font-size="10" font-family="JetBrains Mono,monospace">±'
      + this.scale.toPrecision(3) + '</text>');
    if (this.scale0 && this.scale0 / this.scale > 1.15) {
      p.push('<text x="' + f(cx) + '" y="' + f(cy + R + 16) + '" text-anchor="middle" fill="#ffd166" font-size="10" font-family="JetBrains Mono,monospace">frame zoomed ×'
        + (this.scale0 / this.scale).toFixed(1) + ' since generation 1</text>');
    }
    p.push('<line x1="' + f(cx - R) + '" y1="' + f(cy) + '" x2="' + f(cx + R) + '" y2="' + f(cy) + '" stroke="rgba(130,160,200,.13)" stroke-width="1"/>');
    p.push('<line x1="' + f(cx) + '" y1="' + f(cy - R) + '" x2="' + f(cx) + '" y2="' + f(cy + R) + '" stroke="rgba(130,160,200,.13)" stroke-width="1"/>');

    for (var g = 0; g < this.ghosts.length; g++) {
      var op = 0.05 + 0.05 * g;
      var gs = this.ghosts[g];
      for (var i2 = 0; i2 < gs.length; i2++) {
        p.push('<circle cx="' + f(X(gs[i2][0])) + '" cy="' + f(Y(gs[i2][1])) + '" r="1.8" fill="' + accent + '" opacity="' + op.toFixed(2) + '"/>');
      }
    }

    var e = this.ellipse;
    p.push('<ellipse cx="' + f(cx) + '" cy="' + f(cy) + '" rx="' + f(Math.max(2, e.rx * k)) + '" ry="' + f(Math.max(2, e.ry * k))
      + '" transform="rotate(' + (-e.angle * 180 / Math.PI).toFixed(1) + ' ' + f(cx) + ' ' + f(cy) + ')" fill="' + accent + '" fill-opacity=".08" stroke="' + accent + '" stroke-width="1.6" opacity=".85"/>');
    p.push('<ellipse cx="' + f(cx) + '" cy="' + f(cy) + '" rx="' + f(Math.max(1, e.rx * k * 0.5)) + '" ry="' + f(Math.max(1, e.ry * k * 0.5))
      + '" transform="rotate(' + (-e.angle * 180 / Math.PI).toFixed(1) + ' ' + f(cx) + ' ' + f(cy) + ')" fill="none" stroke="' + accent + '" stroke-width="1" opacity=".4"/>');

    for (var i = 0; i < this.samples.length; i++) {
      var s2 = this.samples[i], keep = i < this.selected;
      p.push('<circle cx="' + f(X(s2[0])) + '" cy="' + f(Y(s2[1])) + '" r="' + (keep ? 4.2 : 2.6)
        + '" fill="' + (keep ? accent : 'none') + '" stroke="' + (keep ? 'rgba(255,255,255,.55)' : '#61708a')
        + '" stroke-width="' + (keep ? 1 : 1.2) + '" opacity="' + (keep ? 0.95 : 0.5) + '"/>');
    }

    var sxp = X(this.step[0]), syp = Y(this.step[1]);
    p.push('<line x1="' + f(cx) + '" y1="' + f(cy) + '" x2="' + f(sxp) + '" y2="' + f(syp) + '" stroke="#ffd166" stroke-width="2" opacity=".9"/>');
    p.push('<circle cx="' + f(sxp) + '" cy="' + f(syp) + '" r="3.4" fill="#ffd166"/>');
    p.push('<text x="' + f(sxp + 7) + '" y="' + f(syp - 5) + '" fill="#ffd166" font-size="10" font-family="JetBrains Mono,monospace">new centre</text>');

    var lx = cx + R + 28;
    var rows = [
      ['sigma', fmt(this.stats.sigma, 4), 'how wide it is sampling'],
      ['condition', fmt(this.stats.condition, 1), 'across all 2,853 axes, not the 2 drawn'],
      ['path', fmt(this.stats.ps, 2), 'is it still moving in one direction'],
      ['kept', this.selected + ' of ' + this.samples.length, 'the better half recombines']
    ];
    for (var rI = 0; rI < rows.length; rI++) {
      var ry = 46 + rI * 34;
      p.push('<text x="' + f(lx) + '" y="' + f(ry) + '" fill="#7d90ad" font-size="10" letter-spacing="1">' + rows[rI][0].toUpperCase() + '</text>');
      p.push('<text x="' + f(lx) + '" y="' + f(ry + 15) + '" fill="#e8eef9" font-size="15" font-family="JetBrains Mono,monospace">' + rows[rI][1] + '</text>');
      p.push('<text x="' + f(lx + 74) + '" y="' + f(ry + 14) + '" fill="#61708a" font-size="10">' + rows[rI][2] + '</text>');
    }
    /* The main frame rescales to the cloud every generation, which is what keeps the dots
       legible as sigma falls — and what makes generation 1 and generation 3,000 the same
       picture. The contraction was reported only in digits. This is the same cloud drawn
       at the scale it had in generation ONE and never rescaled, so the collapse is the
       thing that moves. It is small on purpose: by the end there is not much left of it,
       and that is the point. */
    if (this.scale0) {
      var ix = lx + 58, iy = 46 + rows.length * 34 + 62, ir = 50;
      var ik = ir / this.scale0;
      p.push('<text x="' + f(lx) + '" y="' + f(iy - ir - 12) + '" fill="#7d90ad" font-size="10" letter-spacing="1">AT GENERATION 1\u2019S SCALE</text>');
      p.push('<circle cx="' + f(ix) + '" cy="' + f(iy) + '" r="' + ir + '" fill="none" stroke="rgba(130,160,200,.18)" stroke-width="1"/>');
      p.push('<circle cx="' + f(ix) + '" cy="' + f(iy) + '" r="' + f(ir / 2) + '" fill="none" stroke="rgba(130,160,200,.09)" stroke-width="1"/>');
      for (var q = 0; q < this.samples.length; q++) {
        var sq = this.samples[q];
        var qx = ix + sq[0] * ik, qy = iy - sq[1] * ik;
        var dx = qx - ix, dy = qy - iy;
        if (dx * dx + dy * dy > ir * ir) continue;          // generation 1 sits on the ring
        p.push('<circle cx="' + f(qx) + '" cy="' + f(qy) + '" r="1.7" fill="' + accent + '" opacity=".85"/>');
      }
      p.push('<text x="' + f(ix) + '" y="' + f(iy + ir + 14) + '" text-anchor="middle" fill="#61708a" font-size="10"'
        + ' font-family="JetBrains Mono,monospace">' + Math.round(100 * this.scale / this.scale0)
        + '% of the spread it started with</text>');
    }
    p.push('<text x="16" y="' + (h - 9) + '" fill="#7d90ad" font-size="10.5">generation ' + this.gen
      + ' · filled dots are the half that gets kept · the big frame rescales to the cloud every'
      + ' generation, so read the ring for its size — the small one never rescales</text>');
    p.push('</svg>');
    return p.join('');
  };

  /* ---------------- live decision (playback) ---------------- */

  /* While a saved policy is PLAYING there is no optimiser running, so there is no
     algorithm telemetry to draw. What there is — and what is arguably more interesting
     on camera — is the network's actual output this step: the five action
     probabilities for each animal, and which one it drew. This is the closest thing
     to showing what the brain is thinking, and unlike a "mode" label it is not an
     interpretation, it is the tensor. */
  function LivePanel() {
    this.cat = null; this.mouse = null; this.frame = null;
  }

  LivePanel.prototype.update = function (frame) {
    if (!frame) return;
    this.frame = frame;
    if (frame.cat && frame.cat.probs) this.catT = frame.cat.probs;
    if (frame.mouse && frame.mouse.probs) this.mouseT = frame.mouse.probs;
  };

  LivePanel.prototype.row = function (label, probs, color, y, w, sense, bh) {
    var p = [], f = function (v) { return (+v).toFixed(1); };
    var bx = 16, bw = w - 32;
    bh = bh || 74;
    p.push('<text x="' + bx + '" y="' + (y - 8) + '" fill="' + color + '" font-size="12" letter-spacing="2" font-weight="700">' + label + '</text>');
    p.push('<text x="' + (bx + 70) + '" y="' + (y - 8) + '" fill="#61708a" font-size="11" font-family="JetBrains Mono,monospace">' + sense + '</text>');
    if (!probs) {
      p.push('<text x="' + bx + '" y="' + (y + 30) + '" fill="#3d4a60" font-size="12">waiting for the policy…</text>');
      return p.join('');
    }
    var top = 0;
    for (var i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
    var slot = bw / probs.length;
    for (var k = 0; k < probs.length; k++) {
      var v = Math.max(0, Math.min(1, probs[k]));
      var hgt = Math.max(2, v * bh);
      var x = bx + k * slot + slot * 0.12, ww = slot * 0.76;
      p.push('<rect x="' + f(x) + '" y="' + f(y) + '" width="' + f(ww) + '" height="' + f(bh) + '" rx="3" fill="rgba(255,255,255,.035)"/>');
      p.push('<rect x="' + f(x) + '" y="' + f(y + bh - hgt) + '" width="' + f(ww) + '" height="' + f(hgt) + '" rx="3" fill="' + color
        + '" opacity="' + (k === top ? 0.95 : 0.35 + 0.4 * v).toFixed(2) + '"/>');
      p.push('<text x="' + f(x + ww / 2) + '" y="' + f(y + bh + 14) + '" fill="' + (k === top ? '#e8eef9' : '#61708a')
        + '" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">' + ACTIONS[k] + '</text>');
      p.push('<text x="' + f(x + ww / 2) + '" y="' + f(y + bh - hgt - 5) + '" fill="#8fa4c4" font-size="9.5" text-anchor="middle" font-family="JetBrains Mono,monospace">'
        + Math.round(v * 100) + '</text>');
    }
    return p.join('');
  };

  LivePanel.prototype.draw = function (w, h, accent) {
    // No easing here, unlike the explainer panels. Those smooth telemetry that arrives
    // a couple of times a second; this one gets a fresh tensor every simulation step,
    // and a 25%-per-frame lag would print a running average while the caption says the
    // bars are the network's own numbers. Measured on a recorded session at 4x, the
    // eased version highlighted the wrong action in 10.6% of frames.
    this.cat = this.catT;
    this.mouse = this.mouseT;
    var fr = this.frame || { cat: {}, mouse: {} };
    var p = ['<svg viewBox="0 0 ' + w + ' ' + h + '" width="100%" height="100%" style="display:block">'];
    p.push('<text x="16" y="20" fill="#8fa4c4" font-size="11" letter-spacing="1.4">WHAT EACH BRAIN IS ABOUT TO DO</text>');
    /* The two rows share whatever lies between the heading and the captions. The bars
       were a fixed 74px, which is right for the 420-tall panel this was written against
       and a quarter of the room in the 851-tall card it is actually drawn into on the
       school screen. On camera the bar height IS the reading — it is the one number the
       narration points at — so it gets the space rather than being centred in it. */
    var LABEL = 24, TICK = 20, GAP = 26, HEAD = 38, CAPS = 44;
    var avail = (h - CAPS) - HEAD;
    var bh = Math.max(48, Math.floor((avail - 2 * (LABEL + TICK) - GAP) / 2));
    var y1 = HEAD + LABEL;
    var y2 = y1 + bh + TICK + GAP + LABEL;
    p.push(this.row('TOM', this.catT ? this.cat : null, '#ff8a5c', y1, w,
      (fr.cat.mode || '—') + (fr.cat.frozen ? '  · frozen ' + fr.cat.frozen : ''), bh));
    p.push(this.row('JERRY', this.mouseT ? this.mouse : null, '#7ee0ff', y2, w,
      (fr.mouse.mode || '—') + (fr.mouse.heard ? '  · hears him, confidence ' + fr.mouse.heard.conf.toFixed(2) : '')
      + (fr.mouse.frozen ? '  · frozen ' + fr.mouse.frozen : ''), bh));
    p.push('<text x="16" y="' + (h - 26) + '" fill="#7d90ad" font-size="11">'
      + 'Five bars, five moves. The tall one is what the network wants; the others are how much doubt is left in it.</text>');
    p.push('<text x="16" y="' + (h - 8) + '" fill="#61708a" font-size="11" font-family="JetBrains Mono,monospace">'
      + 'step ' + (fr.step || 0) + '   ' + (fr.nestDist !== undefined ? fr.nestDist + ' cells from home' : '') + '</text>');
    p.push('</svg>');
    return p.join('');
  };

  global.Panels = {
    live: function () { return new LivePanel(); },
    create: function (key) {
      if (key === 'ppo') return new PpoPanel();
      if (key === 'ga') return new GaPanel();
      return new CmaPanel();
    }
  };
})(window);
