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
    this.stats = {
      clipped: t.clippedFrac, kl: t.approxKl, entropy: t.entropy,
      value: t.valueLoss, ev: t.explainedVar, year: t.year, ladder: t.ladderWin
    };
    if (t.entropy !== undefined) {
      this.trail.push(t.entropy);
      if (this.trail.length > 90) this.trail.shift();
    }
  };

  PpoPanel.prototype.draw = function (w, h, accent) {
    this.probs = easeArr(this.probs, this.target || [0.2, 0.2, 0.2, 0.2, 0.2], 0.18);
    this.hist = easeArr(this.hist, this.histTarget || new Array(32).fill(0), 0.22);
    var p = [], f = function (v) { return (+v).toFixed(1); };
    p.push('<svg viewBox="0 0 ' + w + ' ' + h + '" width="100%" height="100%" style="display:block">');

    // Action probabilities
    var bx = 16, by = 34, bw = w * 0.36, bh = h - 76;
    p.push('<text x="' + bx + '" y="20" fill="#8fa4c4" font-size="11" letter-spacing="1.4">WHAT IT WOULD DO</text>');
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
    var hx = bx + bw + 34, hw2 = w - hx - 16, hy = 34, hh = bh * 0.66;
    p.push('<text x="' + hx + '" y="20" fill="#8fa4c4" font-size="11" letter-spacing="1.4">HOW FAR THE UPDATE TRIED TO MOVE</text>');
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
    var ty = hy + hh + 30, th = h - ty - 26;
    if (this.trail.length > 2 && th > 10) {
      var mn = Math.min.apply(null, this.trail), mx = Math.max.apply(null, this.trail);
      var rng = Math.max(1e-6, mx - mn), d = '';
      for (var j = 0; j < this.trail.length; j++) {
        var px = hx + j / (this.trail.length - 1) * hw2;
        var py = ty + th - (this.trail[j] - mn) / rng * th;
        d += (j ? 'L' : 'M') + f(px) + ' ' + f(py);
      }
      p.push('<path d="' + d + '" fill="none" stroke="' + accent + '" stroke-width="1.6" opacity=".8"/>');
      p.push('<text x="' + hx + '" y="' + f(ty - 4) + '" fill="#7d90ad" font-size="10" letter-spacing="1">ENTROPY · how much it is still exploring</text>');
    }

    var s = this.stats;
    p.push('<text x="' + bx + '" y="' + (h - 8) + '" fill="#8fa4c4" font-size="11" font-family="JetBrains Mono,monospace">'
      + 'clipped ' + (s.clipped !== undefined ? Math.round(s.clipped * 100) + '%' : '—')
      + '   KL ' + fmt(s.kl, 4)
      + '   value-fit ' + (s.ev !== undefined ? Math.round(s.ev * 100) + '%' : '—')
      + (s.year ? '   year ' + s.year : '') + '</text>');
    p.push('</svg>');
    return p.join('');
  };

  /* ---------------- GA ---------------- */

  function GaPanel() {
    this.fp = null;
    this.fit = null;
    this.elites = [];
    this.pairs = [];
    this.mut = [];
    this.stats = {};
    this.gen = 0;
    this.flash = 0;
  }

  GaPanel.prototype.update = function (t) {
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
    this.flash = 1;
  };

  GaPanel.prototype.draw = function (w, h, accent) {
    var fpT = this.fpTarget, fitT = this.fitTarget;
    if (!fpT || !fitT) return '<svg viewBox="0 0 ' + w + ' ' + h + '"></svg>';
    this.fit = easeArr(this.fit, fitT, 0.2);
    this.flash = Math.max(0, this.flash - 0.045);
    var p = [], f = function (v) { return (+v).toFixed(1); };
    p.push('<svg viewBox="0 0 ' + w + ' ' + h + '" width="100%" height="100%" style="display:block">');
    p.push('<text x="16" y="20" fill="#8fa4c4" font-size="11" letter-spacing="1.4">FORTY-EIGHT BRAINS · RANKED, THEN BRED</text>');

    var n = fpT.length, cols = 12, rows = Math.ceil(n / cols);
    var gx = 16, gy = 32, gw = w - 32, gh = h - 96;
    var cw = gw / cols, ch = gh / rows;
    var order = this.order || fpT.map(function (_, i) { return i; });
    var pos = {};                       // genome -> slot, so lines can find their ends
    var lo = Math.min.apply(null, this.fit), hi = Math.max.apply(null, this.fit);
    var span = Math.max(1e-6, hi - lo);

    for (var s = 0; s < n; s++) {
      var gi = order[s];
      var col = s % cols, row = (s / cols) | 0;
      var x = gx + col * cw, y = gy + row * ch;
      pos[gi] = { x: x + cw / 2, y: y + ch / 2 };
      var isElite = this.elites.indexOf(gi) >= 0;
      var strip = fpT[gi] || [];
      p.push('<g transform="translate(' + f(x + 2) + ',' + f(y + 2) + ')">');
      p.push('<rect x="0" y="0" width="' + f(cw - 4) + '" height="' + f(ch - 8) + '" rx="3" fill="rgba(255,255,255,.03)" stroke="'
        + (isElite ? accent : 'rgba(130,160,200,.14)') + '" stroke-width="' + (isElite ? 1.4 : 0.8) + '"'
        + (isElite ? ' opacity="' + (0.7 + 0.3 * this.flash).toFixed(2) + '"' : '') + '/>');
      // DNA strip: one bar per projected coordinate, sign as hue, size as magnitude.
      var bw2 = (cw - 10) / strip.length;
      for (var b = 0; b < strip.length; b++) {
        var v = strip[b];
        var mag = Math.min(1, Math.abs(v));
        var hgt = Math.max(1, mag * (ch - 22));
        p.push('<rect x="' + f(3 + b * bw2) + '" y="' + f((ch - 14 - hgt) / 2 + 2) + '" width="' + f(Math.max(0.8, bw2 - 0.4)) + '" height="' + f(hgt)
          + '" fill="' + (v >= 0 ? accent : '#4b5a73') + '" opacity="' + (0.25 + 0.6 * mag).toFixed(2) + '"/>');
      }
      // fitness bar along the bottom
      var fr = (this.fit[gi] - lo) / span;
      p.push('<rect x="3" y="' + f(ch - 11) + '" width="' + f((cw - 10) * fr) + '" height="2.5" rx="1.2" fill="' + (isElite ? accent : '#7d90ad') + '"/>');
      p.push('</g>');
    }

    // Crossover: a handful of children traced back to their two parents.
    if (this.pairs.length) {
      var shown = Math.min(4, this.pairs.length);
      for (var q = 0; q < shown; q++) {
        var childSlot = this.elites.length + q;
        var childGi = order.indexOf ? order[childSlot] : childSlot;
        var c = pos[childGi], a = pos[this.pairs[q][0]], b2 = pos[this.pairs[q][1]];
        if (!c || !a || !b2) continue;
        p.push('<path d="M' + f(a.x) + ' ' + f(a.y) + 'Q' + f((a.x + c.x) / 2) + ' ' + f((a.y + c.y) / 2 - 12) + ' ' + f(c.x) + ' ' + f(c.y)
          + '" fill="none" stroke="' + accent + '" stroke-width="1" opacity="' + (0.35 * this.flash).toFixed(2) + '"/>');
        p.push('<path d="M' + f(b2.x) + ' ' + f(b2.y) + 'Q' + f((b2.x + c.x) / 2) + ' ' + f((b2.y + c.y) / 2 + 12) + ' ' + f(c.x) + ' ' + f(c.y)
          + '" fill="none" stroke="' + accent + '" stroke-width="1" opacity="' + (0.35 * this.flash).toFixed(2) + '"/>');
      }
    }

    var st = this.stats;
    p.push('<text x="16" y="' + (h - 26) + '" fill="#c9d8ee" font-size="11" font-family="JetBrains Mono,monospace">generation '
      + this.gen + '   sigma ' + fmt(st.sigma, 4) + '   diversity ' + fmt(st.diversity, 4) + '</text>');
    p.push('<text x="16" y="' + (h - 9) + '" fill="#7d90ad" font-size="10.5">'
      + 'Left bar = fitness. Lit borders are the elites, carried over untouched. '
      + 'Curves show a child and the two parents it was mixed from.</text>');
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
  };

  CmaPanel.prototype.draw = function (w, h, accent) {
    if (!this.samples) return '<svg viewBox="0 0 ' + w + ' ' + h + '"></svg>';
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
    var cx = w * 0.36, cy = h * 0.5, R = Math.min(w * 0.32, h * 0.40);
    var k = R / this.scale;
    var X = function (v) { return cx + v * k; };
    var Y = function (v) { return cy - v * k; };

    p.push('<svg viewBox="0 0 ' + w + ' ' + h + '" width="100%" height="100%" style="display:block">');
    p.push('<text x="16" y="20" fill="#8fa4c4" font-size="11" letter-spacing="1.4">THE SEARCH CLOUD, AND THE SHAPE IT HAS LEARNED</text>');

    for (var r = 1; r <= 3; r++) {
      p.push('<circle cx="' + f(cx) + '" cy="' + f(cy) + '" r="' + f(R * r / 3) + '" fill="none" stroke="rgba(130,160,200,.10)" stroke-width="1"/>');
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
      ['condition', fmt(this.stats.condition, 1), 'how stretched the shape is'],
      ['path', fmt(this.stats.ps, 2), 'is it still moving in one direction'],
      ['kept', this.selected + ' of ' + this.samples.length, 'the better half recombines']
    ];
    for (var rI = 0; rI < rows.length; rI++) {
      var ry = 46 + rI * 34;
      p.push('<text x="' + f(lx) + '" y="' + f(ry) + '" fill="#7d90ad" font-size="10" letter-spacing="1">' + rows[rI][0].toUpperCase() + '</text>');
      p.push('<text x="' + f(lx) + '" y="' + f(ry + 15) + '" fill="#e8eef9" font-size="15" font-family="JetBrains Mono,monospace">' + rows[rI][1] + '</text>');
      p.push('<text x="' + f(lx + 74) + '" y="' + f(ry + 14) + '" fill="#61708a" font-size="10">' + rows[rI][2] + '</text>');
    }
    p.push('<text x="16" y="' + (h - 9) + '" fill="#7d90ad" font-size="10.5">generation ' + this.gen
      + ' · filled dots are the half that gets kept · the ellipse is the real spread of this generation\'s samples</text>');
    p.push('</svg>');
    return p.join('');
  };

  global.Panels = {
    create: function (key) {
      if (key === 'ppo') return new PpoPanel();
      if (key === 'ga') return new GaPanel();
      return new CmaPanel();
    }
  };
})(window);
