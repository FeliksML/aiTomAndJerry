/* The eight-direction walking sprites.
 *
 * Two characters, one module. Tom is cut from five generated video clips, Jerry from
 * five generated four-frame strips; by the time either reaches here they are the same
 * thing — one transparent PNG of four columns by eight rows, plus a metadata file. In
 * both, the three right-facing directions are exact horizontal mirrors of the
 * left-facing ones, so the two sides can never drift apart. Nothing here touches video
 * or does any image processing: that all happens in tools/, at build time.
 *
 * The module owns three things and deliberately no more:
 *
 *   geometry   Where a given direction and frame live inside the sheet, and where the
 *              sprite's ground anchor sits, so a caller can place the cat by the point
 *              between its feet rather than by a corner of a box.
 *   timing     A walk phase advanced from wall-clock time, not from render count. The
 *              cat therefore walks at the same speed whatever the frame rate, and a
 *              paused game stops the feet instead of freezing them mid-stride.
 *   direction  The movement vector to one of eight names, plus the normalisation that
 *              stops diagonal movement being 1.41x faster than straight movement.
 *
 * It renders nothing itself. paint.js asks for an SVG slice, the sprite lab asks for CSS
 * background values; both come from the same numbers.
 */
(function (global) {
  'use strict';

  var DIRECTIONS = ['up', 'up-right', 'right', 'down-right',
                    'down', 'down-left', 'left', 'up-left'];

  // env.js facings are N E S W; the sheet knows all eight.
  var FROM_FACING = ['up', 'right', 'down', 'left'];

  var CAMEL = {
    'up': 'up', 'up-right': 'upRight', 'right': 'right', 'down-right': 'downRight',
    'down': 'down', 'down-left': 'downLeft', 'left': 'left', 'up-left': 'upLeft'
  };

  /* ---------- one sheet ---------- */

  /* A sheet owns nothing but its metadata, its image and its URL. Everything else is a
     pure function of those, which is why two characters can share one module and why
     nothing here has to be told when the game pauses. */
  function sheet(name, meta_name, home) {
    var st = { name: name, metaName: meta_name, home: home, meta: null, image: null,
               url: null, ready: false, loading: false, waiting: [] };

    /* One PNG and one JSON, fetched once. Callers that render every frame must not touch
       the network, so everything below reads `st` and never re-requests. */
    function load(base, done) {
      if (st.ready) { if (done) done(null, st); return; }
      if (done) st.waiting.push(done);
      if (st.loading) return;
      st.loading = true;
      base = base || st.home;

      var finish = function (err) {
        st.ready = !err;
        st.loading = false;
        var q = st.waiting; st.waiting = [];
        q.forEach(function (cb) { cb(err, st); });
      };

      var req = new XMLHttpRequest();
      req.open('GET', base + st.metaName, true);
      req.onload = function () {
        if (req.status >= 400) { finish(new Error(st.name + ' metadata ' + req.status)); return; }
        var m;
        try { m = JSON.parse(req.responseText); }
        catch (e) { finish(e); return; }
        st.meta = m;
        st.url = base + m.image;
        var img = new Image();
        img.onload = function () { st.image = img; finish(null); };
        img.onerror = function () { finish(new Error(st.name + ' sheet failed to load')); };
        img.src = st.url;
      };
      req.onerror = function () { finish(new Error(st.name + ' metadata failed to load')); };
      req.send();
    }

    function row(direction) {
      if (!st.meta) return 0;
      var r = st.meta.directions[CAMEL[direction] || direction];
      return r === undefined ? 0 : r;
    }

    function cell(direction, frame) {
      var m = st.meta, n = m ? m.framesPerDirection : 4;
      var f = ((frame % n) + n) % n;
      return { x: f * (m ? m.frameWidth : 0), y: row(direction) * (m ? m.frameHeight : 0),
               w: m ? m.frameWidth : 0, h: m ? m.frameHeight : 0 };
    }

    function idleFrame(direction) {
      var m = st.meta;
      if (!m || !m.idleFrame) return 0;
      var v = m.idleFrame[CAMEL[direction] || direction];
      return v === undefined ? 0 : v;
    }

    /* Phase comes from elapsed time so the walk runs at its own rate rather than the
       renderer's, and holding still parks the character on its idle pose instead of on
       whichever frame the last redraw happened to leave behind. */
    function frameAt(direction, moving, nowMs, fps) {
      if (!moving) return idleFrame(direction);
      var m = st.meta, n = m ? m.framesPerDirection : 4;
      return Math.floor(nowMs / 1000 * (fps || (m ? m.defaultFPS : 8))) % n;
    }

    /* A sheet whose first frames are a one-shot, driven by a countdown rather than by the
       clock. Being caught in a trap is not a loop: the snap happens once, and only the
       struggle after it repeats. The environment counts a trapped agent down from
       CFG.freezeSteps, so that counter is the animation's own clock and the two can never
       drift apart — the snap is drawn on exactly the step the jaw closed. */
    function frameForHold(remaining, total) {
      var m = st.meta;
      var n = m ? m.framesPerDirection : 4;
      var lead = m && m.oneShotFrames !== undefined ? m.oneShotFrames : 0;
      if (!lead) return frameAt(null, true, 0, null);
      var k = Math.max(0, (total || n) - remaining);
      return k < lead ? k : lead + ((k - lead) % Math.max(1, n - lead));
    }

    /* A small walker for callers that want the phase to survive stopping and starting.
       Keeping the accumulator here rather than in component state is the point: the walk
       advances on the game's own clock and costs no re-render. */
    function walker(opts) {
      opts = opts || {};
      var w = { direction: opts.direction || 'down', moving: false,
                fps: opts.fps || (st.meta ? st.meta.defaultFPS : 8), phase: 0, last: null };
      w.update = function (nowMs, moving, direction) {
        if (direction) w.direction = direction;
        w.moving = !!moving;
        var dt = w.last === null ? 0 : Math.max(0, nowMs - w.last);
        w.last = nowMs;
        if (w.moving) w.phase += dt / 1000 * w.fps;
        return w;
      };
      w.frame = function () {
        var n = st.meta ? st.meta.framesPerDirection : 4;
        return w.moving ? (Math.floor(w.phase) % n + n) % n : idleFrame(w.direction);
      };
      return w;
    }

    /* An SVG slice. A nested <svg> with a viewBox both crops the sheet to one cell and
       scales it, which keeps the sprite inside the arena's existing SVG layer instead of
       needing a second stacking context on top of it. */
    function svgSlice(direction, frame, x, y, w, h) {
      if (!st.ready) return '';
      var m = st.meta, c = cell(direction, frame);
      return '<svg x="' + x.toFixed(2) + '" y="' + y.toFixed(2) + '" width="' + w.toFixed(2)
        + '" height="' + h.toFixed(2) + '" viewBox="' + c.x + ' ' + c.y + ' ' + c.w + ' ' + c.h
        + '" preserveAspectRatio="xMidYMid meet">'
        + '<image href="' + st.url + '" x="0" y="0" width="' + (m.columns * m.frameWidth)
        + '" height="' + (m.rows * m.frameHeight) + '" image-rendering="auto"/></svg>';
    }

    /* Placed by the ground anchor: (ax, ay) is where the point between the feet should
       land, and `size` is the rendered height of one frame. */
    function svgAt(direction, frame, ax, ay, size) {
      if (!st.ready) return '';
      var a = st.meta.anchor;
      return svgSlice(direction, frame, ax - size * a.x, ay - size * a.y, size, size);
    }

    function backgroundStyle(direction, frame, size) {
      if (!st.ready) return {};
      var m = st.meta, c = cell(direction, frame), k = size / m.frameWidth;
      return {
        backgroundImage: 'url(' + st.url + ')',
        backgroundSize: (m.columns * m.frameWidth * k) + 'px ' + (m.rows * m.frameHeight * k) + 'px',
        backgroundPosition: (-c.x * k) + 'px ' + (-c.y * k) + 'px',
        width: size + 'px', height: size + 'px'
      };
    }

    return {
      name: name, load: load,
      ready: function () { return st.ready; },
      meta: function () { return st.meta; },
      cell: cell, row: row, idleFrame: idleFrame, frameAt: frameAt,
      frameForHold: frameForHold, create: walker,
      svgSlice: svgSlice, svgAt: svgAt, backgroundStyle: backgroundStyle,
      DIRECTIONS: DIRECTIONS, dirFromVector: dirFromVector,
      normalize: normalize, fromFacing: fromFacing
    };
  }

  /* ---------- direction from movement ---------- */

  function dirFromVector(dx, dy) {
    if (!dx && !dy) return null;
    var s = function (v) { return v > 0.0001 ? 1 : (v < -0.0001 ? -1 : 0); };
    var key = s(dx) + ',' + s(dy);
    return {
      '0,-1': 'up', '1,-1': 'up-right', '1,0': 'right', '1,1': 'down-right',
      '0,1': 'down', '-1,1': 'down-left', '-1,0': 'left', '-1,-1': 'up-left'
    }[key] || null;
  }

  /* Diagonals must not be faster. Adding speed to both axes gives sqrt(2) times the
     intended speed on a diagonal, which is the single most common way an eight-direction
     control scheme feels wrong. */
  function normalize(dx, dy) {
    var len = Math.sqrt(dx * dx + dy * dy);
    return len > 0 ? { x: dx / len, y: dy / len } : { x: 0, y: 0 };
  }

  function fromFacing(facing) { return FROM_FACING[facing] || 'down'; }

  /* One entry per character, one sheet per animation. A character whose trapped sheet is
     missing simply keeps walking — nothing here fails hard on an absent file. */
  function character(hero, home) {
    return {
      hero: hero, home: home,
      walk: sheet(hero, hero + '-walk.json', home),
      trapped: sheet(hero, hero + '-trapped.json', home)
    };
  }

  var tom = character('tom', 'assets/cat/');
  var jerry = character('jerry', 'assets/mouse/');

  global.WalkSprite = {
    sheet: sheet, character: character, tom: tom, jerry: jerry,
    ANIMATIONS: ['walk', 'trapped'],
    DIRECTIONS: DIRECTIONS, dirFromVector: dirFromVector,
    normalize: normalize, fromFacing: fromFacing
  };
  // paint.js and the sprite lab reach for the walking pair by name; keep that working.
  global.CatSprite = tom.walk;
  global.MouseSprite = jerry.walk;
})(window);
