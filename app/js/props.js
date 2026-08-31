/* The painted props: the trap and the mouse hole.
 *
 * Same idea as walksprite.js and deliberately much smaller. A character needs eight rows
 * because it walks; a prop sits on its cell and is seen from the one camera the arena is
 * drawn from, so it needs one row of named frames and no mirrors.
 *
 * What it shares with the character sheets is the part that matters — the metadata
 * carries the object's own height as a share of the canvas, so a caller sizes by the
 * OBJECT and not by the padding around it. The two props have very different padding
 * (an open trap is more than half again as wide as it is tall) and sizing by the frame
 * would draw one of them wrong.
 */
(function (global) {
  'use strict';

  function propSheet(home) {
    var st = { home: home, meta: null, image: null, url: null,
               ready: false, loading: false, waiting: [] };

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
      req.open('GET', base + 'props.json', true);
      req.onload = function () {
        if (req.status >= 400) { finish(new Error('props metadata ' + req.status)); return; }
        var m;
        try { m = JSON.parse(req.responseText); }
        catch (e) { finish(e); return; }
        st.meta = m;
        st.url = base + m.image;
        var img = new Image();
        img.onload = function () { st.image = img; finish(null); };
        img.onerror = function () { finish(new Error('prop atlas failed to load')); };
        img.src = st.url;
      };
      req.onerror = function () { finish(new Error('props metadata failed to load')); };
      req.send();
    }

    function has(name) {
      return st.ready && st.meta.frame[name] !== undefined;
    }

    /* How tall to draw one frame so that the object inside it is `cells` map cells high.
       Exactly the sum paint.js does for a character, and for the same reason. */
    function sizeFor(kind, cells, CS) {
      if (!st.ready) return 0;
      var h = st.meta.charHeight[kind] || 0.6;
      return CS * cells / h;
    }

    /* An SVG slice, placed by the ground anchor: (ax, ay) is where the point the object
       stands on should land. Same nested-<svg> trick as the character sheets, so the prop
       lives inside the arena's own SVG layer. */
    function svgAt(name, ax, ay, size) {
      if (!st.ready) return '';
      var m = st.meta, i = m.frame[name];
      if (i === undefined) return '';
      var w = m.frameWidth, h = m.frameHeight;
      var x = ax - size * m.anchor.x, y = ay - size * m.anchor.y;
      return '<svg x="' + x.toFixed(2) + '" y="' + y.toFixed(2) + '" width="' + size.toFixed(2)
        + '" height="' + size.toFixed(2) + '" viewBox="' + (i * w) + ' 0 ' + w + ' ' + h + '"'
        + ' preserveAspectRatio="xMidYMid meet">'
        + '<image href="' + st.url + '" x="0" y="0" width="' + (m.columns * w)
        + '" height="' + (m.rows * h) + '" image-rendering="auto"/></svg>';
    }

    return {
      load: load, has: has, sizeFor: sizeFor, svgAt: svgAt,
      ready: function () { return st.ready; },
      meta: function () { return st.meta; }
    };
  }

  global.PropSprite = propSheet('assets/props/');
  global.PropSprite.create = propSheet;
})(window);
