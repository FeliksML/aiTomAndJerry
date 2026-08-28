/* The recording app.
 *
 * Screens: menu -> level generator -> school -> verdict -> grand final -> leaderboard.
 * The arena and the algorithm panels are driven live by the Python trainer over a
 * WebSocket; everything else is local so the page still works with the trainer down.
 *
 * Two things are deliberate and worth keeping:
 *
 *  - The canvas is a fixed 1920x1080 scaled by a transform. Recording is then exact
 *    and identical whatever window it is previewed in.
 *  - Every read of a school's identity goes through Reveal.view(). A screen cannot
 *    leak a sealed school by forgetting to check, because there is nothing to forget.
 *
 * Keys: 1 2 3 schools · g generator · f final · b leaderboard · esc menu
 *       space pause · enter continue · s skip episode · [ ] speed · t train live
 *       w how the algorithm works · r reveal next school · shift+R re-seal one
 *       c sprite skins on/off
 */
(function () {
  'use strict';

  var E = window.CatMouseEnv;
  var P = window.Paint;

  var ALGOS = {
    ppo: {
      key: 'ppo', short: 'PPO', full: 'Proximal Policy Optimization',
      line: 'gradient · clipped trust region',
      color: '#4ea8ff', light: '#a6d3ff', deep: '#0a2544',
      blurb: 'Gradient school. Every update is scored, then deliberately held back — the '
        + 'new policy is never allowed to stray far from the one that collected the data.',
      specs: [['OPTIMISER', 'policy gradient, clipped'], ['BATCH', '512 arenas × 128 steps'],
              ['SIGNATURE', 'steady, low-variance climb']]
    },
    ga: {
      key: 'ga', short: 'GA', full: 'Genetic Algorithm',
      line: 'population 48 · elitism · tournament selection',
      color: '#3ddc84', light: '#9af0be', deep: '#08331f',
      blurb: 'No gradients at all. Forty-eight whole brains a generation; the ones that '
        + 'survived the room breed, their children are a coin-flip mix with a few weights nudged.',
      specs: [['POPULATION', '48 individuals'], ['OPERATORS', 'uniform crossover + sparse mutation'],
              ['SIGNATURE', 'plateaus, then generational leaps']]
    },
    cmaes: {
      key: 'cmaes', short: 'CMA-ES', full: 'Covariance Matrix Adaptation',
      line: 'σ-adaptation · λ=32 · separable (diagonal)',
      color: '#a97cff', light: '#cdb2ff', deep: '#26134d',
      blurb: 'Fits a Gaussian over strategies and moves the whole distribution toward '
        + 'whatever worked — widening along directions that keep paying off, narrowing where they do not.',
      specs: [['LAMBDA', '32 samples per generation'], ['UPDATE', 'σ-adaptation, rank-μ, diagonal C'],
              ['SIGNATURE', 'shape-aware search']]
    }
  };
  var ORDER = ['ppo', 'ga', 'cmaes'];
  var CP = ['untrained', 'half', 'trained'];
  var CP_NAME = { untrained: 'UNTRAINED', half: 'HALF-TRAINED', trained: 'TRAINED' };
  var CS = 44;

  var App = {
    screen: 'menu',
    school: 'ppo',
    checkpoint: 'trained',
    speed: 4,
    playing: true,
    link: 'offline',
    cat: null,                 // catalogue from the trainer
    frame: null, prev: null, map: null, alpha: 1, mapKey: null,
    results: [], runState: null,
    levels: [], genIndex: 0, lastGen: 0,
    panels: {}, lastTel: {},
    banner: null, bannerAt: 0, highlights: null,
    trainInfo: null, showKeys: false,
    // The six-step explainer (explain.js). 0 is closed; 1..6 is the step being read.
    // While it is open the run behind it is paused, and closing puts playback back the
    // way it was rather than unconditionally resuming.
    explain: 0, explainWasPlaying: true,
    // The sprite skins. Loaded once, up front, so no frame of gameplay ever waits on a
    // request; until they arrive the vector pair is drawn and nothing stalls, and a
    // character whose sheet is missing simply keeps its vector skin rather than vanishing.
    sprites: true, spriteFps: 8, catMoving: false, mouseMoving: false,
    net: null
  };
  window.App = App;

  ORDER.forEach(function (k) { App.panels[k] = window.Panels.create(k); });

  if (window.WalkSprite) {
    [window.WalkSprite.tom, window.WalkSprite.jerry].forEach(function (ch) {
      window.WalkSprite.ANIMATIONS.forEach(function (a) {
        ch[a].load(null, function (err, st) {
          if (err) { console.warn(ch.hero + ' ' + a + ' sheet unavailable:', err.message); return; }
          if (a !== 'walk') return;
          App.spriteFps = st.meta.defaultFPS;
          // The card portraits are built into the HTML at render time, so a screen drawn
          // in the moment before the sheets arrive would keep its vector fallback until
          // something else happened to re-render it. Draw again once they are here.
          if (document.getElementById('screens')) render();
        });
      });
    });
  }
  App.livePanel = window.Panels.live();
  App.mode = 'play';           // 'play' shows the live decision, 'train' the explainer

  /* ---------------- helpers ---------------- */

  function el(id) { return document.getElementById(id); }
  function pct(v) { return (v === undefined || v === null || isNaN(v)) ? '—' : Math.round(v * 100) + '%'; }
  function esc(s) { return String(s === undefined ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  function view(key) { return window.Reveal.view(ALGOS[key]); }

  /* Centre by translating half the canvas back before scaling, rather than relying on
     flex/grid centring — a 1920-wide child inside a narrower box gets clipped at the
     start edge, which silently cuts the left of every frame. */
  var _fitW = 0, _fitH = 0;
  function fitStage(force) {
    var r = el('root');
    if (!r) return;
    var w = window.innerWidth, h = window.innerHeight;
    if (!force && w === _fitW && h === _fitH) return;
    _fitW = w; _fitH = h;
    var s = Math.min(w / 1920, h / 1080);
    r.style.transform = 'translate(' + (-960 * s) + 'px,' + (-540 * s) + 'px) scale(' + s + ')';
  }
  // Checked every frame rather than only on `resize`. A window that changes size while
  // the page is hidden, or a tab restored at a different size, never fires the event —
  // and the result is a frame recorded at the wrong scale, which is not recoverable in
  // the edit. The check is two integer comparisons.
  window.addEventListener('resize', function () { fitStage(); });

  function backdrop(name, opacity) {
    return '<div class="backdrop" style="background-image:url(assets/' + name + '.png);opacity:' + (opacity || 0.5) + '"></div>'
      + '<div class="veil"></div><div class="grid-lines"></div>';
  }

  /* An open socket is not a running simulation. The chip used to read TRAINER LIVE off
     `readyState` alone, so a stalled clock — a crashed pump, a paused replay, a trainer
     mid-garbage-collection — looked identical to a healthy one while the arena sat still.
     It now says how long it has been since a frame arrived, but only while something is
     supposed to be moving: on the menu, with the session idle, silence is correct. */
  var RUNNING = { play: 1, final: 1, race: 1, train: 1 };

  function linkState() {
    if (App.link !== 'live') return App.link === 'connecting'
      ? { cls: 'link-wait', txt: 'CONNECTING' } : { cls: 'link-off', txt: 'TRAINER OFFLINE' };
    var running = App.runState && RUNNING[App.runState.mode] && App.playing;
    var age = App.lastFrameAt ? (performance.now() - App.lastFrameAt) / 1000 : 1e9;
    if (running && age > 2) {
      return { cls: 'link-wait', txt: 'NO FRAMES ' + Math.min(99, Math.round(age)) + 's' };
    }
    return { cls: 'link-live', txt: App.replay ? 'REPLAY' : 'TRAINER LIVE' };
  }

  function statusChip() {
    var st = linkState();
    return '<span class="chip" id="linkchip"><span class="link-dot ' + st.cls + '"></span>' + st.txt + '</span>';
  }

  /* Patched in place from the animation loop rather than by re-rendering: a stalled
     stream is exactly the case where no message arrives to trigger a render. */
  var _linkTxt = null;
  function refreshStatus() {
    var el2 = el('linkchip');
    if (!el2) { _linkTxt = null; return; }
    var st = linkState();
    if (st.txt === _linkTxt) return;
    _linkTxt = st.txt;
    el2.innerHTML = '<span class="link-dot ' + st.cls + '"></span>' + st.txt;
  }

  function revealChip() {
    var lv = window.Reveal.level;
    return '<span class="chip" style="border-color:rgba(255,209,102,.3);color:#ffd166">REVEAL ' + (lv + 1) + ' / '
      + (window.Reveal.max + 1) + ' &nbsp;·&nbsp; R</span>';
  }

  /* ---------------- screens ---------------- */

  function renderMenu() {
    var ready = App.levels.length >= 12;
    var cards = ORDER.map(function (k, i) {
      var v = view(k);
      var prog = (App.cat && App.cat.tournament) ? App.cat.tournament : null;
      var anchor = prog && prog.anchor && prog.anchor[k];
      // A school the run does not contain must say so rather than look ready and then
      // fail on click — during a shoot that reads as a bug in the demo.
      var missing = !!(App.cat && App.cat.schools && App.cat.schools.indexOf(k) < 0);
      var art = v.sealed
        ? '<div class="sealed-art" style="position:absolute;inset:0;background-image:url(assets/school-' + k + '.png);background-size:contain;background-position:center;background-repeat:no-repeat"></div>'
          + '<div class="sealed-plate"><div class="sealed-tag">CLASSIFIED</div></div>'
        : '<div style="position:absolute;inset:0;background-image:url(assets/school-' + k + '.png);background-size:contain;background-position:center;background-repeat:no-repeat"></div>'
          + '<div style="position:absolute;inset:0;background:radial-gradient(70% 78% at 50% 46%,'
          + P.rgba(v.color, .22) + ' 0%,' + P.rgba(v.color, .06) + ' 45%,rgba(0,0,0,0) 78%);mix-blend-mode:screen"></div>';
      return '<div class="card"' + (missing ? '' : ' data-school="' + k + '"')
        + ' style="flex:1;display:flex;flex-direction:column;border-color:'
        + P.rgba(v.color, v.sealed ? .18 : .34) + ';cursor:' + (missing ? 'default' : 'pointer')
        + ';opacity:' + (missing ? '.45' : '1') + '">'
        + '<div style="height:4px;background:' + v.color + ';opacity:' + (v.sealed ? .3 : .9) + '"></div>'
        + '<div style="display:flex;justify-content:space-between;padding:14px 18px 0">'
        + '<span class="chip">SCHOOL 0' + (i + 1) + '</span>'
        + '<span class="chip" style="border-color:' + P.rgba(v.color, .32) + ';color:' + v.light + '">'
        + (v.sealed ? 'SEALED' : missing ? 'NOT IN THIS RUN' : anchor ? 'GRADUATED' : 'READY') + '</span></div>'
        + '<div style="position:relative;height:250px;margin:8px 0">' + art + '</div>'
        + '<div style="padding:0 22px 20px;display:flex;flex-direction:column;gap:12px;flex:1">'
        + '<div style="display:flex;align-items:baseline;gap:12px">'
        + '<span class="title" style="font-size:44px;color:' + (v.sealed ? '#8494ad' : v.color) + '">' + esc(v.short) + '</span>'
        + '<span class="' + (v.sealed ? 'scramble' : 'dim') + '" style="font-size:13px;font-weight:600">' + esc(v.full) + '</span></div>'
        + '<div class="dim" style="font-size:14px;line-height:1.5;min-height:63px">' + esc(v.blurb) + '</div>'
        + '<div style="display:flex;flex-direction:column;gap:6px">'
        + v.specs.map(function (r) {
            return '<div style="display:flex;gap:14px;font-size:12px"><span class="faint mono" style="width:96px;letter-spacing:1px">'
              + esc(r[0]) + '</span><span class="mono" style="color:#c9d8ee">' + esc(r[1]) + '</span></div>';
          }).join('')
        + '</div>'
        + '<div style="display:flex;gap:10px;margin-top:auto">'
        // Name the measurement. Bare "TOM 42%" on the opening screen reads as this
        // school's headline number, and the leaderboard two screens later says 74% —
        // the same cat, scored against the learned field instead of the Examiner.
        + scoreBox('TOM · vs EXAMINER', anchor ? pct(anchor.cat.catch) : '—', 'var(--cat)')
        + scoreBox('JERRY · vs EXAMINER', anchor ? pct(anchor.mouse.escape) : '—', 'var(--mouse)')
        + '</div></div></div>';
    }).join('');

    return '<div class="screen">' + backdrop('bg-academy', .55)
      + '<div style="position:relative;display:flex;justify-content:space-between;align-items:flex-start">'
      + '<div><div class="kicker">Reinforcement learning · pursuit and evasion</div>'
      + '<div class="title" style="font-size:62px;margin-top:6px">CAT &amp; MOUSE ACADEMY</div></div>'
      + '<div style="display:flex;gap:10px;align-items:center">' + revealChip() + statusChip() + '</div></div>'
      + '<div style="position:relative;display:flex;gap:18px;margin-top:18px;flex:1">' + cards + '</div>'
      + '<div style="position:relative;display:flex;gap:12px;align-items:center;margin-top:16px">'
      + '<div class="btn" data-act="gen">' + (ready ? 'NEW LEVEL SET' : 'GENERATE THE LEVELS') + '</div>'
      + '<div class="btn ghost" data-act="board">LEADERBOARD</div>'
      + '<div class="btn ghost" data-act="race">SIDE BY SIDE · X</div>'
      + '<div class="btn ghost" data-act="final">GRAND FINAL</div>'
      + (App.cat && App.cat.highlights && App.cat.highlights.highlights.length
          ? '<div class="btn ghost" data-act="highlights">HIGHLIGHTS · H</div>' : '')
      + '<div class="dim" style="margin-left:auto;font-size:13px">'
      + (ready ? App.levels.length + ' arenas ready · every school trains on the same rooms'
               : 'Generate the shared level set first') + '</div></div></div>';
  }

  function scoreBox(label, value, color) {
    return '<div class="card" style="flex:1;padding:10px 14px;border-radius:10px">'
      + '<div style="width:16px;height:2px;background:' + color + ';margin-bottom:8px"></div>'
      + '<div class="mono" style="font-size:20px">' + value + '</div>'
      + '<div class="faint" style="font-size:10px;letter-spacing:1.6px;margin-top:2px">' + label + '</div></div>';
  }

  function renderGen() {
    var slots = [];
    for (var i = 0; i < 12; i++) {
      var lv = App.levels[i];
      var seeding = !lv && i === App.levels.length;
      slots.push('<div class="card" style="padding:8px;opacity:' + (lv || seeding ? 1 : .45)
        + ';border-color:' + (seeding ? 'rgba(126,232,255,.4)' : 'var(--line)') + '">'
        + '<div style="display:flex;justify-content:space-between;font-size:10px" class="mono">'
        + '<span style="color:' + (lv ? '#c9d8ee' : (seeding ? '#7ee0ff' : '#4f6280')) + '">LEVEL ' + String(i + 1).padStart(2, '0') + '</span>'
        + '<span class="faint">' + (lv ? '#' + (lv.seed % 100000) : (seeding ? 'seeding' : 'queued')) + '</span></div>'
        + '<div id="lv' + i + '" style="margin:6px 0 4px;position:relative;overflow:hidden">'
        + (seeding ? '<div style="position:absolute;left:0;right:0;height:2px;background:#7ee0ff;animation:scan 1.1s linear infinite"></div>' : '')
        + '</div>'
        + '<div class="faint" style="font-size:10px">' + (lv ? lv.route + ' cells · ' + lv.onRoute + ' traps on route' : '&nbsp;') + '</div></div>');
    }
    /* Ranges are read off the rooms actually on screen rather than written down. The
       old copy quoted a fixed 21-32 cell spawn, which was a one-hole rule: with two
       holes the longest possible trek is roughly halved, so ten of the twelve tiles
       beside it printed a number the sentence said was impossible. */
    function span(pick) {
      if (!App.levels.length) return null;
      var lo = Infinity, hi = -Infinity;
      App.levels.forEach(function (l) { var v = pick(l); if (v < lo) lo = v; if (v > hi) hi = v; });
      return lo === hi ? String(lo) : lo + '–' + hi;
    }
    var trek = span(function (l) { return l.route; });
    var onRt = span(function (l) { return l.onRoute; });

    var rules = [
      ['ONE ROOM, MANY SHAPES', 'Seven cover blocks and ten pillars, reseeded until 94% of the floor is one connected region.'],
      ['TWO WAYS HOME', 'Every hole is placed so that walling off any one of its neighbours still leaves it reachable — it always has two independent approaches, so camping it can never be unbeatable.'],
      ['TRAPS ON THE WALKED PATH', 'Up to six traps a room: the first three go on her shortest route, the next two flank it, and a room with fewer than four is reseeded — a hazard off the path is never learned.'
        + (onRt ? ' On this level set, ' + onRt + ' of them sit on the route.' : '')],
      ['MORE THAN ONE WAY OUT', 'With a single hole the cat just stands on it. Every extra hole is kept at least 10 cells from the others, so he cannot cover two: she gets a choice and he has to guess.'],
      ['A REAL CHANCE FOR BOTH', 'She spawns in the farthest slice of the room — at least 72% of that map\'s own longest trek from a hole'
        + (trek ? ', which is ' + trek + ' cells here' : '')
        + '. He spawns at least 10 cells from her, and himself 4 to 85% of that trek from a hole, so camping is available to him but not free.'],
      ['SHARED BY EVERY SCHOOL', 'The same seeds train all three. Nobody gets to memorise one room.']
    ].map(function (r) {
      return '<div style="margin-bottom:16px"><div class="mono" style="font-size:11px;letter-spacing:1.6px;color:#c9d8ee">' + r[0] + '</div>'
        + '<div class="faint" style="font-size:12.5px;line-height:1.5;margin-top:4px">' + r[1] + '</div></div>';
    }).join('');

    return '<div class="screen">' + backdrop('bg-academy', .35)
      + '<div style="position:relative;display:flex;justify-content:space-between;align-items:flex-start">'
      + '<div><div class="kicker">Shared level set</div>'
      + '<div class="title" style="font-size:46px;margin-top:6px">THE ARENAS</div></div>'
      + '<div style="display:flex;gap:10px">' + revealChip() + statusChip() + '</div></div>'
      + '<div style="position:relative;display:flex;gap:22px;margin-top:16px;flex:1">'
      + '<div style="flex:1;display:grid;grid-template-columns:repeat(4,1fr);grid-auto-rows:min-content;gap:12px">' + slots.join('') + '</div>'
      + '<div class="card" style="width:400px;padding:22px">' + rules + '</div></div>'
      + '<div style="position:relative;display:flex;gap:12px;margin-top:14px;align-items:center">'
      + '<div class="btn" data-act="menu">BACK TO THE ACADEMY</div>'
      + '<div class="dim" style="font-size:13px">' + (App.levels.length >= 12
        ? 'Level set complete — ' + App.levels.reduce(function (a, l) { return a + l.onRoute; }, 0) + ' traps sitting on a walked route.'
        : !App.cat
          ? 'Waiting for the trainer — the seeds and the hole count belong to the run, so there is nothing to draw yet.'
          : 'Building arena ' + Math.min(12, App.levels.length + 1) + ' of 12.') + '</div></div></div>';
  }

  function renderSchool() {
    var v = view(App.school);
    var accent = v.color;
    // During the highlight reel the run is the reel, not the twelve-arena level set:
    // ten episodes counted out of twelve, under LV01-LV12 labels that name rooms the
    // reel is not playing, is two wrong numbers in the same corner of the screen.
    var reel = App.highlights;
    var n = reel ? reel.length : (App.cat ? App.cat.levels.length : 12);
    var st = App.runState || {};
    var done = App.results.filter(Boolean);
    var catch_ = done.filter(function (r) { return r === 'catch'; }).length;
    var esc_ = done.filter(function (r) { return r === 'escape'; }).length;

    var pills = CP.map(function (c) {
      var on = App.checkpoint === c;
      return '<div class="chip" data-cp="' + c + '" style="cursor:pointer;padding:8px 14px;'
        + (on ? 'background:' + P.rgba(accent, .18) + ';border-color:' + P.rgba(accent, .45) + ';color:#dceaff' : '') + '">'
        + CP_NAME[c] + '</div>';
    }).join('');

    var strip = [];
    for (var i = 0; i < n; i++) {
      var r = App.results[i];
      var live = st.level === i && App.screen === 'school';
      strip.push('<div style="flex:1;text-align:center;padding:5px 0;border-radius:6px;border:1px solid '
        + (r === 'catch' ? 'rgba(255,138,92,.36)' : r === 'escape' ? 'rgba(126,224,255,.32)' : r ? 'var(--line)' : (live ? 'rgba(124,188,255,.5)' : 'rgba(130,160,200,.1)'))
        + ';background:' + (r === 'catch' ? 'rgba(255,122,84,.14)' : r === 'escape' ? 'rgba(110,226,255,.12)' : r ? 'rgba(255,255,255,.04)' : (live ? 'rgba(124,188,255,.16)' : 'rgba(255,255,255,.02)'))
        + '"><div class="mono" style="font-size:9px;color:var(--faint)">LV'
        + String((reel ? reel[i].arena : i) + 1).padStart(2, '0') + '</div>'
        + '<div class="mono" style="font-size:13px;color:'
        + (r === 'catch' ? '#ff9a72' : r === 'escape' ? '#7ee0ff' : r ? '#8fa4c4' : (live ? '#f2f7ff' : '#3d4a60')) + '">'
        + (r === 'catch' ? 'T' : r === 'escape' ? 'J' : r ? '–' : (live ? '▸' : '·')) + '</div></div>');
    }

    var f = App.frame;
    var mode = f ? (f.cat.mode + ' · ' + f.mouse.mode) : '—';
    var hl = App.highlights && App.highlights[st.level || 0];

    return '<div class="screen">' + backdrop('bg-' + (v.sealed ? 'academy' : App.school), .32)
      + '<div style="position:relative;display:flex;align-items:center;gap:18px">'
      + '<div style="width:46px;height:46px;opacity:' + (v.sealed ? .5 : 1) + '">' + P.emblem(v.emblem, accent) + '</div>'
      + '<div><div class="title" style="font-size:34px;color:' + (v.sealed ? '#8494ad' : '#f2f7ff') + '">'
      + esc(v.short) + ' SCHOOL</div><div class="' + (v.sealed ? 'scramble' : 'dim') + '" style="font-size:12px">' + esc(v.line) + '</div></div>'
      + '<div style="display:flex;gap:8px;margin-left:24px">' + pills + '</div>'
      // Hidden while the school is sealed, for the same reason the panel is: the
      // explainer names the method on its first line, so there must be no door to it.
      + (v.sealed ? '' : '<div data-act="x-open" style="cursor:pointer;margin-left:14px;height:36px;padding:0 15px;display:flex;align-items:center;gap:9px;border-radius:10px;border:1px solid '
        + P.rgba(accent, .34) + ';background:' + v.deep + '">'
        + '<div style="width:17px;height:17px;border-radius:50%;border:1.5px solid ' + accent
        + ';display:flex;align-items:center;justify-content:center;font-family:var(--display);font-size:11px;line-height:1;color:' + accent + '">?</div>'
        + '<span style="font-family:var(--display);font-size:11.5px;letter-spacing:1.6px;color:' + v.light + '">HOW IT WORKS · W</span></div>')
      + '<div style="margin-left:auto;display:flex;gap:10px;align-items:center">'
      + '<span class="chip">' + App.speed + '× &nbsp;[ ]</span>' + revealChip() + statusChip() + '</div></div>'

      + '<div style="position:relative;display:flex;gap:18px;margin-top:14px;flex:1">'
      + '<div style="position:relative;width:' + (E.W * CS) + 'px;flex:0 0 auto">'
      + '<div id="arena" style="position:relative;width:' + (E.W * CS) + 'px;height:' + (E.H * CS) + 'px;border-radius:12px;overflow:hidden;border:1px solid var(--line)">'
      + '<div id="arena-map" style="position:absolute;inset:0"></div>'
      + '<div id="arena-fx" style="position:absolute;inset:0"></div>'
      + '<div id="arena-banner"></div></div>'
      + '<div style="display:flex;gap:4px;margin-top:10px">' + strip.join('') + '</div></div>'

      + '<div style="flex:1;display:flex;flex-direction:column;gap:12px;min-width:0">'
      + '<div style="display:flex;gap:12px">'
      + scoreBox('TOM · CAUGHT HER', catch_ + ' / ' + n, 'var(--cat)')
      + scoreBox('JERRY · GOT HOME', esc_ + ' / ' + n, 'var(--mouse)')
      + scoreBox('ARENA', ((st.level || 0) + 1) + ' / ' + n, 'var(--gold)') + '</div>'
      + '<div class="card" style="padding:14px 18px">'
      + '<div class="faint mono" style="font-size:10px;letter-spacing:1.6px">WHAT THEY CAN SENSE RIGHT NOW</div>'
      + '<div id="thought" class="mono" style="font-size:15px;margin-top:6px;color:#c9d8ee">' + esc(mode) + '</div></div>'
      + '<div class="card" style="flex:1;padding:6px;position:relative;min-height:0">'
      + (v.sealed
        ? '<div class="sealed-plate" style="border-radius:12px"><div style="text-align:center"><div class="sealed-tag">METHOD CLASSIFIED</div>'
          + '<div class="faint" style="font-size:12px;margin-top:12px">The explainer for this school is sealed until it is introduced.</div></div></div>'
        : '<div id="panel" style="width:100%;height:100%"></div>')
      + '</div></div></div>'

      + '<div style="position:relative;display:flex;gap:10px;margin-top:12px;align-items:center">'
      + '<div class="btn ghost" data-act="menu">ESC · ACADEMY</div>'
      + '<div class="btn ghost" data-act="pause">' + (App.playing ? 'PAUSE · SPACE' : 'RESUME · SPACE') + '</div>'
      + '<div class="btn ghost" data-act="skip">SKIP EPISODE · S</div>'
      + '<div class="btn ghost" data-act="train">TRAIN LIVE · T</div>'
      + '<div class="dim" style="margin-left:auto;font-size:12px;text-align:right">'
      + (hl ? '<span class="mono" style="color:var(--gold);letter-spacing:2px">' + esc(hl.kind.toUpperCase())
              + '</span> &nbsp;·&nbsp; ' + esc(hl.why) + '<br>' : '')
      // Name the checkpoint that is actually playing. SHOOT.md sends the author through
      // UNTRAINED and HALF-TRAINED on camera, and the old line said "trained" over both.
      + (App.trainInfo ? esc(App.trainInfo)
         : App.link !== 'live' ? 'Trainer offline — this is the last frame it sent.'
         : !App.frame ? 'Waiting for the first frame from the trainer.'
         : 'Frames are streaming from the ' + CP_NAME[App.checkpoint].toLowerCase()
           + ' policy — nothing here is scripted.')
      + '</div></div></div>';
  }

  function renderFinal() {
    var t = App.cat && App.cat.tournament;
    var st = App.runState || {};
    // Champion versus champion is a claim about a tournament. Without one there is no
    // pairing to show, and defaulting to PPO-versus-PPO would put two invented finalists
    // under a headline saying they earned it.
    if (!t && !st.catSchool) {
      return '<div class="screen">' + backdrop('bg-final', .35)
        + '<div style="position:relative"><div class="kicker">Champion versus champion</div>'
        + '<div class="title" style="font-size:48px;margin-top:6px">THE GRAND FINAL</div>'
        + '<div class="dim" style="margin-top:14px;font-size:15px;max-width:40em;line-height:1.6">'
        + 'There are no champions yet — the final is whoever wins the cross-play tournament, '
        + 'and it has not been run for this training run.<br>'
        + 'Run <span class="mono">python trainer/scripts/tournament_run.py --run runs/v4</span> and reload.</div>'
        + '<div class="btn ghost" style="margin-top:24px" data-act="menu">ESC · ACADEMY</div></div></div>';
    }
    var ck = st.catSchool || (t && t.champion.cat) || 'ppo';
    var mk = st.mouseSchool || (t && t.champion.mouse) || 'ppo';
    var cv = view(ck), mv = view(mk);
    var wins = st.wins || { cat: 0, mouse: 0, draw: 0 };
    return '<div class="screen">' + backdrop('bg-final', .5)
      + '<div style="position:relative;display:flex;justify-content:space-between;align-items:flex-start">'
      + '<div><div class="kicker">Champion versus champion · an arena neither has seen</div>'
      + '<div class="title" style="font-size:50px;margin-top:6px">THE GRAND FINAL</div></div>'
      + '<div style="display:flex;gap:10px">' + revealChip() + statusChip() + '</div></div>'
      + '<div style="position:relative;display:flex;gap:20px;margin-top:14px;flex:1;align-items:flex-start">'
      + finalist('TOM', cv, wins.cat, 'cat')
      + '<div style="position:relative;width:' + (E.W * CS) + 'px;flex:0 0 auto">'
      + '<div id="arena" style="position:relative;width:' + (E.W * CS) + 'px;height:' + (E.H * CS) + 'px;border-radius:12px;overflow:hidden;border:1px solid var(--line)">'
      + '<div id="arena-map" style="position:absolute;inset:0"></div>'
      + '<div id="arena-fx" style="position:absolute;inset:0"></div>'
      + '<div id="arena-banner"></div></div>'
      + '<div class="dim" style="text-align:center;margin-top:10px;font-size:13px">Round ' + ((st.level || 0) + 1)
      + ' of ' + (st.levels || 5) + ' &nbsp;·&nbsp; draws ' + wins.draw + '</div></div>'
      + finalist('JERRY', mv, wins.mouse, 'mouse')
      + '</div>'
      + '<div style="position:relative;display:flex;gap:10px;margin-top:10px">'
      + '<div class="btn ghost" data-act="menu">ESC · ACADEMY</div>'
      + '<div class="btn ghost" data-act="pause">' + (App.playing ? 'PAUSE' : 'RESUME') + '</div>'
      + '<div class="btn ghost" data-act="board">LEADERBOARD · B</div></div></div>';
  }

  function finalist(name, v, wins, role) {
    return '<div class="card" style="flex:1;padding:20px;display:flex;flex-direction:column;gap:12px;border-color:' + P.rgba(v.color, .3) + '">'
      + '<div style="display:flex;align-items:center;gap:12px">'
      + '<div style="width:34px;height:34px">' + P.emblem(v.emblem, v.color) + '</div>'
      + '<div><div class="title" style="font-size:30px;color:' + (role === 'cat' ? 'var(--cat)' : 'var(--mouse)') + '">' + name + '</div>'
      + '<div class="' + (v.sealed ? 'scramble' : 'dim') + '" style="font-size:12px">' + esc(v.short) + ' school</div></div></div>'
      + '<div style="height:190px">' + P.portrait(role, v.color, 130) + '</div>'
      + '<div class="mono" style="font-size:44px;text-align:center">' + wins + '</div>'
      + '<div class="faint" style="text-align:center;font-size:10px;letter-spacing:2px">ROUNDS WON</div></div>';
  }

  /* A full-screen version of the same explainer, for talking over on camera. The panel
     is identical — only bigger — so what is being explained is exactly what was on
     screen a moment ago beside the arena, rather than a separate illustration. */
  var LESSON = {
    ppo: {
      head: 'It takes a step, then refuses to take a big one',
      body: 'PPO collects a batch of episodes, works out which moves did better than '
        + 'expected, and nudges the policy toward them. The trick is the leash: the new '
        + 'policy is scored against the old one, and any move that would change too much '
        + 'is clipped back. The histogram is that leash — almost everything piles up '
        + 'inside the band, because that is the only place the update is allowed to go.',
      watch: ['The bars barely move between updates. That is the point, not a fault.',
              'Anything outside the dashed lines got clipped and contributed nothing.',
              'Entropy falling means it is running out of doubt about what to do.']
    },
    ga: {
      head: 'Forty-eight brains, and only the good ones get children',
      body: 'No gradients, no derivatives, nothing that knows which direction is better. '
        + 'Each generation plays all forty-eight networks, ranks them, keeps the best few '
        + 'untouched, and fills the rest with children: pick two parents, flip a coin per '
        + 'weight to decide which parent it comes from, then nudge a tenth of the weights '
        + 'at random. Repeat a few thousand times.',
      watch: ['Lit borders are the elites — they survive to the next generation unchanged.',
              'Each strip is a fingerprint of one brain. Children look like their parents.',
              'Sigma hunts up and down: it grows while children keep beating parents.']
    },
    cmaes: {
      head: 'It does not just search — it learns the shape of the search',
      body: 'CMA-ES keeps a Gaussian over strategies. Every generation it draws thirty-two '
        + 'brains from it, keeps the better half, and moves the centre toward them — then '
        + 'reshapes the cloud itself, stretching along directions that keep paying off and '
        + 'narrowing where they do not. The ellipse is that shape, measured from the real '
        + 'samples on screen.',
      watch: ['Filled dots are the half that gets kept and recombined.',
              'The gold marker is where the centre moved to this generation.',
              'Sigma shrinking means it has stopped exploring and started converging.']
    }
  };

  /* Three schools, one room. The spec calls this the whole point: identical map,
     identical spawns, identical noise, so the only thing that can differ is the brain.
     Three panes at cell size 22 fit the 1920 canvas with room for the scoreboard. */
  var RCS = 22;

  function renderRace() {
    var schools = App.raceSchools || ORDER;
    var wins = App.raceWins || {};
    var lanes = schools.map(function (k) {
      var v = view(k);
      var w = wins[k] || { catch: 0, escape: 0, draw: 0 };
      var done = App.raceDone && App.raceDone[k];
      return '<div class="card" style="flex:1;padding:12px;border-color:' + P.rgba(v.color, .3) + '">'
        + '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">'
        + '<div style="width:26px;height:26px;flex:0 0 auto">' + P.emblem(v.emblem, v.color) + '</div>'
        + '<div class="title" style="font-size:22px;color:' + (v.sealed ? '#8494ad' : v.color) + '">' + esc(v.short) + '</div>'
        + '<div style="margin-left:auto;display:flex;gap:8px">'
        + '<span class="mono" style="font-size:15px;color:var(--cat)">' + w.catch + '</span>'
        + '<span class="faint">/</span>'
        + '<span class="mono" style="font-size:15px;color:var(--mouse)">' + w.escape + '</span>'
        + '<span class="faint mono" style="font-size:12px">' + w.draw + '</span></div></div>'
        + '<div style="position:relative;width:' + (E.W * RCS) + 'px;height:' + (E.H * RCS) + 'px;border-radius:9px;overflow:hidden;border:1px solid var(--line)">'
        + '<div id="rmap-' + k + '" style="position:absolute;inset:0"></div>'
        + '<div id="rfx-' + k + '" style="position:absolute;inset:0"></div>'
        + (done ? '<div style="position:absolute;inset:0;display:grid;place-items:center;background:rgba(4,7,12,.55)">'
            + '<div class="title" style="font-size:22px;letter-spacing:2px;color:'
            + (done === 'catch' ? 'var(--cat)' : done === 'escape' ? 'var(--gold)' : '#8fa4c4') + '">'
            + (done === 'catch' ? 'TOM' : done === 'escape' ? 'JERRY' : 'TIME') + '</div></div>' : '')
        + '</div></div>';
    }).join('');

    return '<div class="screen">' + backdrop('bg-final', .3)
      + '<div style="position:relative;display:flex;justify-content:space-between;align-items:flex-start">'
      + '<div><div class="kicker">Same room · same spawns · same noise · three different brains</div>'
      + '<div class="title" style="font-size:44px;margin-top:4px">SIDE BY SIDE</div></div>'
      + '<div style="display:flex;gap:10px">' + revealChip() + statusChip() + '</div></div>'
      + '<div style="position:relative;display:flex;gap:14px;margin-top:14px">' + lanes + '</div>'
      + raceMatrix(schools)
      + '<div style="position:relative;display:flex;gap:10px;margin-top:auto;align-items:center">'
      + '<div class="btn ghost" data-act="menu">ESC · ACADEMY</div>'
      + '<div class="btn ghost" data-act="pause">' + (App.playing ? 'PAUSE · SPACE' : 'RESUME · SPACE') + '</div>'
      + '<div class="btn ghost" data-act="race">RESTART · X</div>'
      + '<div class="dim" style="margin-left:auto;font-size:12.5px">'
      + 'Arena ' + (((App.runState || {}).level || 0) + 1) + ' of ' + (App.cat ? App.cat.levels.length : 12)
      + ' &nbsp;·&nbsp; nothing differs between the three panes except the policy.</div></div></div>';
  }

  /* The same twelve rooms, three schools, one grid. Reading down a column tells you
     whether a room is hard or whether one school simply solved it — which is the
     comparison the side-by-side view exists to make. */
  function raceMatrix(schools) {
    var n = App.cat ? App.cat.levels.length : 12;
    var grid = App.raceGrid || {};
    var head = '<div style="display:flex;gap:4px;margin-left:104px">';
    for (var i = 0; i < n; i++) {
      head += '<div class="mono faint" style="flex:1;text-align:center;font-size:9px">'
        + String(i + 1).padStart(2, '0') + '</div>';
    }
    head += '</div>';

    var rows = schools.map(function (k) {
      var v = view(k);
      var r = grid[k] || [];
      var c = r.filter(function (x) { return x === 'catch'; }).length;
      var m = r.filter(function (x) { return x === 'escape'; }).length;
      var cells = '';
      for (var i = 0; i < n; i++) {
        var x = r[i];
        cells += '<div style="flex:1;text-align:center;padding:7px 0;border-radius:5px;border:1px solid '
          + (x === 'catch' ? 'rgba(255,138,92,.36)' : x === 'escape' ? 'rgba(126,224,255,.32)'
             : x ? 'var(--line)' : 'rgba(130,160,200,.09)')
          + ';background:' + (x === 'catch' ? 'rgba(255,122,84,.14)' : x === 'escape' ? 'rgba(110,226,255,.12)'
             : x ? 'rgba(255,255,255,.03)' : 'transparent')
          + '"><span class="mono" style="font-size:12px;color:'
          + (x === 'catch' ? '#ff9a72' : x === 'escape' ? '#7ee0ff' : x ? '#8fa4c4' : '#31405a')
          + '">' + (x === 'catch' ? 'T' : x === 'escape' ? 'J' : x ? '–' : '·') + '</span></div>';
      }
      return '<div style="display:flex;align-items:center;gap:4px;margin-top:6px">'
        + '<div class="mono" style="width:100px;font-size:12px;color:' + (v.sealed ? '#8494ad' : v.color) + '">'
        + esc(v.short) + '</div>' + cells
        + '<div class="mono" style="width:78px;text-align:right;font-size:12px">'
        + '<span style="color:var(--cat)">' + c + '</span><span class="faint">/</span>'
        + '<span style="color:var(--mouse)">' + m + '</span></div></div>';
    }).join('');

    return '<div class="card" style="position:relative;margin-top:16px;padding:16px 20px">'
      + '<div class="mono" style="font-size:10.5px;letter-spacing:1.6px;color:#c9d8ee;margin-bottom:10px">'
      + 'THE SAME TWELVE ROOMS &nbsp;·&nbsp; T = TOM CAUGHT HER &nbsp; J = JERRY GOT HOME &nbsp; – = TIME</div>'
      + head + rows + '</div>';
  }

  function paintRace(now) {
    if (!App.raceFrames || !App.map) return;
    var schools = App.raceSchools || ORDER;
    var local = localMap(App.map);
    for (var i = 0; i < schools.length; i++) {
      var k = schools[i];
      var mh = el('rmap-' + k), fh = el('rfx-' + k);
      if (!mh || !fh) continue;
      if (mh.getAttribute('data-k') !== String(App.map.seed)) {
        mh.setAttribute('data-k', String(App.map.seed));
        mh.innerHTML = P.mapSvg(local, RCS);
      }
      var v = view(k);
      var fr = App.raceFrames[k], pv = (App.racePrev && App.racePrev[k]) || fr;
      if (!fr) continue;
      fh.innerHTML = P.fxSvg({
        frame: fr, prev: pv, alpha: App.alpha, cs: RCS, map: local,
        key: 'r' + k, now: now, catAccent: v.color, mouseAccent: v.color,
        sprites: App.sprites, spriteFps: App.spriteFps, holdSteps: E.CFG.freezeSteps,
        catMoving: App.alpha < 1 && (fr.cat.x !== pv.cat.x || fr.cat.y !== pv.cat.y),
        mouseMoving: App.alpha < 1 && (fr.mouse.x !== pv.mouse.x || fr.mouse.y !== pv.mouse.y)
      });
    }
  }

  function renderLesson() {
    var v = view(App.school);
    var L = LESSON[App.school];
    if (v.sealed) {
      return '<div class="screen">' + backdrop('bg-academy', .4)
        + '<div style="position:relative;flex:1;display:grid;place-items:center">'
        + '<div style="text-align:center"><div class="sealed-tag">METHOD CLASSIFIED</div>'
        + '<div class="dim" style="margin-top:18px;font-size:15px">This school has not been introduced yet.</div>'
        + '<div class="btn ghost" style="margin-top:26px" data-act="menu">ESC · ACADEMY</div></div></div></div>';
    }
    return '<div class="screen">' + backdrop('bg-' + App.school, .3)
      + '<div style="position:relative;display:flex;align-items:center;gap:18px">'
      + '<div style="width:54px;height:54px">' + P.emblem(v.emblem, v.color) + '</div>'
      + '<div><div class="kicker">How it learns</div>'
      + '<div class="title" style="font-size:44px;margin-top:4px;color:' + v.color + '">' + esc(v.short) + '</div></div>'
      + '<div style="margin-left:auto;display:flex;gap:10px">' + revealChip() + statusChip() + '</div></div>'
      + '<div style="position:relative;display:flex;gap:22px;margin-top:18px;flex:1;min-height:0">'
      + '<div class="card" style="flex:1;padding:10px"><div id="panel" style="width:100%;height:100%"></div></div>'
      + '<div style="width:520px;display:flex;flex-direction:column;gap:16px">'
      + '<div class="card" style="padding:24px">'
      + '<div class="title" style="font-size:26px;line-height:1.25">' + esc(L.head) + '</div>'
      + '<div class="dim" style="font-size:14.5px;line-height:1.65;margin-top:14px">' + esc(L.body) + '</div></div>'
      + '<div class="card" style="padding:22px 24px;flex:1">'
      + '<div class="mono faint" style="font-size:10.5px;letter-spacing:1.6px">WHAT TO WATCH</div>'
      + L.watch.map(function (t) {
          return '<div style="display:flex;gap:12px;margin-top:14px"><div style="width:7px;height:7px;border-radius:50%;background:'
            + v.color + ';margin-top:6px;flex:0 0 auto"></div><div class="dim" style="font-size:13.5px;line-height:1.55">'
            + esc(t) + '</div></div>';
        }).join('')
      + '</div></div></div>'
      + '<div style="position:relative;display:flex;gap:10px;margin-top:14px">'
      + '<div class="btn" data-act="school">BACK TO THE ARENA · L</div>'
      + '<div class="btn ghost" data-act="train">TRAIN LIVE · T</div>'
      + '<div class="btn ghost" data-act="menu">ESC · ACADEMY</div></div></div>';
  }

  function renderVerdict() {
    var v = view(App.school);
    // The reel is twelve rooms chosen FOR being dramatic. Scoring it like a straight run
    // and printing "TOM came out ahead" would be a tally of episodes picked to be close.
    var reel = App.highlights;
    var n = reel ? reel.length : (App.cat ? App.cat.levels.length : 12);
    var done = App.results.filter(Boolean);
    var c = done.filter(function (r) { return r === 'catch'; }).length;
    var m = done.filter(function (r) { return r === 'escape'; }).length;
    var d = done.length - c - m;
    var winner = c > m ? 'TOM' : (m > c ? 'JERRY' : 'SPLIT');
    var wcol = c > m ? 'var(--cat)' : (m > c ? 'var(--mouse)' : '#c9d8ee');

    // The three checkpoints, scored against the same fixed opponent. This is the curve
    // that actually shows a student improving — the head-to-head above does not,
    // because both sides moved at once.
    var prog = App.cat && App.cat.progression;
    var bars = '';
    if (prog) {
      bars = ['untrained', 'half', 'trained'].map(function (cp) {
        var row = prog[cp] && prog[cp][App.school];
        if (!row) return '';
        return '<div style="display:flex;align-items:center;gap:14px;margin-bottom:12px">'
          + '<div class="mono faint" style="width:120px;font-size:10.5px;letter-spacing:1.4px">' + CP_NAME[cp] + '</div>'
          + '<div style="flex:1">'
          + ciBar(row.cat, row.catLo, row.catHi, 'var(--cat)')
          + ciBar(row.mouse, row.mouseLo, row.mouseHi, 'var(--mouse)')
          + '</div></div>';
      }).join('');
    }

    var strip = App.results.slice(0, n).map(function (r, i) {
      return '<div style="flex:1;text-align:center;padding:8px 0;border-radius:7px;border:1px solid '
        + (r === 'catch' ? 'rgba(255,138,92,.36)' : r === 'escape' ? 'rgba(126,224,255,.32)' : 'var(--line)')
        + ';background:' + (r === 'catch' ? 'rgba(255,122,84,.14)' : r === 'escape' ? 'rgba(110,226,255,.12)' : 'rgba(255,255,255,.03)') + '">'
        + '<div class="mono faint" style="font-size:9px">LV'
        + String((reel ? reel[i].arena : i) + 1).padStart(2, '0') + '</div>'
        + '<div class="mono" style="font-size:16px;color:' + (r === 'catch' ? '#ff9a72' : r === 'escape' ? '#7ee0ff' : '#8fa4c4') + '">'
        + (r === 'catch' ? 'T' : r === 'escape' ? 'J' : '–') + '</div></div>';
    }).join('');

    return '<div class="screen">' + backdrop('bg-' + (v.sealed ? 'academy' : App.school), .4)
      + '<div style="position:relative;display:flex;justify-content:space-between;align-items:flex-start">'
      + '<div style="display:flex;align-items:center;gap:16px">'
      + '<div style="width:44px;height:44px">' + P.emblem(v.emblem, v.color) + '</div>'
      + '<div><div class="kicker">School verdict</div>'
      + '<div class="title" style="font-size:46px;margin-top:4px;color:' + (v.sealed ? '#8494ad' : '#f2f7ff') + '">'
      + esc(v.short) + ' · ' + winner + '</div></div></div>'
      + '<div style="display:flex;gap:10px">' + revealChip() + statusChip() + '</div></div>'
      + '<div style="position:relative;display:flex;gap:18px;margin-top:20px;flex:1">'
      + '<div class="card" style="flex:1;padding:24px;display:flex;flex-direction:column;gap:18px">'
      + '<div style="display:flex;gap:14px">'
      + scoreBox('TOM · CAUGHT HER', c + ' / ' + n, 'var(--cat)')
      + scoreBox('JERRY · GOT HOME', m + ' / ' + n, 'var(--mouse)')
      + scoreBox('RAN OUT OF TIME', String(d), '#8fa4c4') + '</div>'
      + '<div style="display:flex;gap:6px">' + strip + '</div>'
      + '<div class="dim" style="font-size:13.5px;line-height:1.6">'
      + (reel
         ? 'These are the <b>highlight reel</b>\u2019s ' + n + ' episodes \u2014 chosen for drama, one per room, '
           + 'so this tally is not a measurement of anything. Press <span class="mono">b</span> for the numbers that are. '
         : '')
      + 'This is the head-to-head on the shared level set: <b style="color:' + wcol + '">' + winner
      + '</b> came out ahead <i>inside this school</i>. It is not a ranking across schools — '
      + 'both sides here were raised together, so it says as much about the sparring partner as the student. '
      + 'The leaderboard settles that.</div>'
      + trapStory(prog)
      + '<div style="margin-top:auto;display:flex;gap:10px">'
      + '<div class="btn" data-act="board">SEE THE LEADERBOARD · B</div>'
      + '<div class="btn ghost" data-act="menu">BACK TO THE ACADEMY</div></div></div>'
      + '<div class="card" style="width:520px;padding:24px">'
      + '<div class="mono" style="font-size:11px;letter-spacing:1.6px;color:#c9d8ee">AGAINST A FIXED OPPONENT, AT EACH CHECKPOINT</div>'
      + '<div class="faint" style="font-size:11.5px;line-height:1.5;margin:8px 0 18px">'
      + 'The same Examiner, at a difficulty this school never trained on. Only one side of the room moves, '
      + 'so this curve measures the student rather than the matchup.</div>'
      + (bars || '<div class="faint" style="font-size:12px">Run the tournament to fill this in.</div>')
      + '<div style="display:flex;gap:16px;margin-top:20px">'
      + '<span class="chip" style="border-color:rgba(255,138,92,.32);color:#ff9a72">TOM · CATCH RATE</span>'
      + '<span class="chip" style="border-color:rgba(126,224,255,.32);color:#7ee0ff">JERRY · ESCAPE RATE</span></div>'
      + '</div></div></div>';
  }

  /* Trap hits per episode, checkpoint by checkpoint. Caution is not a rule in this
     environment — it is learned, weighted by competence — so this curve falling is the
     most legible evidence on screen that something was actually learned. */
  function trapStory(prog) {
    if (!prog) return '';
    var rows = ['untrained', 'half', 'trained'].map(function (cp) {
      return { cp: cp, row: prog[cp] && prog[cp][App.school] };
    }).filter(function (r) { return r.row; });
    if (rows.length < 2) return '';
    var max = 0;
    rows.forEach(function (r) { max = Math.max(max, r.row.catTraps, r.row.mouseTraps); });
    max = Math.max(max, 0.2);
    var first = rows[0].row, last = rows[rows.length - 1].row;
    var drop = (first.catTraps + first.mouseTraps) - (last.catTraps + last.mouseTraps);
    return '<div style="border-top:1px solid var(--line);padding-top:16px">'
      + '<div class="mono" style="font-size:11px;letter-spacing:1.6px;color:#c9d8ee">TRAPS SPRUNG PER EPISODE</div>'
      + '<div class="faint" style="font-size:11.5px;margin:6px 0 14px;line-height:1.5">'
      + 'Nothing tells either of them a trap is dangerous. Stepping on one costs five frozen steps, and that is the '
      + 'only lesson available. ' + (drop > 0.05
        ? 'Over this school\'s three checkpoints the pair went from ' + (first.catTraps + first.mouseTraps).toFixed(2)
          + ' snaps an episode down to ' + (last.catTraps + last.mouseTraps).toFixed(2) + '.'
        : 'On this run the count has not fallen yet — the policies are still clumsy enough to walk into them.')
      + '</div>'
      + '<div style="display:flex;gap:22px">' + rows.map(function (r) {
          return '<div style="flex:1"><div class="mono faint" style="font-size:9.5px;letter-spacing:1.2px;margin-bottom:6px">'
            + CP_NAME[r.cp] + '</div>'
            + bar(r.row.catTraps, max, 'var(--cat)') + bar(r.row.mouseTraps, max, 'var(--mouse)') + '</div>';
        }).join('') + '</div></div>';
  }

  /* A rate with its 95% interval drawn on the same track. Eight arenas and a few dozen
     repeats is not a large sample, and three of these bars sit next to each other inviting
     a comparison — so the width of the uncertainty has to be as visible as the rate. */
  function ciBar(v, lo, hi, color) {
    var band = (lo !== undefined && hi !== undefined)
      ? '<div style="position:absolute;top:0;bottom:0;left:' + (lo * 100).toFixed(1) + '%;width:'
        + Math.max(0.4, (hi - lo) * 100).toFixed(1) + '%;background:' + color + ';opacity:.22"></div>'
        + '<div style="position:absolute;top:0;bottom:0;left:' + (lo * 100).toFixed(1) + '%;width:1px;background:' + color + ';opacity:.6"></div>'
        + '<div style="position:absolute;top:0;bottom:0;left:' + (hi * 100).toFixed(1) + '%;width:1px;background:' + color + ';opacity:.6"></div>'
      : '';
    return '<div style="display:flex;gap:8px;align-items:center;margin-bottom:5px">'
      + '<div style="position:relative;flex:1;height:12px;border-radius:6px;background:rgba(255,255,255,.04);overflow:hidden">'
      + band
      + '<div style="position:absolute;top:0;bottom:0;left:' + (v * 100).toFixed(1) + '%;width:2px;background:' + color + '"></div></div>'
      + '<div class="mono" style="width:46px;font-size:12px;color:' + color + '">' + pct(v) + '</div></div>';
  }

  function bar(v, max, color) {
    return '<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">'
      + '<div style="flex:1;height:9px;border-radius:5px;background:rgba(255,255,255,.04);overflow:hidden">'
      + '<div style="height:100%;width:' + Math.round(Math.min(1, v / max) * 100) + '%;background:' + color + ';opacity:.8"></div></div>'
      + '<div class="mono" style="width:34px;font-size:11px;color:' + color + '">' + v.toFixed(2) + '</div></div>';
  }

  function renderBoard() {
    var t = App.cat && App.cat.tournament;
    if (!t) {
      return '<div class="screen">' + backdrop('bg-academy', .4)
        + '<div style="position:relative"><div class="title" style="font-size:46px">LEADERBOARD</div>'
        + '<div class="dim" style="margin-top:14px;font-size:15px">No tournament has been run for this training run yet.<br>'
        + 'Run <span class="mono">python trainer/scripts/tournament_run.py --run runs/latest</span> and reload.</div>'
        + '<div class="btn ghost" style="margin-top:24px" data-act="menu">ESC · ACADEMY</div></div></div>';
    }
    var schools = t.schools;
    /* Two header rows, not one. The columns are the OPPONENT, so for the three cat rows
       they are mice and for the three mouse rows they are cats — a single "… mouse"
       header sitting over both halves says the bottom half is mice playing mice. */
    function head(role, top) {
      return '<tr>' + (top ? '<th></th>' : '<th style="padding-top:18px"></th>')
        + schools.map(function (s) {
            var v = view(s);
            return '<th style="color:' + v.color + (top ? '' : ';padding-top:18px') + '">'
              + esc(v.short) + ' ' + role + '</th>';
          }).join('')
        + '<th' + (top ? '' : ' style="padding-top:18px"') + '>SCORE</th>'
        + '<th' + (top ? '' : ' style="padding-top:18px"') + '>vs EXAMINER</th></tr>';
    }

    var rows = schools.map(function (c) {
      var v = view(c);
      var s = t.catScore[c];
      var a = t.anchor[c].cat;
      return '<tr><th style="color:' + v.color + ';text-align:right">' + esc(v.short) + ' cat</th>'
        + schools.map(function (m) {
            var cell = t.cross[c][m];
            return '<td' + (c === m ? ' class="you"' : '') + '>' + pct(cell.catch) + '</td>';
          }).join('')
        + '<td style="color:#e8eef9;font-weight:700">' + pct(s.rate)
        + '<span class="ci">' + pct(s.lo) + '–' + pct(s.hi) + '</span></td>'
        + '<td class="dim">' + pct(a.catch) + '</td></tr>';
    }).join('');

    var mrows = schools.map(function (m) {
      var v = view(m);
      var s = t.mouseScore[m];
      var a = t.anchor[m].mouse;
      return '<tr><th style="color:' + v.color + ';text-align:right">' + esc(v.short) + ' mouse</th>'
        + schools.map(function (c) {
            var cell = t.cross[c][m];
            return '<td' + (c === m ? ' class="you"' : '') + '>' + pct(cell.escape) + '</td>';
          }).join('')
        + '<td style="color:#e8eef9;font-weight:700">' + pct(s.rate)
        + '<span class="ci">' + pct(s.lo) + '–' + pct(s.hi) + '</span></td>'
        + '<td class="dim">' + pct(a.escape) + '</td></tr>';
    }).join('');

    var champCat = view(t.champion.cat), champMouse = view(t.champion.mouse);
    var contested = (t.contested.cat.length || t.contested.mouse.length);

    var budgets = '';
    if (App.cat.budgets) {
      budgets = '<div style="display:flex;gap:12px;margin-top:14px">' + schools.map(function (s) {
        var b = App.cat.budgets[s], v = view(s);
        if (!b) return '';
        return '<div class="card" style="flex:1;padding:12px 16px"><div class="mono" style="font-size:11px;color:'
          + v.color + '">' + esc(v.short) + '</div>'
          + '<div class="mono" style="font-size:17px;margin-top:6px">' + (b.envSteps / 1e6).toFixed(1) + 'M steps</div>'
          + '<div class="faint" style="font-size:10.5px;margin-top:2px">' + Math.round(b.wall / 60) + ' min · '
          + b.iters + ' updates</div></div>';
      }).join('') + '</div>';
    }

    return '<div class="screen">' + backdrop('bg-final', .35)
      + '<div style="position:relative;display:flex;justify-content:space-between;align-items:flex-start">'
      + '<div><div class="kicker">Every cat against every mouse · arenas nobody trained on</div>'
      + '<div class="title" style="font-size:48px;margin-top:6px">THE LEADERBOARD</div></div>'
      + '<div style="display:flex;gap:10px">' + revealChip() + statusChip() + '</div></div>'
      + '<div style="position:relative;display:flex;gap:22px;margin-top:18px;flex:1">'
      + '<div class="card" style="padding:20px 24px"><table class="grid">'
      + head('mouse', true) + rows + head('cat', false) + mrows + '</table>'
      + '<div class="faint" style="font-size:11.5px;margin-top:14px;max-width:760px;line-height:1.6">'
      + 'Each cell is ' + t.episodesPerPair + ' episodes on ' + t.arenas.length + ' held-out arenas. '
      + 'SCORE is the mean against the OTHER TWO schools. The shaded diagonal is a school playing itself — it is drawn '
      + 'because the gap is the interesting part, but it does not count towards SCORE, because it is exactly the number '
      + 'that flatters a school that raised a weak sparring partner. Ranges are 95% intervals.'
      + '</div></div>'
      + '<div style="flex:1;display:flex;flex-direction:column;gap:14px">'
      + champBox('BEST TOM', champCat, pct(t.catScore[t.champion.cat].rate), 'cat')
      + champBox('BEST JERRY', champMouse, pct(t.mouseScore[t.champion.mouse].rate), 'mouse')
      + (contested ? '<div class="card" style="padding:14px 18px;border-color:rgba(255,209,102,.3)">'
          + '<div class="mono" style="font-size:11px;color:var(--gold);letter-spacing:1.4px">TOO CLOSE TO CALL</div>'
          + '<div class="faint" style="font-size:12px;margin-top:6px;line-height:1.5">'
          + 'The intervals overlap, so this margin is not evidence. '
          // Name the winner as well as the challengers: "Mouse: PPO." on its own reads
          // as the verdict, directly under a BEST JERRY card crowning somebody else.
          + (t.contested.cat.length ? 'Cat: ' + view(t.champion.cat).short + ' over '
              + t.contested.cat.map(function (s) { return view(s).short; }).join(', ') + '. ' : '')
          + (t.contested.mouse.length ? 'Mouse: ' + view(t.champion.mouse).short + ' over '
              + t.contested.mouse.map(function (s) { return view(s).short; }).join(', ') + '.' : '')
          + '</div></div>' : '')
      + budgets + '</div></div>'
      + '<div style="position:relative;display:flex;gap:10px;margin-top:12px">'
      + '<div class="btn ghost" data-act="menu">ESC · ACADEMY</div>'
      + '<div class="btn" data-act="final">RUN THE GRAND FINAL · F</div></div></div>';
  }

  function champBox(label, v, score, role) {
    return '<div class="card" style="padding:18px 20px;display:flex;gap:16px;align-items:center;border-color:' + P.rgba(v.color, .35) + '">'
      + '<div style="width:96px;height:96px;flex:0 0 auto">' + P.portrait(role, v.color, 78) + '</div>'
      + '<div><div class="faint" style="font-size:10px;letter-spacing:2px">' + label + '</div>'
      + '<div class="title" style="font-size:32px;color:' + (v.sealed ? '#8494ad' : v.color) + ';margin-top:4px">' + esc(v.short) + '</div>'
      + '<div class="mono" style="font-size:22px;margin-top:6px">' + score + '</div>'
      + '<div class="faint" style="font-size:10.5px">against the other two schools</div></div></div>';
  }

  /* ---------------- render + hot loop ---------------- */

  /* A cheat sheet, because during a take you will not want to remember which key
     reveals the next school. Deliberately not a screen — it overlays whatever is
     running, so nothing has to be interrupted to check it. */
  var KEYS = [
    ['1 2 3', 'enter a school'], ['x', 'side by side — all three, same room'],
    ['g', 'the level generator'], ['l', 'full-screen lesson'],
    ['w', 'how this algorithm works — six steps'],
    ['h', 'the highlight reel'], ['f', 'the grand final'],
    ['b', 'the leaderboard'], ['v', 'this school\'s verdict'],
    ['space', 'pause / resume'], ['s', 'skip this episode'],
    ['[  ]', 'slower / faster'], ['t', 'train live, on camera'],
    ['esc', 'back to the academy'], ['?', 'this card'],
    ['r', 'REVEAL the next school'], ['shift+R', 're-seal one (for a re-shoot)'],
    ['shift+0', 're-seal everything — back to PPO only'],
  ];

  function keyCard() {
    if (!App.showKeys) return '';
    return '<div style="position:absolute;inset:0;z-index:50;background:rgba(3,5,10,.82);display:grid;place-items:center">'
      + '<div class="card" style="padding:34px 44px;min-width:760px">'
      + '<div class="title" style="font-size:30px;margin-bottom:22px">KEYS</div>'
      + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px 44px">'
      + KEYS.map(function (k) {
          var hot = k[0][0] === 'r' || k[0].indexOf('shift') === 0;
          return '<div style="display:flex;gap:16px;align-items:baseline">'
            + '<span class="mono" style="width:82px;color:' + (hot ? 'var(--gold)' : '#c9d8ee')
            + ';font-size:14px">' + esc(k[0]) + '</span>'
            + '<span class="' + (hot ? '' : 'dim') + '" style="font-size:13.5px;color:'
            + (hot ? 'var(--gold)' : '') + '">' + esc(k[1]) + '</span></div>';
        }).join('')
      + '</div><div class="faint" style="font-size:12px;margin-top:26px">'
      + 'Reveal state survives a reload, so a crash mid-shoot will not unseal the rest of the video.'
      + '</div></div></div>';
  }

  /* The explainer sits over whatever is running, in its own host rather than inside the
     screen, so the arena keeps its DOM and closing it restarts nothing.
     It is rewritten only when the thing being read actually changes. A frame, a status
     message or a reveal landing behind it would otherwise re-run every diagram's entry
     animation under the reader — and only the diagrams are supposed to move. */
  var _xKey = '';
  function renderExplain() {
    var host = el('explain');
    if (!host) return;
    var v = App.explain ? view(App.school) : null;
    var key = (v && !v.sealed) ? App.school + ':' + App.explain : '';
    if (key === _xKey) return;
    var opening = !_xKey && key;
    _xKey = key;
    // Only the open fades in. Stepping recreates the same chrome, which carries no
    // animation of its own, so the swap is invisible and the new diagram plays alone.
    host.innerHTML = key ? window.Explain.overlay(v, App.explain, !!opening) : '';
  }

  function openExplain() {
    if (App.screen !== 'school' || App.explain) return;
    if (view(App.school).sealed) return;
    App.explainWasPlaying = App.playing;
    App.explain = 1;
    if (App.playing) { App.playing = false; App.net.send({ cmd: 'pause' }); }
    render();
  }

  function closeExplain() {
    if (!App.explain) return;
    App.explain = 0;
    if (App.explainWasPlaying && !App.playing) { App.playing = true; App.net.send({ cmd: 'resume' }); }
    render();
  }

  function stepExplain(d) {
    if (!App.explain) return;
    var n = App.explain + d;
    if (n > window.Explain.count(App.school)) { closeExplain(); return; }
    App.explain = Math.max(1, n);
    render();
  }

  function render() {
    var html = { menu: renderMenu, gen: renderGen, school: renderSchool, lesson: renderLesson,
                 race: renderRace, verdict: renderVerdict, final: renderFinal,
                 board: renderBoard }[App.screen]();
    el('screens').innerHTML = html + keyCard();
    renderExplain();
    App.mapKey = null;                 // force a repaint into the new DOM
    fitStage(true);
  }

  function paintArena(now) {
    var host = el('arena-map'), fx = el('arena-fx');
    if (!host || !fx || !App.map) return;
    var key = App.map.seed + ':' + App.map.nest.join(',');
    if (App.mapKey !== key) {
      App.mapKey = key;
      host.innerHTML = P.mapSvg(localMap(App.map), CS);
    }
    if (!App.frame) return;
    var v = view(App.screen === 'final' ? (App.runState && App.runState.catSchool) || App.school : App.school);
    var mv = view(App.screen === 'final' ? (App.runState && App.runState.mouseSchool) || App.school : App.school);
    fx.innerHTML = P.fxSvg({
      frame: App.frame, prev: App.prev, alpha: App.alpha, cs: CS, map: App.map,
      key: 'live', now: now, catAccent: v.color, mouseAccent: mv.color,
      sprites: App.sprites, spriteFps: App.spriteFps, holdSteps: E.CFG.freezeSteps,
      catMoving: App.catMoving, mouseMoving: App.mouseMoving
    });
    var b = el('arena-banner');
    if (b) {
      b.innerHTML = App.banner && now - App.bannerAt < 3200
        ? '<div class="banner" style="color:' + App.banner.c + '">' + App.banner.t + '</div>' : '';
    }
  }

  /* The trainer sends a flat grid; the painters want the richer local shape (blocks,
     pillars, spawns). Regenerating from the seed is exact — the Python port is verified
     bit-for-bit against this same env.js — so the two always agree. */
  function localMap(payload) {
    if (payload._local) return payload._local;
    // The hole count is part of the room's identity: the same seed with one hole and
    // with two is two different rooms. The trainer sends it with the map.
    var m = E.genMap(payload.seed >>> 0, (payload.nests || []).length || 1);
    payload._local = m;
    return m;
  }

  function paintPanel() {
    var host = el('panel');
    if (!host) return;
    var lesson = App.screen === 'lesson';
    // The lesson always shows the algorithm, even during playback: it is a explanation
    // of the method, not a readout of the current episode.
    var p = (lesson || App.mode === 'train') ? App.panels[App.school] : App.livePanel;
    if (!p || !p.draw) return;
    host.innerHTML = lesson ? p.draw(1180, 700, view(App.school).color)
                            : p.draw(700, 420, view(App.school).color);
  }

  var last = 0;
  function loop(now) {
    requestAnimationFrame(loop);
    step(now);
  }

  function step(now) {
    var dt = now - (last || now); last = now;
    fitStage();
    refreshStatus();
    if (App.screen === 'gen') genTick(now);
    // Interpolate between the last two frames so movement is smooth at any speed.
    App.alpha = Math.min(1, App.alpha + dt / 1000 * 9 * App.speed);
    // A character is walking exactly while it is still sliding between two cells; a step
    // it spent standing still leaves the sprite on its idle pose rather than marching in
    // place. The walk phase itself is clock-driven inside WalkSprite, so the feet keep
    // their own 8 fps whatever the render rate or the playback speed.
    var sliding = !!(App.frame && App.prev && App.alpha < 1);
    App.catMoving = sliding
      && (App.frame.cat.x !== App.prev.cat.x || App.frame.cat.y !== App.prev.cat.y);
    App.mouseMoving = sliding
      && (App.frame.mouse.x !== App.prev.mouse.x || App.frame.mouse.y !== App.prev.mouse.y);
    if (App.screen === 'race') paintRace(now);
    else if (App.screen === 'school' || App.screen === 'final') { paintArena(now); paintPanel(); }
    else if (App.screen === 'lesson') paintPanel();
  }

  /* requestAnimationFrame does not run in a hidden tab, but the WebSocket does: the score
     boxes and the sense line kept updating out of the message handler while the arena sat
     frozen on whatever it had last drawn. A window behind another one, or a second screen
     the compositor has parked, is an ordinary thing to happen during a shoot. Timers are
     throttled when hidden rather than stopped, so this keeps the picture and the numbers
     telling the same story, and the return to visible snaps everything back into step. */
  setInterval(function () { if (document.hidden) step(performance.now()); }, 250);
  document.addEventListener('visibilitychange', function () {
    if (document.hidden) return;
    last = 0;
    App.alpha = 1;
    step(performance.now());
  });

  function genTick(now) {
    if (App.levels.length >= 12) return;
    if (now - App.lastGen < 300) return;
    // Without the catalogue there is no level set and no hole count, and the old
    // fallback invented both — twelve one-hole rooms under a two-hole run, drawn as if
    // they were the arenas the schools trained on. Better to draw nothing and say why.
    if (!App.cat || !App.cat.levels) return;
    App.lastGen = now;
    var seed = App.cat.levels[App.levels.length].seed;
    var nn = App.cat.levels[App.levels.length].nests || 1;
    var m = E.genMap(seed >>> 0, nn);
    var on = {};
    (m.route || []).forEach(function (p) { on[p[0] + ',' + p[1]] = 1; });
    App.levels.push({
      seed: seed >>> 0, map: m, route: m.optimal,
      onRoute: m.traps.filter(function (t) { return on[t[0] + ',' + t[1]]; }).length
    });
    render();
    var host = el('lv' + (App.levels.length - 1));
    if (host) host.innerHTML = P.mapSvg(m, 11);
    for (var i = 0; i < App.levels.length; i++) {
      var h2 = el('lv' + i);
      if (h2 && !h2.querySelector('svg')) h2.innerHTML = P.mapSvg(App.levels[i].map, 11);
    }
  }

  /* ---------------- commands ---------------- */

  function go(screen) {
    App.screen = screen;
    if (screen === 'gen') { App.levels = []; App.lastGen = 0; }
    render();
  }

  function playSchool(key, cp) {
    App.school = key;
    App.checkpoint = cp || App.checkpoint;
    App.results = [];
    App.frame = App.prev = null;
    App.trainInfo = null;
    App.mode = 'play';
    App.highlights = null;
    App.screen = 'school';
    render();
    App.net.send({ cmd: 'play', school: key, checkpoint: App.checkpoint });
  }

  /* The episodes the highlight scan picked, in its order.
     Three things have to travel with the request or the caption describes an episode
     nobody is watching: the arena, the episode seed (the server re-seeds both the
     environment and the action stream from it), and the MOUSE'S school — the scan runs
     the champion cat against the champion mouse, and those are usually not the same
     school. highlights.py re-scores every pick through this exact replay before writing
     it, so the label on screen is the label of the episode on screen. */
  function playHighlights() {
    var H = App.cat && App.cat.highlights;
    if (!H || !H.highlights || !H.highlights.length) return;
    App.school = H.catSchool || App.school;
    App.checkpoint = 'trained';
    App.results = [];
    App.frame = App.prev = null;
    App.mode = 'play';
    // Distinct episodes and rooms, not seeds tried: a saturated policy plays a room
    // the same way whatever the seed, so "149 found in 480" would be four episodes
    // counted forty times each.
    App.trainInfo = H.found + ' distinct dramatic episodes'
      + (H.rooms ? ' in ' + H.rooms + ' of the ' + (H.arenas || 12) + ' rooms' : '')
      + ' — playing the best ' + H.highlights.length + ', one per room'
      + (H.mouseSchool && H.mouseSchool !== H.catSchool
         ? ' · ' + view(H.catSchool).short + ' cat vs ' + view(H.mouseSchool).short + ' mouse' : '');
    App.highlights = H.highlights;
    App.screen = 'school';
    render();
    App.net.send({ cmd: 'play', school: App.school, checkpoint: 'trained',
                   mouseSchool: H.mouseSchool || App.school,
                   levels: H.highlights.map(function (h) { return h.arena; }),
                   seeds: H.highlights.map(function (h) { return h.seed; }) });
  }

  function startRace() {
    App.raceFrames = {}; App.racePrev = {}; App.raceDone = {}; App.raceWins = {}; App.raceGrid = {};
    App.screen = 'race';
    render();
    App.net.send({ cmd: 'race', checkpoint: 'trained' });
  }

  function trainLive() {
    App.mode = 'train';
    // Not "training live" yet — the optimiser takes a couple of seconds to set up, and
    // the arena holds still until it exists rather than being filled by anything else.
    App.trainReady = false;
    App.trainInfo = 'Starting the optimiser — the arena waits for the first real policy.';
    App.net.send({ cmd: 'train', school: App.school, minutes: 10 });
    render();
  }

  function bind() {
    el('root').addEventListener('click', function (e) {
      var t = e.target.closest('[data-act],[data-school],[data-cp],[data-xstep]');
      if (!t) return;
      if (t.dataset.xstep) { App.explain = +t.dataset.xstep; render(); return; }
      if (t.dataset.school) { playSchool(t.dataset.school); return; }
      if (t.dataset.cp) { playSchool(App.school, t.dataset.cp); return; }
      var a = t.dataset.act;
      if (a === 'menu') go('menu');
      else if (a === 'gen') go('gen');
      else if (a === 'board') go('board');
      else if (a === 'final') { App.screen = 'final'; render(); App.net.send({ cmd: 'final' }); }
      else if (a === 'pause') { App.playing = !App.playing; App.net.send({ cmd: App.playing ? 'resume' : 'pause' }); render(); }
      else if (a === 'skip') App.net.send({ cmd: 'skip' });
      else if (a === 'train') trainLive();
      else if (a === 'highlights') playHighlights();
      else if (a === 'race') startRace();
      else if (a === 'school') { App.screen = 'school'; render(); }
      else if (a === 'x-open') openExplain();
      else if (a === 'x-next') stepExplain(1);
      else if (a === 'x-prev') stepExplain(-1);
      else if (a === 'x-close') closeExplain();
    });

    window.addEventListener('keydown', function (e) {
      var k = (e.key || '').toLowerCase();
      /* The explainer is a modal read, so it swallows the whole keyboard while it is
         open — otherwise space would silently resume the run behind it and 1/2/3 would
         swap schools under the reader's feet. */
      if (App.explain) {
        if (e.code === 'Space' || k === 'arrowright' || k === 'enter') { e.preventDefault(); stepExplain(1); }
        else if (k === 'arrowleft') { e.preventDefault(); stepExplain(-1); }
        else if (k === 'escape' || k === 'backspace' || k === 'w') closeExplain();
        return;
      }
      // A replay has no clock to talk to: the journal is played back at its recorded
      // pace and nothing is listening for commands. Answering these keys with a banner
      // beats letting them look ignored on camera.
      if (App.replay && (e.code === 'Space' || k === 's' || k === '[' || k === ']')) {
        e.preventDefault();
        App.banner = { t: 'REPLAY · THE PACE IS RECORDED', c: '#8fa4c4' };
        App.bannerAt = performance.now();
        render();
        return;
      }
      if (k === 'escape') go('menu');
      else if (k === '1' || k === '2' || k === '3') playSchool(ORDER[+k - 1]);
      else if (k === 'g') go('gen');
      else if (k === 'b') go('board');
      else if (k === 'h') playHighlights();
      else if (k === 'x') startRace();
      else if (k === '?' || k === '/') { App.showKeys = !App.showKeys; render(); e.preventDefault(); }
      else if (k === 'v' && App.results.length) { App.screen = 'verdict'; render(); }
      else if (k === 'c') { App.sprites = !App.sprites; App.mapKey = null; }
      else if (k === 'f') { App.screen = 'final'; render(); App.net.send({ cmd: 'final' }); }
      else if (k === 't') trainLive();
      else if (k === 'l') { App.screen = App.screen === 'lesson' ? 'school' : 'lesson'; render(); }
      else if (k === 'w') openExplain();
      else if (k === 's') App.net.send({ cmd: 'skip' });
      else if (e.code === 'Space') {
        e.preventDefault();
        App.playing = !App.playing;
        App.net.send({ cmd: App.playing ? 'resume' : 'pause' });
        render();
      } else if (k === '[' || k === ']') {
        var steps = [1, 2, 4, 8, 16];
        var i = steps.indexOf(App.speed);
        App.speed = steps[Math.max(0, Math.min(steps.length - 1, i + (k === ']' ? 1 : -1)))];
        App.net.send({ cmd: 'speed', value: App.speed });
        render();
      }
    });

    window.Reveal.bindKeys();
    window.Reveal.on(function () {
      // Re-sealing a school mid-read (shift+R for a re-shoot) must not leave the modal
      // up with nothing in it — and nothing swallowing the keyboard.
      if (App.explain && view(App.school).sealed) closeExplain();
      else render();
    });
  }

  /* ---------------- wiring ---------------- */

  function start() {
    fitStage();
    App.net = new window.Net();
    App.net
      .on('status', function (m) { App.link = m.status; render(); })
      .on('hello', function (m) {
        App.cat = m;
        if (!App.levels.length && m.levels) {
          App.levels = m.levels.map(function (l) {
            var mm = E.genMap(l.seed >>> 0, l.nests || 1);
            return { seed: l.seed, map: mm, route: l.optimal, onRoute: l.trapsOnRoute,
                     nests: l.nests || 1 };
          });
        }
        render();
      })
      .on('frame', function (m) {
        // `step` advances by exactly one within an episode and resets on a new one, so
        // it is the precise test for whether these two frames are joinable.
        App.lastFrameAt = performance.now();
        var joins = App.frame && m.step === App.frame.step + 1;
        App.prev = joins ? App.frame : m;
        App.frame = m;
        App.alpha = joins ? 0 : 1;
        App.livePanel.update(m);
        if (m.map) { App.map = m.map; App.mapKey = null; }
        if (m.level !== undefined && App.runState) App.runState.level = m.level;
        var th = el('thought');
        if (th) th.textContent = m.cat.mode + ' · ' + m.mouse.mode;
        // The first frame of a train run is the proof the optimiser exists: the server
        // holds the shadow arena until it does, so only now is "training live" true.
        if (m.mode === 'train' && !App.trainReady) {
          App.trainReady = true;
          App.trainInfo = 'Training live — the panel is this run\'s own telemetry.';
          render();
        }
      })
      // A journal replay is a legitimate way to shoot, but it is not a trainer, and the
      // clock keys have nothing to talk to. Say both, rather than letting `space` look
      // broken and the chip claim a live run.
      .on('replay', function (m) {
        App.replay = m.source || true;
        App.trainInfo = 'Replaying ' + (m.source || 'a recorded session')
          + ' — pause, speed and skip do not apply.';
        render();
      })
      .on('replayEnd', function () {
        App.trainInfo = 'Replay finished.';
        render();
      })
      .on('trainWait', function () {
        App.trainReady = false;
        App.trainInfo = 'Starting the optimiser — the arena waits for the first real policy.';
        render();
      })
      .on('race', function (m) {
        App.lastFrameAt = performance.now();
        App.raceSchools = m.schools;
        App.raceWins = m.wins;
        // Lanes finish at different times, so continuity is decided per lane: a pane
        // that has already ended holds still while the others keep moving.
        var prevLanes = {}, next = {};
        m.lanes.forEach(function (l) {
          var was = App.raceFrames && App.raceFrames[l.school];
          prevLanes[l.school] = (was && l.step === was.step + 1) ? was : l;
          next[l.school] = l;
        });
        App.racePrev = prevLanes;
        App.raceFrames = next;
        App.alpha = 0;
        if (m.map) { App.map = m.map; }
        if ((App.runState || {}).level !== m.level) {
          App.runState = Object.assign({}, App.runState, { level: m.level });
          App.raceDone = {};
          render();
        }
      })
      .on('laneEnd', function (m) {
        App.raceDone = App.raceDone || {};
        App.raceDone[m.school] = m.result;
        App.raceGrid = App.raceGrid || {};
        (App.raceGrid[m.school] = App.raceGrid[m.school] || [])[m.level] = m.result;
        render();
      })
      .on('episodeEnd', function (m) {
        App.results[m.level] = m.result;
        App.banner = m.result === 'catch' ? { t: 'TOM CAUGHT HER', c: '#ff8a5c' }
          : m.result === 'escape' ? { t: 'JERRY GOT HOME', c: '#ffd166' }
          : { t: 'TIME OUT', c: '#8fa4c4' };
        App.bannerAt = performance.now();
        render();
      })
      .on('state', function (m) {
        App.runState = m;
        App.playing = m.playing !== false;
        // Follow the stream. A replay only re-sends messages, so without this the app
        // sits on the menu while the recorded episodes play into a screen nobody is
        // looking at. It also means the trainer can drive the app from Python.
        var to = { play: 'school', final: 'final', race: 'race', train: 'school' }[m.mode];
        if (to && App.screen !== to) {
          App.screen = to;
          if (m.school) App.school = m.school;
          if (m.checkpoint) App.checkpoint = m.checkpoint;
          App.mode = m.mode === 'train' ? 'train' : 'play';
          if (m.mode !== 'race') App.results = [];
        }
        render();
      })
      .on('runEnd', function (m) {
        App.runState = m;
        // A school that has finished its twelve arenas goes to its verdict; the final
        // stays where it is, because its own scoreboard is already on screen.
        if (App.screen === 'school') App.screen = 'verdict';
        render();
      })
      .on('train', function (m) {
        if (m.kind === 'algo') {
          var p = App.panels[m.school];
          if (p) {
            // Cat and mouse telemetry both arrive; the panel shows the cat's, which is
            // the side the arena is usually following.
            p.update(Object.assign({ gen: m.iter, role: m.cat ? 'cat' : 'mouse' },
                                   m.cat || m, { year: (m.cat || {}).year }));
          }
        } else if (m.kind === 'eval') {
          // Only while the training screen is the one being watched. `t` starts a run
          // that keeps going for minutes; leaving it for a plain playback screen used to
          // leave its telemetry writing "live · cat 23% · mouse 35%" under an arena that
          // is replaying a saved checkpoint and learning nothing.
          if (App.mode === 'train') {
            App.trainInfo = 'live · cat ' + pct(m.catExam) + ' · mouse ' + pct(m.mouseExam)
              + ' vs the Examiner · ' + (m.steps / 1e6).toFixed(1) + 'M steps';
          }
        } else if (m.kind === 'promotion') {
          App.banner = { t: m.role.toUpperCase() + ' PROMOTED TO YEAR ' + m.year, c: '#ffd166' };
          App.bannerAt = performance.now();
        } else if (m.kind === 'trainDone') {
          App.trainInfo = 'training finished';
        }
      })
      .connect();

    bind();
    render();
    requestAnimationFrame(loop);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
