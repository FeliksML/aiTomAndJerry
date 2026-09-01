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
  var CP_NAME = { untrained: 'UNTRAINED', half: 'HALF-TRAINED', trained: 'TRAINED',
                  // Not a fourth point on the timeline: the strongest policy the run
                  // reached, wherever that was. Usually the end; not always.
                  best: 'BEST' };
  var CS = 44;

  /* Run-level settings — the things that belong to the run rather than to one school.
     The budget and the knobs moved into each academy; what is left here is the tag the
     run writes under, the rooms, and which policy enters the championship. */
  var SETUP_DEFAULTS = { tag: 'v5', nests: '2', envs: 2048, steps: 500e6, minutes: null,
                         seed: 7, scoreWith: 'trained' };

  /* Per academy: its own budget, its own seed, its own shaping, its own knobs. The
     defaults for the knobs arrive from the trainer (`hello.academies`) so this only has
     to carry what the author has actually changed. */
  function acadDefaults() {
    return { steps: 500e6, minutes: null, seed: 11, shaping: {}, hyper: {} };
  }

  function loadAcademies() {
    var out = {};
    ORDER.forEach(function (k) { out[k] = acadDefaults(); });
    try {
      var raw = JSON.parse(localStorage.getItem('cma.acad') || '{}');
      ORDER.forEach(function (k) {
        if (!raw[k]) return;
        var a = out[k];
        if (raw[k].steps !== undefined) a.steps = raw[k].steps;
        if (raw[k].minutes !== undefined) a.minutes = raw[k].minutes;
        if (raw[k].seed !== undefined) a.seed = raw[k].seed;
        a.shaping = raw[k].shaping || {};
        a.hyper = raw[k].hyper || {};
      });
    } catch (e) { /* private window, or a value from an older build */ }
    return out;
  }

  function saveAcademies() {
    try { localStorage.setItem('cma.acad', JSON.stringify(App.acad)); } catch (e) {}
  }

  function loadSetup() {
    var out = {}, k;
    for (k in SETUP_DEFAULTS) out[k] = SETUP_DEFAULTS[k];
    try {
      var raw = JSON.parse(localStorage.getItem('cma.setup') || '{}');
      for (k in SETUP_DEFAULTS) if (raw[k] !== undefined && raw[k] !== null) out[k] = raw[k];
      if (raw.minutes !== undefined) out.minutes = raw.minutes;
    } catch (e) { /* a private window, or a value from an older build */ }
    return out;
  }

  function saveSetup() {
    try { localStorage.setItem('cma.setup', JSON.stringify(App.setup)); } catch (e) {}
  }

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
    // Two-press arming for RESET TO ZERO, so the one destructive control on screen
    // cannot fire on a single stray click during a take.
    resetArmed: 0,
    // The last beat of an episode. The server sends one final frame and then holds the
    // arena still for about nine tenths of a second, which is what the catch and escape
    // sheets are played over; `at` is the stopwatch they run on.
    ending: null, raceDoneAt: {},
    trainInfo: null, showKeys: false, setupNote: null,
    // A refusal or an announcement, drawn on top of whatever screen is up.
    notice: null,
    // Everything the trainer says about where it is: iteration, both clocks, the rate,
    // and the high-water mark each role has reached. Null until a run starts.
    train: null,
    serverMode: null,          // the last mode the server reported, to spot real changes
    // A three-school run: one entry per school, because three processes interleaving
    // their telemetry into one card is a card that reports whichever spoke last.
    runAll: null,
    // Whether an optimiser is running, which is a different fact from what the arena
    // happens to be playing. Beat 7 of the shoot is exactly this: watch an old checkpoint
    // while training carries on, and see both at once.
    training: null,            // {school, finished} while a live run exists
    // The tournament + highlight scan, which is the other thing that used to need a
    // terminal. `lines` is its console output, tailed onto the screen.
    scoring: null,
    // What the next run will be given. Wall-clock and environment steps are the same two
    // clocks the offline trainer takes, and a run ends on whichever arrives first — so
    // a step budget set here is a promise about the work done, not about the time.
    // Survives a reload, because losing it mid-shoot means re-typing a budget.
    setup: loadSetup(),
    // Per-academy settings, and the scrub timeline each academy has produced.
    acad: null,                // filled after ORDER exists
    acadOpen: false,           // the settings drawer on the school screen
    timeline: {},              // school -> [{i, steps, catExam, mouseExam, ...}]
    pinned: null,              // the frame the arena is playing, or null for live
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
  App.acad = loadAcademies();

  if (window.WalkSprite) {
    [window.WalkSprite.tom, window.WalkSprite.jerry].forEach(function (ch) {
      ch.animations.forEach(function (a) {
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

  // The painted trap and the painted hole. Until they arrive the vector pair is drawn,
  // exactly as with the characters — but the map layer is cached, and it is the layer
  // that decides whether to draw the vector arch, so it has to be told to draw again.
  if (window.PropSprite) {
    window.PropSprite.load(null, function (err) {
      if (err) { console.warn('prop atlas unavailable:', err.message); return; }
      App.mapKey = null;
      if (document.getElementById('screens')) render();
    });
  }
  App.livePanel = window.Panels.live();
  App.mode = 'play';           // 'play' shows the live decision, 'train' the explainer

  /* ---------------- helpers ---------------- */

  /* One budget, used by every kind of run the app can start — the three-school run and
     the single-school take on camera. Steps and minutes are both real clocks; setting
     both means "this many steps, but never longer than that", which is how an overnight
     run is kept from overrunning the morning. */
  var STEP_PRESETS = [5e6, 25e6, 100e6, 250e6, 500e6, 1e9, 2e9];
  var ENV_PRESETS = [512, 1024, 2048, 4096];

  /* Measured on this machine: three schools training at once sustain about 90k
     environment steps per second EACH at 2048 envs (~270k in total), and about 70k each
     at the old 512. The estimate is labelled as measured-on-an-M2-Max rather than
     presented as a promise, and it is the only honest way to answer "how long will 500M
     take" before the run has produced a rate of its own. */
  function estimateSeconds(stepsPerSchool, envs) {
    if (!stepsPerSchool) return null;
    var rate = envs >= 2048 ? 90e3 : envs >= 1024 ? 80e3 : 70e3;
    return stepsPerSchool / rate;
  }

  function el(id) { return document.getElementById(id); }
  function pct(v) { return (v === undefined || v === null || isNaN(v)) ? '—' : Math.round(v * 100) + '%'; }

  /* 1_500_000 -> "1.5M". The same rule as the trainer's `human_steps`, so a number read
     off the screen matches the one in the console and in the run's JSON. */
  function steps(n) {
    if (n === undefined || n === null || isNaN(n)) return '—';
    var u = [[1e9, 'B'], [1e6, 'M'], [1e3, 'k']];
    for (var i = 0; i < u.length; i++) {
      if (Math.abs(n) >= u[i][0]) {
        var v = n / u[i][0];
        return (v >= 100 || v === Math.round(v) ? Math.round(v) : v.toFixed(1)) + u[i][1];
      }
    }
    return String(Math.round(n));
  }

  function hms(sec) {
    if (sec === undefined || sec === null || isNaN(sec)) return '—';
    sec = Math.max(0, Math.round(sec));
    var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s2 = sec % 60;
    var p2 = function (v) { return String(v).padStart(2, '0'); };
    return h ? h + ':' + p2(m) + ':' + p2(s2) : m + ':' + p2(s2);
  }
  function esc(s) { return String(s === undefined ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  /* The identity card's numbers belong to the trainer, not to a copy of them kept here.
     POPULATION, LAMBDA and BATCH are all academy knobs — the population slider runs from
     8 to 256 — so a hardcoded "48 individuals" stops being true the moment the author
     moves one, while the panel a few centimetres away reads the real number off the same
     run. Filmed side by side, that is a card contradicting a live readout. The static
     strings stay as the fallback for before the trainer has spoken. */
  var LIVE_SPEC = {
    ppo: { BATCH: function (t) { return t.n_envs + ' arenas × ' + t.horizon + ' steps'; } },
    ga: { POPULATION: function (t) { return t.pop_size + ' individuals'; } },
    cmaes: { LAMBDA: function (t) { return t.lam + ' samples per generation'; } }
  };
  var LIVE_LINE = {
    ga: function (t) { return 'population ' + t.pop_size + ' · elitism · tournament selection'; },
    cmaes: function (t) { return 'σ-adaptation · λ=' + t.lam + ' · separable (diagonal)'; }
  };
  var LIVE_BLURB = {
    ga: function (t) {
      return 'No gradients at all. ' + t.pop_size + ' whole brains a generation; the ones '
        + 'that survived the room breed, their children are a coin-flip mix with a few '
        + 'weights nudged.';
    }
  };

  /* What a run started right now would actually use: the author's override if there is
     one, otherwise the trainer's own default, which is the same rule the academy slider
     reads. Null until `hello` lands — the card then keeps its written text. */
  function tunablesNow(key) {
    var a = (App.cat && App.cat.academies && App.cat.academies[key]) || null;
    if (!a || !a.tunables || !a.tunables.length) return null;
    var out = {};
    a.tunables.forEach(function (t) { out[t.key] = hyperValue(key, t); });
    return out;
  }

  function view(key) {
    var algo = ALGOS[key];
    var t = algo && tunablesNow(key);
    if (t) {
      var spec = LIVE_SPEC[key] || {};
      algo = Object.assign({}, algo, {
        specs: algo.specs.map(function (row) {
          // Only rows this table knows how to source live are replaced; the rest are
          // prose about the method and have no number to go stale.
          return spec[row[0]] ? [row[0], spec[row[0]](t)] : row;
        })
      });
      if (LIVE_LINE[key]) algo.line = LIVE_LINE[key](t);
      if (LIVE_BLURB[key]) algo.blurb = LIVE_BLURB[key](t);
    }
    return window.Reveal.view(algo);
  }

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
    // `display:contents` so the two chips inside still sit in the header's flex row with
    // its gap, while giving the loop one node to rewrite that is not the whole row.
    return '<span id="trainchip" style="display:contents">' + trainChip() + '</span>'
      + '<span class="chip" id="linkchip"><span class="link-dot ' + st.cls + '"></span>' + st.txt + '</span>';
  }

  /* Whether an optimiser is running, on EVERY screen, with the off switch attached.
   *
   * This fact used to be visible on exactly one screen — the school of the one training
   * school — so a ninety-minute run was invisible from the Academy, the leaderboard, the
   * level generator and the lesson, and the only way to stop it was a key nothing
   * mentioned. The chip is part of the header row every screen already draws. */
  function trainChip() {
    var r = App.runAll;
    if (r && !r.done) {
      var fr = ORDER.map(function (k) { return r.schools[k] ? r.schools[k].frac || 0 : 0; });
      var lo = Math.round(Math.min.apply(null, fr) * 100);
      return '<span class="chip" data-act="setup" title="see all three"'
        + ' style="cursor:pointer;border-color:rgba(242,181,68,.6);color:#f2b544">'
        + '<span class="link-dot link-live" style="background:#f2b544"></span>'
        + 'TRAINING ALL THREE · ' + lo + '%+</span>'
        + '<span class="chip" data-act="stop-all" title="stop all three"'
        + ' style="cursor:pointer;border-color:rgba(255,123,114,.5);color:#ff7b72">STOP ALL</span>';
    }
    var t = App.training;
    if (!t || t.finished) {
      return App.scoring && !App.scoring.done
        ? '<span class="chip" style="border-color:rgba(242,181,68,.5);color:#f2b544">SCORING…</span>'
        : '';
    }
    var v = view(t.school);
    var name = v.sealed ? 'SCHOOL ' + (ORDER.indexOf(t.school) + 1) : v.short;
    var pc = App.train && App.train.frac !== undefined ? ' ' + pct(App.train.frac) : '';
    var col = v.sealed ? '#8494ad' : v.color;
    return '<span class="chip" data-act="goto-training" title="go to the run"'
      + ' style="cursor:pointer;border-color:' + P.rgba(col, .55) + ';color:' + col + '">'
      + '<span class="link-dot link-live" style="background:' + col + '"></span>'
      + 'TRAINING ' + esc(name) + pc + '</span>'
      + '<span class="chip" data-act="train-stop" title="stop it"'
      + ' style="cursor:pointer;border-color:rgba(255,123,114,.5);color:#ff7b72">STOP · SHIFT+S</span>';
  }

  /* Patched in place from the animation loop rather than by re-rendering: a stalled
     stream is exactly the case where no message arrives to trigger a render. */
  var _linkTxt = null, _trainTxt = null;
  function refreshStatus() {
    var el2 = el('linkchip');
    if (!el2) { _linkTxt = _trainTxt = null; return; }
    var st = linkState();
    if (st.txt !== _linkTxt) {
      _linkTxt = st.txt;
      el2.innerHTML = '<span class="link-dot ' + st.cls + '"></span>' + st.txt;
    }
    // The training chip carries a percentage, so it changes far more often than a render
    // does. It is rebuilt in place beside the link chip, from the same loop, for the same
    // reason: the case that matters most is the one where no message triggers a render.
    var host = el('trainchip'), want = trainChip();
    if (host && want !== _trainTxt) {
      _trainTxt = want;
      host.innerHTML = want;
    }
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
      + '<div class="btn" data-act="setup">TRAINING · N</div>'
      // Not "NEW LEVEL SET": pressing it does not make one. The screen replays the run's
      // OWN twelve rooms from their own seeds, one at a time — which is the point of the
      // opening shot, and the opposite of what a button promising a new set would do.
      + '<div class="btn ghost" data-act="gen">' + (ready ? 'THE LEVEL SET · G' : 'GENERATE THE LEVELS') + '</div>'
      + '<div class="btn ghost" data-act="board">LEADERBOARD</div>'
      + '<div class="btn ghost" data-act="race">SIDE BY SIDE · X</div>'
      + '<div class="btn ghost" data-act="final">GRAND FINAL</div>'
      + (App.cat && App.cat.highlights && App.cat.highlights.highlights.length
          ? '<div class="btn ghost" data-act="highlights">HIGHLIGHTS · H</div>' : '')
      + resetButton()
      + '<div class="dim" style="margin-left:auto;font-size:13px;text-align:right">'
      + (App.link !== 'live'
        // The trainer is a separate process and the app talks to it over a socket, so
        // it cannot be started from in here. Saying only TRAINER OFFLINE left the one
        // question that matters — what do I type — unanswered on screen.
        ? 'The trainer is not running, so nothing can train or play.<br>'
          + 'Start it with <span class="mono" style="color:#c9d8ee">./run.sh serve</span>'
          + ' in the project folder — this page reconnects on its own.'
        : App.cat && App.cat.zeroed
          ? 'Every weight is a fresh random init — nothing has been trained yet. Enter a school and press t to train one on camera.'
          : ready ? App.levels.length + ' arenas ready · every school trains on the same rooms'
                  : 'Generate the shared level set first')
      + '</div></div></div>';
  }

  /* Two presses, because this is the one control on screen that destroys something, and
     it lives on a menu that gets clicked around during a take. The armed state times out
     on its own, so a stray first press cannot sit there waiting to be completed. */
  function resetButton() {
    var armed = App.resetArmed && (performance.now() - App.resetArmed < 5000);
    if (!armed && App.resetArmed) App.resetArmed = 0;
    return '<div class="btn ghost" data-act="reset" style="border-color:'
      + (armed ? 'rgba(255,138,92,.7)' : 'rgba(255,138,92,.28)') + ';color:'
      + (armed ? '#ff9a72' : '#b98070') + (armed ? ';background:rgba(255,122,84,.12)' : '') + '">'
      + (armed ? 'PRESS AGAIN — THIS WIPES EVERY WEIGHT' : 'RESET TO ZERO') + '</div>';
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
    var training = App.mode === 'train';
    // During the highlight reel the run is the reel, not the twelve-arena level set:
    // ten episodes counted out of twelve, under LV01-LV12 labels that name rooms the
    // reel is not playing, is two wrong numbers in the same corner of the screen.
    var reel = App.highlights;
    var n = reel ? reel.length : (App.cat ? App.cat.levels.length : 12);
    var st = App.runState || {};
    var done = App.results.filter(Boolean);
    var catch_ = done.filter(function (r) { return r === 'catch'; }).length;
    var esc_ = done.filter(function (r) { return r === 'escape'; }).length;

    /* BEST is offered only when this run actually has one. It is written by the trainer
       alongside the other three, so a run made before it existed has three chips and
       nothing to explain — better than a fourth chip that answers with an error.
       A zeroed run has none of them: nothing has trained, so three identical pills would
       invite a comparison of one thing with itself. */
    var have = App.cat && App.cat.available;
    var pills = (App.cat && App.cat.zeroed ? ['untrained']
                 : CP.concat(have && have.best && have.best.indexOf(App.school) >= 0 ? ['best'] : []))
      .map(function (c) {
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
      // The level strip counts twelve arenas; during training the arena is a shadow
      // episode cycling those rooms, so what is worth the space under it is the reel.
      + (trainedHere() || frames().length
        ? timelineStrip()
        : '<div style="display:flex;gap:4px;margin-top:10px">' + strip.join('') + '</div>')
      + '</div>'

      + '<div style="flex:1;display:flex;flex-direction:column;gap:12px;min-width:0">'
      + '<div style="display:flex;gap:12px">'
      /* Two different questions, so two different labels. While the shadow arena is
         replaying the run's own policy these boxes are a rolling tally over a loop that
         never ends; the rest of the time they are the twelve-room playthrough you
         started. Either way they are NOT the reel's percentages a few centimetres below,
         which measure the trainee against the Examiner. */
      + scoreBox(training ? 'TOM · CAUGHT HER, THIS LAP' : 'TOM · THIS PLAYBACK',
                 catch_ + ' / ' + n, 'var(--cat)')
      + scoreBox(training ? 'JERRY · GOT HOME, THIS LAP' : 'JERRY · THIS PLAYBACK',
                 esc_ + ' / ' + n, 'var(--mouse)')
      + scoreBox('ARENA', ((st.level || 0) + 1) + ' / ' + n, 'var(--gold)') + '</div>'
      /* The HUD says how the run is going. This says what the ARENA is, which is the
         first thing anyone asks on pressing `t`: it starts moving and nothing explains
         why. They are not being tested — the optimiser is running thousands of episodes
         a second in a batch nobody could watch, and this is one of them. */
      + (training
         ? '<div class="card" style="padding:13px 18px;border-color:rgba(255,209,102,.28)">'
           + '<div class="mono" style="font-size:10px;letter-spacing:1.6px;color:var(--gold)">SHADOW EPISODE · NOT A TEST</div>'
           + '<div class="faint" style="font-size:12px;line-height:1.5;margin-top:5px">'
           + 'The optimiser is running thousands of episodes a second. This is one of them, '
           + 'replayed at a watchable pace with the policy <b>as it stands right now</b> — '
           + 'it is re-read at the start of every episode, so the pair should visibly get '
           + 'better while you watch. The lap runs the twelve rooms and starts again.'
           + '</div></div>'
         : '')
      // `trainedHere()`, not `training`: the card belongs to the RUN, so it survives a
      // checkpoint click that takes the arena out of train mode.
      + (trainedHere() ? trainHud(accent) : '')
      + (App.acadOpen ? academyPanel(App.school, accent) : ''
      + '<div class="card" style="padding:14px 18px">'
      + '<div class="faint mono" style="font-size:10px;letter-spacing:1.6px">WHAT THEY CAN SENSE RIGHT NOW</div>'
      + '<div id="thought" class="mono" style="font-size:15px;margin-top:6px;color:#c9d8ee">' + esc(mode) + '</div></div>'
      + '<div class="card" style="flex:1;padding:6px;position:relative;min-height:0">'
      + (v.sealed
        ? '<div class="sealed-plate" style="border-radius:12px"><div style="text-align:center"><div class="sealed-tag">METHOD CLASSIFIED</div>'
          + '<div class="faint" style="font-size:12px;margin-top:12px">The explainer for this school is sealed until it is introduced.</div></div></div>'
        : '<div id="panel" style="width:100%;height:100%"></div>')
      + '</div>')
      + '</div></div>'

      + '<div style="position:relative;display:flex;gap:10px;margin-top:12px;align-items:center">'
      + '<div class="btn ghost" data-act="menu">ESC · ACADEMY</div>'
      + '<div class="btn ghost" data-act="pause">' + (App.playing ? 'PAUSE · SPACE' : 'RESUME · SPACE') + '</div>'
      + '<div class="btn ghost" data-act="skip">SKIP EPISODE · S</div>'
      // Only while there IS a run to end — the button used to survive `trainDone`, and
      // pressing it then froze the arena with every indicator still claiming a live run.
      + (trainingHere()
        ? '<div class="btn ghost" data-act="train-stop">STOP RUN · SHIFT+S</div>'
        : '<div class="btn ghost" data-act="acad-open">THIS ACADEMY · N</div>')
      + '<div class="dim" style="margin-left:auto;font-size:12px;text-align:right">'
      + (hl ? '<span class="mono" style="color:var(--gold);letter-spacing:2px">' + esc(hl.kind.toUpperCase())
              + '</span> &nbsp;·&nbsp; ' + esc(hl.why) + '<br>' : '')
      // Name the checkpoint that is actually playing. SHOOT.md sends the author through
      // UNTRAINED and HALF-TRAINED on camera, and the old line said "trained" over both.
      + (App.link !== 'live' && trainedHere()
         ? 'The trainer went away — nothing above is updating any more.'
         : App.trainInfo ? esc(App.trainInfo)
         : App.link !== 'live' ? 'Trainer offline — this is the last frame it sent.'
         : !App.frame ? 'Waiting for the first frame from the trainer.'
         : 'Frames are streaming from the ' + CP_NAME[App.checkpoint].toLowerCase()
           + ' policy — nothing here is scripted.')
      + '</div></div></div>';
  }

  /* Where the live run actually is.
   *
   * The arena already showed that *something* was happening; nothing on screen said
   * which iteration, how far through the budget, how fast, or when it would end. Every
   * number here is off the trainer's own telemetry — `steps`, `iter`, `sps` and `eta`
   * ride on every training event — and a field the trainer has not sent yet reads as a
   * dash rather than a zero, because "0 steps/s" is a measurement and absence is not.
   */
  /* Repaint the run card alone. `progress` lands once a second, and a full render() at
     that rate tore the reel handle out of a drag, dropped focus from the academy's
     fields, and rebuilt the arena SVG every second for nothing. */
  function paintTrainHud() {
    var host = el('train-hud');
    if (!host) { render(); return; }
    var v = view(App.school), html = trainHud(v.sealed ? '#8494ad' : v.color);
    host.innerHTML = html.slice(html.indexOf('>') + 1, html.lastIndexOf('</div>'));
  }

  function trainHud(accent) {
    var t = App.train;
    var frac = t && t.frac !== undefined ? Math.max(0, Math.min(1, t.frac)) : 0;
    var target = t && t.targetSteps ? steps(t.targetSteps) : null;

    function cell(k, v, sub, col) {
      return '<div style="flex:1;min-width:0">'
        + '<div class="mono faint" style="font-size:9px;letter-spacing:1.5px">' + k + '</div>'
        + '<div class="mono" style="font-size:20px;line-height:1.25;color:' + (col || '#f2f7ff') + '">' + v + '</div>'
        + (sub ? '<div class="mono faint" style="font-size:10px">' + sub + '</div>' : '')
        + '</div>';
    }

    // The high-water mark, and where in the run it was reached. This is the pair the
    // run-off at the end chooses between, so naming it while the run is going means the
    // final "best cat / best mouse" card is not the first time it is mentioned.
    function peak(role, col) {
      var p2 = t && t.best && t.best[role];
      if (!p2) return cell('BEST ' + role.toUpperCase() + ' SO FAR', '—', 'no evaluation yet', col);
      return cell('BEST ' + role.toUpperCase() + ' SO FAR', pct(p2.rate),
                  (p2.from === 'policy' ? 'from the brain at ' : 'seen by ')
                  + steps(p2.steps) + ' steps · lower bound ' + pct(p2.lo), col);
    }

    return '<div id="train-hud" class="card" style="padding:14px 18px;border-color:'
      + P.rgba(accent, .3) + ';opacity:' + (App.link === 'live' ? '1' : '.5') + '">'
      + '<div style="display:flex;align-items:baseline;gap:10px">'
      + '<div class="mono faint" style="font-size:10px;letter-spacing:1.6px;color:'
      + (App.link !== 'live' ? '#ff9a72' : '') + '">'
      + (App.link !== 'live' ? 'TRAINER GONE · THIS IS THE LAST THING IT SAID'
         : App.trainFinished ? 'RUN FINISHED' : 'LIVE TRAINING RUN') + '</div>'
      + '<div class="mono" style="font-size:10px;letter-spacing:1.4px;color:' + accent + '">BUDGET '
      + esc((t && t.budget) || budgetLabel()) + '</div>'
      + '<div class="mono" style="margin-left:auto;font-size:19px;color:' + accent + '">' + pct(frac) + '</div></div>'
      + '<div style="height:6px;border-radius:3px;background:rgba(255,255,255,.06);margin:8px 0 12px;overflow:hidden">'
      + '<div style="height:100%;width:' + (frac * 100).toFixed(2) + '%;background:' + accent + ';transition:width .35s linear"></div></div>'
      + '<div style="display:flex;gap:14px">'
      + cell('ITERATION', t ? String(t.iter) : '—')
      + cell('ENV STEPS', t ? steps(t.steps) : '—', target ? 'of ' + target : 'no step budget')
      + cell('SPEED', t && t.sps ? steps(t.sps) + '/s' : '—', 'environment steps per second')
      + cell('ELAPSED', t ? hms(t.wall) : '—', t && t.eta !== null && t.eta !== undefined ? hms(t.eta) + ' left' : 'no estimate yet')
      + '</div>'
      + '<div style="display:flex;gap:14px;margin-top:12px;padding-top:12px;border-top:1px solid var(--line)">'
      + peak('cat', 'var(--cat)') + peak('mouse', 'var(--mouse)') + '</div></div>';
  }

  /* ---------------- the academy: one school's own training ----------------
   *
   * Everything about how THIS school is taught lives on its own screen: how much work
   * it gets, how hard each side is pulled toward its goal, and the knobs its algorithm
   * actually has. Three academies, three budgets, three sets of knobs — set where you
   * are looking at the school, not on a screen outside it.
   *
   * What is deliberately NOT editable here is the terminal reward: +1 for a catch, −1
   * for letting her home, −0.05 for a trap. Those are the rules of the game and the
   * scoreboard counts them directly, so a school taught under different ones would not
   * be playing the same sport as the other two. They are shown, greyed, next to the
   * shaping, so the difference between "the rules" and "how I teach" is on screen
   * rather than something you have to know.
   */

  function slider(kind, key, value, min, max, step, fmt) {
    var id = 'sl-' + kind + '-' + key;
    return '<div style="display:flex;align-items:center;gap:10px">'
      + '<input type="range" id="' + id + '" data-' + kind + '="' + key + '"'
      + ' min="' + min + '" max="' + max + '" step="' + step + '" value="' + value + '"'
      + ' style="flex:1;accent-color:var(--gold);height:18px">'
      + '<span id="' + id + '-out" class="mono" style="width:76px;text-align:right;font-size:14px;color:#f2f7ff">'
      + fmt(value) + '</span></div>';
  }

  function acadOf(key) { return App.acad[key] || (App.acad[key] = acadDefaults()); }

  /* Is an optimiser running for the school on screen? Deliberately NOT `App.mode`, which
     says what the ARENA is playing. Clicking a checkpoint pill sets mode to 'play', and
     the run card, the STOP button and the live caption used to vanish with it — the one
     beat of the shoot that needs both facts at once was the one that could not be shot. */
  function trainingHere() {
    return !!(App.training && !App.training.finished && App.training.school === App.school);
  }

  function trainedHere() {
    return !!(App.training && App.training.school === App.school);
  }

  /* The knobs the trainer says this school has, with the value the author set or the
     trainer's own default. Read from `hello` so a knob added to a config dataclass
     turns up here instead of quietly not existing. */
  function tunablesFor(key) {
    var a = (App.cat && App.cat.academies && App.cat.academies[key]) || null;
    return a ? a.tunables : [];
  }

  function hyperValue(key, t) {
    var h = acadOf(key).hyper;
    return h[t.key] !== undefined && h[t.key] !== null ? h[t.key] : t.value;
  }

  function shapingValue(key, k) {
    var a = acadOf(key), d = (App.cat && App.cat.shapingDefaults) || {};
    return a.shaping[k] !== undefined && a.shaping[k] !== null ? a.shaping[k] : (d[k] || 0);
  }

  var SHAPING_ROWS = [
    ['mouseApproach', 'JERRY PULLED TOWARD A HOLE', 0, 0.2, 0.005],
    ['catApproachVisible', 'TOM PULLED WHEN HE SEES HER', 0, 0.06, 0.002],
    ['catApproachAlways', 'TOM PULLED EVEN BLIND', 0, 0.06, 0.002]
  ];

  function academyPanel(key, accent) {
    var a = acadOf(key), v = view(key);
    var mil = Math.max(1, Math.round((a.steps || 0) / 1e6));
    var eta = estimateSeconds(a.steps, hyperValue(key, { key: 'n_envs', value: 512 }));
    var training = !!(App.training && !App.training.finished && App.training.school === key);

    return '<div class="card" style="padding:16px 18px;border-color:' + P.rgba(accent, .32)
      + ';flex:1;min-height:0;overflow:auto">'

      + '<div style="display:flex;align-items:baseline;gap:10px">'
      + '<div class="mono" style="font-size:11px;letter-spacing:1.8px;color:' + accent + '">'
      + esc(v.sealed ? 'THIS ACADEMY' : v.short + ' ACADEMY') + '</div>'
      + '<div class="mono faint" style="font-size:10px">its own budget · its own knobs</div>'
      + '<div class="btn ghost" data-act="acad-close" style="margin-left:auto;padding:4px 10px;font-size:10px">CLOSE · N</div>'
      + '</div>'

      // ---- the budget ----
      + '<div class="mono faint" style="font-size:9px;letter-spacing:1.5px;margin-top:14px">'
      + 'HOW MUCH WORK THIS ACADEMY GETS</div>'
      + slider('acad', 'steps', mil, 1, 2000, 1, function (m) { return steps(m * 1e6); })
      + '<div style="display:flex;gap:6px;margin-top:6px;flex-wrap:wrap">'
      + [5, 25, 100, 250, 500, 1000, 2000].map(function (m) {
          return '<div class="chip" data-acad-set="steps" data-value="' + (m * 1e6) + '"'
            + ' style="cursor:pointer;padding:5px 9px;font-size:10px;'
            + (mil === m ? 'background:' + P.rgba(accent, .2) + ';border-color:' + P.rgba(accent, .5) + ';color:#dceaff' : '')
            + '">' + steps(m * 1e6) + '</div>';
        }).join('')
      + '</div>'
      + '<div class="mono faint" style="font-size:10px;margin-top:6px">≈ ' + hms(eta)
      + ' alone on this machine · the run reports its own rate once it starts</div>'

      // ---- shaping ----
      + '<div class="mono faint" style="font-size:9px;letter-spacing:1.5px;margin-top:16px">'
      + 'HOW IT IS TAUGHT · does not change how it is scored</div>'
      + SHAPING_ROWS.map(function (r) {
          return '<div style="margin-top:7px">'
            + '<div class="mono faint" style="font-size:9.5px">' + r[1] + '</div>'
            + slider('shape', r[0], shapingValue(key, r[0]), r[2], r[3], r[4],
                     function (x) { return (+x).toFixed(3); })
            + '</div>';
        }).join('')

      // ---- the rules, shown and locked ----
      + '<div class="mono faint" style="font-size:9px;letter-spacing:1.5px;margin-top:16px">'
      + 'THE RULES OF THE GAME · counted by the scoreboard, so not editable</div>'
      + '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:7px;opacity:.62">'
      + (((App.cat && App.cat.rewards) || []).map(function (r) {
          return '<span class="chip mono" style="font-size:10px;padding:4px 9px">'
            + esc(r.label) + ' <b style="color:#c9d8ee">' + (r.value > 0 ? '+' : '') + r.value + '</b></span>';
        }).join('') || '<span class="dim" style="font-size:11px">the trainer has not sent them</span>')
      + '</div>'

      // ---- the algorithm's own knobs ----
      + '<div class="mono faint" style="font-size:9px;letter-spacing:1.5px;margin-top:16px">'
      + (v.sealed ? 'ITS OWN KNOBS' : esc(v.short) + "'S OWN KNOBS") + '</div>'
      + (tunablesFor(key).length
        ? tunablesFor(key).map(function (t) {
            return '<div style="margin-top:7px">'
              + '<div class="mono faint" style="font-size:9.5px">' + esc(t.label)
              + ' <span style="opacity:.6">· ' + esc(t.hint) + '</span></div>'
              + slider('hyper', t.key, hyperValue(key, t), t.min, t.max, t.step,
                       function (x) { return (t.step >= 1 ? Math.round(x) : (+x).toFixed(4).replace(/0+$/, '')); })
              + '</div>';
          }).join('')
        : '<div class="dim" style="font-size:11px;margin-top:6px">Connect the trainer to see them.</div>')

      + '<div style="display:flex;gap:10px;margin-top:18px;align-items:center;flex-wrap:wrap">'
      + (training
        ? '<div class="btn" data-act="train-stop">STOP THIS ACADEMY · SHIFT+S</div>'
        : '<div class="btn" data-act="train">TRAIN THIS ACADEMY</div>')
      + '<div class="mono faint" style="font-size:10px">seed</div>'
      + '<input data-acadnum="seed" value="' + esc(a.seed) + '" spellcheck="false"'
      + ' style="width:64px;background:rgba(255,255,255,.05);border:1px solid var(--line);'
      + 'border-radius:8px;color:#f2f7ff;font-family:JetBrains Mono,monospace;font-size:13px;padding:6px 8px;outline:none">'
      + '<div class="btn ghost" data-act="acad-reset" style="padding:6px 12px;font-size:11px">RESET KNOBS</div>'
      + '</div></div>';
  }

  /* ---------------- scrubbing the run ----------------
   *
   * Every evaluation keeps the weights it was measured on, so the run is a reel rather
   * than a single outcome. Drag the handle and the arena is replayed by the brain the
   * school had at that step — the early frames wander, the late ones hunt — and the two
   * marks say where Tom peaked and where Jerry did, which are rarely the same place.
   *
   * The strip is repainted on its own rather than through `render()`: a snapshot lands
   * every few seconds and a full rebuild would tear the handle out from under a drag.
   */
  /* Holes are the enemy here. A snapshot arrives with an absolute index, and a reload, a
     silent socket reconnect or a second window can leave the array starting part-way
     through — every reader then dereferenced `undefined.catExam` inside render(), which
     throws BEFORE the screen is written, so the school screen froze for good and every
     later event threw again. Filtering is one cheap pass over at most a few hundred
     entries and it makes the rest of the reel unable to reach a hole. */
  function frames(key) { return (App.timeline[key || App.school] || []).filter(Boolean); }

  /* The reel plays into the SHADOW arena, and that arena exists in train mode alone —
     `pin` is refused everywhere else, on purpose, because pinning an arena that goes on
     playing something else is a control that reports success and changes nothing.

     But "the arena is not in train mode" and "there is nothing to scrub" are different
     facts, and treating them as one is what made a two-second look at UNTRAINED cost the
     reel for the rest of a forty-minute run. While a run for THIS school is live the
     shadow arena is one `shadow` command away, and `needsShadow()` is where the handle
     says so — see `pinTo`, which sends it. Scrubbing is off only when there is genuinely
     no live run behind the reel: a finished run read off disk has nothing to go back to. */
  function scrubbable() { return App.mode === 'train' || trainingHere(); }
  function needsShadow() { return App.mode !== 'train' && trainingHere(); }

  function addFrame(m) {
    var tl = App.timeline[m.school] || (App.timeline[m.school] = []);
    while (tl.length < m.i) tl.push(null);
    tl[m.i] = m;
  }

  function peakIndex(role) {
    var f = frames(), best = -1, bv = -1;
    for (var i = 0; i < f.length; i++) {
      var v = role === 'cat' ? f[i].catExam : f[i].mouseExam;
      if (v > bv) { bv = v; best = i; }
    }
    return best;
  }

  function timelineStrip() {
    var f = frames();
    if (!f.length) {
      return '<div id="timeline" class="card" style="padding:12px 14px;margin-top:10px">'
        + '<div class="mono faint" style="font-size:10px;letter-spacing:1.5px">THE REEL</div>'
        + '<div class="dim" style="font-size:12px;margin-top:6px">Nothing has been trained here yet. '
        + 'Train this academy and a frame is kept every time it is scored — this becomes a graph '
        + 'of who is winning and a slider you can drag back through the run.</div></div>';
    }
    // The graph is history and stays readable everywhere. The HANDLE is the part that
    // only means anything while the shadow arena is up, so that is the part that goes
    // grey — a disabled slider explains itself before the drag, where a refusal banner
    // only explains itself after it.
    var can = scrubbable();
    var at = !can || App.pinned === null || App.pinned === undefined ? f.length - 1 : App.pinned;
    return '<div id="timeline" class="card" style="padding:12px 14px;margin-top:10px">'
      + '<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">'
      + '<div class="mono faint" style="font-size:10px;letter-spacing:1.5px">THE REEL</div>'
      + '<div id="tl-legend">' + graphLegend() + '</div>'
      + (can
         ? '<div class="btn ghost" data-act="tl-live" style="margin-left:auto;padding:4px 10px;font-size:10px;'
           + (App.pinned === null ? 'border-color:rgba(124,188,255,.5);color:#dceaff' : '') + '">LIVE</div>'
         : '<div class="mono faint" style="margin-left:auto;font-size:10px;letter-spacing:1.5px">READ ONLY</div>')
      + '</div>'
      + '<div id="tl-marks" style="position:relative;height:' + GRAPH_H + 'px;margin-top:6px">'
      + timelineMarks() + '</div>'
      + '<input type="range" id="tl" min="0" max="' + (f.length - 1) + '" step="1" value="' + at + '"'
      + (can ? '' : ' disabled')
      + ' style="width:100%;height:18px;accent-color:' + (can ? 'var(--gold)' : 'rgba(130,160,200,.3)')
      + (can ? '' : ';opacity:.45;cursor:not-allowed') + '">'
      + '<div id="tl-label" class="mono" style="font-size:12.5px;color:#c9d8ee;min-height:16px">'
      + timelineLabel() + '</div>'
      + '</div>';
  }

  /* Who is winning, across the run. Both series are rates, so they share ONE axis fixed
     at 0..100% — normalising each to its own maximum (which is what this did first) put
     Tom at 4% and Jerry at 62% both against the ceiling, made every crossing an artefact,
     and re-scaled the whole history every time a new maximum arrived, so a climbing line
     visibly flattened while the numbers behind it kept rising. */
  var GRAPH_H = 96;

  function timelineMarks() {
    var f = frames();
    var W = 1000, H = GRAPH_H, pad = 6;
    var y = function (v) { return pad + (1 - Math.max(0, Math.min(1, v))) * (H - 2 * pad); };
    var x = function (j) { return f.length < 2 ? W : j / (f.length - 1) * W; };
    var g = ['<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" width="100%" height="'
             + H + '" style="display:block">'];
    // A grid the eye can read a percentage off, and 50% called out.
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      g.push('<line x1="0" y1="' + y(v).toFixed(1) + '" x2="' + W + '" y2="' + y(v).toFixed(1)
        + '" stroke="rgba(180,205,240,' + (v === 0.5 ? '.20' : '.08') + ')" stroke-width="1"/>');
    });
    if (f.length >= 2) {
      [['catExam', 'var(--cat)'], ['mouseExam', 'var(--mouse)']].forEach(function (r) {
        var d = f.map(function (fr, j) {
          return (j ? 'L' : 'M') + x(j).toFixed(1) + ' ' + y(fr[r[0]]).toFixed(1);
        }).join(' ');
        g.push('<path d="' + d + '" fill="none" stroke="' + r[1] + '" stroke-width="2.2"'
          + ' vector-effect="non-scaling-stroke" opacity=".92"/>');
      });
      [['cat', 'var(--cat)'], ['mouse', 'var(--mouse)']].forEach(function (r) {
        var i = peakIndex(r[0]);
        if (i < 0) return;
        g.push('<line x1="' + x(i).toFixed(1) + '" y1="0" x2="' + x(i).toFixed(1) + '" y2="' + H
          + '" stroke="' + r[1] + '" stroke-width="1" stroke-dasharray="3 3" opacity=".55"/>');
      });
    }
    // Where the handle is, so the slider and the graph are visibly the same axis.
    if (f.length) {
      var at = App.pinned === null || App.pinned === undefined ? f.length - 1
               : Math.max(0, Math.min(f.length - 1, App.pinned));
      g.push('<line x1="' + x(at).toFixed(1) + '" y1="0" x2="' + x(at).toFixed(1) + '" y2="' + H
        + '" stroke="#f2f7ff" stroke-width="1.5" opacity=".75"/>');
    }
    g.push('</svg>');
    return g.join('');
  }

  function graphLegend() {
    var f = frames(), last = f[f.length - 1];
    var sw = function (col, label, val) {
      return '<span class="mono" style="display:inline-flex;align-items:center;gap:5px;font-size:10.5px;color:#9fb2cd">'
        + '<span style="width:14px;height:2px;background:' + col + ';display:inline-block"></span>'
        + label + (val === undefined ? '' : ' <b style="color:#f2f7ff">' + pct(val) + '</b>') + '</span>';
    };
    return '<div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap">'
      // Deliberately NOT the score boxes' words. Those count episodes played on screen;
      // these are the trainee measured against the scripted Examiner on held-out rooms.
      // Two different quantities were sharing one label, a hand's width apart.
      + sw('var(--cat)', 'TOM SCORED', last && last.catExam)
      + sw('var(--mouse)', 'JERRY SCORED', last && last.mouseExam)
      + '<span class="mono faint" style="font-size:9.5px">vs the Examiner · held-out rooms · 0–100%</span>'
      + '</div>';
  }

  function timelineLabel() {
    var f = frames();
    if (!f.length) return '';
    // Read-only is a state worth naming. Without this the line read "PINNED · the brain
    // at 153M steps" while the arena played the finished policy and the trainer had
    // already refused the pin — the screen taking credit for something that did not
    // happen, which is the one thing this app is not allowed to do.
    if (!scrubbable()) {
      var last = f[f.length - 1];
      return f.length + ' checkpoints kept · latest at ' + steps(last.steps) + ' steps · Tom '
        + pct(last.catExam) + ' · Jerry ' + pct(last.mouseExam)
        + ' — read only here: the reel drives the shadow arena, not this one';
    }
    // Where these frames came from. The server serves the newest take for a school when
    // the run directory has no reel of its own, so the numbers on screen can belong to a
    // run nobody in the room started.
    var src = App.training && App.training.school === App.school && !App.training.finished
      ? '' : ' · from the last saved take';
    var live = App.pinned === null || App.pinned === undefined;
    var fr = f[live ? f.length - 1 : Math.max(0, Math.min(f.length - 1, App.pinned))];
    var pc = peakIndex('cat'), pm = peakIndex('mouse');
    var where = 'the brain at ' + steps(fr.steps) + ' steps · Tom ' + pct(fr.catExam)
      + ' · Jerry ' + pct(fr.mouseExam)
      + (!live && App.pinned === pc ? ' · TOM PEAKS HERE' : '')
      + (!live && App.pinned === pm ? ' · JERRY PEAKS HERE' : '');
    if (live) {
      return trainingHere()
        ? 'LIVE · ' + where + ' · training continues behind it'
        : 'the latest brain · ' + where.replace('the brain at ', 'at ') + src;
    }
    // Beat 7 of the shoot: watching an old brain while the run carries on. Both facts,
    // in the one line the author will be pointing at.
    // `trainingHere()`, not "anything is training" — standing on GA's screen while PPO
    // trained used to print "training continues behind it" over GA's frames.
    return (trainingHere() ? 'PINNED · ' : 'showing · ') + where
      + (trainingHere() ? ' · training continues behind it' : src);
  }

  /* Update in place. `max` grows as frames arrive and the handle stays where it was,
     which is what lets you sit on an early frame while the run carries on behind it. */
  function paintTimeline() {
    var host = el('timeline');
    if (!host) return;
    var f = frames(), input = el('tl');
    if (!input) { host.outerHTML = timelineStrip(); return; }
    if (+input.max !== f.length - 1) input.max = String(f.length - 1);
    // `render()` is not guaranteed between a mode change and the next frame — the card is
    // repainted in place. Sync the handle's own enabled state here or a reel that went
    // read-only mid-run stays draggable until something else forces a full rebuild.
    if (input.disabled === scrubbable()) { host.outerHTML = timelineStrip(); return; }
    if (!scrubbable() || App.pinned === null || App.pinned === undefined) {
      input.value = String(f.length - 1);
    }
    var marks = el('tl-marks'), label = el('tl-label'), leg = el('tl-legend');
    if (marks) marks.innerHTML = timelineMarks();
    if (label) label.innerHTML = timelineLabel();
    if (leg) leg.innerHTML = graphLegend();
  }

  /* ---------------- the training screen ----------------
   *
   * Everything a run needs, set here rather than on a command line: the budget, PPO's
   * batch, the holes per room, the seed and the tag it writes under. The same settings
   * drive both kinds of run — the full three-school one, which is the real trainer in
   * three processes, and the single school trained on camera — so there is exactly one
   * place where "how much work" is decided.
   */

  function field(label, key, value, sub, width) {
    return '<div style="width:' + (width || 120) + 'px">'
      + '<div class="mono faint" style="font-size:9px;letter-spacing:1.5px">' + label + '</div>'
      + '<input data-setup="' + key + '" value="' + esc(value === null || value === undefined ? '' : value) + '"'
      + ' spellcheck="false" style="width:100%;margin-top:5px;background:rgba(255,255,255,.05);'
      + 'border:1px solid var(--line);border-radius:8px;color:#f2f7ff;font-family:JetBrains Mono,monospace;'
      + 'font-size:16px;padding:8px 10px;outline:none">'
      + (sub ? '<div class="mono faint" style="font-size:10px;margin-top:4px">' + sub + '</div>' : '')
      + '</div>';
  }

  function chips(key, values, current, fmt) {
    return values.map(function (v) {
      var on = String(current) === String(v);
      return '<div class="chip" data-setup-set="' + key + '" data-value="' + v + '"'
        + ' style="cursor:pointer;padding:7px 12px;' + (on
          ? 'background:rgba(124,188,255,.18);border-color:rgba(124,188,255,.5);color:#dceaff' : '') + '">'
        + (fmt ? fmt(v) : v) + '</div>';
    }).join('');
  }

  /* One school's row in the full run. Deliberately the same four numbers as the live
     HUD — iteration, steps against the target, rate, elapsed and what is left — because
     they answer the same question and should not be learned twice. */
  function runProgressHtml() {
    var sc = App.scoring;
    if (sc) {
      var tail = (sc.lines || []).slice(-14).map(esc).join('<br>');
      return '<div class="card" style="padding:16px 18px;flex:1;display:flex;flex-direction:column;min-height:0">'
        + '<div class="mono faint" style="font-size:10px;letter-spacing:1.6px">'
        + (sc.done ? 'SCAN FINISHED' : 'SCORING · CROSS-PLAY, THEN THE HIGHLIGHT SCAN') + '</div>'
        + '<div class="mono" style="margin-top:10px;font-size:12px;line-height:1.7;color:#8fa4c4;overflow:hidden">'
        + (tail || 'starting…') + '</div>'
        + (sc.done ? '<div class="dim" style="margin-top:auto;font-size:13px">'
            + (sc.ok ? 'The leaderboard and the grand final are live — press B or F.'
                     : 'The scan did not finish. The lines above say where it stopped.')
            + '</div>' : '')
        + '</div>';
    }
    var r = App.runAll;
    if (r) {
      return '<div style="display:flex;flex-direction:column;gap:10px">'
        + '<div class="mono faint" style="font-size:10px;letter-spacing:1.6px">'
        + (r.done ? 'RUN FINISHED · ' + esc(r.tag)
                  : 'TRAINING · ' + esc(r.tag) + ' · three processes, one budget each') + '</div>'
        + ORDER.map(function (k) {
            var v = view(k), t = r.schools[k];
            var frac = t && t.frac !== undefined ? Math.max(0, Math.min(1, t.frac)) : 0;
            var col = v.sealed ? '#8494ad' : v.color;
            var nm = v.sealed ? 'SCHOOL ' + (ORDER.indexOf(k) + 1) : v.short;
            return '<div class="card" style="padding:12px 16px;border-color:' + P.rgba(col, .28) + '">'
              + '<div style="display:flex;align-items:baseline;gap:10px">'
              + '<div class="mono" style="font-size:13px;letter-spacing:1.6px;color:' + col + '">' + esc(nm) + '</div>'
              + '<div class="mono" style="margin-left:auto;font-size:17px;color:' + col + '">' + pct(frac) + '</div></div>'
              + '<div style="height:5px;border-radius:3px;background:rgba(255,255,255,.06);margin:7px 0 9px;overflow:hidden">'
              + '<div style="height:100%;width:' + (frac * 100).toFixed(2) + '%;background:' + col + '"></div></div>'
              + '<div class="mono" style="display:flex;gap:16px;font-size:12px;color:#c9d8ee;flex-wrap:wrap">'
              + '<span>it <b>' + (t ? t.iter : '—') + '</b></span>'
              + '<span>' + (t ? steps(t.steps) : '—') + (t && t.targetSteps ? ' / ' + steps(t.targetSteps) : '') + '</span>'
              + '<span>' + (t && t.sps ? steps(t.sps) + '/s' : '—') + '</span>'
              + '<span>' + (t ? hms(t.wall) : '—') + (t && t.eta != null ? ' · ' + hms(t.eta) + ' left' : '') + '</span>'
              + (t && t.best && t.best.cat ? '<span style="color:var(--cat)">Tom ' + pct(t.best.cat.rate) + '</span>' : '')
              + (t && t.best && t.best.mouse ? '<span style="color:var(--mouse)">Jerry ' + pct(t.best.mouse.rate) + '</span>' : '')
              + '</div></div>';
          }).join('')
        + (r.done ? '<div class="dim" style="font-size:13px;line-height:1.6">Each school kept the best '
            + 'Tom and the best Jerry it reached. Score the run to settle the championship.</div>' : '')
        + '</div>';
    }
    // Idle: the runs that exist, so switching to one is a click rather than a restart.
    var runs = (App.cat && App.cat.runs) || [];
    return '<div style="display:flex;flex-direction:column;gap:10px">'
      + '<div class="mono faint" style="font-size:10px;letter-spacing:1.6px">RUNS ON DISK</div>'
      + (runs.length ? runs.map(function (x) {
          return '<div class="card" style="padding:11px 15px;display:flex;align-items:center;gap:12px;'
            + (x.current ? 'border-color:rgba(124,188,255,.45)' : '') + '">'
            + '<div class="mono" style="font-size:14px;color:#f2f7ff;width:120px">' + esc(x.tag) + '</div>'
            + '<div class="mono faint" style="font-size:11px;flex:1">'
            + (x.schools.length ? x.schools.join(' · ') : 'no checkpoints yet')
            + (x.budget ? ' · ' + esc(x.budget) : '') + '</div>'
            + '<div class="chip" style="font-size:10px">' + (x.scored ? 'SCORED' : 'NOT SCORED') + '</div>'
            + (x.current ? '<div class="chip" style="font-size:10px;color:#7cbcff">WATCHING</div>'
                         : '<div class="btn ghost" data-act="use-run" data-tag="' + esc(x.tag)
                           + '" style="padding:6px 12px;font-size:11px">WATCH</div>')
            + '</div>';
        }).join('') : '<div class="dim" style="font-size:13px">Nothing has been trained yet.</div>')
      + '</div>';
  }

  /* The run-level screen. Training moved into the academies, so what is left here is
     what belongs to the run as a whole: which run is being watched, and settling the
     championship on it. The three cards are a summary and a way in — the budget is set
     inside the academy, because that is where you are looking at the school. */
  function renderSetup() {
    var S2 = App.setup;
    var scoring = App.scoring && !App.scoring.done;

    var cards = ORDER.map(function (k) {
      var v = view(k), a = acadOf(k), n = (App.timeline[k] || []).length;
      var live = !!(App.training && !App.training.finished && App.training.school === k);
      return '<div class="card" data-school="' + k + '" style="flex:1;cursor:pointer;padding:14px 16px;'
        + 'border-color:' + P.rgba(v.color, .3) + '">'
        + '<div class="mono" style="font-size:12px;letter-spacing:1.6px;color:'
        + (v.sealed ? '#8494ad' : v.color) + '">' + esc(v.sealed ? 'SCHOOL ' + (ORDER.indexOf(k) + 1) : v.short) + '</div>'
        + '<div class="mono" style="font-size:19px;margin-top:6px;color:#f2f7ff">' + steps(a.steps) + '</div>'
        + '<div class="mono faint" style="font-size:10px">steps budgeted · seed ' + esc(a.seed) + '</div>'
        + '<div class="mono faint" style="font-size:10px;margin-top:6px">'
        + (live ? '<span style="color:' + v.color + '">training now</span>'
                : n ? n + ' frames on the reel' : 'not trained in this session')
        + '</div></div>';
    }).join('');

    return '<div class="screen">' + backdrop('bg-academy', .42)
      + '<div style="position:relative;display:flex;align-items:center;gap:18px">'
      + '<div><div class="kicker">The run as a whole</div>'
      + '<div class="title" style="font-size:44px;margin-top:4px">RUNS</div></div>'
      + '<div style="margin-left:auto;display:flex;gap:10px;align-items:center">'
      + '<span class="chip">RUN ' + esc((App.cat && App.cat.runTag) || '—') + '</span>'
      + revealChip() + statusChip() + '</div></div>'

      + '<div style="position:relative;display:flex;gap:20px;margin-top:18px;flex:1;min-height:0">'
      + '<div style="width:720px;display:flex;flex-direction:column;gap:16px">'

      + '<div class="card" style="padding:18px 20px">'
      + '<div class="mono faint" style="font-size:10px;letter-spacing:1.6px">'
      + 'THE THREE ACADEMIES · each one is trained from its own screen</div>'
      + '<div style="display:flex;gap:12px;margin-top:12px">' + cards + '</div>'
      + '<div class="dim" style="margin-top:12px;font-size:12.5px;line-height:1.6">'
      + 'Click one, then press <span class="mono" style="color:#c9d8ee">n</span> to set its '
      + 'budget, how hard each side is pulled toward its goal, and its own knobs — and to '
      + 'train it while you watch.</div></div>'

      + '<div class="card" style="padding:18px 20px">'
      + '<div class="mono faint" style="font-size:10px;letter-spacing:1.6px">SETTLE THE CHAMPIONSHIP</div>'
      + '<div style="display:flex;gap:18px;margin-top:12px;align-items:flex-start;flex-wrap:wrap">'
      + field('TAG · WRITES TO runs/', 'tag', S2.tag, 'the run being scored', 180)
      + '<div><div class="mono faint" style="font-size:9px;letter-spacing:1.5px">EACH SCHOOL ENTERS</div>'
      + '<div style="display:flex;gap:8px;margin-top:6px">'
      + chips('scoreWith', ['trained', 'best'], S2.scoreWith, function (v) {
          return v === 'best' ? 'BEST REACHED' : 'AS FINISHED'; })
      + '</div></div></div>'
      + '<div style="display:flex;gap:10px;margin-top:16px;flex-wrap:wrap">'
      + '<div class="btn" data-act="score"' + (scoring || App.link !== 'live' ? ' style="opacity:.45"' : '')
      + '>SCORE THE RUN</div>'
      + '<div class="btn ghost" data-act="menu">ESC · ACADEMY</div></div>'
      + (App.setupNote ? '<div class="mono" style="font-size:12.5px;color:#ff9a72;margin-top:12px;line-height:1.5">'
          + esc(App.setupNote) + '</div>' : '')
      + '<div class="dim" style="margin-top:12px;font-size:12.5px;line-height:1.6">'
      + (App.link === 'live'
        ? 'The cross-play tournament and the highlight scan, on arenas nobody trained on. '
          + 'The leaderboard and the grand final are live the moment it finishes.'
        : '<span style="color:#ff9a72">The trainer is not running.</span> Start it from the '
          + 'project folder with <span class="mono" style="color:#c9d8ee">./run.sh serve</span> — '
          + 'this page reconnects by itself.')
      + '</div></div></div>'

      + '<div id="run-progress" style="flex:1;min-width:0;display:flex;flex-direction:column">'
      + runProgressHtml() + '</div>'
      + '</div></div>';
  }

  /* Progress arrives a couple of times a second, and this screen has text fields in it.
     A full re-render would take the caret out of whatever the author was typing, so the
     right-hand column is repainted on its own. */
  function paintRunProgress() {
    var host = el('run-progress');
    if (host) host.innerHTML = runProgressHtml();
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
        + 'Score it from the RUNS screen — press <span class="mono">n</span>, then SCORE THE RUN.</div>'
        + '<div class="btn ghost" style="margin-top:24px" data-act="menu">ESC · ACADEMY</div></div></div>';
    }
    var ck = st.catSchool || (t && t.champion.cat) || 'ppo';
    var mk = st.mouseSchool || (t && t.champion.mouse) || 'ppo';
    var cv = view(ck), mv = view(mk);
    // `wins` is two different shapes. The final counts {cat, mouse, draw}; a race counts
    // one {catch, escape, draw} per LANE, keyed by lane. Both ride in on `runState`, so
    // coming to the final straight off a race printed "draws undefined" on screen.
    var w0 = st.wins;
    var wins = (w0 && typeof w0.cat === 'number' && typeof w0.draw === 'number')
      ? w0 : { cat: 0, mouse: 0, draw: 0 };
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
      + '<div class="btn ghost" data-act="board">LEADERBOARD · B</div>'
      + '<div class="btn ghost" data-act="refresh">REFRESH RESULTS</div></div></div>';
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
  /* The lesson prose carries the same numbers the identity cards do, off the same knobs.
     Filmed exactly as SHOOT.md directs — press 2, then l — the heading read "Forty-eight
     brains" with the panel beside it reading "120 BRAINS · 15 SURVIVE, 105 ARE REPLACED",
     both on screen at once. `mu = lam // 2` in the CMA-ES setup, so "the better half"
     stays true at any lambda and is left alone. */
  var LIVE_LESSON = {
    ga: function (L, t) {
      return Object.assign({}, L, {
        head: t.pop_size + ' brains, and only the good ones get children',
        body: 'No gradients, no derivatives, nothing that knows which direction is better. '
          + 'Each generation plays all ' + t.pop_size + ' networks, ranks them, keeps the best '
          + t.elite + ' untouched, and fills the rest with children: pick two parents, flip a '
          + 'coin per weight to decide which parent it comes from, then nudge '
          + Math.round(t.mutation_rate * 100) + '% of the weights at random. Repeat a few '
          + 'thousand times.'
      });
    },
    cmaes: function (L, t) {
      return Object.assign({}, L, {
        body: 'CMA-ES keeps a Gaussian over strategies. Every generation it draws ' + t.lam
          + ' brains from it, keeps the better half, and moves the centre toward them — then '
          + 'reshapes the cloud itself, stretching along directions that keep paying off and '
          + 'narrowing where they do not. The ellipse is that shape, measured from the real '
          + 'samples on screen.'
      });
    }
  };

  function lessonOf(key) {
    var L = LESSON[key];
    if (!L || !LIVE_LESSON[key]) return L;
    var t = tunablesNow(key);
    return t ? LIVE_LESSON[key](L, t) : L;
  }

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

  /* Twenty-two was sized for three panes: 3 x 27 x 22 is 1782 and fits. Six panes at the
     same cell is 3564 across a canvas 1920 wide, so the last ones ran off the right edge
     with nothing saying they had.
     Six in one row does fit, at ten pixels a cell, and leaves the bottom third of the
     canvas empty — width is the binding constraint there, so spare height cannot help.
     Two rows of three is what actually uses the frame: the cell roughly doubles. */
  function raceLayout(n) {
    n = Math.max(1, n);
    // Every arrangement is tried and the one with the biggest cell wins, rather than a
    // rule of thumb per lane count. Three across was right for six lanes and wrong for
    // four: four in one row gives a 16px cell against 15 in two rows of two, and it also
    // leaves 3px of slack per card instead of 244 — the arena inside a card is a grid of
    // square cells, so it cannot stretch to fill a card that is wider than it needs.
    // Ties go to fewer rows.
    var compact = n > 3;
    // One row per lane in the twelve-room matrix, so its height is not a constant: the
    // 190px written for three lanes put the last row of six at y=1082 on a canvas 1080
    // tall and ran the one above it through the button bar.
    var matrix = 46 + n * (compact ? 26 : 44);
    var best = null;
    for (var perRow = 1; perRow <= n; perRow++) {
      var rows = Math.ceil(n / perRow);
      var wAvail = 1920 - 40 - 14 * (perRow - 1) - 24 * perRow;  // padding, gaps, card padding
      // The 56 is the button bar; the 30 after it is slack, because the header and each
      // card's own title row measure a little larger than this estimate, and the failure
      // mode is the matrix sliding under the buttons rather than anything visibly broken.
      var hAvail = 1080 - 36 - 84 - 14 * (rows - 1) - matrix - 56 - 30;
      var cs = Math.max(6, Math.min(RCS, Math.floor(wAvail / perRow / E.W),
                                    Math.floor((hAvail / rows - 70) / E.H)));
      if (!best || cs > best.cs || (cs === best.cs && rows < best.rows)) {
        best = { cs: cs, perRow: perRow, rows: rows, compact: compact };
      }
    }
    // Once the cell is bound by WIDTH — four lanes in one row is — the height left over
    // is not small, and it has to go somewhere. Giving the matrix `flex:1` and letting it
    // take all of it produced 130px cells holding one letter, which looks more broken
    // than the gap did. The rows get the slack up to a ceiling instead, and whatever is
    // still spare becomes bottom margin, which reads as composition rather than a void.
    var paneBlock = best.rows * (E.H * best.cs + 70) + 14 * (best.rows - 1);
    var left = 1080 - 36 - 84 - paneBlock - 56 - 14 - 46 - 32;
    best.rowH = Math.max(26, Math.min(96, Math.floor(left / n)));
    return best;
  }

  /* A lane is a SCHOOL in the three-school race and a GENOME in the population race.
     Everything that draws one asks here, because `view()` looks a key up in ALGOS and a
     genome has no entry there — it is one member of one generation of one school. */
  function laneView(k) {
    if (App.raceKind !== 'population') return view(k);
    var lanes = App.raceLanes || [];
    var i = -1;
    for (var j = 0; j < lanes.length; j++) if (lanes[j].key === k) { i = j; break; }
    var L = i >= 0 ? lanes[i] : null;
    // The distribution's centre is not a rank and has no fitness -- it was never drawn
    // and never scored. Gold, and named for what it is, because it is the only lane the
    // arena is actually playing.
    if (L && L.mean) {
      return { color: 'var(--gold)', short: 'THE CENTRE', sub: 'never sampled · in the arena',
               kept: true, mean: true, sealed: false, emblem: null };
    }
    var scored = lanes.filter(function (x) { return !x.mean; });
    var si = scored.indexOf(L);
    var half = Math.ceil(scored.length / 2) || 1;
    var kept = si >= 0 && si < half;
    // Two families of colour, not N arbitrary ones: better and worse is the split the
    // screen exists to show, and unrelated hues would hide it.
    var hue = kept ? 96 + si * 24 : 6 + (si - half) * 14;
    return {
      color: 'hsl(' + hue + ',64%,' + (kept ? 58 : 56) + '%)',
      short: L ? 'RANK ' + L.rank : k,
      sub: L ? 'fitness ' + (+L.fitness).toFixed(3) : '',
      kept: kept, sealed: false, emblem: null
    };
  }

  function renderRace() {
    var pop = App.raceKind === 'population';
    var schools = App.raceSchools || ORDER;
    var lay = raceLayout(schools.length), cs = lay.cs;
    var wins = App.raceWins || {};
    var lanes = schools.map(function (k) {
      var v = laneView(k);
      var w = wins[k] || { catch: 0, escape: 0, draw: 0 };
      var done = App.raceDone && App.raceDone[k];
      return '<div class="card" style="padding:12px;border-color:' + P.rgba(v.color, .3)
        + ';flex:' + (lay.rows === 1 ? '1'
            : '0 0 calc((100% - ' + (14 * (lay.perRow - 1)) + 'px) / ' + lay.perRow + ')') + '">'
        + '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">'
        + (pop
           ? '<div style="width:26px;height:26px;flex:0 0 auto;border-radius:6px;background:'
             + P.rgba(v.color, .22) + ';border:1px solid ' + P.rgba(v.color, .55) + '"></div>'
           : '<div style="width:26px;height:26px;flex:0 0 auto">' + P.emblem(v.emblem, v.color) + '</div>')
        + '<div><div class="title" style="font-size:' + (pop ? 18 : 22) + 'px;color:'
        + (v.sealed ? '#8494ad' : v.color) + '">' + esc(v.short) + '</div>'
        + (pop ? '<div class="mono faint" style="font-size:10px">' + esc(v.sub) + '</div>' : '')
        + '</div>'
        + '<div style="margin-left:auto;display:flex;gap:8px">'
        + '<span class="mono" style="font-size:15px;color:var(--cat)">' + w.catch + '</span>'
        + '<span class="faint">/</span>'
        + '<span class="mono" style="font-size:15px;color:var(--mouse)">' + w.escape + '</span>'
        + '<span class="faint mono" style="font-size:12px">' + w.draw + '</span></div></div>'
        + '<div style="position:relative;width:' + (E.W * cs) + 'px;height:' + (E.H * cs) + 'px;border-radius:9px;overflow:hidden;border:1px solid var(--line)">'
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
      + (pop
         ? '<div><div class="kicker">Same room · same spawns · same noise · same opponent · '
           + schools.length + ' brains out of one generation</div>'
           + '<div class="title" style="font-size:44px;margin-top:4px">'
           // CMA-ES keeps nobody. There is no elite to carry over and no pair to breed:
           // the whole sample is thrown away every generation and the next one is drawn
           // fresh from a distribution, so KEPT AND REPLACED would name a mechanism this
           // school does not have.
           + (App.raceSchool === 'cmaes' ? 'THE BEST AND WORST DRAWS' : 'KEPT AND REPLACED')
           + '</div></div>'
         : '<div><div class="kicker">Same room · same spawns · same noise · three different brains</div>'
           + '<div class="title" style="font-size:44px;margin-top:4px">SIDE BY SIDE</div></div>')
      + '<div style="display:flex;gap:10px">' + revealChip() + statusChip() + '</div></div>'
      + '<div style="position:relative;display:flex;flex-wrap:wrap;gap:14px;margin-top:14px">' + lanes + '</div>'
      + raceMatrix(schools, lay.compact, lay.rowH)
      // Anchored to the bottom of the frame, like every other screen's button bar. Any
      // slack then sits BETWEEN two blocks of content rather than as a black band under
      // the last one, which is what it looks like in a recorded frame.
      + '<div style="position:relative;display:flex;gap:10px;margin-top:auto;align-items:center">'
      + '<div class="btn ghost" data-act="menu">ESC · ACADEMY</div>'
      + '<div class="btn ghost" data-act="pause">' + (App.playing ? 'PAUSE · SPACE' : 'RESUME · SPACE') + '</div>'
      + (pop ? '' : '<div class="btn ghost" data-act="race">RESTART · X</div>')
      + '<div class="dim" style="margin-left:auto;font-size:12.5px">'
      + 'Arena ' + (((App.runState || {}).level || 0) + 1) + ' of ' + (App.cat ? App.cat.levels.length : 12)
      + (pop
         ? ' &nbsp;·&nbsp; generation ' + (App.raceGen === undefined ? '—' : App.raceGen)
           + ' \u00b7 ' + (App.racePop || '—') + ' brains · '
           + (App.raceSchool === 'cmaes'
              // The one fact that separates this school from the other two, and it is not
              // visible in any pane: what the arena plays is the centre of the cloud, a
              // brain that was never drawn and never scored. See `best = mean` in tell().
              ? 'none of these survives — the cloud keeps the direction they point in, not '
                + 'the brains · the arena plays its centre, which was never one of them'
              : 'the left ones were kept, the right ones were replaced')
           + ' · nothing differs between the panes except the genome.'
         : ' &nbsp;·&nbsp; nothing differs between the three panes except the policy.')
      + '</div></div></div>';
  }

  /* The same twelve rooms, three schools, one grid. Reading down a column tells you
     whether a room is hard or whether one school simply solved it — which is the
     comparison the side-by-side view exists to make. */
  /* Sized by flex rather than arithmetic. Once the cell is bound by WIDTH — four lanes
     in one row is — the height left over is not small: the matrix ended at y=660 with the
     buttons at 1018, and 358px of nothing between them. Rather than compute that gap and
     spend it, the matrix is given `flex:1` and its rows share whatever is there. */
  function raceMatrix(schools, compact, rowH) {
    var pad = compact ? '3px 0' : '7px 0';
    var gap = compact ? 3 : 6;
    var n = App.cat ? App.cat.levels.length : 12;
    var grid = App.raceGrid || {};
    var head = '<div style="display:flex;gap:4px;margin-left:104px">';
    for (var i = 0; i < n; i++) {
      head += '<div class="mono faint" style="flex:1;text-align:center;font-size:9px">'
        + String(i + 1).padStart(2, '0') + '</div>';
    }
    head += '</div>';

    var rows = schools.map(function (k) {
      var v = laneView(k);
      var r = grid[k] || [];
      var c = r.filter(function (x) { return x === 'catch'; }).length;
      var m = r.filter(function (x) { return x === 'escape'; }).length;
      var cells = '';
      for (var i = 0; i < n; i++) {
        var x = r[i];
        cells += '<div style="flex:1;align-self:stretch;display:flex;align-items:center;'
          + 'justify-content:center;text-align:center;padding:' + pad + ';border-radius:5px;border:1px solid '
          + (x === 'catch' ? 'rgba(255,138,92,.36)' : x === 'escape' ? 'rgba(126,224,255,.32)'
             : x ? 'var(--line)' : 'rgba(130,160,200,.09)')
          + ';background:' + (x === 'catch' ? 'rgba(255,122,84,.14)' : x === 'escape' ? 'rgba(110,226,255,.12)'
             : x ? 'rgba(255,255,255,.03)' : 'transparent')
          + '"><span class="mono" style="font-size:12px;color:'
          + (x === 'catch' ? '#ff9a72' : x === 'escape' ? '#7ee0ff' : x ? '#8fa4c4' : '#31405a')
          + '">' + (x === 'catch' ? 'T' : x === 'escape' ? 'J' : x ? '–' : '·') + '</span></div>';
      }
      return '<div style="display:flex;align-items:stretch;gap:4px;height:' + rowH
        + 'px;margin-top:' + gap + 'px">'
        + '<div class="mono" style="width:100px;font-size:12px;display:flex;align-items:center;color:'
        + (v.sealed ? '#8494ad' : v.color) + '">' + esc(v.short) + '</div>' + cells
        + '<div class="mono" style="width:78px;justify-content:flex-end;display:flex;'
        + 'align-items:center;font-size:12px">'
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
    var cs = raceLayout(schools.length).cs;
    var local = localMap(App.map);
    for (var i = 0; i < schools.length; i++) {
      var k = schools[i];
      var mh = el('rmap-' + k), fh = el('rfx-' + k);
      if (!mh || !fh) continue;
      // Keyed by cell size as well as seed: six lanes and three lanes draw the same room
      // at different scales, and a cache that only watched the seed kept the old one.
      if (mh.getAttribute('data-k') !== App.map.seed + ':' + cs) {
        mh.setAttribute('data-k', App.map.seed + ':' + cs);
        mh.innerHTML = P.mapSvg(local, cs, { sprites: App.sprites });
      }
      var v = laneView(k);
      var fr = App.raceFrames[k], pv = (App.racePrev && App.racePrev[k]) || fr;
      if (!fr) continue;
      fh.innerHTML = P.fxSvg({
        frame: fr, prev: pv, alpha: App.alpha, cs: cs, map: local,
        key: 'r' + k, now: now, catAccent: v.color, mouseAccent: v.color,
        sprites: App.sprites, spriteFps: App.spriteFps, holdSteps: E.CFG.freezeSteps,
        catMoving: App.alpha < 1 && (fr.cat.x !== pv.cat.x || fr.cat.y !== pv.cat.y),
        mouseMoving: App.alpha < 1 && (fr.mouse.x !== pv.mouse.x || fr.mouse.y !== pv.mouse.y),
        ending: App.raceDone && App.raceDone[k]
          ? { result: App.raceDone[k], ms: now - (App.raceDoneAt[k] || now) } : null
      });
    }
  }

  function renderLesson() {
    var v = view(App.school);
    var L = lessonOf(App.school);
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
      + '<div class="btn ghost" data-act="acad-open">THIS ACADEMY · N</div>'
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
        + 'Score it from the RUNS screen — press <span class="mono">n</span>, then SCORE THE RUN.</div>'
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
      + '<div class="btn ghost" data-act="refresh">REFRESH RESULTS</div>'
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
    ['p', 'kept and replaced — the best and the worst of one live generation, same room'],
    ['g', 'the level generator'], ['l', 'full-screen lesson'],
    ['w', 'how this algorithm works — six steps'],
    ['h', 'the highlight reel'], ['f', 'the grand final'],
    ['b', 'the leaderboard'], ['v', 'this school\'s verdict'],
    ['space', 'pause / resume'], ['s', 'skip this episode'],
    ['[  ]', 'slower / faster'], ['t', 'train live, on camera'], ['n / shift+t', 'training settings and the full run'], ['shift+s', 'end the live run early'],
    ['esc', 'back to the academy'], ['?', 'this card'],
    ['r', 'REVEAL the next school'], ['shift+R', 're-seal one (for a re-shoot)'],
    ['shift+0', 're-seal everything — back to PPO only'],
  ];

  /* The card is an overlay, and it used to be an opaque one: a bare div across the whole
     stage with no pointer rule, so while it was up every button in the app was
     unclickable while every key still worked. It now lets clicks through and closes on
     one. */
  function keyCard() {
    if (!App.showKeys) return '';
    return '<div data-act="keys-close" style="position:absolute;inset:0;z-index:50;'
      + 'background:rgba(3,5,10,.82);display:grid;place-items:center;cursor:pointer">'
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
      + '<br>Click anywhere, or press ? again, to close.'
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
    // The crowd's size is an academy knob, and the explainer says it out loud in three
    // places — two in the GA steps and one in the CMA-ES step that compares itself to a
    // GA. All three mean the same population, so it is passed whatever GA would use.
    var ga = tunablesNow('ga'), cma = tunablesNow('cmaes');
    var facts = {};
    if (ga) facts.POP = ga.pop_size;
    if (cma) facts.LAM = cma.lam;
    host.innerHTML = key ? window.Explain.overlay(v, App.explain, !!opening, facts) : '';
  }

  function openExplain() {
    if (App.explain) return;
    if (App.screen !== 'school') {
      notice('The explainer belongs to a school — press 1, 2 or 3 first, then w.', true);
      return;
    }
    if (view(App.school).sealed) {
      notice('That school is still sealed. Press r to reveal it, then w.', true);
      return;
    }
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

  /* What the settings screen's *form* depends on. Frames, episode results and progress
     all call render() several times a second; rebuilding the form on each of them takes
     the caret out of whatever is being typed and swaps the buttons out from under a
     click. When only the run's progress has moved, repaint just the progress column. */
  function setupSignature() {
    return JSON.stringify([App.setup,
      !!(App.scoring && !App.scoring.done), !!App.scoring,
      !!App.runAll, !!(App.runAll && App.runAll.done),
      // The signature is what lets this screen skip a render. Leaving these out made `?`
      // dead here and left a refused SCORE wedged on "starting…" for good.
      App.showKeys, App.bannerAt,
      ORDER.map(function (k) { return (App.timeline[k] || []).length; }).join(','),
      !!App.training && !App.training.finished, App.school,
      App.cat && App.cat.runTag, ((App.cat && App.cat.runs) || []).length,
      // `live` or not — the screen branches on nothing finer, and the socket flips
      // between 'offline' and 'connecting' every couple of seconds while it retries,
      // which would rebuild the form on each attempt.
      App.link === 'live', App.setupNote]);
  }
  var _setupSig = null;

  /* A refusal has to be readable wherever it was raised. The arena banner needs an arena
     and a frame, so it is invisible on six of the nine screens; this strip is appended to
     every screen and disappears on its own. */
  function noticeStrip() {
    var n = App.notice;
    if (!n || performance.now() - n.at > 9000) return '';
    return '<div id="notice" style="position:absolute;left:50%;top:22px;transform:translateX(-50%);'
      + 'z-index:60;max-width:820px;display:flex;align-items:center;gap:12px;'
      + 'background:rgba(12,17,22,.94);border:1px solid ' + (n.bad ? 'rgba(255,123,114,.55)' : 'var(--line)')
      + ';border-left:3px solid ' + (n.bad ? '#ff7b72' : '#f2b544') + ';border-radius:10px;padding:11px 16px">'
      + '<span class="mono" style="font-size:12.5px;color:' + (n.bad ? '#ff9a8f' : '#e6edf6') + '">'
      + esc(n.text) + '</span>'
      + '<span data-act="notice-close" style="cursor:pointer;color:#7d90ad;font-size:14px;line-height:1">✕</span>'
      + '</div>';
  }

  function notice(text, bad) {
    App.notice = { text: text, at: performance.now(), bad: bad !== false };
    render();
  }

  function render() {
    if (App.screen === 'setup' && _setupSig !== null) {
      var sig = setupSignature();
      if (sig === _setupSig) { paintRunProgress(); return; }
    }
    _setupSig = App.screen === 'setup' ? setupSignature() : null;
    var html = { menu: renderMenu, gen: renderGen, school: renderSchool, lesson: renderLesson,
                 race: renderRace, verdict: renderVerdict, final: renderFinal,
                 board: renderBoard, setup: renderSetup }[App.screen]();
    el('screens').innerHTML = html + noticeStrip() + keyCard();
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
      host.innerHTML = P.mapSvg(localMap(App.map), CS, { sprites: App.sprites });
    }
    if (!App.frame) return;
    var v = view(App.screen === 'final' ? (App.runState && App.runState.catSchool) || App.school : App.school);
    var mv = view(App.screen === 'final' ? (App.runState && App.runState.mouseSchool) || App.school : App.school);
    fx.innerHTML = P.fxSvg({
      frame: App.frame, prev: App.prev, alpha: App.alpha, cs: CS, map: App.map,
      key: 'live', now: now, catAccent: v.color, mouseAccent: mv.color,
      sprites: App.sprites, spriteFps: App.spriteFps, holdSteps: E.CFG.freezeSteps,
      catMoving: App.catMoving, mouseMoving: App.mouseMoving,
      ending: App.ending ? { result: App.ending, ms: now - App.bannerAt } : null
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
    var p = (lesson || trainingHere()) ? App.panels[App.school] : App.livePanel;
    if (!p || !p.draw) return;
    // Drawn at the size the host actually has. `clientWidth`/`clientHeight` are layout
    // pixels and the stage is scaled by a transform, so they ARE canvas pixels — no
    // conversion. The fixed 700x420 letterboxed inside whatever box it was given: on the
    // school screen in playback the host is 666x851, so 452px of that card — more than
    // half of it — was empty, on the screen this video spends most of its time on. Every
    // panel already derives its layout from the width and height it is handed.
    var pw = Math.max(320, host.clientWidth || (lesson ? 1180 : 700));
    var ph = Math.max(240, host.clientHeight || (lesson ? 700 : 420));
    host.innerHTML = p.draw(pw, ph, view(App.school).color);
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
    // The drawer belongs to a school screen. Left set, it drew the academy panel where
    // the explainer should be, and swallowed the next `n` press on the menu.
    if (screen !== 'school') App.acadOpen = false;
    App.screen = screen;
    if (screen === 'gen') { App.levels = []; App.lastGen = 0; }
    render();
  }

  function playSchool(key, cp) {
    // The number keys are documented, so they must obey the same guard the greyed-out
    // school cards enforce — otherwise 1/2/3 answers with an error banner over a dead
    // arena for a school this run does not contain.
    if (App.cat && App.cat.schools && App.cat.schools.indexOf(key) < 0) {
      // Not a banner. The banner needs an arena and a frame to be drawn at all, and this
      // refusal fires from the menu, where there is neither — so the key looked dead.
      notice('That school is not in run ' + String((App.cat && App.cat.runTag) || '?')
             + '. The menu greys the ones this run does not contain.', true);
      return;
    }
    App.acadOpen = false;
    // A pin belongs to the school it was taken from; the server drops it on `play`.
    App.pinned = null;
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
    // A documented key that does nothing, silently, is the one thing the command dispatch
    // was fixed not to do — "a typo, a stale client and a dropped packet were the same
    // event: nothing happened and nothing said so". The same rule has to hold on this side
    // of the socket, because on camera an unexplained key looks like a broken build.
    if (!H || !H.highlights || !H.highlights.length) {
      notice('This run has no highlight scan yet — `./run.sh score` writes one, then h '
             + 'plays it.', true);
      return;
    }
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
    // What the lanes ARE has to go too. This renders before the first lane arrives, and
    // `renderRace` falls back to whatever it was last told — so x pressed after p drew
    // SIDE BY SIDE as a population race, ranks and all, until a frame landed.
    App.raceKind = null; App.raceLanes = null; App.raceSchool = null;
    App.raceSchools = null; App.raceSig = null;
    App.screen = 'race';
    render();
    App.net.send({ cmd: 'race', checkpoint: 'trained' });
  }

  /* What pressing TRAIN overwrote, so a refusal can put it back. Nulling was right when
     the refusal meant "there is no run", and destructive when it meant "training already
     running" -- the run being erased was the live one in ANOTHER school. */
  var _trainSent = false, _trainWas = null;

  function trainLive() {
    var a = acadOf(App.school);
    var b = { minutes: a.minutes || null, steps: a.steps || null };
    if (App.link !== 'live') {
      App.screen = 'setup';
      return note('The trainer is not running — start it with ./run.sh serve.');
    }
    if (!b.steps && !b.minutes) {
      App.screen = 'setup';
      return note('Set a budget first: a step count, a minutes cap, or both.');
    }
    _trainWas = { training: App.training, train: App.train, mode: App.mode,
                  trainReady: App.trainReady, trainFinished: App.trainFinished,
                  pinned: App.pinned, results: App.results,
                  highlights: App.highlights, runState: App.runState };
    _trainSent = true;
    App.setupNote = null;
    App.mode = 'train';
    // `t` is a global key, so it can be pressed from the verdict or the leaderboard.
    // Starting a run and leaving the viewer on a screen that cannot show it is the one
    // outcome nobody wants.
    App.screen = 'school';
    // Not "training live" yet — the optimiser takes a couple of seconds to set up, and
    // the arena holds still until it exists rather than being filled by anything else.
    App.trainReady = false;
    // A fresh run, so the previous take's iteration count and high-water marks go with
    // it. Leaving them up would have the HUD reporting the last run's peak under a bar
    // that just went back to zero.
    App.train = { iter: 0, steps: 0, wall: 0, frac: 0, sps: 0, eta: null,
                  targetSteps: null, budget: b.label, best: {} };
    App.trainInfo = 'Starting the optimiser — the arena waits for the first real policy.';
    // The previous reel stays until the new run lands its first frame — see the snapshot
    // handler. Blanking it here meant a mis-pressed `t` erased the checkpoint that was on
    // screen, mid-sentence, with nothing to put in its place for another two seconds.
    App.pinned = null;
    App.trainFinished = false;
    App.training = { school: App.school, finished: false };
    // The previous playthrough's tally, highlight reel and run state would otherwise sit
    // over the training HUD: score boxes reading 2 / 12 from a different policy, and an
    // arena counter out of a highlight reel that is 10 episodes long, not 12.
    App.results = [];
    App.highlights = null;
    App.runState = null;
    App.net.send({
      cmd: 'train', school: App.school, minutes: b.minutes, steps: b.steps,
      seed: a.seed, shaping: a.shaping, hyper: a.hyper
    });
    render();
  }

  /* The budget in words, for buttons and captions. Both clocks, when both are set. */
  function budgetLabel(key) {
    var a = acadOf(key || App.school), p2 = [];
    if (a.steps) p2.push(steps(a.steps) + ' STEPS');
    if (a.minutes) p2.push(a.minutes + ' MIN CAP');
    return p2.length ? p2.join(' · ') : 'NO BUDGET SET';
  }

  /* The arena banner is drawn into the arena, which the settings screen does not have —
     a refusal announced there would be invisible. This puts it under the buttons, where
     the click that was refused actually happened. */
  function note(msg) {
    App.setupNote = msg;
    render();
  }

  /* Coerce what was typed into what the trainer wants. A step field takes 500M or 2e8
     as happily as 500000000; an empty minutes field means "no cap", not zero. */
  function setSetup(key, raw) {
    var v = String(raw === undefined || raw === null ? '' : raw).trim();
    if (key === 'tag') App.setup.tag = v.replace(/[^A-Za-z0-9._-]/g, '') || 'v5';
    else if (key === 'nests') App.setup.nests = v || '2';
    else if (key === 'minutes') App.setup.minutes = v === '' ? null : Math.max(0, +v || 0) || null;
    else if (key === 'seed') App.setup.seed = Math.max(0, Math.round(+v || 0));
    else if (key === 'envs') App.setup.envs = Math.max(64, Math.round(+v || 512));
    else if (key === 'scoreWith') App.setup.scoreWith = v === 'best' ? 'best' : 'trained';
    else if (key === 'steps') {
      var m = /^([0-9.eE+-]+)\s*([kKmMbBgG]?)$/.exec(v.replace(/[_,\s]/g, ''));
      var n = m ? parseFloat(m[1]) : NaN;
      var mult = m ? ({ k: 1e3, m: 1e6, b: 1e9, g: 1e9 }[m[2].toLowerCase()] || 1) : 1;
      App.setup.steps = isNaN(n) ? null : Math.round(n * mult) || null;
    }
    saveSetup();
  }

  /* One writer for every academy control, so a slider, a chip and the seed box cannot
     disagree about where the value lives. */
  function setAcad(kind, key, raw) {
    var a = acadOf(App.school), v = +raw;
    if (kind === 'acad') {
      if (key === 'steps') a.steps = Math.max(1e6, Math.round(v) * 1e6);
    } else if (kind === 'acadraw') {
      if (key === 'steps') a.steps = Math.max(0, Math.round(v)) || null;
      else if (key === 'seed') a.seed = Math.max(0, Math.round(v) || 0);
    } else if (kind === 'shape') {
      a.shaping[key] = v;
    } else if (kind === 'hyper') {
      a.hyper[key] = v;
    }
    saveAcademies();
  }

  function acadReset() {
    var a = acadOf(App.school);
    a.shaping = {};
    a.hyper = {};
    saveAcademies();
    render();
  }

  /* Pinning during a drag: the arena should follow the handle, but not at sixty
     messages a second. One in flight at a time, with the last position sent after. */
  var _pinAt = null, _pinBusy = false;
  /* The last position the TRAINER agreed to, which is not the same thing as the last
     position the handle was dragged to. A refusal has to put the reel back to this. */
  var _pinSent = false, _pinOk = null;
  function pinTo(i) {
    _pinAt = i;
    if (_pinBusy) return;
    _pinBusy = true;
    setTimeout(function () {
      _pinBusy = false;
      if (_pinAt === null) return;
      var at = _pinAt; _pinAt = null;
      // Ordered on one socket, so the mode is already `train` by the time the pin is
      // read. Cheap to be sure rather than to remember: re-entering the shadow arena
      // when it is already up would rewind the episode for nothing.
      if (needsShadow()) {
        App.net.send({ cmd: 'shadow' });
        notice('Back to the shadow arena — the reel plays into that one, not the checkpoint '
               + 'you were watching.', false);
      }
      _pinSent = true;
      App.net.send({ cmd: 'pin', at: at, school: App.school });
    }, 120);
  }

  function bind() {
    // Typing must not re-render — the caret would jump out of the field on every key.
    // The value is parsed as it is typed and only drawn back on blur.
    el('root').addEventListener('input', function (e) {
      var t = e.target.closest('[data-setup]');
      if (t) { setSetup(t.dataset.setup, t.value); return; }

      // The reel. The label follows the handle immediately; the arena follows a beat
      // later, because swapping the policy is a message and a drag is not.
      if (e.target.id === 'tl') {
        App.pinned = +e.target.value;
        var lab = el('tl-label');
        if (lab) lab.innerHTML = timelineLabel();
        pinTo(App.pinned);
        return;
      }

      // Academy sliders: write the value and update the readout beside it, without a
      // re-render — rebuilding the panel mid-drag would drop the handle.
      var kinds = ['acad', 'shape', 'hyper'];
      for (var i = 0; i < kinds.length; i++) {
        var k = kinds[i], node = e.target.closest('[data-' + k + ']');
        if (!node) continue;
        var key = node.dataset[k];
        setAcad(k, key, node.value);
        var out = el('sl-' + k + '-' + key + '-out');
        if (out) {
          out.textContent = k === 'acad' ? steps(+node.value * 1e6)
            : k === 'shape' ? (+node.value).toFixed(3)
            : (+node.step >= 1 ? Math.round(+node.value) : (+node.value).toFixed(4).replace(/0+$/, ''));
        }
        return;
      }
      var num = e.target.closest('[data-acadnum]');
      if (num) setAcad('acadraw', num.dataset.acadnum, num.value);
    });
    el('root').addEventListener('change', function (e) {
      var t = e.target.closest('[data-setup]');
      if (t && App.screen === 'setup') { render(); return; }
      if (e.target.id === 'tl') { pinTo(+e.target.value); return; }
      // Only the budget slider changes anything derived (the estimate, the chips).
      if (e.target.closest('[data-acad]') && App.acadOpen) render();
    });

    el('root').addEventListener('click', function (e) {
      var chip = e.target.closest('[data-setup-set]');
      if (chip) { setSetup(chip.dataset.setupSet, chip.dataset.value); render(); return; }
      var achip = e.target.closest('[data-acad-set]');
      if (achip) { setAcad('acadraw', achip.dataset.acadSet, achip.dataset.value); render(); return; }
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
      else if (a === 'setup') go('setup');
      else if (a === 'acad-open') { App.acadOpen = true; App.screen = 'school'; render(); }
      else if (a === 'acad-close') { App.acadOpen = false; render(); }
      else if (a === 'acad-reset') acadReset();
      else if (a === 'notice-close') { App.notice = null; render(); }
      else if (a === 'keys-close') { App.showKeys = false; render(); }
      else if (a === 'stop-all') { App.net.send({ cmd: 'stopAll' }); notice('Asking all three to finish their iteration and save…', false); }
      else if (a === 'refresh') {
        // The leaderboard, the BEST chip and the run list were stale until the server
        // volunteered a greeting or the page was reloaded. The command has always been
        // dispatched; nothing ever sent it.
        App.net.send({ cmd: 'hello' });
        notice('Asked the trainer for fresh results…', false);
      }
      else if (a === 'goto-training') {
        // The chip is on every screen; clicking it takes you to the run it is reporting.
        if (App.training) { App.school = App.training.school; App.screen = 'school'; App.acadOpen = false; }
        render();
      }
      else if (a === 'tl-live') {
        App.pinned = null;
        // LIVE off a checkpoint used to light the chip and change nothing: `pin(null)`
        // has no mode guard, so it answered "pinned" while the arena went on playing
        // UNTRAINED. Going back to the shadow arena is what LIVE means here.
        if (needsShadow()) App.net.send({ cmd: 'shadow' });
        App.net.send({ cmd: 'pin', at: null });
        render();
      }
      else if (a === 'score') {
        App.scoring = { lines: [], done: false, ok: false };
        App.net.send({ cmd: 'score', tag: App.setup.tag,
                       checkpoint: App.setup.scoreWith });
        render();
      }
      else if (a === 'use-run') App.net.send({ cmd: 'useRun', tag: t.dataset.tag });
      else if (a === 'train-stop') App.net.send({ cmd: 'stop' });
      else if (a === 'highlights') playHighlights();
      else if (a === 'race') startRace();
      else if (a === 'reset') {
        if (App.resetArmed && performance.now() - App.resetArmed < 5000) {
          App.resetArmed = 0;
          App.results = []; App.highlights = null; App.frame = App.prev = null;
          App.trainInfo = null; App.runState = null; App.mode = 'play';
          App.checkpoint = 'untrained';
          App.net.send({ cmd: 'reset' });
        } else {
          App.resetArmed = performance.now();
          setTimeout(function () { if (App.screen === 'menu') render(); }, 5100);
        }
        render();
      }
      else if (a === 'school') { App.screen = 'school'; render(); }
      else if (a === 'x-open') openExplain();
      else if (a === 'x-next') stepExplain(1);
      else if (a === 'x-prev') stepExplain(-1);
      else if (a === 'x-close') closeExplain();
    });

    window.addEventListener('keydown', function (e) {
      // Single letters are the whole control scheme, and the academy drawer and the RUNS
      // screen have text fields in them. Typing a seed of `11` used to fire the school
      // switch twice; typing a tag of `test` sent `t` (start an optimiser) and then `s`.
      var tgt = e.target;
      if (tgt && (tgt.tagName === 'INPUT' || tgt.tagName === 'TEXTAREA' || tgt.isContentEditable)
          && e.key !== 'Escape') {
        return;
      }
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
      else if (k === 'p') {
        // Only while a run is live: the brains that lost exist for one generation and
        // are written over, so a finished run has nothing to put in the losing lanes.
        if (!trainingHere()) {
          notice('The population race needs a live run in this school — the brains that '
                 + 'lost are not kept once it ends.', true);
        } else {
          App.raceFrames = {}; App.racePrev = {}; App.raceDone = {};
          App.raceWins = {}; App.raceGrid = {};
          App.net.send({ cmd: 'popRace', role: 'cat', lanes: 4 });
        }
      }
      else if (k === 'b') go('board');
      else if (k === 'h') playHighlights();
      else if (k === 'x') startRace();
      else if (k === '?' || k === '/') { App.showKeys = !App.showKeys; render(); e.preventDefault(); }
      else if (k === 'v') {
        if (App.results.length) { App.screen = 'verdict'; render(); }
        else notice('No verdict yet — it needs a finished playthrough of the twelve rooms.', true);
      }
      // It never re-rendered, so on any screen without an arena it was a key that did
      // nothing at all, silently, with no way to tell which way it had been left.
      else if (k === 'c') {
        App.sprites = !App.sprites; App.mapKey = null;
        notice(App.sprites ? 'Sprite skins on' : 'Sprite skins off — vector pair', false);
      }
      else if (k === 'f') { App.screen = 'final'; render(); App.net.send({ cmd: 'final' }); }
      // `e.key` is already 'T' when shift is held; `shiftKey` is checked too because a
      // remapped or on-screen keyboard can send one without the other.
      // shift+T opens the settings the run will use; plain t starts one with them.
      // shiftKey only. Testing `e.key === 'T'` as well meant caps lock silently turned
      // `t` into the other action — and for `s` that other action ended the run.
      else if (k === 't') { if (e.shiftKey) go('setup'); else trainLive(); }
      // `n` is the training key. On a school it opens THAT academy; anywhere else it
      // opens the run-level screen, which is now only runs and scoring.
      else if (k === 'n') {
        // On a school, or on its verdict, `n` is that academy. Elsewhere it is the
        // run-level screen.
        if (App.screen === 'school' || App.screen === 'verdict' || App.acadOpen) {
          App.acadOpen = !App.acadOpen;
          if (App.acadOpen) App.screen = 'school';
          render();
        } else go('setup');
      }
      else if (k === 'l') { App.screen = App.screen === 'lesson' ? 'school' : 'lesson'; render(); }
      else if (k === 'w') openExplain();
      // shift+S ends a live run; plain s skips the episode on screen. The two are very
      // different sizes of action, so they are not the same key press — and the test is
      // `shiftKey` alone, because with caps lock on `e.key` is 'S' for an unshifted press.
      else if (k === 's') App.net.send({ cmd: e.shiftKey ? 'stop' : 'skip' });
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
      /* The one message that says what is running. The server now pushes it whenever
         that changes, so every screen learns about a run starting and finishing instead
         of only the window that pressed the button. */
      .on('training', function (m) {
        var was = App.training && !App.training.finished ? App.training.school : null;
        // The server accepted a run, so there is nothing left to undo.
        if (m.live && m.liveSchool) _trainSent = false;
        if (m.live && m.liveSchool) {
          App.training = { school: m.liveSchool, finished: false };
          App.trainFinished = false;
          if (was !== m.liveSchool) {
            if (!App.train) App.train = { best: {}, budget: null, targetSteps: null, iter: 0, steps: 0 };
            notice('Training started · ' + esc(view(m.liveSchool).sealed
              ? 'school ' + (ORDER.indexOf(m.liveSchool) + 1) : view(m.liveSchool).short), false);
          }
        } else if (App.training && !App.training.finished) {
          // `trainDone` usually lands first and raises the notice, because it is the event
          // that knows the outcome. This is the fallback for a run that ended while this
          // window was not connected.
          App.training.finished = true;
          App.trainFinished = true;
          if (!App.notice || performance.now() - App.notice.at > 4000) {
            notice('Training finished · its checkpoints are saved', false);
          }
        }
        if (!m.scoring && App.scoring && !App.scoring.done) App.scoring.done = true;
        render();
      })
      .on('status', function (m) {
        // A note saying "the trainer is not running" must not outlive the trainer
        // starting — the caption underneath it has already gone back to normal.
        if (m.status === 'live' && App.link !== 'live') App.setupNote = null;
        // Losing the socket used to change nothing but a chip: the run card kept reading
        // LIVE TRAINING RUN 43% with a caption of catch rates, for a process that was no
        // longer there. `hello` reconciles it properly when the socket comes back.
        if (m.status !== 'live' && App.link === 'live' && App.training && !App.training.finished) {
          App.trainInfo = 'The trainer went away' + (App.train && App.train.frac !== undefined
            ? ' at ' + pct(App.train.frac) : '') + ' — everything above is its last word.';
        }
        App.link = m.status;
        render();
      })
      /* The greeting is a complete snapshot of the session, and the app used to keep
         only the catalogue out of it. A reload or a second window during a run therefore
         landed on the menu with no reel, no run card, and an academy offering to start a
         run that was already going. Everything below is read from what the server has
         been sending all along. */
      .on('hello', function (m) {
        App.cat = m;
        if (!App.levels.length && m.levels) {
          App.levels = m.levels.map(function (l) {
            var mm = E.genMap(l.seed >>> 0, l.nests || 1);
            return { seed: l.seed, map: mm, route: l.optimal, onRoute: l.trapsOnRoute,
                     nests: l.nests || 1 };
          });
        }
        // The reel, for every school: whatever is on disk, or the live one if a run is
        // going. Without this a finished run had no slider at all and a mid-run reload
        // started the array at frame 40.
        ORDER.forEach(function (k) {
          var a = m.academies && m.academies[k];
          if (a && a.timeline && a.timeline.length) App.timeline[k] = a.timeline.slice();
          // This only ever OVERWROTE, so it could not express "there is no reel any
          // more". RESET TO ZERO answers with a hello, the wiped session rightly offers
          // no timeline, and the frames the discarded run had already streamed stayed on
          // screen — a graph of the run right under a caption saying nothing on screen
          // comes from it. A zeroed session has trained nothing, so it has no reel.
          else if (m.zeroed) App.timeline[k] = [];
        });
        if (m.zeroed) App.pinned = null;
        var t = m.training || {};
        if (t.live && t.liveSchool) {
          App.training = { school: t.liveSchool, finished: false };
          App.school = t.liveSchool;
          App.mode = 'train';
          if (App.screen === 'menu') App.screen = 'school';
          if (!App.train) App.train = { best: {}, budget: null, targetSteps: null, iter: 0, steps: 0 };
          App.trainFinished = false;
        } else if (App.training && !App.training.finished) {
          // It finished while we were away, or this is a different trainer.
          App.training.finished = true;
          App.trainFinished = true;
        }
        // Which frame the arena is on, so a second window does not claim LIVE over an
        // arena the first window pinned.
        App.pinned = (m.pinnedAt === undefined ? null : m.pinnedAt);
        if (t.scoring && !App.scoring) App.scoring = { lines: [], done: false, ok: false };
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
        // A frame that does not join the last one is a new episode, and the ending it
        // was holding on is over.
        if (!joins) App.ending = null;
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
        // Cleared, so the guard that answers space / s / shift+S with a banner stops
        // applying. It used to last until the page was reloaded.
        App.replay = false;
        render();
      })
      /* The trainer answers some commands with a refusal — a second training run while
         one is already going, a school the run does not contain. Nothing listened for
         those, so on camera the key press simply did nothing and there was no way to
         tell a refusal from a bug. */
      .on('error', function (m) {
        App.banner = { t: (m.message || 'the trainer refused that').toUpperCase(), c: '#ff9a72' };
        App.bannerAt = performance.now();
        render();
      })
      .on('trainWait', function () {
        App.trainReady = false;
        App.trainInfo = 'Starting the optimiser — the arena waits for the first real policy.';
        render();
      })
      .on('race', function (m) {
        App.lastFrameAt = performance.now();
        // WHAT the lanes are can change without the level changing, and the screen is
        // first drawn from the `state` reply — before any lane has arrived, so it falls
        // back to the three schools. Pressing p with a run already on level 0 therefore
        // drew SIDE BY SIDE, with PPO / GA / CMA-ES over a population race that had
        // genuinely started: the level test below never fired, so nothing corrected it.
        var laneSig = (m.schools || []).join(',') + '|' + (m.raceKind || '');
        var laneChanged = laneSig !== App.raceSig;
        App.raceSig = laneSig;
        App.raceSchools = m.schools;
        // Only the population race sends these. Cleared rather than left standing, or a
        // three-school race after one would draw its lanes as somebody's genomes.
        App.raceKind = m.raceKind || null;
        App.raceLanes = m.raceLanes || null;
        App.raceSchool = m.raceSchool || null;
        App.raceGen = m.raceGen;
        App.racePop = m.racePop;
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
        if (laneChanged || (App.runState || {}).level !== m.level) {
          App.runState = Object.assign({}, App.runState, { level: m.level });
          App.raceDone = {};
          App.raceDoneAt = {};
          render();
        }
      })
      .on('laneEnd', function (m) {
        App.raceDone = App.raceDone || {};
        App.raceDone[m.school] = m.result;
        App.raceDoneAt[m.school] = performance.now();
        App.raceGrid = App.raceGrid || {};
        (App.raceGrid[m.school] = App.raceGrid[m.school] || [])[m.level] = m.result;
        render();
      })
      .on('episodeEnd', function (m) {
        // Both fire for every episode. The tally used to be the problem — a training
        // run's shadow episodes are one policy early, a different one later, sometimes a
        // pinned checkpoint in between, so twelve of them are not a score. The fix is the
        // label, not the counter: during a run the boxes read "THIS LAP", which is what
        // they are. The `render()` below is still skipped in train mode, because the
        // level strip it exists to redraw is not on screen then.
        // The lap wraps, and the trainer clears its own tally when it does — `advance()`
        // in train mode sets the level back to 0 and empties `results`. This one did not,
        // so the entries for the arenas the new lap had not reached yet were still last
        // lap's, and a box labelled THIS LAP counted episodes from the one before it. The
        // wrap is a level that goes backwards; nothing else moves the level down.
        if (App._lapLevel !== undefined && m.level < App._lapLevel) App.results = [];
        App._lapLevel = m.level;
        App.results[m.level] = m.result;
        App.ending = m.result === 'catch' || m.result === 'escape' ? m.result : null;
        App.banner = m.result === 'catch' ? { t: 'TOM CAUGHT HER', c: '#ff8a5c' }
          : m.result === 'escape' ? { t: 'JERRY GOT HOME', c: '#ffd166' }
          : { t: 'TIME OUT', c: '#8fa4c4' };
        App.bannerAt = performance.now();
        // `paintArena` draws the banner every frame; only the level strip needs the
        // screen back, and that is not what is under the arena during a run.
        if (App.mode !== 'train') render();
      })
      .on('state', function (m) {
        App.runState = m;
        App.playing = m.playing !== false;
        // One session, one clock: another window's `]` changes the speed for everybody, so
        // this window's chip has to follow rather than keep its own stale number.
        if (m.speed) App.speed = m.speed;
        // Follow the stream. A replay only re-sends messages, so without this the app
        // sits on the menu while the recorded episodes play into a screen nobody is
        // looking at. It also means the trainer can drive the app from Python.
        // Follow a CHANGE of mode, not every state message. `space`, `[` and `]` each
        // make the server broadcast a state, and following those meant pressing pause on
        // the leaderboard threw you onto the school screen.
        var changed = m.mode !== App.serverMode;
        App.serverMode = m.mode;
        // What the ARENA is playing is not a navigation decision, and it was being kept
        // inside one: the assignment below only ran when the SCREEN also changed, and
        // `play` and `train` both map to 'school'. So a mode change that stayed on the
        // school screen never reached `App.mode` — which is exactly what `shadow` does,
        // and it left the reel believing it still had to send `shadow` before every pin,
        // rewinding the episode on each drag.
        if (changed) App.mode = m.mode === 'train' ? 'train' : 'play';
        var to = changed ? { play: 'school', final: 'final', race: 'race', train: 'school' }[m.mode] : null;
        // ...except while a screen the author is working on is open. Both the run
        // screen and the academy drawer are places you sit and type; an episode playing
        // behind them must not drag you off mid-sentence.
        if (App.screen === 'setup' || App.acadOpen) to = null;
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
        // stays where it is, because its own scoreboard is already on screen. An open
        // academy drawer stays — the run ending is not a reason to close a panel somebody
        // is setting up — and so does a live training run, whose HUD and reel are the
        // whole point of the screen the verdict would replace.
        if (App.screen === 'school' && !App.acadOpen && !trainingHere()) App.screen = 'verdict';
        render();
      })
      /* The server refuses some commands — a second live run while one is still going,
         a checkpoint this run does not have — and until now the app dropped the reply on
         the floor. On camera that read as a key doing nothing, and worse: the HUD had
         already been reset to the budget of the run that was never started. */
      .on('error', function (m) {
        App.banner = { t: String(m.message || 'the trainer refused that').toUpperCase(), c: '#ff9a72' };
        App.bannerAt = performance.now();
        // The banner needs an arena and a frame to be drawn at all, so on six of the nine
        // screens it was raised into nothing. The strip is drawn by every screen.
        App.notice = { text: m.message || 'the trainer refused that',
                       at: performance.now(), bad: true };
        // A refusal must undo whatever the click optimistically put on screen. Without
        // this, a refused SCORE left the panel reading "SCORING · starting…" for the rest
        // of the session, and a refused TRAIN left a run card for a run that never began.
        if (App.scoring && !App.scoring.done && !App.scoring.lines.length) App.scoring = null;
        // Put back exactly what the press overwrote. Nulling instead cost a live run in
        // another school its run card, its STOP button and its scrub -- `trainingHere()`
        // reads `App.training`, so erasing it took the reel read-only while the run was
        // still going -- and left the HUD counting that run's steps against the budget of
        // the run that never started: 63M steps shown as 6% of a 1B budget nobody set.
        if (_trainSent) {
          _trainSent = false;
          App.training = _trainWas.training;
          App.train = _trainWas.train;
          App.mode = _trainWas.mode;
          App.trainReady = _trainWas.trainReady;
          App.trainFinished = _trainWas.trainFinished;
          App.pinned = _trainWas.pinned;
          App.results = _trainWas.results;
          App.highlights = _trainWas.highlights;
          App.runState = _trainWas.runState;
        } else if (App.training && !App.training.finished && (!App.train || !App.train.iter)) {
          App.training = null;
          App.train = null;
          if (App.mode === 'train') App.mode = 'play';
        }
        // A refused PIN left the reel reading "PINNED · the brain at 153M steps" over an
        // arena playing something else, for the rest of the session — the label is written
        // optimistically on the drag so the handle stays smooth, and nothing ever put it
        // back. The trainer refuses this exact lie server-side; the client was telling it
        // anyway, one line lower.
        if (_pinSent) {
          _pinSent = false;
          App.pinned = _pinOk;
          var slider = el('tl');
          if (slider) {
            var n = frames().length;
            slider.value = String(App.pinned === null || App.pinned === undefined
                                  ? Math.max(0, n - 1) : App.pinned);
          }
        }
        App.setupNote = m.message || null;
        App.trainInfo = m.message || null;
        render();
      })
      .on('trainStopping', function () {
        App.trainInfo = 'Stopping after this iteration — the run still saves its checkpoints '
          + 'and still picks its best Tom and Jerry.';
        render();
      })
      .on('snapshot', function () { /* snapshots arrive inside `train` events */ })
      .on('pinned', function (m) {
        _pinSent = false;
        App.pinned = m.at === null || m.at === undefined ? null : m.at;
        _pinOk = App.pinned;
        var lab = el('tl-label');
        if (lab) lab.innerHTML = timelineLabel(); else render();
      })
      .on('trainAllStarted', function (m) {
        App.runAll = { tag: m.tag, schools: {}, done: false };
        App.screen = 'setup';
        notice('Training all three · ' + esc(m.tag) + ' · one budget each', false);
        render();
      })
      .on('trainAllDone', function (m) {
        if (App.runAll) App.runAll.done = true;
        notice('All three finished · ' + esc(m.run || '') + ' · score it to settle the championship', false);
        render();
      })
      .on('scoreStarted', function () {
        App.scoring = { lines: [], done: false, ok: false };
        render();
      })
      .on('scoreStep', function (m) {
        if (!App.scoring) App.scoring = { lines: [], done: false, ok: false };
        App.scoring.lines.push('— ' + m.step);
        if (App.screen === 'setup') paintRunProgress(); else render();
      })
      .on('scoreLine', function (m) {
        if (!App.scoring) App.scoring = { lines: [], done: false, ok: false };
        App.scoring.lines.push(m.line);
        if (App.screen === 'setup') paintRunProgress();
      })
      .on('scoreDone', function (m) {
        if (!App.scoring) App.scoring = { lines: [], done: false, ok: false };
        App.scoring.done = true;
        App.scoring.ok = !!m.ok;
        render();
      })
      .on('runSwitched', function (m) {
        // A different run means different policies, so nothing on screen from the old
        // one is still true. Back to the Academy rather than leaving a stale arena up.
        // A different run means different policies, a different reel and a different
        // pin. Leaving any of it up put the old run's frames under a slider that now
        // indexes the new run's timeline.
        App.scoring = null; App.results = [];
        App.timeline = {}; App.pinned = null;
        App.train = null; App.training = null; App.trainFinished = false;
        App.frame = App.prev = null; App.trainInfo = null; App.mode = 'play';
        App.screen = 'menu';
        App.banner = { t: 'WATCHING ' + String(m.tag || '').toUpperCase(), c: '#7cbcff' };
        App.bannerAt = performance.now();
        render();
      })
      .on('train', function (m) {
        // Every training event carries the run's position on both clocks, so the HUD is
        // updated from all of them rather than only from the ones that happen to be
        // frequent. `algo` fires once per iteration — which for the population schools
        // is tens of seconds — so on its own it would leave the bar visibly stalled.
        // A window that joins mid-run — a reload during a shoot, a second monitor — gets
        // the HUD from the next event rather than staying blank until the run ends.
        if (!App.train) App.train = { best: {}, budget: null, targetSteps: null };
        {
          // Field by field. `trainDone` is emitted by the server rather than by the
          // school's own `emit()`, so it carries no clocks at all — copying them blindly
          // wrote `undefined` over the finished run's iteration and reset the bar to 0%
          // at the exact moment the numbers mattered most.
          if (m.iter !== undefined) App.train.iter = m.iter;
          if (m.steps !== undefined) App.train.steps = m.steps;
          if (m.wall !== undefined) App.train.wall = m.wall;
          if (m.frac !== undefined) App.train.frac = m.frac;
          if (m.targetSteps !== undefined) App.train.targetSteps = m.targetSteps;
          if (m.sps) App.train.sps = m.sps;
          if (m.eta !== undefined) App.train.eta = m.eta;
          if (m.budget) App.train.budget = m.budget;
        }
        // Never a full re-render while the settings screen is up: it has text fields,
        // and a repaint mid-word takes the caret with it.
        /* Two senders, one event shape. `run` is set when the event was tailed out of the
           three-school trainer; those belong to that school's row, never to the single
           live take's card. */
        if (m.run) {
          if (!App.runAll) App.runAll = { tag: m.run, schools: {}, done: false };
          var row = App.runAll.schools[m.school] || (App.runAll.schools[m.school] = { best: {} });
          row.iter = m.iter; row.steps = m.steps; row.wall = m.wall; row.frac = m.frac;
          if (m.targetSteps !== undefined) row.targetSteps = m.targetSteps;
          if (m.sps) row.sps = m.sps;
          if (m.eta !== undefined) row.eta = m.eta;
          if (m.kind === 'snapshot') { if (m.i === 0) App.timeline[m.school] = []; addFrame(m); }
          if (m.kind === 'best') row.best[m.role] = { rate: m.rate, lo: m.lo, steps: m.steps };
          if (m.kind === 'bestFinal') row.best[m.role] = { rate: m.rate, lo: m.lo, steps: m.fromSteps, settled: m.pick };
          if (App.screen === 'setup') paintRunProgress();
          else if (App.screen === 'school' && App.school === m.school && el('tl')) paintTimeline();
          return;
        }
        if (m.kind === 'snapshot') {
          // A frame of the reel. Appended and painted in place, never through render():
          // one lands every few seconds and a rebuild would drop a handle mid-drag.
          // Frame 0 is a new run's first word. That is the moment to drop the old reel,
          // not the click that started it — and it is what keeps a stale tail from the
          // previous, longer run hanging off the end of this one's slider.
          if (m.i === 0) App.timeline[m.school] = [];
          addFrame(m);
          // Frame 0 is the untrained policy, taken before a single update. Frame 1 is the
          // first thing the run actually produced, and it is a beat worth calling: the
          // author says "our first checkpoint just appeared" and the screen should agree.
          if (m.i === 1 && App.training && !App.training.finished) {
            App.banner = { t: 'FIRST CHECKPOINT · ' + steps(m.steps) + ' STEPS', c: '#ffd166' };
            App.bannerAt = performance.now();
          }
          if (App.screen === 'school' && App.school === m.school) {
            // The first frame turns the placeholder into a real slider, which needs the
            // screen once; after that the strip repaints itself.
            if (!el('tl')) render(); else paintTimeline();
          }
          return;
        }
        if (m.kind === 'progress' || m.kind === 'best') {
          if (m.kind === 'best' && App.train) {
            // Ranked on the lower end of the confidence interval, so that is what the
            // card reports beside the headline rate — a peak that is only a lucky
            // evaluation should not read as a new champion.
            App.train.best[m.role] = { rate: m.rate, lo: m.lo, steps: m.steps, from: 'now' };
          }
          if (App.screen === 'school') paintTrainHud();
          else if (App.screen === 'setup') paintRunProgress();
          return;
        }
        if (m.kind === 'bestFinal') {
          if (App.train) {
            // At the end the same field means something else: where the WINNING policy
            // came from. Tagged, so the label can say which.
            App.train.best[m.role] = { rate: m.rate, lo: m.lo, steps: m.fromSteps,
                                       settled: m.pick, from: 'policy' };
          }
          App.banner = { t: 'BEST ' + (m.role === 'cat' ? 'TOM' : 'JERRY') + ' · '
                            + (m.pick === 'peak' ? 'THE PEAK, NOT THE FINISH' : 'THE FINAL POLICY'),
                         c: m.role === 'cat' ? '#ff9a72' : '#7ee0ff' };
          App.bannerAt = performance.now();
          render();
          return;
        }
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
          if (trainedHere()) {
            App.trainInfo = 'live · cat ' + pct(m.catExam) + ' · mouse ' + pct(m.mouseExam)
              + ' vs the Examiner · ' + (m.steps / 1e6).toFixed(1) + 'M steps';
          }
        } else if (m.kind === 'promotion') {
          App.banner = { t: m.role.toUpperCase() + ' PROMOTED TO YEAR ' + m.year, c: '#ffd166' };
          App.bannerAt = performance.now();
        } else if (m.kind === 'trainDone') {
          App.trainFinished = true;
          if (App.training) App.training.finished = true;
          var nm = view(m.school || App.school);
          // A run the wipe threw away must not be announced like one that landed. This
          // used to read "finished · best Tom = its final policy · the BEST chip plays
          // them" over a session that had just reported itself empty — and the BEST chip
          // did NOT play them, because the weights behind it were wiped. What is worth
          // saying is the one thing the wipe did not touch: where the run is on disk.
          if (m.discarded) {
            App.trainInfo = 'reset to zero threw this run away while it was training — '
              + 'nothing on screen comes from it'
              + (m.savedTo ? ' · its checkpoints are still on disk at ' + m.savedTo : '');
            notice((nm.sealed ? 'That academy' : nm.short) + '\u2019s run was thrown away by '
              + 'RESET TO ZERO while it was training.'
              + (m.savedTo ? ' Its checkpoints are still on disk at ' + m.savedTo + '.' : ''), true);
            return;
          }
          notice(m.failed
            ? 'The run stopped: ' + (m.message || 'the trainer raised')
            : (nm.sealed ? 'That academy' : nm.short) + ' finished · best Tom = '
              + (((m.best || {}).cat === 'peak') ? 'its peak' : 'its final policy')
              + ' · best Jerry = ' + (((m.best || {}).mouse === 'peak') ? 'its peak' : 'its final policy')
              + ' · saved', !!m.failed);
          if (m.failed) {
            App.trainInfo = 'the run stopped: ' + (m.message || 'the trainer raised');
            render();
            return;
          }
          var b = m.best || {};
          // Name what the run settled on. "training finished" said nothing about the
          // thing the run existed to produce.
          App.trainInfo = 'training finished · best Tom = ' + (b.cat === 'peak' ? 'its peak' : 'its final policy')
            + ' · best Jerry = ' + (b.mouse === 'peak' ? 'its peak' : 'its final policy')
            + ' · the BEST chip plays them'
            + (m.savedTo ? ' · saved to ' + m.savedTo : '');
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
