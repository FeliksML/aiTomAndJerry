/* The reveal layer — what the audience is allowed to know yet.
 *
 * The intro names only the first school. The other two stay sealed until they are
 * introduced on camera, so nothing has to be blurred in the edit: press the key and
 * the page reveals the next one live.
 *
 * Level 0  only PPO is named. GA and CMA-ES are sealed.
 * Level 1  GA revealed.
 * Level 2  CMA-ES revealed. Everything is open.
 *
 * A sealed school must look *deliberately* sealed, never broken. It keeps its card,
 * its layout and its number; what it loses is its name, its emblem, its accent colour,
 * its building and its explainer. In their place: a redaction plate, a scrambled
 * codename that is stable per school, and neutral steel instead of the accent.
 *
 * The state lives in localStorage, so a crash or an accidental reload mid-shoot does
 * not unseal the rest of the video.
 */
(function (global) {
  'use strict';

  var KEY = 'cm-reveal-v1';
  var ORDER = ['ppo', 'ga', 'cmaes'];
  var SEALED = '#8494ad';
  var CODENAMES = { ga: 'SCHOOL BETA', cmaes: 'SCHOOL GAMMA' };

  var level = 0;
  var listeners = [];

  function load() {
    try {
      var v = parseInt(localStorage.getItem(KEY), 10);
      level = isNaN(v) ? 0 : Math.max(0, Math.min(ORDER.length - 1, v));
    } catch (e) { level = 0; }
  }

  function save() {
    try { localStorage.setItem(KEY, String(level)); } catch (e) { /* not worth failing over */ }
  }

  function set(v) {
    var next = Math.max(0, Math.min(ORDER.length - 1, v));
    if (next === level) return;
    level = next;
    save();
    listeners.forEach(function (fn) { fn(level); });
  }

  function revealed(key) {
    var i = ORDER.indexOf(key);
    return i < 0 || i <= level;
  }

  /* One place that decides how a school is allowed to look. Every screen asks here
     rather than checking the level itself, so a new screen cannot leak by omission. */
  function view(algo) {
    if (!algo) return null;
    if (revealed(algo.key)) {
      return {
        key: algo.key, sealed: false, short: algo.short, full: algo.full,
        line: algo.line, blurb: algo.blurb, specs: algo.specs,
        color: algo.color, light: algo.light, deep: algo.deep,
        emblem: algo.key
      };
    }
    return {
      key: algo.key, sealed: true,
      short: '?????', full: CODENAMES[algo.key] || 'CLASSIFIED',
      line: 'sealed until it is introduced',
      blurb: 'This school has not been introduced yet. Its method, its emblem and its '
        + 'colours stay sealed until the reveal.',
      specs: [['METHOD', 'REDACTED'], ['POPULATION', 'REDACTED'], ['SIGNATURE', 'REDACTED']],
      color: SEALED, light: '#b7c4d8', deep: '#171d27',
      emblem: 'sealed'
    };
  }

  function on(fn) { listeners.push(fn); }

  /* Keyboard: `r` reveals the next school, `shift+R` takes one back (for a re-shoot),
     `shift+0` reseals everything back to PPO-only. Deliberately off the common keys —
     a stray keypress mid-take must not unseal the rest of the video. */
  function bindKeys() {
    window.addEventListener('keydown', function (e) {
      if (e.key === 'r') { set(level + 1); e.preventDefault(); }
      else if (e.key === 'R') { set(level - 1); e.preventDefault(); }
      else if (e.key === ')' || (e.key === '0' && e.shiftKey)) { set(0); e.preventDefault(); }
    });
  }

  load();
  global.Reveal = {
    get level() { return level; },
    set: set, revealed: revealed, view: view, on: on, bindKeys: bindKeys,
    ORDER: ORDER, SEALED: SEALED,
    max: ORDER.length - 1
  };
})(window);
