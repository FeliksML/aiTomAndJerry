/* The link to the Python trainer.
 *
 * Reconnects on its own, forever, with a short backoff. A dropped socket during a take
 * should heal without anyone touching a keyboard — the app keeps painting the last
 * frame it had and picks the stream back up.
 *
 * Everything the app knows about the run arrives through here: the catalogue on
 * connect (schools, checkpoints, level seeds, tournament results), then frames while
 * something is playing, then algorithm telemetry while something is training.
 */
(function (global) {
  'use strict';

  /* `?ws=8799` or `?ws=ws://host:port` points this window at a different trainer.
     Two runs can be up at once — a long one training overnight and a short one being
     shot — and without this the second window silently attaches to the first server. */
  function Net(url) {
    var q = null;
    try { q = new URLSearchParams(location.search).get('ws'); } catch (e) { q = null; }
    if (q) q = /^wss?:\/\//.test(q) ? q : ('ws://' + (location.hostname || '127.0.0.1') + ':' + q);
    this.url = q || url || ('ws://' + (location.hostname || '127.0.0.1') + ':8765');
    this.ws = null;
    this.status = 'offline';
    this.handlers = {};
    this.queue = [];
    this.retry = 0;
    this._timer = null;
  }

  Net.prototype.on = function (type, fn) {
    (this.handlers[type] || (this.handlers[type] = [])).push(fn);
    return this;
  };

  Net.prototype._emit = function (type, msg) {
    (this.handlers[type] || []).forEach(function (fn) { fn(msg); });
    (this.handlers['*'] || []).forEach(function (fn) { fn(msg); });
  };

  Net.prototype.connect = function () {
    var self = this;
    clearTimeout(this._timer);
    try {
      this.ws = new WebSocket(this.url);
    } catch (e) {
      return this._reconnect();
    }
    this.status = 'connecting';
    this._emit('status', { status: this.status });

    this.ws.onopen = function () {
      self.status = 'live';
      self.retry = 0;
      self._emit('status', { status: self.status });
      var q = self.queue;
      self.queue = [];
      q.forEach(function (m) { self.send(m); });
    };
    this.ws.onmessage = function (ev) {
      var msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      self._emit(msg.type || 'message', msg);
    };
    this.ws.onclose = function () {
      self.status = 'offline';
      self._emit('status', { status: self.status });
      self._reconnect();
    };
    this.ws.onerror = function () { try { self.ws.close(); } catch (e) { /* onclose handles it */ } };
    return this;
  };

  Net.prototype._reconnect = function () {
    var self = this;
    this.retry = Math.min(this.retry + 1, 8);
    var wait = Math.min(4000, 300 * this.retry);
    this._timer = setTimeout(function () { self.connect(); }, wait);
  };

  Net.prototype.send = function (msg) {
    if (this.ws && this.ws.readyState === 1) {
      this.ws.send(JSON.stringify(msg));
    } else if (this.queue.length < 32) {
      // Hold the intent rather than dropping it: a command typed a moment before the
      // socket came back should still land.
      this.queue.push(msg);
    }
  };

  global.Net = Net;
})(window);
