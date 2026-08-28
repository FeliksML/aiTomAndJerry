"""Static server for the app, with caching defeated properly.

`python -m http.server` sends no cache headers, so a browser will happily keep an old
copy of env.js after the environment changes underneath it — which looks exactly like a
bug in the environment and wastes a take. Observed doing exactly that: the page kept
serving a single-hole `genMap` after the two-hole one had shipped.

`no-store` alone did not fix it, because a script tag can still be answered from the
memory/disk cache without a revalidation. So index.html is rewritten on the way out with
a `?v=<mtime>` on every local script and stylesheet: change a file, and its URL changes,
and there is nothing left for a cache to match.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import pathlib
import re
import socketserver

ROOT = pathlib.Path(__file__).resolve().parents[1] / "app"


ASSET = re.compile(rb'(?P<attr>(?:src|href)=")(?P<path>(?:js|css)/[^"?]+)"')


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def do_GET(self) -> None:                      # noqa: N802 (stdlib naming)
        if self.path in ("/", "/index.html"):
            return self._send_index()
        return super().do_GET()

    def _send_index(self) -> None:
        html = (ROOT / "index.html").read_bytes()

        def stamp(m):
            f = ROOT / m.group("path").decode()
            v = int(f.stat().st_mtime) if f.exists() else 0
            return m.group("attr") + m.group("path") + f"?v={v}".encode() + b'"'

        body = ASSET.sub(stamp, html)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):        # keep the console readable during a shoot
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8778)
    a = ap.parse_args()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", a.port), functools.partial(Handler, directory=str(ROOT))) as s:
        print(f"app on http://localhost:{a.port}  (no-store, so a reload always gets the real files)",
              flush=True)
        s.serve_forever()


if __name__ == "__main__":
    main()
