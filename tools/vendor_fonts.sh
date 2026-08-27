#!/usr/bin/env bash
# Self-host the three faces so a recording session needs no network at all.
# Run once; app/index.html then prefers app/fonts/fonts.css if it exists.
set -euo pipefail
cd "$(dirname "$0")/../app"
mkdir -p fonts && cd fonts
UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
URL='https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap'
curl -sS -A "$UA" "$URL" -o fonts.css
grep -o 'https://fonts.gstatic.com[^)]*' fonts.css | sort -u | while read -r u; do
  f="$(basename "$u")"
  curl -sS "$u" -o "$f"
  # rewrite to the local copy
  python3 - "$u" "$f" <<'PY'
import sys, pathlib
u, f = sys.argv[1], sys.argv[2]
p = pathlib.Path("fonts.css"); p.write_text(p.read_text().replace(u, f))
PY
done
echo "vendored $(ls *.woff2 2>/dev/null | wc -l | tr -d ' ') font files into app/fonts/"
