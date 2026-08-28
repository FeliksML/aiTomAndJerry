#!/usr/bin/env bash
# One entry point for the whole pipeline.
#
#   ./run.sh train [minutes] [tag]   train all three schools in parallel
#   ./run.sh score [tag]             tournament + checkpoint progression + highlights
#   ./run.sh serve [tag]             live trainer + the app, ready to record
#   ./run.sh verify                  the environment parity gates
#   ./run.sh all   [minutes] [tag]   train, score, then serve
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
TAG_DEFAULT=v3

train() { $PY trainer/scripts/train.py --minutes "${1:-45}" --tag "${2:-$TAG_DEFAULT}"; }

score() {
  local tag="${1:-$TAG_DEFAULT}"
  $PY trainer/scripts/tournament_run.py --run "runs/$tag"
  $PY trainer/scripts/highlights.py --run "runs/$tag" --episodes 400
}

serve() {
  local tag="${1:-$TAG_DEFAULT}"
  $PY trainer/scripts/serve.py --run "runs/$tag" &
  local ws=$!
  python3 tools/serve_app.py --port 8778 >/dev/null 2>&1 &
  local http=$!
  trap 'kill $ws $http 2>/dev/null || true' EXIT INT TERM
  echo
  echo "  app      http://localhost:8778"
  echo "  trainer  ws://127.0.0.1:8765   (run runs/$tag)"
  echo "  keys     1 2 3 schools · l lesson · h highlights · f final · b leaderboard · r REVEAL"
  echo
  wait $ws
}

verify() {
  $PY trainer/scripts/check_arenas.py
  # Both hole counts: the generator branches on it, so one is not evidence for the other.
  for n in 1 2; do
    node trainer/scripts/dump_js.js 1 150 "$n" > "runs/parity-js-$n.json"
    $PY trainer/scripts/parity.py "runs/parity-js-$n.json"
    $PY trainer/scripts/vec_parity.py 5 "$n"
  done
  $PY trainer/scripts/balance.py 480 2
}

case "${1:-serve}" in
  train)  train "${2:-45}" "${3:-$TAG_DEFAULT}" ;;
  score)  score "${2:-$TAG_DEFAULT}" ;;
  serve)  serve "${2:-$TAG_DEFAULT}" ;;
  verify) verify ;;
  all)    train "${2:-45}" "${3:-$TAG_DEFAULT}"; score "${3:-$TAG_DEFAULT}"; serve "${3:-$TAG_DEFAULT}" ;;
  *)      sed -n '2,9p' "$0" >&2; exit 1 ;;
esac
