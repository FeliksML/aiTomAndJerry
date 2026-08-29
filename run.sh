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
TAG_DEFAULT=v4

# Extra flags after [minutes] [tag] are forwarded, so the --nests in the README is the
# --nests that runs. They used to be dropped silently, which is the worst way to be wrong
# about a command you typed on camera.
train() {
  local mins="${1:-45}" tag="${2:-$TAG_DEFAULT}"
  shift 2 2>/dev/null || shift $# 
  $PY trainer/scripts/train.py --minutes "$mins" --tag "$tag" "$@"
}

score() {
  local tag="${1:-$TAG_DEFAULT}"
  $PY trainer/scripts/tournament_run.py --run "runs/$tag"
  $PY trainer/scripts/highlights.py --run "runs/$tag" --episodes 400
}

# A port already in use is the one failure that looks exactly like a bug in the app:
# the page loads from the OLD server still on 8778 and then sits there saying TRAINER
# OFFLINE, with nothing anywhere saying why. Check first and say so.
port_owner() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1
}

require_free() {
  local port="$1" what="$2" pid
  pid="$(port_owner "$port")"
  [ -z "$pid" ] && return 0
  echo "  port $port ($what) is already taken by pid $pid:" >&2
  ps -o command= -p "$pid" 2>/dev/null | sed 's/^/    /' >&2
  echo "  stop it first:  kill $pid" >&2
  return 1
}

serve() {
  local tag="${1:-$TAG_DEFAULT}"
  echo
  require_free 8765 "trainer" || exit 1
  require_free 8778 "app" || exit 1
  $PY trainer/scripts/serve.py --run "runs/$tag" &
  local ws=$!
  python3 tools/serve_app.py --port 8778 &
  local http=$!
  trap 'kill $ws $http 2>/dev/null || true' EXIT INT TERM
  # Both have to actually be up. `serve.py` imports torch, which is not instant.
  local i=0
  while [ $i -lt 40 ]; do
    [ -n "$(port_owner 8765)" ] && [ -n "$(port_owner 8778)" ] && break
    kill -0 $ws 2>/dev/null || { echo "  the trainer exited on start-up — see above" >&2; exit 1; }
    sleep 0.5
    i=$((i + 1))
  done
  echo
  echo "  app      http://localhost:8778"
  echo "  trainer  ws://localhost:8765   (run runs/$tag)"
  echo "  keys     n TRAINING (budget, train all three, score) · 1 2 3 schools · l lesson"
  echo "           h highlights · f final · b leaderboard · x side by side · r REVEAL"
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
  # Needs a recorded session; the renderer gates are only meaningful against real frames.
  if ls runs/journals/*.jsonl >/dev/null 2>&1; then
    node tools/check_render.js
  else
    echo "render gate skipped — no journal yet (serve once and it records one)"
  fi
}

case "${1:-serve}" in
  train)  shift; train "$@" ;;
  score)  score "${2:-$TAG_DEFAULT}" ;;
  serve)  serve "${2:-$TAG_DEFAULT}" ;;
  verify) verify ;;
  all)    shift; train "$@"; score "${2:-$TAG_DEFAULT}"; serve "${2:-$TAG_DEFAULT}" ;;
  *)      sed -n '2,9p' "$0" >&2; exit 1 ;;
esac
