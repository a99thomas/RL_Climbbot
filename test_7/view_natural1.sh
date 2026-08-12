#!/usr/bin/env bash
# Launch natural1 in the MuJoCo viewer. Prefer the climbbot conda env — its
# path has no spaces (unlike Climbing Robot/.venv, whose mjpython shebang breaks).
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x "$HOME/miniconda3/envs/climbbot/bin/mjpython" ]]; then
  PY="$HOME/miniconda3/envs/climbbot/bin/mjpython"
elif command -v mjpython >/dev/null 2>&1; then
  PY="$(command -v mjpython)"
else
  echo "No mjpython found. Activate climbbot:  conda activate climbbot" >&2
  exit 1
fi

exec "$PY" view_ppo.py \
  --tag natural1 --which latest --treadmill \
  --spacing 0.15 --jitter-y 0.05 --jitter-z 0.25 --max-steps 1200 \
  "$@"
