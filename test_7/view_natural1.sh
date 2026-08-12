#!/usr/bin/env bash
# Watch the natural1 PPO checkpoint in the MuJoCo viewer.
#
# Prerequisites:
#   conda activate climbbot          # NumPy 2 + mujoco; path must have no spaces
#   cd test_7
#   ./view_natural1.sh
#
# Do NOT use a venv living under "Climbing Robot/" — macOS mjpython shebangs
# break on spaces ("bad interpreter: .../Climbing").
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x "${HOME}/miniconda3/envs/climbbot/bin/mjpython" ]]; then
  MJPYTHON="${HOME}/miniconda3/envs/climbbot/bin/mjpython"
elif [[ -x "${HOME}/mambaforge/envs/climbbot/bin/mjpython" ]]; then
  MJPYTHON="${HOME}/mambaforge/envs/climbbot/bin/mjpython"
elif command -v mjpython >/dev/null 2>&1; then
  MJPYTHON="$(command -v mjpython)"
else
  echo "mjpython not found. Create/activate the env first:" >&2
  echo "  conda create -n climbbot python=3.12 -y && conda activate climbbot" >&2
  echo "  pip install -r ../requirements.txt" >&2
  exit 1
fi

echo "Using: $MJPYTHON"
exec "$MJPYTHON" view_ppo.py \
  --tag natural1 --which latest --treadmill \
  --spacing 0.15 --jitter-y 0.05 --jitter-z 0.25 --max-steps 1200 \
  "$@"
