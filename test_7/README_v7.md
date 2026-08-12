# ClimbBot v7 — user guide

This folder is the **only supported** training stack. It replaces `test_6/`, which could
not learn to climb by construction (see [History](#history--why-v7-exists) at the bottom).

For a short project overview and install steps, start at the repo
[README.md](../README.md). This file is the detailed runbook for `test_7/`.

```
test_7/
  climb_env_v7.py      # Gymnasium env (control, obs, reward, gait)
  train_v7.py          # PPO trainer (stable-baselines3)
  view_ppo.py          # watch a checkpoint (use mjpython on macOS)
  view_natural1.sh     # convenience launcher for --tag natural1
  view_hang.py         # static hang pose
  eval_headless.py     # moves / falls / tilt metrics
  eval_hands.py        # left-first vs right-first split
  prove_learning_cem.py
  cem_train_v7.py
  hang_start_*.npy     # saved hang poses for reset
../assets/
  robot_v7.xml
  scene_v7_treadmill.xml         # recommended
  scene_v7_treadmill_holds.xml   # mesh holds (harder)
  scene_v7_ladder.xml / scene_v7.xml / …
```

Outputs (not committed): `runs_v7/<tag>/checkpoints/`, `best/`, `tb/`, VecNormalize pickles.

---

## Prerequisites checklist

1. `conda activate climbbot` (or your env with packages from `../requirements.txt`).
2. Working directory is **`test_7/`**.
3. Env path has **no spaces** (do not put a venv inside `Climbing Robot/`).
4. **NumPy 2.x** (`python -c "import numpy; print(numpy.__version__)"`).
5. On macOS, viewer commands use **`mjpython`**, train/eval use **`python`**.

```bash
conda activate climbbot
cd test_7
python -c "import mujoco, gymnasium, stable_baselines3, torch, numpy; print('numpy', numpy.__version__)"
```

---

## 5-minute smoke tests

### Env can learn (no torch)

```bash
python prove_learning_cem.py
# writes learning_curve_v7.png; reward should rise sharply vs random
```

### Hang is stable under zero action

```bash
python - <<'PY'
import numpy as np
from climb_env_v7 import ClimbBotEnv
e = ClimbBotEnv(
    xml_path="../assets/scene_v7_treadmill.xml",
    treadmill=True, max_moves=1, tm_spacing=0.15, gravity_scale=0.4,
    init_qpos_file="hang_start_treadmill.npy",
    max_steps=200, init_joint_noise=0.02,
)
o, _ = e.reset(seed=0)
for _ in range(150):
    o, r, t, tr, info = e.step(np.zeros(6))
print("base_z", round(info["base_z"], 3), "survived", not t)
PY
```

Healthy result: base ≈ 0.09 m, episode does **not** terminate.

### Open the scene

```bash
python -m mujoco.viewer --mjcf=../assets/scene_v7_treadmill.xml
# or hang pose:
mjpython view_hang.py
```

---

## Train

### Recommended first run: easy curriculum

`--simple` = treadmill + **0.4 g** + **0.15 m** rung spacing + **1 move** per episode.

```bash
python train_v7.py --simple --timesteps 3000000 --n-envs 8 --tag simple
```

Watch curves:

```bash
tensorboard --logdir runs_v7
# http://localhost:6006 → climb/moves_done, rollout/success_rate, rollout/ep_rew_mean
```

Checkpoints every ~50k steps; best model every ~25k. Training is headless — re-run
`view_ppo` on `--which latest` to “watch it train.”

### Useful CLI flags (`train_v7.py`)

| Flag | Meaning |
|------|---------|
| `--simple` | Easy curriculum (0.4 g, 0.15 m, 1 move) |
| `--treadmill` | Scrolling mocap rungs (recommended scene) |
| `--climb` | Ladder climb mode (non-treadmill) |
| `--gravity-scale F` | Scale gravity (curriculum) |
| `--spacing M` | Rung spacing in meters |
| `--max-moves N` | End episode after N moves (`0` = unlimited) |
| `--max-steps N` | Horizon (use `1200` for slow prismatics / multi-move) |
| `--jitter-y M` / `--jitter-z F` | Randomize hold placement |
| `--natural` | Stronger upright / smoothness / pull-then-reach priors |
| `--time-penalty F` | Per-step cost (default `0.03` with `--natural`) |
| `--push N` | Random lateral shoves up to N newtons |
| `--tag NAME` | Output folder under `runs_v7/NAME` |
| `--resume` | Continue latest checkpoint of `--tag` |
| `--init-from TAG` | Warm-start weights from another tag |
| `--ent-coef` / `--lr` / `--target-kl` / `--init-std` | PPO fine-tune knobs |

### Curriculum that actually works

Do **not** jump gravity 0.4 → 1.0 in one step (success collapses). Anneal, warm-starting each stage:

1. `--simple` (0.4 g, 1 move)
2. `--gravity-scale 0.7` → `0.85` → `1.0` (still 1 move)
3. `--max-moves 0` for endless chaining
4. Optional: `--jitter-y/--jitter-z`, then `--natural`, then `--push`

Example natural-form fine-tune (after a strong 1 g policy such as `slow1` / `endless*`):

```bash
python train_v7.py --treadmill --spacing 0.15 --max-steps 1200 \
  --jitter-y 0.05 --jitter-z 0.25 \
  --init-from slow1 --tag natural1 --natural \
  --lr 5e-5 --ent-coef 0.0003 --target-kl 0.03 \
  --n-envs 8 --timesteps 5000000
```

Interrupted run:

```bash
python train_v7.py --resume --tag natural1 --treadmill --spacing 0.15 \
  --max-steps 1200 --jitter-y 0.05 --jitter-z 0.25 --natural \
  --lr 5e-5 --ent-coef 0.0003 --target-kl 0.03 --n-envs 8 --timesteps 5000000
```

Long laptop runs:

```bash
caffeinate -dims python train_v7.py …   # keeps Mac awake (lid + idle)
```

### Fine-tune hygiene

- Default `ent_coef=0.005` can inflate action noise on long hard runs → use `0.0003–0.001`.
- Prefer `--target-kl 0.02–0.03` so PPO does not thrash (`approx_kl` should sit near the target).
- Changing obs/reward and `--resume`-ing causes ~100–200k steps of value re-fit — expected.

---

## View a trained policy

```bash
# Easy
mjpython view_ppo.py --tag simple --simple --which latest

# Full gravity treadmill (match train-time flags)
mjpython view_ppo.py --tag slow1 --which latest --treadmill \
  --spacing 0.15 --jitter-y 0.05 --jitter-z 0.25 --max-steps 1200

# Natural-form tag
./view_natural1.sh
```

`--which`: `latest` | `best` | `final`. Or pass `--ckpt path/to.zip`.

The script loads the matching `ppo_v7_vecnormalize_*_steps.pkl`. Without those stats the
policy will look broken — never relocate a `.zip` without its VecNormalize pickle.

### macOS viewer rules

| Do | Don't |
|----|-------|
| `mjpython view_ppo.py …` from the **conda** env | `python view_ppo.py …` |
| Conda env under `~/miniconda3/envs/climbbot` | `../.venv/bin/mjpython` inside `Climbing Robot/` (space breaks shebang) |
| NumPy 2.x | NumPy 1.x (cannot unpickle training stats) |

---

## Evaluate without a viewer

```bash
python eval_headless.py --tag natural1 --which latest --episodes 12 \
  --gravity-scale 1.0 --spacing 0.15 --max-moves 0 --max-steps 1200 \
  --jitter-y 0.05 --jitter-z 0.25 --natural

python eval_hands.py --tag slow1 --episodes 10 --spacing 0.15 \
  --max-steps 1200 --jitter-y 0.05 --jitter-z 0.25
```

`eval_headless` reports success rate, moves/episode, falls, mean tilt, and steps to first move.

---

## How the system works (short)

### Control

- Action ∈ [-1, 1]^6 → integrate `action * DELTA_SCALE` into position-servo targets.
- Prismatics are intentionally slow (1.5 mm/control-step) for realism; full stroke ~3.6 s.
- `frame_skip=20`, `dt=0.002` → 25 Hz control.

### Grip (do not remove)

Passive hooks alone slip off cylinders. Each gripper has a **catch bar** (`r_catch` /
`l_catch` in `robot_v7.xml`). Keep them — without catch bars the robot cannot hang.

### Treadmill task

Six mocap rungs scroll. On a successful grab the lowest rung on that side recycles to the
top, so there are never rungs below the body (avoids base-jam). Only hook geoms collide
with rungs.

### Gait

- **Active** hand reaches next same-side rung; **support** hand must stay load-bearing.
- Advance requires several consecutive steps of: close distance + cradled geometry +
  signed upward force on both hooks (scaled to body weight).
- **Mirror** mode (default): policy always “sees” a left-hand reach; right-hand turns are
  mirrored. Required for chaining more than one move.

### Reward (climb mode)

Dense terms are **potential-based** (`Φ(s′) − Φ(s)`). Do not switch back to large
per-step reach bonuses — that taught policies to park at the rung and farm reward.
`--natural` tightens uprightness, adds action-jerk cost, wall proximity, and
pull-then-reach cadence weights, and defaults `time_penalty=0.03`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `launch_passive requires mjpython` | Use `mjpython` for viewers on macOS. |
| `bad interpreter: .../Climbing` | Space in env path. Use conda, not a venv under this repo folder. |
| `BitGenerator` / `PCG64` / `state must be a dict` | NumPy 1.x loading NumPy 2 pickles → upgrade NumPy in that env. |
| Policy flails / falls immediately on replay | Missing VecNormalize; use `view_ppo.py` / eval scripts. |
| `approx_kl` 0.1–0.3, `clip_fraction` ~0.5 | Set `--target-kl 0.03`; lower `--lr`. |
| `log_std` / action std blows up | Lower `--ent-coef` (fine-tunes: `0.0003–0.001`). |
| 0% success after stacking mesh + large jitter | Change **one** difficulty axis per stage; warm-start. |
| Success collapses 0.7g → 1.0g | Insert 0.85 g stage. |
| Torch dylib import error | Reinstall torch in the active env. |

---

## Mesh holds (optional, harder)

Cylinder treadmill is the default. Mesh STL holds (`--mesh-holds`) have a narrow lip
(±3 cm). Train insertion **without** jitter first, then anneal jitter carefully.
See `test_mesh_holds.py` and notes in `CLAUDE.md`. Do not resume the failed `mesh1` tag.

---

## History / why v7 exists

`test_6` failed for three independent reasons:

1. No physical grip constraint — “grasp” was force detection only; robot always fell.
2. Gait state machine was commented out — stuck in one phase forever.
3. Grasp/pull reward weights were zero — only a weak distance term remained.

Plus an IK solver ran inside every RL step (slow, non-stationary, NaN `QACC` blowups).

v7 uses joint-space control, physical catch-bar grips, potential-based climbing rewards,
and a treadmill curriculum. Design diary and dead-ends: [`CLAUDE.md`](CLAUDE.md).

### Hardware caveat

~3.1 kg robot; hanging shoulder torque exceeds DS3225MG-class servos. The sim allows
higher torque so policies can be developed; real transfer needs stronger actuators,
gearing, or a counterbalance.
