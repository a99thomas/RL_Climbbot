# ClimbBot RL

Teach a **two-arm climbing robot** to climb a wall of rungs with reinforcement learning
(MuJoCo + Gymnasium + Stable-Baselines3 PPO).

**Current, working code lives in [`test_7/`](test_7/).** Older folders (`test_5/`, `test_6/`,
`archive/`) are historical and should not be used.

| Goal | Command (from `test_7/`) |
|------|--------------------------|
| Install | see [Setup](#setup) |
| Sanity-check the env (~20 s) | `python prove_learning_cem.py` |
| Train easy climb (recommended start) | `python train_v7.py --simple --timesteps 3000000 --n-envs 8 --tag simple` |
| Watch a policy (macOS) | `mjpython view_ppo.py --tag simple --simple --which latest` |
| Full docs | [`test_7/README_v7.md`](test_7/README_v7.md) |

---

## What the robot does

- Two arms, each with 2 revolute joints + 1 prismatic (telescoping) stage.
- 6-D continuous actions: small joint-target deltas sent to position servos (**no IK** in the RL loop).
- Grip is physical: hook geometry + a catch bar seat onto horizontal rungs (contact + friction).
- Default task is a **scrolling treadmill of rungs**: hand-over-hand moves at full gravity after a short curriculum.

Status (as of the v7 stack): continuous climbing at 1 g works; form fine-tuning (`--natural`) and mesh holds are optional next steps. See `test_7/README_v7.md` for metrics and caveats.

---

## Setup

### Requirements

- macOS or Linux (viewer instructions below assume **macOS**)
- Python **3.11 or 3.12**
- Conda or a virtualenv whose path **contains no spaces**

> **Important:** Do **not** create a `venv` inside this repo if the folder is named
> `Climbing Robot` (space). macOS `mjpython` uses a shebang that breaks on spaces
> (`bad interpreter: .../Climbing`). Prefer a conda env under `~/miniconda3/envs/...`.

### 1. Create the environment

```bash
conda create -n climbbot python=3.12 -y
conda activate climbbot
cd "/path/to/Climbing Robot"   # quote the path if it has a space
pip install -r requirements.txt
```

### 2. Verify imports

```bash
python -c "import mujoco, gymnasium, stable_baselines3, torch, numpy; print('ok', numpy.__version__)"
```

You need **NumPy 2.x**. VecNormalize stats from training are pickled with NumPy 2; NumPy 1.x
will fail to load them (`BitGenerator` / `PCG64` errors).

### 3. Enter the working directory

```bash
cd test_7
```

All train / view / eval commands below are run from `test_7/` unless noted.

---

## Quickstart (copy-paste)

```bash
conda activate climbbot
cd test_7

# A) prove the env can learn (no torch; ~20 seconds)
python prove_learning_cem.py

# B) train the easy curriculum (0.4 g, one move per episode)
python train_v7.py --simple --timesteps 3000000 --n-envs 8 --tag simple

# C) watch the latest checkpoint (macOS: mjpython, not python)
mjpython view_ppo.py --tag simple --simple --which latest

# D) metrics while training
tensorboard --logdir runs_v7
# open http://localhost:6006 — watch climb/moves_done and rollout/ep_rew_mean
```

On a laptop, expect roughly **1.5k–3k env steps/sec** with `--n-envs 8`. A 3M-step run is
on the order of **20–40 minutes**; a full gravity curriculum is hours.

Keep the machine awake for long runs (on macOS, `caffeinate -dims` around the train command).

---

## Viewing policies

On **macOS**, the interactive MuJoCo viewer must be launched with **`mjpython`**
(installed with the `mujoco` pip package). Plain `python` raises
`launch_passive requires mjpython`.

```bash
conda activate climbbot
cd test_7

# Easy curriculum policy
mjpython view_ppo.py --tag simple --simple --which latest

# Full-gravity treadmill (match the flags used at train time)
mjpython view_ppo.py --tag slow1 --which latest --treadmill \
  --spacing 0.15 --jitter-y 0.05 --jitter-z 0.25 --max-steps 1200

# Natural-form fine-tune (if you trained --tag natural1)
./view_natural1.sh
# or:
mjpython view_ppo.py --tag natural1 --which latest --treadmill \
  --spacing 0.15 --jitter-y 0.05 --jitter-z 0.25 --max-steps 1200
```

`--which` may be `latest`, `best`, or `final`. Always load the matching **VecNormalize**
stats (the scripts do this automatically from `runs_v7/<tag>/`).

Static hang pose (no policy):

```bash
mjpython view_hang.py          # add --sim to let physics run
```

---

## Training paths

### Easy → hard curriculum (recommended)

Warm-start each stage from the previous with `--init-from <tag>`:

```bash
# 1) discover a single move at reduced gravity
python train_v7.py --simple --timesteps 3000000 --n-envs 8 --tag simple2

# 2) anneal gravity (one stage at a time — do not jump 0.4→1.0)
python train_v7.py --treadmill --spacing 0.15 --max-moves 1 --gravity-scale 0.7 \
  --init-from simple2 --tag anneal_g07 --timesteps 2000000 --n-envs 8
python train_v7.py --treadmill --spacing 0.15 --max-moves 1 --gravity-scale 0.85 \
  --init-from anneal_g07 --tag anneal_g085 --timesteps 2000000 --n-envs 8
python train_v7.py --treadmill --spacing 0.15 --max-moves 1 --gravity-scale 1.0 \
  --init-from anneal_g085 --tag anneal_g10 --timesteps 2000000 \
  --ent-coef 0.001 --lr 1e-4 --n-envs 8

# 3) unlimited moves at 1 g
python train_v7.py --treadmill --spacing 0.15 --max-moves 0 --gravity-scale 1.0 \
  --max-steps 1200 --init-from anneal_g10 --tag endless --timesteps 5000000 \
  --ent-coef 0.001 --lr 1e-4 --n-envs 8

# 4) optional: quieter / more natural form
python train_v7.py --treadmill --spacing 0.15 --max-steps 1200 \
  --jitter-y 0.05 --jitter-z 0.25 --init-from endless --tag natural1 --natural \
  --lr 5e-5 --ent-coef 0.0003 --target-kl 0.03 --n-envs 8 --timesteps 5000000
```

Resume a tag that was interrupted:

```bash
python train_v7.py --resume --tag natural1 --treadmill --spacing 0.15 --max-steps 1200 \
  --jitter-y 0.05 --jitter-z 0.25 --natural --lr 5e-5 --ent-coef 0.0003 \
  --target-kl 0.03 --n-envs 8 --timesteps 5000000
```

Outputs land in `test_7/runs_v7/<tag>/` (`checkpoints/`, `best/`, `tb/`, VecNormalize pickles).

### Headless evaluation

```bash
python eval_headless.py --tag natural1 --which latest --episodes 12 \
  --gravity-scale 1.0 --spacing 0.15 --max-moves 0 --max-steps 1200 \
  --jitter-y 0.05 --jitter-z 0.25 --natural

python eval_hands.py --tag slow1 --episodes 10 --max-steps 1200 \
  --spacing 0.15 --jitter-y 0.05 --jitter-z 0.25
```

---

## Repository layout

```
Climbing Robot/
  README.md                 ← you are here
  requirements.txt
  assets/
    robot_v7.xml            ← robot + catch bars
    scene_v7_treadmill.xml  ← recommended scene (scrolling rungs)
    scene_v7_*.xml          ← other scenes (ladder, mesh holds, stage-0)
  test_7/                   ← ONLY supported training code
    climb_env_v7.py
    train_v7.py
    view_ppo.py
    eval_headless.py
    README_v7.md            ← deep dive, design notes, troubleshooting
  test_5/, test_6/, archive/  ← obsolete; do not train from these
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `launch_passive requires mjpython` | Use `mjpython …`, not `python …`, for the viewer on macOS. |
| `bad interpreter: .../Climbing` | Env path has a space. Use conda `climbbot` under `~/miniconda3`, not a venv inside this folder. |
| `BitGenerator` / `PCG64` pickle error loading VecNormalize | Install **NumPy ≥ 2** in the same env you use to view/train. |
| Policy looks random / falls immediately | Missing or mismatched `vecnormalize_*.pkl`. Use `view_ppo.py` / `eval_*.py` (they load the matching stats). Do not move checkpoints without their `*_vecnormalize_*` files. |
| `import torch` fails (missing dylib) | `pip uninstall -y torch && pip install --force-reinstall torch` |
| Training dies when the lid closes | Wrap with `caffeinate -dims python train_v7.py …` |
| Entropy / std explodes on fine-tunes | Pass `--ent-coef 0.001` (or lower) and `--lr 1e-4` / `5e-5`; use `--target-kl 0.03`. |

More design history, reward details, and dead-ends: [`test_7/README_v7.md`](test_7/README_v7.md)
and [`test_7/CLAUDE.md`](test_7/CLAUDE.md) (agent working notes).

---

## Hardware note

The simulated robot is ~3.1 kg. Shoulder torque while hanging exceeds what DS3225MG-class
servos can provide. The sim uses higher force limits for learning; real transfer needs
stronger actuators, gearing, or a counterbalance.
