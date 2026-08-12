# ClimbBot RL — v7 redesign

This folder is a working, learnable rebuild of the climbing-robot simulator. It replaces
the `test_6` setup, which could not learn to climb **by construction** (see diagnosis
below). Everything here is self-contained.

```
test_7/
  climb_env_v7.py        # the redesigned Gymnasium environment
  train_v7.py            # PPO trainer (stable-baselines3) — for real training runs
  prove_learning_cem.py  # dependency-free proof the env learns (no torch needed)
  learning_curve_v7.png  # evidence: reward rises, hands reach holds, hooks engage
../assets/
  robot_v7.xml           # retuned robot (actuator gains, force limits, contacts)
  scene_v7.xml           # wall + holds + stage-0 torso anchor
```

---

## Why the old version never worked (diagnosis)

Three independent showstoppers, each fatal on its own:

1. **Nothing held the robot to the wall.** The torso is a `freejoint` and there was no
   weld/connect anywhere. "Grasping" was only *detected* via contact force — it never
   *created a constraint*. The robot fell every episode. No policy can climb a wall it
   can't hang onto.
2. **The phase machine was dead.** In the old `step()`, the `right_reach -> right_pull`
   transition was commented out, so the episode froze in phase 1 forever; pull-up and
   the left arm never ran.
3. **The reward didn't describe climbing.** `w_grasp = w_pull = w_control = 0`. The only
   live term was `-0.5 * (right-hand-to-hold distance)`. Best case: hover one hand near
   one hold while the body falls.

Plus it was impractical to train: a nonlinear least-squares **IK solver ran inside every
RL step** (up to 6 restarts × 1500 evals); on failure it silently reused a stale solution,
jolting the `kp = 15000` position servos and producing the `NaN in QACC` blow-ups in
`MUJOCO_LOG.TXT`. There was also a latent bug — the left-arm branch called
`ik_left(target_r, ...)` (wrong arm's target).

## What changed in v7

- **Joint-space control.** The policy outputs 6 small joint-target deltas straight to the
  position servos. No IK in the loop → ~20–30× faster and far more stable.
- **Physical grip (no cheating).** Holding on is real contact + friction between the hook
  geoms and the holds. Nothing welds a hook to a hold.
- **Retuned actuators / contacts.** `kp` 15000 → 120 (revolute) / 4000 (prismatic);
  force limits 222 N·m → 12 N·m / 150 N; dropped the odd `timeconst`; softened contact
  `solref` to `0.02 1` (≥ 2× timestep) and raised hold friction so hooks can bear load.
- **Climbing-shaped dense reward:** reach the holds + press the hooks in (contact force)
  + stay upright + small control cost; large penalty + episode end on a fall.
- **Correct Gym semantics:** time-limit → `truncated`, fall → `terminated`.
- **Stage-0 curriculum:** `scene_v7.xml` has a `base_anchor` weld that pins the torso to
  the world at its spawn pose so the agent first learns reach+hook on a stable base. The
  env toggles it off (`freeze_base=False`) for free-climbing stages.

### Proof it learns (no torch required)
`prove_learning_cem.py` trains a tiny policy with the Cross-Entropy Method on Stage-0:

| metric | random baseline | after 12 CEM iters |
|---|---|---|
| episode return | −13.8 | **+474** |
| right hand → hold | — | **0.046 m** (< 0.06 grasp) |
| left hand → hold | — | **0.011 m** |
| both hooks engaged | — | **97% of steps** |

See `learning_curve_v7.png`.

---

## How to run it (on your Mac)

From `Climbing Robot/test_7/`:

```bash
# 0) one-time deps (your Mac, not the sandbox)
pip install mujoco gymnasium stable-baselines3 scipy matplotlib

# 1) quick, dependency-light sanity check that the env learns (~20s, no torch)
python prove_learning_cem.py
#    -> prints the table above and writes learning_curve_v7.png

# 2) real training — Stage 0 (torso anchored): learn reach + hook
python train_v7.py --timesteps 300000 --n-envs 8 --tag stage0
#    logs to runs_v7/stage0/  (tensorboard, checkpoints, best model)

# 3) watch training curves
tensorboard --logdir runs_v7
#    open http://localhost:6006  — look at rollout/ep_rew_mean and climb/*

# 4) Stage 1 — release the torso for free climbing (longer run)
python train_v7.py --timesteps 5000000 --n-envs 8 --no-freeze --tag stage1
```

### How to *watch the robot* in the MuJoCo viewer
On macOS the interactive viewer MUST run under `mjpython` (ships with the mujoco pip
package), not plain `python` — otherwise you get a "launch_passive requires mjpython"
error. Headless training runs fine under normal `python`.

```bash
# watch the trained CEM policy (note: mjpython on macOS)
mjpython cem_train_v7.py --tag stage0b --play

# just look at the scene / poke the model
python -m mujoco.viewer --mjcf=../assets/scene_v7.xml

# or roll out a trained policy with the live viewer:
python - <<'PY'
from climb_env_v7 import ClimbBotEnv
from stable_baselines3 import PPO
env = ClimbBotEnv(freeze_base=True, render_mode="human")
model = PPO.load("runs_v7/stage0/best/best_model")
obs,_ = env.reset()
for _ in range(2000):
    a,_ = model.predict(obs, deterministic=True)
    obs,r,term,trunc,info = env.step(a)
    if term or trunc: obs,_ = env.reset()
PY
```

---

## v7.1 — physical rung grip + climbing (passive hook)

A feasibility test proved the original flat holds **could not be gripped**: with pure
friction the hooks slipped and the robot fell. Worse, the original hooks don't catch a
plain cylinder either — what looked like a hang was actually the torso impaling a rung.
The working fix (grip stays fully physical, no welds):

- **Holds -> horizontal rungs** on a dense ladder (`assets/scene_v7_ladder.xml`), 0.2 m
  spacing per side so each hand-over-hand move is reachable.
- **One minimal capture bar per gripper** (`r_catch`/`l_catch` in `robot_v7.xml`): a thin
  rod just above the grip point. The existing hook cradles the rung from below; this bar
  stops it sliding out under load. Geometry of the original gripper is otherwise unchanged.
- **Torso/arm-link vs rung collisions are excluded** so only the hooks touch rungs (the
  body passes freely instead of jamming).

Validated headlessly:
- **The robot truly hangs from its hooks** — 34 N total (≈ body weight), stable for 1000+
  steps even with the floor removed.
- During a hand-coded pull-up the **support hook holds firm at 33 N** (it used to slip to 0)
  and the body **pulls up 0.31 m**. So grip + pull-up both work; what remains is the policy
  learning to coordinate "pull up while reaching" to fully seat the next rung — a PPO job.
- The robot **hangs stably** from the saved start pose (`hang_start_qpos.npy`): base holds
  at 0.25 m, 0 fall, for the whole episode.
- Caveat: the arm has no wrist roll (2 revolute + 1 prismatic), so claw orientation is tied
  to arm position — some rung positions give an upright (catching) claw, others tilt it.
  The policy has to favour stances where the claw catches; load isn't always shared evenly.

Climbing mode (`climb=True`) starts from the hang pose, frees the torso, and runs a
**hand-over-hand gait state machine**:

- Each hand owns its same-side rungs: left = hold_1/3/5, right = hold_2/4/6.
- Hands start on the bottom rungs (hold_1, hold_2). The ACTIVE hand is rewarded for
  reaching its next same-side rung (potential-based distance shaping); the SUPPORT hand is
  rewarded for keeping a firm grip (contact force > `firm_thresh`, default 12 N).
- A move only "counts" (big +10 reward, +height bonus) when the active hook grasps its
  next rung AND the support hand is firmly holding. Then the hands swap roles:
  left -> right -> left ... climbing the alternating rungs. Moving with a weak support
  grip is penalized; a fall ends the episode.
- Observation in climb mode (25-dim) is gait-aware: active-hand->next-rung vector,
  support-hand->its-rung vector, both hook forces, and an active-hand flag.

Each same-side move is a 0.4 m reach that requires a **pull-up** (the support arm hauls the
torso up). This is verified kinematically feasible: the left hand can reach hold_3 (0 mm
error) while the right stays on hold_2, with the base rising to z=0.32. It is, however, a
hard coordination problem: CEM in-session learns to hang and grip firmly but does not
discover the pull-up. **This is what the PPO run is for.** If PPO struggles, add
intermediate rungs (smaller spacing) as a curriculum, or add an explicit pull-up shaping
term (reward the support prismatic retracting while the active hand reaches).

```bash
# dependency-free climbing sanity run (no torch)
python cem_train_v7.py --tag climb --climb --ep-len 150 --seconds 60
mjpython cem_train_v7.py --tag climb --climb --play          # watch it (macOS)

# real climbing training with PPO (your Mac) — point train_v7 at the rung scene + climb env
python train_v7.py --xml ../assets/scene_v7_rung.xml --no-freeze --timesteps 5000000 --tag climb
#   (set the env to climb=True / init_qpos_file in make_env first — see climb_env_v7.py args)
```

## Roadmap to a full climb (honest expectations)

Stage 0 (proven here) is the reach+hook primitive on a fixed torso. To get to multi-move
climbing you still need a real (hours-long, ideally GPU) training run through a curriculum:

1. **Stage 0** — torso anchored; reach + engage both hooks. *(working)*
2. **Stage 1** — start hanging from one engaged hook; reach the free hand to the next hold
   and engage it without falling. Raise `fall_z` so "fell" triggers on real drops.
3. **Stage 2** — alternate hands to ascend N holds; reward net height gained.

Each stage should warm-start from the previous stage's weights.

### Real-world caveat worth knowing
The robot is ~3.1 kg (≈30 N). Hanging puts ≈12 N·m at each shoulder, but the DS3225MG
servos are ~2.45 N·m. The v7 sim uses higher force limits so the policy can be developed,
but the **physical** robot as modeled can't hold its own weight on two arms — you'll need
higher-torque actuators, gearing, or a counterbalance before sim policies transfer.
