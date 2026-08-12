# ClimbBot RL — project context (read me first)

**Humans setting up / training:** use [`../README.md`](../README.md) and
[`README_v7.md`](README_v7.md) — those are the standalone runbooks.

This file is **agent working memory**: goal, design, what's been tried, what works, what
doesn't. Read it before changing code so you don't re-introduce already-fixed bugs.

Everything current lives in `test_7/` (code) and `../assets/` (MJCF models). The older
`test_6/` is the abandoned original (kept for reference only — see history).

---

## TL;DR status

- **WORKING (2026-07): continuous hand-over-hand climbing at FULL gravity.** Best policy:
  `runs_v7/endless7` — 16 moves / 28 s episode on average (max 26), 1.6 m height gain
  mean, 0 falls in 20 episodes, base tilt mean 14°, works starting with either hand.
  Watch it: `mjpython view_ppo.py --tag endless7 --which latest --treadmill --spacing 0.15`
- Run lineage (each warm-started from the previous via `--init-from`):
  `simple2` (0.4 g) → `anneal_g07` (0.7 g) → `anneal_g085` (0.85 g) → `anneal_g10b` (1 g,
  single move) → `grip3_g10` (1 g, load-bearing grip gate) → `endless6` (unlimited moves,
  mirror gait) → `endless7` (stronger upright shaping) → `endless8` (RANDOMIZED holds
  ±5 cm lateral / ±25% gap + level-shoulders form penalty; 10.2 moves mean, 0 falls/20 on
  randomized holds). ~50M steps total, all PPO.
- Handhold randomization: env args `rung_jitter_y` (m) / `rung_jitter_z` (fraction of
  spacing), CLI `--jitter-y/--jitter-z` on train/view/eval scripts. Rungs above the
  starting pair spawn jittered; each recycled rung gets a fresh random offset.
- **SLOW PRISMATICS (2026-07-02):** `DELTA_SCALE` prismatic entries cut 4× (0.006 →
  0.0015 m/step; full stroke now ~3.6 s). This is a physical-realism change requested by
  the user — policies trained before it (endless8 and earlier) drop to ~1-2 moves and MUST
  be replayed/compared with the old value if ever needed. Current best on slow actuators:
  `slow1` (~29M steps, warm-started from endless8): 7.4 moves/ep left-first, 3.5
  right-first on randomized holds, 1200-step episodes. Still improving with training;
  resume with `--tag slow1 --resume --ent-coef 0.0003 --lr 5e-5`. Use `caffeinate -dims`
  for long runs (plain `caffeinate -i` doesn't stop lid-close sleep, runs kept dying).
- Form shaping: `r_level = -10·(|z_r1−z_l1| + |z_r2−z_l2|)` (paired shoulder joint anchors
  level → punishes base roll) on top of the tilt penalty.
- **The recommended scene/setup is the treadmill** (`scene_v7_treadmill.xml`, `--treadmill`/`--simple`).
- **MESH HOLDS (2026-07):** Original STL convex mesh parts are the grip geometry
  (`scene_v7_treadmill_holds.xml`). Site at the hook engagement point (same as `scene_v7.xml`).
  Feasibility: `python test_mesh_holds.py`. Train:
  `python train_v7.py --mesh-holds --treadmill ... --init-from slow1`. View:
  `mjpython view_ppo.py --tag mesh1 --mesh-holds --treadmill ...`. Hang pose:
  `hang_start_treadmill_holds.npy` (treadmill hang ports directly now).
- **NATURAL FORM (2026-08):** `--natural` fine-tune prior for quieter, more human-like
  climbs. Env flag `natural_form=True` (CLI `--natural`): tighter upright/level
  penalties, action-jerk smoothness, wall-proximity potential, distance-gated
  pull-then-reach cadence weights, default `time_penalty=0.03`. Warm-start from
  `slow1` (cylinder treadmill, slow prismatics):
  `python train_v7.py --treadmill --spacing 0.15 --max-steps 1200
  --jitter-y 0.05 --jitter-z 0.25 --init-from slow1 --tag natural1 --natural
  --lr 5e-5 --ent-coef 0.0003 --target-kl 0.03 --n-envs 8 --timesteps 5000000`.
  Watch `climb/tilt`, `climb/moves_done`, `climb/ang_speed`. Defaults stay OFF so
  older tags remain bit-identical.
- **`mesh1` POST-MORTEM (2026-07-05): 0% success at 3.3M steps — do not resume it.**
  Root causes, verified headless: (a) the mesh's grippable lip is only ±0.03 m wide in y,
  flanked by horns rising 0.036-0.040 m above it (cylinder rung: ±0.09 m, nothing above) —
  a seated hook tolerates lateral shift to ~±0.04 m but the reach must INSERT into that
  pocket, a precision `slow1` never needed; (b) the run stacked mesh + jitter-y 0.05
  (> pocket half-width) + jitter-z 0.25 in one jump; (c) with zero successes ever, PPO had
  no gradient toward the grab — training logs show d_support 0.11-0.15 m (support hand off
  its hold most of every episode), approx_kl 0.06-0.31 / clip_fraction ~0.5 (no target_kl
  → policy thrash) and std collapsed to 0.117 (too cold to discover the pocket).
  Fixes in place: `_cradled` now gates |dy|<0.04 on mesh (honest grabs), `cradle_gauss`
  y-sigma 0.06→0.035 on mesh (shaping points at pocket center), `default_hang_qpos`
  prefers `hang_start_treadmill_holds.npy`, `train_v7.py` gained `--target-kl` and
  `--init-std` (and `--ent-coef/--lr` now also apply to fresh runs).
  Restart recipe: fresh tag from `slow1`, mesh with NO jitter first, then anneal:
  `python train_v7.py --mesh-holds --treadmill --spacing 0.15 --max-steps 1200
  --init-from slow1 --tag mesh2 --lr 5e-5 --ent-coef 0.002 --init-std 0.3
  --target-kl 0.03 --n-envs 8 --timesteps 10000000`, then a new tag with
  `--jitter-y 0.02 --jitter-z 0.1`, then `0.04/0.25`. Watch `climb/moves_done` and
  `train/approx_kl` (should sit ≤0.03 now).

---

## Quickstart (macOS)

```bash
# deps (once). If torch import fails with a missing libtorch_cpu.dylib, reinstall it:
#   pip uninstall -y torch && pip install --force-reinstall torch
pip install mujoco gymnasium stable-baselines3 scipy matplotlib

# TRAIN the easy curriculum (recommended starting point)
python train_v7.py --simple --timesteps 3000000 --n-envs 8 --tag simple

# WATCH the latest checkpoint (macOS needs mjpython for the viewer)
mjpython view_ppo.py --tag simple --simple --which latest

# metrics
tensorboard --logdir runs_v7      # watch climb/moves_done, climb/d_l, rollout/success_rate

# just look at the hang pose statically (no policy)
mjpython view_hang.py             # add --sim to let physics run
```

Training is **headless** (no live window). To "watch it train," re-run `view_ppo` on the
latest checkpoint every minute or two. Checkpoints save every 50k steps (~18 s at ~2800 fps);
best model every 25k.

---

## File map

| File | What it is |
|---|---|
| `climb_env_v7.py` | The Gymnasium env. All control / obs / reward / gait logic. |
| `train_v7.py` | PPO trainer (stable-baselines3). Flags below. Needs torch. |
| `cem_train_v7.py` | Dependency-free Cross-Entropy-Method trainer (no torch). Sanity/eval. |
| `view_ppo.py` | Watch a trained PPO checkpoint in the viewer (mjpython). |
| `eval_headless.py` | Headless eval: success rate, moves, falls, approach dist per episode. |
| `eval_hands.py` | Headless eval split by starting hand (left-first vs right-first). |
| `view_hang.py` | Open a scene at the saved hang pose (static or `--sim`). |
| `prove_learning_cem.py` | Standalone CEM proof that the reach task is learnable. |
| `hang_start_treadmill.npy` | Saved hang qpos for the treadmill scene (env loads on reset). |
| `hang_start_treadmill_holds.npy` | Hang qpos for mesh-hold treadmill (grip rails; pose searched). |
| `test_mesh_holds.py` | Feasibility test: can hooks load on mesh-hold scene? |
| `hang_start_ladder.npy` | Saved hang qpos for the ladder scene. |
| `../assets/robot_v7.xml` | Tuned robot (actuators, masses) + the **catch bars** (grip fix). |
| `../assets/scene_v7_treadmill.xml` | **Recommended** scene: 6 mocap rungs the env scrolls. |
| `../assets/scene_v7_treadmill_holds.xml` | Treadmill with mesh hold collision (original STLs). |
| `../assets/scene_v7_ladder.xml` | Fixed dense ladder (0.2 m), 10 rungs. Pre-treadmill. |
| `../assets/scene_v7.xml` | Reach task (mesh holds), torso anchored. Stage-0 only. |
| `runs_v7/<tag>/` | PPO outputs: `checkpoints/`, `best/`, `vecnormalize.pkl`, `tb/`. |

---

## The robot & control

- Two arms, each = 2 revolute joints + 1 prismatic. The prismatic is a 3-stage coupled
  linear actuator (joints `r3_1=r3_2=r3_3` coupled via `<equality>`), tripling its travel.
- 6 actuated joints, in this order everywhere (`ACT_JOINTS`): `r1, r2, r3_1, l1, l2, l3_1`.
- Floating base (`freejoint`), ~3.1 kg.
- **Control = joint-space residual position.** Action is 6-dim in [-1,1]; each step the
  per-joint target integrates `action * DELTA_SCALE`
  (`[0.05,0.05,0.006, 0.05,0.05,0.006]` rad/m), clipped to joint limits, sent to position
  servos. There is **no IK in the loop** (that was the old design's main failure).
- Actuators (`robot_v7.xml`): position servos, revolute `kp=120`, prismatic `kp=4000`,
  force limits ±12 N·m / ±150 N. `actuatorgravcomp=true` on the joints.
- `frame_skip=20`, `dt=0.002` → 0.04 s/step (25 Hz control).

### The grip (IMPORTANT — do not "simplify" this away)
The original gripper hooks **cannot** grip a plain cylinder — proven three ways (the robot
just slides off / falls). The fix is a **minimal catch bar** per gripper (`r_catch`,
`l_catch` in `robot_v7.xml`): a thin cylinder just above the grip point. The existing hook
cradles the rung from below; the catch bar stops it sliding out under load. With it, a single
arm holds the full body weight. **Keep the catch bars.** Removing them = the robot can't hang.

---

## Modes (env flags)

- **reach / stage-0** (`scene_v7.xml`, `freeze_base=True`): torso welded to world; learn to
  reach the holds. Not climbing; legacy.
- **climb** (`climb=True`): free torso, start from a saved hang pose, hand-over-hand gait,
  height reward. Uses `scene_v7_ladder.xml`.
- **treadmill** (`treadmill=True`, implies climb): **recommended.** Rungs are mocap bodies the
  env repositions — a small scrolling window so there are **never rungs below the body**
  (this structurally kills the base-jam problem). On a successful grab the lowest rung
  recycles to the top. Uses `scene_v7_treadmill.xml`.
- **simple** (`--simple`): the easy curriculum on top of treadmill — `gravity_scale=0.4`
  (body ~12 N, easy pull-up, little grip load), `tm_spacing=0.15`, `max_moves=1`
  (episode ends after ONE successful move → short, clean, lots of reps).

Tunable knobs (env args / CLI): `gravity_scale`, `tm_spacing`, `max_moves`,
`init_joint_noise`. CLI overrides on `train_v7.py`: `--gravity-scale`, `--spacing`,
`--max-moves` (use these to anneal the curriculum).

`train_v7.py` flags: `--climb --treadmill --simple --resume --tag --timesteps --n-envs
--gravity-scale --spacing --max-moves`. `--resume` continues from the latest checkpoint of
the tag (loads model + VecNormalize stats).

---

## Observation (climb/treadmill mode, 25-dim)
`[ joint_qpos(6, normalized), joint_qvel(6, scaled), active_hand→target_rung vec(3),
   support_hand→its_rung vec(3), base_z(1), base_up(3), f_active(1), f_support(1),
   active_is_left(1) ]`. (reach mode is 24-dim, slightly different.)

## Gait & success detection (2026-07 design — load-bearing)
- The **active** hand reaches its next same-side rung; the **support** hand must hold.
- **Force = `_uplift_on`:** SIGNED upward (world +z) force the rung exerts on the hook
  geoms (contact-frame force rotated to world). A real hold carrying weight reads strongly
  positive; a frontal "fistbump" against the rung reads ~0. Sign calibrated at the hang
  (both hooks sum to body weight, +30.4 N). Catch bars ARE in the sensing sets now.
- **`_cradled`:** grip site within a tight window of the rung center (|dx|<0.03,
  −0.005≤dz≤0.035) — the rung is inside the hook cradle, not poked from the side/bottom.
- A move counts (advance) only when ALL of these hold for **3 consecutive control steps**:
  `d_active < 0.07` AND `uplift_active > 0.15·W` AND `uplift_support > 0.20·W` AND
  `cradled` (W = body weight, so thresholds auto-scale with `gravity_scale`).
- On advance: `moves_done += 1`, hands swap roles, and (treadmill) the lowest rung recycles
  to the top (endless regenerating handholds).
- **MIRROR gait (`mirror=True`, default):** when the RIGHT hand is active the obs is
  left-right mirrored (swap arm blocks with signs (−,−,+), negate y of world vectors) and
  the action mirrored back. The policy only ever learns "reach with the left hand" and the
  skill transfers to both roles exactly (mirror map verified kinematically to ~3 mm).
  Without this, the policy could do one move and then deadlocked on the mirrored reach.

## Reward (climb/gait, `_climb_reward`, 2026-07)
- ALL dense guidance is **potential-based** (`r_shape = Φ(s′) − Φ(s)`): hovering anywhere
  nets ~0/step, so reward only accumulates by completing moves. (A per-step version taught
  the policy to park at the rung and farm the dense terms — 0% success. Don't go back.)
- `Φ = -8·min(d_active,0.6) + 3·cradle_gauss + 4·load + 1.5·pullup + 3·sup_load
     − 2·act_load_other + 4·base_z` where:
  - `cradle_gauss` = Gaussian of grip-site offset from the hooked-over pose (leads the
    hand up-and-over the rung, not into its face),
  - `load` = clipped uplift on the target rung, only when cradled,
  - `pullup` = support-arm prismatic retraction while support is firm ("pull all the way
    up with the holding arm to bring the next rung into reach"),
  - `sup_load`/`act_load_other` = weight transfer: hang off the SUPPORT hook, unload the
    hand that must move next (fixes the "hangs off the reaching hand and deadlocks" mode).
  - Φ re-anchors (`_prev_phi=None`) on reset and on every hand swap.
- Per-step penalties: support gate (−2 if support not load-bearing), support-hold
  (−6·min(d_support,0.3), stops dangling the support arm), anti-skip
  (−12·max(0, active_z−target_z−0.06)), uprightness (−2·tilt − 3·tilt² − 0.02·|ω|, tilt
  measured against the STARTING attitude → keeps the base vertical), ctrl (−0.002‖a‖²).
- On advance: `+40` and `+15·Δmax_height`. Fall (base_z < max_z − 0.25): `−10 − 2·(steps
  remaining)` and terminates (scaling makes early bailouts never reward-optimal).
- `--simple` ends the episode after `max_moves` with `+5`.
- Reset: **40-step settle** + rungs are placed relative to the post-noise grip sites so
  both hooks always start seated (24/24 noisy seeds stable); starting hand randomized.

---

## History / decisions (so you don't redo dead ends)

1. **`test_6` was unfixable by design:** nothing attached the robot to the wall, the gait
   state machine was commented out, the reward had grasp/pull weights = 0, and a nonlinear
   IK solver ran inside every RL step (slow + NaN blowups). v7 is a ground-up redesign.
2. **Joint-space control** replaced in-loop IK → ~20-30× faster, stable.
3. **Holds → rungs:** flat holds can't be gripped; horizontal cylindrical rungs can.
4. **Catch bar** added to grippers — the only thing that makes the passive hook bear load.
   (A bigger "claw" worked too but the user wanted minimal change to the gripper.)
5. **Base jam:** the bulky base hangs *among* lower rungs and jams them at 10-30 kN if they
   collide. Fix = collision excludes so **only the gripper hooks touch rungs** (base/arms
   pass freely). DO NOT re-enable base↔rung collision (it jams). Sloped rungs do NOT help
   (the base statically overlaps, it doesn't slide past).
6. **Freezing the support arm makes the move impossible** (no pull-up → can't reach; IK is
   92 mm short). Don't do it.
7. **Treadmill** (mocap scrolling rungs) is the clean fix: no rungs below the body, and the
   task becomes a repeatable "reach the next rung."
8. **Gravity assist** (`--simple`, 0.4 g) is the chosen way to make learning easier without
   breaking feasibility.
9. **`seated` check** added because contact force is unsigned — a bottom-touch used to count
   as a grab.

### Known-good facts
- Single arm holds full body weight (~34 N at 1 g) on a rung via the catch bar; hang stable
  1000+ steps even with the floor removed.
- A hand-over-hand move is kinematically feasible (IK solves to 0 mm with both arms).
- Reset settle → robust spawn (survives 12/12 seeds at noise 0.02-0.03).

### Solved since (2026-07)
- Curriculum annealing works but **jumping 0.7g→1.0g directly fails** (3% success);
  0.7→0.85→1.0 works (80%+ at each stage). Use `--init-from <tag>` to warm-start a new
  run from another tag's latest checkpoint (fresh timestep count).
- Grip detection is now load-bearing (signed uplift + cradle geometry + 3-step hold);
  catch bars are in the force sets. This is what fixed the "fistbump" false grabs.
- **Entropy blow-up:** long fine-tuning runs with default `ent_coef=0.005` inflated
  `log_std` to +3.5 (pure noise) and destroyed the policy. Fine-tune stages should pass
  `--ent-coef 0.001 --lr 1e-4` (train_v7 also clamps a warm-started log_std to ≤0).
- **One-move deadlock** was two problems: (a) the mirrored reach was out-of-distribution
  (fixed by the mirror trick — instant transfer, no retraining needed), (b) the policy
  hung off the reaching hand so the support hook never loaded (fixed by the weight-transfer
  terms in Φ).

### Open problems / next steps
- Anneal spacing 0.15 → 0.2 m if wider rungs are wanted (endless7 trained at 0.15).
- Tilt p95 is ~30°; tighten the upright weights if a stricter attitude is needed.
- `eval_headless.py` / `eval_hands.py` are the quick verification tools (see below).
- **Competence/robustness roadmap (2026-07-05, mesh2 era).** New opt-in knobs (defaults
  off, running runs unaffected): `--time-penalty` (constant per-step cost; the potential
  shaping is time-indifferent, so this rewards FAST insertion — try 0.05),
  `--push` (random 0.2 s mostly-lateral shoves on the base up to N newtons, robustness —
  try 3-8, body is ~30 N), `--joint-noise` (reset randomization, default 0.02).
  Staging discipline (mesh1's lesson: ONE new axis per stage, warm-start each):
  1) mesh2 to 10M (no jitter); if moves_done <1 at end, insertion stage `--max-moves 1
     --time-penalty 0.05`, then unlimited-moves chaining stage;
  2) hold randomization `--jitter-y 0.02 --jitter-z 0.1` → `0.04/0.25` (y capped ~0.04
     by the mesh pocket width);
  3) pushes `--push 4` → `--push 8`; optionally `--joint-noise 0.04`.
  Evaluate each stage with eval_headless + eval_hands (both starting hands) before moving on.
- **VERIFIED SEATING + HONEST GRABS (2026-07-05).** User observed starts with a hook
  visibly off / "secure" grips that weren't. Audit confirmed: 11/30 mesh resets at noise
  0.02 started non-load-bearing (37%; cylinders 0/30). Three env fixes: (a) reset now
  RETRIES the noise draw (≤10×) until both hooks are cradled AND carrying ≥15% W after
  the settle (`_start_seated`, diagnostic `env._reset_tries`); (b) `_tm_set_window`
  matches the starting holds' lateral y to the post-noise hook positions on mesh (fixed
  y + ±3 cm pocket was the "starts perched on a horn" bug; cylinders keep nominal y);
  (c) `secure_steps` 3→5 on mesh only (0.2 s of sustained load to count a grab — kills
  bounce-fake advances; cylinder lineage untouched at 3). Post-fix audit: 30/30 verified
  seats at noise 0.02 AND 0.05. NOTE: a run resumed across this change sees a slightly
  stricter gate + cleaner start distribution — expect a brief moves_done dip, not a
  regression.

---

## Gotchas
- **macOS viewer needs `mjpython`** (ships with the mujoco pip pkg), not `python`. Headless
  training/CEM run under normal `python`.
- **torch on macOS is fragile** — if `import torch` fails with a missing dylib, reinstall.
- **VecNormalize:** PPO trains on normalized obs. The trainer saves `vecnormalize_*.pkl`
  alongside every checkpoint; `view_ppo.py` loads the matching one. Without those stats a
  checkpoint replays as garbage — don't move/delete them.
- Each `train_v7.py` run is one tag; use `--resume` to continue, or a new `--tag` for a
  fresh curve.
- If you change the env's observation size or reward, a `--resume` run will wobble for
  ~100-200k steps as the value function re-fits — expected, not a regression.
- Don't override `info["is_success"]` for climb mode (it's set in the reward from `moves_done`).

---

## How to verify a change quickly (headless, no torch)
```bash
python - <<'PY'
import numpy as np; from climb_env_v7 import ClimbBotEnv
e=ClimbBotEnv(xml_path="../assets/scene_v7_treadmill.xml", treadmill=True, max_moves=1,
              tm_spacing=0.15, gravity_scale=0.4, init_qpos_file="hang_start_treadmill.npy",
              max_steps=200, init_joint_noise=0.02)
o,_=e.reset(seed=0)
for _ in range(150): o,r,t,tr,info=e.step(np.zeros(6))
print("base_z", round(info["base_z"],3), "survived", not t, "Rhook", round(e._hook_force(e.r_hook_gids)))
PY
```
A healthy treadmill hang stays up (base ≈ 0.09, doesn't terminate) under zero action.
