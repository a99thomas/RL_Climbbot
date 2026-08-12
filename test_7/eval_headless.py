"""
eval_headless.py — quantify a checkpoint's climbing competence without a viewer.

Reports per-episode: moves completed, closest approach of the active hand to its
target rung, whether it fell, and the step at which each move landed. Aggregates
success rate and the dominant failure mode.

Usage:
    python eval_headless.py --tag simple --which latest --episodes 30 --simple
    python eval_headless.py --tag anneal --which best --episodes 30 \
        --gravity-scale 1.0 --spacing 0.2 --max-moves 0
"""
import argparse, os, glob, re
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from climb_env_v7 import (ClimbBotEnv, DEFAULT_XML, resolve_treadmill_xml, default_hang_qpos,
                          is_mesh_holds_xml)


def newest(paths):
    def steps(p):
        m = re.search(r"_(\d+)_steps", p)
        return int(m.group(1)) if m else -1
    return max(paths, key=steps) if paths else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="simple")
    ap.add_argument("--which", choices=["best", "final", "latest"], default="latest")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--simple", action="store_true")
    ap.add_argument("--mesh-holds", action="store_true",
                    help="treadmill with original mesh hold collision (not cylinder rungs)")
    ap.add_argument("--gravity-scale", type=float, default=None)
    ap.add_argument("--spacing", type=float, default=None)
    ap.add_argument("--max-moves", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--jitter-y", type=float, default=0.0)
    ap.add_argument("--jitter-z", type=float, default=0.0)
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--outdir", default="runs_v7")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    base = os.path.join(args.outdir, args.tag)
    if args.ckpt:
        model_path = args.ckpt
    elif args.which == "best":
        model_path = os.path.join(base, "best", "best_model.zip")
    elif args.which == "final":
        model_path = os.path.join(base, "ppo_v7_final.zip")
    else:
        model_path = newest(glob.glob(os.path.join(base, "checkpoints", "ppo_v7_*_steps.zip")))
    assert model_path and os.path.exists(model_path), f"model not found: {model_path}"

    gs = 0.4 if args.simple else 1.0
    sp = 0.15 if args.simple else 0.2
    mm = 1 if args.simple else 0
    if args.gravity_scale is not None: gs = args.gravity_scale
    if args.spacing is not None: sp = args.spacing
    if args.max_moves is not None: mm = args.max_moves

    xml = resolve_treadmill_xml(args.mesh_holds)
    init_qpos = default_hang_qpos(xml)

    def make():
        return ClimbBotEnv(xml_path=xml, treadmill=True, max_moves=mm, tm_spacing=sp,
                           gravity_scale=gs, init_qpos_file=init_qpos,
                           rung_jitter_y=args.jitter_y, rung_jitter_z=args.jitter_z,
                           natural_form=args.natural,
                           max_steps=args.max_steps, init_joint_noise=0.02)

    venv = DummyVecEnv([make])
    vp = os.path.join(base, "vecnormalize.pkl")
    if args.which == "latest" or not os.path.exists(vp):
        cand = newest(glob.glob(os.path.join(base, "checkpoints", "*vecnormalize*.pkl")))
        vp = cand or vp
    assert vp and os.path.exists(vp), "no VecNormalize stats found"
    venv = VecNormalize.load(vp, venv)
    venv.training = False
    venv.norm_reward = False
    print(f"model: {model_path}\nvecnorm: {vp}\ngravity={gs} spacing={sp} max_moves={mm}")

    model = PPO.load(model_path, device="cpu")
    env0 = venv.venv.envs[0].unwrapped if hasattr(venv.venv.envs[0], "unwrapped") else venv.venv.envs[0]

    results = []
    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        obs = venv.reset()
        min_d = np.inf
        fell = False
        moves = 0
        move_steps = []
        steps = 0
        tilts, ang_speeds = [], []
        while True:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, done, infos = venv.step(action)
            info = infos[0]
            steps += 1
            d = info.get("d_r", np.inf)
            min_d = min(min_d, d)
            if "tilt" in info:
                tilts.append(float(info["tilt"]))
            if "ang_speed" in info:
                ang_speeds.append(float(info["ang_speed"]))
            if info.get("moves_done", 0) > moves:
                moves = info["moves_done"]
                move_steps.append(steps)
            if done[0]:
                # terminal info: 'terminal_observation' present; check fall via base_z
                fell = bool(info.get("base_z", 1.0) < (env0._start_z - 0.25)) if hasattr(env0, "_start_z") else False
                break
        mean_tilt = float(np.mean(tilts)) if tilts else float("nan")
        results.append(dict(moves=moves, min_d=min_d, fell=fell, steps=steps,
                            move_steps=move_steps, mean_tilt=mean_tilt,
                            mean_ang=float(np.mean(ang_speeds)) if ang_speeds else float("nan")))
        print(f"ep {ep:2d}: moves={moves} min_d={min_d:.3f} fell={fell} "
              f"tilt={mean_tilt:.3f} steps={steps} at={move_steps}")

    moves = np.array([r["moves"] for r in results])
    min_ds = np.array([r["min_d"] for r in results])
    fells = np.array([r["fell"] for r in results])
    tilts = np.array([r["mean_tilt"] for r in results])
    print("\n===== SUMMARY =====")
    print(f"episodes: {len(results)}")
    print(f"success (>=1 move): {np.mean(moves >= 1)*100:.0f}%")
    print(f"moves: mean {moves.mean():.2f}  max {moves.max()}")
    print(f"min approach dist: mean {min_ds.mean():.3f}  median {np.median(min_ds):.3f}")
    print(f"fell: {np.mean(fells)*100:.0f}%")
    print(f"mean tilt (1-cosθ): {np.nanmean(tilts):.3f}  p95 {np.nanpercentile(tilts, 95):.3f}")
    if (moves >= 1).any():
        first = [r["move_steps"][0] for r in results if r["move_steps"]]
        print(f"steps to first move: mean {np.mean(first):.0f}")


if __name__ == "__main__":
    main()
