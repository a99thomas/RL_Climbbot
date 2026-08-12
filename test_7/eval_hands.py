"""Evaluate a checkpoint separately for LEFT-first and RIGHT-first starts."""
import argparse, glob, re, os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from climb_env_v7 import ClimbBotEnv, resolve_treadmill_xml, default_hang_qpos


def newest(paths):
    steps = lambda p: int(re.search(r"_(\d+)_steps", p).group(1))
    return max(paths, key=steps) if paths else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mesh-holds", action="store_true",
                    help="treadmill with original mesh hold collision (not cylinder rungs)")
    ap.add_argument("--which", choices=["best", "final", "latest"], default="latest")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-moves", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--spacing", type=float, default=0.15)
    ap.add_argument("--gravity-scale", type=float, default=1.0)
    ap.add_argument("--jitter-y", type=float, default=0.0)
    ap.add_argument("--jitter-z", type=float, default=0.0)
    args = ap.parse_args()

    xml = resolve_treadmill_xml(args.mesh_holds)
    init_qpos = default_hang_qpos(xml)
    def make():
        return ClimbBotEnv(xml_path=xml, treadmill=True, max_moves=args.max_moves,
                           tm_spacing=args.spacing, gravity_scale=args.gravity_scale,
                           rung_jitter_y=args.jitter_y, rung_jitter_z=args.jitter_z,
                           init_qpos_file=init_qpos,
                           max_steps=args.max_steps, init_joint_noise=0.02)
    venv = DummyVecEnv([make])
    venv = VecNormalize.load(newest(glob.glob(f"runs_v7/{args.tag}/checkpoints/*vecnormalize*.pkl")), venv)
    venv.training = False; venv.norm_reward = False
    mp = newest(glob.glob(f"runs_v7/{args.tag}/checkpoints/ppo_v7_*_steps.zip"))
    print("ckpt:", mp)
    model = PPO.load(mp, device="cpu")
    e = venv.venv.envs[0].unwrapped

    for first in ("left", "right"):
        moves_all = []
        for ep in range(args.episodes):
            obs = venv.reset()
            e.active = first
            e._prev_active_d = e._active_target_dist()
            e._prev_phi = None
            obs = venv.normalize_obs(np.array([e._get_obs()]))
            m = 0
            for t in range(args.max_steps):
                a, _ = model.predict(obs, deterministic=True)
                obs, r, done, infos = venv.step(a)
                m = infos[0].get("moves_done", m)
                if done[0]:
                    break
            moves_all.append(m)
        print(f"{first}-first: moves per ep {moves_all}  mean {np.mean(moves_all):.2f}  "
              f">=1: {np.mean(np.array(moves_all)>=1)*100:.0f}%  >=2: {np.mean(np.array(moves_all)>=2)*100:.0f}%")


if __name__ == "__main__":
    main()
