"""
view_ppo.py  —  Watch a trained PPO checkpoint in the MuJoCo viewer.

macOS: run with the PROJECT venv's mjpython (NumPy 2 — matches training pickles):
    ../.venv/bin/mjpython view_ppo.py --tag natural1 --which latest --treadmill \
        --spacing 0.15 --jitter-y 0.05 --jitter-z 0.25 --max-steps 1200
    ../.venv/bin/mjpython view_ppo.py --tag slow1 --which latest --treadmill --spacing 0.15
System/pyenv mjpython often has NumPy 1.x and cannot load VecNormalize stats saved
under NumPy 2.x (BitGenerator pickle error).

--which:  best  -> <tag>/best/best_model.zip
          final -> <tag>/ppo_v7_final.zip
          latest-> newest <tag>/checkpoints/ppo_v7_*_steps.zip
Or pass an explicit --ckpt path.

IMPORTANT: PPO here trains on VecNormalize-normalized observations. To replay faithfully
the matching normalization stats must be loaded. This script looks for (in order):
    <tag>/vecnormalize.pkl                  (saved at end of training)
    <tag>/checkpoints/ppo_v7_vecnormalize_*_steps.pkl   (saved per-checkpoint; newest)
If none is found it warns and runs UNNORMALIZED — the policy will look wrong. In that case
re-run training with the updated train_v7.py (it now saves stats every checkpoint).
"""
import argparse, os, glob, re, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from climb_env_v7 import (ClimbBotEnv, DEFAULT_XML, resolve_treadmill_xml,
                          default_hang_qpos, is_mesh_holds_xml)


def newest(paths):
    def steps(p):
        m = re.search(r"_(\d+)_steps", p)
        return int(m.group(1)) if m else -1
    return max(paths, key=steps) if paths else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gait")
    ap.add_argument("--which", choices=["best", "final", "latest"], default="best")
    ap.add_argument("--ckpt", default=None, help="explicit model .zip path (overrides --which)")
    ap.add_argument("--climb", action="store_true")
    ap.add_argument("--treadmill", action="store_true")
    ap.add_argument("--mesh-holds", action="store_true",
                    help="treadmill with original mesh hold collision (not cylinder rungs)")
    ap.add_argument("--simple", action="store_true")
    ap.add_argument("--spacing", type=float, default=None, help="rung spacing (m)")
    ap.add_argument("--gravity-scale", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--jitter-y", type=float, default=0.0, help="rung lateral randomization (m)")
    ap.add_argument("--jitter-z", type=float, default=0.0, help="vertical gap randomization (frac)")
    ap.add_argument("--xml", default=None)
    ap.add_argument("--init-qpos", default=None)
    ap.add_argument("--outdir", default="runs_v7")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--episodes", type=int, default=20)
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
    print("loading model:", model_path)

    if args.simple:
        args.treadmill = True
    if args.mesh_holds:
        args.treadmill = True
    xml = args.xml
    if xml is None:
        if args.treadmill:
            xml = resolve_treadmill_xml(args.mesh_holds)
        elif args.climb:
            xml = os.path.join(os.path.dirname(DEFAULT_XML), "scene_v7_ladder.xml")
    else:
        args.mesh_holds = args.mesh_holds or is_mesh_holds_xml(xml)
    init_qpos = args.init_qpos or default_hang_qpos(xml)

    def make():
        kw = dict(render_mode="human", max_steps=args.max_steps)
        if xml: kw["xml_path"] = xml
        if args.climb or args.treadmill:
            kw.update(climb=True, init_qpos_file=init_qpos)
        if args.treadmill:
            kw.update(treadmill=True)
        if args.simple:
            kw.update(tm_spacing=0.15, gravity_scale=0.4)
        if args.spacing is not None:
            kw.update(tm_spacing=args.spacing)
        if args.gravity_scale is not None:
            kw.update(gravity_scale=args.gravity_scale)
        if args.treadmill:
            kw.update(rung_jitter_y=args.jitter_y, rung_jitter_z=args.jitter_z)
        return ClimbBotEnv(**kw)

    venv = DummyVecEnv([make])

    # find normalization stats
    vp = os.path.join(base, "vecnormalize.pkl")
    if not os.path.exists(vp):
        cand = newest(glob.glob(os.path.join(base, "checkpoints", "*vecnormalize*.pkl")))
        vp = cand
    if vp and os.path.exists(vp):
        print("loading VecNormalize stats:", vp)
        try:
            venv = VecNormalize.load(vp, venv)
        except (ValueError, TypeError) as e:
            raise SystemExit(
                f"\nFailed to load VecNormalize ({e}).\n"
                "This is almost always a NumPy 1.x vs 2.x pickle mismatch.\n"
                "Use the project venv's mjpython instead of system mjpython:\n"
                "  ../.venv/bin/mjpython view_ppo.py --tag natural1 --which latest "
                "--treadmill --spacing 0.15 --jitter-y 0.05 --jitter-z 0.25 "
                "--max-steps 1200\n"
            ) from e
        venv.training = False
        venv.norm_reward = False
    else:
        print("\n*** WARNING: no VecNormalize stats found — running UNNORMALIZED. "
              "The policy will likely look wrong. Re-train with the updated train_v7.py "
              "(saves stats every checkpoint), or let training finish. ***\n")

    model = PPO.load(model_path)
    obs = venv.reset()
    print("playing in real time — close the viewer window or Ctrl-C to stop.")
    ep = 0
    while ep < args.episodes:
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, done, info = venv.step(action)
        time.sleep(0.04)   # ~real time (frame_skip 20 * dt 0.002 = 0.04 s/step)
        if done[0]:
            ep += 1
            obs = venv.reset()


if __name__ == "__main__":
    main()
