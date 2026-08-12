"""
train_v7.py  —  PPO training for ClimbBot v7.

Full setup and curriculum: ../README.md and README_v7.md.

Quick start (from test_7/, conda env climbbot activated):
    python train_v7.py --simple --timesteps 3000000 --n-envs 8 --tag simple
    python train_v7.py --treadmill --spacing 0.15 --max-steps 1200 \\
        --init-from simple --tag climb --timesteps 5000000 --n-envs 8
    python train_v7.py --resume --tag climb --treadmill --spacing 0.15 ...

Outputs: runs_v7/<tag>/{checkpoints,best,tb}/ plus VecNormalize pickles.
"""
import argparse, os, glob, re
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from climb_env_v7 import (ClimbBotEnv, DEFAULT_XML, resolve_treadmill_xml,
                          default_hang_qpos, is_mesh_holds_xml, LADDER_XML)


def make_env(xml, freeze, max_steps, rank, seed=0, climb=False, init_qpos=None, treadmill=False,
             max_moves=0, tm_spacing=0.2, gravity_scale=1.0, jitter_y=0.0, jitter_z=0.0,
             time_penalty=0.0, push=0.0, joint_noise=0.02, natural_form=False):
    def _thunk():
        kw = dict(xml_path=xml, freeze_base=freeze, max_steps=max_steps, render_mode=None)
        if climb or treadmill:
            kw.update(climb=True, init_qpos_file=init_qpos, freeze_base=False,
                      init_joint_noise=joint_noise, gravity_scale=gravity_scale,
                      time_penalty=time_penalty, push_newton=push,
                      natural_form=natural_form)
        if treadmill:
            kw.update(treadmill=True, max_moves=max_moves, tm_spacing=tm_spacing,
                      rung_jitter_y=jitter_y, rung_jitter_z=jitter_z)
        env = ClimbBotEnv(**kw)
        env.reset(seed=seed + rank)
        keys = ("d_r", "d_l", "force_r", "force_l", "is_success")
        if climb or treadmill:
            keys = keys + ("moves_done", "tilt", "ang_speed")
        return Monitor(env, info_keywords=keys)
    return _thunk


class ClimbLogger(BaseCallback):
    """Log climbing-specific diagnostics (means over recent steps) to TensorBoard."""
    def __init__(self, window=2000):
        super().__init__()
        self.window = window
        self.buf = {k: [] for k in ["d_r", "d_l", "force_r", "force_l", "is_success",
                                    "moves_done", "tilt", "ang_speed"]}

    def _on_step(self):
        for info in self.locals.get("infos", []):
            for k in self.buf:
                if k in info:
                    self.buf[k].append(float(info[k]))
                    if len(self.buf[k]) > self.window:
                        self.buf[k].pop(0)
        for k, v in self.buf.items():
            if v:
                self.logger.record(f"climb/{k}", float(np.mean(v)))
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default=DEFAULT_XML)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--no-freeze", action="store_true", help="release torso anchor (free climbing)")
    p.add_argument("--climb", action="store_true", help="climbing mode: hang start + height reward")
    p.add_argument("--treadmill", action="store_true", help="scrolling-rung treadmill (recommended)")
    p.add_argument("--mesh-holds", action="store_true",
                   help="treadmill with original mesh hold collision (not cylinder rungs)")
    p.add_argument("--simple", action="store_true",
                   help="easiest curriculum: 0.4 g, 0.15 m rungs, 1 move/episode")
    p.add_argument("--gravity-scale", type=float, default=None, help="override gravity (e.g. 0.6) for curriculum")
    p.add_argument("--spacing", type=float, default=None, help="override rung spacing (m)")
    p.add_argument("--jitter-y", type=float, default=0.0,
                   help="randomize each upcoming rung's lateral position by ±this (m)")
    p.add_argument("--jitter-z", type=float, default=0.0,
                   help="randomize each vertical gap by ±this fraction of spacing")
    p.add_argument("--max-moves", type=int, default=None, help="override moves/episode (0 = unlimited)")
    p.add_argument("--init-qpos", default=None)
    p.add_argument("--tag", default="stage0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="runs_v7")
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint of --tag")
    p.add_argument("--init-from", default=None,
                   help="warm-start: tag whose latest checkpoint (policy + vecnorm stats) seeds "
                        "this NEW run (fresh timestep count, fresh outdir)")
    p.add_argument("--ent-coef", type=float, default=None,
                   help="override entropy coefficient (use 0.0-0.002 for fine-tuning stages; "
                        "the default 0.005 can blow up the action std on hard tasks)")
    p.add_argument("--lr", type=float, default=None, help="override learning rate")
    p.add_argument("--time-penalty", type=float, default=0.0,
                   help="constant per-step cost in climb mode (e.g. 0.05): makes faster "
                        "moves strictly better (the potential shaping is time-indifferent)")
    p.add_argument("--push", type=float, default=0.0,
                   help="robustness: random pushes on the base up to this many N "
                        "(occasional 0.2 s mostly-lateral shoves; try 3-8 N, body is ~30 N)")
    p.add_argument("--joint-noise", type=float, default=0.02,
                   help="init joint randomization at reset (robustness: try 0.03-0.05)")
    p.add_argument("--target-kl", type=float, default=None,
                   help="early-stop PPO epochs when approx_kl exceeds this (0.02-0.03 "
                        "recommended; mesh1 ran unbounded and thrashed at KL 0.06-0.3)")
    p.add_argument("--init-std", type=float, default=None,
                   help="reset the action std at warm start (e.g. 0.3): restores exploration "
                        "when transferring to new hold geometry, where the inherited std "
                        "(~0.12 after a long fine-tune) is too cold to discover new grabs")
    p.add_argument("--natural", action="store_true",
                   help="natural-form fine-tune: stronger upright/level/smoothness priors, "
                        "pull-then-reach cadence, wall proximity, default time_penalty=0.03")
    args = p.parse_args()

    freeze = (not args.no_freeze) and (not args.climb) and (not args.treadmill)
    if args.simple:
        args.treadmill = True
    if args.mesh_holds:
        args.treadmill = True
    mm = 1 if args.simple else 0     # one move per episode
    sp = 0.15 if args.simple else 0.2  # rung spacing
    gs = 0.4 if args.simple else 1.0   # gravity scale (lighter body = easy pull-up)
    # explicit overrides win (use these to anneal the curriculum back toward full difficulty)
    if args.gravity_scale is not None: gs = args.gravity_scale
    if args.spacing is not None: sp = args.spacing
    if args.max_moves is not None: mm = args.max_moves
    if args.treadmill:
        args.mesh_holds = args.mesh_holds or is_mesh_holds_xml(args.xml)
        args.xml = resolve_treadmill_xml(args.mesh_holds, args.xml)
    elif args.climb and args.xml == DEFAULT_XML:
        args.xml = LADDER_XML
    if args.init_qpos is None:
        args.init_qpos = default_hang_qpos(args.xml)
    outdir = os.path.join(args.outdir, args.tag)
    os.makedirs(outdir, exist_ok=True)
    print(f"scene: {args.xml}")
    print(f"hang:  {args.init_qpos}  |  mesh_holds={args.mesh_holds}")
    if args.natural and args.time_penalty <= 0.0:
        args.time_penalty = 0.03
    print(f"gravity={gs}  spacing={sp}  max_moves={mm}  jitter_y={args.jitter_y}  jitter_z={args.jitter_z}")
    print(f"natural_form={args.natural}  time_penalty={args.time_penalty}  push={args.push}")

    VecCls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecCls([make_env(args.xml, freeze, args.max_steps, i, args.seed, args.climb, args.init_qpos,
                            args.treadmill, mm, sp, gs, args.jitter_y, args.jitter_z,
                            args.time_penalty, args.push, args.joint_noise, args.natural)
                   for i in range(args.n_envs)])
    eval_raw = DummyVecEnv([make_env(args.xml, freeze, args.max_steps, 999, args.seed, args.climb,
                            args.init_qpos, args.treadmill, mm, sp, gs, args.jitter_y, args.jitter_z,
                            args.time_penalty, args.push, args.joint_noise, args.natural)])

    # --- resume from latest checkpoint of this tag, if requested ---
    def latest_ckpt(d=None):
        d = d or outdir
        cks = glob.glob(os.path.join(d, "checkpoints", "ppo_v7_*_steps.zip"))
        if not cks:
            return None, None
        steps = lambda p: int(re.search(r"_(\d+)_steps", p).group(1))
        m = max(cks, key=steps)
        vn = os.path.join(d, "checkpoints", f"ppo_v7_vecnormalize_{steps(m)}_steps.pkl")
        return m, (vn if os.path.exists(vn) else None)

    ckpt, vnorm = latest_ckpt() if args.resume else (None, None)
    if not ckpt and args.init_from:
        ckpt, vnorm = latest_ckpt(os.path.join(args.outdir, args.init_from))
        if ckpt:
            print(f"WARM-STARTING from {ckpt}")
    if ckpt:
        if args.resume:
            print(f"RESUMING from {ckpt}")
        # VecNormalize pickles embed a numpy Generator; numpy 1.x cannot unpickle
        # stats written by numpy 2.x (BitGenerator state format). Prefer a matching
        # numpy, but fall back to a fresh wrapper if load fails so training can start.
        if vnorm:
            try:
                venv = VecNormalize.load(vnorm, venv)
            except Exception as e:
                print(f"WARNING: could not load VecNormalize ({e}); starting fresh stats")
                venv = VecNormalize(venv, clip_obs=10.0, gamma=0.99)
        else:
            venv = VecNormalize(venv, clip_obs=10.0, gamma=0.99)
        venv.training = True; venv.norm_reward = True
        custom = {}
        if args.ent_coef is not None: custom["ent_coef"] = args.ent_coef
        if args.lr is not None: custom["learning_rate"] = args.lr
        if args.target_kl is not None: custom["target_kl"] = args.target_kl
        model = PPO.load(ckpt, env=venv, tensorboard_log=os.path.join(outdir, "tb"),
                         custom_objects=custom)
        # cap a blown-up exploration std from the previous stage: resample-friendly but
        # keeps the warm-started competence usable instead of drowning it in noise
        import torch
        with torch.no_grad():
            if args.init_std is not None:
                model.policy.log_std.fill_(float(np.log(args.init_std)))
            else:
                model.policy.log_std.clamp_(max=0.0)
    else:
        if args.resume:
            print("no checkpoint found; starting fresh")
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO(
            "MlpPolicy", venv,
            n_steps=1024, batch_size=2048, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, target_kl=args.target_kl,
            ent_coef=args.ent_coef if args.ent_coef is not None else 0.005,
            learning_rate=args.lr if args.lr is not None else 3e-4,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=os.path.join(outdir, "tb"),
            verbose=1, seed=args.seed,
        )

    eval_env = VecNormalize(eval_raw, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)

    callbacks = [
        ClimbLogger(),
        CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1),
                           save_path=os.path.join(outdir, "checkpoints"), name_prefix="ppo_v7",
                           save_vecnormalize=True),   # save obs-normalization stats too
        EvalCallback(eval_env, best_model_save_path=os.path.join(outdir, "best"),
                     log_path=os.path.join(outdir, "eval"),
                     eval_freq=max(25_000 // args.n_envs, 1), n_eval_episodes=5, deterministic=True),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False,
                reset_num_timesteps=not (args.resume and ckpt))
    model.save(os.path.join(outdir, "ppo_v7_final"))
    venv.save(os.path.join(outdir, "vecnormalize.pkl"))
    print("Saved model + vecnormalize to", outdir)


if __name__ == "__main__":
    main()
