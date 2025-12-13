# train_climbbot.py
# top of your train_stage0.py (or any script)
try:
    import gymnasium as gym
except Exception:
    import gym  # fallback, but prefer gymnasium

import os
import argparse
import multiprocessing as mp
from typing import Callable
import numpy as np

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# your env
from envs.climb_env import ClimbBotStage0Env

# ------------------- wrapper & factory -------------------
class GymnasiumToGymWrapper(Monitor):
    """
    Wrap a Gymnasium env so it behaves like classic gym.Env for SB3:
      - reset() returns obs only (Monitor returns (obs, info) so we override)
      - step() returns (obs, reward, done, info) where done = terminated or truncated
    We inherit Monitor to get episode logging for SB3.
    """
    def __init__(self, env):
        # Monitor expects a gym.Env-like interface: we implement reset/step to match that
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Monitor.reset returns obs, so call parent's wrapper to initialize Monitor internals
        super_data = super().reset(seed=kwargs.get("seed", None))
        # parent Monitor.reset returns (obs, info) in newer versions; we return obs only to SB3
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        # use Monitor.step for episode bookkeeping (it expects gym-style step)
        # Monitor.step expects (obs, reward, done, info) where obs is the observation AFTER the step
        # We'll call Monitor.step by delegating to the parent wrapper's underlying attribute
        _ = super().step(reward, obs=obs, done=done)
        return obs, float(reward), bool(done), info

def make_env_fn(xml_path: str, seed: int = 0, render_mode: str = None, debug: bool = False) -> Callable:
    def _init():
        env = ClimbBotStage0Env(xml_path=xml_path, render_mode=render_mode, debug=debug)
        # Gymnasium env: wrap for SB3 compatibility and monitoring
        env = GymnasiumToGymWrapper(env)
        # seed env (Gymnasium-style)
        try:
            env.env.reset(seed=seed)
        except Exception:
            # older envs might not accept seed this way; ignore if it fails
            pass
        return env
    return _init

# ------------------- Render / Eval callback -------------------
class RenderEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 1, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        # run evaluation every eval_freq environment steps
        if (self.num_timesteps - self.last_eval_step) >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            if self.verbose:
                print(f"\n[RenderEvalCallback] Running evaluation at step {self.num_timesteps} ...")

            # quick metric evaluation (no render)
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=1, render=False, deterministic=True)
            if self.verbose:
                print(f"[RenderEvalCallback] Eval (no render) mean_reward={mean_reward:.3f} +- {std_reward:.3f}")

            # explicit rendered episodes (use model.predict on wrapped env)
            for ep in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                total_r = 0.0
                steps = 0
                while True:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    total_r += float(reward)
                    steps += 1
                    try:
                        self.eval_env.env.render()  # underlying gymnasium env's render
                    except Exception:
                        print("[RenderEvalCallback] Warning: render() failed during eval render.")
                    if terminated or truncated:
                        if self.verbose:
                            print(f"[RenderEvalCallback] Eval episode {ep+1} finished: reward={total_r:.3f}, steps={steps}")
                        break
        return True

    def _on_training_end(self) -> None:
        try:
            self.eval_env.close()
        except Exception:
            pass

# ------------------- main training -------------------
def main(args):
    # ensure spawn on macOS / safety across platforms
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # create vectorized training envs
    seeds = [args.seed + i for i in range(args.num_cpu)]
    if args.num_cpu > 1:
        env_fns = [make_env_fn(args.xml, seed=s, render_mode=None, debug=False) for s in seeds]
        train_env = SubprocVecEnv(env_fns)
    else:
        env_fns = [make_env_fn(args.xml, seed=args.seed, render_mode=None, debug=False)]
        train_env = DummyVecEnv(env_fns)

    # wrap with monitoring and normalization
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # evaluation env (non-vectorized) — wrap for SB3 compatibility & rendering
    eval_wrapped = GymnasiumToGymWrapper(ClimbBotStage0Env(xml_path=args.xml, render_mode=None, debug=False))
    # set seed for eval env
    try:
        eval_wrapped.env.reset(seed=args.seed + 999)
    except Exception:
        pass

    # configure logger
    tmp_path = os.path.join(args.logdir, "sb3_logs")
    new_logger = configure(tmp_path, ["stdout", "tensorboard"])

    # compute sensible n_steps per CPU (keep total rollout length near 2048)
    n_steps_per_env = max(256, args.n_steps // max(1, args.num_cpu))

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=args.policy_kwargs,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
        ent_coef=args.ent_coef,
        learning_rate=args.lr,
        n_steps=n_steps_per_env,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gae_lambda=args.gae_lambda,
        device=args.device,
    )
    model.set_logger(new_logger)

    # callbacks
    render_cb = RenderEvalCallback(eval_env=eval_wrapped, eval_freq=args.eval_freq, n_eval_episodes=args.n_eval_episodes, verbose=1)

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=render_cb)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Save final model and VecNormalize (obs stats)
        model.save(os.path.join(args.save_dir, "ppo_climbbot_final"))
        try:
            # VecNormalize has save() on the object for sb3 v1.x
            train_env.save(os.path.join(args.save_dir, "vecnormalize.pkl"))
        except Exception:
            # older/newer APIs differ; attempt to reach underlying VecNormalize if wrapped
            try:
                # if DummyVecEnv / SubprocVecEnv were wrapped: find VecNormalize instance
                env_unwrapped = train_env
                # iterate attrs to find VecNormalize
                if hasattr(env_unwrapped, "venv"):
                    env_unwrapped = env_unwrapped.venv
                if hasattr(env_unwrapped, "save"):
                    env_unwrapped.save(os.path.join(args.save_dir, "vecnormalize.pkl"))
            except Exception:
                print("Warning: failed to save VecNormalize stats; you'll need to recreate normalization at inference.")

        train_env.close()
        eval_wrapped.close()
        print("Training finished. Artifacts saved to:", args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="robot_mjcf.xml", help="Path to MJCF xml file for the robot")
    parser.add_argument("--logdir", type=str, default="logs/climbbot_ppo", help="logging / tensorboard directory")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="where to save models and VecNormalize")
    parser.add_argument("--num-cpu", type=int, default=5, help="number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="total training timesteps")
    parser.add_argument("--n-steps", type=int, default=2048, help="total rollout length (will be divided across num-cpu)")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="entropy coefficient")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="evaluation frequency (env steps)")
    parser.add_argument("--n-eval-episodes", type=int, default=1, help="number of rendered eval episodes at each eval")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="torch device (cpu / cuda)")
    parser.add_argument("--policy-kwargs", type=eval, default="dict(net_arch=[dict(pi=[256,256], vf=[256,256])])", help="policy kwargs (python dict literal)")
    args = parser.parse_args()

    # if user passed policy-kwargs as string, ensure it's a dict
    if isinstance(args.policy_kwargs, str):
        args.policy_kwargs = eval(args.policy_kwargs)

    main(args)
