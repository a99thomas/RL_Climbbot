"""
train.py

Train a policy to reach the two handholds using PPO (continuous actions).
Requires: stable-baselines3, gymnasium, mujoco (python bindings), torch.

Usage:
    python train.py --xml scene.xml --timesteps 200000
"""

import argparse
import os
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

# Try to import your environment. This will work if you named it climbbot_env.py (ClimbBotEnv)
# or climb_env_3.py (ClimbEnv). Adjust the import if your file name differs.
try:
    from climb_env_4 import ClimbBotEnv as UserClimbEnv
except Exception:
    try:
        from climb_env_3 import ClimbEnv as UserClimbEnv
    except Exception:
        # Last resort: try a generic name
        try:
            from climb_env import ClimbEnv as UserClimbEnv
        except Exception as e:
            raise ImportError(
                "Could not import your climb environment. Make sure one of "
                "(climbbot_env.ClimbBotEnv, climb_env_3.ClimbEnv, climb_env.ClimbEnv) is available."
            ) from e


class SB3EnvInfoLogger(BaseCallback):
    """
    Callback for logging environment info dict values (d_right, d_left, reward components,
    and success) to TensorBoard via SB3 logger.

    This implementation is robust:
      - Looks for 'infos' in self.locals (typical during on_step).
      - Falls back to inspecting underlying envs (Monitor-wrapped envs) for last_info.
      - Attempts to log the following keys if present:
          'd_right','d_left','reward_total','reward_dist','reward_climb',
          'reward_shaping','reward_ctrl_pen','is_success','success'
    """

    def __init__(self, verbose=0, window: int = 100):
        super().__init__(verbose)
        self.window = int(window)

        # windows for metrics
        self.d_rights = deque(maxlen=self.window)
        self.d_lefts = deque(maxlen=self.window)
        self.successes = deque(maxlen=self.window)

        self.r_totals = deque(maxlen=self.window)
        self.r_climbs = deque(maxlen=self.window)
        self.r_dists = deque(maxlen=self.window)
        self.r_shapings = deque(maxlen=self.window)
        self.r_ctrls = deque(maxlen=self.window)

    def _process_info(self, info: dict):
        if not isinstance(info, dict):
            return

        # distances
        dr = info.get("d_right", None)
        dl = info.get("d_left", None)
        if dr is not None:
            try:
                self.d_rights.append(float(dr))
            except Exception:
                pass
        if dl is not None:
            try:
                self.d_lefts.append(float(dl))
            except Exception:
                pass

        # reward components
        rt = info.get("reward_total", None) or info.get("reward", None)
        rc = info.get("reward_climb", None)
        rd = info.get("reward_dist", None)
        rs = info.get("reward_shaping", None)
        rcp = info.get("reward_ctrl_pen", None)

        if rt is not None:
            try:
                self.r_totals.append(float(rt))
            except Exception:
                pass
        if rc is not None:
            try:
                self.r_climbs.append(float(rc))
            except Exception:
                pass
        if rd is not None:
            try:
                self.r_dists.append(float(rd))
            except Exception:
                pass
        if rs is not None:
            try:
                self.r_shapings.append(float(rs))
            except Exception:
                pass
        if rcp is not None:
            try:
                self.r_ctrls.append(float(rcp))
            except Exception:
                pass

        # success detection (many envs use 'is_success' or 'success')
        success_flag = None
        for k in ("is_success", "success", "succeeded"):
            if k in info:
                try:
                    success_flag = bool(info[k])
                    break
                except Exception:
                    success_flag = None

        # fallback: if distances present, compare to env threshold if available
        if success_flag is None and dr is not None:
            try:
                envs = getattr(self.training_env, "envs", None)
                if envs and len(envs) > 0:
                    env0 = envs[0]
                    # env may expose reward params or close_thresh
                    thresh = None
                    if hasattr(env0, "reward_params"):
                        rp = getattr(env0, "reward_params")
                        if isinstance(rp, dict):
                            thresh = rp.get("close_thresh", None)
                    if thresh is None:
                        # try an attribute directly
                        thresh = getattr(env0, "close_thresh", None)
                    if thresh is None:
                        thresh = 0.03
                    success_flag = (float(dr) < float(thresh))
            except Exception:
                success_flag = False

        if success_flag is None:
            success_flag = False

        try:
            self.successes.append(1 if bool(success_flag) else 0)
        except Exception:
            pass

    def _on_step(self) -> bool:
        infos = None
        if isinstance(self.locals, dict):
            infos = self.locals.get("infos", None)

        if infos:
            # infos typically a list (one per env)
            if isinstance(infos, (list, tuple)):
                for info in infos:
                    if isinstance(info, dict):
                        self._process_info(info)
            elif isinstance(infos, dict):
                self._process_info(infos)
        else:
            # fallback: inspect training_env.envs[...] for last_info (Monitor wrapper often stores it)
            try:
                envs = getattr(self.training_env, "envs", None)
                if envs:
                    for e in envs:
                        last_info = getattr(e, "last_info", None)
                        # Some Monitor wrappers store last_info in env.unwrapped or env.envs[0].last_info
                        if not isinstance(last_info, dict):
                            last_info = getattr(e, "env_method", lambda *_: None)  # no-op fallback
                        if isinstance(last_info, dict):
                            self._process_info(last_info)
            except Exception:
                pass

        # Write aggregated means to SB3/TensorBoard logger
        if len(self.d_rights) > 0:
            self.logger.record("env/d_right_mean", float(np.mean(self.d_rights)))
            self.logger.record("env/d_left_mean", float(np.mean(self.d_lefts)))
            self.logger.record("env/success_rate", float(np.mean(self.successes)))

        if len(self.r_totals) > 0:
            self.logger.record("env/reward_total_mean", float(np.mean(self.r_totals)))
        if len(self.r_climbs) > 0:
            self.logger.record("env/reward_climb_mean", float(np.mean(self.r_climbs)))
        if len(self.r_dists) > 0:
            self.logger.record("env/reward_dist_mean", float(np.mean(self.r_dists)))
        if len(self.r_shapings) > 0:
            self.logger.record("env/reward_shaping_mean", float(np.mean(self.r_shapings)))
        if len(self.r_ctrls) > 0:
            self.logger.record("env/reward_ctrl_mean", float(np.mean(self.r_ctrls)))

        return True


def make_env_ctor(xml_path, render=False):
    """
    Returns a thunk that constructs and returns a Monitor-wrapped environment.
    Tries both xml_path= and model_path= constructor signatures.
    """
    def _thunk():
        # try common constructor signatures
        kwargs = dict(render_mode=None)
        # first try xml_path
        try:
            env = UserClimbEnv(xml_path=xml_path, **kwargs)
        except TypeError:
            try:
                env = UserClimbEnv(model_path=xml_path, **kwargs)
            except TypeError:
                # fallback: try positional argument
                try:
                    env = UserClimbEnv(xml_path)
                except Exception:
                    env = UserClimbEnv(xml_path)  # let exception propagate if still wrong
        # wrap with Monitor for episode stats
        return Monitor(env)
    return _thunk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=str, default="/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene.xml", help="Path to MJCF / scene xml")
    p.add_argument("--timesteps", type=int, default=600_000)
    p.add_argument("--save", type=str, default="ppo_climbbot.zip")
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--tensorboard", type=str, default="./ppo_climb_tensorboard")
    p.add_argument("--best-model-dir", type=str, default="./logs/best_model/")
    p.add_argument("--eval-log-dir", type=str, default="./logs/eval_logs/")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    os.makedirs(args.best_model_dir, exist_ok=True)
    os.makedirs(args.eval_log_dir, exist_ok=True)
    os.makedirs(args.tensorboard, exist_ok=True)

    # single-process vectorized env (easy to debug and matches your example)
    env = DummyVecEnv([make_env_ctor(args.xml)])
    eval_env = DummyVecEnv([make_env_ctor(args.xml)])

    # set up callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.best_model_dir,
        log_path=args.eval_log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=True,  # set to True if you want to see evaluation rollout windows
    )

    checkpoint_callback = CheckpointCallback(save_freq=max(1, args.eval_freq // 2),
                                             save_path=os.path.dirname(args.save) or ".",
                                             name_prefix="ppo_climbbot_ckpt")

    info_logger = SB3EnvInfoLogger(verbose=1, window=200)

    # Configure SB3 logger to write TensorBoard logs alongside the folder
    new_logger = configure(args.tensorboard, ["stdout", "csv", "tensorboard"])

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=2,
        tensorboard_log=args.tensorboard,
        learning_rate=1e-4,
        ent_coef=0.0,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    )

    # model = PPO.load("//Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/envs/logs/best_model/best_model.zip", env=env)

    model.set_logger(new_logger)

    # Train
    callbacks = [info_logger, eval_callback, checkpoint_callback]
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    # Save final model
    model.save(args.save)
    print("Saved model to", args.save)


if __name__ == "__main__":
    main()
