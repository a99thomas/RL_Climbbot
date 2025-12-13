"""
train.py

Train a policy to reach the two handholds using PPO (continuous actions).
Requires: stable-baselines3, gymnasium, mujoco (python bindings), torch.

Usage:
    python train.py --xml scene.xml --timesteps 200000
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
import os

from climb_env_5 import ClimbBotEnv

# add these imports at the top of train.py
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np

class TensorboardInfoCallback(BaseCallback):
    """
    Custom callback for logging additional env info scalars to TensorBoard.

    Works with vectorized envs (e.g., DummyVecEnv). The callback reads the `infos`
    produced by the environment at each step (your env already returns keys like
    'd_right', 'd_left', 'contact_right', 'contact_left', 'base_height', ...).
    It logs per-step means across the vectorized envs and also keeps short
    rolling averages to smoother curves in TensorBoard (logged each rollout end).
    """

    def __init__(self, verbose=0, smoothing_window=200):
        super().__init__(verbose)
        self.smoothing_window = smoothing_window
        # deques for smoothing (key -> deque)
        self._buffers = {}
        # keys we expect from your env's info dict (will adapt if some are missing)
        self.default_keys = [
            "d_right", "d_left",
            "contact_right", "contact_left",
            "base_height",
        ]

    def _ensure_buffer(self, key):
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.smoothing_window)

    def _on_training_start(self) -> None:
        # Called before training starts
        for k in self.default_keys:
            self._ensure_buffer(k)

    def _on_step(self) -> bool:
        """
        Called at every training step. We access the 'infos' dicts coming from the
        environment via self.locals. This is standard in SB3 callbacks.
        """
        infos = self.locals.get("infos", None)
        if infos is None:
            # fallback: nothing to log this step
            return True

        # If vec env: infos is a list (one per env). If not vectorized, it might be a dict.
        if isinstance(infos, dict):
            infos = [infos]

        # Collect values across all envs, for each key
        for info in infos:
            if not info:
                continue
            for key, val in info.items():
                # only handle scalars (numbers) - skip complex objects
                try:
                    scalar = float(val)
                except Exception:
                    continue
                # log instantaneous (mean across envs will be recorded below)
                self._ensure_buffer(key)
                self._buffers[key].append(scalar)

        # At every step, record the instantaneous mean across envs for the default keys.
        # This produces dense time-series in TensorBoard.
        for k in list(self._buffers.keys()):
            buf = self._buffers[k]
            if len(buf) == 0:
                continue
            mean_val = float(np.mean(buf))  # mean across recent steps across envs
            # Write to SB3 logger. This is visible in TensorBoard under the tag you choose below.
            # 'env/...' tags are conventional
            self.logger.record(f"env/{k}", mean_val)

        # Also ensure PPO's default logging still occurs
        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout. Log smoothed (rolling average) values which
        produce nicer, less noisy curves in TensorBoard.
        """
        for k, buf in self._buffers.items():
            if len(buf) == 0:
                continue
            smoothed = float(np.mean(buf))
            self.logger.record(f"smoothed/{k}", smoothed)

        # flush logger so values appear promptly in TensorBoard
        self.logger.dump(self.num_timesteps)



def make_train_env(xml, control_speed, render_mode, render_speed):
    def _thunk():
        # Training env should be headless (render_mode=None) for speed.
        env = ClimbBotEnv(model_path=xml, control_speed=control_speed,
                          render_mode=render_mode, render_speed=render_speed)
        return Monitor(env)
    return _thunk


def make_eval_env(xml, control_speed, render_mode, render_speed):
    def _thunk():
        # Eval env: typically headless during training. Set render_mode="human" only if you want to watch.
        env = ClimbBotEnv(model_path=xml, control_speed=control_speed,
                          render_mode=render_mode, render_speed=render_speed)
        return Monitor(env)
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str,
                        default="/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene copy.xml")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--save", type=str, default="ppo_climbbot.zip")
    parser.add_argument("--eval-freq", type=int, default=100_000,
                        help="How many training timesteps between evaluations")
    parser.add_argument("--n-eval-episodes", type=int, default=5,
                        help="Number of episodes per evaluation")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                        help="Save a checkpoint every this many timesteps")
    parser.add_argument("--control-speed", type=int, default=1,
                        help="How many physics steps per action")
    parser.add_argument("--render-speed", type=int, default=1,
                        help="Viewer sync rate if rendering (eval only)")
    parser.add_argument("--render-train", action="store_true",
                        help="If set, training env will use render_mode='human' (not recommended)")
    args = parser.parse_args()

    # -------- Create training env (headless unless user asks otherwise) ----------
    train_render_mode = "human" if args.render_train else None
    train_env = DummyVecEnv([make_train_env(args.xml, args.control_speed, train_render_mode, args.render_speed)])

    # -------- Create eval env (separate instance) ----------
    # Keep eval env headless during training for speed. If you want to *watch* evaluation,
    # change eval_render_mode to "human" here or create a separate ad-hoc script.
    eval_render_mode = None
    eval_env = DummyVecEnv([make_eval_env(args.xml, args.control_speed, eval_render_mode, args.render_speed)])

    # -------- Make directories for callbacks ----------
    best_model_dir = "best_model"
    chkpt_dir = "checkpoints"
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    # -------- Callbacks ----------
    # EvalCallback will evaluate the current model every `eval_freq` timesteps on `n_eval_episodes`,
    # and save the best model by mean reward to `best_model_dir`.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path="./eval_logs",
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,   # set True only if eval_env uses render_mode="human" and you want to see it
    )



    # Optional: periodic checkpoint saving (keeps last checkpoint and versioned files)
    checkpoint_callback = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=chkpt_dir,
                                             name_prefix="ppo_climb")


    # Custom tensorboard callback for logging extra env info
    tb_info_callback = TensorboardInfoCallback(verbose=0, smoothing_window=200)

    # -------- Create model ----------
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=2,
        tensorboard_log="./ppo_climb_tensorboard",
        learning_rate=1e-4,
        # you can tune other hyperparams here (n_steps, batch_size, etc.)
    )

    # -------- Train with callbacks ----------
    model.learn(total_timesteps=args.timesteps, callback=[eval_callback, checkpoint_callback, tb_info_callback])

    # -------- Save final policy ----------
    model.save(args.save)
    print("Saved final model to", args.save)
    print("Best model (by eval mean reward) saved to:", best_model_dir)

    # Close envs
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
