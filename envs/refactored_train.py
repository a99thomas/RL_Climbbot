# train.py
"""
Train a policy to reach the two handholds using PPO (continuous actions).
Requires: stable-baselines3, gymnasium, mujoco (python bindings), torch.

Usage examples:
    python train.py --xml scene.xml --timesteps 200000
    python train.py --xml scene.xml --timesteps 500000 --climb_scale 50.0 --climb_success_bonus 300.0
"""

import argparse
import json
from collections import deque
from dataclasses import asdict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Dict

from climb_env_2 import ClimbEnv, RewardConfig


class SB3EnvInfoLogger(BaseCallback):
    """
    Callback for logging environment info dict values to TensorBoard via SB3 logger.

    Expects env info dicts to contain keys:
      - 'd_right', 'd_left'
      - 'reward_total' (float)
      - 'reward_components' (dict) with named components (dist, climb, shaping, ctrl_pen, contact, ...)
      - 'is_success'
    Aggregates a sliding window (mean) of values and writes them via self.logger.record().
    """

    def __init__(self, verbose=0, window: int = 200):
        super().__init__(verbose)
        self.window = int(window)

        # rolling windows
        self.d_rights = deque(maxlen=self.window)
        self.d_lefts = deque(maxlen=self.window)
        self.successes = deque(maxlen=self.window)

        self.r_totals = deque(maxlen=self.window)
        # components map name -> deque
        self.r_components = {}

    def _ensure_component(self, name: str):
        if name not in self.r_components:
            self.r_components[name] = deque(maxlen=self.window)

    def _on_step(self) -> bool:
        # `infos` is usually available in locals during _on_step
        infos = None
        if isinstance(self.locals, dict):
            infos = self.locals.get("infos", None)

        # Process infos (vectorized env -> list of infos)
        if infos is None:
            # fallback: attempt to read last_info from underlying envs (Monitor wrapper stores episode stats but not last_info by default)
            try:
                envs = getattr(self.training_env, "envs", None)
                if envs:
                    for e in envs:
                        last_info = getattr(e, "last_info", None) or getattr(e, "info", None)
                        if isinstance(last_info, dict):
                            self._process_info(last_info)
            except Exception:
                pass
        else:
            if isinstance(infos, (list, tuple)):
                for info in infos:
                    if isinstance(info, dict):
                        self._process_info(info)
            elif isinstance(infos, dict):
                self._process_info(infos)

        # Log aggregated stats if available
        if len(self.d_rights) > 0:
            self.logger.record("env/d_right_mean", float(np.mean(self.d_rights)))
            self.logger.record("env/d_left_mean", float(np.mean(self.d_lefts)))
            self.logger.record("env/success_rate", float(np.mean(self.successes)))

        if len(self.r_totals) > 0:
            self.logger.record("env/reward_total_mean", float(np.mean(self.r_totals)))

        # reward components
        for name, dq in self.r_components.items():
            if len(dq) > 0:
                self.logger.record(f"env/reward_comp/{name}_mean", float(np.mean(dq)))

        return True

    def _process_info(self, info: Dict):
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

        # total reward (top-level)
        rt = info.get("reward_total", None)
        if rt is not None:
            try:
                self.r_totals.append(float(rt))
            except Exception:
                pass

        # reward components (dict expected from the new env)
        rcomps = info.get("reward_components", None)
        if isinstance(rcomps, dict):
            for k, v in rcomps.items():
                try:
                    self._ensure_component(k)
                    self.r_components[k].append(float(v))
                except Exception:
                    pass
        else:
            # Backwards compatibility: check old flat keys if present
            for oldk in ("reward_climb", "reward_dist", "reward_shaping", "reward_ctrl_pen", "reward_contact"):
                if oldk in info:
                    try:
                        name = oldk.replace("reward_", "")
                        self._ensure_component(name)
                        self.r_components[name].append(float(info[oldk]))
                    except Exception:
                        pass

        # success detection
        success_flag = None
        for k in ("is_success", "success", "succeeded"):
            if k in info:
                try:
                    success_flag = bool(info[k])
                    break
                except Exception:
                    success_flag = None

        # fallback via distances and env threshold if not provided
        if success_flag is None and (dr is not None):
            try:
                envs = getattr(self.training_env, "envs", None)
                if envs and len(envs) > 0:
                    env0 = envs[0]
                    # Try RewardConfig on env (new API uses RewardConfig dataclass)
                    thresh = None
                    if hasattr(env0, "reward_config"):
                        rc = getattr(env0, "reward_config")
                        # rc may be dataclass -> convert to dict or try attribute
                        try:
                            if isinstance(rc, dict):
                                thresh = rc.get("close_thresh", None)
                            else:
                                thresh = getattr(rc, "close_thresh", None)
                        except Exception:
                            thresh = None
                    # fallback older attr name for compatibility
                    if thresh is None and hasattr(env0, "reward_params"):
                        rp = getattr(env0, "reward_params")
                        if isinstance(rp, dict):
                            thresh = rp.get("close_thresh", None)
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


def make_env(xml: str, reward_kwargs: Optional[Dict] = None, use_actuators: bool = True):
    """
    Returns a thunk that creates a Monitor-wrapped ClimbEnv with an optional RewardConfig override.
    reward_kwargs: dict of RewardConfig fields to override, e.g. {'climb_scale': 50.0}
    """
    def _thunk():
        rc = RewardConfig()
        if reward_kwargs:
            # apply overrides
            for k, v in reward_kwargs.items():
                if hasattr(rc, k):
                    setattr(rc, k, v)
                else:
                    raise KeyError(f"Unknown RewardConfig field: {k}")
        env = ClimbEnv(xml_path=xml, render_mode='human', use_actuators=use_actuators, reward_config=rc)
        return Monitor(env)
    return _thunk

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from collections import deque
import os
import numpy as np

class BestModelOnSuccessRate(BaseCallback):
    """
    Save the model when the rolling success rate (from env.info dicts) improves.

    - save_path: directory where best model files will be written.
    - window: size of rolling window (in samples of info entries, not episodes) used to compute the mean.
    - metric_key: the info key to use to compute success (default 'is_success'). If that key
                  isn't present, falls back to distance-based heuristic if available.
    - min_steps: minimum number of info samples before the first save attempt (avoid noisy early saves).
    """
    def __init__(self, save_path: str, window: int = 200, metric_key: str = "is_success",
                 min_steps: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = os.path.expanduser(save_path)
        os.makedirs(self.save_path, exist_ok=True)
        self.window = int(window)
        self.metric_key = metric_key
        self.min_steps = int(min_steps)

        self._deque = deque(maxlen=self.window)
        self.best_mean = -np.inf
        self._n_seen = 0

    def _on_step(self) -> bool:
        # extract infos available in locals (SB3 provides "infos" during on_step)
        infos = None
        if isinstance(self.locals, dict):
            infos = self.locals.get("infos", None)

        if infos is None:
            # nothing to update this step
            return True

        # infos can be a list (vec env) or a dict (single env)
        if isinstance(infos, dict):
            infos_iter = [infos]
        else:
            infos_iter = infos

        changed = False
        for info in infos_iter:
            if not isinstance(info, dict):
                continue
            val = None
            # prefer explicit success flag
            if self.metric_key in info:
                try:
                    val = 1.0 if bool(info[self.metric_key]) else 0.0
                except Exception:
                    val = None

            # fallback heuristic: if d_right is present and env exposes threshold, consider success if d_right < thresh
            if val is None and "d_right" in info:
                try:
                    dr = float(info.get("d_right", np.inf))
                    # try env-provided threshold if available
                    thresh = None
                    envs = getattr(self.training_env, "envs", None)
                    if envs and len(envs) > 0:
                        env0 = envs[0]
                        # prefer reward_config dataclass if present
                        if hasattr(env0, "reward_config"):
                            rc = getattr(env0, "reward_config")
                            thresh = getattr(rc, "close_thresh", None)
                        elif hasattr(env0, "reward_params"):
                            rp = getattr(env0, "reward_params")
                            if isinstance(rp, dict):
                                thresh = rp.get("close_thresh", None)
                    if thresh is None:
                        thresh = 0.03
                    val = 1.0 if (dr < float(thresh)) else 0.0
                except Exception:
                    val = 0.0

            if val is not None:
                self._deque.append(float(val))
                self._n_seen += 1
                changed = True

        # if we haven't collected enough samples yet, don't try to save
        if self._n_seen < self.min_steps or len(self._deque) == 0:
            return True

        mean_success = float(np.mean(self._deque))
        if mean_success > self.best_mean:
            # Save model
            self.best_mean = mean_success
            filename = os.path.join(self.save_path, f"best_model_success_{mean_success:.4f}.zip")
            # self.model is available and is the RL model
            try:
                self.model.save(filename)
                if self.verbose:
                    print(f"[BestModelOnSuccessRate] New best success rate {mean_success:.4f} -> saved {filename}")
            except Exception as e:
                if self.verbose:
                    print(f"[BestModelOnSuccessRate] Failed to save model: {e}")

        return True



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=str,
                   default="/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene copy.xml")
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--save", type=str, default="models/ppo_climbbot.zip")
    # reward tuning overrides (examples)
    p.add_argument("--climb_scale", type=float, default=None)
    p.add_argument("--climb_success_bonus", type=float, default=None)
    p.add_argument("--left_weight", type=float, default=None)
    p.add_argument("--ctrl_penalty_scale", type=float, default=None)
    p.add_argument("--use_actuators", action="store_true", help="Run with actuators enabled (default False in many test setups).")
    p.add_argument("--tensorboard", type=str, default="./ppo_climb_tensorboard")
    p.add_argument("--window", type=int, default=200, help="window size for callback logging")
    return p.parse_args()


def main():
    args = parse_args()

    # build reward override dict from provided CLI args
    reward_overrides = {}
    if args.climb_scale is not None:
        reward_overrides["climb_scale"] = float(args.climb_scale)
    if args.climb_success_bonus is not None:
        reward_overrides["climb_success_bonus"] = float(args.climb_success_bonus)
    if args.left_weight is not None:
        reward_overrides["left_weight"] = float(args.left_weight)
    if args.ctrl_penalty_scale is not None:
        reward_overrides["ctrl_penalty_scale"] = float(args.ctrl_penalty_scale)

    # single-process vectorized env
    env = DummyVecEnv([make_env(args.xml, reward_kwargs=reward_overrides, use_actuators=args.use_actuators)])

    # PPO hyperparams: tweak as you like
    model = PPO("MlpPolicy", env,
                verbose=2,
                tensorboard_log=args.tensorboard,
                learning_rate=3e-4,
                # you can change n_steps, batch_size, n_epochs for sample efficiency
                )
    
    callback = SB3EnvInfoLogger(verbose=1, window=args.window)
    from stable_baselines3.common.callbacks import CallbackList

    best_saver = BestModelOnSuccessRate(save_path="./best_models_by_success", window=args.window, min_steps=1000)
    callback_list = CallbackList([callback, best_saver])

    model.learn(total_timesteps=int(args.timesteps), callback=callback_list)
    model.save(args.save)
    print("Saved model to", args.save)

    # optionally save the reward config used to train for record-keeping
    try:
        # retrieve underlying env to inspect reward_config
        envs = getattr(env, "envs", None)
        if envs and len(envs) > 0:
            rc = getattr(envs[0].env, "reward_config", None)  # Monitor(env) -> .env is underlying env
            if rc is not None:
                fname = args.save + ".reward_config.json"
                with open(fname, "w") as f:
                    json.dump(asdict(rc) if hasattr(rc, "__dict__") or hasattr(rc, "__dataclass_fields__") else rc, f, indent=2)
                print("Saved reward config to", fname)
    except Exception:
        pass


if __name__ == "__main__":
    main()
