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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from collections import deque
import numpy as np
import os

from climb_env_3 import ClimbEnv


class SB3EnvInfoLogger(BaseCallback):
    """
    Callback for logging environment info dict values (d_right, d_left, success,
    and reward components) to TensorBoard via SB3 logger.

    Expects env.info to contain keys:
      'd_right','d_left','reward_total','reward_dist','reward_climb',
      'reward_shaping','reward_ctrl_pen','is_success'
    """

    def __init__(self, verbose=0, window: int = 100):
        super().__init__(verbose)
        self.window = int(window)
        # distance / success windows
        self.d_rights = deque(maxlen=self.window)
        self.d_lefts = deque(maxlen=self.window)
        self.successes = deque(maxlen=self.window)
        # reward component windows
        self.r_totals = deque(maxlen=self.window)
        self.r_climbs = deque(maxlen=self.window)
        self.r_dists = deque(maxlen=self.window)
        self.r_shapings = deque(maxlen=self.window)
        self.r_ctrls = deque(maxlen=self.window)

    def _on_step(self) -> bool:
        infos = None
        if isinstance(self.locals, dict):
            infos = self.locals.get("infos", None)

        if infos is None:
            # fallback: try to inspect underlying envs (works for DummyVecEnv)
            try:
                envs = getattr(self.training_env, "envs", None)
                if envs:
                    for e in envs:
                        last_info = getattr(e, "last_info", None)
                        if isinstance(last_info, dict):
                            self._process_info(last_info)
            except Exception:
                pass
        else:
            # infos is typically a list (one entry per env)
            if isinstance(infos, (list, tuple)):
                for info in infos:
                    if isinstance(info, dict):
                        self._process_info(info)
            elif isinstance(infos, dict):
                # single-env case
                self._process_info(infos)

        # write aggregated stats to TensorBoard
        if len(self.d_rights) > 0:
            self.logger.record("env/d_right_mean", float(np.mean(self.d_rights)))
            self.logger.record("env/d_left_mean",  float(np.mean(self.d_lefts)))
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

    def _process_info(self, info: dict):
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
        rt = info.get("reward_total", None)
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

        # success detection
        success_flag = None
        for k in ("is_success", "success", "succeeded"):
            if k in info:
                try:
                    success_flag = bool(info[k])
                    break
                except Exception:
                    success_flag = None

        # fallback via distances and env threshold
        if success_flag is None and (dr is not None):
            try:
                envs = getattr(self.training_env, "envs", None)
                if envs and len(envs) > 0:
                    env0 = envs[0]
                    thresh = None
                    if hasattr(env0, "reward_params"):
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


def make_env(xml):
    def _thunk():
        env = ClimbEnv(xml_path=xml, render_mode=None, use_actuators=True)
        # it's useful to wrap with Monitor to produce episode-level stats
        return Monitor(env)
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene.xml")
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--save", type=str, default="ppo_climbbot.zip")
    args = parser.parse_args()

    # single-process vectorized env
    env = DummyVecEnv([make_env(args.xml)])
    eval_env = DummyVecEnv([make_env(args.xml)])
    # Save best model and periodic checkpoints
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval_logs/",
        eval_freq=50000,          # evaluate every 10k steps
        n_eval_episodes=3,
        deterministic=True,
        render=True
    )



    model = PPO("MlpPolicy", env, verbose=2, tensorboard_log="./ppo_climb_tensorboard", learning_rate=10e-4) 
    # model = PPO.load("/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/logs/best_model/best_model.zip", env=env)
    # model = PPO.load("/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/models/ppo_climbbot_right9.zip", env=env, 
    # custom_objects={"ent_coef": 0.01})
    callback = [SB3EnvInfoLogger(verbose=1, window=200),
            eval_callback]

    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.save)
    print("Saved model to", args.save)


if __name__ == "__main__":
    main()

# """
# train.py

# Train a policy to reach the two handholds using PPO (continuous actions).
# Requires: stable-baselines3, gymnasium, mujoco (python bindings), torch.

# Usage:
#     python train.py --xml scene.xml --timesteps 200000
# """

# import argparse
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from climb_env import ClimbEnv
# import numpy as np
# import os




# # put this in train.py (above main) or in a helper module
# from stable_baselines3.common.callbacks import BaseCallback
# from collections import deque
# import numpy as np

# class SB3EnvInfoLogger(BaseCallback):
#     """
#     Callback for logging environment info dict values (d_right, d_left, success)
#     to TensorBoard via SB3 logger.

#     - Looks for 'infos' in self.locals (typical during on_step).
#     - Falls back to reading env attributes if needed (training_env.envs[...]).

#     window: number of recent samples to average for logging.
#     """

#     def __init__(self, verbose=0, window: int = 100):
#         super().__init__(verbose)
#         self.window = int(window)
#         self.d_rights = deque(maxlen=self.window)
#         self.d_lefts  = deque(maxlen=self.window)
#         self.successes = deque(maxlen=self.window)
#         # when using vectorized env, we may get many info dicts per _on_step

#     def _on_step(self) -> bool:
#         # Try to get 'infos' from the learner locals (most robust during on_step)
#         infos = self.locals.get("infos") if isinstance(self.locals, dict) else None

#         if infos is None:
#             # fallback: try to read last step info from envs (if they stored it)
#             try:
#                 # training_env is a VecEnv. For DummyVecEnv it has .envs list
#                 envs = getattr(self.training_env, "envs", None)
#                 if envs:
#                     # attempt to read 'last_info' or similar attribute from each env
#                     for e in envs:
#                         last_info = getattr(e, "last_info", None)
#                         if isinstance(last_info, dict):
#                             self._process_info(last_info)
#             except Exception:
#                 # if fallback fails, don't crash the callback
#                 pass
#         else:
#             # infos is typically a list (one dict per env)
#             if isinstance(infos, list) or isinstance(infos, tuple):
#                 for info in infos:
#                     if isinstance(info, dict):
#                         self._process_info(info)

#         # periodically write aggregated stats to TensorBoard (every step is OK; SB3 will handle smoothing in the UI)
#         if len(self.d_rights) > 0:
#             self.logger.record("env/d_right_mean", float(np.mean(self.d_rights)))
#             self.logger.record("env/d_left_mean",  float(np.mean(self.d_lefts)))
#             self.logger.record("env/success_rate", float(np.mean(self.successes)))
#         return True

#     def _process_info(self, info: dict):
#         # record distances if present
#         dr = info.get("d_right", None)
#         dl = info.get("d_left", None)

#         if (dr is not None) and (dl is not None):
#             try:
#                 self.d_rights.append(float(dr))
#                 self.d_lefts.append(float(dl))
#             except Exception:
#                 pass

#         # success detection:
#         # 1) prefer explicit flags in info ('is_success', 'success', 'terminated' might indicate success)
#         # 2) fallback: compare distances with env.reward_params['close_thresh'] (if available)
#         success_flag = None
#         for k in ("is_success", "success", "succeeded"):
#             if k in info:
#                 success_flag = bool(info[k])
#                 break
#         if success_flag is None:
#             # some envs include terminal/terminated info — check it
#             if "terminated" in info:
#                 # Note: `terminated` in info sometimes indicates termination reason; treat as not necessarily success
#                 # so only use this if there is also a success flag.
#                 pass

#         if success_flag is None:
#             # fallback heuristic: if distances are present, compare to env's threshold
#             if (dr is not None) and (dl is not None):
#                 try:
#                     # get first underlying env to read reward_params; this assumes DummyVecEnv or similar
#                     envs = getattr(self.training_env, "envs", None)
#                     if envs and len(envs) > 0:
#                         env0 = envs[0]
#                         thresh = None
#                         if hasattr(env0, "reward_params"):
#                             rp = getattr(env0, "reward_params")
#                             thresh = rp.get("close_thresh", None) if isinstance(rp, dict) else None
#                         if thresh is None:
#                             thresh = 0.03
#                         success_flag = (float(dr) < float(thresh)) #and (float(dl) < float(thresh))
#                 except Exception:
#                     success_flag = False

#         # if still None, treat as False
#         if success_flag is None:
#             success_flag = False

#         # append integer success (1/0)
#         try:
#             self.successes.append(1 if bool(success_flag) else 0)
#         except Exception:
#             pass


# def make_env(xml):
#     def _thunk():
#         return ClimbEnv(xml_path=xml, render_mode= None, use_actuators=True)
#     return _thunk

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--xml", type=str, default="/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene.xml")
#     parser.add_argument("--timesteps", type=int, default=2_000_000)
#     parser.add_argument("--save", type=str, default="ppo_climbbot.zip")
#     args = parser.parse_args()

#     env = DummyVecEnv([make_env(args.xml)])

#     model = PPO("MlpPolicy", env, verbose=2, tensorboard_log="./ppo_climb_tensorboard")
#     # model = PPO(
#     # "MlpPolicy", env, verbose=2, tensorboard_log="./ppo_climb_tensorboard",
#     # ent_coef=0.01,  # encourage exploration
#     # learning_rate=3e-4,
#     # n_steps=2048,
#     # batch_size=64,
#     # n_epochs=10,
#     # clip_range=0.2
#     # )

#     # model = PPO.load("/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/models/ppo_left_arm.zip", env=env)

#     callback = SB3EnvInfoLogger(verbose=1, window=200)
#     model.learn(total_timesteps=args.timesteps, callback=callback)
#     model.save(args.save)
#     print("Saved model to", args.save)

# if __name__ == "__main__":
#     main()