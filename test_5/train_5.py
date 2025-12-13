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

from climb_env_5 import ClimbBotEnv


def make_env(xml):
    def _thunk():
        env = ClimbBotEnv(model_path=xml, control_speed=1, render_mode = "human", render_speed=10)
        # it's useful to wrap with Monitor to produce episode-level stats
        return Monitor(env)
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene copy.xml")
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--save", type=str, default="ppo_climbbot.zip")
    args = parser.parse_args()

    # single-process vectorized env
    env = DummyVecEnv([make_env(args.xml)])


    model = PPO("MlpPolicy", env, verbose=2, tensorboard_log="./ppo_climb_tensorboard", learning_rate=10e-4) 
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    print("Saved model to", args.save)


if __name__ == "__main__":
    main()