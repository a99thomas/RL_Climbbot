# play_policy.py — robust playback that falls back to offscreen rendering when needed
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.settrace

from stable_baselines3 import PPO
from envs.climbbot_stage0_env import ClimbBotStage0Env

import mujoco

# --- Update these to your paths ---
XML_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/robot_mjcf.xml"
MODEL_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/checkpoints/ppo_climb_stage0_150000_steps.zip"
# ----------------------------------

# small helper: try to launch high-level viewer if available
def try_launch_viewer(env):
    if hasattr(mujoco, "viewer"):
        try:
            v = mujoco.viewer.launch_passive(env.model, env.data)
            print("Using mujoco.viewer (mujoco-python-viewer).")
            return v
        except Exception as e:
            print("mujoco.viewer present but failed to launch:", e)
    return None

# low-level renderer setup (MjvScene + MjrContext), reused across frames
class LowLevelRenderer:
    def __init__(self, model, width=800, height=600):
        self.model = model
        self.width = int(width)
        self.height = int(height)
        # scene and context
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        # font scale enum varies; use default numeric cast
        try:
            self.ctx = mujoco.MjrContext(model, 150)  # older bindings accept int fontscale
        except Exception:
            # some bindings expose mjFONTSCALE enum; try that
            try:
                self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            except Exception as e:
                raise RuntimeError("Failed to create MjrContext: " + str(e))
        # one camera object reused
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # adjust camera distance/position a bit for a good viewpoint
        try:
            self.cam.distance = 2.5
        except Exception:
            pass

    def render(self, data):
        # update scene & render
        mujoco.mjv_updateScene(self.model, data, mujoco.mjtCatBit.mjCAT_ALL,
                               self.cam, None, self.scene)
        mujoco.mjr_render(self.width, self.height, self.scene, self.ctx)
        # read pixels
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, self.width, self.height, self.ctx)
        # mjr_readPixels returns bottom->top; flip to top->bottom
        rgb = np.flipud(rgb)
        return rgb

def main():
    print("Creating environment...")
    env = ClimbBotStage0Env(xml_path=XML_PATH, dt=0.005, max_episode_seconds=20.0, render_mode="human")

    print("Loading model:", MODEL_PATH)
    model = PPO.load(MODEL_PATH, env=None)

    viewer = try_launch_viewer(env)
    renderer = None
    use_matplotlib = False

    if viewer is None:
        # fallback: try low-level renderer
        try:
            renderer = LowLevelRenderer(env.model, width=960, height=720)
            print("Falling back to low-level MJR rendering.")
            # prepare matplotlib window
            plt.ion()
            fig, ax = plt.subplots(figsize=(8,6))
            im = ax.imshow(np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8))
            ax.axis("off")
            plt.show()
            use_matplotlib = True
        except Exception as e:
            print("Low-level rendering failed:", e)
            print("You can install the high-level viewer with: pip install mujoco-python-viewer")
            # still attempt to run but without rendering
            renderer = None
            use_matplotlib = False

    obs, _ = env.reset()
    episodes_to_run = 5

    for ep in range(episodes_to_run):
        print(f"Episode {ep+1}/{episodes_to_run} starting...")
        done = False
        step = 0
        while not done and step < env.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # render via whichever available backend
            if viewer is not None:
                try:
                    viewer.render()
                except Exception as e:
                    # viewer failed during runtime — fallback to low-level
                    print("viewer.render() failed, switching to low-level render:", e)
                    viewer = None
            if viewer is None and renderer is not None:
                try:
                    frame = renderer.render(env.data)
                    if use_matplotlib:
                        im.set_data(frame)
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                except Exception as e:
                    print("Renderer error:", e)
                    renderer = None

            # small sleep to make motion viewable
            time.sleep(env.dt)

            step += 1
            if terminated or truncated:
                done = True

        print(f"Episode {ep+1} finished. info={info}")
        obs, _ = env.reset()

    print("Run finished. Closing...")
    # close viewer if present
    try:
        if viewer is not None:
            viewer.close()
    except Exception:
        pass
    env.close()
    if use_matplotlib:
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    main()
