# view_model.py
from stable_baselines3 import PPO
from climb_env_5 import ClimbBotEnv
import time

# === Edit these paths ===
MODEL_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/test_5/checkpoints/ppo_climb_1000000_steps.zip"   # your trained model file
XML_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene copy.xml"            # your MuJoCo scene file

# === Load environment & model ===
env = ClimbBotEnv(model_path=XML_PATH, render_mode="human", control_speed=10, render_speed=1)
model = PPO.load(MODEL_PATH)
print("Loaded model:", MODEL_PATH)

# === Run the policy ===
obs, _ = env.reset()
for step in range(10000):  # adjust number of steps as needed
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    # print(f"Step {step}: reward={reward:.3f}, d_right={info['d_right']:.3f}, d_left={info['d_left']:.3f}")
    # if terminated or truncated:
    #     print("Episode done, resetting...")
    #     obs, _ = env.reset()
    # time.sleep(0.02)  # slow down to real-time

env.close()
