# view_model.py
from stable_baselines3 import PPO
from climb_env import ClimbEnv
import time

# === Edit these paths ===
MODEL_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/logs/best_model/best_model.zip"   # your trained model file
XML_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene copy.xml"            # your MuJoCo scene file

# === Load environment & model ===
env = ClimbEnv(xml_path=XML_PATH, render_mode="human", use_actuators=True)
model = PPO.load(MODEL_PATH)
print("Loaded model:", MODEL_PATH)

# === Run the policy ===
obs, _ = env.reset()
for step in range(10000):  # adjust number of steps as needed
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: reward={reward:.3f}, d_right={info['d_right']:.3f}, d_left={info['d_left']:.3f}")
    if terminated or truncated:
        print("Episode done, resetting...")
        obs, _ = env.reset()
    time.sleep(0.02)  # slow down to real-time

env.close()
