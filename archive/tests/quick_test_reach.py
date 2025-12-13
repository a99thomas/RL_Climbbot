# quick_test_reach.py
from envs.climbbot_stage0_env import ClimbBotReachEnv

xml = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/robot_mjcf.xml"
env = ClimbBotReachEnv(xml_path=xml, dt=0.1, sim_substeps=10, goal_interval_seconds=5.0)
obs, info = env.reset()
print("obs shape", obs.shape)
for i in range(600):
    a = env.action_space.sample() * 0.2
    obs, rew, done, truncated, info = env.step(a)
    if i % 50 == 0:
        print(f"step {i} reward {rew:.3f}")
    if done:
        break
env.close()
