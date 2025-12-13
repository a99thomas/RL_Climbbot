'''
Custom Gym environment
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import mujoco_tools as mt
import mujoco_viewer as cr
import kinematics as kinematics
import numpy as np

import time
from collections import deque

# simple aggregator to log rolling averages
TIMINGS = {}
TIMING_HIST = {}
TIMING_HISTORY_LEN = 200

def _record_timing(name, t):
    if name not in TIMINGS:
        TIMINGS[name] = 0.0
        TIMING_HIST[name] = deque(maxlen=TIMING_HISTORY_LEN)
    TIMING_HIST[name].append(t)
    TIMINGS[name] = sum(TIMING_HIST[name]) / len(TIMING_HIST[name])



# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/
class ClimbBotEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(self, model_path="climbbot.xml", control_speed = 1, max_steps=10000, render_mode=None, render_speed = 2):
        super().__init__()
        self.render_mode = render_mode
        self.render_speed = int(render_speed)
        self.frame_skip = render_speed
        self.control_speed = control_speed
        self.env_step_count = 0
        self.frame_step_count = 0
        self.physics_step_count = 0

        # Initialize the WarehouseRobot problem
        self.climbbot = cr.ClimbingRobot(xml_path=model_path, render_mode=render_mode)

        self.action_space = spaces.Box(low=np.array([-0.003]*6, dtype=np.float32),
                                    high=np.array([ 0.003]*6, dtype=np.float32),
                                    shape=(6,), dtype=np.float32)

        big = np.finfo(np.float32).max / 10.0
        self.NUM_HOLDS = 2
        # shape = 3+3+3+3 + 3 + 4 + (NUM_HOLDS*3) + 1 = 3+3+3+3+3+4+6+1 = 26
        obs_dim = 26
        self.observation_space = spaces.Box(
            low=-big, high=big, shape=(obs_dim,), dtype=np.float32
        )


        # --------------------
        # internal & robot params
        # --------------------
        # initialize internal absolute goals for each EE (must be within workspace)
        # Use the current measured ee positions as initial goals.
        self.ee_cmd_r = np.zeros(3, dtype=np.float32)
        self.ee_cmd_l = np.zeros(3, dtype=np.float32)

        # joint limits for clamping (fill with your model limits)
        self.joint_limits_min = np.array([-1.5708, -1.5708, 0.0, -1.5708, -1.5708, 0.0], dtype=np.float32)
        self.joint_limits_max = np.array([ 1.5708,  1.5708, 0.134,  1.5708,  1.5708, 0.134], dtype=np.float32)
        
        self.workspace_min = np.array([-0.5, -0.5, 0.0], dtype=np.float32)
        self.workspace_max = np.array([ 0.8,  0.5, 0.5], dtype=np.float32)


        self.max_steps = max_steps
        self.sim_time = 0

        #Book keeping
        self.armr_ee_pos = np.zeros((3,))
        self.arml_ee_pos = np.zeros((3,))
        self.base_pos, self.base_quat = np.zeros((3,)), np.zeros((4,))
        self.hold_pos1 = np.zeros((3,))
        self.hold_pos2 = np.zeros((3,))

        # ----------
        # IK / control caches & thresholds
        # ----------
        self._last_ik_r = np.zeros(3, dtype=np.float64)   # last successful joint solution for right arm
        self._last_ik_l = np.zeros(3, dtype=np.float64)   # last successful joint solution for left arm
        self._last_target_r = None                        # last endpoint target used for IK (3,)
        self._last_target_l = None

        # threshold to skip IK if target moved less than this (meters)
        self.ik_skip_thresh = 1e-4   # 0.1 mm; increase to 1e-3 for more skipping

        # fallback joint targets if IK fails (initialized later in reset)
        self._fallback_joint_targets = np.zeros(6, dtype=np.float32)



    # Gym required function (and parameters) to reset the environment
    def reset(self, seed: int | None = None, options: dict | None = None):
        # reproducible randomness
        if seed is not None:
            # prefer gym seeding helper for compatibility
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            np.random.seed(seed)

        # reset the underlying simulator (should reset internal sim counters too)
        self.climbbot.reset()

        # zero or set initial commands (choose one strategy; here: set to measured pose)
        obs = self._get_obs()  # updates armr_ee_pos / arml_ee_pos, base_pos, etc.

        # set desired/commanded EE to current measured values (no sudden jump)
        self.ee_initial_r = self.armr_ee_pos.copy()
        self.ee_initial_l = self.arml_ee_pos.copy()

        # If you use absolute ee_cmds, initialize to measured pose:
        self.ee_cmd_r = self.armr_ee_pos.copy()
        self.ee_cmd_l = self.arml_ee_pos.copy()
        # OR if you prefer start from zero-deltas, uncomment:
        # self.ee_cmd_r = np.zeros(3, dtype=np.float32)
        # self.ee_cmd_l = np.zeros(3, dtype=np.float32)

        # reset IK caches so warm-start doesn't use stale targets
        self._last_target_r = None
        self._last_target_l = None
        self._last_ik_r = np.zeros(3, dtype=np.float64)
        self._last_ik_l = np.zeros(3, dtype=np.float64)
        self._fallback_joint_targets = np.zeros(6, dtype=np.float32)

        # reset counters
        self.env_step_count = 0
        self.frame_step_count = 0
        self.physics_step_count = 0
        self.sim_time = 0

        # ensure controller output is safe (no leftover torques/targets)
        try:
            self.climbbot.data.ctrl[:] = 0.0
        except Exception:
            pass

        # recompute observation and return (Gymnasium: obs, info)
        obs = self._get_obs()
        return obs, {}




    def clip_workspace(self, pos):
        return np.clip(pos, self.workspace_min, self.workspace_max)
    
    def _get_obs(self):
        # --- End effector positions ---
        T_r = mt.get_relative_transform(self.climbbot.model, self.climbbot.data,
                                        "body", "robot_base_tilted", "site", "r_grip_site")
        self.armr_ee_pos, _ = mt.tf_to_pos_quat(T_r)

        T_l = mt.get_relative_transform(self.climbbot.model, self.climbbot.data,
                                        "body", "robot_base_tilted", "site", "l_grip_site")
        self.arml_ee_pos, _ = mt.tf_to_pos_quat(T_l)

        # --- Base pose ---
        self.base_pos, base_quat = mt.get_body_pose(
            self.climbbot.model, self.climbbot.data,
            "body", "robot_base_tilted"
        )

        # --- Holds ---
        T_h1 = mt.get_relative_transform(self.climbbot.model, self.climbbot.data,
                                        "body", "robot_base_tilted", "site", "hold_1_site")
        self.hold_pos1, _ = mt.tf_to_pos_quat(T_h1)

        T_h2 = mt.get_relative_transform(self.climbbot.model, self.climbbot.data,
                                        "body", "robot_base_tilted", "site", "hold_2_site")
        self.hold_pos2, _ = mt.tf_to_pos_quat(T_h2)

        holds_pos = np.stack([self.hold_pos1, self.hold_pos2], axis=0).astype(np.float32)

        # --- Time remaining ---
        time_remaining = self.climbbot.physics_dt * (self.max_steps - self.climbbot.physics_steps)

        # --- Flatten into 26-dim vector ---
        obs_vector = np.concatenate([
            self.armr_ee_pos.astype(np.float32),     # (3,)
            self.arml_ee_pos.astype(np.float32),     # (3,)
            self.ee_cmd_r.astype(np.float32),        # (3,)
            self.ee_cmd_l.astype(np.float32),        # (3,)
            self.base_pos.astype(np.float32),        # (3,)
            base_quat.astype(np.float32),            # (4,)
            holds_pos.flatten(),                      # (6,)
            np.array([time_remaining], np.float32),   # (1,)
        ])

        return obs_vector


    def compute_reward(self):
        """
        Balanced reward:
        + progress toward each target hold (negative distance -> larger reward when closer)
        + small per-step survival bonus (encourages staying upright but tiny)
        + contact/grasp bonus when within grasp threshold AND contact force indicates contact
        - penalty for large contact forces (discourage crashing/slamming)
        - penalty for motor effort (encourage smoothness)
        - large negative if touches floor (truncate)
        + large success bonus when both hands are on target holds
        """

        # weights (tune these)
        w_dist = 0.1            # distance-to-hold shaping
        w_grasp = 0.0           # bonus for grasp/contact near holds
        w_height = 1.0          # reward for raising base height
        w_survive = 0.0        # tiny per-step survival
        w_force_pen = -0.0      # penalty for large contact forces (crashes)
        w_energy = -0.0        # penalty on motor commands / joint velocity to encourage smoothness
        success_bonus = 0.0    # large reward for completing the objective
        fall_penalty = 0.0   # large negative when touching the floor

        # thresholds
        grasp_thresh = 0.04     # within 3 cm considered "grasp proximity"
        contact_force_thresh = 0.04  # small threshold to consider contact meaningful (N or normalized)

        # distances (shorter is better) — use holds you defined earlier
        d_right = float(np.linalg.norm(self.armr_ee_pos - self.hold_pos2))
        d_left  = float(np.linalg.norm(self.arml_ee_pos - self.hold_pos1))

        # distance reward (negative distance -> closer gives higher reward)
        # convert to positive reward: larger when closer. We use exponential or inverse to shape.
        # Use an amplitude that drops off smoothly.
        dist_reward_right = d_right   # values in (0,1], steeper near 0
        # dist_reward_left  = 1.0 / (1.0 + d_left*10.0)
        dist_reward_left = 0

        # base height reward (encourage rising)
        base_height = float(self.base_pos[2])

        # contact forces (clamp to reasonable range first)
        try:
            contact_r = float(mt.get_contact_force(self.climbbot.model, self.climbbot.data,
                                                "assembly_12_collision_1_2", 2))
            if contact_r != 0:
                print(contact_r)
        except Exception:
            contact_r = 0.0
        try:
            # contact_l = float(mt.get_contact_force(self.climbbot.model, self.climbbot.data,
            #                                     "assembly_11_collision_1_2", 2))
            contact_l = 0
        except Exception:
            contact_l = 0.0

        # clamp/scale contact forces to a small range so they don't dominate
        contact_r = np.clip(contact_r, -1.0, 1.0)
        contact_l = np.clip(contact_l, -1.0, 1.0)

        # grasp bonus: require both proximity AND contact > small threshold
        grasp_bonus = 0.0
        # if (d_right < grasp_thresh) and (contact_r > contact_force_thresh):
        #     grasp_bonus += 1.0
        # if (d_left < grasp_thresh) and (contact_l > contact_force_thresh):
        #     grasp_bonus += 1.0

        # energy / effort penalty: prefer smaller joint commands
        # estimate by ctrl commands magnitude (assuming climbbot.data.ctrl holds joint targets/commands)
        # if your controller uses different signals, change this.
        try:
            effort = float(np.linalg.norm(self.climbbot.data.ctrl[0:6]))
        except Exception:
            effort = 0.0

        # detect floor contact (fall)
        try:
            floor_force = float(mt.get_contact_force(self.climbbot.model, self.climbbot.data, "floor", 2))
        except Exception:
            floor_force = 0.0

        # compose reward
        reward = 0.0
        reward += w_dist * (dist_reward_right + dist_reward_left)    # closeness reward
        reward += w_height * base_height                             # encourage higher base
        reward += w_survive                                          # tiny survive bonus each step
        reward += w_grasp * grasp_bonus                              # encourage grasping both holds
        reward += w_force_pen * (abs(contact_r) + abs(contact_l))    # penalize large contact forces
        reward += w_energy * effort                                  # penalize large motor commands

        # success condition: both hands grasping => big bonus and terminate as success
        success = (d_right < grasp_thresh) and (d_left < grasp_thresh) and (contact_r > contact_force_thresh) and (contact_l > contact_force_thresh)
        if success:
            reward += success_bonus

        # huge penalty if touches floor — mark truncated in step()
        if floor_force != 0.0:
            reward += fall_penalty

        # Clip reward to avoid exploding values
        reward = float(np.clip(reward, -200.0, 200.0))

        return reward
   
    
    # def compute_reward(self): 
    #     #Reward Weights
    #     d_right_weight = 1.0
    #     d_left_weight = 0.0
    #     base_height_weight = 0.0
    #     contact_right_weight = 2.0
    #     contact_left_weight = 0.0
    #     grasp_weight = 0.0

    #     grasp_threshold = 0.03

    #     # Optional: implement a custom reward function if needed
    #     d_right = -np.linalg.norm(self.armr_ee_pos - self.hold_pos2)
    #     d_left = -np.linalg.norm(self.arml_ee_pos - self.hold_pos1)
    #     base_height = self.base_pos[2]

    #     contact_right = mt.get_contact_force(self.climbbot.model, self.climbbot.data, "assembly_12_collision_1_2", 2)
    #     contact_left = mt.get_contact_force(self.climbbot.model, self.climbbot.data, "assembly_11_collision_1_2", 2)

    #     contact_right = np.clip(contact_right, -.5, .5)
    #     contact_left = np.clip(contact_left, -.5, .5)

    #     reward = d_right_weight * d_right + d_left_weight * d_left + base_height_weight * base_height + \
    #                 contact_right_weight * contact_right + contact_left_weight * contact_left 
    #     return reward

    def render(self, mode="human"):
        # if your mujoco_viewer provides an rgb frame:
        if self.env_step_count % self.render_speed == 0:
            if hasattr(self.climbbot, "render"):
                self.frame_step_count += 1
                return self.climbbot.render()
        return None

    def close(self):
        try:
            self.climbbot.close()
        except Exception:
            pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    # Gym required function (and parameters) to perform an action
    def step(self, action):
        """
        Accepts a flat action vector length-6: [dr_x,dr_y,dr_z, dl_x,dl_y,dl_z]
        """
        # -------------------------
        # parse & clamp action
        # -------------------------
        a = np.asarray(action, dtype=np.float32).ravel()
        assert a.shape[0] == 6, "Expected action shape (6,)"
        dr = a[0:3]
        # dl = a[3:6]
        dl = [0, 0, 0]

        dr = np.clip(dr, -0.003, 0.003)
        dl = np.clip(dl, -0.003, 0.003)
    
        # -------------------------
        # update internal EE goals
        # -------------------------
        # workspace limits as numpy arrays
        min_limits = np.array([0.2, -0.4, -0.5], dtype=np.float32)
        max_limits = np.array([0.8,  0.4,  0.2], dtype=np.float32)

        # apply deltas
        self.ee_cmd_r = self.ee_cmd_r + dr
        self.ee_cmd_l = self.ee_cmd_l + dl

        # clamp into workspace box
        self.ee_cmd_r = np.clip(self.ee_cmd_r, min_limits, max_limits)
        self.ee_cmd_l = np.clip(self.ee_cmd_l, min_limits, max_limits)


        # -------------------------
        # build absolute targets
        # -------------------------
        # If your ee_cmd_* are absolute positions use them directly;
        # if they are deltas relative to some initial pose, uncomment the lines below.

        target_r = self.ee_cmd_r
        target_l = self.ee_cmd_l


        # target_r = self.ee_initial_r + self.ee_cmd_r
        # target_l = self.ee_initial_l + self.ee_cmd_l

        # -------------------------
        # decide whether to run IK (skip if target barely changed)
        # -------------------------
        do_ik_r = True
        do_ik_l = True

        if self._last_target_r is not None:
            if np.linalg.norm(target_r - self._last_target_r) < self.ik_skip_thresh:
                do_ik_r = False

        if self._last_target_l is not None:
            if np.linalg.norm(target_l - self._last_target_l) < self.ik_skip_thresh:
                do_ik_l = False

        # -------------------------
        # IK (warm-started, fallback to last solution)
        # -------------------------
        if do_ik_r:
            ikr = kinematics.ik_right(
                target_r,
                q_init=self._last_ik_r,
                restarts=1,
                use_warm_start=True,
                tol=1e-3,
                max_nfev=500,
                skip_if_close=False,
            )
            if ikr and ikr.get("success", False):
                q_r = np.asarray(ikr["q"], dtype=np.float64)
                self._last_ik_r = q_r.copy()
                self._last_target_r = target_r.copy()
            else:
                q_r = self._last_ik_r.copy()
        else:
            q_r = self._last_ik_r.copy()

        if do_ik_l:
            ikl = kinematics.ik_left(
                target_l,
                q_init=self._last_ik_l,
                restarts=1,
                use_warm_start=True,
                tol=1e-3,
                max_nfev=500,
                skip_if_close=False,
            )
            if ikl and ikl.get("success", False):
                q_l = np.asarray(ikl["q"], dtype=np.float64)
                self._last_ik_l = q_l.copy()
                self._last_target_l = target_l.copy()
            else:
                q_l = self._last_ik_l.copy()
        else:
            q_l = self._last_ik_l.copy()

        # -------------------------
        # full joint target vector, clip to joint limits
        # -------------------------
        joint_targets = np.concatenate([q_r, q_l], axis=0).astype(np.float32)
        joint_targets = np.clip(joint_targets, self.joint_limits_min, self.joint_limits_max)
        self._fallback_joint_targets = joint_targets.copy()

        # send to controller
        self.climbbot.data.ctrl[0:6] = joint_targets

        # -------------------------
        # step physics
        # -------------------------
        for _ in range(self.control_speed):
            self.climbbot.step()
            self.physics_step_count += 1

        # -------------------------
        # observations & reward
        # -------------------------
        obs = self._get_obs()
        reward = self.compute_reward()

        # -------------------------
        # termination / truncation
        # -------------------------
        terminated = False
        truncated = False

        if self.physics_step_count >= self.max_steps:
            terminated = True

        try:
            floor_force = mt.get_contact_force(self.climbbot.model, self.climbbot.data, "floor", 2)
        except Exception:
            floor_force = 0.0

        # if floor_force != 0.0:
        #     truncated = True

        # -------------------------
        # info dict
        # -------------------------
        info = {
            "d_right": float(np.linalg.norm(self.armr_ee_pos - self.hold_pos2)),
            "d_left": float(np.linalg.norm(self.arml_ee_pos - self.hold_pos1)),
            "contact_right": float(mt.get_contact_force(self.climbbot.model, self.climbbot.data,
                                                        "assembly_12_collision_1_2", 2)),
            "contact_left": float(mt.get_contact_force(self.climbbot.model, self.climbbot.data,
                                                    "assembly_11_collision_1_2", 2)),
            "base_height": float(self.base_pos[2]),
            "ee_cmd_r": self.ee_cmd_r.copy(),
            "ee_cmd_l": self.ee_cmd_l.copy(),
        }

        # -------------------------
        # book-keeping & optional rendering (note: rendering in step() is fine for manual debugging,
        # but avoid it during training; prefer calling env.render() from the evaluation loop)
        # -------------------------
        self.env_step_count += 1

        if self.render_mode == "human":
            # If you prefer to only sync the viewer every N env steps, keep this.
            if self.env_step_count % self.render_speed == 0:
                self.render()

        # -------------------------
        # return
        # -------------------------
        return obs, reward, terminated, truncated, info


# For unit testing
if __name__=="__main__":
    env = gym.make('ClimbingRobot', render_mode='human')

    # Use this to check our custom environment
    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    while(True):
        rand_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(rand_action)


        if(terminated):
            obs = env.reset()[0]


