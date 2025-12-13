# climb_env.py
"""
ClimbEnv using numeric-Jacobian dual-arm IK (damped pseudoinverse) with joint/ctrl clamping.
Drop-in replacement for your previous climb_env.py.
"""

import os
import time
from typing import Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco_viewer

# -------------------- helpers (transforms) --------------------
def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),   1-2*(x*x+z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x+y*y)]
    ], dtype=float)
    return R

def world_to_local(point_world, base_pos_world, base_quat):
    R = quat_to_mat(base_quat)
    return (R.T @ (point_world - base_pos_world))

def mat3_to_quat(m):
    m = np.asarray(m).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)

def site_quat_from_xmat(data, site_id):
    flat = data.site_xmat[site_id]
    R = np.asarray(flat).reshape(3, 3)
    return mat3_to_quat(R)

# -------------------- numeric Jacobian IK utils --------------------
def forward_site_pos(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    mujoco.mj_forward(model, data)
    return data.site_xpos[site_id].copy()

def numeric_jacobian(model: mujoco.MjModel, data: mujoco.MjData,
                     site_id: int, qpos_indices: List[int], eps: float = 1e-6) -> np.ndarray:
    base_qpos = data.qpos.copy()
    f0 = forward_site_pos(model, data, site_id)
    n = len(qpos_indices)
    J = np.zeros((3, n), dtype=float)
    for i, qi in enumerate(qpos_indices):
        dq = np.zeros_like(base_qpos)
        dq[qi] = eps
        data.qpos[:] = base_qpos + dq
        mujoco.mj_forward(model, data)
        f1 = data.site_xpos[site_id].copy()
        J[:, i] = (f1 - f0) / eps
    data.qpos[:] = base_qpos
    mujoco.mj_forward(model, data)
    return J

def damped_pinv(J: np.ndarray, damping: float) -> np.ndarray:
    JTJ = J.T @ J
    n = JTJ.shape[0]
    A = JTJ + (damping**2) * np.eye(n)
    try:
        pinv = np.linalg.solve(A, J.T)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(A) @ J.T
    return pinv  # shape (n, 3)

def clamp_delta_by_limits(qpos: np.ndarray, delta_q: np.ndarray, qpos_addrs: List[int],
                          limits: List[Tuple[float, float]]) -> np.ndarray:
    clipped = delta_q.copy()
    for i, addr in enumerate(qpos_addrs):
        qmin, qmax = limits[i]
        if not (np.isfinite(qmin) or np.isfinite(qmax)):
            continue
        cur = qpos[addr]
        proposed = cur + clipped[i]
        clipped[i] = float(np.clip(proposed, qmin, qmax) - cur)
    return clipped

def clamp_qpos_by_limits(qpos: np.ndarray, qpos_addrs: List[int], limits: List[Tuple[float, float]]) -> np.ndarray:
    qpos_new = qpos.copy()
    for addr, (qmin, qmax) in zip(qpos_addrs, limits):
        if np.isfinite(qmin) or np.isfinite(qmax):
            qpos_new[addr] = float(np.clip(qpos_new[addr], qmin, qmax))
    return qpos_new

# -------------------- environment --------------------
class ClimbEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "mujoco"], "render_fps": 60}

    def __init__(self, xml_path: str,
                 render_mode: str = "None",
                 action_scale: float = 1.0,
                 ik_step_scale: float = 1.0,
                 ik_damping: float = 0.5,
                 ik_fd_eps: float = 1e-6,
                 ik_update_every: int = 3,
                 use_actuators: bool = True,
                 max_episode_time: int = 10):
        assert os.path.exists(xml_path), f"{xml_path} not found"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.viewer = None
        self.render_mode = render_mode
        self.action_scale = action_scale

        self.dt = self.model.opt.timestep
        self.max_episode_steps = int(max_episode_time / self.dt)
        print(f"Max episode steps: {self.max_episode_steps}")

        # IK params (sensible defaults)
        self.ik_step_scale = ik_step_scale
        self.ik_damping = ik_damping if ik_damping is not None else 1e-2
        self.ik_fd_eps = ik_fd_eps if ik_fd_eps is not None else 1e-5
        self.ik_update_every = ik_update_every
        self.use_actuators = use_actuators
        

        # site ids (require these names exist)
        self.base_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "base_site")
        self.r_grip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_grip_site")
        self.l_grip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "l_grip_site")
        self.hold1_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hold_1_site")
        self.hold2_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hold_2_site")
        
        # optional contact reward target (example names; must exist in XML)
        try:
            self.r_grip_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "assembly_12_collision_1_2")
            self.l_grip_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "assembly_11_collision_1_2")
        except Exception:
            self.r_grip_geom_id = -1
            self.l_grip_geom_id = -1


        # joint lists (change to the real joint names in your XML)
        self.right_joint_names = ["r1", "r2", "r3_1"]
        self.left_joint_names = ["l1", "l2", "l3_1"]
        # resolve joint ids and qpos addresses
        self.right_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.right_joint_names]
        self.left_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.left_joint_names]

        # qpos indices for numeric jacobian (index into data.qpos)
        self.right_qpos_indices = [int(self.model.jnt_qposadr[jid]) for jid in self.right_joint_ids]
        self.left_qpos_indices = [int(self.model.jnt_qposadr[jid]) for jid in self.left_joint_ids]

        # qpos addrs + limits for clamping
        self.right_qpos_addrs, self.right_limits = self._get_addrs_and_limits(self.right_joint_ids)
        self.left_qpos_addrs, self.left_limits = self._get_addrs_and_limits(self.left_joint_ids)

        # map actuators (optional) by naming convention "<joint>_ctrl"
        self.right_actuator_ids = self._find_actuator_ids_for_joint_names(self.right_joint_names) if self.use_actuators else None
        self.left_actuator_ids  = self._find_actuator_ids_for_joint_names(self.left_joint_names) if self.use_actuators else None
        if self.use_actuators and (self.right_actuator_ids is None or self.left_actuator_ids is None):
            print("Actuator mapping not found for all joints; falling back to kinematic qpos updates.")
            self.use_actuators = False

        # sanity asserts: fail early if sites/joints missing
        for name, sid in [
            ("base_site", self.base_site_id),
            ("r_grip_site", self.r_grip_site_id),
            ("l_grip_site", self.l_grip_site_id),
            ("hold_1_site", self.hold1_site_id),
            ("hold_2_site", self.hold2_site_id),
        ]:
            assert sid != -1, f"Site {name} not found in XML"

        for jn, jid in zip(self.right_joint_names + self.left_joint_names,
                           self.right_joint_ids + self.left_joint_ids):
            assert jid != -1, f"Joint {jn} not found in XML"


        # action space: 6 deltas (r dx,dy,dz, l dx,dy,dz) in base frame [-1,1] scaled by action_scale
        act_high = np.ones(6, dtype=np.float32)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)

        # build observation_space dynamically from a sample obs to avoid mismatches
        sample_obs = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32)


        # training bookkeeping for shaping reward
        self.prev_d_right = None
        self.prev_d_left  = None


        # reward params
        # reward params
        self.reward_params = {
            "right_weight": 1.0,
            "left_weight": 0.0,
            "close_bonus": 1.0,
            "close_thresh": 0.03,
            "distance_penalty_scale": 1.0,

            # climb-specific params (new)
            "climb_scale": 0.0,            # reward per meter of upward base movement (scale to taste)
            "climb_success_delta": 0.1,    # success if base rises this much (meters)
            "climb_success_bonus": 200.0,   # large sparse success bonus
            "ctrl_penalty_scale": 0.0       # optional control (effort) penalty; set >0 to penalize big controls
        }


        # runtime counters
        self._step_count = 0
        self._physics_step_counter = 0

        mujoco.mj_forward(self.model, self.data)

    # -------------------- utility helpers --------------------
    def _get_addrs_and_limits(self, joint_ids: List[int]):
        addrs = []
        limits = []
        for jid in joint_ids:
            addr = int(self.model.jnt_qposadr[jid])
            addrs.append(addr)
            limited = bool(self.model.jnt_limited[jid]) if hasattr(self.model, "jnt_limited") else True
            if limited:
                limits.append((float(self.model.jnt_range[jid, 0]), float(self.model.jnt_range[jid, 1])))
            else:
                limits.append((-np.inf, np.inf))
        return addrs, limits

    def _find_actuator_ids_for_joint_names(self, joint_names: List[str]):
        act_ids = []
        for jn in joint_names:
            aname = jn + "_ctrl"
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid == -1:
                # actuator not found, return None to indicate not using actuators
                return None
            act_ids.append(int(aid))
        return act_ids

    # -------------------- reward and observation --------------------
    def reward_fn(self, d_right: float, d_left: float) -> float:
        rp = self.reward_params
        r = - (rp["right_weight"] * d_right + rp["left_weight"] * d_left) * rp["distance_penalty_scale"]
        if d_right < rp["close_thresh"]:
            r += rp["close_bonus"]
        # if d_left < rp["close_thresh"]:
        #     r += rp["close_bonus"]
        return float(r)

    def _get_obs(self):
        base_pos = self.data.site_xpos[self.base_site_id].copy()
        base_quat = site_quat_from_xmat(self.data, self.base_site_id)
        r_pos_w = self.data.site_xpos[self.r_grip_site_id].copy()
        l_pos_w = self.data.site_xpos[self.l_grip_site_id].copy()
        r_quat_w = site_quat_from_xmat(self.data, self.r_grip_site_id)
        l_quat_w = site_quat_from_xmat(self.data, self.l_grip_site_id)
        l_target_w = self.data.site_xpos[self.hold1_site_id].copy()
        r_target_w = self.data.site_xpos[self.hold2_site_id].copy()

        r_pos_b = world_to_local(r_pos_w, base_pos, base_quat)
        l_pos_b = world_to_local(l_pos_w, base_pos, base_quat)
        r_target_b = world_to_local(r_target_w, base_pos, base_quat)
        l_target_b = world_to_local(l_target_w, base_pos, base_quat)

        # quaternion transform for sites
        def quat_inv(q):
            w,x,y,z = q
            return np.array([w, -x, -y, -z])
        def quat_mul(a,b):
            aw,ax,ay,az = a
            bw,bx,by,bz = b
            return np.array([
                aw*bw - ax*bx - ay*by - az*bz,
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw
            ])
        base_q_inv = quat_inv(base_quat)
        r_quat_b = quat_mul(base_q_inv, r_quat_w)
        l_quat_b = quat_mul(base_q_inv, l_quat_w)

        obs = np.concatenate([
            base_pos.flatten(), base_quat.flatten(),
            r_pos_b.flatten(), r_quat_b.flatten(),
            l_pos_b.flatten(), l_quat_b.flatten(),
            r_target_b.flatten(), l_target_b.flatten()
        ]).astype(np.float32)
        return obs

    def step(self, action: np.ndarray):
        """
        Replaces the original step(). Includes:
        - action clipping & base->world transform (same as before)
        - numeric IK solve (same as before)
        - per-joint delta clipping
        - exponential smoothing of qpos target
        - actuator ctrl smoothing & ramping OR kinematic interpolation (depending on self.use_actuators)
        """
        self._step_count += 1

        # -------------------- tuning knobs (tweak these) --------------------
        # smaller = slower but safer / smoother
        # you were previously using larger action_scale/ik_step_scale; reduce if jumpy
        # (you may want to set action_scale when constructing the env instead)
        max_joint_delta = 0.05       # radians per joint per env.step (cap)
        n_interp = 8                 # micro physics steps to interpolate qpos for kinematic path
        interp_alpha = 0.25          # exponential smoothing factor for qpos target (0=no smoothing, 1=immediate)
        ctrl_interp_steps = 12       # ramping steps for actuator ctrl path
        ctrl_alpha = 0.25            # exponential smoothing for ctrl target
        # --------------------------------------------------------------------

        # clip & scale action (end-effector delta in base frame)
        action = np.clip(action, self.action_space.low, self.action_space.high) * self.action_scale
        dr = action[:3].copy()
        dl = action[3:6].copy()

        # TEMP: if you want to freeze an arm set its delta to zero (left enabled by default)
        # dr = np.zeros_like(dr)   # uncomment to freeze right
        dl = np.zeros_like(dl)   # uncomment to freeze left

        # transform deltas from base -> world
        base_pos = self.data.site_xpos[self.base_site_id].copy()
        base_quat = site_quat_from_xmat(self.data, self.base_site_id)
        Rb = quat_to_mat(base_quat)
        dr_world = Rb @ dr
        dl_world = Rb @ dl

        # desired gripper world positions
        r_cur_w = self.data.site_xpos[self.r_grip_site_id].copy()
        l_cur_w = self.data.site_xpos[self.l_grip_site_id].copy()
        r_des_w = r_cur_w + dr_world
        l_des_w = l_cur_w + dl_world

        # ---- numeric Jacobians (current base state) ----
        Jr = numeric_jacobian(self.model, self.data, self.r_grip_site_id, self.right_qpos_indices, eps=self.ik_fd_eps)
        Jl = numeric_jacobian(self.model, self.data, self.l_grip_site_id, self.left_qpos_indices, eps=self.ik_fd_eps)

        # position errors (end-effector deltas in world frame)
        err_r = (r_des_w - r_cur_w).reshape(3,)
        err_l = (l_des_w - l_cur_w).reshape(3,)

        # combined block-diagonal Jacobian
        J_top = np.hstack([Jr, np.zeros((3, len(self.left_qpos_indices)))])
        J_bot = np.hstack([np.zeros((3, len(self.right_qpos_indices))), Jl])
        J_big = np.vstack([J_top, J_bot])   # shape (6, nr+nl)
        err_big = np.concatenate([err_r, err_l])  # shape (6,)

        # damped pseudo-inverse solve
        pinv_big = damped_pinv(J_big, self.ik_damping)
        delta_big = (pinv_big @ err_big).squeeze()

        # split back into right / left deltas
        nr = len(self.right_qpos_indices)
        delta_r = delta_big[:nr]
        delta_l = delta_big[nr:]

        # -------------------- safety + smoothing pipeline --------------------
        # 1) per-joint hard cap (pre-scale for safety)
        delta_r = np.clip(delta_r, -max_joint_delta, max_joint_delta)
        delta_l = np.clip(delta_l, -max_joint_delta, max_joint_delta)

        # 2) original scaling by ik_step_scale (kept but now safer)
        delta_r = self.ik_step_scale * delta_r
        delta_l = self.ik_step_scale * delta_l

        # 3) enforce joint limits on the deltas (prevents proposing out-of-range qpos)
        delta_r = clamp_delta_by_limits(self.data.qpos, delta_r, self.right_qpos_addrs, self.right_limits)
        delta_l = clamp_delta_by_limits(self.data.qpos, delta_l, self.left_qpos_addrs, self.left_limits)

        # 4) propose qpos target
        qpos_prop = self.data.qpos.copy()
        for i, addr in enumerate(self.right_qpos_addrs):
            qpos_prop[addr] += float(delta_r[i])
        for i, addr in enumerate(self.left_qpos_addrs):
            qpos_prop[addr] += float(delta_l[i])

        # 5) final clamp qpos to joint limits
        qpos_prop = clamp_qpos_by_limits(qpos_prop, self.right_qpos_addrs, self.right_limits)
        qpos_prop = clamp_qpos_by_limits(qpos_prop, self.left_qpos_addrs, self.left_limits)

        # 6) exponential smoothing of target to prevent spikes
        if not hasattr(self, "_last_qpos_target"):
            self._last_qpos_target = self.data.qpos.copy()
        qpos_smoothed_target = interp_alpha * qpos_prop + (1.0 - interp_alpha) * self._last_qpos_target
        self._last_qpos_target = qpos_smoothed_target.copy()

        # -------------------- apply either via actuators OR kinematic interpolation --------------------
        if self.use_actuators and (self.right_actuator_ids is not None and self.left_actuator_ids is not None):
            # Build ctrl_target from qpos_smoothed_target (assumes actuator maps to joint position)
            ctrl_target = self.data.ctrl.copy()
            for aid, addr in zip(self.right_actuator_ids, self.right_qpos_addrs):
                if 0 <= aid < self.model.nu:
                    ctrl_val = float(qpos_smoothed_target[addr])
                    if self.model.nu and hasattr(self.model, "actuator_ctrlrange"):
                        low, high = float(self.model.actuator_ctrlrange[aid,0]), float(self.model.actuator_ctrlrange[aid,1])
                        ctrl_val = float(np.clip(ctrl_val, low, high))
                    ctrl_target[aid] = ctrl_val
            # for aid, addr in zip(self.left_actuator_ids, self.left_qpos_addrs):
            #     if 0 <= aid < self.model.nu:
            #         ctrl_val = float(qpos_smoothed_target[addr])
            #         if self.model.nu and hasattr(self.model, "actuator_ctrlrange"):
            #             low, high = float(self.model.actuator_ctrlrange[aid,0]), float(self.model.actuator_ctrlrange[aid,1])
            #             ctrl_val = float(np.clip(ctrl_val, low, high))
            #         ctrl_target[aid] = ctrl_val

            # smooth ctrl target with EMA
            if not hasattr(self, "_last_ctrl_target"):
                self._last_ctrl_target = self.data.ctrl.copy()
            ctrl_target_smoothed = ctrl_alpha * ctrl_target + (1.0 - ctrl_alpha) * self._last_ctrl_target
            self._last_ctrl_target = ctrl_target_smoothed.copy()

            # Ramp ctrl over several physics steps to let actuators move gradually
            for step_i in range(ctrl_interp_steps):
                t = float(step_i + 1) / float(ctrl_interp_steps)
                ctrl_step = (1.0 - t) * self.data.ctrl + t * ctrl_target_smoothed
                self.data.ctrl[:] = ctrl_step
                mujoco.mj_step(self.model, self.data)

            # ensure final ctrl applied and forward to update site_xpos/xmat
            self.data.ctrl[:] = ctrl_target_smoothed
            mujoco.mj_forward(self.model, self.data)

        else:
            # Kinematic path: interpolate current->target qpos across n_interp micro steps
            qpos_current = self.data.qpos.copy()
            for k in range(n_interp):
                t = float(k + 1) / float(n_interp)
                qpos_step = qpos_current + t * (qpos_smoothed_target - qpos_current)
                self.data.qpos[:] = qpos_step
                mujoco.mj_forward(self.model, self.data)
                mujoco.mj_step(self.model, self.data)

            # finalize at smoothed target
            self.data.qpos[:] = qpos_smoothed_target
            mujoco.mj_forward(self.model, self.data)

        # one more physics step (keeps behavior consistent with your original single-step)
        mujoco.mj_step(self.model, self.data)

        # -------------------- observations & reward (modified for climb) --------------------
        obs = self._get_obs()
        l_target_w = self.data.site_xpos[self.hold1_site_id].copy()
        r_target_w = self.data.site_xpos[self.hold2_site_id].copy()
        r_grip_w = self.data.site_xpos[self.r_grip_site_id].copy()
        l_grip_w = self.data.site_xpos[self.l_grip_site_id].copy()
        d_right = float(np.linalg.norm(r_grip_w - r_target_w))
        d_left  = float(np.linalg.norm(l_grip_w - l_target_w))

        # base dense distance reward (existing behavior)
        base_dist_reward = self.reward_fn(d_right, d_left)

        # ---------- climb shaping (right-arm focused) ----------
        # current base height
        base_pos = self.data.site_xpos[self.base_site_id].copy()
        base_z = float(base_pos[2])

        # positive z gain since last step (only reward upwards progress)
        if hasattr(self, "prev_base_z") and (self.prev_base_z is not None):
            z_gain = max(0.0, base_z - self.prev_base_z)
        else:
            z_gain = 0.0

        climb_reward = float(self.reward_params.get("climb_scale", 50.0) * z_gain)

        # right-arm progress shaping (dense): reward reduction in distance to hold
        # (only use right arm shaping to avoid left noise)
        right_shaping = 0.0
        if self.prev_d_right is not None:
            right_shaping = float(self.prev_d_right - d_right)

        # optional control penalty (useful if actuators used)
        ctrl_pen = 0.0
        if (self.model.nu if hasattr(self.model, "nu") else 0) > 0 and self.reward_params.get("ctrl_penalty_scale", 0.0) > 0.0:
            ctrl_pen = float(self.reward_params["ctrl_penalty_scale"] * np.sum(np.square(self.data.ctrl)))

        # combine rewards:
        # keep distance-based base reward (so previous behaviors still apply),
        # then add climb reward and right-only shaping, subtract control penalty
        reward = float(base_dist_reward + climb_reward + 0.5 * right_shaping - ctrl_pen)

        # success termination: either right gripper close OR base climbed enough
        close_thresh = self.reward_params.get("close_thresh", 0.03)
        terminated = False

        # If gripper is very close to its hold: small success bonus (existing)
        if d_right < close_thresh:
            reward += float(self.reward_params.get("close_bonus", 1.0))
            # (optional) do not force termination on reach alone if you prefer
            # terminated = True

        # -------------------- contact-based reward --------------------
        contact_bonus = 0.0
        if self.r_grip_geom_id != -1:
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                # if either geom in the contact pair matches the target geom
                if c.geom[0] == self.r_grip_geom_id or c.geom[1] == self.r_grip_geom_id:
                    f6 = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    frame = np.array(c.frame).reshape(3, 3)
                    f_world = frame @ f6[:3]
                    contact_bonus += f_world[2]
                    break
        reward += contact_bonus


        # Big success if base_z increased enough from start
        climb_delta = float(self.reward_params.get("climb_success_delta", 0.08))
        if hasattr(self, "_start_base_z") and (base_z >= (self._start_base_z + climb_delta)):
            reward += float(self.reward_params.get("climb_success_bonus", 200.0))
            terminated = True

        # store for next step
        self.prev_d_right = d_right
        self.prev_d_left  = d_left
        self.prev_base_z = base_z


        info = {
            "d_right": d_right,
            "d_left": d_left,
            "reward_total": float(reward),
            "reward_dist": float(base_dist_reward),     # original distance-based reward
            "reward_climb": float(climb_reward),        # added climb reward
            "reward_shaping": float(0.5 * right_shaping),# shaping component
            "reward_ctrl_pen": float(ctrl_pen),
            "is_success": bool(terminated),
            "reward_contact": float(contact_bonus),
        }

        truncated = self._step_count >= self.max_episode_steps
        if truncated:
            print("Episode truncated due to max episode steps.")
            print(r_grip_w, r_target_w, d_right)

        # render if requested
        if self.render_mode in ("human", "mujoco"):
            self.render()
        return obs, reward, terminated, truncated, info

    def euler_to_quat_x(euler_x):
        # convert rotation of angle euler_x about X to quaternion (w,x,y,z)
        # here euler is [rx, ry, rz] but we only need rx = pi in your case
        rx, ry, rz = euler_x
        # Build quaternion from euler using X->Y->Z convention if needed.
        # For your specific euler (3.14,0,0) we can directly return (0,1,0,0) which is pi about X.
        # We'll implement general conversion for safety:
        cx = np.cos(rx/2.0); sx = np.sin(rx/2.0)
        cy = np.cos(ry/2.0); sy = np.sin(ry/2.0)
        cz = np.cos(rz/2.0); sz = np.sin(rz/2.0)
        # quaternion in w,x,y,z (assuming rotations applied X then Y then Z)
        w = cx*cy*cz + sx*sy*sz
        x = sx*cy*cz - cx*sy*sz
        y = cx*sy*cz + sx*cy*sz
        z = cx*cy*sz - sx*sy*cz
        return np.array([w, x, y, z], dtype=np.float64)

    def _find_free_joint_qpos_addr(self):
        # return qpos address for first free joint found, or None
        try:
            # iterate joint ids
            for jid in range(self.model.nj):
                # `jnt_type` contains the integer code for joint type
                if int(self.model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                    addr = int(self.model.jnt_qposadr[jid])
                    return addr
        except Exception:
            pass
        return None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Robust reset that:
        - uses model.key_qpos if available,
        - otherwise sets free-base pose if present,
        - otherwise resets data to model defaults,
        - zeros qvel/ctrl, forwards simulation, and clears smoothing buffers.
        Returns (obs, info) like Gymnasium.
        """
        self._step_count = 0
        if seed is not None:
            np.random.seed(seed)

        # If model has a keyframe initial qpos, use it (best option)
        if getattr(self.model, "nkey", 0) > 0:
            try:
                kp = self.model.key_qpos[0].copy()
                # reset data cleanly to model defaults then set qpos to key state
                mujoco.mj_resetData(self.model, self.data)
                self.data.qpos[:] = kp
            except Exception:
                # fallback to safer path below
                mujoco.mj_resetData(self.model, self.data)
        else:
            # Try to find a free joint (floating base). If found, inject the desired base pose
            free_addr = self._find_free_joint_qpos_addr()
            if free_addr is not None:
                # reset data to model defaults first (ensures body tree is consistent)
                mujoco.mj_resetData(self.model, self.data)

                # desired pose from your XML (adjust if you want different start pose)
                base_pos = np.array([0.0, 0.0, 0.10287843], dtype=np.float64)
                base_euler = np.array([3.14, 0.0, 0.0], dtype=np.float64)  # rx,ry,rz
                base_quat = self.euler_to_quat_x(base_euler)

                # start from model defaults then write base pose
                qpos_init = self.data.qpos.copy()
                qpos_init[free_addr:free_addr+3] = base_pos
                qpos_init[free_addr+3:free_addr+7] = base_quat
                self.data.qpos[:] = qpos_init
            else:
                # No keyframe and no free joint: resetData -> leaves qpos as XML defaults
                mujoco.mj_resetData(self.model, self.data)
                # (optional) if you want to zero only actuated joints instead of all, do it here

        # Zero velocities & controls
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        # Make sure actuator internal buffers won't try to drive from an old value:
        # If you maintain any custom smoothing buffers, initialize/clear them here:
        self.prev_d_right = None
        self.prev_d_left  = None

        # Reset smoothing/last-target buffers so no stale interpolation occurs
        self._last_qpos_target = self.data.qpos.copy()
        self._last_ctrl_target = self.data.ctrl.copy()

        # Forward simulate once to populate site_xpos / site_xmat / etc.
        mujoco.mj_forward(self.model, self.data)
        
        # record start base z for climb shaping
        base_pos = self.data.site_xpos[self.base_site_id].copy()
        self._start_base_z = float(base_pos[2])
        self.prev_base_z = self._start_base_z


        # If a viewer exists, recreate it so camera/view state doesn't show stale pose
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

        # Return obs, info per gymnasium API
        return self._get_obs(), {}

    def render(self):
        if self.render_mode in ("human", "mujoco"):
            if self.viewer is None:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.render()
        elif self.render_mode == "rgb_array":
            if self.viewer is None:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, offscreen=True)
            return self.viewer.read_pixels(depth=False)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# quick smoke test if run as script
if __name__ == "__main__":
    xml = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene.xml"
    env = ClimbEnv(xml, render_mode="human", use_actuators=False)
    obs, _ = env.reset()
    print("obs shape:", obs.shape)
    a = np.zeros(6, dtype=np.float32)
    obs, r, t, tr, info = env.step(a)
    print("step -> reward, info:", r, info)
