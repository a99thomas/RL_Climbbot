# climb_env.py
"""
ClimbEnv using numeric-Jacobian dual-arm IK (damped pseudoinverse) with joint/ctrl clamping.
Drop-in replacement for your previous climb_env.py with clearer reward structure.
"""
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco_viewer

# -------------------- transforms / small helpers --------------------
def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R

def world_to_local(point_world: np.ndarray, base_pos_world: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
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
    # data.site_xmat[site_id] should be a length-9/ or 3x3 matrix in row-major
    try:
        flat = data.site_xmat[site_id]
        R = np.asarray(flat).reshape(3, 3)
        return mat3_to_quat(R)
    except Exception:
        # fallback: if site_xquat exists (some mujoco versions)
        try:
            return data.site_xquat[site_id].copy()
        except Exception:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# -------------------- numeric Jacobian IK utils --------------------
def forward_site_pos(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    mujoco.mj_forward(model, data)
    return data.site_xpos[site_id].copy()

def numeric_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_id: int, qpos_indices: List[int], eps: float = 1e-6) -> np.ndarray:
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
    # J: m x n (m rows, n cols)  -> we want pseudo-inverse shape (n x m)
    # Implementation: (J^T J + lambda^2 I)^{-1} J^T
    JTJ = J.T @ J
    n = JTJ.shape[0]
    A = JTJ + (damping**2) * np.eye(n)
    try:
        pinv = np.linalg.solve(A, J.T)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(A) @ J.T
    return pinv

def clamp_delta_by_limits(qpos: np.ndarray, delta_q: np.ndarray, qpos_addrs: List[int], limits: List[Tuple[float, float]]) -> np.ndarray:
    clipped = delta_q.copy()
    for i, addr in enumerate(qpos_addrs):
        qmin, qmax = limits[i]
        # if neither bound is finite -> no clamping
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

# -------------------- reward configuration --------------------
@dataclass
class RewardConfig:
    # distance-based shaping
    right_weight: float = 1.0
    left_weight: float = 0.0
    distance_penalty_scale: float = 1.0
    close_bonus: float = 0.1
    close_thresh: float = 0.035

    # climb-specific sparse/dense shaping
    climb_scale: float = 0.0
    climb_success_delta: float = 10.0
    climb_success_bonus: float = 0.0

    # shaping / smoothing / control penalty
    shaping_scale: float = 0.0
    ctrl_penalty_scale: float = 0.0

    # contact reward options
    contact_enabled: bool = True
    contact_max_terms: int = 1

    def to_dict(self) -> Dict:
        return asdict(self)

# -------------------- environment --------------------
class ClimbEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "mujoco"], "render_fps": 5}

    def __init__(
        self,
        xml_path: str,
        render_mode: Optional[str] = None,
        action_scale: float = 1.0,
        ik_step_scale: float = 1.0,
        ik_damping: float = 0.5,
        ik_fd_eps: float = 1e-6,
        ik_update_every: int = 3,
        use_actuators: bool = True,
        max_episode_time: int = 10,
        reward_config: Optional[RewardConfig] = None,
    ):
        assert os.path.exists(xml_path), f"{xml_path} not found"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.render_mode = render_mode
        self.action_scale = action_scale
        self.dt = float(self.model.opt.timestep)
        self.max_episode_steps = int(max_episode_time / self.dt)
        print(f"[ClimbEnv] Max episode steps: {self.max_episode_steps}")

        # IK params
        self.ik_step_scale = ik_step_scale
        self.ik_damping = ik_damping if ik_damping is not None else 1e-2
        self.ik_fd_eps = ik_fd_eps if ik_fd_eps is not None else 1e-6
        self.ik_update_every = ik_update_every
        self.use_actuators = use_actuators

        # sites (must exist in XML)
        self.base_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "base_site")
        self.r_grip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_grip_site")
        self.l_grip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "l_grip_site")
        self.hold1_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hold_1_site")
        self.hold2_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hold_2_site")

        # optional contact geoms (example names; if not present -> -1)
        try:
            self.r_grip_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "assembly_12_collision_1_2")
            self.l_grip_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "assembly_11_collision_1_2")
        except Exception:
            self.r_grip_geom_id = -1
            self.l_grip_geom_id = -1

        # joint names - change to match your XML if necessary
        self.right_joint_names = ["r1", "r2", "r3_1"]
        self.left_joint_names = ["l1", "l2", "l3_1"]

        # resolve joint ids and qpos addresses
        self.right_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.right_joint_names]
        self.left_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.left_joint_names]
        self.right_qpos_indices = [int(self.model.jnt_qposadr[jid]) for jid in self.right_joint_ids]
        self.left_qpos_indices = [int(self.model.jnt_qposadr[jid]) for jid in self.left_joint_ids]
        self.right_qpos_addrs, self.right_limits = self._get_addrs_and_limits(self.right_joint_ids)
        self.left_qpos_addrs, self.left_limits = self._get_addrs_and_limits(self.left_joint_ids)

        # actuator mapping by convention "<joint>_ctrl" if using actuators
        self.right_actuator_ids = self._find_actuator_ids_for_joint_names(self.right_joint_names) if use_actuators else None
        self.left_actuator_ids = self._find_actuator_ids_for_joint_names(self.left_joint_names) if use_actuators else None
        if use_actuators and (self.right_actuator_ids is None or self.left_actuator_ids is None):
            print("[ClimbEnv] Actuator mapping not found for all joints; falling back to kinematic updates.")
            self.use_actuators = False

        # sanity check required_sites and joints
        required_sites = {
            "base_site": self.base_site_id,
            "r_grip_site": self.r_grip_site_id,
            "l_grip_site": self.l_grip_site_id,
            "hold_1_site": self.hold1_site_id,
            "hold_2_site": self.hold2_site_id,
        }
        for name, sid in required_sites.items():
            assert sid != -1, f"Site {name} not found in XML"
        for jn, jid in zip(self.right_joint_names + self.left_joint_names, self.right_joint_ids + self.left_joint_ids):
            assert jid != -1, f"Joint {jn} not found in XML"

        # action space: 6 deltas (r dx,dy,dz, l dx,dy,dz) in base frame [-1,1]
        act_high = np.ones(6, dtype=np.float32)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)

        # build observation space dynamically from a sample obs
        mujoco.mj_forward(self.model, self.data)
        sample_obs = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32)

        # reward config (easy to change)
        self.reward_config = reward_config if reward_config is not None else RewardConfig()

        # bookkeeping
        self.prev_d_right = None
        self.prev_d_left = None
        self.prev_base_z = None
        self._start_base_z = None
        self._step_count = 0
        self._physics_step_counter = 0

        # smoothing buffers
        self._last_qpos_target = self.data.qpos.copy()
        self._last_ctrl_target = self.data.ctrl.copy()

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
                return None
            act_ids.append(int(aid))
        return act_ids

    def set_reward_params(self, **kwargs):
        """Update reward parameters by keyword. Example: env.set_reward_params(climb_scale=10.0)"""
        for k, v in kwargs.items():
            if hasattr(self.reward_config, k):
                setattr(self.reward_config, k, v)
            else:
                raise KeyError(f"Unknown reward parameter: {k}")

    # -------------------- reward decomposition (easy to tweak) --------------------
    def _reward_distance(self, d_right: float, d_left: float) -> float:
        """Base dense reward from distances to targets. Lower distance -> higher reward (we use negative penalty)."""
        rc = self.reward_config
        r = - (rc.right_weight * d_right + rc.left_weight * d_left) * rc.distance_penalty_scale
        # close bonus (small fixed bonus to avoid infinite values)
        if d_right < rc.close_thresh:
            r += rc.close_bonus
        return float(r)

    def _reward_climb(self, base_z: float, prev_base_z: Optional[float]) -> float:
        """Dense reward for positive upward base movement. Only rewards upward progress (no negative)."""
        rc = self.reward_config
        if prev_base_z is None:
            return 0.0
        z_gain = max(0.0, base_z - prev_base_z)
        return float(rc.climb_scale * z_gain)

    def _reward_shaping(self, d_right: float, prev_d_right: Optional[float]) -> float:
        """Dense shaping based on reduction in right distance (positive if we get closer)."""
        rc = self.reward_config
        if prev_d_right is None:
            return 0.0
        return float(rc.shaping_scale * max(0.0, (prev_d_right - d_right)))

    def _reward_control_penalty(self) -> float:
        rc = self.reward_config
        try:
            nu = int(self.model.nu) if hasattr(self.model, "nu") else 0
        except Exception:
            nu = 0
        if nu <= 0 or rc.ctrl_penalty_scale <= 0.0:
            return 0.0
        return float(rc.ctrl_penalty_scale * np.sum(np.square(self.data.ctrl)))

    def _reward_contact(self) -> float:
        rc = self.reward_config
        if not rc.contact_enabled or self.r_grip_geom_id == -1:
            return 0.0
        contact_bonus = 0.0
        count = 0
        try:
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                # geometry indices may be stored in different fields; try multiple access patterns
                geom0 = getattr(c, "geom1", None)
                geom1 = getattr(c, "geom2", None)
                if geom0 is None or geom1 is None:
                    # older/newer API sometimes packs them differently
                    try:
                        geom0 = c.geom[0]
                        geom1 = c.geom[1]
                    except Exception:
                        continue
                if int(geom0) == int(self.r_grip_geom_id) or int(geom1) == int(self.r_grip_geom_id):
                    # get contact force - mj_contactForce writes into an array of length 6
                    f6 = np.zeros(6)
                    try:
                        mujoco.mj_contactForce(self.model, self.data, i, f6)
                        # f6[:3] is force in contact frame; try to project to world using contact frame if available
                        # If contact frame isn't accessible, fallback to using normal z-component of f6
                        # We attempt to use c.frame if present (3x3)
                        fz = f6[2]
                        try:
                            frame = np.array(c.frame).reshape(3, 3)
                            f_world = frame @ f6[:3]
                            fz = float(f_world[2])
                        except Exception:
                            pass
                        contact_bonus += float(fz)
                    except Exception:
                        # fallback: add small fixed bonus for contact existence
                        contact_bonus += 0.0
                    count += 1
                    if count >= rc.contact_max_terms:
                        break
        except Exception:
            return 0.0
        return float(contact_bonus)

    def _compose_reward(self, d_right: float, d_left: float, base_z: float) -> (float, Dict):
        rc = self.reward_config
        reward_components = {}
        reward_components['dist'] = self._reward_distance(d_right, d_left)
        reward_components['climb'] = self._reward_climb(base_z, getattr(self, "prev_base_z", None))
        reward_components['shaping'] = self._reward_shaping(d_right, self.prev_d_right)
        reward_components['ctrl_pen'] = self._reward_control_penalty()
        reward_components['contact'] = self._reward_contact()
        # combine (you can change combination rules here)
        total = reward_components['dist'] + reward_components['climb'] + reward_components['shaping'] - reward_components['ctrl_pen'] + reward_components['contact']
        reward_components['total'] = float(total)
        return float(total), reward_components

    # -------------------- observation --------------------
    def _get_obs(self) -> np.ndarray:
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

        # quaternion helpers
        def quat_inv(q):
            w, x, y, z = q
            return np.array([w, -x, -y, -z])
        def quat_mul(a, b):
            aw, ax, ay, az = a
            bw, bx, by, bz = b
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

    # -------------------- core step --------------------
    def step(self, action: np.ndarray):
        """
        Single environment step. Handles:
          - action transforms & scaling
          - numeric IK (damped pseudoinverse)
          - per-joint delta clipping and qpos smoothing
          - actuator ctrl smoothing/ramping or kinematic interpolation
          - reward computation using modular reward config

        Returns: obs, reward, terminated, truncated, info
        """
        self._step_count += 1

        # ---------- tuning knobs (local, easy to tweak) ----------
        max_joint_delta = 0.5  # rad per joint per env.step
        n_interp = 8           # micro interpolation steps (kinematic path)
        interp_alpha = 0.5     # smoothing for qpos target (0=no smoothing, 1=immediate)
        ctrl_interp_steps = 12  # actuator ramp steps
        ctrl_alpha = 0.25      # smoothing for ctrl targets
        # --------------------------------------------------------

        # clip & scale action (end-effector delta in base frame)
        action = np.clip(action, self.action_space.low, self.action_space.high) * self.action_scale
        dr = action[:3].copy()
        dl = action[3:6].copy()

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

        # numeric Jacobians
        Jr = numeric_jacobian(self.model, self.data, self.r_grip_site_id, self.right_qpos_indices, eps=self.ik_fd_eps)
        Jl = numeric_jacobian(self.model, self.data, self.l_grip_site_id, self.left_qpos_indices, eps=self.ik_fd_eps)

        err_r = (r_des_w - r_cur_w).reshape(3,)
        err_l = (l_des_w - l_cur_w).reshape(3,)

        # block-diagonal combined Jacobian (6 x (nr+nl))
        J_top = np.hstack([Jr, np.zeros((3, len(self.left_qpos_indices)))])
        J_bot = np.hstack([np.zeros((3, len(self.right_qpos_indices))), Jl])
        J_big = np.vstack([J_top, J_bot])
        err_big = np.concatenate([err_r, err_l])

        pinv_big = damped_pinv(J_big, self.ik_damping)
        delta_big = (pinv_big @ err_big).squeeze()

        # split into arms
        nr = len(self.right_qpos_indices)
        delta_r = delta_big[:nr] if delta_big.size >= nr else np.zeros(nr)
        delta_l = delta_big[nr:] if delta_big.size > nr else np.zeros(len(self.left_qpos_indices))

        # safety + smoothing pipeline
        delta_r = np.clip(delta_r, -max_joint_delta, max_joint_delta)
        delta_l = np.clip(delta_l, -max_joint_delta, max_joint_delta)
        delta_r = self.ik_step_scale * delta_r
        delta_l = self.ik_step_scale * delta_l
        delta_r = clamp_delta_by_limits(self.data.qpos, delta_r, self.right_qpos_addrs, self.right_limits)
        delta_l = clamp_delta_by_limits(self.data.qpos, delta_l, self.left_qpos_addrs, self.left_limits)

        # propose qpos target
        qpos_prop = self.data.qpos.copy()
        for i, addr in enumerate(self.right_qpos_addrs):
            qpos_prop[addr] += float(delta_r[i])
        for i, addr in enumerate(self.left_qpos_addrs):
            qpos_prop[addr] += float(delta_l[i])
        qpos_prop = clamp_qpos_by_limits(qpos_prop, self.right_qpos_addrs, self.right_limits)
        qpos_prop = clamp_qpos_by_limits(qpos_prop, self.left_qpos_addrs, self.left_limits)

        # exponential smoothing of qpos target
        qpos_smoothed_target = interp_alpha * qpos_prop + (1.0 - interp_alpha) * self._last_qpos_target
        self._last_qpos_target = qpos_smoothed_target.copy()

        # apply via actuators or kinematic interpolation
        if self.use_actuators and (self.right_actuator_ids is not None and self.left_actuator_ids is not None):
            ctrl_target = self.data.ctrl.copy()
            # map qpos target into actuator ctrl range based on assumed mapping
            for aid, addr in zip(self.right_actuator_ids + self.left_actuator_ids, self.right_qpos_addrs + self.left_qpos_addrs):
                if 0 <= aid < self.model.nu:
                    ctrl_val = float(qpos_smoothed_target[addr])
                    if hasattr(self.model, "actuator_ctrlrange"):
                        low, high = float(self.model.actuator_ctrlrange[aid, 0]), float(self.model.actuator_ctrlrange[aid, 1])
                        ctrl_val = float(np.clip(ctrl_val, low, high))
                    ctrl_target[aid] = ctrl_val
            # smooth ctrl target with EMA
            ctrl_target_smoothed = ctrl_alpha * ctrl_target + (1.0 - ctrl_alpha) * self._last_ctrl_target
            self._last_ctrl_target = ctrl_target_smoothed.copy()

            # Ramp ctrl over several physics steps
            for step_i in range(ctrl_interp_steps):
                t = float(step_i + 1) / float(ctrl_interp_steps)
                ctrl_step = (1.0 - t) * self.data.ctrl + t * ctrl_target_smoothed
                self.data.ctrl[:] = ctrl_step
                mujoco.mj_step(self.model, self.data)
            # ensure final ctrl applied and forward to update sites
            self.data.ctrl[:] = ctrl_target_smoothed
            mujoco.mj_forward(self.model, self.data)
        else:
            # Kinematic path: interpolate qpos_current -> qpos_smoothed_target
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
            # one extra physics step to keep single-step semantics
            mujoco.mj_step(self.model, self.data)

        # -------------------- observations & rewards --------------------
        obs = self._get_obs()

        # compute distances
        r_target_w = self.data.site_xpos[self.hold2_site_id].copy()
        l_target_w = self.data.site_xpos[self.hold1_site_id].copy()
        r_grip_w = self.data.site_xpos[self.r_grip_site_id].copy()
        l_grip_w = self.data.site_xpos[self.l_grip_site_id].copy()
        d_right = float(np.linalg.norm(r_grip_w - r_target_w))
        d_left = float(np.linalg.norm(l_grip_w - l_target_w))

        # base z and climb shaping
        base_pos = self.data.site_xpos[self.base_site_id].copy()
        base_z = float(base_pos[2])

        # compute modular reward
        total_reward, reward_components = self._compose_reward(d_right, d_left, base_z)

        # success conditions
        terminated = False
        # small close bonus / optional termination on reach
        if d_right < self.reward_config.close_thresh:
            # we leave termination optional; flag success in info
            pass

        # big success: base rises enough from start
        if self._start_base_z is not None:
            climb_delta = float(self.reward_config.climb_success_delta)
            if base_z >= (self._start_base_z + climb_delta):
                total_reward += float(self.reward_config.climb_success_bonus)
                terminated = True

        # update trackers
        self.prev_d_right = d_right
        self.prev_d_left = d_left
        self.prev_base_z = base_z

        info = {
            "d_right": d_right,
            "d_left": d_left,
            "reward_total": float(total_reward),
            "reward_components": reward_components,
            "is_success": bool(terminated),
        }

        truncated = self._step_count >= self.max_episode_steps
        if truncated:
            print("[ClimbEnv] Episode truncated due to max steps.")

        # render if requested
        if self.render_mode in ("human", "mujoco"):
            try:
                self.render()
            except Exception:
                pass

        return obs, float(total_reward), bool(terminated), bool(truncated), info

    # -------------------- helpers --------------------
    def euler_to_quat_x(self, euler_x):
        """Convert euler (rx,ry,rz) to quaternion (w,x,y,z). Uses X->Y->Z convention."""
        rx, ry, rz = euler_x
        cx = np.cos(rx/2.0); sx = np.sin(rx/2.0)
        cy = np.cos(ry/2.0); sy = np.sin(ry/2.0)
        cz = np.cos(rz/2.0); sz = np.sin(rz/2.0)
        w = cx*cy*cz + sx*sy*sz
        x = sx*cy*cz - cx*sy*sz
        y = cx*sy*cz + sx*cy*sz
        z = cx*cy*sz - sx*cy*cz
        return np.array([w, x, y, z], dtype=np.float64)

    def _find_free_joint_qpos_addr(self):
        try:
            for jid in range(self.model.nj):
                if int(self.model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                    return int(self.model.jnt_qposadr[jid])
        except Exception:
            pass
        return None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset robustly:
          - use model.key_qpos if available
          - otherwise set a free-base pose if present
          - otherwise use XML defaults

        Returns (obs, info) per gymnasium API.
        """
        self._step_count = 0
        if seed is not None:
            np.random.seed(seed)

        try:
            if getattr(self.model, "nkey", 0) > 0:
                kp = self.model.key_qpos[0].copy()
                mujoco.mj_resetData(self.model, self.data)
                self.data.qpos[:] = kp
            else:
                free_addr = self._find_free_joint_qpos_addr()
                if free_addr is not None:
                    mujoco.mj_resetData(self.model, self.data)
                    base_pos = np.array([0.0, 0.0, 0.10287843], dtype=np.float64)
                    base_euler = np.array([3.14, 0.0, 0.0], dtype=np.float64)
                    base_quat = self.euler_to_quat_x(base_euler)
                    qpos_init = self.data.qpos.copy()
                    qpos_init[free_addr:free_addr+3] = base_pos
                    qpos_init[free_addr+3:free_addr+7] = base_quat
                    self.data.qpos[:] = qpos_init
                else:
                    mujoco.mj_resetData(self.model, self.data)
        except Exception:
            mujoco.mj_resetData(self.model, self.data)

        # zero velocities & controls
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        # clear smoothing buffers and trackers
        self.prev_d_right = None
        self.prev_d_left = None
        self._last_qpos_target = self.data.qpos.copy()
        self._last_ctrl_target = self.data.ctrl.copy()

        mujoco.mj_forward(self.model, self.data)

        # record start base z for climb shaping
        base_pos = self.data.site_xpos[self.base_site_id].copy()
        self._start_base_z = float(base_pos[2])
        self.prev_base_z = self._start_base_z

        # recreate viewer if needed
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        self.viewer = None

        return self._get_obs(), {}

    def render(self):
        if self.render_mode in ("human", "mujoco"):
            if self.viewer is None:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.render()
        elif self.render_mode == "rgb_array":
            if self.viewer is None:
                # offscreen viewer
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, offscreen=True)
            return self.viewer.read_pixels(depth=False)

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        self.viewer = None

# quick smoke test
if __name__ == "__main__":
    # Replace this path with your actual local XML file path.
    xml = os.environ.get("CLIMB_XML_PATH", "/path/to/your/scene.xml")
    if not os.path.exists(xml):
        print("Please set CLIMB_XML_PATH environment variable to point to your scene XML, or edit the script.")
    else:
        env = ClimbEnv(xml, render_mode="human", use_actuators=False)
        obs, _ = env.reset()
        print("obs shape:", obs.shape)
        a = np.zeros(6, dtype=np.float32)
        obs, r, t, tr, info = env.step(a)
        print("step -> reward, info:", r, info)
        env.close()
