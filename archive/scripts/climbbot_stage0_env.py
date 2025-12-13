# climbbot_stage0_env.py
import os
import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

# -------------------- Utility functions --------------------
def mat3_to_quat(mat3: np.ndarray) -> np.ndarray:
    m = mat3
    trace = m[0,0] + m[1,1] + m[2,2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2,1] - m[1,2]) * s
        y = (m[0,2] - m[2,0]) * s
        z = (m[1,0] - m[0,1]) * s
    else:
        if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)

def name2id(model: mujoco.MjModel, objtype: str, name: str) -> int:
    if objtype == 'site':
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if objtype == 'joint':
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if objtype == 'body':
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if objtype == 'actuator':
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    raise ValueError("unsupported objtype")

def clamp_q_targets(model: mujoco.MjModel, joint_ids: list, q_targets: np.ndarray) -> np.ndarray:
    qt = q_targets.copy().astype(np.float64)
    for i, jid in enumerate(joint_ids):
        qpos_adr = int(model.jnt_qposadr[jid])
        limited = bool(model.jnt_limited[jid]) if hasattr(model, "jnt_limited") else True
        if limited:
            qmin = float(model.jnt_range[jid, 0])
            qmax = float(model.jnt_range[jid, 1])
            qt[i] = float(np.clip(qt[i], qmin, qmax))
    return qt

# -------------------- Environment --------------------
class ClimbBotStage0Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1.0 / 0.005}

    def __init__(
        self,
        xml_path: str,
        ee_max_step: float = 0.05,
        dt: float = 0.005,
        max_episode_seconds: float = 20.0,
        success_radius: float = 0.03,
        render_mode: Optional[str] = None,
        debug: bool = False,
        n_settle: int = 6,                 # physics steps after writing ctrl
        max_joint_step: float = 0.12,      # rad (per IK iteration) max
        ik_damping: float = 1e-3,          # base damping
        ik_damping_growth: float = 10.0,   # factor to multiply damping on failure
    ):
        assert os.path.exists(xml_path), f"xml not found: {xml_path}"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # env params
        self.model.opt.timestep = float(dt)
        self.dt = float(dt)
        self.ee_max_step = float(ee_max_step)
        self.max_steps = int(np.ceil(max_episode_seconds / dt))
        self.success_radius = float(success_radius)
        self.render_mode = render_mode
        self.debug = bool(debug)
        self.n_settle = int(n_settle)
        self.max_joint_step = float(max_joint_step)
        self.ik_damping = float(ik_damping)
        self.ik_damping_growth = float(ik_damping_growth)

        # required sites
        try:
            self.site_ids = {
                'r': name2id(self.model, 'site', 'r_grip_site'),
                'l': name2id(self.model, 'site', 'l_grip_site'),
                'base': name2id(self.model, 'site', 'base_site'),
            }
        except Exception as e:
            raise RuntimeError("Missing required site: r_grip_site, l_grip_site, base_site") from e

        # joints & actuators for both arms
        self.joint_names_r = ["r1", "r2", "r3_1"]
        self.joint_names_l = ["l1", "l2", "l3_1"]
        self.joint_ids_r = [name2id(self.model, 'joint', n) for n in self.joint_names_r]
        self.joint_ids_l = [name2id(self.model, 'joint', n) for n in self.joint_names_l]

        self.actuator_names_r = ["r1_ctrl", "r2_ctrl", "r3_1_ctrl"]
        self.actuator_names_l = ["l1_ctrl", "l2_ctrl", "l3_1_ctrl"]
        self.actuator_ids_r = []
        self.actuator_ids_l = []
        for n in self.actuator_names_r:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            if aid != -1:
                self.actuator_ids_r.append(aid)
        for n in self.actuator_names_l:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            if aid != -1:
                self.actuator_ids_l.append(aid)

        # holds discovery
        self.hold_bodies = []
        i = 1
        while True:
            name = f"hold_{i}"
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid == -1:
                break
            self.hold_bodies.append((name, bid))
            i += 1
        if len(self.hold_bodies) == 0:
            raise RuntimeError("No holds found: expect hold_1 ... hold_N bodies in MJCF")

        # observation & action spaces
        obs_dim = 3 + 4 + 3 + 3 + 3 + 3 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # IK scratch
        self._jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        # bookkeeping
        self._step_count = 0
        self._right_target_idx = 0
        self._left_target_idx = 0
        self._rng = np.random.RandomState()
        self._last_site_pos = np.zeros((self.model.nsite, 3), dtype=np.float32)
        self._max_holds = len(self.hold_bodies)

        # safety: store last good state
        self._last_good_qpos = self.data.qpos.copy()
        self._last_good_qvel = self.data.qvel.copy()

        # forward to initialize caches
        mujoco.mj_forward(self.model, self.data)
        self._update_last_site_pos()

    # -------------------- helpers --------------------
    def set_hold_limit(self, n_holds: int):
        self._max_holds = max(1, min(n_holds, len(self.hold_bodies)))

    def _site_quat(self, site_id: int) -> np.ndarray:
        if hasattr(self.data, "site_xquat"):
            return self.data.site_xquat[site_id].copy().astype(np.float32)
        if hasattr(self.data, "site_xmat"):
            mat_flat = self.data.site_xmat[site_id].copy()
            mat3 = np.asarray(mat_flat).reshape(3,3)
            return mat3_to_quat(mat3)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _site_linvel_world(self, site_id: int) -> np.ndarray:
        if hasattr(self.data, "site_xvel"):
            return self.data.site_xvel[site_id].copy().astype(np.float32)
        cur = self.data.site_xpos[site_id].copy()
        last = self._last_site_pos[site_id].copy()
        vel = (cur - last) / max(self.dt, 1e-8)
        return vel.astype(np.float32)

    def _update_last_site_pos(self):
        for i in range(self.model.nsite):
            self._last_site_pos[i] = self.data.site_xpos[i].copy()

    def _get_all_hold_world_positions(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        poses = []
        for name, bid in self.hold_bodies[:self._max_holds]:
            poses.append(self.data.xpos[bid].copy())
        return np.stack(poses, axis=0)

    # -------------------- IK solver (analytic Jacobian + adaptive damping) --------------------
    def _ik_solve_arm(
        self,
        site_id: int,
        joint_ids: list,
        desired_world_pos: np.ndarray,
        max_iters: int = 8,
        tol: float = 1e-4,
        damping: Optional[float] = None
    ) -> np.ndarray:
        if damping is None:
            damping = self.ik_damping
        active_dofs = [int(self.model.jnt_dofadr[jid]) for jid in joint_ids]
        # keep backup of qpos so updates for one arm don't permanently change state during iterative solve
        qpos_backup = self.data.qpos.copy()

        # simple adaptive damping: if a step increases error, increase damping and retry
        cur_damping = float(damping)
        last_err_norm = np.inf

        for it in range(max_iters):
            mujoco.mj_forward(self.model, self.data)
            p_cur = self.data.site_xpos[site_id].copy()
            err = desired_world_pos - p_cur
            err_norm = float(np.linalg.norm(err))

            if err_norm < tol:
                break

            # analytic jacobian: fills _jacp (3 x nv) and _jacr
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, site_id)
            J_full = self._jacp  # 3 x nv
            J_small = J_full[:, active_dofs]  # 3 x m

            # build damped least squares solve for delta q (in joint-space)
            A = J_small.T @ J_small
            A += (cur_damping**2) * np.eye(A.shape[0])
            b = J_small.T @ err

            # solve robustly
            try:
                dq = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(A, b, rcond=None)[0]

            # sanitize dq
            dq = np.nan_to_num(dq, nan=0.0, posinf=np.sign(dq)*1e2, neginf=np.sign(dq)*-1e2)

            # clamp per-iteration joint step magnitude
            nd = float(np.linalg.norm(dq))
            if nd > self.max_joint_step:
                dq = dq * (self.max_joint_step / (nd + 1e-12))

            # apply dq into qpos for the active dofs
            for i, dof in enumerate(active_dofs):
                self.data.qpos[dof] += float(dq[i])

            mujoco.mj_forward(self.model, self.data)
            new_p = self.data.site_xpos[site_id].copy()
            new_err_norm = float(np.linalg.norm(desired_world_pos - new_p))

            # if error got worse, increase damping and undo step
            if new_err_norm > err_norm + 1e-6:
                # undo
                self.data.qpos[:] = qpos_backup[:]
                cur_damping *= max(2.0, self.ik_damping_growth)  # increase damping
                # try a smaller step by scaling dq
                scaled = dq * 0.5
                for i, dof in enumerate(active_dofs):
                    self.data.qpos[dof] += float(scaled[i])
                mujoco.mj_forward(self.model, self.data)
                new_p2 = self.data.site_xpos[site_id].copy()
                new_err_norm2 = float(np.linalg.norm(desired_world_pos - new_p2))
                # if still worse, increase damping more and try again in next iter
                if new_err_norm2 > err_norm:
                    self.data.qpos[:] = qpos_backup[:]
                    cur_damping *= 4.0
            else:
                # accept this qpos as new baseline for next iter
                qpos_backup = self.data.qpos.copy()

            last_err_norm = err_norm

        # extract q_targets in same ordering as joint_ids
        q_targets = [self.data.qpos[self.model.jnt_qposadr[jid]] for jid in joint_ids]
        q_targets = np.array(q_targets, dtype=np.float32)
        # clamp to joint limits before returning
        q_targets = clamp_q_targets(self.model, joint_ids, q_targets)
        # restore qpos to the state before IK (env will set actuators or write qpos explicitly)
        self.data.qpos[:] = qpos_backup[:]
        mujoco.mj_forward(self.model, self.data)
        return q_targets

    # -------------------- observations --------------------
    def _get_obs(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        base_pos_world = self.data.site_xpos[self.site_ids['base']].copy()
        base_quat = self._site_quat(self.site_ids['base'])
        base_xmat = (np.asarray(self.data.site_xmat[self.site_ids['base']]).reshape(3,3)
                     if hasattr(self.data, "site_xmat") else np.eye(3))
        base_linvel_world = self._site_linvel_world(self.site_ids['base'])
        R_base = base_xmat
        base_linvel_base = R_base.T @ base_linvel_world

        rpos_world = self.data.site_xpos[self.site_ids['r']].copy()
        lpos_world = self.data.site_xpos[self.site_ids['l']].copy()
        rvel_world = self._site_linvel_world(self.site_ids['r'])
        lvel_world = self._site_linvel_world(self.site_ids['l'])

        rpos_base = R_base.T @ (rpos_world - base_pos_world)
        lpos_base = R_base.T @ (lpos_world - base_pos_world)
        rvel_base = R_base.T @ rvel_world
        lvel_base = R_base.T @ lvel_world

        all_holds_world = self._get_all_hold_world_positions()
        r_idx = min(self._right_target_idx, len(all_holds_world)-1)
        l_idx = min(self._left_target_idx, len(all_holds_world)-1)
        r_target_world = all_holds_world[r_idx].copy()
        l_target_world = all_holds_world[l_idx].copy()
        r_target_base = R_base.T @ (r_target_world - base_pos_world)
        l_target_base = R_base.T @ (l_target_world - base_pos_world)

        obs = np.concatenate([
            base_pos_world.astype(np.float32),
            base_quat.astype(np.float32),
            base_linvel_base.astype(np.float32),
            rpos_base.astype(np.float32),
            rvel_base.astype(np.float32),
            lpos_base.astype(np.float32),
            lvel_base.astype(np.float32),
            r_target_base.astype(np.float32),
            l_target_base.astype(np.float32),
        ], axis=0)
        return obs

    # -------------------- reset/step --------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        start_right = 1
        start_left = 0
        if options is not None:
            start_right = int(options.get('start_right_idx', start_right))
            start_left = int(options.get('start_left_idx', start_left))
            max_holds = options.get('max_holds', None)
            if max_holds is not None:
                self.set_hold_limit(int(max_holds))
        self._right_target_idx = max(0, min(start_right, self._max_holds-1))
        self._left_target_idx = max(0, min(start_left, self._max_holds-1))
        self._step_count = 0
        self._update_last_site_pos()
        # store last-good
        self._last_good_qpos = self.data.qpos.copy()
        self._last_good_qvel = self.data.qvel.copy()
        return self._get_obs(), {}

    def _sanitize_ctrl_and_state(self):
        # sanitize ctrl
        if not np.all(np.isfinite(self.data.ctrl)):
            if self.debug: print("Sanitize: non-finite ctrl detected -> nan_to_num")
            self.data.ctrl[:] = np.nan_to_num(self.data.ctrl, nan=0.0, posinf=1e3, neginf=-1e3)
        # clamp ctrl to reasonable range to avoid huge instantaneous forces
        MAX_CTRL = 1e4
        self.data.ctrl[:] = np.clip(self.data.ctrl, -MAX_CTRL, MAX_CTRL)

        # sanitize qpos/qvel
        if not np.all(np.isfinite(self.data.qpos)):
            if self.debug: print("Sanitize: non-finite qpos -> reset to last good qpos")
            self.data.qpos[:] = self._last_good_qpos.copy()
        if not np.all(np.isfinite(self.data.qvel)):
            if self.debug: print("Sanitize: non-finite qvel -> zero")
            self.data.qvel[:] = np.nan_to_num(self.data.qvel, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape == (6,)
        # clip action
        action = np.clip(action, -1.0, 1.0)

        delta_r_base = action[0:3] * self.ee_max_step
        delta_l_base = action[3:6] * self.ee_max_step

        mujoco.mj_forward(self.model, self.data)

        base_world = self.data.site_xpos[self.site_ids['base']].copy()
        base_xmat = (np.asarray(self.data.site_xmat[self.site_ids['base']]).reshape(3,3)
                     if hasattr(self.data, "site_xmat") else np.eye(3))
        R_base = base_xmat

        rpos_world = self.data.site_xpos[self.site_ids['r']].copy()
        lpos_world = self.data.site_xpos[self.site_ids['l']].copy()
        r_target_world_pos = rpos_world + (R_base @ delta_r_base)
        l_target_world_pos = lpos_world + (R_base @ delta_l_base)

        # compute IK per-arm (we don't let the right-arm IK permanently modify qpos for the left solve)
        qpos_backup = self.data.qpos.copy()

        # RIGHT
        try:
            q_targets_r = self._ik_solve_arm(self.site_ids['r'], self.joint_ids_r, r_target_world_pos,
                                             max_iters=8, tol=1e-4)
        except Exception as e:
            if self.debug: print("IK right exception:", e)
            q_targets_r = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.joint_ids_r], dtype=np.float32)
        q_targets_r = clamp_q_targets(self.model, self.joint_ids_r, q_targets_r)

        # write to actuators or fallback mapping
        if len(self.actuator_ids_r) == len(q_targets_r) and len(self.actuator_ids_r) > 0:
            for aid, qt in zip(self.actuator_ids_r, q_targets_r):
                self.data.ctrl[aid] = float(qt)
        else:
            # fallback: write first three ctrl slots (preserve others)
            for i, qt in enumerate(q_targets_r):
                if i < self.data.ctrl.shape[0]:
                    self.data.ctrl[i] = float(qt)

        # restore qpos and solve left
        self.data.qpos[:] = qpos_backup[:]
        mujoco.mj_forward(self.model, self.data)

        # LEFT
        try:
            q_targets_l = self._ik_solve_arm(self.site_ids['l'], self.joint_ids_l, l_target_world_pos,
                                             max_iters=8, tol=1e-4)
        except Exception as e:
            if self.debug: print("IK left exception:", e)
            q_targets_l = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.joint_ids_l], dtype=np.float32)
        q_targets_l = clamp_q_targets(self.model, self.joint_ids_l, q_targets_l)

        if len(self.actuator_ids_l) == len(q_targets_l) and len(self.actuator_ids_l) > 0:
            for aid, qt in zip(self.actuator_ids_l, q_targets_l):
                self.data.ctrl[aid] = float(qt)
        else:
            start = len(self.actuator_ids_r) if len(self.actuator_ids_r) > 0 else 3
            for i, qt in enumerate(q_targets_l):
                idx = start + i
                if idx < self.data.ctrl.shape[0]:
                    self.data.ctrl[idx] = float(qt)

        # SANITIZE BEFORE STEP
        self._sanitize_ctrl_and_state()

        # apply settle steps so actuators can apply forces gradually
        unstable_flag = False
        for _ in range(self.n_settle):
            mujoco.mj_step(self.model, self.data)
            # detect instability
            if not np.all(np.isfinite(self.data.qacc)):
                unstable_flag = True
                if self.debug:
                    print(f"Unstable sim detected (non-finite qacc). Time={self.data.time}")
                break

        if unstable_flag:
            # attempt a safe recovery: revert to last good state
            if self.debug: print("Attempting recovery: revert to last good qpos/qvel and zero ctrl")
            self.data.qpos[:] = self._last_good_qpos.copy()
            self.data.qvel[:] = self._last_good_qvel.copy()
            self.data.ctrl[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            # mark truncated and return early
            truncated = True
            terminated = False
            self._step_count += 1
            self._update_last_site_pos()
            obs = self._get_obs()
            reward = -50.0  # heavy penalty for instability
            info = {"unstable_sim": True}
            return obs.astype(np.float32), float(reward), terminated, truncated, info

        # If we reached here, update last-good state
        self._last_good_qpos = self.data.qpos.copy()
        self._last_good_qvel = self.data.qvel.copy()

        # normal bookkeeping
        self._step_count += 1
        self._update_last_site_pos()
        obs = self._get_obs()

        # compute reward & termination (same shaping as before)
        all_holds_world = self._get_all_hold_world_positions()
        r_goal = all_holds_world[min(self._right_target_idx, len(all_holds_world)-1)]
        l_goal = all_holds_world[min(self._left_target_idx, len(all_holds_world)-1)]
        rpos = self.data.site_xpos[self.site_ids['r']].copy()
        lpos = self.data.site_xpos[self.site_ids['l']].copy()
        r_dist = float(np.linalg.norm(rpos - r_goal))
        l_dist = float(np.linalg.norm(lpos - l_goal))

        reward_shaping = -(r_dist + l_dist)
        energy_pen = -0.01 * float(np.linalg.norm(action) ** 2)
        time_pen = -1.0 * self.dt
        reward = reward_shaping + energy_pen + time_pen

        terminated = False
        truncated = False
        info = {}

        if (r_dist < self.success_radius) and (self._right_target_idx < len(all_holds_world)-1):
            reward += 5.0
            self._right_target_idx += 1
        if (l_dist < self.success_radius) and (self._left_target_idx < len(all_holds_world)-1):
            reward += 5.0
            self._left_target_idx += 1

        if (self._right_target_idx >= len(all_holds_world)-1) and (self._left_target_idx >= len(all_holds_world)-1):
            reward += 200.0
            terminated = True
            info["task_complete"] = True

        if self._step_count >= self.max_steps:
            truncated = True
            info["TimeLimit.truncated"] = True

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    # -------------------- render/close --------------------
    def render(self):
        try:
            if not hasattr(self, "_viewer") or self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if hasattr(self._viewer, "sync"):
                self._viewer.sync()
            elif hasattr(self._viewer, "render"):
                self._viewer.render()
        except Exception as e:
            if self.debug:
                print("Render error:", e)

    def close(self):
        try:
            if hasattr(self, "_viewer") and self._viewer is not None:
                try:
                    self._viewer.close()
                except Exception:
                    pass
        except Exception:
            pass
