"""
climb_env_v7.py  —  Redesigned ClimbBot RL environment.

Key differences vs the test_6 env (which could not learn to climb by construction):
  * Joint-space control. The policy outputs 6 small joint-target deltas that drive the
    position servos directly. No inverse-kinematics solver runs inside the RL loop
    (that was the main source of slowness, non-stationarity, and the QACC NaN blowups).
  * Physical grip. "Grasping" is real contact + friction between the hook geoms and the
    holds. Nothing magically welds the hook to a hold.
  * A dense, climbing-shaped reward: reach the target holds, press the hooks into them
    (contact force), stay upright, gain height; large penalty for falling.
  * Correct Gymnasium semantics: time-limit -> truncated, fall -> terminated.
  * A `freeze_base` curriculum flag. Stage 0 anchors the torso to the world (via the
    `base_anchor` weld in scene_v7.xml) so the agent first learns the reach+hook skill
    on a stable base. Later stages release it for free climbing.

Author: redesign pass, 2026-06.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
DEFAULT_XML = os.path.join(ASSETS, "scene_v7.xml")
TREADMILL_XML = os.path.join(ASSETS, "scene_v7_treadmill.xml")
MESH_TREADMILL_XML = os.path.join(ASSETS, "scene_v7_treadmill_holds.xml")
LADDER_XML = os.path.join(ASSETS, "scene_v7_ladder.xml")
ENV_DIR = os.path.dirname(os.path.abspath(__file__))


def is_mesh_holds_xml(xml_path):
    return "treadmill_holds" in os.path.basename(xml_path)


def resolve_treadmill_xml(mesh_holds=False, xml_path=None):
    """Return treadmill MJCF path. Explicit non-default xml_path wins."""
    if xml_path and xml_path != DEFAULT_XML and os.path.isfile(xml_path):
        return xml_path
    return MESH_TREADMILL_XML if mesh_holds else TREADMILL_XML


def default_hang_qpos(xml_path):
    """Saved hang pose for a scene. Mesh treadmill prefers its own searched pose
    (hang_start_treadmill_holds.npy, written by test_mesh_holds.py when the random
    search finds a stable hang) and falls back to the cylinder-rung pose."""
    if is_mesh_holds_xml(xml_path):
        mesh_pose = os.path.join(ENV_DIR, "hang_start_treadmill_holds.npy")
        if os.path.exists(mesh_pose):
            return mesh_pose
    if is_mesh_holds_xml(xml_path) or "treadmill" in os.path.basename(xml_path):
        return os.path.join(ENV_DIR, "hang_start_treadmill.npy")
    return os.path.join(ENV_DIR, "hang_start_ladder.npy")

# The 6 actuated joints, in actuator order (r1,r2,r3_1, l1,l2,l3_1)
ACT_JOINTS = ["r1", "r2", "r3_1", "l1", "l2", "l3_1"]
# Per-joint target delta applied per control step when |action|=1 (rad for hinges, m for slide).
# Prismatic rate is 1.5 mm/step (was 6 mm): a realistic slow linear actuator — full 0.134 m
# stroke (x3 stages = 0.4 m travel) takes ~90 control steps (3.6 s) instead of ~22.
DELTA_SCALE = np.array([0.05, 0.05, 0.0015, 0.05, 0.05, 0.0015], dtype=np.float64)

# Mirror mapping between the arms (verified kinematically to ~3 mm): the left-arm pose
# that mirrors a right-arm pose across the robot's mid-plane is (l1,l2,l3)=(-r1,-r2,+r3).
MIRROR_SIGN = np.array([-1.0, -1.0, 1.0])

# Gripper hook collision geoms (load-bearing part of each hand).
# The catch bars (r_catch / l_catch) are what actually bear the hanging load, so they
# MUST be part of the force-sensing set — without them a firm grip can read ~0 N.
RIGHT_HOOK_GEOMS = [f"assembly_12_collision_1_{i}" for i in range(5)] + ["r_catch"]
LEFT_HOOK_GEOMS = [f"assembly_11_collision_1_{i}" for i in range(5)] + ["l_catch"]


class ClimbBotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self,
                 xml_path=DEFAULT_XML,
                 frame_skip=20,
                 max_steps=250,
                 freeze_base=True,
                 right_hold="hold_2",
                 left_hold="hold_1",
                 init_joint_noise=0.05,
                 climb=False,
                 treadmill=False,
                 freeze_support=False,
                 mirror=True,
                 max_moves=0,
                 tm_spacing=0.2,
                 rung_jitter_y=0.0,
                 rung_jitter_z=0.0,
                 gravity_scale=1.0,
                 time_penalty=0.0,
                 push_newton=0.0,
                 natural_form=False,
                 init_qpos_file=None,
                 render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.freeze_base = bool(freeze_base)
        self.init_joint_noise = float(init_joint_noise)
        # climb mode: free base, start from a hanging pose, reward upward progress
        self.treadmill = bool(treadmill)
        self.climb = bool(climb) or self.treadmill
        self.freeze_support = bool(freeze_support)   # policy commands only the active arm
        # mirror trick: when the RIGHT hand is active, present the policy with the
        # left-right mirrored obs and mirror its action back. The policy then only ever
        # solves "left hand reaches" and the skill transfers to both gait roles for free.
        self.mirror = bool(mirror)
        self.max_moves = int(max_moves)              # >0: end episode after this many moves
        self._tm_spacing_cfg = float(tm_spacing)
        # handhold randomization: every rung (except the two the hands start on) is placed
        # at nominal_y ± rung_jitter_y (m, sideways along the wall) and its vertical gap is
        # spacing ± rung_jitter_z fraction. The policy must reach wherever the hold IS.
        self.rung_jitter_y = float(rung_jitter_y)
        self.rung_jitter_z = float(rung_jitter_z)
        # URGENCY: small constant per-step cost (climb mode). The potential-based shaping
        # is time-indifferent — hovering nets ~0 — so without this a slow move scores the
        # same as a fast one. A constant can't be farmed; falls still dominate it by far.
        self.time_penalty = float(time_penalty)
        # ROBUSTNESS: random pushes on the base. Every so often a random mostly-lateral
        # force (up to this many N) shoves the torso for ~5 control steps (0.2 s); the
        # policy must learn to keep its grip and recover. 0 = off.
        self.push_newton = float(push_newton)
        # NATURAL FORM: stronger human-like climbing priors — upright torso, level
        # shoulders, smooth joint commands, keep the body close to the wall, and
        # sequence pull-up before a long reach (cadence). Defaults stay off so older
        # runs are bit-identical; --natural turns this on for fine-tuning.
        self.natural_form = bool(natural_form)
        if self.natural_form and self.time_penalty <= 0.0:
            self.time_penalty = 0.03
        if self.climb:
            self.freeze_base = False  # climbing requires a free torso, always
        self._init_qpos = np.load(init_qpos_file) if init_qpos_file else None

        self.xml_path = xml_path
        self.mesh_holds = is_mesh_holds_xml(xml_path)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        # gravity-assist curriculum: lighter body -> easy pull-up, far less grip load to slip
        self.model.opt.gravity[2] *= float(gravity_scale)

        # --- resolve ids we need ---
        self.jnt_ids = [self._jid(j) for j in ACT_JOINTS]
        self.qpos_adr = np.array([self.model.jnt_qposadr[j] for j in self.jnt_ids])
        self.dof_adr = np.array([self.model.jnt_dofadr[j] for j in self.jnt_ids])
        self.jnt_range = np.array([self.model.jnt_range[j] for j in self.jnt_ids])  # (6,2)
        self.act_ids = [self._aid(f"{j}_ctrl") for j in ACT_JOINTS]

        self.base_bid = self._bid("robot_base_tilted")
        self.r_site = self._sid("r_grip_site")
        self.l_site = self._sid("l_grip_site")
        self.r_hold_site = self._sid(right_hold + "_site")
        self.l_hold_site = self._sid(left_hold + "_site")

        self.r_hook_gids = [self._gid(g) for g in RIGHT_HOOK_GEOMS]
        self.l_hook_gids = [self._gid(g) for g in LEFT_HOOK_GEOMS]
        # all world-collidable geoms (wall + holds)
        self.world_gids = set(int(g) for g in range(self.model.ngeom)
                              if self.model.geom_contype[g] == 1)
        # ONLY the hold geoms: engagement reward counts contact with a hold, not the flat
        # wall. Without this the policy farms the contact bonus by mashing the wall.
        hold_bids = {self._bid(f"hold_{i}") for i in range(1, 60)}
        hold_bids.discard(-1)
        self.hold_gids = set(int(g) for g in range(self.model.ngeom)
                             if int(self.model.geom_bodyid[g]) in hold_bids)

        # --- hand-over-hand gait (climb mode) ---
        # Detect all rungs generically and split by side (world-y sign), sorted by height.
        # Left hand climbs the left-side rungs; right hand the right-side rungs. Works for
        # any rung layout (alternating wall or dense ladder).
        if self.climb:
            mujoco.mj_forward(self.model, self.data)
            left, right = [], []
            for n in range(1, 60):
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"hold_{n}")
                if bid < 0:
                    continue
                sid = self._sid(f"hold_{n}_site")
                gids = self._hold_geoms(bid)
                y = float(self.data.site_xpos[sid][1])
                z = float(self.data.site_xpos[sid][2])
                (left if y > 0 else right).append((z, sid, gids))
            left.sort(); right.sort()
            self.left_sites = [s for _, s, _ in left]
            self.right_sites = [s for _, s, _ in right]
            self.left_rgeoms = {i: gids for i, (_, _, gids) in enumerate(left)}
            self.right_rgeoms = {i: gids for i, (_, _, gids) in enumerate(right)}
            # LOAD-BEARING grip thresholds, as fractions of actual body weight (gravity
            # already scaled into the model). A hold only counts when the hook is truly
            # carrying weight (upward rung-on-hook force), not merely touching the rung —
            # a sideways "fistbump" against the rung reads ~0 uplift and never advances.
            self.body_weight = float(self.model.body_subtreemass[self.base_bid]) \
                * abs(self.model.opt.gravity[2])
            self.firm_thresh = 0.20 * self.body_weight    # support hook must carry >=20% W
            self.advance_grasp_dist = 0.07
            self.advance_force = 0.15 * self.body_weight  # new hold must carry >=15% W
            # a grab only counts if the full gate holds this many consecutive control
            # steps — rejects grazes and guarantees a genuinely seated grip. Mesh pocket
            # edges can carry a dynamic bounce for ~0.12 s and fake a grab, so mesh
            # requires 5 steps (0.2 s); cylinders keep the original 3 (0.12 s).
            self.secure_steps = 5 if self.mesh_holds else 3

        if self.treadmill:
            # mocap holds we can reposition: split by side, keep mocap id + site + geoms
            self.left_mocap, self.right_mocap = [], []
            for n in range(1, 60):
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"hold_{n}")
                if bid < 0:
                    continue
                mid = int(self.model.body_mocapid[bid])
                if mid < 0:
                    continue
                sid = self._sid(f"hold_{n}_site"); gids = self._hold_geoms(bid)
                y = float(self.data.site_xpos[sid][1])
                rec = {"mid": mid, "sid": sid, "gids": gids, "y": y}
                (self.left_mocap if y > 0 else self.right_mocap).append(rec)
            self.tm_spacing = self._tm_spacing_cfg
            self.tm_x = -0.044          # rung protrusion (world x)
            self.tm_base_z = 0.9        # lowest rung height (where the hands naturally hang)

        # --- base anchor weld: set per-instance (curriculum stage) ---
        self.base_weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "base_anchor")
        if self.base_weld_id >= 0:
            self.model.eq_active0[self.base_weld_id] = 1 if self.freeze_base else 0

        # --- spaces ---
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        if self.climb:
            # qpos, qvel, active->target vec, support->hold vec, base_z, base_up,
            # force_active, force_support, active_is_left
            obs_dim = 6 + 6 + 3 + 3 + 1 + 3 + 2 + 1
        else:
            obs_dim = 6 + 6 + 3 + 3 + 1 + 3 + 2  # qpos, qvel, vec_r, vec_l, base_z, base_up, contacts
        big = np.float32(50.0)
        self.observation_space = spaces.Box(-big, big, shape=(obs_dim,), dtype=np.float32)

        # reward weights
        self.w_dist = 1.0      # distance shaping
        self.w_progress = 8.0  # reward for closing distance (potential-based)
        self.w_contact = 0.5   # reward for hook engagement (contact force, clipped)
        self.w_upright = 0.2   # stay oriented
        self.w_ctrl = 0.002    # action penalty
        self.alive_bonus = 0.05
        self.fall_penalty = 10.0
        self.success_bonus = 5.0
        self.contact_thresh = 2.0   # N, counts as "engaged"
        self.grasp_dist = 0.06      # m, hook close enough to a hold
        self.fall_z = 0.0           # base below this world-z (free mode) => fell

        self.target = np.zeros(6)
        self._prev_dr = None
        self._prev_dl = None
        self.step_count = 0

    # ---------- id helpers ----------
    def _jid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
    def _aid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    def _bid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
    def _sid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
    def _gid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)

    def _hold_geoms(self, bid):
        """Colliding geoms on a hold body (mesh parts, or a single rung cylinder)."""
        gids = {int(g) for g in range(self.model.ngeom)
                if int(self.model.geom_bodyid[g]) == bid and self.model.geom_contype[g] > 0}
        return gids

    def _tm_hold_body_pos(self, body_world):
        """Mocap body origin in world frame (tm_x is always body x, not site x)."""
        return np.asarray(body_world, dtype=float)

    # ---------- core ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # VERIFIED SEATING: a noisy start can leave a hook "on" the hold but not bearing
        # load (on mesh: perched on a horn / off the lip). Each attempt draws fresh joint
        # noise, settles, then CHECKS both hooks are cradled and load-bearing; retry on
        # failure so no episode ever starts from a fake grip.
        max_tries = 10 if self.treadmill else 1
        for _try in range(max_tries):
            mujoco.mj_resetData(self.model, self.data)

            # climb mode: start from the saved hanging pose
            if self.climb and self._init_qpos is not None:
                self.data.qpos[:] = self._init_qpos

            # small randomization of starting joint config (within range)
            q0 = self.data.qpos[self.qpos_adr].copy()
            if self.init_joint_noise > 0:
                noise = self.np_random.uniform(-self.init_joint_noise, self.init_joint_noise, size=6)
                q0 = np.clip(q0 + noise, self.jnt_range[:, 0], self.jnt_range[:, 1])
                self.data.qpos[self.qpos_adr] = q0
            mujoco.mj_forward(self.model, self.data)

            # initialize position targets to current joint config (no jump on first step)
            self.target = q0.copy()
            self.data.ctrl[self.act_ids] = self.target

            if not self.treadmill:
                break
            self._tm_set_window()
            mujoco.mj_forward(self.model, self.data)
            # SETTLE: hold the start pose for a few frames so the hooks seat onto the rungs.
            # Without this, startup noise can leave the grippers just below a rung (no grip)
            # and the robot drops at t=0.
            self.data.ctrl[self.act_ids] = self.target
            for _ in range(40):
                mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if self._start_seated():
                break
        self._reset_tries = _try + 1   # diagnostic: how many draws until a verified seat

        self.step_count = 0

        if self.treadmill:
            # Randomize which hand moves first: the gait must work from BOTH roles or the
            # robot can only ever do one move (the mirrored reach stays out-of-distribution).
            self.active = "left" if self.np_random.random() < 0.5 else "right"
            self.moves_done = 0
        self._secure = 0
        self._prev_phi = None   # potential-based shaping re-anchors on reset
        self._prev_action = np.zeros(6)
        self._push_left = 0
        self._push_vec = np.zeros(3)
        if self.base_bid >= 0:
            self.data.xfrc_applied[self.base_bid, :3] = 0.0

        self._prev_dr, self._prev_dl = self._distances()
        self._start_z = float(self.data.xpos[self.base_bid][2])
        self._prev_z = self._start_z
        self._max_z = self._start_z
        # reference orientation: keep the base near its starting (vertical) attitude
        self._start_up = self.data.xmat[self.base_bid].reshape(3, 3)[:, 2].copy()

        if self.treadmill:
            self._prev_active_d = self._active_target_dist()
        elif self.climb:
            # sync gait state to whichever rungs the hands are actually nearest at reset
            lp = self.data.site_xpos[self.l_site]
            rp = self.data.site_xpos[self.r_site]
            self.li = int(np.argmin([np.linalg.norm(lp - self.data.site_xpos[s]) for s in self.left_sites]))
            self.ri = int(np.argmin([np.linalg.norm(rp - self.data.site_xpos[s]) for s in self.right_sites]))
            self.active = "left"
            self.moves_done = 0
            self._prev_active_d = self._active_target_dist()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        # MIRROR: when the right hand is active the policy acted on a mirrored obs, so its
        # "left arm" command drives the real right arm (and vice versa) with mirrored signs.
        if self.mirror and self.climb and self.active == "right":
            action = np.concatenate([MIRROR_SIGN * action[3:6], MIRROR_SIGN * action[0:3]])
        # SIMPLE MODE: only command the ACTIVE arm; the support arm holds rigid (its target
        # is left unchanged). Removes the "drop the support hand" failure and halves the DoF.
        if self.freeze_support and self.climb:
            mask = np.zeros(6)
            active_idx = [3, 4, 5] if self.active == "left" else [0, 1, 2]
            mask[active_idx] = 1.0
            action = action * mask
        # integrate joint-target deltas, clip to joint limits
        self.target = np.clip(self.target + action * DELTA_SCALE,
                              self.jnt_range[:, 0], self.jnt_range[:, 1])
        self.data.ctrl[self.act_ids] = self.target

        # random push perturbations (robustness curriculum): occasional mostly-lateral
        # shove on the torso for ~5 control steps, magnitude up to push_newton
        if self.climb and self.push_newton > 0:
            if self._push_left == 0 and self.np_random.random() < 0.02:
                v = self.np_random.normal(size=3)
                v[2] *= 0.3   # mostly sideways/outward, not straight up/down
                v /= (np.linalg.norm(v) + 1e-9)
                self._push_vec = v * self.push_newton * self.np_random.uniform(0.3, 1.0)
                self._push_left = 5
            if self._push_left > 0:
                self.data.xfrc_applied[self.base_bid, :3] = self._push_vec
                self._push_left -= 1
                if self._push_left == 0:
                    self.data.xfrc_applied[self.base_bid, :3] = 0.0

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()
        base_z = float(self.data.xpos[self.base_bid][2])

        if self.climb:
            reward, info = self._climb_reward(action, base_z)
            # fell = dropped well below the highest point reached (not the start), so the
            # check stays meaningful during continuous multi-move climbs
            fell = base_z < (self._max_z - 0.25)
        else:
            reward, info = self._reward(action)
            fell = (not self.freeze_base) and (base_z < self.fall_z)

        terminated = bool(fell)
        if fell:
            # Penalty scales with the episode time remaining: cutting an episode short must
            # never beat staying on the wall (otherwise the per-step penalties make an early
            # fall the reward-optimal "escape" and PPO learns to dive off the rungs).
            reward -= self.fall_penalty + 2.0 * max(0, self.max_steps - self.step_count)
        # SIMPLE MODE: end the episode after the target number of moves (e.g. 1) with a bonus
        if self.climb and self.max_moves > 0 and self.moves_done >= self.max_moves:
            terminated = True
            reward += 5.0
        truncated = self.step_count >= self.max_steps

        info["base_z"] = base_z
        if not self.climb:   # climb already sets is_success (moves_done based) in its reward
            info["is_success"] = bool(info["engaged_r"] and info["engaged_l"]
                                      and info["d_r"] < self.grasp_dist and info["d_l"] < self.grasp_dist)
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    # ---------- observation / reward ----------
    def _distances(self):
        r = self.data.site_xpos[self.r_site] - self.data.site_xpos[self.r_hold_site]
        l = self.data.site_xpos[self.l_site] - self.data.site_xpos[self.l_hold_site]
        return float(np.linalg.norm(r)), float(np.linalg.norm(l))

    # ---- treadmill helpers ----
    def _tm_sorted(self, side):
        rungs = self.left_mocap if side == "left" else self.right_mocap
        return sorted(rungs, key=lambda r: float(self.data.mocap_pos[r["mid"]][2]))

    def _rung_offsets(self):
        """Random placement offsets for one rung: (dy, spacing_scale)."""
        dy = self.np_random.uniform(-self.rung_jitter_y, self.rung_jitter_y) \
            if self.rung_jitter_y > 0 else 0.0
        sz = 1.0 + (self.np_random.uniform(-self.rung_jitter_z, self.rung_jitter_z)
                    if self.rung_jitter_z > 0 else 0.0)
        return dy, sz

    def _tm_set_window(self):
        """Place the mocap holds as a small window. The lowest hold on each side is placed
        relative to where that hand's grip site actually is (post-noise), so BOTH hooks
        start properly seated regardless of the joint randomization.
        Holds ABOVE the starting pair are randomized (y and vertical gap) so the policy
        learns to grab holds wherever they are, not a memorized fixed ladder.
        Mesh holds use the same contract as cylinder rungs: site at body origin, tm_x is
        body x for recycled holds."""
        for lst, y, site in ((self.left_mocap, 0.1, self.l_site),
                             (self.right_mocap, -0.2, self.r_site)):
            g = self.data.site_xpos[site]
            z = float(g[2]) - 0.011
            for k, r in enumerate(lst):
                if k == 0:
                    # Match hook height (z), keep x on the wall plane (tm_x). On MESH also
                    # match the hook's lateral y: joint noise shifts the hook sideways by
                    # up to ~2 cm, and the mesh pocket is only ±3 cm wide — a fixed-y hold
                    # can start perched on a horn ("grip looks on but bears nothing").
                    # Cylinders span ±9 cm, so nominal y is fine there.
                    y0 = float(g[1]) if self.mesh_holds else y
                    target = np.array([self.tm_x, y0, z])
                else:
                    dy, sz = self._rung_offsets()
                    z += self.tm_spacing * sz
                    target = np.array([self.tm_x, y + dy, z])
                self.data.mocap_pos[r["mid"]] = self._tm_hold_body_pos(target)

    def _tm_recycle(self, side):
        """Move the lowest rung on a side up to the top of that side's stack (the scroll).
        The recycled rung is placed at a randomized lateral offset and vertical gap, so the
        upcoming handholds keep appearing in different spots."""
        rungs = self._tm_sorted(side)
        top_z = float(self.data.mocap_pos[rungs[-1]["mid"]][2])
        dy, sz = self._rung_offsets()
        target = np.array([self.tm_x, rungs[0]["y"] + dy, top_z + self.tm_spacing * sz])
        self.data.mocap_pos[rungs[0]["mid"]] = self._tm_hold_body_pos(target)

    # ---- hand-over-hand gait bookkeeping ----
    def _active_parts(self):
        """Return (active_hook_gids, active_grip_site, active_target_site, active_target_geoms,
                   support_hook_gids, support_grip_site, support_hold_geoms)."""
        if self.treadmill:
            if self.active == "left":
                a_hook, a_site = self.l_hook_gids, self.l_site
                a = self._tm_sorted("left"); s = self._tm_sorted("right")
                sh, s_site = self.r_hook_gids, self.r_site
            else:
                a_hook, a_site = self.r_hook_gids, self.r_site
                a = self._tm_sorted("right"); s = self._tm_sorted("left")
                sh, s_site = self.l_hook_gids, self.l_site
            a_target = a[1] if len(a) > 1 else a[0]   # 2nd-lowest rung = next one up
            s_hold = s[0]                             # support hand on its lowest rung
            return (a_hook, a_site, a_target["sid"], a_target["gids"],
                    sh, s_site, s_hold["gids"])
        if self.active == "left":
            tgt = min(self.li + 1, len(self.left_sites) - 1)
            return (self.l_hook_gids, self.l_site, self.left_sites[tgt], self.left_rgeoms[tgt],
                    self.r_hook_gids, self.r_site, self.right_rgeoms[self.ri])
        else:
            tgt = min(self.ri + 1, len(self.right_sites) - 1)
            return (self.r_hook_gids, self.r_site, self.right_sites[tgt], self.right_rgeoms[tgt],
                    self.l_hook_gids, self.l_site, self.left_rgeoms[self.li])

    def _active_target_dist(self):
        ah, as_, at, atg, sh, ss, sg = self._active_parts()
        return float(np.linalg.norm(self.data.site_xpos[as_] - self.data.site_xpos[at]))

    def _get_obs(self):
        q = self.data.qpos[self.qpos_adr]
        qd = self.data.qvel[self.dof_adr]
        mid = 0.5 * (self.jnt_range[:, 0] + self.jnt_range[:, 1])
        half = 0.5 * (self.jnt_range[:, 1] - self.jnt_range[:, 0]) + 1e-6
        q_n = (q - mid) / half
        qd_s = np.clip(qd / 10.0, -5, 5)
        base_z = np.array([self.data.xpos[self.base_bid][2]])
        base_up = self.data.xmat[self.base_bid].reshape(3, 3)[:, 2]

        if self.climb:
            ah, as_, at, atg, sh, ss, sg = self._active_parts()
            vec_active = self.data.site_xpos[as_] - self.data.site_xpos[at]      # active hand -> next rung
            # support hand vector to the rung it should be holding:
            if self.treadmill:
                sup_side = "right" if self.active == "left" else "left"
                sup_site = self._tm_sorted(sup_side)[0]["sid"]
            else:
                sup_site = self.right_sites[self.ri] if self.active == "left" else self.left_sites[self.li]
            vec_support = self.data.site_xpos[ss] - self.data.site_xpos[sup_site]
            # load-bearing (upward) forces, same signals the advance gate is built on
            f_active = np.clip(self._uplift_on(ah, atg), 0.0, 50.0) / 50.0
            f_support = np.clip(self._uplift_on(sh), 0.0, 50.0) / 50.0
            active_is_left = 1.0 if self.active == "left" else 0.0
            if self.mirror and self.active == "right":
                # left-right mirror: swap arm blocks (with joint-sign flips) and negate the
                # y component of every world-frame vector. The policy always sees itself
                # reaching with its LEFT hand.
                q_n = np.concatenate([MIRROR_SIGN * q_n[3:6], MIRROR_SIGN * q_n[0:3]])
                qd_s = np.concatenate([MIRROR_SIGN * qd_s[3:6], MIRROR_SIGN * qd_s[0:3]])
                vec_active = vec_active * np.array([1.0, -1.0, 1.0])
                vec_support = vec_support * np.array([1.0, -1.0, 1.0])
                base_up = base_up * np.array([1.0, -1.0, 1.0])
                active_is_left = 1.0
            obs = np.concatenate([q_n, qd_s, vec_active, vec_support, base_z, base_up,
                                  [f_active, f_support, active_is_left]])
            return obs.astype(np.float32)

        vec_r = self.data.site_xpos[self.r_site] - self.data.site_xpos[self.r_hold_site]
        vec_l = self.data.site_xpos[self.l_site] - self.data.site_xpos[self.l_hold_site]
        cr = min(self._hook_force(self.r_hook_gids), 50.0) / 50.0
        cl = min(self._hook_force(self.l_hook_gids), 50.0) / 50.0
        obs = np.concatenate([q_n, qd_s, vec_r, vec_l, base_z, base_up, [cr, cl]])
        return obs.astype(np.float32)

    def _force_on(self, hook_gids, target_gids):
        """Normal contact-force magnitude between hook geoms and a specific target geom set."""
        hset = set(hook_gids)
        total = 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in hset and g2 in target_gids) or (g2 in hset and g1 in target_gids):
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                total += abs(f6[0])
        return total

    def _uplift_on(self, hook_gids, target_gids=None):
        """Signed UPWARD (world +z) contact force exerted by rungs ON the hook geoms.
        This is the load-bearing measure of a hold: a proper cradled grip supporting the
        body weight reads strongly positive, while a sideways poke / "fistbump" against
        the rung reads ~0 (its contact normal is horizontal). target_gids=None -> any hold."""
        hset = set(hook_gids)
        tset = self.hold_gids if target_gids is None else set(target_gids)
        total = 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            # sign convention calibrated at the stable hang: total uplift across both
            # hooks equals body weight (+30 N), so positive = rung supporting the hook
            if g1 in hset and g2 in tset:
                sign = -1.0
            elif g2 in hset and g1 in tset:
                sign = 1.0
            else:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, f6)
            frame = np.array(c.frame).reshape(3, 3)   # rows = contact-frame axes in world coords
            f_world = frame.T @ f6[:3]                # contact-frame force -> world frame
            total += sign * float(f_world[2])
        return total

    def _start_seated(self):
        """Reset-time check: BOTH hooks load-bearing (>=15% W each) and cradled on their
        lowest rungs after the settle. A start that fails this is a fake grip."""
        thr = 0.15 * self.body_weight
        for side, hooks, site in (("left", self.l_hook_gids, self.l_site),
                                  ("right", self.r_hook_gids, self.r_site)):
            r = self._tm_sorted(side)[0]
            if self._uplift_on(hooks) < thr or not self._cradled(site, r["sid"]):
                return False
        return True

    def _cradled(self, grip_site, target_site):
        """Geometric check that the rung sits INSIDE the hook cradle: grip site within a
        tight x-z window of the rung center and at/above it (hooked over the top, between
        hook and catch bar) — not touching the side or the underside."""
        g = self.data.site_xpos[grip_site]
        t = self.data.site_xpos[target_site]
        dx = float(g[0] - t[0])
        dz = float(g[2] - t[2])
        # Mesh holds: the grippable lip is only ~±0.03 m wide in y (flanked by raised
        # horns); measured uplift dies at |dy|>=0.05, so gate the grab on |dy|<0.04.
        # Cylinder rungs span ±0.09 m in y — no lateral check needed there.
        y_ok = (abs(float(g[1] - t[1])) < 0.04) if self.mesh_holds else True
        return (abs(dx) < 0.03) and (-0.005 <= dz <= 0.035) and y_ok

    def _hook_force(self, hook_gids):
        """Total normal contact-force magnitude between the given hook geoms and the HOLDS.
        Contact with the flat wall does NOT count, so the policy must reach a hold."""
        hset = set(hook_gids)
        total = 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            hook_hit = (g1 in hset and g2 in self.hold_gids) or (g2 in hset and g1 in self.hold_gids)
            if hook_hit:
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                total += abs(f6[0])  # normal component in contact frame
        return total

    def _reward(self, action):
        d_r, d_l = self._distances()
        fr = self._hook_force(self.r_hook_gids)
        fl = self._hook_force(self.l_hook_gids)
        engaged_r = fr > self.contact_thresh
        engaged_l = fl > self.contact_thresh

        # distance shaping (negative, bounded)
        r_dist = -self.w_dist * (min(d_r, 1.0) + min(d_l, 1.0))
        # potential-based progress (closing the gap)
        prog = (self._prev_dr - d_r) + (self._prev_dl - d_l)
        r_prog = self.w_progress * prog
        self._prev_dr, self._prev_dl = d_r, d_l
        # contact engagement (clipped so it can't dominate)
        r_contact = self.w_contact * (min(fr, 20.0) / 20.0 + min(fl, 20.0) / 20.0)
        # upright: torso local-z should point "up the wall" (world +z-ish)
        up = self.data.xmat[self.base_bid].reshape(3, 3)[:, 2]
        r_up = self.w_upright * float(up[2])
        # control penalty + alive
        r_ctrl = -self.w_ctrl * float(np.sum(action ** 2))
        reward = r_dist + r_prog + r_contact + r_up + r_ctrl + self.alive_bonus

        success = (engaged_r and engaged_l and d_r < self.grasp_dist and d_l < self.grasp_dist)
        if success:
            reward += self.success_bonus

        info = {
            "d_r": d_r, "d_l": d_l, "force_r": fr, "force_l": fl,
            "engaged_r": bool(engaged_r), "engaged_l": bool(engaged_l),
            "r_dist": r_dist, "r_prog": r_prog, "r_contact": r_contact, "r_up": r_up,
        }
        return reward, info

    def _climb_reward(self, action, base_z):
        """Hand-over-hand gait. The ACTIVE hand is rewarded for reaching its next same-side
        rung; the SUPPORT hand must keep a firm grip. When the active hand grasps its next
        rung AND the support hand is firm, the gait advances (big reward) and the hands
        swap roles: left -> right -> left ... climbing up the alternating rungs."""
        ah, as_, at, atg, sh, ss, sg = self._active_parts()
        d_active = float(np.linalg.norm(self.data.site_xpos[as_] - self.data.site_xpos[at]))
        # LOAD-BEARING measures: upward force the rung exerts on each hook. This is what
        # distinguishes a real hold (carrying body weight) from a touch ("fistbump").
        f_active = self._uplift_on(ah, atg)    # weight carried by active hook on its TARGET rung
        f_support = self._uplift_on(sh)        # weight carried by support hook on any rung
        support_firm = f_support > self.firm_thresh
        # how far the support hand has drifted from the rung it is supposed to hold
        if self.treadmill:
            sup_site = self._tm_sorted("right" if self.active == "left" else "left")[0]["sid"]
        else:
            sup_site = self.right_sites[self.ri] if self.active == "left" else self.left_sites[self.li]
        d_support = float(np.linalg.norm(self.data.site_xpos[ss] - self.data.site_xpos[sup_site]))
        # "seated" = the hook came OVER the rung (grip at/above it), not poking the underside.
        active_z = float(self.data.site_xpos[as_][2])
        target_z = float(self.data.site_xpos[at][2])
        seated = active_z >= (target_z - 0.02)
        # "cradled" = the rung sits INSIDE the hook cradle (tight x-z window), i.e. a real
        # hooked-over grip rather than a frontal "fistbump" against the rung.
        cradled = self._cradled(as_, at)

        # ALL dense guidance is POTENTIAL-BASED (reward = Φ(s') − Φ(s)). Sitting in any
        # state — including hovering with the hook pressed on the rung — nets ~0 per step,
        # so the ONLY way to accumulate reward is to actually complete moves. (A previous
        # per-step version of these terms taught the policy to park at the rung and farm
        # the dense payouts forever instead of grabbing it.)
        dx = float(self.data.site_xpos[as_][0] - self.data.site_xpos[at][0])
        dzc = active_z - target_z - 0.015          # ideal grip site ~15 mm above rung center
        dy = float(self.data.site_xpos[as_][1] - self.data.site_xpos[at][1])
        # mesh pocket is ~±0.03 m wide: tighten the lateral shaping so the policy is
        # actually pulled toward the pocket CENTER, not merely near the hold
        y_sig = 0.035 if self.mesh_holds else 0.06
        cradle_gauss = float(np.exp(-((dx / 0.03) ** 2 + (dzc / 0.03) ** 2 + (dy / y_sig) ** 2)))
        load = float(np.clip(f_active / max(self.advance_force * 2.0, 1e-6), 0.0, 1.0)) \
            * (1.0 if cradled else 0.0)
        # support-arm pull-up: retraction of the SUPPORT prismatic hauls the torso up and
        # brings the next rung into reach; only counts while that hand is bearing load.
        # active=="left" -> support is the RIGHT arm -> r3_1 (index 2), and vice versa.
        sup_prism = 2 if self.active == "left" else 5
        prism_lo, prism_hi = self.jnt_range[sup_prism]
        retract = 1.0 - (float(self.data.qpos[self.qpos_adr[sup_prism]]) - prism_lo) \
            / max(prism_hi - prism_lo, 1e-6)
        pullup = retract * (1.0 if support_firm else 0.0)
        # WEIGHT TRANSFER: climbing requires the support hook to carry the body so the
        # active hook can let go. Load on the support hook is rewarded; load still on the
        # active hook ANYWHERE (its old rung included) blocks the move and is charged.
        # Without these the policy pulls to max height hanging off the REACHING hand and
        # deadlocks (support hook floats unloaded in the cradle; observed failure mode).
        half_w = 0.5 * self.body_weight
        sup_load = float(np.clip(f_support / half_w, 0.0, 1.0))
        act_load_other = float(np.clip((self._uplift_on(ah) - max(f_active, 0.0)) / half_w, 0.0, 1.0))
        # NATURAL CADENCE: when the next rung is still far, weight pull-up more than
        # reach; once close, weight cradle insertion. Humans pull the body up, THEN
        # place the free hand — not a long stretch from a low hang.
        w_reach, w_pull = 8.0, 1.5
        if self.natural_form:
            far = float(np.clip(d_active / 0.35, 0.0, 1.0))
            w_reach = 6.0 + 2.0 * (1.0 - far)   # softer reach when far
            w_pull = 1.5 + 2.5 * far            # insist on pull-up before long reach
        # keep torso near the wall plane (hang starts ~x=-0.05); large lean-back is ugly
        wall_prox = 0.0
        if self.natural_form:
            base_x = float(self.data.xpos[self.base_bid][0])
            wall_prox = -float(np.clip(abs(base_x + 0.05) - 0.08, 0.0, 0.25))
        phi = (-w_reach * min(d_active, 0.6)  # reach the next rung
               + 3.0 * cradle_gauss           # get the rung INTO the hook cradle (over the top)
               + 4.0 * load                   # transfer real weight onto the new hold
               + w_pull * pullup              # pull all the way up with the support arm
               + 3.0 * sup_load               # hang off the SUPPORT hook...
               - 2.0 * act_load_other         # ...not off the hand that must move next
               + 4.0 * base_z                 # gain height
               + 2.0 * wall_prox)             # stay close to the wall (natural_form)
        r_shape = phi - self._prev_phi if self._prev_phi is not None else 0.0
        self._prev_phi = phi
        self._prev_active_d = d_active
        self._prev_z = base_z

        # Per-step PENALTIES (safe: they can only be avoided, not farmed).
        # support grip: hard gate if not load-bearing + dense pull keeping the support hand
        # ON its rung (otherwise the policy dangles it and hangs off one hook)
        r_support_gate = 0.0 if support_firm else -2.0
        r_support_hold = -6.0 * min(d_support, 0.3)
        # ANTI-SKIP: active hand should not rise far above its target rung. Margin 0.06 so
        # the up-and-over hooking motion (swings ~0.05 above before dropping in) stays free.
        r_overshoot = -12.0 * max(0.0, active_z - (target_z + 0.06))
        # UPRIGHTNESS: penalize tilting away from the STARTING (vertical) attitude.
        # tilt = 1 - cos(angle to start orientation): 0 when vertical, 1 at 90 deg.
        # Quadratic in tilt: gentle near vertical (normal climbing sway is free) but
        # increasingly expensive for the big swings, plus damping on body angular rate.
        up = self.data.xmat[self.base_bid].reshape(3, 3)[:, 2]
        tilt = 1.0 - float(np.dot(up, self._start_up))
        base_dof = self.model.body_dofadr[self.base_bid]
        ang_speed = float(np.linalg.norm(self.data.qvel[base_dof + 3:base_dof + 6]))
        if self.natural_form:
            # tighter attitude: ~14° mean tilt was "ok but swayy"; push toward quieter climbs
            r_up = -4.0 * tilt - 8.0 * tilt * tilt - 0.06 * ang_speed
        else:
            r_up = -2.0 * tilt - 3.0 * tilt * tilt - 0.02 * ang_speed
        # LEVEL SHOULDERS (climbing form): the paired shoulder joints (r1/l1, r2/l2) should
        # stay at the same height — this specifically punishes base ROLL, which the tilt
        # term alone under-weights. Anchors are level by construction when unrolled.
        dz_sh1 = float(self.data.xanchor[self.jnt_ids[0]][2] - self.data.xanchor[self.jnt_ids[3]][2])
        dz_sh2 = float(self.data.xanchor[self.jnt_ids[1]][2] - self.data.xanchor[self.jnt_ids[4]][2])
        level_w = 16.0 if self.natural_form else 10.0
        r_level = -level_w * (abs(dz_sh1) + abs(dz_sh2))
        r_ctrl = -0.002 * float(np.sum(action ** 2))
        # SMOOTHNESS: penalize action jerk (Δa). Makes moves look less twitchy without
        # changing the potential landscape — only farmable by staying still (which loses
        # to time_penalty + move bonuses).
        r_jerk = 0.0
        if self.natural_form:
            da = action - self._prev_action
            r_jerk = -0.01 * float(np.sum(da ** 2))
        self._prev_action = np.asarray(action, dtype=np.float64).copy()
        reward = (r_shape + r_support_gate + r_support_hold + r_overshoot + r_up
                  + r_level + r_ctrl + r_jerk - self.time_penalty)

        # (7) successful hand-over-hand move: the FULL gate must hold for `secure_steps`
        #     consecutive control steps. "Good hold" = rung inside the cradle (geometry)
        #     AND the hook is carrying real weight (uplift), AND the support hand is also
        #     load-bearing. A single-step graze or a fistbump can never advance the gait.
        gate = (d_active < self.advance_grasp_dist and f_active > self.advance_force
                and support_firm and cradled)
        self._secure = self._secure + 1 if gate else 0
        if self._secure >= self.secure_steps:
            self._secure = 0
            if self.treadmill:
                # active hand just grabbed its next rung; recycle that side's lowest rung
                # (the one it left) up to the top so there's never a rung below the body.
                self._tm_recycle(self.active)
                self.active = "right" if self.active == "left" else "left"
            elif self.active == "left":
                self.li = min(self.li + 1, len(self.left_sites) - 1); self.active = "right"
            else:
                self.ri = min(self.ri + 1, len(self.right_sites) - 1); self.active = "left"
            self.moves_done += 1
            reward += 40.0                                    # per successful rung move
            reward += 15.0 * max(0.0, base_z - self._max_z)   # bonus for net height gained
            self._prev_active_d = self._active_target_dist()  # re-anchor for the new active hand
            self._prev_phi = None   # hand swap changes Φ discontinuously; re-anchor shaping
        self._max_z = max(self._max_z, base_z)

        fr = self._hook_force(self.r_hook_gids); fl = self._hook_force(self.l_hook_gids)
        # d_r = active-hand reach distance, d_l = support-hand hold distance (diagnostics)
        success_moves = self.max_moves if self.max_moves > 0 else 4
        info = {"d_r": d_active, "d_l": d_support, "force_r": fr, "force_l": fl,
                "base_z": base_z, "height_gain": base_z - self._start_z,
                "active": self.active, "moves_done": self.moves_done,
                "support_firm": bool(support_firm),
                "tilt": tilt, "ang_speed": ang_speed,
                "engaged_r": bool(fr > self.contact_thresh), "engaged_l": bool(fl > self.contact_thresh),
                "is_success": self.moves_done >= success_moves}
        return reward, info

    # ---------- rendering ----------
    def render(self):
        if self.viewer is None:
            import mujoco.viewer  # submodule isn't auto-imported by `import mujoco`
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
