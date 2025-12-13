"""
climbbot_env.py

Gymnasium environment for the climbbot MuJoCo model.

Usage example:
    python -m climbbot_env        # runs a simple random demo loop

Recommended training:
    Use a continuous control RL algorithm (PPO, SAC) to output 6 actions:
      [r1_target, r2_target, r3_target, l1_target, l2_target, l3_target]

Dependencies:
    pip install mujoco gymnasium numpy

If you have stable-baselines3, use it to train (PPO).
"""
import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import mujoco
    from mujoco import MjSim, MjRenderContextOffscreen, MjViewer, load_model_from_path
except Exception as e:
    raise RuntimeError("This environment requires the official 'mujoco' python bindings. "
                       "Install mujoco >= 2.3 and the python package. Error: " + str(e))


class ClimbBotEnv(gym.Env):
    """
    Gym-style environment wrapping your MuJoCo model.
    Action space: 6 continuous values in [-1,1] mapped to joint target ranges for:
        r1, r2, r3_1 (linear), l1, l2, l3_1 (linear)
    Observation: concatenation of:
        - joint positions and velocities for these 6 joints
        - world positions of r_grip_site and l_grip_site (6 values)
        - world positions of hold sites (hold_1..hold_6) flattened (6 * 3)
        - base site pos (3) and base z velocity (1)
    """

    def __init__(self, model_path="climbbot.xml", frame_skip=10, max_steps=1000, render=False):
        super().__init__()
        assert os.path.exists(model_path), f"Model XML not found: {model_path}"

        self.model = load_model_from_path(model_path)
        self.sim = MjSim(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_mode = render

        # List of actuators we will control (names from your xml).
        self.ctrl_names = ["r1_ctrl", "r2_ctrl", "r3_1_ctrl", "l1_ctrl", "l2_ctrl", "l3_1_ctrl"]
        # Map to actuator indices
        self.ctrl_idxs = [self.model.actuator_name2id(name) for name in self.ctrl_names]

        # Joints that correspond to these actuators (conventionally)
        # We'll expose joint positions and velocities for: r1,r2,r3_1,l1,l2,l3_1
        joint_names = ["r1", "r2", "r3_1", "l1", "l2", "l3_1"]
        self.joint_qpos_idxs = [self.model.joint_name2id(j) for j in joint_names]  # indices in model joint list
        # But to read qpos we need the offset in sim.data.qpos: use joint_qpos_addr
        # We'll build helpers to read by name.

        # Sites of interest
        self.r_grip_site = "r_grip_site"
        self.l_grip_site = "l_grip_site"
        self.base_site = "base_site"
        self.hold_sites = [f"hold_{i}_site" for i in range(1, 7)]  # hold_1_site ... hold_6_site

        # Action / observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        obs_dim = 0
        # 6 joint pos + 6 joint vel
        obs_dim += 12
        # 2 grips * 3 coords
        obs_dim += 6
        # 6 holds * 3 coords
        obs_dim += 6 * 3
        # base pos (3) and base z vel (1)
        obs_dim += 4

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Joint target ranges to map [-1,1] actions -> actual ctrl values (these match your joint ctrlrange)
        # For revolute joints r1,r2,l1,l2 range [-pi/2, pi/2]; linear joints 0..0.134
        self.joint_ctrl_ranges = [
            (-1.5708, 1.5708),  # r1
            (-1.5708, 1.5708),  # r2
            (0.0, 0.134),       # r3_1 (linear)
            (-1.5708, 1.5708),  # l1
            (-1.5708, 1.5708),  # l2
            (0.0, 0.134),       # l3_1 (linear)
        ]

        # bookkeeping
        self.step_count = 0

        # Render context if requested
        self.viewer = None
        if self.render_mode:
            self.viewer = MjViewer(self.sim)

        # Convenience caches: model indices for sites and geoms
        self.site_ids = {name: self.model.site_name2id(name) for name in self.model.site_names}
        # For contact checking, map gripper collision geom names (the xml uses many; we'll treat all geoms whose name contains "gripper" or "assembly_12_collision_1" etc.)
        # Simpler: we'll check contact pairs and look for contacts involving the hold meshes (hold_* geoms are in bodies named hold_1..hold_6).
        self.hold_geom_ids = []
        for i, gname in enumerate(self.model.geom_names):
            if ("wall_part" in gname) or ("hold" in gname and "hold_" in gname) or ("hold_material" in gname):
                # This is heuristic (names depend on xml). Instead, find geoms that are children of hold bodies:
                pass
        # Instead of heuristics, create list of hold geom ids by searching bodies named hold_1..hold_6
        self.hold_geom_ids = []
        for bname in [f"hold_{i}" for i in range(1, 7)]:
            if bname in self.model.body_names:
                bid = self.model.body_name2id(bname)
                # collect geoms with this body id
                for gid, gbody in enumerate(self.model.geom_bodyid):
                    if int(gbody) == int(bid):
                        self.hold_geom_ids.append(gid)

        # Gripper geom ids -- find geoms on assembly_12 and assembly_11 branches containing 'gripper'
        self.gripper_geom_ids = []
        for gid, gname in enumerate(self.model.geom_names):
            if "gripper" in gname or "gripper_hook" in gname or "flat_gripper" in gname:
                self.gripper_geom_ids.append(gid)
        # If we found none, fallback to some known names from the XML
        if len(self.gripper_geom_ids) == 0:
            for name in self.model.geom_names:
                if "part_6" in name or "assembly_12" in name or "assembly_11" in name:
                    self.gripper_geom_ids.append(self.model.geom_name2id(name))

        # Small helper for qpos indexing
        self.joint_qpos_addr = {}
        for jname in self.model.joint_names:
            # qpos_adr maps to where this joint's qpos starts in sim.data.qpos
            jid = self.model.joint_name2id(jname)
            qpos_adr = self.model.jnt_qposadr[jid]
            self.joint_qpos_addr[jname] = qpos_adr

        # keep track of which holds are currently contacted by each gripper
        self.r_grasp_hold = None
        self.l_grasp_hold = None

        self.reset()

    # ---------- Utility helpers ----------
    def _get_site_worldpos(self, site_name):
        """Return world coordinates (x,y,z) of a site."""
        sid = self.model.site_name2id(site_name)
        # model.site_pos is local pos; but sim.data.site_xpos has world positions
        return self.sim.data.site_xpos[sid].copy()

    def _get_base_pos(self):
        sid = self.model.site_name2id(self.base_site)
        return self.sim.data.site_xpos[sid].copy()

    def _get_joint_qpos(self, joint_name):
        adr = self.joint_qpos_addr[joint_name]
        # For hinge joints qpos is scalar, for slides scalar (both occupy 1)
        return float(self.sim.data.qpos[adr])

    def _get_joint_qvel(self, joint_name):
        # need qveladr
        jid = self.model.joint_name2id(joint_name)
        adr = self.model.jnt_dofadr[jid]
        return float(self.sim.data.qvel[adr])

    def _set_actuator_targets(self, targets):
        """Map 6-element targets (in joint ranges) into sim.data.ctrl for actuators."""
        for i, idx in enumerate(self.ctrl_idxs):
            self.sim.data.ctrl[idx] = float(targets[i])

    def _action_to_targets(self, action):
        """Map normalized action in [-1,1] to the real joint targets."""
        targets = []
        for a, (lo, hi) in zip(action, self.joint_ctrl_ranges):
            # scale
            tgt = (a + 1.0) / 2.0 * (hi - lo) + lo
            targets.append(tgt)
        return np.array(targets, dtype=np.float32)

    def _step_sim(self):
        # step the sim forward frame_skip times
        for _ in range(self.frame_skip):
            self.sim.step()

    def _compute_contacts(self):
        """Return set of (geom1, geom2) contacts for current step."""
        contacts = []
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            contacts.append((int(c.geom1), int(c.geom2)))
        return contacts

    def _check_grasp_contacts(self):
        """
        Check contacts between gripper geoms and hold geoms.
        Return (r_hold_idx or None, l_hold_idx or None).
        """
        r_found = None
        l_found = None
        contacts = self._compute_contacts()
        for (g1, g2) in contacts:
            # if g1 is gripper and g2 in hold geoms (or vice versa)
            if g1 in self.gripper_geom_ids and g2 in self.hold_geom_ids:
                # figure out if this gripper is left or right by proximity to site positions
                pos = self.sim.data.geom_xpos[g1]
                rpos = self._get_site_worldpos(self.r_grip_site)
                lpos = self._get_site_worldpos(self.l_grip_site)
                # distance
                if np.linalg.norm(pos - rpos) < np.linalg.norm(pos - lpos):
                    r_found = self.hold_geom_ids.index(g2) if g2 in self.hold_geom_ids else None
                else:
                    l_found = self.hold_geom_ids.index(g2) if g2 in self.hold_geom_ids else None
            if g2 in self.gripper_geom_ids and g1 in self.hold_geom_ids:
                pos = self.sim.data.geom_xpos[g2]
                rpos = self._get_site_worldpos(self.r_grip_site)
                lpos = self._get_site_worldpos(self.l_grip_site)
                if np.linalg.norm(pos - rpos) < np.linalg.norm(pos - lpos):
                    r_found = self.hold_geom_ids.index(g1) if g1 in self.hold_geom_ids else None
                else:
                    l_found = self.hold_geom_ids.index(g1) if g1 in self.hold_geom_ids else None
        return r_found, l_found

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        self.step_count = 0
        # randomize initial joint positions slightly
        self.sim.reset()
        # set qpos to default (let xml define), then small randomization of arm joints
        for jname in ["r1", "r2", "r3_1", "l1", "l2", "l3_1"]:
            q = self._get_joint_qpos(jname)
            # small jitter
            jitter = (np.random.rand() - 0.5) * 0.2
            adr = self.joint_qpos_addr[jname]
            self.sim.data.qpos[adr] = q + jitter

        # zero velocities
        self.sim.data.qvel[:] = 0.0
        # zero controls
        self.sim.data.ctrl[:] = 0.0

        # forward kinematics update
        self.sim.forward()
        self.r_grasp_hold = None
        self.l_grasp_hold = None

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        obs = []
        # joint pos + vel
        for jname in ["r1", "r2", "r3_1", "l1", "l2", "l3_1"]:
            obs.append(self._get_joint_qpos(jname))
        for jname in ["r1", "r2", "r3_1", "l1", "l2", "l3_1"]:
            obs.append(self._get_joint_qvel(jname))
        # grip world positions
        obs.extend(self._get_site_worldpos(self.r_grip_site).tolist())
        obs.extend(self._get_site_worldpos(self.l_grip_site).tolist())
        # holds world positions
        for hs in self.hold_sites:
            obs.extend(self._get_site_worldpos(hs).tolist())
        # base pos + base z vel
        basepos = self._get_base_pos()
        obs.extend(basepos.tolist())
        # base z vel (approx from sim.data.qvel of base? we'll use site linear velocity)
        # There is a velocimeter sensor named base_site_vel in xml; read from sim.data.sensordata if present
        vel_z = 0.0
        try:
            # find sensor index by name
            s_idx = list(self.model.sensor_names).index("base_site_vel")
            vel_z = float(self.sim.data.sensordata[s_idx][2]) if self.sim.data.sensordata.shape[0] > s_idx else 0.0
        except Exception:
            # fallback: compute finite difference approx from sim data (not necessary on reset)
            vel_z = 0.0
        obs.append(vel_z)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        targets = self._action_to_targets(action)
        # set as actuator targets (position actuators)
        self._set_actuator_targets(targets)

        # Step the sim
        self._step_sim()

        # simulate forward kinematics updated already
        self.sim.forward()

        # Update grasp detection via contacts
        r_found, l_found = self._check_grasp_contacts()
        # Convert to hold indexes (1..6) or None
        self.r_grasp_hold = (r_found + 1) if r_found is not None else None
        self.l_grasp_hold = (l_found + 1) if l_found is not None else None

        obs = self._get_obs()

        # Reward shaping:
        reward = 0.0
        done = False
        info = {}

        # encourage upward progress of base (climb)
        # base_z = self._get_base_pos()[2]
        # small reward for upward movement relative to previous step (approx)
        # store previous base_z in info or self._prev_base_z
        # prev_z = getattr(self, "_prev_base_z", base_z)
        # reward += 50.0 * max(0.0, base_z - prev_z)  # reward upward motion
        # self._prev_base_z = base_z

        # encourage closeness of grips to nearest holds
        rpos = self._get_site_worldpos(self.r_grip_site)
        lpos = self._get_site_worldpos(self.l_grip_site)
        hold_positions = [self._get_site_worldpos(s) for s in self.hold_sites]

        # distance-based penalty (smaller when closer)
        def min_dist_to_holds(pos):
            dists = [np.linalg.norm(pos - h) for h in hold_positions]
            return min(dists) if len(dists) > 0 else 1.0

        # rdist = min_dist_to_holds(rpos)
        # ldist = min_dist_to_holds(lpos)
        # reward -= 1.0 * (rdist + ldist)

        # grasp reward bonuses
        # if self.r_grasp_hold is not None:
        #     reward += 10.0
        # if self.l_grasp_hold is not None:
        #     reward += 10.0

        # big success reward if both grippers are holding different holds and base has climbed past a threshold
        # if (self.r_grasp_hold is not None) and (self.l_grasp_hold is not None) and (self.r_grasp_hold != self.l_grasp_hold):
        #     # encourage using two hands
        #     reward += 25.0
        #     # if base z above top-most hold height -> big success
        #     top_hold_z = max([h[2] for h in hold_positions])
        #     if base_z > top_hold_z - 0.2:
        #         reward += 200.0
        #         done = True
        #         info["success"] = True

        
        contact_bonus = 0.0
        if self.r_grip_geom_id != -1:
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if c.geom[0] == self.r_grip_geom_id or c.geom[1] == self.r_grip_geom_id:
                    f6 = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    frame = np.array(c.frame).reshape(3, 3)
                    f_world = frame @ f6[:3]
                    contact_bonus += f_world[2]
                    info["r_grip_contact_force"] = f_world[2]
                    print("Right gripper contact force:", f_world[2])
                    break
            
            
        reward += contact_bonus

        # negative reward for falling / base below floor
        if base_z < 0.1:
            reward -= 50.0
            done = True
            info["fell"] = True
            

        # time termination
        if self.step_count >= self.max_steps:
            done = True
            info["TimeLimit.truncated"] = True

        return obs, float(reward), done, False, info

    def render(self, mode='human'):
        if self.render_mode:
            if self.viewer is None:
                self.viewer = MjViewer(self.sim)
            self.viewer.render()
        else:
            # create a short-lived offscreen context to render an image
            ctx = MjRenderContextOffscreen(self.sim, 0)
            ctx.render(width=640, height=480)
            img = ctx.read_pixels(640, 480, depth=False)[0]
            return img

    def close(self):
        if self.viewer is not None:
            # viewer doesn't need explicit close in mujoco python bindings
            self.viewer = None


if __name__ == "__main__":
    # Simple random policy demo.
    env = ClimbBotEnv(model_path="climbbot.xml", frame_skip=10, max_steps=500, render=True)
    obs, _ = env.reset()
    for t in range(1000):
        a = env.action_space.sample()
        obs, r, done, truncated, info = env.step(a)
        env.render()
        if done:
            print("Done:", info)
            obs, _ = env.reset()
    env.close()
