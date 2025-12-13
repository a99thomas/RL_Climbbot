# IK_both_arms_reload_on_reset.py
"""
Dual-arm IK launcher with reload-on-reset.

- Uses numeric Jacobian + damped pseudoinverse (LM-style fixed-damping step).
- Controls both arms each IK cycle, applying both arms' commands together.
- Reloads MJCF from disk when viewer Reset is clicked.
"""

import numpy as np
import time
import os
import mujoco
import mujoco.viewer

# ---------------- USER CONFIG ----------------
XML_FILE_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene.xml"

SIMULATION_SPEED = 1.0

# Per-arm target lists (world coordinates). Adjust as needed.
TARGETS_R = [
    np.array([0.55, 0.03, 0.3]),
    np.array([0.45, 0.03, 0.2]),
    np.array([0.3, -0.43, 0.4]),
]
TARGETS_L = [
    np.array([0.55, 0.13, 0.3]),  # example left-arm targets (edit to suit your scene)
    np.array([0.45, 0.23, 0.2]),
    np.array([0.5, 0.0, 0.3]),
]

# IK / runtime params
USE_ACTUATORS = True             # True: command position actuators via data.ctrl; False: write data.qpos directly
IK_UPDATE_EVERY_N_STEPS = 3
FD_EPS = 1e-6
DAMPING = 0.0
STEP_SCALE = 0.5
TOL = 0.02                       # meters (coarse); tune smaller for precision
MAX_ITERS = 800
POSE_HOLD_SECONDS = 2.0
# ----------------------------------------------

# ---------- helper functions (same as before) ----------
def get_arm_joint_names(arm):
    return ["r1", "r2", "r3_1"] if arm == "right" else ["l1", "l2", "l3_1"]

def qpos_indices_for_joint_names(model, joint_names):
    idxs = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid == -1:
            raise RuntimeError(f"Joint name '{jn}' not found in model")
        idxs.append(int(model.jnt_qposadr[jid]))
    return idxs

def get_joint_limits_for_joint_names(model, joint_names):
    qpos_addrs = []
    limits = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid == -1:
            raise RuntimeError(f"Joint '{jn}' not found in model")
        adr = int(model.jnt_qposadr[jid])
        qpos_addrs.append(adr)
        limited = bool(model.jnt_limited[jid]) if hasattr(model, "jnt_limited") else True
        if limited:
            qmin = float(model.jnt_range[jid, 0])
            qmax = float(model.jnt_range[jid, 1])
        else:
            qmin = -np.inf
            qmax = np.inf
        limits.append((qmin, qmax))
    return qpos_addrs, limits

def clamp_qpos_by_limits(qpos, qpos_addrs, limits):
    qpos_new = qpos.copy()
    for addr, (qmin, qmax) in zip(qpos_addrs, limits):
        if np.isfinite(qmin) or np.isfinite(qmax):
            qpos_new[addr] = float(np.clip(qpos_new[addr], qmin, qmax))
    return qpos_new

def clamp_delta_by_limits(qpos, delta_q, qpos_addrs, limits):
    clipped = delta_q.copy()
    for i, addr in enumerate(qpos_addrs):
        qmin, qmax = limits[i]
        if not (np.isfinite(qmin) or np.isfinite(qmax)):
            continue
        cur = qpos[addr]
        proposed = cur + clipped[i]
        clipped[i] = float(np.clip(proposed, qmin, qmax) - cur)
    return clipped

def forward_site_pos(model, data, site_id):
    mujoco.mj_forward(model, data)
    return data.site_xpos[site_id].copy()

def numeric_jacobian(model, data, site_id, qpos_indices, eps=1e-6):
    base_qpos = data.qpos.copy()
    f0 = forward_site_pos(model, data, site_id)
    n = len(qpos_indices)
    J = np.zeros((3, n))
    for i, qi in enumerate(qpos_indices):
        dq = np.zeros_like(base_qpos)
        dq[qi] = eps
        data.qpos[:] = base_qpos + dq
        mujoco.mj_forward(model, data)
        f1 = forward_site_pos(model, data, site_id)
        J[:, i] = (f1 - f0) / eps
    data.qpos[:] = base_qpos
    mujoco.mj_forward(model, data)
    return J

def damped_pinv(J, damping):
    JTJ = J.T @ J
    n = JTJ.shape[0]
    A = JTJ + (damping**2) * np.eye(n)
    try:
        pinv = np.linalg.solve(A, J.T)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(A) @ J.T
    return pinv

# ---------- model/load helper ----------
def load_model_and_setup(xml_path, use_actuators):
    """Load mjcf, create model/data and compute useful indices & limits for both arms."""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML not found at '{xml_path}'")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # site ids
    r_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_grip_site")
    l_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "l_grip_site")

    # per-arm joint info
    info = {}
    for arm, site_id in (("right", r_site_id), ("left", l_site_id)):
        joint_names = get_arm_joint_names(arm)
        qpos_indices = qpos_indices_for_joint_names(model, joint_names)
        qpos_addrs, limits = get_joint_limits_for_joint_names(model, joint_names)
        actuator_indices = None
        if use_actuators:
            actuator_indices = []
            for jn in joint_names:
                aname = jn + "_ctrl"
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
                if aid == -1:
                    raise RuntimeError(f"Actuator '{aname}' not found; update actuator map or set USE_ACTUATORS=False")
                actuator_indices.append(int(aid))
        info[arm] = {
            "site_id": site_id,
            "joint_names": joint_names,
            "qpos_indices": qpos_indices,
            "qpos_addrs": qpos_addrs,
            "limits": limits,
            "actuator_indices": actuator_indices
        }

    # warm forward
    mujoco.mj_forward(model, data)
    return model, data, info

# ---------------- main outer loop that supports reload-on-reset ----------------
print("Starting dual-arm launcher. MJCF:", XML_FILE_PATH)
reload_count = 0

# outer reload loop
while True:
    # load or reload model & info
    try:
        model, data, arms_info = load_model_and_setup(XML_FILE_PATH, USE_ACTUATORS)
    except Exception as e:
        print("Error loading model:", e)
        raise

    # site ids
    r_site_id = arms_info["right"]["site_id"]
    l_site_id = arms_info["left"]["site_id"]

    # prepare per-arm IK state
    arm_state = {
        "right": {
            "targets": [t.astype(np.float64) for t in TARGETS_R],
            "target_idx": 0,
            "hold_until": None,
            "ik_iter": 0,
        },
        "left": {
            "targets": [t.astype(np.float64) for t in TARGETS_L],
            "target_idx": 0,
            "hold_until": None,
            "ik_iter": 0,
        }
    }

    # if using actuators, prepare ctrl vector
    if USE_ACTUATORS:
        ctrl_vector = np.zeros(model.nu)

    print(f"[run #{reload_count}] Model loaded. Actuators? {USE_ACTUATORS}")
    reload_count += 1

    last_sim_time = 0.0
    reset_was_requested = False
    physics_step_counter = 0

    # open passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched. Close window to exit. Click Reset to reload the MJCF file.")
        start_time = time.time()

        while viewer.is_running():
            # detect reset
            if data.time < last_sim_time:
                print("Detected viewer reset -> will reload MJCF from disk.")
                reset_was_requested = True
                break
            last_sim_time = data.time

            # real-time sync
            elapsed_real_time = time.time() - start_time
            target_real_time = data.time / SIMULATION_SPEED
            time_to_wait = target_real_time - elapsed_real_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

            # step physics
            mujoco.mj_step(model, data)
            physics_step_counter += 1

            # perform IK updates every N steps
            if physics_step_counter % IK_UPDATE_EVERY_N_STEPS == 0:
                # collect per-arm info and prepare to compute deltas
                # (we compute both Jacobians on the same base state)
                apply_updates = {"right": False, "left": False}
                deltas = {"right": None, "left": None}
                target_qpos_proposals = {"right": None, "left": None}
                ctrl_updates = {}  # actuator index -> value

                # loop arms to decide whether to compute an IK step
                for arm in ("right", "left"):
                    info = arms_info[arm]
                    state = arm_state[arm]
                    if state["target_idx"] >= len(state["targets"]):
                        continue  # no more targets for this arm

                    tgt = state["targets"][state["target_idx"]]
                    site_id = info["site_id"]
                    cur = forward_site_pos(model, data, site_id)
                    err = tgt - cur
                    err_norm = np.linalg.norm(err)

                    # debugging print (optional)
                    print(f"{arm} cur {cur} tgt {tgt} err {err_norm:.4f}")

                    if err_norm < TOL:
                        print(f"{arm} reached target {state['target_idx']} (err {err_norm:.4f})")
                        state["hold_until"] = time.time() + POSE_HOLD_SECONDS
                        state["ik_iter"] = 0
                        state["target_idx"] += 1
                        continue
                    if state["ik_iter"] >= MAX_ITERS:
                        print(f"{arm} max iters for target {state['target_idx']} (err {err_norm:.4f}); skipping")
                        state["ik_iter"] = 0
                        state["target_idx"] += 1
                        continue

                    # if currently holding, skip until hold_until
                    if state["hold_until"] is not None:
                        if time.time() >= state["hold_until"]:
                            state["hold_until"] = None
                        else:
                            continue

                    # need IK step for this arm
                    apply_updates[arm] = True

                    # compute numeric jacobian (3 x n)
                    J = numeric_jacobian(model, data, info["site_id"], info["qpos_indices"], eps=FD_EPS)
                    pinv = damped_pinv(J, DAMPING)  # shape (n,3)
                    delta_q = (pinv @ err).squeeze()
                    delta_q = STEP_SCALE * delta_q

                    # clamp so we don't propose outside joint limits
                    delta_q = clamp_delta_by_limits(data.qpos, delta_q, info["qpos_addrs"], info["limits"])

                    # store delta and proposed target qpos
                    # proposed qpos (for the affected joint addresses)
                    qpos_prop = data.qpos.copy()
                    for i, addr in enumerate(info["qpos_addrs"]):
                        qpos_prop[addr] += float(delta_q[i])
                    qpos_prop = clamp_qpos_by_limits(qpos_prop, info["qpos_addrs"], info["limits"])

                    deltas[arm] = delta_q
                    target_qpos_proposals[arm] = qpos_prop
                    # don't apply yet — we will combine both arms' proposals below

                # After both arms processed, apply combined updates
                if USE_ACTUATORS:
                    # map each arm's target_qpos_proposals to ctrl_vector entries (actuator indices)
                    # reset ctrl_vector only for actuators we will set (so others remain as-is)
                    # we keep previous ctrl for actuators not controlled by IK
                    for arm in ("right", "left"):
                        if not apply_updates[arm]:
                            continue
                        info = arms_info[arm]
                        targ_qpos = target_qpos_proposals[arm]
                        ai_list = info["actuator_indices"]
                        # set each actuator's ctrl to the desired joint position
                        for ai, j_addr in zip(ai_list, info["qpos_addrs"]):
                            if ai < model.nu:
                                ctrl_vector[ai] = float(targ_qpos[j_addr])
                    # write once
                    data.ctrl[:] = ctrl_vector
                    # let actuators run a bit (small settling window)
                    for _ in range(8):
                        mujoco.mj_step(model, data)
                else:
                    # KINEMATIC mode: combine both arms proposals into a single qpos and apply
                    qpos_new = data.qpos.copy()
                    # apply right then left (they affect disjoint qpos addresses in this model)
                    for arm in ("right", "left"):
                        if not apply_updates[arm]:
                            continue
                        qpos_prop = target_qpos_proposals[arm]
                        for addr in arms_info[arm]["qpos_addrs"]:
                            qpos_new[addr] = float(qpos_prop[addr])
                    # clamp for both arms (safety)
                    # combine addrs/limits
                    for arm in ("right", "left"):
                        qpos_new = clamp_qpos_by_limits(qpos_new, arms_info[arm]["qpos_addrs"], arms_info[arm]["limits"])
                    # assign and forward
                    data.qpos[:] = qpos_new
                    mujoco.mj_forward(model, data)

                # increment ik_iter counters for arms that had an update (we treated both simultaneously)
                for arm in ("right", "left"):
                    if apply_updates[arm]:
                        arm_state[arm]["ik_iter"] += 1

            # render
            try:
                viewer.sync()
            except Exception:
                pass

        # end viewer
        if reset_was_requested:
            print("Reloading model and reopening viewer...")
            time.sleep(0.2)
            continue
        else:
            print("Viewer closed by user. Exiting.")
            break

# end outer loop
print("Script finished.")
