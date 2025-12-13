# ik_least_squares_mujoco.py
"""
Analytic + numeric IK demo for the R-R-P right arm using mujoco-python + scipy.
This script:
 - loads the MJCF via mujoco.MjModel.from_xml_path
 - uses data.qpos and mujoco.mj_forward to evaluate FK inside the optimizer residual
 - solves for [r1, r2, r3_1] using least_squares with joint bounds from the model
 - applies solution to position actuators (if present) or to qpos directly
 - steps/prints final site error

Requirements:
 - mujoco Python bindings (mujoco>=2.x)
 - scipy
"""
import time
import numpy as np
from math import sqrt, atan2
from scipy.optimize import least_squares
import mujoco

# -------- CONFIG ----------
MODEL_XML_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene copy.xml"
# name of the 3 DOF we optimize (right arm)
independent_joints = ["r1", "r2", "r3_1"]
# site to match
site_name = "r_grip_site"
# optional: whether to step forward a bit after applying solution
STEP_AFTER = True
NSTEP = 40
# --------------------------

# load model + data (mujoco low-level Python API)
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

# helpers for name <-> id
def joint_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
def actuator_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
def site_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

# qpos index helper (address into data.qpos)
def qpos_index(jname):
    jid = joint_id(jname)
    return int(model.jnt_qposadr[jid])

# site id
sid = site_id(site_name)

# build list of qpos indices for variables
q_indices = [ qpos_index(n) for n in independent_joints ]
nvars = len(q_indices)

# build joint bounds from model.jnt_range (if available)
lb = np.full((nvars,), -np.inf, dtype=float)
ub = np.full((nvars,),  np.inf, dtype=float)
for i, jname in enumerate(independent_joints):
    jid = joint_id(jname)
    # model.jnt_range is shape (njnt, 2)
    jr = model.jnt_range[jid]    # [min, max]
    lb[i] = float(jr[0])
    ub[i] = float(jr[1])

print("Optimizing joints:", independent_joints)
print("Bounds low :", lb)
print("Bounds high:", ub)

# pack/unpack helpers
def pack_full_q(x):
    """
    Return a full qpos vector (length model.nq) where the selected entries are replaced
    by x (which has length nvars). Does not write into data; caller must write if needed.
    """
    q = data.qpos.copy()
    for i, val in enumerate(x):
        q[q_indices[i]] = float(val)
    return q

# residual function for least_squares -> returns 3 residuals (site world pos - target)
def residual(x, target_world):
    q_full = pack_full_q(x)
    # set qpos and run forward
    data.qpos[:] = q_full
    # run forward kinematics
    mujoco.mj_forward(model, data)
    site_pos = data.site_xpos[sid]
    res = site_pos - target_world
    # return flattened 3-vector
    return res.astype(float)

# pick a target in world coords (example)
target_world = np.array([0.35, 0.03, 0.35], dtype=float)

# initial guess: use current qpos values
x0 = np.array([ float(data.qpos[idx]) for idx in q_indices ], dtype=float)
print("Initial guess x0 =", x0)

# ensure target is visible in output
print("Target world:", target_world)

# run least_squares with bounds
res = least_squares(
    residual,
    x0,
    args=(target_world,),
    bounds=(lb, ub),
    verbose=2,
    xtol=1e-8,
    ftol=1e-8,
    gtol=1e-8,
    max_nfev=1000
)

print("Optimizer success:", res.success, "message:", res.message)
print("Solution (joint values):", res.x)

# evaluate final residual and site pos
data.qpos[:] = pack_full_q(res.x)
mujoco.mj_forward(model, data)
final_site = data.site_xpos[sid].copy()
err = final_site - target_world
print("Final site pos:", final_site)
print("Target pos    :", target_world)
print("Residual (meters):", err, "norm =", np.linalg.norm(err))

# apply solution to actuators (prefer writing to actuator ctrl if you have position actuators)
for jname, val in zip(independent_joints, res.x):
    # expected actuator name convention: "<joint>_ctrl"
    a_name = jname + "_ctrl"
    a_id = actuator_id(a_name)
    if a_id >= 0:
        # position actuators take desired joint position in data.ctrl
        try:
            data.ctrl[a_id] = float(val)
            print(f"Set actuator {a_name} ctrl -> {val:.6f}")
        except Exception:
            # fallback: set qpos directly
            data.qpos[qpos_index(jname)] = float(val)
            print(f"Fallback set qpos for {jname} -> {val:.6f}")
    else:
        # no actuator found; set qpos directly
        data.qpos[qpos_index(jname)] = float(val)
        print(f"No actuator {a_name}; set qpos for {jname} -> {val:.6f}")

mujoco.mj_forward(model, data)

# (optional) step the simulation and (optionally) launch viewer if you want to visualize
if STEP_AFTER:
    print(f"Stepping {NSTEP} frames to let actuators settle...")
    for i in range(NSTEP):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    final_site2 = data.site_xpos[sid].copy()
    print("Final site (after stepping):", final_site2, "error norm:", np.linalg.norm(final_site2 - target_world))

print("Done.")
