# ik_benchmark.py
"""
Benchmark analytic R-R-P IK vs scipy least_squares numeric IK
for your MuJoCo model. Produces timing and accuracy statistics.

Usage:
    python3 ik_benchmark.py

Notes:
 - This script uses the low-level mujoco API (mujoco.MjModel / mujoco.MjData).
 - It samples reachable targets only (so least_squares has a chance to converge).
 - Analytic IK is computed in world coords relative to joint2 origin (assembly_13).
"""

import time
import math
import numpy as np
from math import atan2, hypot
from scipy.optimize import least_squares
import mujoco

# -------- USER CONFIG ----------
MODEL_XML_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene copy.xml"
independent_joints = ["r1", "r2", "r3_1"]
site_name = "r_grip_site"
joint2_body_name = "assembly_13"
N_SAMPLES = 200            # total random targets to test
VERBOSE = False            # set True for detailed per-sample prints
SEED = 1234
# optional: limit solver iterations
LS_MAX_NFEV = 300
# --------------------------------

# ---- load model/data ----
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

# helpers
def jid(name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
def aid(name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
def sid(name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
def bid(name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
def qidx(jname): return int(model.jnt_qposadr[jid(jname)])

sid_target = sid(site_name)
joint2_bid = bid(joint2_body_name)

# qpos indices for optimization variables
q_indices = [ qidx(n) for n in independent_joints ]
nvars = len(q_indices)

# joint ranges
lb = np.full((nvars,), -np.inf, dtype=float)
ub = np.full((nvars,),  np.inf, dtype=float)
for i, jname in enumerate(independent_joints):
    jr = model.jnt_range[jid(jname)]
    lb[i] = float(jr[0])
    ub[i] = float(jr[1])

# prismatic (r3) limits from model.jnt_range use same approach above (already in lb/ub)
d_min, d_max = lb[-1], ub[-1]

# measure L (estimate) from current pose
def estimate_L_right():
    mujoco.mj_forward(model, data)
    joint2_pos = np.array(data.xpos[joint2_bid])
    site_pos = np.array(data.site_xpos[sid_target])
    rho_now = np.linalg.norm(site_pos - joint2_pos)
    d_now = float(data.qpos[q_indices[-1]])
    return rho_now - d_now

L = estimate_L_right()
print(f"Estimated L = {L:.6f} m, prismatic d range = [{d_min:.6f}, {d_max:.6f}]")
print("Joint bounds low:", lb, "high:", ub)

# analytic IK (world target, joint2 frame)
def analytic_rrp_ik_world(target_world, joint2_pos, L_local):
    # vector from joint2 origin to target (world coords)
    dx = np.array(target_world) - np.array(joint2_pos)
    x_rel, y_rel, z_rel = float(dx[0]), float(dx[1]), float(dx[2])
    theta1 = math.atan2(y_rel, x_rel)
    r = math.hypot(x_rel, y_rel)
    s = z_rel
    rho = math.hypot(r, s)
    theta2 = math.atan2(s, r)
    d = rho - L_local
    return theta1, theta2, d, rho

# helper to set qpos (pack) and forward
def set_qpos_from_vars(x):
    q = data.qpos.copy()
    for i,val in enumerate(x):
        q[q_indices[i]] = float(val)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

# residual for least squares (world-space site position error)
def residual_ls(x, target_world):
    set_qpos_from_vars(x)
    site_pos = np.array(data.site_xpos[sid_target])
    return (site_pos - target_world).astype(float)

# sample random reachable targets:
# sample yaw in joint1 limits, pitch (theta2) in joint2 limits, rho between L and L + d_max
rng = np.random.default_rng(SEED)
theta1_low, theta1_high = lb[0], ub[0]
theta2_low, theta2_high = lb[1], ub[1]

def sample_reachable():
    # sample theta1, theta2 uniformly within joint limits
    th1 = rng.uniform(theta1_low, theta1_high)
    th2 = rng.uniform(theta2_low, theta2_high)
    rho = rng.uniform(max(L + 1e-6, L), L + d_max)  # (L to L+d_max)
    # convert spherical/polar back to world displacement from joint2
    x = rho * math.cos(th2) * math.cos(th1)
    y = rho * math.cos(th2) * math.sin(th1)
    z = rho * math.sin(th2)
    joint2_pos = np.array(data.xpos[joint2_bid])
    return joint2_pos + np.array([x,y,z])

# prepare sample set
targets = []
for _ in range(N_SAMPLES):
    # ensure using current forward kinematics (joint2 pos may change)
    mujoco.mj_forward(model, data)
    targets.append(sample_reachable())

# benchmarking arrays
times_analytic = []
times_ls = []
ls_iters = []
ls_success = []
analytical_feasible_count = 0
analytical_infeasible_count = 0
residual_norms_analytic = []
residual_norms_ls = []

# run benchmark
for i, tgt in enumerate(targets):
    # recompute joint2 origin in case sim moved
    mujoco.mj_forward(model, data)
    joint2_pos = np.array(data.xpos[joint2_bid])

    # analytic solve + timing
    t0 = time.perf_counter()
    th1, th2, d, rho = analytic_rrp_ik_world(tgt, joint2_pos, L)
    t1 = time.perf_counter()
    times_analytic.append((t1 - t0)*1000.0)  # ms

    # feasibility check (angle & d limits)
    ok = True
    reasons = []
    if d < d_min - 1e-9 or d > d_max + 1e-9:
        ok = False
        reasons.append("d out")
    if th1 < lb[0] - 1e-9 or th1 > ub[0] + 1e-9:
        ok = False
        reasons.append("th1 out")
    if th2 < lb[1] - 1e-9 or th2 > ub[1] + 1e-9:
        ok = False
        reasons.append("th2 out")
    if ok:
        analytical_feasible_count += 1
    else:
        analytical_infeasible_count += 1

    # compute analytic residual error (place those joint values into qpos and measure site)
    x_analytic = np.array([th1, th2, d], dtype=float)
    set_qpos_from_vars(x_analytic)
    analytic_site = np.array(data.site_xpos[sid_target])
    err_analytic = analytic_site - tgt
    residual_norms_analytic.append(np.linalg.norm(err_analytic))

    # Run least_squares with analytic init (fast) and timing
    x0 = x_analytic.copy()
    t0 = time.perf_counter()
    try:
        res = least_squares(residual_ls, x0, args=(tgt,), bounds=(lb, ub),
                            xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=LS_MAX_NFEV, verbose=0)
        t1 = time.perf_counter()
        times_ls.append((t1 - t0)*1000.0)
        ls_iters.append(res.nfev)
        ls_success.append(bool(res.status > 0))
        # final residual
        set_qpos_from_vars(res.x)
        final_site = np.array(data.site_xpos[sid_target])
        residual_norms_ls.append(np.linalg.norm(final_site - tgt))
    except Exception as e:
        t1 = time.perf_counter()
        times_ls.append((t1 - t0)*1000.0)
        ls_iters.append(None)
        ls_success.append(False)
        residual_norms_ls.append(np.nan)
        if VERBOSE:
            print("LS exception on sample", i, ":", e)

    if VERBOSE:
        print(f"sample {i}: analytic_time={times_analytic[-1]:.6f}ms, ls_time={times_ls[-1]:.6f}ms, analytic_res={residual_norms_analytic[-1]:.6e}, ls_res={residual_norms_ls[-1]:.6e}, ls_nfev={ls_iters[-1]}, ok={ok}, reasons={reasons}")

# Summarize results
def stats(arr):
    a = np.array(arr, dtype=float)
    return np.nanmean(a), np.nanmedian(a), np.nanstd(a)

print("\n=== BENCHMARK SUMMARY ===")
print(f"Samples: {N_SAMPLES}")
print(f"Analytic feasible: {analytical_feasible_count}, infeasible: {analytical_infeasible_count}")
mean_a, med_a, std_a = stats(times_analytic)
mean_ls, med_ls, std_ls = stats(times_ls)
print(f"Analytic IK time (ms): mean={mean_a:.6f} median={med_a:.6f} std={std_a:.6f}")
print(f"LeastSquares time (ms): mean={mean_ls:.6f} median={med_ls:.6f} std={std_ls:.6f}")
# residual statistics
print(f"Analytic residual norm (m): mean={np.mean(residual_norms_analytic):.6e}, max={np.nanmax(residual_norms_analytic):.6e}")
print(f"LS residual norm (m): mean={np.nanmean(residual_norms_ls):.6e}, max={np.nanmax(residual_norms_ls):.6e}")
# LS success rate
succ_rate = 100.0 * sum(1 for s in ls_success if s)/len(ls_success)
print(f"LeastSquares success rate: {succ_rate:.2f}% (success counted by res.status > 0)")

# show some distributions (top slowest LS samples)
if len(times_ls) > 0:
    slow_idx = np.argsort(times_ls)[-5:]
    print("\nTop 5 slowest LS times (ms) with residuals and nfev:")
    for idx in reversed(slow_idx):
        print(f"  idx={idx} time={times_ls[idx]:.3f}ms nfev={ls_iters[idx]} ls_res={residual_norms_ls[idx]:.3e} analytic_res={residual_norms_analytic[idx]:.3e}")

print("\nDone.")
