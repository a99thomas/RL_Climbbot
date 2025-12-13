import numpy as np
from scipy.optimize import least_squares

# ============================================================
# FORWARD KINEMATICS (RIGHT ARM) — ALL UNITS IN METERS
# ============================================================

def fk_right(q1, q2, q3,
             *,
             x0=0.180853,
             y0=0.1192,
             z0=0.075378,
             x2=0.000563,
             y2=0.0565,
             z2=0.024,
             x3=0.2415,
             y3=0.03501,
             z3=0.012378):
    """
    Forward kinematics for the RIGHT ARM in meters.

    q1, q2 : joint angles in radians
    q3     : linear joint extension in meters
    """

    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    q3 = np.asarray(q3)

    s1 = np.sin(q1); c1 = np.cos(q1)
    s2 = np.sin(q2); c2 = np.cos(q2)

    v1 = y2 + y3
    v2 = x2
    v3 = z3
    v4 = x3 + q3   # linear extension in METERS

    p0x = x0
    p0y = y0
    p0z = z0 - z2

    x = p0x + (-s1)*v1 + (-c1)*v2 + (c1*s2)*v3 + (c1*c2)*v4
    y = p0y + ( c1)*v1 + (-s1)*v2 + (s1*s2)*v3 + (s1*c2)*v4
    z = p0z + (-c2)*v3 + (s2)*v4

    return np.stack([x, y, z], axis=-1)


# ============================================================
# FORWARD KINEMATICS (LEFT ARM) — ALL UNITS IN METERS
# ============================================================

def fk_left(q4, q5, q6,
            *,
            x4=0.180853,
            y4=0.019158,
            z4=0.025378,
            x5=0.000563,
            y5=0.0565,
            z5=0.029,
            x6=0.2415,
            y6=0.03501,
            z6=0.012378):
    """
    Forward kinematics for the LEFT ARM in meters.

    q4, q5 : joint angles in radians
    q6     : linear joint extension in meters
    """

    q4 = np.asarray(q4)
    q5 = np.asarray(q5)
    q6 = np.asarray(q6)

    s4 = np.sin(q4); c4 = np.cos(q4)
    s5 = np.sin(q5); c5 = np.cos(q5)

    v1 = y5 + y6
    v2 = x5
    v3 = z6
    v4 = x6 + q6   # linear extension in METERS

    p0x = x4
    p0y = y4
    p0z = z4 - z5

    x = p0x + ( s4)*v1 + ( c4)*v2 + (-c4*s5)*v3 + (c4*c5)*v4
    y = p0y + (-c4)*v1 + ( s4)*v2 + (-s4*s5)*v3 + (s4*c5)*v4
    z = p0z + (-c5)*v3 + (-s5)*v4

    return np.stack([x, y, z], axis=-1)


# ============================================================
# Helper: small numerical Jacobian for 3-DOF FK (cheap)
# ============================================================
def _numerical_jacobian_3d(q, fk_func, eps=1e-6):
    """
    q: length-3 array-like
    fk_func: callable(q1,q2,q3) -> length-3 position
    returns: 3x3 Jacobian (columns are partial derivatives wrt q[i])
    """
    q = np.asarray(q, dtype=float)
    f0 = fk_func(q[0], q[1], q[2])
    J = np.zeros((3, 3), dtype=float)
    for i in range(3):
        dq = np.zeros(3, dtype=float)
        dq[i] = eps
        f1 = fk_func(q[0] + dq[0], q[1] + dq[1], q[2] + dq[2])
        J[:, i] = (f1 - f0) / eps
    return J


# ============================================================
# INVERSE KINEMATICS (RIGHT ARM) — FASTER VERSION
# ============================================================

# simple cache for warm-starting
_ik_cache_right = {"last_p": None, "last_q": None}

def ik_right(
    p_des,
    q_init=None,
    restarts=1,               # default 1 (use warm-start)
    use_warm_start=True,
    tol=1e-6,
    max_nfev=500,
    skip_if_close=True,
    skip_thresh=1e-4,         # meters: skip if target moved less than this
):
    """
    Faster IK for RIGHT arm by warm-starting and using practical tolerances.
    Returns dict like {"q": array(3,), "success": bool, "residual_norm": float, "message": str}
    """
    p_des = np.asarray(p_des, dtype=float).ravel()

    # Skip solve if target hasn't changed significantly
    if skip_if_close and _ik_cache_right["last_p"] is not None:
        if np.linalg.norm(p_des - _ik_cache_right["last_p"]) < skip_thresh:
            q_cached = _ik_cache_right["last_q"].copy()
            return {
                "q": q_cached,
                "success": True,
                "residual_norm": float(np.linalg.norm(fk_right(*q_cached) - p_des)),
                "message": "cached"
            }

    # initial guess
    if q_init is None:
        if use_warm_start and _ik_cache_right["last_q"] is not None:
            q_init = _ik_cache_right["last_q"].copy()
        else:
            q_init = np.array([0.0, 0.0, 0.0], dtype=float)

    rng = np.random.default_rng(42)

    guesses = [q_init]
    for _ in range(max(0, restarts - 1)):
        guesses.append(np.array([
            rng.uniform(-np.pi, np.pi),
            rng.uniform(-np.pi/2, np.pi/2),
            rng.uniform(-0.1, 0.3)
        ], dtype=float))

    best = None

    # wrapper to use fk_right with scalar args
    fk_wrapper = lambda q1, q2, q3: fk_right(q1, q2, q3)

    lsq_opts = dict(ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev)

    for g in guesses:
        try:
            # residual function accepts vector q
            def residuals(q):
                q1, q2, q3 = q
                return fk_right(q1, q2, q3) - p_des

            # try TRF with numeric jacobian (2-point) — good default for robustness/perf
            sol = least_squares(residuals, g, jac='2-point', method='trf', **lsq_opts)
            norm = float(np.linalg.norm(sol.fun))

            if best is None or norm < best["residual_norm"]:
                best = {"q": sol.x.copy(), "success": bool(sol.success), "residual_norm": norm, "message": sol.message}

            if norm < tol:
                break
        except Exception:
            # solver failed for this guess — continue
            continue

    if best is None:
        # fallback: return initial guess (unsuccessful)
        best = {"q": q_init.copy(), "success": False, "residual_norm": np.inf, "message": "no_solution"}

    # cache the result for next call
    _ik_cache_right["last_p"] = p_des.copy()
    _ik_cache_right["last_q"] = best["q"].copy()

    return best


# ============================================================
# INVERSE KINEMATICS (LEFT ARM) — FASTER VERSION
# ============================================================

# cache for left arm
_ik_cache_left = {"last_p": None, "last_q": None}

def ik_left(
    p_des,
    q_init=None,
    restarts=1,
    use_warm_start=True,
    tol=1e-4,
    max_nfev=500,
    skip_if_close=True,
    skip_thresh=1e-4,
):
    """
    Faster IK for LEFT arm with same strategies as ik_right.
    """
    p_des = np.asarray(p_des, dtype=float).ravel()

    if skip_if_close and _ik_cache_left["last_p"] is not None:
        if np.linalg.norm(p_des - _ik_cache_left["last_p"]) < skip_thresh:
            q_cached = _ik_cache_left["last_q"].copy()
            return {
                "q": q_cached,
                "success": True,
                "residual_norm": float(np.linalg.norm(fk_left(*q_cached) - p_des)),
                "message": "cached"
            }

    # initial guess
    if q_init is None:
        if use_warm_start and _ik_cache_left["last_q"] is not None:
            q_init = _ik_cache_left["last_q"].copy()
        else:
            q_init = np.array([0.0, 0.0, 0.0], dtype=float)

    rng = np.random.default_rng(24)

    guesses = [q_init]
    for _ in range(max(0, restarts - 1)):
        guesses.append(np.array([
            rng.uniform(-np.pi, np.pi),
            rng.uniform(-np.pi/2, np.pi/2),
            rng.uniform(-0.1, 0.3)
        ], dtype=float))

    best = None
    lsq_opts = dict(ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev)

    for g in guesses:
        try:
            def residuals(q):
                q4, q5, q6 = q
                return fk_left(q4, q5, q6) - p_des

            sol = least_squares(residuals, g, jac='2-point', method='trf', **lsq_opts)
            norm = float(np.linalg.norm(sol.fun))

            if best is None or norm < best["residual_norm"]:
                best = {"q": sol.x.copy(), "success": bool(sol.success), "residual_norm": norm, "message": sol.message}

            if norm < tol:
                break
        except Exception:
            continue

    if best is None:
        best = {"q": q_init.copy(), "success": False, "residual_norm": np.inf, "message": "no_solution"}

    _ik_cache_left["last_p"] = p_des.copy()
    _ik_cache_left["last_q"] = best["q"].copy()

    return best


# ============================================================
# EXAMPLE USAGE & BASIC BENCHMARK
# ============================================================
if __name__ == "__main__":
    # Example FK
    print("Right FK:", fk_right(0.5, 0.3, 0.05))  # 5 cm extension
    print("Left FK :", fk_left(0.5, 0.3, 0.05))

    # Example IK solve (single call)
    target = np.array([0.35, 0.31, 0.12])
    sol = ik_right(target)
    print("IK Right:", sol)

    # Quick micro-benchmark: compare old vs new settings (approx)
    p_test = target
    N = 50

    # fast settings (warm-start, restarts=1)
    t0 = __import__("time").perf_counter()
    for _ in range(N):
        _ = ik_right(p_test, restarts=1, tol=1e-6, max_nfev=500)
    t1 = __import__("time").perf_counter()
    print(f"New IK avg: {(t1-t0)/N*1000:.3f} ms per call")

    # If you want to test the 'old' heavy solver (ONLY for comparison) you could copy your original ik_right body
    # and run it here with restarts=6 and tighter tolerances — expect it to be much slower.
