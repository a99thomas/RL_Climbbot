"""
test_mesh_holds.py — feasibility: can the passive hook+catch-bar grip body weight on the
ORIGINAL mesh climbing holds (not cylinders)?

Usage:
    python test_mesh_holds.py              # run all tests, print verdict
    mjpython test_mesh_holds.py --view     # watch the best hang attempt
"""
import argparse
import os
import numpy as np
import mujoco
from climb_env_v7 import ClimbBotEnv

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "assets", "scene_v7_treadmill_holds.xml")
RUNG_SCENE = os.path.join(os.path.dirname(SCENE), "scene_v7_treadmill.xml")
HANG_RUNG = "hang_start_treadmill.npy"
HANG_MESH = "hang_start_treadmill_holds.npy"


def load_hang(path):
    return np.load(path) if os.path.exists(path) else None


def hang_stats(e, steps=800):
    """Run zero-action hang; return summary dict."""
    fr, fl, ur, ul, bz = [], [], [], [], []
    fell = False
    for t in range(steps):
        _, _, term, _, info = e.step(np.zeros(6))
        fr.append(e._hook_force(e.r_hook_gids))
        fl.append(e._hook_force(e.l_hook_gids))
        ur.append(e._uplift_on(e.r_hook_gids))
        ul.append(e._uplift_on(e.l_hook_gids))
        bz.append(info["base_z"])
        if term:
            fell = True
            break
    W = e.body_weight
    ur = np.array(ur); ul = np.array(ul)
    return {
        "steps": t + 1,
        "fell": fell,
        "uplift_r_mean": float(ur.mean()),
        "uplift_l_mean": float(ul.mean()),
        "uplift_total_mean": float((ur + ul).mean()),
        "uplift_r_last": float(ur[-1]),
        "uplift_l_last": float(ul[-1]),
        "base_z_last": float(bz[-1]),
        "base_z_start": float(e._start_z),
        "both_firm": bool(ur[-1] > 0.2 * W and ul[-1] > 0.2 * W),
        "weight": W,
    }


def try_pose(e, qpos, label):
    """Place robot at qpos, run treadmill settle, then zero-action hang."""
    mujoco.mj_resetData(e.model, e.data)
    e.data.qpos[:] = qpos
    q0 = e.data.qpos[e.qpos_adr].copy()
    mujoco.mj_forward(e.model, e.data)
    e.target = q0.copy()
    e.data.ctrl[e.act_ids] = e.target
    e.step_count = 0
    if e.treadmill:
        e._tm_set_window()
        mujoco.mj_forward(e.model, e.data)
        for _ in range(40):
            mujoco.mj_step(e.model, e.data)
        mujoco.mj_forward(e.model, e.data)
        e.active = "left"
        e.moves_done = 0
    e._secure = 0
    e._prev_phi = None
    e._start_z = float(e.data.xpos[e.base_bid][2])
    e._prev_z = e._start_z
    e._max_z = e._start_z
    e._start_up = e.data.xmat[e.base_bid].reshape(3, 3)[:, 2].copy()
    if e.treadmill:
        e._prev_active_d = e._active_target_dist()

    s = hang_stats(e)
    s["label"] = label
    return s


def search_hang_pose(e, q0, n=80, seed=0):
    """Random search around a seed qpos to maximize total uplift after settle+hang."""
    rng = np.random.default_rng(seed)
    best_q, best_score, best_s = q0.copy(), -1.0, None
    for i in range(n):
        q = q0.copy()
        q[e.qpos_adr] = np.clip(
            q0[e.qpos_adr] + rng.uniform(-0.04, 0.04, 6),
            e.jnt_range[:, 0], e.jnt_range[:, 1])
        s = try_pose(e, q, f"search_{i}")
        score = s["uplift_total_mean"]
        if score > best_score and not s["fell"]:
            best_score, best_q, best_s = score, q.copy(), s
            best_s["label"] = f"search_best (trial {i}, score {score:.1f} N)"
    return best_q, best_s


def compare_rungs_baseline():
    """Same test on cylinder rungs for reference."""
    e = ClimbBotEnv(xml_path=RUNG_SCENE, treadmill=True, max_moves=0, tm_spacing=0.15,
                    gravity_scale=1.0, init_qpos_file=HANG_RUNG, max_steps=400,
                    init_joint_noise=0.0)
    q = load_hang(HANG_RUNG)
    return try_pose(e, q, "CYLINDER RUNGS (baseline)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true", help="watch best search pose (needs a passing search)")
    ap.add_argument("--view-scene", action="store_true",
                    help="open mesh-hold scene in viewer (uses treadmill hang pose; add --sim to run physics)")
    ap.add_argument("--sim", action="store_true", help="with --view-scene: run physics instead of static pose")
    ap.add_argument("--search", type=int, default=80, help="random pose search trials")
    args = ap.parse_args()

    if args.view_scene:
        import mujoco.viewer
        import time
        m = mujoco.MjModel.from_xml_path(SCENE)
        d = mujoco.MjData(m)
        q = load_hang(HANG_RUNG)
        if q is not None and q.shape[0] == m.nq:
            d.qpos[:] = q
        mujoco.mj_forward(m, d)
        print("Mesh-hold scene:", SCENE)
        print("Pose:", HANG_RUNG, "| sim:", args.sim, "| Esc to quit")
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running():
                if args.sim:
                    mujoco.mj_step(m, d)
                else:
                    mujoco.mj_forward(m, d)
                v.sync()
                time.sleep(0.005 if args.sim else 0.001)
        return

    print("=" * 60)
    print("MESH HOLD FEASIBILITY TEST")
    print("=" * 60)

    # --- baseline: cylinder rungs ---
    print("\n[1] Cylinder rung baseline (should pass)...")
    s_rung = compare_rungs_baseline()
    print(f"    uplift R={s_rung['uplift_r_last']:+.1f} L={s_rung['uplift_l_last']:+.1f} N  "
          f"total={s_rung['uplift_r_last']+s_rung['uplift_l_last']:+.1f} / {s_rung['weight']:.1f} N  "
          f"fell={s_rung['fell']}  both_firm={s_rung['both_firm']}")

    # --- mesh holds: treadmill hang pose ---
    print("\n[2] Mesh holds + treadmill hang pose (direct port)...")
    e = ClimbBotEnv(xml_path=SCENE, treadmill=True, max_moves=0, tm_spacing=0.15,
                    gravity_scale=1.0, init_qpos_file=HANG_RUNG, max_steps=400,
                    init_joint_noise=0.0)
    q_rung = load_hang(HANG_RUNG)
    s_direct = try_pose(e, q_rung, "mesh: treadmill qpos")
    print(f"    uplift R={s_direct['uplift_r_last']:+.1f} L={s_direct['uplift_l_last']:+.1f} N  "
          f"total={s_direct['uplift_r_last']+s_direct['uplift_l_last']:+.1f} / {s_direct['weight']:.1f} N  "
          f"fell={s_direct['fell']}  both_firm={s_direct['both_firm']}")

    # --- random search for best hang on mesh holds ---
    print(f"\n[3] Random search ({args.search} trials) around treadmill hang pose...")
    e3 = ClimbBotEnv(xml_path=SCENE, treadmill=True, max_moves=0, tm_spacing=0.15,
                     gravity_scale=1.0, init_qpos_file=HANG_RUNG, max_steps=400,
                     init_joint_noise=0.0)
    best_q, s_best = search_hang_pose(e3, q_rung, n=args.search)
    if s_best:
        print(f"    BEST: uplift R={s_best['uplift_r_last']:+.1f} L={s_best['uplift_l_last']:+.1f} N  "
              f"total={s_best['uplift_r_last']+s_best['uplift_l_last']:+.1f} / {s_best['weight']:.1f} N  "
              f"fell={s_best['fell']}  both_firm={s_best['both_firm']}")
        if s_best["both_firm"] and s_best["steps"] >= 800:
            np.save(HANG_MESH, best_q)
            print(f"    Saved stable hang pose -> {HANG_MESH}")

    # --- verdict ---
    W = s_rung["weight"]
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    def verdict(s):
        if s is None:
            return "NO DATA"
        tot = s["uplift_r_last"] + s["uplift_l_last"]
        if s["fell"]:
            return "FAIL (fell)"
        if tot > 0.8 * W and s["both_firm"] and s["steps"] >= 800:
            return "PASS — both hooks load-bearing (800+ steps)"
        if tot > 0.8 * W and s["both_firm"]:
            return "SHORT-PASS — loads at first but may slip (check steps)"
        if tot > 0.4 * W:
            return "MARGINAL — partial load only"
        return "FAIL — hooks not loading"

    print(f"  Cylinder rungs:     {verdict(s_rung)}")
    print(f"  Mesh (direct port): {verdict(s_direct)}")
    print(f"  Mesh (best search): {verdict(s_best)}")
    print()
    if s_best and s_best["both_firm"] and (s_best["uplift_r_last"] + s_best["uplift_l_last"]) > 0.5 * W \
            and s_best["steps"] >= 800:
        print("  -> Mesh holds CAN bear load stably. Training is worth trying.")
        print("     Next: warm-start from slow1, use --xml scene_v7_treadmill_holds")
    elif s_best and s_best["both_firm"]:
        print("  -> Mesh holds load briefly (~300 steps) then SLIP — not stable enough to climb.")
        print("     Recommend adding a hidden grip lip inside the mesh, then re-test.")
    else:
        print("  -> Mesh holds do NOT reliably load the hooks with current geometry.")
        print("     Options: add a hidden grip lip inside the mesh, or reshape the hold STL.")
    print("=" * 60)

    if args.view and s_best and best_q is not None:
        e3.reset()
        e3.data.qpos[:] = best_q
        e3.render_mode = "human"
        try_pose(e3, best_q, "view")
        print("Viewer open — close window to exit.")
        for _ in range(2000):
            e3.step(np.zeros(6))
            e3.render()


if __name__ == "__main__":
    main()
