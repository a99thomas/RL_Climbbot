# ik_gui_multiproc.py
import time
import math
import numpy as np
import multiprocessing as mp
import mujoco
import mujoco.viewer
import tkinter as tk
from tkinter import ttk

# ---------------- CONFIG ----------------
XML_FILE_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene copy.xml"
simulation_speed = 1.0
# ----------------------------------------

def run_tk_gui(shared):
    """
    Runs in a separate process. Updates shared['x'], ['y'], ['z'] periodically.
    Provide simple Reset L button that toggles shared['reset_L'] to True once.
    """
    root = tk.Tk()
    root.title("IK Target Sliders (GUI Process)")

    # Get initial bounds from shared if present, otherwise use defaults
    # shared may contain 'init_joint2' passed as a tuple.
    init_joint2 = shared.get('init_joint2', (0.0, 0.0, 0.0))
    jx, jy, jz = init_joint2

    # sensible slider ranges relative to joint2
    x_min, x_max = jx + 0.1, jx + 0.54
    y_min, y_max = jy - 0.25, jy + 0.25
    z_min, z_max = jz - 0.10, jz + 0.25

    # Set initial target from shared if exists
    init_t = shared.get('init_target', (jx + 0.12, jy + 0.0, jz + 0.02))

    sx = tk.DoubleVar(value=init_t[0])
    sy = tk.DoubleVar(value=init_t[1])
    sz = tk.DoubleVar(value=init_t[2])

    def make_slider(parent, label, var, mn, mx, row):
        ttk.Label(parent, text=label).grid(column=0, row=row, sticky="w")
        s = ttk.Scale(parent, from_=mn, to=mx, variable=var, orient="horizontal")
        s.grid(column=1, row=row, sticky="we", padx=6, pady=6)
        ttk.Label(parent, textvariable=var, width=10).grid(column=2, row=row, sticky="e", padx=4)
        parent.columnconfigure(1, weight=1)

    frame = ttk.Frame(root, padding=8)
    frame.pack(fill="both", expand=True)

    make_slider(frame, "Target X", sx, x_min, x_max, row=0)
    make_slider(frame, "Target Y", sy, y_min, y_max, row=1)
    make_slider(frame, "Target Z", sz, z_min, z_max, row=2)

    # Buttons
    def do_reset_L():
        # set a flag the main process will see
        shared['reset_L'] = True

    def do_quit():
        shared['quit'] = True
        root.quit()

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(column=0, row=3, columnspan=3, pady=(8,0), sticky="we")
    ttk.Button(btn_frame, text="Reset L", command=do_reset_L).pack(side="left", padx=6)
    ttk.Button(btn_frame, text="Quit GUI", command=do_quit).pack(side="left", padx=6)

    # Periodically write slider values to shared dict (20 Hz)
    def gui_update_loop():
        shared['x'] = float(sx.get())
        shared['y'] = float(sy.get())
        shared['z'] = float(sz.get())
        root.after(50, gui_update_loop)

    root.after(50, gui_update_loop)
    root.mainloop()


def main_loop(shared):
    # load model
    try:
        model = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print("Failed to load model:", e)
        return

    # launch viewer (in main process)
    viewer = mujoco.viewer.launch_passive(model, data)

    # names and ids (same as before)
    r1_name = "r1"; r2_name = "r2"; r3_name = "r3_1"; r_site_name = "r_grip_site"
    r1_act_name = "r1_ctrl"; r2_act_name = "r2_ctrl"; r3_act_name = "r3_1_ctrl"

    r1_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, r1_name)
    r2_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, r2_name)
    r3_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, r3_name)

    r1_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r1_act_name)
    r2_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r2_act_name)
    r3_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r3_act_name)

    joint2_body_name = "assembly_13"
    joint2_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, joint2_body_name)
    r_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, r_site_name)

    r1_qpos_adr = model.jnt_qposadr[r1_jid]
    r2_qpos_adr = model.jnt_qposadr[r2_jid]
    r3_qpos_adr = model.jnt_qposadr[r3_jid]

    r1_min, r1_max = model.jnt_range[r1_jid]
    r2_min, r2_max = model.jnt_range[r2_jid]
    r3_min, r3_max = model.jnt_range[r3_jid]

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # estimate L stored in a mutable container so GUI reset can update it
    def estimate_L_now():
        joint2_pos = np.array(data.xpos[joint2_bid]).copy()
        site_pos = np.array(data.site_xpos[r_site_id]).copy()
        vec = site_pos - joint2_pos
        rho_now = math.hypot(math.hypot(vec[0], vec[1]), vec[2])
        d_now = float(data.qpos[r3_qpos_adr])
        return rho_now - d_now

    L_const = [estimate_L_now()]
    print("Estimated L =", L_const[0])

    def analytic_rrp_ik_world(target_xyz, joint2_pos, L):
        dx = np.array(target_xyz, dtype=float) - np.array(joint2_pos, dtype=float)
        x_rel, y_rel, z_rel = float(dx[0]), float(dx[1]), float(dx[2])
        theta1 = math.atan2(y_rel, x_rel)
        r = math.hypot(x_rel, y_rel)
        s = z_rel
        rho = math.hypot(r, s)
        theta2 = math.atan2(s, r)
        d = rho - L
        return theta1, theta2, d, rho

    # smoothing
    alpha = 0.25
    last_cmd = {
        "r1": float(data.qpos[r1_qpos_adr]),
        "r2": float(data.qpos[r2_qpos_adr]),
        "r3": float(data.qpos[r3_qpos_adr])
    }

    time0 = time.time()
    last_sim_time = 0.0

    # ensure shared has initial target if not set
    if 'x' not in shared:
        # sensible default based on current site
        shared['x'], shared['y'], shared['z'] = tuple(data.site_xpos[r_site_id])
    # write joint2 pos for GUI to compute sensible ranges (optional)
    if 'init_joint2' not in shared:
        shared['init_joint2'] = tuple(data.xpos[joint2_bid])
    if 'init_target' not in shared:
        shared['init_target'] = tuple(data.site_xpos[r_site_id])

    while viewer.is_running():
        # gracefully exit if GUI requested quit
        if shared.get('quit', False):
            print("GUI requested quit -> closing viewer.")
            viewer.close()
            break

        # detect reset and resync clocks
        if data.time < last_sim_time:
            time0 = time.time() - (data.time / simulation_speed)
        last_sim_time = data.time

        # realtime sync
        elapsed_real_time = time.time() - time0
        target_real_time = data.time / simulation_speed
        time_to_wait = target_real_time - elapsed_real_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        # read shared target (multiproc-safe)
        tx = shared.get('x', data.site_xpos[r_site_id][0])
        ty = shared.get('y', data.site_xpos[r_site_id][1])
        tz = shared.get('z', data.site_xpos[r_site_id][2])
        target = np.array([tx, ty, tz], dtype=float)

        # if GUI requested L reset, do it and clear flag
        if shared.get('reset_L', False):
            L_const[0] = estimate_L_now()
            print("Re-estimated L ->", L_const[0])
            shared['reset_L'] = False

        joint2_pos = data.xpos[joint2_bid]
        times_start = time.time()
        theta1_des, theta2_des, d_des, rho = analytic_rrp_ik_world(target, joint2_pos, L_const[0])

        print("IK computation time:", time.time() - times_start)

        # feasibility checks & clamp
        feasible = True
        reasons = []
        if d_des < r3_min or d_des > r3_max:
            feasible = False
            reasons.append(f"d {d_des:.4f} out of [{r3_min:.4f},{r3_max:.4f}]")
        if theta1_des < r1_min or theta1_des > r1_max:
            feasible = False
            reasons.append(f"theta1 {theta1_des:.3f} out of [{r1_min:.3f},{r1_max:.3f}]")
        if theta2_des < r2_min or theta2_des > r2_max:
            feasible = False
            reasons.append(f"theta2 {theta2_des:.3f} out of [{r2_min:.3f},{r2_max:.3f}]")

        if not feasible:
            theta1_cmd = clamp(theta1_des, r1_min, r1_max)
            theta2_cmd = clamp(theta2_des, r2_min, r2_max)
            d_cmd      = clamp(d_des, r3_min, r3_max)
            # print a short message occasionally
            if int(time.time() - time0) % 2 == 0:
                print("IK infeasible, clamped:", reasons)
        else:
            theta1_cmd = theta1_des
            theta2_cmd = theta2_des
            d_cmd      = d_des

        # smoothing
        theta1_cmd = (1.0 - alpha) * last_cmd["r1"] + alpha * theta1_cmd
        theta2_cmd = (1.0 - alpha) * last_cmd["r2"] + alpha * theta2_cmd
        d_cmd      = (1.0 - alpha) * last_cmd["r3"] + alpha * d_cmd

        # send commands
        data.ctrl[r1_act_id] = float(theta1_cmd)
        data.ctrl[r2_act_id] = float(theta2_cmd)
        data.ctrl[r3_act_id] = float(d_cmd)
        last_cmd["r1"], last_cmd["r2"], last_cmd["r3"] = theta1_cmd, theta2_cmd, d_cmd

        # step and render
        mujoco.mj_step(model, data)

        # optional telemetry
        t_now = time.time() - time0
        if int(t_now*10) % 50 == 0:
            print(f"t={t_now:.2f} target={target.round(3)} ik=[{theta1_des:.3f},{theta2_des:.3f},{d_des:.3f}] rho={rho:.3f}")

        viewer.sync()

    # cleanup
    try:
        viewer.close()
    except Exception:
        pass


if __name__ == "__main__":
    mp.set_start_method('spawn')   # safer on macOS
    manager = mp.Manager()
    shared = manager.dict()

    # start GUI process
    p = mp.Process(target=run_tk_gui, args=(shared,), daemon=True)
    p.start()

    try:
        main_loop(shared)
    finally:
        # ensure GUI process is asked to quit
        shared['quit'] = True
        p.join(timeout=1.0)
        if p.is_alive():
            p.terminate()
        print("Exited.")
