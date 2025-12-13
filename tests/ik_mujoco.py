# ik_gui_multiproc_with_left.py
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
    Runs in a separate process. Updates shared['x'], ['y'], ['z'] for right arm and
    shared['lx'], ['ly'], ['lz'] for left arm periodically.
    Provides Reset L (left) and Reset R (right) buttons that toggle shared['reset_L'] / ['reset_R'].
    """
    root = tk.Tk()
    root.title("IK Target Sliders (GUI Process)")

    # Get initial bounds from shared if present, otherwise use defaults
    init_joint2 = shared.get('init_joint2', (0.0, 0.0, 0.0))
    jx, jy, jz = init_joint2

    # sensible slider ranges relative to joint2
    x_min, x_max = jx + 0.1, jx + 0.6
    y_min, y_max = jy - 0.35, jy + 0.35
    z_min, z_max = jz - 0.10, jz + 0.25

    # Right arm initial target (keep compatibility with existing keys 'x','y','z')
    init_t = shared.get('init_target', (jx + 0.12, jy + 0.0, jz + 0.02))
    sx = tk.DoubleVar(value=init_t[0])
    sy = tk.DoubleVar(value=init_t[1])
    sz = tk.DoubleVar(value=init_t[2])

    # Left arm initial target (new keys 'lx','ly','lz')
    init_t_l = shared.get('init_target_left', (jx + 0.12, jy - 0.2, jz + 0.02))
    slx = tk.DoubleVar(value=init_t_l[0])
    sly = tk.DoubleVar(value=init_t_l[1])
    slz = tk.DoubleVar(value=init_t_l[2])

    def make_slider(parent, label, var, mn, mx, row):
        ttk.Label(parent, text=label).grid(column=0, row=row, sticky="w")
        s = ttk.Scale(parent, from_=mn, to=mx, variable=var, orient="horizontal")
        s.grid(column=1, row=row, sticky="we", padx=6, pady=6)
        ttk.Label(parent, textvariable=var, width=10).grid(column=2, row=row, sticky="e", padx=4)
        parent.columnconfigure(1, weight=1)

    frame = ttk.Frame(root, padding=8)
    frame.pack(fill="both", expand=True)

    # Right target group
    right_label = ttk.Label(frame, text="Right Target", font=("TkDefaultFont", 10, "bold"))
    right_label.grid(column=0, row=0, columnspan=3, sticky="w", pady=(0,4))
    make_slider(frame, "Target X (R)", sx, x_min, x_max, row=1)
    make_slider(frame, "Target Y (R)", sy, y_min, y_max, row=2)
    make_slider(frame, "Target Z (R)", sz, z_min, z_max, row=3)

    # Left target group
    left_label = ttk.Label(frame, text="Left Target", font=("TkDefaultFont", 10, "bold"))
    left_label.grid(column=0, row=4, columnspan=3, sticky="w", pady=(10,4))
    make_slider(frame, "Target X (L)", slx, x_min, x_max, row=5)
    make_slider(frame, "Target Y (L)", sly, y_min, y_max, row=6)
    make_slider(frame, "Target Z (L)", slz, z_min, z_max, row=7)

    # Buttons
    def do_reset_L():
        shared['reset_L'] = True

    def do_reset_R():
        shared['reset_R'] = True

    def do_quit():
        shared['quit'] = True
        root.quit()

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(column=0, row=8, columnspan=3, pady=(12,0), sticky="we")
    ttk.Button(btn_frame, text="Reset L", command=do_reset_L).pack(side="left", padx=6)
    ttk.Button(btn_frame, text="Reset R", command=do_reset_R).pack(side="left", padx=6)
    ttk.Button(btn_frame, text="Quit GUI", command=do_quit).pack(side="left", padx=6)

    # Periodically write slider values to shared dict (~20 Hz)
    def gui_update_loop():
        # Right arm keys (kept as 'x','y','z' for backward compatibility)
        shared['x'] = float(sx.get())
        shared['y'] = float(sy.get())
        shared['z'] = float(sz.get())
        # Left arm keys (new)
        shared['lx'] = float(slx.get())
        shared['ly'] = float(sly.get())
        shared['lz'] = float(slz.get())
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
    viewer = mujoco.viewer.launch_passive(model, data)

    # ---------------- Right side names (as before) ----------------
    r1_name = "r1"; r2_name = "r2"; r3_name = "r3_1"; r_site_name = "r_grip_site"
    r1_act_name = "r1_ctrl"; r2_act_name = "r2_ctrl"; r3_act_name = "r3_1_ctrl"

    # Attempt to build left names by replacing leading 'r' with 'l'
    l1_name = r1_name.replace('r', 'l', 1)
    l2_name = r2_name.replace('r', 'l', 1)
    l3_name = r3_name.replace('r', 'l', 1)
    l_site_name = r_site_name.replace('r', 'l', 1)
    l1_act_name = r1_act_name.replace('r', 'l', 1)
    l2_act_name = r2_act_name.replace('r', 'l', 1)
    l3_act_name = r3_act_name.replace('r', 'l', 1)

    # fetch ids for right (these must exist)
    r1_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, r1_name)
    r2_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, r2_name)
    r3_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, r3_name)

    r1_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r1_act_name)
    r2_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r2_act_name)
    r3_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r3_act_name)

    # try to fetch left ids; if not found, set to None and continue (we'll skip left control)
    def safe_name2id(obj_type, name):
        try:
            return mujoco.mj_name2id(model, obj_type, name)
        except Exception:
            return -1

    l1_jid = safe_name2id(mujoco.mjtObj.mjOBJ_JOINT, l1_name)
    l2_jid = safe_name2id(mujoco.mjtObj.mjOBJ_JOINT, l2_name)
    l3_jid = safe_name2id(mujoco.mjtObj.mjOBJ_JOINT, l3_name)

    l1_act_id = safe_name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, l1_act_name)
    l2_act_id = safe_name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, l2_act_name)
    l3_act_id = safe_name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, l3_act_name)

    # joint2 body and right site (we reuse same joint2 body for both arms unless separate bodies are available)
    joint2_body_name = "assembly_7"
    joint2_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, joint2_body_name)
    r_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, r_site_name)

    # left site id (may be missing)
    l_site_id = safe_name2id(mujoco.mjtObj.mjOBJ_SITE, l_site_name)

    # --- NEW BLOCK: compute init_joint2 from zeroed joints in base frame assembly_7 ---
    # If shared['init_joint2'] not already set, compute it by setting all joints to 0,
    # forwarding the model, reading the right site position, and expressing that in the base frame.
    if 'init_joint2' not in shared:
        try:
            # save original qpos
            qpos_orig = data.qpos.copy()

            # set all joint qpos to zero
            data.qpos[:] = 0.0
            mujoco.mj_forward(model, data)

            # read site world position (right site) and base/world pose
            if r_site_id != -1 and joint2_bid != -1:
                site_world = np.array(data.site_xpos[r_site_id], dtype=float)
                base_world = np.array(data.xpos[joint2_bid], dtype=float)
                base_xmat = np.array(data.xmat[joint2_bid]).reshape(3,3)  # body->world rotation

                # express site in base frame: rel = R_base^T * (site_world - base_world)
                rel = base_xmat.T @ (site_world - base_world)
                jx, jy, jz = float(rel[0]), float(rel[1]), float(rel[2])
                shared['init_joint2'] = (jx, jy, jz)

                # also set sensible initial targets (in base frame coordinates) so GUI aligns
                # Convert back to world coords for init_target (GUI expects world-like values earlier),
                # but keep init_joint2 as base-frame vector since GUI uses it only for slider bounds.
                # For compatibility, set init_target to the world site pos:
                shared['init_target'] = tuple(site_world)
                # set left initial target a bit offset in world coordinates
                shared['init_target_left'] = tuple(site_world + np.array([0.0, -0.2, 0.0]))
                print("Computed init_joint2 (site in base frame with q=0):", shared['init_joint2'])
            else:
                # fallback: use joint2 body world position if site missing
                shared['init_joint2'] = tuple(data.xpos[joint2_bid])
                shared['init_target'] = tuple(data.site_xpos[r_site_id]) if r_site_id != -1 else tuple(data.xpos[joint2_bid])
                shared['init_target_left'] = (shared['init_target'][0], shared['init_target'][1] - 0.2, shared['init_target'][2])
                print("Fallback init_joint2 (world joint2 pos):", shared['init_joint2'])
        except Exception as e:
            print("Failed to compute init_joint2 from zero qpos:", e)
            # fallback: use current joint2 world pos
            shared['init_joint2'] = tuple(data.xpos[joint2_bid])
            shared['init_target'] = tuple(data.site_xpos[r_site_id]) if r_site_id != -1 else tuple(data.xpos[joint2_bid])
            shared['init_target_left'] = (shared['init_target'][0], shared['init_target'][1] - 0.2, shared['init_target'][2])
        finally:
            # restore original qpos and forward again
            try:
                data.qpos[:] = qpos_orig
                mujoco.mj_forward(model, data)
            except Exception:
                pass
    # --- END NEW BLOCK ---

    # qpos addresses
    r1_qpos_adr = model.jnt_qposadr[r1_jid]
    r2_qpos_adr = model.jnt_qposadr[r2_jid]
    r3_qpos_adr = model.jnt_qposadr[r3_jid]

    l1_qpos_adr = model.jnt_qposadr[l1_jid] if l1_jid != -1 else None
    l2_qpos_adr = model.jnt_qposadr[l2_jid] if l2_jid != -1 else None
    l3_qpos_adr = model.jnt_qposadr[l3_jid] if l3_jid != -1 else None

    # joint ranges (for feasibility checks)
    r1_min, r1_max = model.jnt_range[r1_jid]
    r2_min, r2_max = model.jnt_range[r2_jid]
    r3_min, r3_max = model.jnt_range[r3_jid]

    if l1_jid != -1:
        try:
            l1_min, l1_max = model.jnt_range[l1_jid]
            l2_min, l2_max = model.jnt_range[l2_jid]
            l3_min, l3_max = model.jnt_range[l3_jid]
        except Exception:
            # If left ranges aren't available, fall back to right ranges (conservative)
            l1_min, l1_max = r1_min, r1_max
            l2_min, l2_max = r2_min, r2_max
            l3_min, l3_max = r3_min, r3_max
    else:
        l1_min = l1_max = l2_min = l2_max = l3_min = l3_max = None

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # estimate L stored in a mutable container so GUI reset can update it
    def estimate_L_now(site_id):
        joint2_pos = np.array(data.xpos[joint2_bid]).copy()
        site_pos = np.array(data.site_xpos[site_id]).copy()
        vec = site_pos - joint2_pos
        rho_now = math.hypot(math.hypot(vec[0], vec[1]), vec[2])
        # attempt to read the telescoping d from qpos if available (use right r3_qpos_adr by default)
        d_now = float(data.qpos[r3_qpos_adr]) if r3_qpos_adr is not None else 0.0
        return rho_now - d_now

    # Two L constants: one for right, one for left
    L_right = [estimate_L_now(r_site_id)]
    print("Estimated L (right) =", L_right[0])

    if l_site_id != -1:
        L_left = [estimate_L_now(l_site_id)]
        print("Estimated L (left) =", L_left[0])
    else:
        # fallback to using same as right if left site missing
        L_left = [L_right[0]]

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
    alpha = 0.95
    last_cmd = {
        "r1": float(data.qpos[r1_qpos_adr]),
        "r2": float(data.qpos[r2_qpos_adr]),
        "r3": float(data.qpos[r3_qpos_adr]),
        # left defaults: if not present, we'll ignore them later
        "l1": float(data.qpos[l1_qpos_adr]) if l1_qpos_adr is not None else 0.0,
        "l2": float(data.qpos[l2_qpos_adr]) if l2_qpos_adr is not None else 0.0,
        "l3": float(data.qpos[l3_qpos_adr]) if l3_qpos_adr is not None else 0.0
    }

    time0 = time.time()
    last_sim_time = 0.0

    # ensure shared has initial target if not set (right)
    if 'x' not in shared:
        shared['x'], shared['y'], shared['z'] = tuple(data.site_xpos[r_site_id])
    # ensure left shared defaults
    if 'lx' not in shared:
        # place left slightly offset in Y by -0.2 as a reasonable default
        lx0, ly0, lz0 = tuple(np.array(data.site_xpos[r_site_id]) + np.array([0.0, -0.2, 0.0]))
        shared['lx'], shared['ly'], shared['lz'] = lx0, ly0, lz0

    # write joint2 pos for GUI to compute sensible ranges (optional)
    if 'init_joint2' not in shared:
        shared['init_joint2'] = tuple(data.xpos[joint2_bid])
    if 'init_target' not in shared:
        shared['init_target'] = tuple(data.site_xpos[r_site_id])
    if 'init_target_left' not in shared:
        # left initial target guess
        shared['init_target_left'] = (shared['lx'], shared['ly'], shared['lz'])

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

        # read shared targets (multiproc-safe)
        tx = shared.get('x', data.site_xpos[r_site_id][0])
        ty = shared.get('y', data.site_xpos[r_site_id][1])
        tz = shared.get('z', data.site_xpos[r_site_id][2])
        target_r = np.array([tx, ty, tz], dtype=float)

        lx = shared.get('lx', data.site_xpos[r_site_id][0])
        ly = shared.get('ly', data.site_xpos[r_site_id][1])
        lz = shared.get('lz', data.site_xpos[r_site_id][2])
        target_l = np.array([lx, ly, lz], dtype=float)

        # if GUI requested resets, do them and clear flags
        if shared.get('reset_R', False):
            L_right[0] = estimate_L_now(r_site_id)
            print("Re-estimated L (right) ->", L_right[0])
            shared['reset_R'] = False
        if shared.get('reset_L', False):
            # only re-estimate left L if we have a left site id
            if l_site_id != -1:
                L_left[0] = estimate_L_now(l_site_id)
            else:
                # fallback to using right site pos if left site missing
                L_left[0] = estimate_L_now(r_site_id)
            print("Re-estimated L (left) ->", L_left[0])
            shared['reset_L'] = False

        joint2_pos = data.xpos[joint2_bid]
        times_start = time.time()

        # RIGHT IK
        theta1_des_r, theta2_des_r, d_des_r, rho_r = analytic_rrp_ik_world(target_r, joint2_pos, L_right[0])
        # feasibility checks & clamp for right
        feasible_r = True
        reasons_r = []
        if d_des_r < r3_min or d_des_r > r3_max:
            feasible_r = False
            reasons_r.append(f"d {d_des_r:.4f} out of [{r3_min:.4f},{r3_max:.4f}]")
        if theta1_des_r < r1_min or theta1_des_r > r1_max:
            feasible_r = False
            reasons_r.append(f"theta1 {theta1_des_r:.3f} out of [{r1_min:.3f},{r1_max:.3f}]")
        if theta2_des_r < r2_min or theta2_des_r > r2_max:
            feasible_r = False
            reasons_r.append(f"theta2 {theta2_des_r:.3f} out of [{r2_min:.3f},{r2_max:.3f}]")

        if not feasible_r:
            theta1_cmd_r = clamp(theta1_des_r, r1_min, r1_max)
            theta2_cmd_r = clamp(theta2_des_r, r2_min, r2_max)
            d_cmd_r      = clamp(d_des_r, r3_min, r3_max)
            if int(time.time() - time0) % 2 == 0:
                print("Right IK infeasible, clamped:", reasons_r)
        else:
            theta1_cmd_r = theta1_des_r
            theta2_cmd_r = theta2_des_r
            d_cmd_r      = d_des_r

        # smoothing (right)
        theta1_cmd_r = (1.0 - alpha) * last_cmd["r1"] + alpha * theta1_cmd_r
        theta2_cmd_r = (1.0 - alpha) * last_cmd["r2"] + alpha * theta2_cmd_r
        d_cmd_r      = (1.0 - alpha) * last_cmd["r3"] + alpha * d_cmd_r

        # SEND RIGHT commands
        data.ctrl[r1_act_id] = float(theta1_cmd_r)
        data.ctrl[r2_act_id] = float(theta2_cmd_r)
        data.ctrl[r3_act_id] = float(d_cmd_r)
        last_cmd["r1"], last_cmd["r2"], last_cmd["r3"] = theta1_cmd_r, theta2_cmd_r, d_cmd_r

        # LEFT IK (only if left joints/actuators present)
        if l1_jid != -1 and l1_act_id != -1 and l2_act_id != -1 and l3_act_id != -1:
            theta1_des_l, theta2_des_l, d_des_l, rho_l = analytic_rrp_ik_world(target_l, joint2_pos, L_left[0])

            feasible_l = True
            reasons_l = []
            if d_des_l < l3_min or d_des_l > l3_max:
                feasible_l = False
                reasons_l.append(f"d {d_des_l:.4f} out of [{l3_min:.4f},{l3_max:.4f}]")
            if theta1_des_l < l1_min or theta1_des_l > l1_max:
                feasible_l = False
                reasons_l.append(f"theta1 {theta1_des_l:.3f} out of [{l1_min:.3f},{l1_max:.3f}]")
            if theta2_des_l < l2_min or theta2_des_l > l2_max:
                feasible_l = False
                reasons_l.append(f"theta2 {theta2_des_l:.3f} out of [{l2_min:.3f},{l2_max:.3f}]")

            if not feasible_l:
                theta1_cmd_l = clamp(theta1_des_l, l1_min, l1_max)
                theta2_cmd_l = clamp(theta2_des_l, l2_min, l2_max)
                d_cmd_l      = clamp(d_des_l, l3_min, l3_max)
                if int(time.time() - time0) % 2 == 0:
                    print("Left IK infeasible, clamped:", reasons_l)
            else:
                theta1_cmd_l = theta1_des_l
                theta2_cmd_l = theta2_des_l
                d_cmd_l      = d_des_l

            # smoothing (left)
            theta1_cmd_l = (1.0 - alpha) * last_cmd["l1"] + alpha * theta1_cmd_l
            theta2_cmd_l = (1.0 - alpha) * last_cmd["l2"] + alpha * theta2_cmd_l
            d_cmd_l      = (1.0 - alpha) * last_cmd["l3"] + alpha * d_cmd_l

            # send left commands
            data.ctrl[l1_act_id] = float(theta1_cmd_l)
            data.ctrl[l2_act_id] = float(theta2_cmd_l)
            data.ctrl[l3_act_id] = float(d_cmd_l)
            last_cmd["l1"], last_cmd["l2"], last_cmd["l3"] = theta1_cmd_l, theta2_cmd_l, d_cmd_l
        else:
            # left not present / incomplete: skip left control
            # helpful one-time messages
            if l_site_id == -1 and (l1_jid != -1 or l1_act_id != -1):
                # site missing but some left joints found: a warning
                pass  # suppressed to reduce log spam

        # step and render
        mujoco.mj_step(model, data)

        # optional telemetry
        t_now = time.time() - time0
        if int(t_now*10) % 50 == 0:
            # include left info if available
            if l1_jid != -1:
                print(f"t={t_now:.2f} Rtarget={target_r.round(3)} Rik=[{theta1_des_r:.3f},{theta2_des_r:.3f},{d_des_r:.3f}] rho={rho_r:.3f} "
                      f"Ltarget={target_l.round(3)} Lik=[{theta1_des_l:.3f},{theta2_des_l:.3f},{d_des_l:.3f}] rho_l={rho_l:.3f}")
            else:
                print(f"t={t_now:.2f} target={target_r.round(3)} ik=[{theta1_des_r:.3f},{theta2_des_r:.3f},{d_des_r:.3f}] rho={rho_r:.3f}")

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
