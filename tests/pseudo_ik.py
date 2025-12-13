"""
mujoco_dh_table.py

Compute a standard Denavit-Hartenberg (a, alpha, d, theta) table for an ordered chain
of joints in a MuJoCo model.

Usage:
    - edit XML_FILE_PATH to point to your model
    - set JOINT_CHAIN list to the ordered joint names (base --> end-effector)
    - run: python mujoco_dh_table.py

Assumptions & caveats:
    - Each joint's frame origin is taken as data.xpos[ body_of_joint ] (world position).
    - Joint axis in world frame is computed from model.jnt_axis[jid] rotated by body xmat.
    - Standard DH convention is used.
    - For prismatic joints, d is variable and theta is constant; for revolute joints theta is variable and d is constant.
"""
import numpy as np
import math
import mujoco

# ---------- USER CONFIG ----------
XML_FILE_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene copy.xml"

# Specify joint chain in order from base (closest to robot base) to the end-effector
# Example for your robot's right arm:
JOINT_CHAIN = ["r1", "r2", "r3_1"]
# ---------------------------------

def vec_norm(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v, 0.0
    return v / n, n

def make_frame_from_joint(model, data, jid):
    """
    Build a homogeneous transform T_world_j (4x4) that maps points from the
    joint frame j to world frame.
    Joint frame convention used here:
      - origin: world position of the body to which the joint is attached (data.xpos[bodyid])
      - z-axis: joint axis (model.jnt_axis[jid]) rotated into world with data.xmat[bodyid]
      - x-axis: arbitrary unit vector perpendicular to z, constructed using world reference
      - y-axis: z x x (right handed)
    Returns: T (4x4), and z_axis_world (3,)
    """
    # get joint id -> body id mapping
    # model.jnt_bodyid exists in MuJoCo python API: gives the body index the joint is attached to
    bodyid = int(model.jnt_bodyid[jid])

    # origin in world
    origin_world = np.array(data.xpos[bodyid], dtype=float)

    # body rotation matrix body->world (data.xmat is flattened 3x3 row-major)
    xmat_flat = np.array(data.xmat[bodyid], dtype=float)
    R_body2world = xmat_flat.reshape(3,3)

    # joint axis in joint-local coordinates (model.jnt_axis[jid])
    axis_local = np.array(model.jnt_axis[jid], dtype=float)
    # axis in world coordinates:
    z_world = R_body2world @ axis_local
    z_world, _ = vec_norm(z_world)

    # choose reference for constructing x-axis
    # if z nearly parallel to world Z, use world X as reference; else use world Z
    world_z = np.array([0.0, 0.0, 1.0])
    world_x = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(z_world, world_z)) > 0.95:
        ref = world_x
    else:
        ref = world_z

    x_candidate = np.cross(ref, z_world)
    x_candidate, nx = vec_norm(x_candidate)
    if nx < 1e-8:
        # fallback to some orthogonal vector
        if abs(z_world[0]) < 0.9:
            x_candidate = np.cross(z_world, np.array([1.0,0.0,0.0]))
        else:
            x_candidate = np.cross(z_world, np.array([0.0,1.0,0.0]))
        x_candidate, _ = vec_norm(x_candidate)

    x_world = x_candidate
    y_world = np.cross(z_world, x_world)
    y_world, _ = vec_norm(y_world)

    # Compose rotation matrix R = [x y z] as columns mapping joint-frame->world
    R = np.column_stack([x_world, y_world, z_world])  # 3x3

    # Homogeneous transform
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = origin_world

    return T, z_world

def extract_dh_from_relative_transform(T_rel):
    """
    Given relative homogeneous transform T_{i}^{i+1} (4x4) expressed in frame i,
    extract standard DH parameters (a_i, alpha_i, d_i, theta_i) such that:
      T = RotZ(theta) @ TransZ(d) @ TransX(a) @ RotX(alpha)
    For standard DH the mapping from T to parameters is:
      alpha = atan2(T[2,1], T[2,2])
      d     = T[2,3]
      theta = atan2(T[1,0], T[0,0])
      a     = T[0,3] / cos(theta)  (if cos(theta) too small, use T[1,3]/sin(theta))
    Returns (a, alpha, d, theta)
    """
    R = T_rel[0:3, 0:3]
    p = T_rel[0:3, 3]

    # numeric guard
    def safe_atan2(y, x):
        return math.atan2(float(y), float(x))

    # alpha from R[2,1], R[2,2]
    alpha = safe_atan2(R[2,1], R[2,2])
    d = float(p[2])

    # theta from R[1,0], R[0,0]
    theta = safe_atan2(R[1,0], R[0,0])

    # a from position; watch for cos(theta) near zero
    cth = math.cos(theta)
    sth = math.sin(theta)

    if abs(cth) > 1e-6:
        a = float(p[0] / cth)
    elif abs(sth) > 1e-6:
        a = float(p[1] / sth)
    else:
        # fallback: compute length of the projection in X-Y plane
        a = float(math.hypot(p[0], p[1]))

    return a, alpha, d, theta

def compute_dh_table(xml_path, joint_names):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    # forward once to populate data.xpos / xmat etc.
    mujoco.mj_forward(model, data)

    # collect transforms for each joint
    frames = []
    z_axes = []
    jtypes = []
    jranges = []
    for jname in joint_names:
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        except Exception as e:
            raise RuntimeError(f"Joint name '{jname}' not found in model: {e}")
        Tj, z_world = make_frame_from_joint(model, data, jid)
        frames.append((jname, jid, Tj))
        z_axes.append(z_world)
        # joint type and range for info
        jtype = int(model.jnt_type[jid])  # 0: hinge, 1: slide, etc. (model-specific)
        jtypes.append(jtype)
        jranges.append(tuple(model.jnt_range[jid]))

    # compute relative transforms and extract DH params
    dh_rows = []
    for i in range(len(frames)-1):
        name_i, jid_i, T_i = frames[i]
        name_j, jid_j, T_j = frames[i+1]
        # relative transform T_i_to_j = inv(T_i) * T_j
        T_i_inv = np.eye(4)
        R_i = T_i[0:3, 0:3]; p_i = T_i[0:3, 3]
        R_i_inv = R_i.T
        T_i_inv[0:3,0:3] = R_i_inv
        T_i_inv[0:3,3] = -R_i_inv @ p_i
        T_rel = T_i_inv @ T_j

        a, alpha, d, theta = extract_dh_from_relative_transform(T_rel)

        # identify which parameter is variable by joint type of joint j (for classic DH the i-th row corresponds to joint i+1 mapping)
        # We'll report the joint type associated with transformation from i -> i+1
        row = {
            "from_joint": name_i,
            "to_joint": name_j,
            "a": a,
            "alpha": alpha,
            "d": d,
            "theta": theta,
            "joint_index": i+1,
            "joint_name": name_j,
            "joint_type": ("revolute" if model.jnt_type[jid_j] == 0 else "prismatic" if model.jnt_type[jid_j] == 1 else f"type_{int(model.jnt_type[jid_j])}"),
            "joint_range": jranges[i+1]
        }
        dh_rows.append(row)

    # For completeness, also return a "last" row from last joint to end-effector frame (if you have an EE site, you can include it)
    return dh_rows, frames

if __name__ == "__main__":
    dh_table, frames = compute_dh_table(XML_FILE_PATH, JOINT_CHAIN)

    print("Frames (world):")
    for (name, jid, T) in frames:
        pos = T[0:3,3]
        z = T[0:3,2]
        print(f"  Joint '{name}' (jid={jid}) origin={pos.round(6)} z_axis={z.round(6)}")

    print("\nDH table rows (standard DH)  — each row corresponds to transform from joint i -> joint i+1:")
    print("Columns: from_joint, to_joint, a (m), alpha (rad), d (m), theta (rad), joint_name (variable)")
    for i, row in enumerate(dh_table):
        print(f"Row {i}: from='{row['from_joint']}' -> to='{row['to_joint']}'  a={row['a']:.6f}  alpha={row['alpha']:.6f}  d={row['d']:.6f}  theta={row['theta']:.6f}  joint='{row['joint_name']}' type={row['joint_type']} range={row['joint_range']}")

    # If you want a NumPy matrix of the DH values:
    import numpy as _np
    dh_mat = _np.array([[r['a'], r['alpha'], r['d'], r['theta']] for r in dh_table])
    print("\nDH matrix (a, alpha, d, theta):\n", dh_mat)
