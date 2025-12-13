import numpy as np
import mujoco

def _body_xmat_to_R(xmat_flat):
    # data.xmat is flattened 3x3; reshape row-major -> (3,3)
    return np.array(xmat_flat, dtype=float).reshape(3,3)

def _make_T_from_body(model, data, body_id):
    """Return 4x4 transform T_world_body mapping body-frame -> world-frame."""
    R = _body_xmat_to_R(data.xmat[body_id])      # body->world rotation
    p = np.array(data.xpos[body_id], dtype=float)  # body origin in world
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,3]   = p
    return T

def _make_T_from_site(model, data, site_id):
    """Return 4x4 transform T_world_site mapping site-frame -> world-frame.
       Note: MuJoCo sites have site_xmat / site_xpos available on data."""
    # Some MuJoCo versions expose data.site_xmat; if not, we approximate by body's transform + site local pose.
    try:
        R = np.array(data.site_xmat[site_id]).reshape(3,3)
        p = np.array(data.site_xpos[site_id], dtype=float)
        T = np.eye(4)
        T[0:3,0:3] = R
        T[0:3,3]   = p
        return T
    except Exception:
        # fallback: use site_xpos and the body xmat of the site body (less accurate for rotated site frames)
        p = np.array(data.site_xpos[site_id], dtype=float)
        # attempt to find site->body mapping from model.site_... (site has .bodyid in model)
        bodyid = int(model.site_bodyid[site_id])
        R = _body_xmat_to_R(data.xmat[bodyid])
        T = np.eye(4)
        T[0:3,0:3] = R
        T[0:3,3]   = p
        return T

def get_body_pose(model, data, object_type, object_name):
    """
    Return the (position, quaternion) of a body in world coordinates.

    Args:
        model: MuJoCo model
        data:  MuJoCo data
        body_name: string name of the body
    
    Returns:
        pos:  np.array shape (3,) - world position
        quat: np.array shape (4,) - world quaternion [w, x, y, z]
    """
    # Make sure kinematics are updated
    mujoco.mj_forward(model, data)

    # Look up body id
    if object_type == "body":
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if object_id < 0:
            raise ValueError(f"Body '{object_name}' not found in the model.")
    elif object_type == "site":
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, object_name)
        if object_id < 0:
            raise ValueError(f"Body '{object_name}' not found in the model.")
    # World position and quaternion
    pos = np.array(data.xpos[object_id], dtype=float)
    quat = np.array(data.xquat[object_id], dtype=float)

    return pos, quat

def get_relative_transform(model, data, base_type, base_name, target_type, target_name):
    """
    Return transform of `target` expressed in `base` frame.
    base_type/target_type: 'body' or 'site'
    base_name/target_name: string names in the model

    Returns:
      T_base_target : 4x4 numpy array such that
        [p_target_in_base; 1] = T_base_target @ [p_target_in_world; 1]
      (equivalently) p_target_in_base = (R_base^T @ (p_target_world - p_base_world))
      Also returns (pos, R) where pos is 3-vector of target origin in base frame and R is 3x3 rotation.
    """
    # ensure kinematics current
    mujoco.mj_forward(model, data)

    # resolve ids
    def name2id(objtype, name):
        if objtype == 'body':
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        elif objtype == 'site':
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        else:
            raise ValueError("objtype must be 'body' or 'site'")

    base_id = name2id(base_type, base_name)
    target_id = name2id(target_type, target_name)

    if base_id < 0 or target_id < 0:
        raise ValueError("Could not find base or target by name")

    # build world transforms
    if base_type == 'body':
        T_world_base = _make_T_from_body(model, data, base_id)
    else:
        T_world_base = _make_T_from_site(model, data, base_id)

    if target_type == 'body':
        T_world_target = _make_T_from_body(model, data, target_id)
    else:
        T_world_target = _make_T_from_site(model, data, target_id)

    # compute T_base_target = inv(T_world_base) @ T_world_target
    Rb = T_world_base[0:3,0:3]
    pb = T_world_base[0:3,3]
    # inverse of T_world_base: [Rb^T  -Rb^T pb; 0 1]
    T_base_inv = np.eye(4)
    T_base_inv[0:3,0:3] = Rb.T
    T_base_inv[0:3,3] = -Rb.T @ pb

    T_base_target = T_base_inv @ T_world_target

    pos_in_base = T_base_target[0:3,3]
    R_in_base = T_base_target[0:3,0:3]

    return T_base_target

def get_contact_force(model, data, geom_name, axis):
    """
    Return the contact force vector (3D) acting on the geom specified by geom_name, expressed in world frame.
    axis: 0,1,2 for x,y,z components
    """
    # ensure kinematics current
    mujoco.mj_forward(model, data)

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"Could not find geom named '{geom_name}'")

    # find body id of the geom
    body_id = model.geom_bodyid[geom_id]

    # accumulate contact forces on this body
    f_total = np.zeros(3)
    ncon = data.ncon
    for i in range(ncon):
        c = data.contact[i]
        # check if this contact involves our geom
        if c.geom1 == geom_id or c.geom2 == geom_id:
            f6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f6)
            # f6[:3] is force vector in contact frame; transform to world frame
            frame = np.array(c.frame).reshape(3,3)
            f_world = frame @ f6[:3]
            f_total += f_world

    return f_total[axis]

def _find_joint_id(model, joint_name):
    """
    Try multiple ways to find a joint id from a model.
    Returns jid >= 0 on success, or -1 if not found.
    """
    # 1) Official API common method if present
    try:
        # many wrappers (official mujoco) have model.joint_name2id(name)
        jid = model.joint_name2id(joint_name)
        return int(jid)
    except Exception:
        pass

    # 2) mujoco.mj_name2id fallback (if mujoco module is imported externally)
    try:
        import mujoco
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(jid)
    except Exception:
        pass

    # 3) some wrappers expose an iterable of names
    try:
        if hasattr(model, "joint_names"):
            return int(list(model.joint_names).index(joint_name))
    except Exception:
        pass

    # 4) fallback: scan any attribute that looks like joint name array
    for attr in ("joint_name", "joint_names", "jnt_names", "names"):
        if hasattr(model, attr):
            try:
                names = list(getattr(model, attr))
                if joint_name in names:
                    return int(names.index(joint_name))
            except Exception:
                pass

    return -1

def _quat_to_euler_xyzw(q):
    """
    Convert quaternion [w, x, y, z] -> Euler (roll, pitch, yaw) in radians.
    Uses the z-y-x (yaw-pitch-roll) convention common in robotics.
    Note: MuJoCo uses [w, x, y, z].
    """
    w, x, y, z = q
    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=float)

def get_joint_value(model, data, joint_name, as_euler=False, return_slice=False):
    """
    Robustly return the qpos value(s) for a named joint.

    Args:
        model: mujoco model object (official or wrapper).
        data: mujoco data object (official or wrapper).
        joint_name: str joint name to query.
        as_euler: bool. If True and the joint qpos is a quaternion (len==4),
                  convert it to Euler angles (roll, pitch, yaw) and return that (len==3).
        return_slice: bool. If True, always return the raw qpos slice (np.ndarray),
                      even when length==1.

    Returns:
        float (scalar) if joint has 1 qpos DOF and return_slice==False,
        np.ndarray otherwise.

    Raises:
        ValueError if joint name not found.
    """
    jid = _find_joint_id(model, joint_name)
    if jid < 0:
        raise ValueError(f"Could not find joint named '{joint_name}' in model.")

    # starting address of this joint in qpos
    addr = int(model.jnt_qposadr[jid])

    # Build sorted list of all joint qpos addresses to compute span/length robustly
    # Only consider first model.njnt entries
    try:
        all_addrs = np.array([int(a) for a in model.jnt_qposadr[:model.njnt]])
    except Exception:
        # fallback - try to use full array if slicing not available
        all_addrs = np.array([int(a) for a in model.jnt_qposadr])

    # Sort addresses to find the next address after addr
    addrs_sorted = np.sort(all_addrs)
    # find index of this addr in sorted array
    idx = int(np.where(addrs_sorted == addr)[0][0])
    if idx < len(addrs_sorted) - 1:
        next_addr = int(addrs_sorted[idx + 1])
        length = next_addr - addr
    else:
        # last joint -> goes until model.nq
        length = int(model.nq) - addr

    if length <= 0:
        raise RuntimeError(f"Computed non-positive qpos length {length} for joint '{joint_name}' (addr {addr}).")

    qpos_slice = np.array(data.qpos[addr: addr + length], dtype=float)

    # Return according to preferences
    if return_slice:
        return qpos_slice

    if length == 1:
        return float(qpos_slice[0])

    # If quaternion and user asked for Euler, convert
    if as_euler and length == 4:
        # MuJoCo quaternions are [w, x, y, z]
        q = qpos_slice.copy()
        # normalize for safety
        q = q / np.linalg.norm(q)
        return _quat_to_euler_xyzw(q)

    return qpos_slice

def tf_to_pos_quat(T):
    """
    Convert a 4x4 transform matrix to (pos, quat).

    Args:
        T: (4,4) homogeneous transform, rotation in T[:3,:3], translation in T[:3,3].

    Returns:
        pos:  (3,) numpy array
        quat: (4,) numpy array in [w, x, y, z] format
    """
    # extract position
    pos = np.array(T[0:3, 3], dtype=float)

    # extract rotation matrix
    R = np.array(T[0:3, 0:3], dtype=float)

    # convert to quaternion
    quat = rotmat_to_quat(R)  # [w, x, y, z]
    return pos, quat

def rotmat_to_quat(R):
    R = np.asarray(R, dtype=float).reshape(3,3)
    t = R[0,0] + R[1,1] + R[2,2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    quat /= np.linalg.norm(quat)
    return quat
