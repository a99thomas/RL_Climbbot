"""
view_hang.py  —  open the climbing ladder scene at its hang pose in the MuJoCo viewer.

macOS (required):   mjpython view_hang.py
   add --sim to let physics run (watch it settle/fall):   mjpython view_hang.py --sim
   --scene / --qpos to override the defaults.

This imports ONLY mujoco + numpy (no gym), so it's clean and fast.
"""
import os, sys, time, argparse
import numpy as np
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=os.path.join(ASSETS, "scene_v7_ladder.xml"))
    ap.add_argument("--qpos", default=os.path.join(HERE, "hang_start_ladder.npy"))
    ap.add_argument("--sim", action="store_true", help="run physics instead of holding the pose")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    if os.path.exists(args.qpos):
        q = np.load(args.qpos)
        if q.shape[0] == m.nq:
            d.qpos[:] = q
            print("loaded hang pose:", args.qpos)
        else:
            print(f"qpos size {q.shape[0]} != model nq {m.nq}; ignoring")
    mujoco.mj_forward(m, d)
    print("scene:", args.scene, "| nq", m.nq, "| sim:", args.sim)
    print("drag to orbit; press Esc / close window to quit.")

    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            if args.sim:
                mujoco.mj_step(m, d)
            else:
                mujoco.mj_forward(m, d)   # hold the static pose
            v.sync()
            time.sleep(0.005)


if __name__ == "__main__":
    main()
