'''
This module models the problem to be solved. In this very simple example, the problem is to optimze a Robot that works in a Warehouse.
The Warehouse is divided into a rectangular grid. A Target is randomly placed on the grid and the Robot's goal is to reach the Target.
'''
import random
from enum import Enum
import sys
from os import path
import mujoco
import mujoco.viewer
import mujoco_tools
import kinematics as kinematics
import time


class ClimbingRobot:

    # Initialize the grid size. Pass in an integer seed to make randomness (Targets) repeatable.
    def __init__(self, xml_path="scene copy.xml", render_mode='human'):
        # self.reset()
        self.xml_path = xml_path
        self.render_mode = render_mode
        
        try:
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            print("Failed to load model:", e)
            return
            
        if render_mode == 'human':
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # physics timestep (seconds)
        self.physics_dt = float(self.model.opt.timestep)
        self.physics_steps = 0

    def reset(self):
        # Initialize Robot's starting position
        mujoco.mj_resetData(self.model, self.data)
    
    def step(self):
        mujoco.mj_step(self.model, self.data)
        self.physics_steps += 1   

    def render(self):
        # if self.render_mode == 'human':
        self.viewer.sync()
            # self.physics_steps = 0

    def close(self):
        # cleanup
        try:
            self.viewer.close()
        except Exception:
            pass

                


# For unit testing
if __name__=="__main__":
    cr = ClimbingRobot(xml_path="/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/scene copy.xml", render_mode='human')
    goal_right = [0.45, 0.31, 0.12]
    start_time = time.time()
    while (cr.viewer.is_running()):
        a11_force = mujoco_tools.get_contact_force(cr.model, cr.data, "assembly_11_collision_1_2", 2)
        a12_force = mujoco_tools.get_contact_force(cr.model, cr.data, "assembly_12_collision_1_2", 2)
        # joint_vals = mujoco_tools.get_joint_value(cr.model, cr.data, 
        print(mujoco_tools.get_contact_force(cr.model, cr.data, "floor", 2))
        # print("Joint values:", kinematics.ik_right(goal_right)["q"])
        # cr.data.ctrl[0:3] = kinematics.ik_right(goal_right)["q"]
        print(mujoco_tools.get_relative_transform(cr.model, cr.data,
                                        "body", "robot_base_tilted", "site", "l_grip_site")[0:3,3])
        
        print(mujoco_tools.get_relative_transform(cr.model, cr.data,
                                        "body", "robot_base_tilted", "site", "r_grip_site")[0:3,3])
        # kinematics.ik_right(goal_pose, )


        # print("Assembly 11 contact force:", a11_force)
        # print("Assembly 12 contact force:", a12_force)
        # print("r1", mujoco_tools.get_joint_value(cr.model, cr.data, "r1"))
        # print("r2", mujoco_tools.get_joint_value(cr.model, cr.data, "r2"))
        # print("r3", mujoco_tools.get_joint_value(cr.model, cr.data, "r3_1"))
        cr.step()
        if cr.render_mode == 'human':
            cr.render()

        # if time.time()-start_time > 10:
        #     cr.reset()
        #     start_time = time.time()
        #     print("-----")
        # time.sleep(1.0 / cr.fps)
    cr.close()