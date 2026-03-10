'''
Credits: https://github.com/real-dex-suite/REAL-ROBO
'''
import rospy
from copy import deepcopy as copy
from termcolor import cprint
from pynput import keyboard
import numpy as np
from .robot import RobotController
from transforms3d.quaternions import quat2mat, mat2quat
from scipy.spatial.transform import Rotation as R

# Movement step sizes
translation_step = 0.01  # meters
rotation_step = 0.05      # radians

class KBArmTeleop(object):
    def __init__(self, simulator=None, gripper=None, arm_type="franka", gripper_init_state="open", lock_rotation=["pitch", "roll"], lock_z=False):
        self.arm_type = arm_type
        self.robot = RobotController(
            teleop=True,
            simulator=simulator,
            gripper=gripper,
            arm_type=arm_type,
            gripper_init_state=gripper_init_state,
        )
        self.lock_rotation = lock_rotation
        self.lock_z = lock_z
        # Pose delta: [x, y, z, roll, pitch, yaw]
        self.desired_delta_pose = np.zeros(6)  # xyz, rpy, gripper
        self.desired_gripper_state = 0.0
        self.gripper_processed = True
        
    def _on_press(self, key):
        try:
            if key.char == "w":
                self.desired_delta_pose = np.array([translation_step, 0.0, 0.0, 0.0, 0.0, 0.0])
            elif key.char == "s":
                self.desired_delta_pose = np.array([-translation_step, 0.0, 0.0, 0.0, 0.0, 0.0])
            elif key.char == "a":
                self.desired_delta_pose = np.array([0.0, translation_step, 0.0, 0.0, 0.0, 0.0])
            elif key.char == "d":
                self.desired_delta_pose = np.array([0.0, -translation_step, 0.0, 0.0, 0.0, 0.0])
            elif key.char == "q":
                if not self.lock_z:
                    self.desired_delta_pose = np.array([0.0, 0.0, translation_step, 0.0, 0.0, 0.0])
            elif key.char == "e":
                if not self.lock_z:
                    self.desired_delta_pose = np.array([0.0, 0.0, -translation_step, 0.0, 0.0, 0.0])
            elif key.char == "r":
                if not "roll" in self.lock_rotation:
                    self.desired_delta_pose = np.array([0.0, 0.0, 0.0, rotation_step, 0.0, 0.0])
            elif key.char == "f":
                if not "roll" in self.lock_rotation:
                    self.desired_delta_pose = np.array([0.0, 0.0, 0.0, -rotation_step, 0.0, 0.0])
            elif key.char == "t":
                if not "pitch" in self.lock_rotation:
                    self.desired_delta_pose = np.array([0.0, 0.0, 0.0, 0.0, rotation_step, 0.0])
            elif key.char == "g":
                if not "pitch" in self.lock_rotation:
                    self.desired_delta_pose = np.array([0.0, 0.0, 0.0, 0.0, -rotation_step, 0.0])
            elif key.char == "y":
                if not "yaw" in self.lock_rotation:
                    self.desired_delta_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, rotation_step])
            elif key.char == "h":
                if not "yaw" in self.lock_rotation:
                    self.desired_delta_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -rotation_step])
            elif key.char == "i":
                self.desired_gripper_state = 1.0
                self.gripper_processed = False
            elif key.char == "o":
                self.desired_gripper_state = 0.0
                self.gripper_processed = False
            else:
                cprint(f"Unknown key pressed: {key.char}", "red", attrs=["bold"])
        except AttributeError:
            cprint(f"Special key {key} pressed", "red", attrs=["bold"])

    def move(self):
        cprint("\n" + "*" * 78, "green", attrs=["bold"])
        cprint("[   ok   ]     Controller initiated. ", "green", attrs=["bold"])
        cprint("*" * 78 + "\n", "green", attrs=["bold"])
        cprint("Start controlling the robot arm using the Keyboard.\n", "green", attrs=["bold"])

        keyboard_listener = keyboard.Listener(on_press=self._on_press)
        keyboard_listener.start()

        while True:
            if not np.all(self.desired_delta_pose == np.zeros(6)) or not self.gripper_processed:
                init_arm_ee_pose = self.robot.get_arm_tcp_position()
                # Convert both init_arm_ee_pose (could be joint state or pose) and self.desired_delta_pose (delta translation + delta rotation) to xyz+wxyz
                pose_xyz = np.array(init_arm_ee_pose[:3])
                pose_quat = np.array(init_arm_ee_pose[3:7])  # assume quaternion order is wxyz

                delta_xyz = np.array(self.desired_delta_pose[:3])
                # roll, pitch, yaw small angles = axis-angle (rotvec) in radians about fixed x, y, z axis
                delta_rot_rotvec = np.array(self.desired_delta_pose[3:6])

                # Apply translation (add delta to xyz)
                new_xyz = pose_xyz + delta_xyz

                # Axis-angle to quaternion: rotvec = (roll, pitch, yaw), i.e., rotate around x/y/z axes
                delta_rot = R.from_rotvec(delta_rot_rotvec).as_quat()  # returns xyzw
                delta_quat = np.array([delta_rot[3], delta_rot[0], delta_rot[1], delta_rot[2]])  # xyzw -> wxyz

                # Quaternion multiplication (q_new = delta_quat * pose_quat)
                # transforms3d uses wxyz order
                # Multiplies as: q_new = q2 * q1, meaning q2 applied after q1
                new_quat = mat2quat(quat2mat(delta_quat) @ quat2mat(pose_quat))  # still wxyz

                # Concatenate new translation and quaternion to form new pose: xyz + wxyz
                new_arm_ee_pose = np.concatenate([new_xyz, new_quat])
                cprint(new_arm_ee_pose, "yellow", attrs=["bold"])
                self.robot.move(np.concatenate([new_arm_ee_pose, [self.desired_gripper_state]]))
                self.desired_delta_pose = np.zeros(6)
                self.gripper_processed = True
            else:
                continue


if __name__ == "__main__":
    teleop = KBArmTeleop()
    teleop.move()
