'''
Credits: https://github.com/real-dex-suite/REAL-ROBO
'''
import os
import sys
sys.path.insert(0, os.getcwd())

from scipy.spatial.transform import Rotation as R
from termcolor import cprint
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
import xrobotoolkit_sdk as xrt
from .robot import RobotController

def swap_y_z_axis(T):
    """
    Swap Y and Z axes in a 4x4 transformation matrix.
    
    Args:
        T (np.ndarray): 4x4 transformation matrix
    
    Returns:
        np.ndarray: New transformation matrix with Y and Z swapped
    """
    # Make a copy to avoid modifying the original
    T_new = T.copy()
    
    # Swap rotation rows (Y and Z)
    T_new[1, :], T_new[2, :] = T[2, :], T[1, :]
    
    # Swap rotation columns (Y and Z)
    T_new[:, 1], T_new[:, 2] = T_new[:, 2], T_new[:, 1].copy()
    
    return T_new

def rbu_to_flu(T_rbu):
    """
    Convert a transformation matrix from RBU (Right, Back, Up) to FLU (Front, Left, Up).
    FLU front = -RBU back (-Y), FLU left = -RBU right (-X), FLU up = RBU up (Z).
    
    Args:
        T_rbu (np.ndarray): 4x4 transformation matrix in RBU coordinates
    
    Returns:
        np.ndarray: 4x4 transformation matrix in FLU coordinates
    """
    C = np.array([
        [0, -1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    C_inv = C.T
    T_flu = C @ T_rbu @ C_inv
    return T_flu

def remove_euler_component_scipy(quat, remove_roll=False, remove_pitch=False, remove_yaw=False):
    """
    Remove specified Euler angle components from a quaternion using SciPy
    
    Args:
        quat: Input quaternion in [w, x, y, z] format (scalar first)
        remove_roll: Flag to remove roll component (x-axis rotation)
        remove_pitch: Flag to remove pitch component (y-axis rotation)
        remove_yaw: Flag to remove yaw component (z-axis rotation)
        
    Returns:
        New quaternion with specified components removed [w, x, y, z]
    """
    # Note: SciPy uses xyzw format (scalar last), so we need to convert input
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    
    # Create Rotation object from quaternion
    rotation = R.from_quat(quat_xyzw)
    
    # Extract Euler angles (using 'xyz' convention: roll, pitch, yaw)
    euler_angles = rotation.as_euler('xyz', degrees=False)
    
    # Zero out the components we want to remove
    if remove_roll:
        euler_angles[0] = 0.0  # Roll (x-axis)
    if remove_pitch:
        euler_angles[1] = 0.0  # Pitch (y-axis)
    if remove_yaw:
        euler_angles[2] = 0.0  # Yaw (z-axis)
    
    # Create new rotation from modified Euler angles
    new_rotation = R.from_euler('xyz', euler_angles)
    
    # Convert back to quaternion and adjust to wxyz format
    new_quat_xyzw = new_rotation.as_quat()
    new_quat = np.array([new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]])
    
    return new_quat

class PICOArmTeleOp:
    def __init__(self, simulator=None, gripper=None, arm_type="franka", gripper_init_state="open", lock_rotation=["pitch", "roll"], lock_z=False):
        self.trans_scale = 1
        self.gripper_control = float(gripper_init_state == "close")

        self._setup_xrt()

        # Initialize robot controller
        self.robot = RobotController(teleop=True, simulator=simulator, gripper=gripper, arm_type=arm_type, gripper_init_state=gripper_init_state)
        self.init_arm_ee_pose = self._get_tcp_position()
        self.init_arm_ee_to_world = np.eye(4)
        self.init_arm_ee_to_world[:3, 3] = self.init_arm_ee_pose[:3]
        self.init_arm_ee_to_world[:3, :3] = quat2mat(self.init_arm_ee_pose[3:7])
        self.joystick_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # xyz, wxyz
        self.lock_rotation = lock_rotation
        self.lock_z = lock_z
        # Only use relative pose when right_grip is True; record the pose when the grip is pressed for the first time (robot frame 4x4)
        self._first_grip_T_robot = None

    def __del__(self):
        self._close_xrt()
        
    def _setup_xrt(self):
        xrt.init()

    def _close_xrt(self):
        xrt.close()
        
    def _get_joystick_pose(self):
        # xrt returns pose as [x, y, z, qx, qy, qz, qw] (xyzw), in VR left-handed frame (right-up-back)
        pose = xrt.get_right_controller_pose()
        right_grip = xrt.get_right_grip() > 0.5
        if not right_grip:
            self._first_grip_T_robot = None
            return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Identity pose
        pos = np.array(pose[:3])
        quat_xyzw = pose[3:]  # [qx, qy, qz, qw]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        # VR controller coordinates (right-up-back) -> Robot coordinates (front-left-up)
        rot = quat2mat(quat_wxyz)
        transmat = np.eye(4)
        transmat[:3, :3] = rot
        transmat[:3, 3] = pos
        transmat = swap_y_z_axis(transmat)
        transmat = rbu_to_flu(transmat)
        T_robot = transmat.copy()
        if self._first_grip_T_robot is None:
            self._first_grip_T_robot = T_robot.copy()
        T_rel = np.linalg.inv(self._first_grip_T_robot) @ T_robot
        pos = T_rel[:3, 3]
        quat_wxyz = mat2quat(T_rel[:3, :3])
        return np.concatenate((pos, quat_wxyz), axis=0)

    def _get_gripper_control(self):
        return xrt.get_right_trigger() > 0.5

    def _get_tcp_position(self):
        """Get the TCP position based on the arm type"""
        return self.robot.arm.get_tcp_position()

    def _retarget_base(self):
        """Retarget the base position of the robot arm"""
        self.joystick_pose = self._get_joystick_pose()
        self.gripper_control = self._get_gripper_control()
        current_arm_pose = self.init_arm_ee_pose.copy()
        if self.lock_z:
            # If lock_z is enabled, do not update z axis
            current_arm_pose[:2]  = self.joystick_pose[:2] * self.trans_scale + self.init_arm_ee_to_world[:2, 3]
            current_arm_pose[2:3] = self.init_arm_ee_to_world[2:3, 3].copy()
        else:
            current_arm_pose[:3]  = self.joystick_pose[:3] * self.trans_scale + self.init_arm_ee_to_world[:3, 3]
        # NOTE: quat is wxyz.
        filtered_quat = remove_euler_component_scipy(self.joystick_pose[3:7], 
                                                    remove_roll="roll" in self.lock_rotation,
                                                    remove_pitch="pitch" in self.lock_rotation,
                                                    remove_yaw="yaw" in self.lock_rotation,)
        current_arm_pose[3:7] = mat2quat(quat2mat(filtered_quat) @ self.init_arm_ee_to_world[:3, :3])
        if self.robot.arm.with_gripper:
            current_arm_pose = np.concatenate([current_arm_pose, np.expand_dims(self.gripper_control, axis=0)])
        return current_arm_pose
    
    def move(self):
        """Main control loop for robot movement"""
        print("\n" + "*" * 78)
        cprint("[   ok   ]     Controller initiated. ", "green", attrs=["bold"])
        print("*" * 78 + "\n")
        print("Start controlling the robot hand using the PICO VR.\n")

        while True:
            desired_cmd = self._retarget_base()
            self.robot.move(desired_cmd)