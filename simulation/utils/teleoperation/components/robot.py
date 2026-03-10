'''
Credits: https://github.com/real-dex-suite/REAL-ROBO
'''
import rospy
import warnings
import os
import sys

sys.path.insert(0, os.getcwd())
from simulation.utils.teleoperation.franka_genesis_env_wrapper import FrankaGenesisEnvWrapper
warnings.filterwarnings(
    "ignore",
    message="Link .* is of type 'fixed' but set as active in the active_links_mask.*",
)
    
class RobotController(object):
    def __init__(
        self,
        teleop,
        arm_type="franka",
        simulator=None,
        gripper=None,
        gripper_init_state="open",
    ) -> None:
        self.arm = FrankaGenesisEnvWrapper(gripper=gripper, gripper_init_state=gripper_init_state) 

    def home_robot(self):
        self.arm.home_robot()

    def get_arm_position(self):
        return self.arm.get_arm_position()
    
    def get_arm_tcp_position(self):
        return self.arm.get_tcp_position()

    def move(self, input_cmd):
        self.arm.move(input_cmd)