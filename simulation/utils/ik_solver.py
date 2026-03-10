import time
import torch
import os

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig, JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from termcolor import cprint

# Enable PyTorch performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Constants
JS_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]


class FrankaSolver:
    """Inverse Kinematics solver for Franka Emika Panda robot."""

    def __init__(self, ik_type="motion_gen", ik_sim=False, simulator="genesis", no_solver=False):
        """
        Initialize the Franka IK Solver.

        Args:
            ik_type (str): Type of IK solver to use. Options: "ik_solver" or "motion_gen"
        """
        self.tensor_args = TensorDeviceType()
        self._initialize_robot_config(ik_sim, simulator=simulator)
        if not no_solver:
            self._initialize_solver(ik_type, ik_sim, simulator=simulator)

        # Pre-allocate tensors for efficiency
        self.pos_tensor_buffer = torch.zeros(
            3, device=self.tensor_args.device, dtype=torch.float32
        )
        self.quat_tensor_buffer = torch.zeros(
            4, device=self.tensor_args.device, dtype=torch.float32
        )

    def _initialize_robot_config(self, ik_sim=False, simulator="genesis"):
        """Load and initialize robot configuration."""
        if ik_sim:
            config_file = load_yaml(join_path(os.path.dirname(__file__), f"franka_{simulator}.yml"))
        else:
            config_file = load_yaml(join_path(os.path.dirname(__file__), "franka.yml"))
        urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
        base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

        self.robot_cfg = RobotConfig.from_basic(
            urdf_file, base_link, ee_link, self.tensor_args
        )
        self.kin_model = CudaRobotModel(self.robot_cfg.kinematics)

    def _initialize_solver(self, ik_type, ik_sim=False, simulator="genesis"):
        """Initialize the appropriate solver based on ik_type."""
        if ik_type == "ik_solver":
            self._initialize_ik_solver()
        elif ik_type == "motion_gen":
            self._initialize_motion_gen(ik_sim=ik_sim, simulator=simulator)
        else:
            raise ValueError(f"Unsupported IK type: {ik_type}")

    def _initialize_ik_solver(self):
        """Initialize the IK solver."""
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=100,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(self.ik_config)

    def _initialize_motion_gen(self, ik_sim=False, simulator="genesis"):
        """Initialize the motion generator."""
        if ik_sim:
            config_file = load_yaml(join_path(os.path.dirname(__file__), f"franka_{simulator}.yml"))
        else:
            config_file = load_yaml(join_path(os.path.dirname(__file__), "franka.yml"))
        urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
        base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
        self.plan_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            tensor_args=self.tensor_args,
            interpolation_dt=0.01,
            ee_link_name=ee_link,
        )
        self.motion_gen = MotionGen(self.plan_config)
        cprint("warming up motion gen solver", "green")

        self.motion_gen.warmup(warmup_js_trajopt=False)
        self.plan_config_temp = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=4,
            max_attempts=2,
            enable_finetune_trajopt=True,
            time_dilation_factor=0.5,
        )

    def solve_ik_by_motion_gen(self, curr_joint_state, target_trans, target_quat):
        """
        Solve IK using motion generation.

        Args:
            curr_joint_state (list): Current joint state
            target_trans (list): Target position [x, y, z]
            target_quat (list): Target quaternion [w, x, y, z]

        Returns:
            list or None: Joint solution if successful, None otherwise
        """
        cu_js = JointState(
            position=self.tensor_args.to_device(curr_joint_state),
            velocity=self.tensor_args.to_device(curr_joint_state) * 0.0,
            acceleration=self.tensor_args.to_device(curr_joint_state) * 0.0,
            jerk=self.tensor_args.to_device(curr_joint_state) * 0.0,
            joint_names=JS_NAMES,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        ik_goal = Pose(
            position=self.tensor_args.to_device(target_trans),
            quaternion=self.tensor_args.to_device(target_quat),
        )
        try:
            result = self.motion_gen.plan_single(
                cu_js.unsqueeze(0), ik_goal, self.plan_config_temp
            )

            if result.success.item():
                motion_plan = result.get_interpolated_plan()
                motion_plan = self.motion_gen.get_full_js(motion_plan)
                motion_plan = motion_plan.get_ordered_joint_state(JS_NAMES)
                return motion_plan.position.cpu().numpy().tolist()
        except:
            return None
        return None

    def solve_ik(self, target_position, target_quaternion):
        """
        Solve a single IK problem for the Franka robot.

        Args:
            target_position (list): Target position [x, y, z].
            target_quaternion (list): Target quaternion [w, x, y, z].

        Returns:
            list or None: Solution (7 joint angles) if successful, otherwise None.
        """
        # Copy data to pre-allocated tensors
        self.pos_tensor_buffer.copy_(torch.tensor(target_position, dtype=torch.float32))
        self.quat_tensor_buffer.copy_(
            torch.tensor(target_quaternion, dtype=torch.float32)
        )

        # Create pose using the buffer tensors
        goal = Pose(self.pos_tensor_buffer, self.quat_tensor_buffer)

        # Solve IK
        result = self.ik_solver.solve(goal)

        if result.success.item():
            solution = result.solution.cpu().numpy().flatten()
            return solution
        return None

    def compute_fk(self, joint_angles):
        """
        Compute the forward kinematics for the Franka robot.

        Args:
            joint_angles (list): Joint angles [q1, q2, ..., q7]

        Returns:
            tuple: (position, quaternion) where position is [x, y, z] and quaternion is [w, x, y, z]
        """
        joint_angles = torch.tensor(
            joint_angles, device=self.tensor_args.device, dtype=torch.float32
        )
        out = self.kin_model.get_state(joint_angles)
        position = out.ee_position.cpu().numpy().flatten()
        quaternion = out.ee_quaternion.cpu().numpy().flatten()
        return position, quaternion