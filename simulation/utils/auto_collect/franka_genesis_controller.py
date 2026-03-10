import sys
import numpy as np
import os
sys.path.insert(0, os.getcwd())
from simulation.utils.ik_solver import FrankaSolver
import h5py
from simulation.utils.auto_collect.utils import rotate_quaternion_around_world_z_axis
from simulation.utils.constants import DEFAULT_JOINT_ANGLES, JOINT_NAMES
import torch
try:
    import rospy
    from std_msgs.msg import Float64MultiArray, Bool
    ROSPY_ENABLED = True
except:
    ROSPY_ENABLED = False
    print("rospy not loaded. Teleop mode will be disabled.")

class franka_controller:
    def __init__(self, scene, robot, close_thres, teleop=False, evaluation=False, default_gripper_state=False, default_joint_angles=DEFAULT_JOINT_ANGLES):
        """
        Initialize the Franka controller with the specified IK type and simulation settings.

        Args:
            ik_type (str): Type of IK solver to use. Options: "ik_solver" or "motion_gen"
            ik_sim (bool): Whether to use simulation mode.
            simulator (str): The simulator to use, e.g., "genesis".
        """
        self.scene = scene
        self.franka = robot
        if evaluation:
            # Use IK in policy
            self.franka_solver = FrankaSolver(ik_type="motion_gen", ik_sim=False, simulator=None, no_solver=True)
        elif teleop:
            # Use IK in teleop mode
            self.franka_solver = FrankaSolver(ik_type="motion_gen", ik_sim=True, simulator="genesis", no_solver=True)
        else:
            # Use IK in simulator
            self.franka_solver = FrankaSolver(ik_type="motion_gen", ik_sim=True, simulator="genesis", no_solver=False)
        self.real_franka_solver = FrankaSolver(ik_type="motion_gen", ik_sim=False, simulator=None, no_solver=True)
        self.record_started = False
        self.default_joint_angles = default_joint_angles
        self.close_state = [close_thres / 100, close_thres / 100]
        self.open_state = [0.04, 0.04]
        self.current_control = np.array(self.default_joint_angles[:7])
        self.current_gripper_control = default_gripper_state
        self.default_gripper_state = default_gripper_state
        self.teleop = teleop
        self.evaluation = evaluation
        self.all_dof_ids = [robot.get_joint(name).dof_idx for name in JOINT_NAMES]
        if self.default_gripper_state:
            self.franka.set_dofs_position(
                self.close_state,
                self.all_dof_ids[7:9], 
            )
            self.default_joint_angles[7:9] = self.close_state
        else:
            self.franka.set_dofs_position(
                self.open_state,
                self.all_dof_ids[7:9], 
            )
            self.default_joint_angles[7:9] = self.open_state

        if teleop or evaluation:
            if ROSPY_ENABLED:
                self.init_teleop()
            else:
                raise RuntimeError("rospy is not enabled! Install ROS first if in teleop mode.")
        
    def _callback_joint_control(self, msg):
        self.current_control = np.array(msg.data)
        self.franka.control_dofs_position(
            np.array(msg.data),
            self.all_dof_ids[:7],
        )

    def _callback_gripper_control(self, msg):
        if not self.default_gripper_state:
            if msg.data:
                self.current_gripper_control = True
                self.franka.control_dofs_position(
                    self.close_state,
                    self.all_dof_ids[7:9], 
                )
            else:
                self.current_gripper_control = False
                self.franka.control_dofs_position(
                    self.open_state,
                    self.all_dof_ids[7:9], 
                )
            
    def init_teleop(self):
        rospy.init_node('genesis_sim', anonymous=True)
        # Use ROS to publish pred_mano_params to the client
        self.pub_joint = rospy.Publisher('/genesis/joint_states', Float64MultiArray, queue_size=1)
        self.pub_ee    = rospy.Publisher('/genesis/ee_states', Float64MultiArray, queue_size=1)
        self.sub_joint_control = rospy.Subscriber(
            "/genesis/joint_control",
            Float64MultiArray,
            self._callback_joint_control,
            queue_size=1,
        )
        self.sub_gripper_control = rospy.Subscriber(
            "/genesis/gripper_control",
            Bool,
            self._callback_gripper_control,
            queue_size=1,
        )

    def publish_states(self):
        joint_pos = self.franka.get_dofs_position()
        joint_pos = joint_pos.cpu().numpy()
        joint_pos_list = joint_pos.flatten().tolist()
        joint_pos_msg = Float64MultiArray(data=joint_pos_list)
        self.pub_joint.publish(joint_pos_msg)
        
        joint_pos = self.franka.get_dofs_position().cpu().numpy()
        trans, rot_quat = self.franka_solver.compute_fk(joint_pos)
        current_ee_pose = np.concatenate([trans, rot_quat])
        current_ee_pose_list = current_ee_pose.flatten().tolist()
        current_ee_msg = Float64MultiArray(data=current_ee_pose_list)
        self.pub_ee.publish(current_ee_msg)

    def move_to_goal(self, pos, quat, gripper_open=True):
        """
        Move the Franka robot to the specified goal position and orientation.

        Args:
            pos (list or np.ndarray): The target position in Cartesian coordinates.
            quat (list or np.ndarray): The target orientation as a quaternion.
        """
        joint_pos = self.franka.get_dofs_position().cpu().numpy()
        current_joint_angles = joint_pos[:7]  # Current joint angles
        result = self.franka_solver.solve_ik_by_motion_gen(
            curr_joint_state=current_joint_angles, 
            target_trans=pos,
            target_quat=quat,
        )
        if gripper_open:
            self.current_gripper_control = False # gripper open --> gripper = False
        else:
            self.current_gripper_control = True  # gripper close --> gripper = True
            
        if result and len(result):
            for waypoint in result:
                self.current_control = np.array(waypoint)
                if gripper_open:
                    self.current_gripper_control = False # gripper open --> gripper = False
                else:
                    self.current_gripper_control = True  # gripper close --> gripper = True
                self.franka.control_dofs_position((waypoint+self.open_state if gripper_open else waypoint+self.close_state))
                self.step()
                self.step()
        else:
            raise RuntimeError("IK Failed")
        
        joint_pos = self.franka.get_dofs_position().cpu().numpy()
        ee_position, ee_quat = self.franka_solver.compute_fk(joint_pos)
        dist = np.sqrt(np.sum((ee_position - pos) ** 2))
        print(dist)
        loop_count = 0
        while dist > 1e-2 :
            self.step()
            joint_pos = self.franka.get_dofs_position().cpu().numpy()
            ee_position, ee_quat = self.franka_solver.compute_fk(joint_pos)
            dist = np.sqrt(np.sum((ee_position - pos) ** 2))
            print(dist)
            loop_count += 1
            if loop_count > 1000:
                raise RuntimeError
    def close_gripper(self, wait_steps=100):
        """
        Close the gripper of the Franka robot.
        """
        joint_pos = self.franka.get_dofs_position().cpu().numpy()
        current_joint_angles = joint_pos
        current_joint_angles[7:] = self.close_state
        self.current_gripper_control = True
        self.franka.control_dofs_position(current_joint_angles)
        for i in range(wait_steps):
            self.current_gripper_control = True
            self.step()

    def open_gripper(self, wait_steps=100):
        """
        Open the gripper of the Franka robot.
        """
        joint_pos = self.franka.get_dofs_position().cpu().numpy()
        current_joint_angles = joint_pos
        current_joint_angles[7:] = self.open_state
        self.current_gripper_control = False
        self.franka.control_dofs_position(current_joint_angles)
        for i in range(wait_steps):
            self.current_gripper_control = False
            self.step()

    def reset_franka(self):
        """
        Reset the Franka robot to its initial state.
        """
        self.current_control = np.array(self.default_joint_angles[:7])
        self.current_gripper_control = self.default_gripper_state
        self.franka.set_dofs_position(self.default_joint_angles)
        self.scene.step()
        
    def step(self):
        """
        Step the simulation forward.
        """
        if self.default_gripper_state:
            self.franka.control_dofs_position(
                self.close_state,
                self.all_dof_ids[7:9], 
            )
        if self.teleop or self.evaluation:
            self.publish_states()
        self.scene.step()
        if self.record_started:
            self.record_step()

    def record_step(self):
        raise NotImplementedError("Recording functionality is not implemented yet.")
            
class pick_and_place_controller(franka_controller):
    """Controller for pick and place operations using the Franka robot.
    Inherits from the `franka_controller` class.
    """
    def __init__(self, scene, scene_config, robot, object_active, object_passive, default_poses, close_thres=1, teleop=False, evaluation=False,
                 physics_params=[1, 0, 0, 0, 0],
                 default_joint_angles=DEFAULT_JOINT_ANGLES):
        """
        Initialize the pick and place controller.

        Args:
            scene: The simulation scene.
            robot: The Franka robot instance.
            ik_type (str): Type of IK solver to use. Options: "ik_solver" or "motion_gen".
            simulator (str): The simulator to use, e.g., "genesis".
        """
        super().__init__(scene, robot, close_thres, teleop=teleop, evaluation=evaluation, default_gripper_state=False, default_joint_angles=default_joint_angles)
        self.scene_config = scene_config
        self.object_active = object_active
        self.object_passive = object_passive
        self.default_poses = default_poses
        
        self.franka.set_dofs_position(self.default_joint_angles)
        self.scene.step()
        self.default_hand_quat = self.franka.get_link("hand").get_quat().cpu().numpy()
        self.default_hand_pos = self.franka.get_link("hand").get_pos().cpu().numpy()
        self.record_started = False
        self.timestamp = 0
        self.record = []
        self.physics_params = {
            "friction_ratio": physics_params[0],
            "mass_shift": physics_params[1],
            "com_shift": physics_params[2:5],
        }        
        # Record related variables
        self.traj_cnt = 0
        
    def reset_scene(self, given_data=None):
        def set_friction_ratio(rigid_entity, friction_ratio, link_indices, envs_idx=None):
            """
            Set the friction ratio of the geoms of the specified links.
            Parameters
            ----------
            friction_ratio : torch.Tensor, shape (n_envs, n_links)
                The friction ratio
            link_indices : array_like
                The indices of the links to set friction ratio.
            envs_idx : None | array_like, optional
                The indices of the environments. If None, all environments will be considered. Defaults to None.
            """
            geom_indices = []
            for i in link_indices:
                for j in range(rigid_entity._links[i].n_geoms):
                    geom_indices.append(rigid_entity._links[i]._geom_start + j)
            rigid_entity._solver.set_geoms_friction_ratio(
                torch.cat(
                    [
                        ratio.unsqueeze(-1).repeat(1, rigid_entity._links[j].n_geoms)
                        for j, ratio in zip(link_indices, friction_ratio.unbind(-1))
                    ],
                    dim=-1,
                ).squeeze(0),
                geom_indices,
                envs_idx,
            )
            return rigid_entity

        self.franka.set_dofs_position(self.default_joint_angles)
        self.franka.control_dofs_position(self.default_joint_angles)
        
        if given_data:
            self.object_active.set_pos(given_data["object_states"]["active"][:3])
            self.object_passive.set_pos(given_data["object_states"]["passive"][:3])
            self.object_active.set_quat(given_data["object_states"]["active"][3:7])
            self.object_passive.set_quat(given_data["object_states"]["passive"][3:7])
        else:
            active_x_min = self.scene_config.object_active.pos_range.x[0] / 100
            active_x_max = self.scene_config.object_active.pos_range.x[1] / 100
            active_y_min = self.scene_config.object_active.pos_range.y[0] / 100
            active_y_max = self.scene_config.object_active.pos_range.y[1] / 100
            rand_x = np.random.rand() * (active_x_max - active_x_min) + active_x_min
            rand_y = np.random.rand() * (active_y_max - active_y_min) + active_y_min
            fixed_z = self.default_poses["active_pos"][2]
            self.object_active.set_pos(np.array([rand_x, rand_y, fixed_z]))

            passive_x_min = self.scene_config.object_passive.pos_range.x[0] / 100
            passive_x_max = self.scene_config.object_passive.pos_range.x[1] / 100
            passive_y_min = self.scene_config.object_passive.pos_range.y[0] / 100
            passive_y_max = self.scene_config.object_passive.pos_range.y[1] / 100
            rand_x = np.random.rand() * (passive_x_max - passive_x_min) + passive_x_min
            rand_y = np.random.rand() * (passive_y_max - passive_y_min) + passive_y_min
            fixed_z = self.default_poses["passive_pos"][2]
            self.object_passive.set_pos(np.array([rand_x, rand_y, fixed_z]))


            rand_angle = np.random.rand() * 360
            # wxyz --> xyzw
            quat_scipy = np.array([self.default_poses["active_quat"][1], self.default_poses["active_quat"][2], self.default_poses["active_quat"][3], self.default_poses["active_quat"][0]])
            quat_scipy = rotate_quaternion_around_world_z_axis(quat_scipy, rand_angle)
            # xyzw --> wxyz
            rand_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
            self.object_active.set_quat(rand_quat)

            rand_angle = np.random.rand() * 360
            # wxyz --> xyzw
            quat_scipy = np.array([self.default_poses["passive_quat"][1], self.default_poses["passive_quat"][2], self.default_poses["passive_quat"][3], self.default_poses["passive_quat"][0]])
            quat_scipy = rotate_quaternion_around_world_z_axis(quat_scipy, rand_angle)
            # xyzw --> wxyz
            rand_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
            self.object_passive.set_quat(rand_quat)
        
        # Set physics properties
        set_friction_ratio(self.object_active, torch.tensor([self.physics_params["friction_ratio"]]).to("cuda"), link_indices = [0],)
        self.object_active.set_mass_shift(torch.tensor([self.physics_params["mass_shift"]]).to("cuda"), link_indices = [0])
        self.object_active.set_COM_shift(
            com_shift=torch.tensor(self.physics_params["com_shift"]).to("cuda").unsqueeze(0),
            link_indices = [0],
        )
        if not self.teleop and not self.evaluation:
            self.scene.step()
    
    def get_states(self):
        object_active_pos_state = self.object_active.get_pos()
        object_active_quat_state = self.object_active.get_quat()
        object_passive_pos_state = self.object_passive.get_pos()
        object_passive_quat_state = self.object_passive.get_quat()
        
        joint_pos = self.franka.get_dofs_position().cpu().numpy()
        trans, rot_quat = self.real_franka_solver.compute_fk(joint_pos)
        current_ee_pose = np.concatenate([trans, rot_quat])
        current_ee_control = np.concatenate(self.real_franka_solver.compute_fk(self.current_control))
        states = {
            "timestamp": np.array([self.timestamp]),
            "joint_states": joint_pos, 
            "ee_states": current_ee_pose,
            "joint_control": self.current_control,
            "ee_control": current_ee_control,
            "gripper_control": np.array([self.current_gripper_control]),
            "object_states": {
                "active": np.concatenate([object_active_pos_state.cpu().numpy(), object_active_quat_state.cpu().numpy()]),
                "passive": np.concatenate([object_passive_pos_state.cpu().numpy(), object_passive_quat_state.cpu().numpy()]),
            },
        }
        return states
    
    def record_step(self):
        self.record.append(self.get_states())
        self.timestamp += 1

    def save_traj(self, output_dir):
        def save_dict_to_hdf5(dic, filename):
            with h5py.File(filename, 'w') as h5file:
                _save_dict_to_hdf5(dic, h5file)

        def _save_dict_to_hdf5(dic, h5grp):
            for key, item in dic.items():
                if isinstance(item, dict):
                    # Create group and recursively save sub-dictionaries
                    subgroup = h5grp.create_group(key)
                    _save_dict_to_hdf5(item, subgroup)
                elif isinstance(item, np.ndarray) or isinstance(item, list):
                    # Save numpy array (with compression enabled)
                    h5grp.create_dataset(key, data=np.array(item), compression="gzip")
                else:
                    print(f"Unknown item {key}, {type(item)}")
        self.record_started = False
        current_traj_dir = os.path.join(output_dir, "{:05d}".format(self.traj_cnt))
        os.makedirs(current_traj_dir, exist_ok=True)
        for idx, state in enumerate(self.record):
            save_dict_to_hdf5(state, os.path.join(current_traj_dir, f"{idx}.h5"))
        self.clean_traj()
        self.traj_cnt += 1

    def clean_traj(self):
        self.record = []
    
    def start_record(self):
        self.record_started = True
        self.record = []
        self.timestamp = 0

    def end_record(self):
        self.record_started = False