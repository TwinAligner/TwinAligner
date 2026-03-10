import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import sys

sys.path.insert(0, os.getcwd())
import genesis as gs
import yaml
from pathlib import Path
from easydict import EasyDict
from simulation.utils.constants import XML_DEFAULT_CFG, BEST_PARAMS, JOINT_NAMES
from simulation.gs_viewer import render_and_save_specific_view, GenesisGaussianViewer
from simulation.utils.auto_collect.utils import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix
from simulation.utils.auto_collect.franka_genesis_controller import pick_and_place_controller
from termcolor import cprint
from scipy.spatial.transform import Rotation as R

def init_anygrasp():
    sys.path.insert(0, "simulation/anygrasp_sdk/grasp_detection")
    from gsnet import AnyGrasp
    args = argparse.Namespace()
    args.checkpoint_path = "checkpoints/checkpoint_detection.tar"
    args.gripper_height = 0.107 # franka ee to object
    args.max_gripper_width = 0.08 # franka gripper width
    args.top_down_grasp = True
    args.debug = False
    anygrasp = AnyGrasp(args)
    anygrasp.load_net()
    return anygrasp

def rotate_axis_quaternion(ori_axis):
    if ori_axis == 'x':
        return (np.sqrt(2) / 2, 0, -np.sqrt(2) / 2, 0)
    elif ori_axis == 'y':
        return (np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0)
    elif ori_axis == 'z':
        return (1, 0, 0, 0)
    elif ori_axis == "-x":
        return (np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0)
    elif ori_axis == "-y":
        return (np.sqrt(2) / 2, -np.sqrt(2) / 2, 0, 0)
    elif ori_axis == "-z":
        return (0, 1, 0, 0)
    else:
        return (1, 0, 0, 0)

def get_object_bbox_and_height(object_ply, ori_axis):
    ply_o3d = o3d.io.read_triangle_mesh(object_ply)
    bbox = ply_o3d.get_axis_aligned_bounding_box()
    bbox_minbound = bbox.get_min_bound()
    bbox_maxbound = bbox.get_max_bound()
    if "x" in ori_axis:
        height = (bbox_maxbound[0] - bbox_minbound[0])
    elif "y" in ori_axis:
        height = (bbox_maxbound[1] - bbox_minbound[1])
    elif "z" in ori_axis:
        height = (bbox_maxbound[2] - bbox_minbound[2])
    else:
        raise NotImplementedError(f"unknown axis {ori_axis}")
    return bbox, height

def design_scene(scene_config, show_viewer=True):
    scene = gs.Scene(
        viewer_options = gs.options.ViewerOptions(
            camera_fov    = 60,
            max_FPS       = 30,
        ),
        vis_options = gs.options.VisOptions(
            show_world_frame = False,
            background_color=(0.8, 0.8, 0.8),
            ambient_light=(0.3, 0.3, 0.3),
        ),
        sim_options = gs.options.SimOptions(
            dt = 0.01,
            substeps=2,
        ),
        show_viewer = show_viewer,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file  = scene_config.robot.genesis_asset,
            pos   = (0, 0, 0),
            quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
            scale = 1.0,
        ),
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )
    background = scene.add_entity(
        gs.morphs.Plane(pos   = (0, 0, -0.008)),
        material=gs.materials.Rigid(friction=0.1),
    )
    grasp_cam = scene.add_camera(
        res    = (1280, 720),
        pos    = (0.62, -0.05, 1.0),
        lookat = (0.62, -0.05, 0.0),
        fov    = 30,
        GUI    = False
    )
    object_active_bbox, object_active_height = get_object_bbox_and_height(scene_config.object_active.mesh_asset, scene_config.object_active.up_axis)
    # cm --> m
    active_pos   = (scene_config.object_active.default_position.x / 100,
                    scene_config.object_active.default_position.y / 100, 
                    object_active_height / 2 - scene_config.robot.robot_to_table_height / 100 + scene_config.object_active.height_thres / 100)
    active_quat  = rotate_axis_quaternion(scene_config.object_active.up_axis)
    object_active = scene.add_entity(
        material=gs.materials.Rigid(rho=3000, friction=0.1),
        morph=gs.morphs.URDF(
            file  = scene_config.object_active.asset,
            pos   = active_pos,
            quat  = active_quat, # we use w-x-y-z convention for quaternions,
            scale = 1.0,
            convexify=False,
            fixed=False,
        ),
        visualize_contact=False,
        vis_mode = "collision",
    )
    object_passive_bbox, object_passive_height = get_object_bbox_and_height(scene_config.object_passive.mesh_asset, scene_config.object_passive.up_axis)
    # cm --> m
    passive_pos = (scene_config.object_passive.default_position.x / 100, 
                   scene_config.object_passive.default_position.y / 100, 
                   object_passive_height / 2 - scene_config.robot.robot_to_table_height / 100)
    passive_quat = rotate_axis_quaternion(scene_config.object_passive.up_axis)
    object_passive = scene.add_entity(
        morph=gs.morphs.URDF(
            file  = scene_config.object_passive.asset,
            pos   = passive_pos,
            quat  = passive_quat, # we use w-x-y-z convention for quaternions,
            scale = 1.0,
            convexify=False,
            fixed=True,
        ),
        visualize_contact=False,
        vis_mode = "collision",
    )
    scene_dict = {
        "robot": robot,
        "background": background,
        "object_active": object_active,
        "object_passive": object_passive,
    }
    scene_asset_path_dict = {
        "robot": scene_config.robot.urdf_asset,
        "background": scene_config.background.asset,
        "object_active": scene_config.object_active.asset,
        "object_passive": scene_config.object_passive.asset,
    }
    default_poses = {
        "active_pos": active_pos,
        "active_quat": active_quat,
        "passive_pos": passive_pos,
        "passive_quat": passive_quat,
    }
    return scene, scene_config, scene_dict, scene_asset_path_dict, grasp_cam, default_poses

def render_gs(grasp_cam, gs_viewer_nonbackground, gs_viewer_background):
    topcam_intr = np.array(grasp_cam.intrinsics)
    topcam_extr = np.array(grasp_cam.extrinsics)
    R_topcam = topcam_extr[:3, :3]
    T_topcam = topcam_extr[:3, 3]
    topcam_width, topcam_height = grasp_cam.res
    intr_dict = {
        "image_height": topcam_height,
        "image_width": topcam_width,
    }
    intr_dict["fx"] = topcam_intr[0, 0]
    intr_dict["fy"] = topcam_intr[1, 1]
    intr_dict["cx"] = topcam_intr[0, 2]
    intr_dict["cy"] = topcam_intr[1, 2]
    
    rendered_nontable = render_and_save_specific_view(
        gs_viewer_nonbackground.viewer_renderer, 
        torch.device("cuda"),
        None,
        intr_dict,
        R = R_topcam,
        T = T_topcam,
        verbose=False,
        render_alpha=True,
        render_depth=False,
        return_outputs=True,
        save=False,
    )
    nontable_rgb_image = rendered_nontable["rgb_image"]
    nontable_alpha_image = rendered_nontable["alpha_image"]

    rendered_table = render_and_save_specific_view(
        gs_viewer_background.viewer_renderer, 
        torch.device("cuda"),
        None,
        intr_dict,
        R = R_topcam,
        T = T_topcam,
        verbose=False,
        render_alpha=True,
        render_depth=False,
        return_outputs=True,
        save=False,
    )
    rgb_image = rendered_table["rgb_image"]
    rgb_image = nontable_rgb_image * nontable_alpha_image[:, :, None] + (1 - nontable_alpha_image)[:, :, None] * rgb_image  # Replace RGB in transparent regions with non-transparent image RGB
    rgb_image = rgb_image.astype(np.uint8)
    return rgb_image

def remove_non_z_rotation(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Removes rotation about the world's X and Y axes from a 4x4 transform matrix,
    keeping only the rotation around the world's Z-axis and the original translation.

    Args:
        rotation_matrix (np.ndarray): The original 3x3 rotation matrix
                                       of the gripper in the world frame.

    Returns:
        np.ndarray: The new 3x3 transform matrix with the gripper's Z-axis aligned
                    with the world's Z-axis.
    """
    # 2. Extract the gripper's original Z-axis vector from the rotation matrix
    # The third column of the rotation matrix is the Z-axis vector.
    gripper_z_axis = rotation_matrix[:, 2]

    # 3. Create a new Z-axis vector that aligns with the world's Z-axis
    new_z_axis = np.array([0, 0, -1])

    # 4. Extract the gripper's original X-axis vector
    gripper_x_axis = rotation_matrix[:, 0]
    
    # 5. Project the original X-axis onto the world's XY plane
    # to maintain the original heading (yaw)
    projected_x_axis = np.array([gripper_x_axis[0], gripper_x_axis[1], 0])
    
    # 6. Normalize the projected X-axis vector
    # Avoid division by zero if the original X-axis was perfectly aligned with Z.
    if np.linalg.norm(projected_x_axis) > 1e-6:
        new_x_axis = projected_x_axis / np.linalg.norm(projected_x_axis)
    else:
        # If the original X-axis was nearly vertical, fall back to a default X-axis.
        new_x_axis = np.array([1, 0, 0])

    # 7. Create the new Y-axis vector by taking the cross product of the new Z and X axes
    new_y_axis = np.cross(new_z_axis, new_x_axis)

    # 8. Construct the new rotation matrix using the new axes
    new_rotation_matrix = np.vstack((new_x_axis, new_y_axis, new_z_axis)).T
    
    return new_rotation_matrix

def do_anygrasp(anygrasp_pipeline, topcam_rgb, topcam_depth, topcam_extr, topcam_intr, controller, collision_detection=True, debug=False, debug_anygrasp_showall=False):
    default_joint_angles = controller.default_joint_angles
    fx, fy = topcam_intr[0, 0], topcam_intr[1, 1]
    cx, cy = topcam_intr[0, 2], topcam_intr[1, 2]
    
    xmin, xmax = -0.59, 0.52
    ymin, ymax = -0.30, 0.30
    zmin, zmax = 0.5, 1.5
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]
    
    xmap, ymap = np.arange(topcam_depth.shape[1]), np.arange(topcam_depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = topcam_depth
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    
    mask = (points_z > 0) & (points_z < 5)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = topcam_rgb[mask].astype(np.float32) / 255
        
    gg, cloud = anygrasp_pipeline.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=collision_detection)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')
    
    gg = gg.nms().sort_by_score()
    gg_pick = gg[0]
    
    if debug_anygrasp_showall:
        gg_show = gg
    else:
        gg_show = gg[0:1]

    for gg_pick_each in gg_show:
        if debug:
            print(f"Original:")
            print(f"  Position: {gg_pick_each.translation}")
            print(f"  Rotation:\n{gg_pick_each.rotation_matrix}")
            print(f"  Width: {gg_pick_each.width:.4f} m")
            print(f"  Score: {gg_pick_each.score:.4f}")
            
    if debug:
        gripper = gg_show.to_open3d_geometry_list()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries(gripper + [pcd])
    
    grasp_pos = np.array(gg_pick.translation)
    grasp_rot = np.array(gg_pick.rotation_matrix)

    inv_extr = np.linalg.inv(topcam_extr)
    # cam --> world @ x_gripper --> cam @ z_gripper --> x_gripper
    grasp_rot_w = inv_extr[:3,:3] @ grasp_rot @ np.array([[ 0.0,0.0,1.0],
                                                          [ 0.0,1.0,0.0],
                                                          [-1.0,0.0,0.0],])
    grasp_rot_w = remove_non_z_rotation(grasp_rot_w)
    
    # to make joint7 shorter distance
    angle_rad = np.arctan2(grasp_rot_w[0, 1], -grasp_rot_w[1, 1])
    default_quat = controller.default_hand_quat
    default_rot_w = quaternion_to_rotation_matrix(default_quat)
    default_rad = np.arctan2(default_rot_w[0, 1], -default_rot_w[1, 1])
    possible_rads = np.array([angle_rad - 2*np.pi, angle_rad - np.pi, angle_rad, angle_rad + np.pi, angle_rad + 2 * np.pi])
    possible_diff_rads = np.abs(possible_rads - default_rad)
    best_rad = possible_rads[np.argmin(possible_diff_rads)]
    new_grasp_rot_w = grasp_rot_w.copy()
    new_grasp_rot_w[0, 0] = np.cos(best_rad)
    new_grasp_rot_w[1, 0] = np.sin(best_rad)
    new_grasp_rot_w[0, 1] = np.sin(best_rad)
    new_grasp_rot_w[1, 1] = -np.cos(best_rad)
    grasp_pos_w = inv_extr[:3,3].reshape(-1,3)+ (inv_extr[:3,:3]@grasp_pos.reshape(3,-1)).reshape(-1,3)
    grasp_pos_w = grasp_pos_w.reshape(-1)
    grasp_quat_w = rotation_matrix_to_quaternion(new_grasp_rot_w)
    
    return grasp_rot_w, grasp_pos_w, grasp_quat_w

def main(args):
    scene_config = EasyDict(yaml.safe_load(Path(args.cfg_path).open('r')))
    
    from datetime import datetime
    now = datetime.now()
    milliseconds = now.microsecond // 1000
    timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{milliseconds:03d}"
    task_name = scene_config.task_name
    process_output_dir = os.path.join(args.output_dir, task_name, timestamp)
    os.makedirs(process_output_dir, exist_ok=True)
                             
    cprint("*" * 40, "green")
    cprint("  Initializing Genesis", "green")
    cprint("*" * 40, "green")
    
    # Init Genesis
    gs.init(backend=gs.gpu, logging_level = 'warning')
    scene, scene_config, scene_dict, scene_asset_path_dict, grasp_cam, default_poses = design_scene(scene_config, show_viewer=args.show_viewer)
    scene.build()
    
    cprint("*" * 40, "green")
    cprint("  Initializing AnyGrasp", "green")
    cprint("*" * 40, "green")
    # Init AnyGrasp
    anygrasp_pipeline = init_anygrasp()
    
    # Assets
    robot = scene_dict["robot"]
    object_active = scene_dict["object_active"]
    object_passive = scene_dict["object_passive"]
    
    # Set phys params
    all_dof_ids = [robot.get_joint(name).dof_idx for name in JOINT_NAMES]
    robot.set_dofs_kp(kp = BEST_PARAMS["kp"], dofs_idx_local=all_dof_ids[:7])
    robot.set_dofs_kv(kv = BEST_PARAMS["kv"], dofs_idx_local=all_dof_ids[:7])
    robot.set_dofs_kp(kp = [50000, 50000], dofs_idx_local=all_dof_ids[7:9])
    robot.set_dofs_kv(kv = [10000, 10000], dofs_idx_local=all_dof_ids[7:9])
    robot.set_dofs_force_range([-100, -100], [100, 100], dofs_idx_local=all_dof_ids[7:9])
    
    object_active.get_link("object").set_mass(0.1)
    object_active.get_link("object").set_friction(0.2)

    cprint("*" * 40, "green")
    cprint("  Initializing Gaussians", "green")
    cprint("*" * 40, "green")
    
    # Init Gaussian Viewers
    gs_viewer_nonbackground = GenesisGaussianViewer({key: value for key, value in scene_asset_path_dict.items() if key != "background"}, 
                                                    {key: value for key, value in scene_dict.items() if key != "background"},
                                                    args_cli=None,)
    gs_viewer_background = GenesisGaussianViewer({key: value for key, value in scene_asset_path_dict.items() if key == "background"}, 
                                                 {key: value for key, value in scene_dict.items() if key == "background"},
                                                 args_cli=None,)
    
    cprint("*" * 40, "green")
    cprint("  Initializing Controller", "green")
    cprint("*" * 40, "green")
    
    # Init Controller
    controller = pick_and_place_controller(scene=scene, scene_config=scene_config, robot=robot, object_active=object_active, object_passive=object_passive, default_poses=default_poses, close_thres=scene_config.robot.close_thres)

    while True:
        cprint("*" * 40, "green")
        cprint(f"Trajectory count {controller.traj_cnt}", "green")
        cprint("*" * 40, "green")
        
        cprint(">>> Reset.", "yellow")
        
        # Reset Layout
        while True:
            controller.reset_scene()
            passive_pos = object_passive.get_pos().cpu().numpy()
            active_pos = object_active.get_pos().cpu().numpy()
            passive_aabb = object_passive.get_link("object").get_AABB()
            active_aabb = object_active.get_link("object").get_AABB()
            
            # object aabb must in the borders
            if active_aabb[0, 0] > scene_config.object_active.aabb_range.x[0] / 100 and \
               active_aabb[0, 1] > scene_config.object_active.aabb_range.y[0] / 100 and \
               active_aabb[1, 0] < scene_config.object_active.aabb_range.x[1] / 100 and \
               active_aabb[1, 1] < scene_config.object_active.aabb_range.y[1] / 100 :
                pass
            else:
                cprint("active out of range", "yellow")
                continue
            if passive_aabb[0, 0] > scene_config.object_passive.aabb_range.x[0] / 100 and \
               passive_aabb[0, 1] > scene_config.object_passive.aabb_range.y[0] / 100 and \
               passive_aabb[1, 0] < scene_config.object_passive.aabb_range.x[1] / 100 and \
               passive_aabb[1, 1] < scene_config.object_passive.aabb_range.y[1] / 100 :
                pass
            else:
                cprint("passive out of range", "yellow")
                continue
            active_xyaabb = active_aabb.cpu().numpy()[:2, :2]
            passive_xyaabb = passive_aabb.cpu().numpy()[:2, :2]
            x_overlap = active_xyaabb[0, 0] < passive_xyaabb[1, 0] and active_xyaabb[1, 0] > passive_xyaabb[0, 0]
            y_overlap = active_xyaabb[0, 1] < passive_xyaabb[1, 1] and active_xyaabb[1, 1] > passive_xyaabb[0, 1]
            if x_overlap and y_overlap:
                cprint("active-passive overlap box", "yellow")
                continue
            # the objects should not be too near
            if np.linalg.norm(passive_pos - active_pos) > scene_config.far_threshold:
                cprint("active-passive too near", "yellow")
                break
            
        default_ee_quat = controller.franka.get_link("hand").get_quat().cpu().numpy()
        controller.start_record()
        for i in range(50):
            robot.set_dofs_position([0, 0, 0, 0, 0, 0, 0, 0, 0], dofs_idx_local=all_dof_ids)
            scene.step()
            
        
        
        cprint(">>> Doing AnyGrasp.", "yellow")
        try:
            passive_pos = object_passive.get_pos().cpu().numpy()
            object_passive.set_pos(passive_pos + [0, 0, 10])
            
            # Update GS buffer
            scene.step()
            gs_viewer_nonbackground.update()
            gs_viewer_background.update()
            
            # AnyGrasp
            topcam_rgb = render_gs(grasp_cam, gs_viewer_nonbackground, gs_viewer_background)[:, :, :3]
            topcam_depth = grasp_cam.render(rgb=False, depth=True)[1]
            topcam_extr = np.array(grasp_cam.extrinsics)
            topcam_intr = np.array(grasp_cam.intrinsics)
            grasp_rot_w, grasp_pos_w, grasp_quat_w = do_anygrasp(anygrasp_pipeline, topcam_rgb, topcam_depth, topcam_extr, topcam_intr, controller, collision_detection=scene_config.collision_detection, debug=args.debug_anygrasp, debug_anygrasp_showall=args.debug_anygrasp_showall)
            object_passive.set_pos(passive_pos)
            object_passive_pos = object_passive.get_pos().cpu().numpy()
            
            # Update GS buffer
            scene.step()
            gs_viewer_nonbackground.update()
            gs_viewer_background.update()
            
        except:
            cprint(">>> AnyGrasp Failed. Reset.", "yellow")
            continue
        
        cprint(">>> Planning.", "yellow")
        # Plan
        ee_goals = []
        ee_goals.append(np.hstack(((grasp_pos_w + [0, 0, scene_config.object_active.motion_1_height / 100]), grasp_quat_w)))
        skill1_grasp_pos = grasp_pos_w + [0, 0, scene_config.object_active.skill_1_height / 100]
        skill1_grasp_pos[2] = np.clip(skill1_grasp_pos[2], 0.120 - scene_config.robot.robot_to_table_height / 100, np.inf)
        ee_goals.append(np.hstack(((skill1_grasp_pos), grasp_quat_w)))
        ee_goals.append(np.hstack(((grasp_pos_w + [0, 0, scene_config.object_active.motion_2_height / 100], grasp_quat_w))))
        ee_goals.append(np.hstack(((object_passive_pos + [0, 0, scene_config.object_active.motion_2_height / 100],  default_ee_quat))))
        ee_goals.append(np.hstack(((object_passive_pos + [0, 0, scene_config.object_active.skill_2_height / 100],  default_ee_quat))))
        ee_goals.append(np.hstack(((object_passive_pos + [0, 0, scene_config.object_active.motion_2_height / 100],  default_ee_quat))))
        ee_goals = np.array(ee_goals)

        cprint(">>> Executing.", "yellow")
        try:
            # Execute plan
            controller.reset_franka()
            controller.move_to_goal(ee_goals[0, 0:3], ee_goals[0, 3:7], gripper_open=True)
            controller.move_to_goal(ee_goals[1, 0:3], ee_goals[1, 3:7], gripper_open=True)
            for i in range(20):
                controller.step()
            controller.close_gripper()
            for i in range(20):
                controller.step()
            controller.move_to_goal(ee_goals[2, 0:3], ee_goals[2, 3:7], gripper_open=False)
            controller.move_to_goal(ee_goals[3, 0:3], ee_goals[3, 3:7], gripper_open=False)
            controller.move_to_goal(ee_goals[4, 0:3], ee_goals[4, 3:7], gripper_open=False)
            controller.open_gripper(wait_steps=50)
            controller.move_to_goal(ee_goals[5, 0:3], ee_goals[5, 3:7], gripper_open=True)
        except:
            cprint(">>> Executing Failed. Reset.", "yellow")
            continue
        
        # Judge and save
        passive_pos = object_passive.get_pos().cpu().numpy()
        active_pos = object_active.get_pos().cpu().numpy()
        if np.linalg.norm(passive_pos - active_pos) > scene_config.near_threshold :
            print("failed")
        else:
            print("success")
            print(f"saving {len(controller.record)} steps")
            controller.save_traj(process_output_dir)
        controller.end_record()
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/records")
    parser.add_argument("--cfg_path", type=str, default="simulation/configs/carrot_plate.yaml")
    parser.add_argument("--show_viewer", action='store_true')
    parser.add_argument("--debug_anygrasp", action='store_true')
    parser.add_argument("--debug_anygrasp_showall", action='store_true')
    args = parser.parse_args()
    main(args)