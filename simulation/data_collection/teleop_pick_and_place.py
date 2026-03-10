import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import sys
sys.path.insert(0, os.getcwd())

if os.environ.get("TRACKER_TYPE") == "keyboard":
    import genesis.ext.pyrender.viewer as _pr_viewer
    _pr_viewer.Viewer.on_key_press = lambda self, symbol, modifiers: None

import genesis as gs
import yaml
from pathlib import Path
from easydict import EasyDict
from simulation.utils.constants import BEST_PARAMS, JOINT_NAMES
from simulation.utils.auto_collect.franka_genesis_controller import pick_and_place_controller
from simulation.data_collection.auto_collect_pick_and_place import design_scene as design_pnp_scene
from termcolor import cprint
from tqdm import tqdm
from pynput import keyboard

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
    scene, scene_config, scene_dict, scene_asset_path_dict, grasp_cam, default_poses = design_pnp_scene(scene_config, show_viewer=show_viewer)
    left_cam = scene.add_camera(
        res    = (1280, 720),
        pos    = (0.62, -0.5, 0.5),
        lookat = (0.62, 0, 0),
        fov    = 60,
        GUI    = True,
    )
    front_cam = scene.add_camera(
        res    = (1280, 720),
        pos    = (1.5, -0.05, 0.3),
        lookat = (0.5, -0.05, 0.3),
        fov    = 30,
        GUI    = True,
    )

    annotation_cams = {
        "left_cam": left_cam,
        "front_cam": front_cam,
    }
    return scene, scene_config, scene_dict, scene_asset_path_dict, grasp_cam, default_poses, annotation_cams

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
    scene, scene_config, scene_dict, scene_asset_path_dict, grasp_cam, default_poses, annotation_cams = design_scene(scene_config, show_viewer=True)
    scene.build()
    
    scene.viewer.on_key_press = lambda symbol, modifiers: None
    
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
    cprint("  Initializing Controller", "green")
    cprint("*" * 40, "green")
    
    # Init Controller
    controller = pick_and_place_controller(scene=scene, scene_config=scene_config, robot=robot, object_active=object_active, object_passive=object_passive, default_poses=default_poses, close_thres=scene_config.robot.close_thres, teleop=True)
    
    def reset_layout():
        # Reset Layout
        while True:
            controller.reset_scene()
            passive_pos = object_passive.get_pos().cpu().numpy()
            active_pos = object_active.get_pos().cpu().numpy()
            passive_aabb = object_passive.get_link("object").get_AABB()
            active_aabb = object_active.get_link("object").get_AABB()
            
            # object aabb must in the borders
            if active_aabb[0, 0] > scene_config.object_active.pos_range.x[0] / 100 and \
               active_aabb[0, 1] > scene_config.object_active.pos_range.y[0] / 100 and \
               active_aabb[1, 0] < scene_config.object_active.pos_range.x[1] / 100 and \
               active_aabb[1, 1] < scene_config.object_active.pos_range.y[1] / 100 :
                pass
            else:
                cprint("active out of range", "yellow")
                continue
            if passive_aabb[0, 0] > scene_config.object_passive.pos_range.x[0] / 100 and \
               passive_aabb[0, 1] > scene_config.object_passive.pos_range.y[0] / 100 and \
               passive_aabb[1, 0] < scene_config.object_passive.pos_range.x[1] / 100 and \
               passive_aabb[1, 1] < scene_config.object_passive.pos_range.y[1] / 100 :
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
            
    def on_press(key):
        pass
        # print(f"Key pressed: {key}")

    def on_release(key):
        nonlocal process_output_dir
        nonlocal controller
        nonlocal reset_layout
        if key.char == "1":
            print('recollect !!!')
            controller.clean_traj()
            reset_layout()
        elif key.char == "2":
            print('start recording !!!')
            controller.start_record()
        elif key.char == "3":
            print("end recording !!!")
            controller.end_record()
        elif key.char == "4":
            print("save record ...")
            controller.save_traj(process_output_dir)
            reset_layout()
    
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    pbar = tqdm(total=None, bar_format='Teleop: {rate_fmt}', unit='frames')

    while True:
        if not listener.is_alive():
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
        for cam in annotation_cams.values():
            cam.render()
        controller.step()
        pbar.update()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/records")
    parser.add_argument("--cfg_path", type=str, default="simulation/configs/carrot_plate.yaml")
    args = parser.parse_args()
    main(args)