import os
import argparse
import torch
import numpy as np
import open3d as o3d
import sys

sys.path.insert(0, os.getcwd())
import genesis as gs
import yaml
from pathlib import Path
from easydict import EasyDict
from simulation.utils.constants import BEST_PARAMS, JOINT_NAMES
from simulation.utils.auto_collect.franka_genesis_controller import pick_and_place_controller
from simulation.data_collection.auto_collect_pick_and_place import design_scene as design_pnp_scene
from simulation.data_collection.render_pick_and_place import init_renderers, prepare_background, render_scene
from termcolor import cprint
from tqdm import tqdm
from pynput import keyboard

import queue
import threading
import time
import h5py

try:
    import rospy
    from sensor_msgs.msg import Image
    ROSPY_ENABLED = True
except:
    ROSPY_ENABLED = False
    raise NotImplementedError
        
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

def design_scene(scene_config, show_viewer=False):
    scene, scene_config, scene_dict, scene_asset_path_dict, grasp_cam, default_poses = design_pnp_scene(scene_config, show_viewer=show_viewer)
    return scene, scene_config, scene_dict, scene_asset_path_dict, grasp_cam, default_poses

image_queue = queue.Queue(maxsize=1) 

def consumer_thread():
    import cv2
    """
    Consumer thread: fetch images from the queue and display them.
    """
    try:
        cv2.namedWindow("Live Stream", cv2.WINDOW_AUTOSIZE)

        while True:
            # Try to get an image from the queue, with 1s timeout
            try:
                # The consumer thread will block here until there is data in the queue
                img = image_queue.get(timeout=1)
                if img is not None:
                    cv2.imshow("Live Stream", img)
            except queue.Empty:
                # If the queue is empty, do nothing and wait again
                pass
            
            # Window event handling
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Consumer error: {e}")
    finally:
        cv2.destroyAllWindows()

def read_h5_file(file_path):
    """
    Read a hierarchical HDF5 file and return a nested dictionary with NumPy arrays as leaf values.

    Args:
        file_path (str): Path to the HDF5 file

    Returns:
        dict: Nested dictionary representing all groups and datasets in the file
    """
    def traverse_group(group):
        data = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                # Read dataset as a NumPy array
                data[key] = item[()]
            elif isinstance(item, h5py.Group):
                # Recursively traverse subgroup
                data[key] = traverse_group(item)
        return data
    
    with h5py.File(file_path, 'r') as f:
        return traverse_group(f)
    
def main(args):
    scene_config = EasyDict(yaml.safe_load(Path(args.cfg_path).open('r')))
    render_types = ["rgb"]
    
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
    gs.init(backend=gs.gpu, logging_level = 'error')
    scene, scene_config, scene_dict, scene_asset_path_dict, grasp_cam, default_poses = design_scene(scene_config, show_viewer=False)
    scene.build()
    gs_viewer_nontable, gs_viewer_table, offscreen_renderer, w2c, camera_intr = init_renderers(scene_asset_path_dict, scene_dict, args, render_types=render_types)
    table_rgb_image, table_depth = prepare_background(offscreen_renderer, gs_viewer_table, camera_intr, w2c, args,  render_types=render_types)

    # Assets
    robot = scene_dict["robot"]
    object_active = scene_dict["object_active"]
    object_passive = scene_dict["object_passive"]
    
    # Set physical parameters
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
    controller = pick_and_place_controller(scene=scene, scene_config=scene_config, robot=robot, object_active=object_active, object_passive=object_passive, default_poses=default_poses, close_thres=scene_config.robot.close_thres, teleop=False, evaluation=False)

    cnt = 0
    
    def on_press(key):
        print(f"Key pressed: {key}")

    def on_release(key):
        nonlocal cnt
        if key.char == "1":
            cnt += 1
        if key.char == "2":
            cnt -= 1
        print(f"cnt={cnt}")
        
    def reset_layout(data=None):
        if data:
            controller.reset_scene(data)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    pbar = tqdm(total=None, bar_format='Replay: {rate_fmt}', unit='frames')

    # Create and start the consumer thread
    consumer = threading.Thread(target=consumer_thread, daemon=True)
    consumer.start()
    target = None
    
    while True:
        target_record = sorted(os.listdir(args.record_dir))[cnt % 50]
        h5_filenames = os.listdir(os.path.join(args.record_dir, target_record))
        h5_filenames = sorted([int(x.split(".")[0]) for x in h5_filenames if x.endswith(".h5")])
        h5_filenames = [str(h5_filename)+".h5" for h5_filename in h5_filenames]
        for h5_filename in h5_filenames:
            h5_path = os.path.join(args.record_dir, target_record, h5_filename)
            h5_stepidx = int(h5_filename.split(".")[0])
            if h5_stepidx == 0:
                first_h5 = read_h5_file(h5_path)
                reset_layout(first_h5)
            if h5_stepidx % 5 == 0:
                new_h5_path = os.path.join(args.record_dir, target_record, f"{h5_stepidx+5}.h5")
                new_h5 = read_h5_file(new_h5_path)
                target = [new_h5["joint_states"], new_h5["gripper_control"]]
            print(target)
            controller.franka.control_dofs_position(
                np.array(target[0][:7]),
                controller.all_dof_ids[:7],
            )
            if target[1][0]:
                controller.franka.control_dofs_position(
                    controller.close_state,
                    controller.all_dof_ids[7:9], 
                )
            else:
                controller.franka.control_dofs_position(
                    controller.open_state,
                    controller.all_dof_ids[7:9], 
                )
                
            rgb_image, depth_image = render_scene(scene_dict, None, gs_viewer_nontable, table_rgb_image, table_depth, args, w2c, camera_intr, offscreen_renderer, render_types=render_types)
            
            if not image_queue.empty():
                try:
                    image_queue.get_nowait()
                except queue.Empty:
                    pass
            controller.step()
            image_queue.put(rgb_image[:, :, ::-1])
            pbar.update()
        break
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/records")
    parser.add_argument("--cfg_path", type=str, default="simulation/configs/carrot_plate.yaml")
    parser.add_argument("--record_dir", type=str, default="")
    parser.add_argument("--depth_only", type=bool, default=False)
    parser.add_argument("--rgb_only", type=bool, default=True)
    args = parser.parse_args()
    main(args)