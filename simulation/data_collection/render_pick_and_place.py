import argparse
import os
import json

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import genesis as gs
import torch
import sys
import numpy as np
from tqdm import tqdm
import imageio
from scipy.spatial.transform import Rotation
import h5py
import cv2
import yaml
from pathlib import Path
from easydict import EasyDict
import open3d as o3d
import time

sys.path.insert(0, os.getcwd())
from simulation.utils.gs_viewer_utils import load_camera_params
from simulation.data_collection.auto_collect_pick_and_place import design_scene as design_pnp_scene
from simulation.gs_viewer import render_and_save_specific_view, GenesisGaussianViewer

def render_scene(scene_dict, data, gs_viewer_nontable, table_rgb_image, table_depth, args_cli, w2c, camera_intr, znear=0.1, render_types=["rgb", "depth"]):
    if data is not None:
        robot = scene_dict["robot"]
        object_active = scene_dict["object_active"]
        object_passive = scene_dict["object_passive"]
        robot.set_dofs_position(data["joint_states"])
        passive_pos = data["object_states"]["passive"][:3]
        passive_quat = data["object_states"]["passive"][3:7]
        active_pos = data["object_states"]["active"][:3]
        active_quat = data["object_states"]["active"][3:7]
        object_passive.set_pos(passive_pos)
        object_passive.set_quat(passive_quat)
        object_active.set_pos(active_pos)
        object_active.set_quat(active_quat)
    
    # time33 = time.time()
    gs_viewer_nontable.update()
    # time44 = time.time()
    # print("update time", time44-time33)
    if "rgb" in render_types:
        # time1 = time.time()
        rendered_nontable = render_and_save_specific_view(
            gs_viewer_nontable.viewer_renderer, 
            torch.device("cuda"),
            None,
            camera_intr,
            R = w2c[:3, :3],
            T = w2c[:3, 3],
            verbose=False,
            render_alpha=True,
            render_depth=True,
            return_outputs=True,
            save=False,
            return_torch = True,
        )
        # time2 = time.time()
        # print("real rendertime", time2-time1)
        nontable_rgb_image = rendered_nontable["rgb_image"][:, :, :3]
        nontable_alpha_image = rendered_nontable["alpha_image"]
        # time11 = time.time()
        rgb_image = nontable_rgb_image * nontable_alpha_image[:, :, None] + (1 - nontable_alpha_image)[:, :, None] * table_rgb_image # Replace the RGB values of the transparent image with those from the opaque image
        rgb_image = rgb_image.cpu().numpy()
        rgb_image = np.ascontiguousarray(rgb_image.astype(np.uint8))
        # time222 = time.time()
        # print("composite time", time222 - time11)
    else:
        rgb_image = None
    if "depth" in render_types:
        nontable_depth = gs_viewer_nontable.render_depth_for_frame(w2c, camera_intr, torch.device("cuda"), znear=znear)
        nontable_depth[nontable_depth <= znear] = 5.0
        all_depth = table_depth.copy()
        # all_depth = np.minimum(nontable_depth, table_depth) # z-buffer occlusion
        all_depth[nontable_depth < 5] = nontable_depth[nontable_depth < 5] # direct copy. For depth sim this is better...
        depth_image = np.clip(all_depth * 1000, 0, 65535).astype('uint16')
    else:
        depth_image = None
    return rgb_image, depth_image
            
def read_h5_file(file_path):
    """
    Read hierarchical HDF5 file and return a nested dictionary, with values as NumPy arrays.

    Args:
        file_path (str): Path to the HDF5 file

    Returns:
        dict: Nested dictionary containing all groups and datasets in the file
    """
    def traverse_group(group):
        data = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                # Read dataset and store as NumPy array
                data[key] = item[()]
            elif isinstance(item, h5py.Group):
                # Recursively traverse subgroup
                data[key] = traverse_group(item)
        return data
    
    with h5py.File(file_path, 'r') as f:
        return traverse_group(f)

def load_extrinsics(file_path):
    T = np.loadtxt(file_path)
    assert T.shape == (4, 4), "Input must be a 4x4 matrix"

    R_inv = T[:3, :3].T  # R^T
    t_inv = -R_inv @ T[:3, 3]  # -R^T * t

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    
    position = T_inv[:3, 3].tolist()
    rotation_matrix = T_inv[:3, :3]
    
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat().tolist()  # (x, y, z, w)
    quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    
    return position, quaternion, T[:3, :3], T[:3, 3]

def visualize_depth(depth_map, max_depth=None, view=True):
    """
    Visualize depth map

    :param depth_map: Input depth map (single channel)
    :param max_depth: Optional. Specifies the maximum depth value for normalization
    :return: Colored depth map for visualization
    """

    # Use the maximum value from the depth map if max_depth is not specified
    if max_depth is None:
        max_depth = np.max(depth_map)
    
    # Normalize depth values to 0-255 and convert to uint8
    depth_vis = np.clip(depth_map, 0, max_depth)  # Limit maximum depth
    depth_vis = (depth_vis / max_depth * 255).astype(np.uint8)
    
    # Apply color mapping (JET colormap is used here)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    if view:
        cv2.imshow("Depth Visualization", depth_colormap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return depth_colormap

def visualize_rgb(rgb):
    cv2.imshow("RGB Visualization", rgb[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def visualize_rgbd(rgb, depth, camera_intr, depth_scale=1000.0):
    """
    Args:
        rgb: HxWx3 RGB image (np.uint8 format, 0-255)
        depth: HxW depth map (np.float32, usually in millimeters)
        depth_scale: Depth scale factor (default 1000 means the depth is in mm)
    """
    # 1. Create Open3D RGB and depth image objects
    rgb_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)

    # 2. Build RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=rgb_o3d,
        depth=depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=3.0,  # Depth truncation value (adjust as needed)
        convert_rgb_to_intensity=False
    )

    # 3. Generate point cloud and visualize
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=camera_intr["image_width"],
        height=camera_intr["image_height"],
        fx=camera_intr["fx"],  # Default focal length (adjust according to camera parameters)
        fy=camera_intr["fy"],
        cx=camera_intr["cx"],
        cy=camera_intr["cy"]
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )
    
    # 4. Visualization
    o3d.io.write_point_cloud("debug.ply", pcd)

def init_renderers(scene_asset_path_dict, scene_dict, args_cli, render_types=["rgb", "depth"]):
    gs_viewer_nontable = GenesisGaussianViewer({key: value for key, value in scene_asset_path_dict.items() if key != "background"}, 
                                {key: value for key, value in scene_dict.items() if key != "background"},
                                args_cli=args_cli,
                                render_depth="depth" in render_types)
    gs_viewer_table = GenesisGaussianViewer({key: value for key, value in scene_asset_path_dict.items() if key == "background"}, 
                                    {key: value for key, value in scene_dict.items() if key == "background"},
                                    args_cli=args_cli,
                                    render_depth="depth" in render_types)
    w2c, camera_intr = load_camera_params()
    return gs_viewer_nontable, gs_viewer_table, w2c, camera_intr

def prepare_background(gs_viewer_table, camera_intr, w2c, args_cli, render_types=["rgb", "depth"]):
    if "depth" in render_types:
        print("Preparing background depth")
        table_depth = gs_viewer_table.render_depth_for_frame(w2c, camera_intr, torch.device("cuda"))
        print('done.')
    else:
        table_depth = None
    if "rgb" in render_types:
        rendered_table = render_and_save_specific_view(
            gs_viewer_table.viewer_renderer, 
            torch.device("cuda"),
            None,
            camera_intr,
            R = w2c[:3, :3],
            T = w2c[:3, 3],
            verbose=False,
            render_alpha=False,
            render_depth=False,
            return_outputs=True,
            save=False,
            return_torch = True,
        )
        table_rgb_image = rendered_table["rgb_image"][:, :, :3]
    else:
        table_rgb_image = None
    return table_rgb_image, table_depth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument('--cfg_path', type=str, default=None)
    parser.add_argument("--mod", type=int, default=10)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--demo_min_idx", type=int, default=0)
    parser.add_argument('--demo_num', type=int, default=200)
    parser.add_argument("--nots", action="store_true", default=False)
    parser.add_argument('--render_types', type=str, default="rgb,depth")
    args_cli = parser.parse_args()
    args_cli.render_types = args_cli.render_types.split(',')
    scene_config = EasyDict(yaml.safe_load(Path(args_cli.cfg_path).open('r')))
    
    gs.init(backend=gs.gpu, logging_level = 'error')
    scene, scene_config, scene_dict, scene_asset_path_dict, grasp_cam, default_poses = design_pnp_scene(scene_config, show_viewer=False)
    gs_viewer_nontable, gs_viewer_table, w2c, camera_intr = init_renderers(scene_asset_path_dict, scene_dict, args_cli, render_types=args_cli.render_types)
    scene.build()
    
    table_rgb_image, table_depth = prepare_background(gs_viewer_table, camera_intr, w2c, args_cli, render_types=args_cli.render_types)
    
    def render_demos(demo_dirs):
        for demo_dir in tqdm(demo_dirs, desc="demo"):
            h5_file_idxs = [int(file.replace(".h5", "")) for file in os.listdir(demo_dir) if file.endswith(".h5") and int(file.replace(".h5", "")) % args_cli.mod == 0]
            h5_file_idxs = sorted(h5_file_idxs)
            for h5_file_idx in tqdm(h5_file_idxs, desc="frame"):
                h5_path = os.path.join(demo_dir, f"{h5_file_idx}.h5")
                data = read_h5_file(h5_path)
                rgb_image, depth_image = render_scene(scene_dict, data, gs_viewer_nontable, table_rgb_image, table_depth, args_cli, w2c, camera_intr)
                if args_cli.debug:
                    if "rgb" in args_cli.render_types and "depth" in args_cli.render_types:
                        visualize_rgbd(rgb_image, depth_image, camera_intr)
                        print("saved rgbd to debug.ply in current directory")
                    if "rgb" in args_cli.render_types:
                        visualize_rgb(rgb_image)
                    if "depth" in args_cli.render_types:                        
                        visualize_depth(depth_image)
                else:
                    if "rgb" in args_cli.render_types:                        
                        imageio.imwrite(h5_path.replace(".h5", ".jpg"), rgb_image, quality=90)
                    if "depth" in args_cli.render_types:
                        cv2.imwrite(h5_path.replace(".h5", "_depth.png"), depth_image)
                        depth_vis = visualize_depth(depth_image, view=False)
                        cv2.imwrite(h5_path.replace(".h5", "_depth_render.jpg"), depth_vis)
    def collect_demo_dirs():
        if not args_cli.nots:
            demo_dirs = []
            # Iterate each timestamp directory
            timestamps = os.listdir(args_cli.record_dir)
            for timestamp in timestamps:
                timestamp_dir = os.path.join(args_cli.record_dir, timestamp)
                # Append all sub demo directories under the timestamp directory
                sub_demo_dirs = [os.path.join(timestamp_dir, x) for x in sorted(os.listdir(timestamp_dir))]
                demo_dirs.extend(sub_demo_dirs)
        else:
            # Directly get the demo directories under record_dir
            demo_dirs = [os.path.join(args_cli.record_dir, x) for x in sorted(os.listdir(args_cli.record_dir))]
        return demo_dirs
    demo_dirs = collect_demo_dirs()
    demo_dirs = demo_dirs[args_cli.demo_min_idx : args_cli.demo_min_idx + args_cli.demo_num]
    print(f"Render {len(demo_dirs)} demos.")
    render_demos(demo_dirs)