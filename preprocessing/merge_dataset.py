import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.align_meshes_o3d_cpu import align_meshes
import torch
import open3d as o3d
import numpy as np
import shutil
import yaml
from pathlib import Path
import json
from tqdm import tqdm
import pymeshlab
import copy

def main(args):
    all_json = dict()
    all_json["camera_model"] = "OPENCV"
    all_json["frames"] = []
    all_frame_idx = 0
    for scan_idx in range(args.num_scans):
        real_scan_idx = scan_idx + 1
        this_scan_recons_dir = args.scan_recons_dir.format(scan_idx=real_scan_idx)
        config_path = os.path.join(this_scan_recons_dir, "config.yml")
        config = yaml.load(Path(config_path).read_text(), Loader=yaml.Loader)
        new_c2w_path = os.path.join(this_scan_recons_dir, "new_c2w.npz")
        new_c2ws = np.load(new_c2w_path)["c2w"]
        data_path = str(config.pipeline.datamanager.dataparser.data)
        transforms = json.load(open(os.path.join(data_path, "transforms.json")))
        if args.edited:
            mesh_path = os.path.join(this_scan_recons_dir, "parts", "part_0.ply")
        else:
            mesh_path = os.path.join(this_scan_recons_dir, "mesh_w_vertex_color.ply")
        if args.simplify_meshes:
            new_mesh_path = mesh_path.replace(".ply", "_simplified_merge_dataset.ply")
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(mesh_path)
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=2000)
            ms.save_current_mesh(new_mesh_path, save_face_color=False)
        else:
            new_mesh_path = mesh_path
        if real_scan_idx == 1:
            # base dataset
            new_mesh = o3d.io.read_point_cloud(new_mesh_path)
            trans_matrix = np.eye(4)
        else:
            # additional dataset
            old_mesh = copy.deepcopy(new_mesh)
            new_mesh = o3d.io.read_point_cloud(new_mesh_path)
            results = align_meshes(new_mesh, old_mesh, tolerance=args.tolerance, show_fitting=args.show_fitting, pbar_desc=f"Fitting scan {real_scan_idx} to scan {real_scan_idx-1}",
                                   annotation=args.annotation, icp_iteration=args.icp_iteration)
            if results is None:
                print(f"Fitting scan {real_scan_idx} failed.")
                continue
            else:
                mid_trans_matrix, dist = results
                trans_matrix = trans_matrix @ mid_trans_matrix
                print(f"Fitting scan {real_scan_idx} success, final dist is {dist}.")
        for frame_idx, frame_info in enumerate(tqdm(transforms["frames"], desc=f"Merging dataset {real_scan_idx} ...")):
            new_frame_info = dict()
            new_frame_info["file_path"] = "images/frame_{:05d}.jpg".format(all_frame_idx)
            new_frame_info["transform_matrix"] = (trans_matrix @ np.concatenate([new_c2ws[frame_idx], np.array([[0, 0, 0, 1]])], axis=0)).tolist()
            new_frame_info["fl_x"] = transforms["fl_x"]
            new_frame_info["fl_y"] = transforms["fl_y"]
            new_frame_info["cx"] = transforms["cx"]
            new_frame_info["cy"] = transforms["cy"]
            new_frame_info["w"] = transforms["w"]
            new_frame_info["h"] = transforms["h"]
            new_frame_info["k1"] = transforms["k1"]
            new_frame_info["k2"] = transforms["k2"]
            new_frame_info["p1"] = transforms["p1"]
            new_frame_info["p2"] = transforms["p2"]
            all_json["frames"].append(new_frame_info)
            old_image_path = os.path.join(data_path, transforms["frames"][frame_idx]["file_path"])
            new_image_path = os.path.join(args.output_dir, new_frame_info["file_path"])
            dst_folder = os.path.dirname(new_image_path)
            os.makedirs(dst_folder, exist_ok=True)
            shutil.copyfile(old_image_path, new_image_path)
            
            old_mask_path = old_image_path.replace("images", "masks")
            if os.path.exists(old_mask_path):
                new_mask_path = new_image_path.replace("images", "masks")
                dst_folder_mask = os.path.dirname(new_mask_path)
                os.makedirs(dst_folder_mask, exist_ok=True)
                shutil.copyfile(old_mask_path, new_mask_path)
            old_mono_depth_path = os.path.splitext(old_image_path.replace("images", "mono_depths"))[0] + ".png"
            if os.path.exists(old_mono_depth_path):
                new_mono_depth_path = os.path.splitext(new_image_path.replace("images", "mono_depths"))[0] + ".png"
                dst_folder_mono_depth = os.path.dirname(new_mono_depth_path)
                os.makedirs(dst_folder_mono_depth, exist_ok=True)
                shutil.copyfile(old_mono_depth_path, new_mono_depth_path)
            old_mono_depth_render_path = old_image_path.replace("images", "mono_depths_render")
            if os.path.exists(old_mono_depth_render_path):
                new_mono_depth_render_path = new_image_path.replace("images", "mono_depths_render")
                dst_folder_mono_depth_render = os.path.dirname(new_mono_depth_render_path)
                os.makedirs(dst_folder_mono_depth_render, exist_ok=True)
                shutil.copyfile(old_mono_depth_render_path, new_mono_depth_render_path)
            old_image_ori_path = old_image_path.replace("images", "images_ori")
            if os.path.exists(old_image_ori_path):
                new_image_ori_path = new_image_path.replace("images", "images_ori")
                dst_folder_ori = os.path.dirname(new_image_ori_path)
                os.makedirs(dst_folder_ori, exist_ok=True)
                shutil.copyfile(old_image_ori_path, new_image_ori_path)
            all_frame_idx += 1
            
    print("Dumping transforms.json ...")
    with open(os.path.join(args.output_dir, "transforms.json"), "w", encoding="utf-8") as f:
        json.dump(all_json, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-scans', type=int, default=3, help='number of scans to merge')
    parser.add_argument('--scan-recons-dir', type=str, default='outputs/component-20240717-{scan_idx}', help='path of the reconstructed scans')
    parser.add_argument('--output-dir', type=str, default='datasets/nerfstudio-data/component-20240717', help='path of the merged dataset')
    parser.add_argument('--edited', action='store_true')
    parser.add_argument("--show_fitting", action="store_true")
    parser.add_argument('--simplify_meshes', action='store_true')
    parser.add_argument('--tolerance', type=float, default=0.1)
    parser.add_argument("--annotation", action="store_true")
    parser.add_argument('--no_alignment', action='store_true')
    parser.add_argument("-icp", "--icp-iteration", type=int, default=10000)
    args = parser.parse_args()
    
    main(args)