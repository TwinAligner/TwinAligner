from viser.extras.colmap import read_images_binary
from pathlib import Path
from tqdm import tqdm
import viser.transforms as tf
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d
import os
import sys
import argparse
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from sklearn import neighbors
import matplotlib
import torch

sys.path.insert(0, os.getcwd())
from preprocessing.align_meshes_o3d_cpu import align_meshes
from editing.forward_kinematics.urdf_fk import urdf_fk
from nerfstudio.scripts.exporter import ExportGaussianSplat
from simulation.fast_gaussian_model_manager import construct_from_ply, matrix_to_quaternion, save_to_ply

def cluster_filter_pointcloud(pointcloud, eps=0.1, min_samples=35, visualize=False, filename="clusters.html"):
    """
    Filter noise points using DBSCAN.
    If visualize=True, saves a Plotly HTML showing all detected clusters.
    """
    if pointcloud is None or len(pointcloud) == 0:
        return pointcloud
    
    # 1. Prepare points
    points = pointcloud[:, :3]
    
    # 2. Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(points)

    # 3. Optional Visualization
    if visualize:
        fig = go.Figure()
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            mask = (cluster_labels == label)
            cluster_pts = points[mask]
            
            # Formatting: Noise is -1, usually rendered in grey
            name = f"Cluster {label}" if label != -1 else "Noise"
            color = None if label != -1 else "lightgrey"
            opacity = 0.8 if label != -1 else 0.2
            size = 2 if label != -1 else 1

            fig.add_trace(go.Scatter3d(
                x=cluster_pts[:, 0], y=cluster_pts[:, 1], z=cluster_pts[:, 2],
                mode='markers',
                marker=dict(size=size, opacity=opacity, color=color),
                name=name
            ))

        fig.update_layout(title="DBSCAN Clustering Results", scene=dict(aspectmode='data'))
        fig.write_html(filename)
        print(f"Cluster visualization saved to {filename}")

    # 4. Return results (usually keeping the largest cluster, label 0)
    keep_mask = cluster_labels == 0
    filtered_pointcloud = pointcloud[keep_mask]
    
    print(f"DBSCAN: Original {len(pointcloud)}, Filtered {len(filtered_pointcloud)}")
    return filtered_pointcloud, keep_mask

def calculate_hull_and_crop(np_points, c2w_points, visualize=False, filename="hull_crop.html"):
    """
    Crops np_points using the convex hull of c2w_points.
    If visualize=True, saves a Plotly HTML with the mesh and points.
    """
    # 1. Compute Convex Hull using Open3D
    hull_pcd = o3d.geometry.PointCloud()
    hull_pcd.points = o3d.utility.Vector3dVector(c2w_points)
    hull_mesh, _ = hull_pcd.compute_convex_hull()
    
    # 2. Raycasting for Cropping logic
    hull_t = o3d.t.geometry.TriangleMesh.from_legacy(hull_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(hull_t)
    
    query_tensor = o3d.core.Tensor(np_points, dtype=o3d.core.Dtype.Float32)
    occupancy = scene.compute_occupancy(query_tensor).numpy().astype(bool)
    
    inside_points = np_points[occupancy]

    # 3. Optional Visualization
    if visualize:
        outside_points = np_points[~occupancy]
        fig = go.Figure()

        # Points Inside
        fig.add_trace(go.Scatter3d(
            x=inside_points[:, 0], y=inside_points[:, 1], z=inside_points[:, 2],
            mode='markers', marker=dict(size=2, color='blue'), name='Inside Points'
        ))

        # Points Outside (low opacity)
        fig.add_trace(go.Scatter3d(
            x=outside_points[:, 0], y=outside_points[:, 1], z=outside_points[:, 2],
            mode='markers', marker=dict(size=1, color='red', opacity=0.1), name='Outside Points'
        ))

        # The Hull Mesh surface
        verts = np.asarray(hull_mesh.vertices)
        faces = np.asarray(hull_mesh.triangles)
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='green', opacity=0.15, name='Convex Hull Boundary'
        ))

        fig.update_layout(title="Convex Hull Cropping Result", scene=dict(aspectmode='data'))
        fig.write_html(filename)
        print(f"Crop visualization saved to {filename}")

    # 4. Wrap result in Open3D object
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(inside_points)
    
    print(f"Cropping: Original {len(np_points)}, Kept {len(inside_points)}")
    return hull_mesh, cropped_pcd, occupancy

def align_robot_gs_knn(fk_dict, robot_plydata, s2t, target_urdf, visualize=False, knn_neighbors=10, filename="robot_gs_knn.html"):
    knn = neighbors.KNeighborsClassifier(n_neighbors=knn_neighbors)
    link_names = []
    X_train = []
    y_train = []
    cnt = 0
    for link_name, part in fk_dict.items():
        each_part_pcd = part.sample_points_uniformly(number_of_points=5000)
        each_part_pcd = np.asarray(each_part_pcd.points)
        X_train.append(each_part_pcd)
        y_train.append(np.ones((each_part_pcd.shape[0],), dtype=int) * cnt)
        link_names.append(link_name)
        cnt += 1
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    knn.fit(X_train, y_train)
    X_pred = np.stack((
        np.asarray(robot_plydata["x"]),
        np.asarray(robot_plydata["y"]),
        np.asarray(robot_plydata["z"]),
    ), axis=1)
    X_pred = (s2t[:3, :3] @ X_pred.T + s2t[:3, 3:4]).T
    y_pred = knn.predict(X_pred)
    if visualize:
        cmap = matplotlib.colormaps['plasma']
        normalized_labels = y_pred / np.max(y_pred)
        colors = cmap(normalized_labels)[:, :3]

        x, y, z = X_pred[:, 0], X_pred[:, 1], X_pred[:, 2]
        marker_colors = [
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            for r, g, b in colors
        ]

        fig = go.Figure(
            data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=2,
                    color=marker_colors,
                ),
                text=[str(lbl) for lbl in y_train],
            )]
        )
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            title="Point Cloud Visualization (Plotly)"
        )
        fig.write_html(filename)
        print(f"knn visualization saved to {filename}")
        
    gaussian_part_dir = os.path.join(os.path.dirname(os.path.abspath(target_urdf)), "parts")
    os.makedirs(gaussian_part_dir, exist_ok=True)
    
    scale = np.ones((3,)) * np.linalg.svd(s2t[:3, :3])[1][0]
    rotation = s2t[:3, :3] / scale
    translation = s2t[:3, 3]
    
    for link_idx, link_name in enumerate(link_names):
        valid_part_gs_idxs = y_pred == link_idx
        new_ply_data = dict()
        for propname in robot_plydata.keys():
            new_ply_data[propname] = np.asarray(robot_plydata[propname])[valid_part_gs_idxs]
        part_3dgs_path = os.path.join(gaussian_part_dir, f'{link_name}_3dgs.ply')
        ExportGaussianSplat.write_ply(part_3dgs_path, len(np.where(valid_part_gs_idxs)[0]), new_ply_data)
        part_3dgs_path_abs = part_3dgs_path.replace(".ply", "_abs.ply")
        gs_model = construct_from_ply(part_3dgs_path, torch.device("cpu"))
        gs_model.scale(torch.from_numpy(scale).to(gs_model._scaling.dtype).to("cpu"))
        gs_model.rotate(matrix_to_quaternion(torch.from_numpy(rotation).to(gs_model._rotation.dtype).to("cpu")))
        gs_model.translate(torch.from_numpy(translation).to(gs_model._xyz.dtype).to("cpu"))
        save_to_ply(gs_model, part_3dgs_path_abs)
        
def align_background(gs_points_path, colmap_path, expand_factor=0.2, robot_expand_factor=0.1, save_txt=False, target_urdf="assets/fr3/fr3.urdf"):
    images = read_images_binary(Path(colmap_path) / "sparse" / "images.bin")
    gs_points_dir = os.path.dirname(gs_points_path)
    os.makedirs(os.path.join(gs_points_dir, "vis"), exist_ok=True)
    c2ws = [tf.SE3.from_rotation_and_translation(
        tf.SO3(img.qvec), img.tvec
    ).inverse() for img in images.values()]
    c2w_xyzs = np.array([c2w.translation() for c2w in c2ws])
    fk_dict, all_fk = urdf_fk(target_urdf, save=False, clean_vertex_normals=True)
    print(">>> Loading scene gs ...")
    plydata = PlyData.read(gs_points_path)
    gs_xyz = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"]),
    ), axis=1)
    print(">>> Cropping robot gs with hull...")
    _, source_point_cloud, hull_keep = calculate_hull_and_crop(gs_xyz, c2w_xyzs, visualize=True, filename=os.path.join(gs_points_dir, "vis", "1_crop_hull.html"))
    source_point_cloud_xyz = np.asarray(source_point_cloud.points)
    print(">>> Cleaning robot gs ...")
    source_point_cloud_xyz_filtered, clean_keep = cluster_filter_pointcloud(source_point_cloud_xyz, visualize=True, filename=os.path.join(gs_points_dir, "vis", "2_clean_gs.html"))
    print(">>> Aligning robot gs with URDF ...")
    source_point_cloud.points = o3d.utility.Vector3dVector(source_point_cloud_xyz_filtered)
    # Sampling more points helps in partial matching
    target_pcd = all_fk.sample_points_uniformly(number_of_points=50000)
    # s2t: colmap axes --> urdf axes
    s2t, rmse = align_meshes(source_point_cloud, target_pcd, show_fitting=True, filename=os.path.join(gs_points_dir, "vis", "3_align_robot_gs.html"))
    if save_txt:
        s2t_txt_path = os.path.join(gs_points_dir, "background_gs2cad.txt")
        np.savetxt(s2t_txt_path, s2t, fmt="%.8f", delimiter=" ")
        print(f"Saved transformation matrix to {s2t_txt_path}")
    print(">>> Cropping scene + robot gs ... ")
    c2w_xyzs_transformed = (s2t[:3, :3] @ c2w_xyzs.T + s2t[:3, 3:4]).T
    xy_min = c2w_xyzs_transformed[:, :2].min(axis=0)
    xy_max = c2w_xyzs_transformed[:, :2].max(axis=0)
    xy_range = xy_max - xy_min
    xy_min_exp = xy_min - expand_factor * xy_range
    xy_max_exp = xy_max + expand_factor * xy_range
    gs_xyz_transformed = (s2t[:3, :3] @ gs_xyz.T + s2t[:3, 3:4]).T
    scene_robot_keep = (
        (gs_xyz_transformed[:, 0] >= xy_min_exp[0]) & (gs_xyz_transformed[:, 0] <= xy_max_exp[0]) &
        (gs_xyz_transformed[:, 1] >= xy_min_exp[1]) & (gs_xyz_transformed[:, 1] <= xy_max_exp[1])
    )    
    print(">>> Cropping robot gs ...")
    fk_xyz = np.asarray(target_pcd.points)
    fk_bbox = np.array([fk_xyz.min(axis=0), fk_xyz.max(axis=0)])
    gs_points = gs_xyz_transformed[scene_robot_keep]
    fk_bbox_range = fk_bbox[1] - fk_bbox[0]
    fk_bbox_min_exp = fk_bbox[0] - robot_expand_factor * fk_bbox_range
    fk_bbox_max_exp = fk_bbox[1] + robot_expand_factor * fk_bbox_range
    robot_keep = (
        (gs_points[:, 0] >= fk_bbox_min_exp[0]) & (gs_points[:, 0] <= fk_bbox_max_exp[0]) &
        (gs_points[:, 1] >= fk_bbox_min_exp[1]) & (gs_points[:, 1] <= fk_bbox_max_exp[1]) &
        (gs_points[:, 2] >= fk_bbox[0][2]) & (gs_points[:, 2] <= fk_bbox[1][2])
    )
    robot_plydata = dict()
    for prop in plydata.elements[0].properties:
        robot_plydata[prop.name] = np.asarray(plydata.elements[0][prop.name])[scene_robot_keep][robot_keep]
    ExportGaussianSplat.write_ply(os.path.join(gs_points_dir, "robot_gs.ply"), len(robot_plydata["x"]), robot_plydata)
    print(">>> Cropping scene gs ... ")
    scene_keep = ~robot_keep
    scene_plydata = dict()
    for prop in plydata.elements[0].properties:
        scene_plydata[prop.name] = np.asarray(plydata.elements[0][prop.name])[scene_robot_keep][scene_keep]
    ExportGaussianSplat.write_ply(os.path.join(gs_points_dir, "scene_gs.ply"), len(scene_plydata["x"]), scene_plydata)
    print(">>> Aligning robot parts gs with URDF ...") 
    align_robot_gs_knn(fk_dict, robot_plydata, s2t, target_urdf, visualize=True, filename=os.path.join(gs_points_dir, "vis", "4_align_robot_part_gs.html"))
    print(">>> Done!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_points_path", type=str, default="outputs/background-asset/point_cloud/iteration_30000/point_cloud.ply")
    parser.add_argument("--colmap_path", type=str, default="datasets/nerfstudio-data/background-asset")
    parser.add_argument("--expand_factor", type=float, default=0.2)
    parser.add_argument("--robot_expand_factor", type=float, default=0.05)
    parser.add_argument('--save_txt', action="store_true")
    args = parser.parse_args()
    align_background(args.gs_points_path, args.colmap_path, args.expand_factor, args.robot_expand_factor, args.save_txt)