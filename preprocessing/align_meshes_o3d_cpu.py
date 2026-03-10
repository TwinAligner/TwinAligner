import open3d as o3d
import numpy as np
import time
import random
from scipy.spatial.transform import Rotation as R
from tqdm import trange
import argparse
import copy
import plotly.graph_objects as go

def draw_registration_result(source, target, transformation, filename="registration_result.html"):
    """
    Visualizes the registration result using Plotly and saves it as an HTML file.
    
    Args:
        source: Open3D PointCloud (the moving cloud)
        target: Open3D PointCloud (the fixed cloud)
        transformation: 4x4 numpy array (transformation matrix)
        filename: Path to save the HTML file
    """
    # 1. Transform the source point cloud
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    
    # 2. Extract points as numpy arrays
    source_pts = np.asarray(source_temp.points)
    target_pts = np.asarray(target.points)
    
    # 3. Create Plotly Figure
    fig = go.Figure()

    # Add Source Trace (Transformed) - Using the orange-ish color from your original code
    fig.add_trace(go.Scatter3d(
        x=source_pts[:, 0], y=source_pts[:, 1], z=source_pts[:, 2],
        mode='markers',
        marker=dict(size=2, color='rgb(255, 180, 0)', opacity=0.8),
        name='Source (Transformed)'
    ))

    # Add Target Trace - Using the blue-ish color from your original code
    fig.add_trace(go.Scatter3d(
        x=target_pts[:, 0], y=target_pts[:, 1], z=target_pts[:, 2],
        mode='markers',
        marker=dict(size=2, color='rgb(0, 166, 237)', opacity=0.8),
        name='Target'
    ))

    # 4. Update layout for better viewing
    fig.update_layout(
        title="Point Cloud Registration Result",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data' # Ensures 1:1:1 scale
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 5. Save to HTML
    fig.write_html(filename)
    print(f"Registration visualization saved to {filename}")
    
def align_meshes(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    num_init_trials: int = 200,
    tolerance: float = 0.01,
    show_fitting: bool = False,
    pbar_desc: str = "Trials",
    annotation: bool = False,
    icp_iteration: int = 10000,
    filename="registration_result.html",
):
    bbox1 = source.get_oriented_bounding_box()
    bbox2 = target.get_oriented_bounding_box()
    init_scale = bbox2.extent.max() / bbox1.extent.max()
    source_copied = copy.deepcopy(source)
    target_copied = copy.deepcopy(target)
    source_copied = source_copied.scale(init_scale, np.array([0,0,0]))
    
    # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
    max_correspondence_distance = 1.0
        
    while True:
        # random seed
        current_time = time.time() * 1000 # time in milliseconds
        random.seed(int(current_time) % 114514)
        np.random.seed(int(current_time) % 114514)
    
        transformations = []
        rmses = []
        for trial_idx in trange(num_init_trials):
            # init euler angles
            init_x_angle = np.random.rand() * 360 - 180
            init_y_angle = np.random.rand() * 360 - 180
            init_z_angle = np.random.rand() * 360 - 180
            angles = np.array([init_x_angle, init_y_angle, init_z_angle])
            # Initial alignment or source to target transform.
            init_source_to_target = np.eye(4)
            init_source_to_target[:3, :3] = R.from_euler('xyz', angles, degrees=True).as_matrix()
            # print(f"angles: {angles}")

            # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
            registration_icp = o3d.pipelines.registration.registration_icp(
            source_copied, target_copied, max_correspondence_distance, init_source_to_target, 
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
            o3d.pipelines.registration.ICPConvergenceCriteria())
            transformations.append(registration_icp.transformation)
            rmses.append(registration_icp.inlier_rmse)
        
        trans_matrix = transformations[np.argmin(rmses)]
        print(f"rmse is {np.min(rmses):.3f} with rigid transformation")
        registration_icp = o3d.pipelines.registration.registration_icp(
            source_copied, target_copied, max_correspondence_distance, trans_matrix, 
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iteration))
        print(f"rmse is {registration_icp.inlier_rmse:.3f} after scaling")
        if registration_icp.inlier_rmse < tolerance:
            break
        else:
            print("rmse {:.3f} > tolerance {}, trying again...".format(registration_icp.inlier_rmse, tolerance))
    trans_matrix = registration_icp.transformation
    if show_fitting:
        draw_registration_result(source_copied, target_copied, trans_matrix, filename=filename)
    trans_matrix_dump = trans_matrix.copy()
    scale_mat = np.array([
        [init_scale, 0, 0, 0],
        [0, init_scale, 0, 0],
        [0, 0, init_scale, 0],
        [0, 0,          0, 1]
    ])
    trans_matrix_dump = trans_matrix_dump @ scale_mat
    return trans_matrix_dump, registration_icp.inlier_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source-path', type=str, default='/home/hwfan/twinmanip/outputs/robot-20240930_150308_876/mesh_w_vertex_color.ply', help='source mesh path')
    parser.add_argument('-t', '--target-path', type=str, default='/home/hwfan/twinmanip/outputs/robot-20240825_181120_727/mesh_w_vertex_color.ply', help='target mesh path')
    parser.add_argument('--tolerance', type=float, default=0.2, help='tolerance threshold')
    parser.add_argument('-p', '--show_fitting', action='store_true')
    parser.add_argument("-pc", "--point-cloud", action='store_true')
    parser.add_argument("-anno", "--annotation", action="store_true")
    parser.add_argument("-icp", "--icp-iteration", type=int, default=10000)
    args = parser.parse_args()
    if args.point_cloud:
        source = o3d.io.read_point_cloud(args.source_path)
    else:
        source = o3d.io.read_triangle_mesh(args.source_path).sample_points_uniformly(number_of_points=50000)
    target = o3d.io.read_triangle_mesh(args.target_path).sample_points_uniformly(number_of_points=50000)
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    trans_matrix_dump, dist = align_meshes(source, target, tolerance=args.tolerance, show_fitting=args.show_fitting, 
                                           annotation=args.annotation, icp_iteration=args.icp_iteration)