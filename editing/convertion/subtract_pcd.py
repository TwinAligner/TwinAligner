import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree
import open3d as o3d  # for visualization only

def subtract_point_clouds_vectorized_2(target_ply_path, subtract_ply_path, threshold=0.01):
    """
    Subtract one point cloud from the target point cloud while preserving all attributes, using SciPy KDTree.
    """
    print(f"Reading files with plydata: {target_ply_path} and {subtract_ply_path}...")
    try:
        target_ply = PlyData.read(target_ply_path)
        subtract_ply = PlyData.read(subtract_ply_path)
    except Exception as e:
        print(f"Error reading files: {e}")
        return None

    target_points_data = target_ply['vertex'].data
    subtract_points_data = subtract_ply['vertex'].data

    target_points_np = np.vstack([target_points_data['x'], target_points_data['y'], target_points_data['z']]).T
    subtract_points_np = np.vstack([subtract_points_data['x'], subtract_points_data['y'], subtract_points_data['z']]).T

    # 1. Build SciPy KDTree
    print("Building KDTree...")
    kdtree = KDTree(subtract_points_np)

    # 2. Batch query with SciPy KDTree; 'query' returns nearest-neighbor distance and index per point
    print("Querying points to remove...")
    distances, _ = kdtree.query(target_points_np)

    # 3. Find indices below threshold using NumPy where
    indices_to_remove = np.where(distances < threshold)[0]

    # 4. Remove those rows from PlyData
    print(f"Found {len(indices_to_remove)} points to remove.")
    all_indices = np.arange(len(target_points_np))
    indices_to_keep = np.setdiff1d(all_indices, indices_to_remove)
    
    vertex_element = target_ply.elements[0]
    filtered_data = vertex_element.data[indices_to_keep]
    
    new_vertex_element = PlyElement.describe(filtered_data, vertex_element.name)
    result_ply = PlyData([new_vertex_element], text=False)
    
    print(f"Final point cloud has {len(result_ply['vertex'].data)} points.")
    
    return result_ply

# --- Main ---
if __name__ == "__main__":
    # Replace with your file paths
    point_cloud_path = "/home/hwfan/workspace/twinmanip/outputs/background-asset/point_cloud/iteration_30000/point_cloud.ply"
    franka_path = "/home/hwfan/workspace/twinmanip/outputs/background-asset/splitted_point_cloud/franka_whole_gs.ply"
    
    # Run point cloud subtraction
    result_ply_data = subtract_point_clouds_vectorized_2(point_cloud_path, franka_path, threshold=0.01)
    
    if result_ply_data:
        output_path = "/home/hwfan/workspace/twinmanip/outputs/background-asset/splitted_point_cloud/table_new.ply"
        result_ply_data.write(output_path)