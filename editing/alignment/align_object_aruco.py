import cv2
import numpy as np
from scipy.spatial.distance import euclidean
import argparse
import os
import open3d as o3d
import copy

def detect_and_measure_aruco_3d(image_path, pointmap, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    """
    Reads a JPG image, detects ArUco markers, extracts their 3D coordinates 
    using a provided pointmap, calculates the average length of the 3D edges, 
    and draws the results on the image.

    :param image_path: File path of the input JPG image.
    :param pointmap: HxWx3 NumPy array (Point Cloud Map) where pointmap[y, x] = [X, Y, Z] 
                     in 3D space (e.g., camera coordinate system).
    :param aruco_dict_type: The type of ArUco dictionary to use.
    :return: 
        - detection_results: List of dictionaries containing detected 2D marker ID and corner coordinates.
        - image_with_markers: The image with detected markers drawn on it.
        - average_3d_edge_distance: The average length of all detected marker edges in 3D.
    """
    
    # --- 1. Detect ArUco Markers in 2D ---

    print(f"Attempting to read image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image. Check path and existence.")
        return [], None, 0.0

    # Initialize ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = aruco_detector.detectMarkers(gray)

    detection_2d_results = []
    marker_3d_data = []
    image_with_markers = image.copy()
    
    if ids is not None and len(ids) > 0:
        print(f"Successfully detected {len(ids)} ArUco markers in 2D.")
        
        # Draw detected markers on the image
        cv2.aruco.drawDetectedMarkers(image_with_markers, corners, ids)
        
        # --- 2. Process Detection Results and Extract 3D Coordinates ---
        
        for marker_corners, marker_id in zip(corners, ids):
            # marker_corners shape: (1, 4, 2) -> reshape to (4, 2)
            corners_2d = marker_corners.reshape((4, 2)).astype(int)
            
            # The order of corners is: [0] TL, [1] TR, [2] BR, [3] BL
            corners_map = {
                "top_left": (corners_2d[0][0], corners_2d[0][1]),
                "top_right": (corners_2d[1][0], corners_2d[1][1]),
                "bottom_right": (corners_2d[2][0], corners_2d[2][1]),
                "bottom_left": (corners_2d[3][0], corners_2d[3][1]),
            }
            
            # Store 2D detection result
            detection_2d_results.append({
                "id": int(marker_id[0]),
                "corners_2d": corners_map
            })
            
            # Extract 3D coordinates from pointmap
            corners_3d = {}
            valid_3d_data = True
            
            for name, (x, y) in corners_map.items():
                # pointmap indexing is [y, x]
                if 0 <= y < pointmap.shape[0] and 0 <= x < pointmap.shape[1]:
                    # Extract 3D coordinates [X, Y, Z]
                    coords_3d = pointmap[y, x]
                    # Check for invalid Z values (depth sensor "no-data" markers, often 0 or infinity)
                    if np.isclose(coords_3d[2], 0.0) or np.isnan(coords_3d).any():
                         valid_3d_data = False
                         print(f"Warning: Invalid 3D data (near zero Z or NaN) found for marker {marker_id[0]} at {name}.")
                         break
                    corners_3d[name] = coords_3d
                else:
                    valid_3d_data = False
                    print(f"Warning: 2D corner for marker {marker_id[0]} is outside pointmap boundaries.")
                    break
            
            if valid_3d_data:
                 marker_3d_data.append({
                    "id": int(marker_id[0]),
                    "corners_3d": corners_3d
                })

        # --- 3. Calculate Average 3D Edge Distance ---
        
        all_edge_distances = []
        
        for marker in marker_3d_data:
            corners = marker['corners_3d']
            
            # Extract 3D coordinates (TL, TR, BR, BL)
            P_TL = corners['top_left']
            P_TR = corners['top_right']
            P_BR = corners['bottom_right']
            P_BL = corners['bottom_left']
            
            # Calculate 3D Euclidean distance for the four edges
            dist_top = euclidean(P_TL, P_TR)
            dist_right = euclidean(P_TR, P_BR)
            dist_bottom = euclidean(P_BR, P_BL)
            dist_left = euclidean(P_BL, P_TL)
            
            marker_distances = [dist_top, dist_right, dist_bottom, dist_left]
            all_edge_distances.extend(marker_distances)

        if all_edge_distances:
            average_distance = np.mean(all_edge_distances)
            print(f"Calculated average 3D edge distance across all valid markers: {average_distance:.4f}")
            return detection_2d_results, image_with_markers, average_distance, all_edge_distances
        else:
            print("No valid 3D data available to calculate edge distances.")
            return detection_2d_results, image_with_markers, 0.0, []
    
    else:
        print("No ArUco markers detected.")
        return [], image, 0.0, []

def main(args):
    image_file_path = os.path.join(args.data, "images_ori", "frame_00000.jpg")
    point_map_path = os.path.join(args.data, "ori_world_pointmap", "frame_00000.npz")
    pointmap = np.load(point_map_path)["world_pointmap"]
    _, aruco_image, dist, all_edge_distances = detect_and_measure_aruco_3d(image_file_path, pointmap)
    colmap2real_factor = args.aruco_size / dist
    print("average aruco dist in colmap:", dist)
    print("real aruco dist:", args.aruco_size)
    print("colmap2real factor:", colmap2real_factor)
    if args.debug:
        print("all edge distances:", all_edge_distances)
    aruco_image_path = os.path.join(args.data, "ori_world_pointmap", "frame_00000_aruco.jpg")
    print("saving aruco detection image ...")
    cv2.imwrite(aruco_image_path, aruco_image)

    meshfile = os.path.join(args.output_path, "mesh_w_vertex_color.ply")
    
    mesh = o3d.io.read_triangle_mesh(meshfile)
    ori_mesh = copy.deepcopy(mesh)
    ori_center = mesh.get_center()
    T_transmat_all = np.eye(4)
    T_transmat_all[:3, 3] = -ori_center
    mesh.translate(-ori_center)
    
    S_transmat_all = np.eye(4)
    S_transmat_all[0, 0] = colmap2real_factor
    S_transmat_all[1, 1] = colmap2real_factor
    S_transmat_all[2, 2] = colmap2real_factor
    mesh.transform(S_transmat_all)
    
    np.savez_compressed(os.path.join(args.output_path, 'rel2abs.npz'), 
                        rel2abs=S_transmat_all,
                        scale=np.array([colmap2real_factor, colmap2real_factor, colmap2real_factor]),
                        rot=np.array([0, 0, 0]),
                        trans=-ori_center,)
    o3d.io.write_triangle_mesh(meshfile.replace(".ply", "_abs.ply"), mesh)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("align object scale with aruco markers")
    parser.add_argument('-i', '--data', type=str, default="", help='data path')
    parser.add_argument('-o', '--output_path', type=str, help='output path')
    parser.add_argument('--aruco_size', type=float, default=0.035)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    main(args)