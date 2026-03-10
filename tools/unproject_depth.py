#!/usr/bin/env python3
"""Unproject a depth map into a point cloud and visualize it.

Depth map convention: PNG as uint16, units are millimeters (consistent with render_pick_and_place output).
Camera intrinsics: by default, use assets/realsense/cam_K.txt (3x3 matrix, fx/fy/cx/cy).
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d

try:
    import imageio.v3 as iio
except ImportError:
    iio = None


def load_depth(path: str, depth_scale: float = 1000.0) -> np.ndarray:
    """Load depth map and convert to meters.

    Args:
        path: Path to the depth map (PNG, uint16, millimeter units)
        depth_scale: Depth scaling, 1000 means stored in mm

    Returns:
        (H, W) float32, units in meters, invalid/zero depth remains 0
    """
    if iio is not None:
        raw = iio.imread(path)
    else:
        from PIL import Image
        raw = np.array(Image.open(path))
    if raw is None or raw.size == 0:
        raise FileNotFoundError(f"Unable to read depth map: {path}")
    raw = np.squeeze(raw)
    if raw.ndim != 2:
        raise ValueError(f"Depth map should be 2D, got shape {raw.shape}")
    if raw.dtype != np.uint16:
        raw = np.asarray(raw, dtype=np.uint16)
    depth_m = raw.astype(np.float32) / depth_scale
    return depth_m


def load_intrinsics(path: str) -> tuple[np.ndarray, int, int]:
    """Load intrinsic parameters from a 3x3 K matrix file.

    File format: Three numbers per line, three lines in total (OpenCV style K matrix).
    If the image size does not match the depth map, please ensure the input depth image size is consistent where used.
    """
    K = np.loadtxt(path)
    assert K.shape == (3, 3), f"Expected 3x3 matrix, got {K.shape}"
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return (fx, fy, cx, cy), K


def unproject_depth_to_point_cloud(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1.0,
    depth_trunc: float | None = None,
    invalid_depth: float = 0.0,
) -> np.ndarray:
    """Unproject the depth map into point cloud in camera coordinates.

    Args:
        depth: (H, W) depth map, in meters
        fx, fy, cx, cy: Camera intrinsics
        depth_scale: Depth multiplier (usually 1.0)
        depth_trunc: Depths beyond this value are considered invalid and not added to the point cloud; None means no truncation
        invalid_depth: Threshold for invalid depth (pixels <= this value are skipped)

    Returns:
        (N, 3) point cloud, camera coordinate system, units consistent with "depth"
    """
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    z = depth.astype(np.float32) * depth_scale
    valid = z > invalid_depth
    if depth_trunc is not None:
        valid &= z < depth_trunc
    z_valid = z[valid]
    u_valid = uu[valid]
    v_valid = vv[valid]
    x = (u_valid - cx) / fx * z_valid
    y = (v_valid - cy) / fy * z_valid
    points = np.stack([x, y, z_valid], axis=1)
    return points


def depth_to_color_colormap(depth: np.ndarray, valid_mask: np.ndarray, depth_trunc: float) -> np.ndarray:
    """Generate color from depth values (used for coloring when no RGB). Uses a simple linear colormap."""
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid_mask] = np.clip(depth[valid_mask] / depth_trunc, 0, 1)
    # Linear from blue to red: R=norm, G peaks in the middle, B=1-norm
    r = (np.clip(norm * 255, 0, 255)).astype(np.uint8)
    g = (np.clip(2 * np.minimum(norm, 1 - norm) * 255, 0, 255)).astype(np.uint8)
    b = (np.clip((1 - norm) * 255, 0, 255)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Unproject a depth map into a point cloud and visualize it")
    parser.add_argument(
        "depth_path",
        nargs="?",
        default="/home/hwfan/twinaligner/datasets/records/carrot_plate/20260309_165115_643/00000/0_depth.png",
        help="Path to depth map (PNG, uint16, units: mm)",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=None,
        help="Path to camera intrinsics 3x3 matrix file (default: assets/realsense/cam_K.txt)",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor to convert depth values to meters (default 1000 for mm)",
    )
    parser.add_argument(
        "--depth-trunc",
        type=float,
        default=3.0,
        help="Points with depth beyond this value (meters) will not be added to the point cloud",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional: Path to save point cloud as PLY",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not pop up visualization window (useful when only saving)",
    )
    args = parser.parse_args()

    # Set project root directory: script is in tools/
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.intrinsics is None:
        args.intrinsics = os.path.join(repo_root, "assets", "realsense", "cam_K.txt")
    if not os.path.isfile(args.intrinsics):
        print(f"Error: Intrinsics file not found: {args.intrinsics}", file=sys.stderr)
        sys.exit(1)

    depth = load_depth(args.depth_path, depth_scale=args.depth_scale)
    (fx, fy, cx, cy), _ = load_intrinsics(args.intrinsics)
    h, w = depth.shape

    points = unproject_depth_to_point_cloud(
        depth,
        fx=fx, fy=fy, cx=cx, cy=cy,
        depth_trunc=args.depth_trunc,
        invalid_depth=0.0,
    )
    if points.size == 0:
        print("No valid depth points. Please check the depth map and depth_scale/depth_trunc.", file=sys.stderr)
        sys.exit(1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color with depth (when no RGB)
    valid = (depth > 0) & (depth < args.depth_trunc)
    colors = depth_to_color_colormap(depth, valid, args.depth_trunc)
    pcd.colors = o3d.utility.Vector3dVector(colors[valid] / 255.0)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        o3d.io.write_point_cloud(args.save, pcd)
        print(f"Point cloud saved: {args.save}")

    if not args.no_show:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        o3d.visualization.draw_geometries(
            [pcd, coord],
            window_name="Depth unprojection",
            width=1280,
            height=720,
        )


if __name__ == "__main__":
    main()
