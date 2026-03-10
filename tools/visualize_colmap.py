"""COLMAP visualizer

Visualize COLMAP sparse reconstruction outputs. To get demo data, see `./assets/download_colmap_garden.sh`.
"""

import random
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as onp
import tyro
import viser
import viser.transforms as tf
from tqdm.auto import tqdm
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)


def main(
    colmap_path: Path,
    images_path: Path,
    downsample_factor: int = 8,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    server = viser.ViserServer()
    server.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load the colmap info.
    cameras = read_cameras_binary(colmap_path / "cameras.bin")
    images = read_images_binary(colmap_path / "images.bin")
    points3d = read_points3d_binary(colmap_path / "points3D.bin")
    gui_reset_up = server.add_gui_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ onp.array(
            [0.0, -1.0, 0.0]
        )

    gui_point_size = server.add_gui_number("Point size", initial_value=0.01)

    img_ids = [im.id for im in images.values()]
    
    visualize_dict = dict()            
    for img_id in tqdm(img_ids):
        img = images[img_id]
        cam = cameras[img.camera_id]

        # Skip images that don't exist.
        image_filename = images_path / img.name
        if not image_filename.exists():
            continue
        
        image = iio.imread(image_filename)
        image = image[::downsample_factor, ::downsample_factor]
        
        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(img.qvec), img.tvec
        ).inverse()
        
        visualize_dict[img_id] = {
            "cam": cam,
            "image": image,
            "T_world_camera": T_world_camera,
        }
        
    def visualize_colmap() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""
        # Set the point cloud.
        points = onp.array([points3d[p_id].xyz for p_id in points3d])
        colors = onp.array([points3d[p_id].rgb for p_id in points3d])
        server.add_point_cloud(
            name="/colmap/pcd",
            points=points,
            colors=colors,
            point_size=gui_point_size.value,
        )

        # Interpret the images and cameras.
        def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in visualize_dict:
            cam = visualize_dict[img_id]["cam"]
            image = visualize_dict[img_id]["image"]
            T_world_camera = visualize_dict[img_id]["T_world_camera"]
            frame = server.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )
            H, W = cam.height, cam.width
            fy = cam.params[1]
            frustum = server.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=2 * onp.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.15,
                image=image,
            )
            attach_callback(frustum, frame)

    need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    while True:
        if need_update:
            need_update = False

            server.reset_scene()
            visualize_colmap()

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)