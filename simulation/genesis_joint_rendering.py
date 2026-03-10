import argparse
import os
import json
import genesis as gs
import torch
import sys
import numpy as np
from urdfpy import URDF
import viser
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Literal, List
from fast_gaussian_model_manager import FastGaussianModelManager, construct_from_ply, matrix_to_quaternion
sys.path.insert(0, os.getcwd())
from simulation.utils.gs_viewer_utils import ClientThread, ViewerRenderer, GSPlatRenderer
from simulation.utils.gs_viewer_utils import *
import imageio
from simulation.utils.constants import FR3_DEFAULT_CFG
from gs_viewer import render_and_save_specific_view, GenesisGaussianViewer
from simulation.utils.gs_viewer_utils import load_camera_params

def design_scene(background_timestamp="asset", intr="examples/franka-track/cam_K.txt"):
    genesis_scene_path_dict = {
        "robot": "assets/fr3/fr3.urdf",
        "scene": f"outputs/background-{background_timestamp}/point_cloud/iteration_30000/scene_gs_abs.ply",
    }
    scene = gs.Scene(
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (0, -3.5, 2.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 30,
            max_FPS       = 60,
        ),
        sim_options = gs.options.SimOptions(
            dt = 0.01,
        ),
        show_viewer = True,
    )
    
    background = scene.add_entity(
        gs.morphs.Plane(pos   = (0, 0, -0.008)),
        material=gs.materials.Rigid(friction=0.1),
    )
    
    franka = scene.add_entity(
        gs.morphs.URDF(
            file  = genesis_scene_path_dict["robot"],
            pos   = (0, 0, 0),
            quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
            scale = 1.0,
            merge_fixed_links=False,
            fixed=True,
            convexify=False,
            
        ),
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )
    
    genesis_scene_dict = {
        "robot": franka,
        "scene": background,
    }
    
    gs_viewer = GenesisGaussianViewer(genesis_scene_path_dict, 
                                      genesis_scene_dict)
    camera_extr, camera_intr = load_camera_params()
    return scene, gs_viewer, genesis_scene_dict, camera_intr, camera_extr

def run_simulator(scene, gs_viewer, genesis_scene_dict, camera_intr, camera_extr):
    gs_viewer.start()
    gs_viewer.update()
    robot = genesis_scene_dict["robot"]

    count = 0
    jnt_names = [
        'fr3_joint1',
        'fr3_joint2',
        'fr3_joint3',
        'fr3_joint4',
        'fr3_joint5',
        'fr3_joint6',
        'fr3_joint7',
        'fr3_finger_joint1',
        'fr3_finger_joint2',
    ]
    all_dof_ids = [robot.get_joint(name).dof_idx for name in jnt_names]
    all_joint_ids = [robot.get_joint(name).idx_local for name in jnt_names]
    pbar = tqdm(total=None, desc='frames')
    default_joint_pos = [FR3_DEFAULT_CFG[name] for name in jnt_names]
    with gs_viewer.server.add_gui_folder("Joints"):
        gs_viewer.joint_modifiers = dict()
        for i, name in enumerate(jnt_names[:7]):
            joint_limits = robot.get_dofs_limit([i])
            gs_viewer.joint_modifiers[name] = gs_viewer.server.add_gui_slider(
                name,
                min=float(joint_limits[0].item()),
                max=float(joint_limits[1].item()),
                step=np.pi / 180,
                initial_value=float(default_joint_pos[i]),
            )
            gs_viewer.joint_modifiers[name].on_update(gs_viewer._handle_option_updated)
        gs_viewer.joint_modifiers["fr3_finger_joint"] = gs_viewer.server.add_gui_slider(
            "fr3_finger_joint",
            min=0,
            max=0.04,
            step=0.01,
            initial_value=0.04
        )
        gs_viewer.joint_modifiers["fr3_finger_joint"].on_update(gs_viewer._handle_option_updated)
        
    # Simulation loop
    while True:
        joint_angles = torch.tensor([gs_viewer.joint_modifiers[joint_name].value if "finger" not in joint_name \
                                     else gs_viewer.joint_modifiers["fr3_finger_joint"].value \
                                     for joint_name in jnt_names])
        dt = 100
        robot.control_dofs_position(joint_angles.to("cuda"))

        count += 1
        # Update buffers
        scene.step()
        gs_viewer.update()
        pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Genesis joint rendering.")
    parser.add_argument("-b", "--background_timestamp", type=str, default="asset", help="Background timestamp.")
    args_cli = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level = 'warning')
    scene, gs_viewer, genesis_scene_dict, camera_intr, camera_extr = design_scene(args_cli.background_timestamp)
    scene.build()
    run_simulator(scene, gs_viewer, genesis_scene_dict, camera_intr, camera_extr)

if __name__ == "__main__":
    main()
