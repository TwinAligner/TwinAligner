# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
import os
from tqdm import tqdm
import genesis as gs
import torch
import sys
import numpy as np
sys.path.insert(0, os.getcwd())
from simulation.gs_viewer import render_and_save_specific_view, GenesisGaussianViewer
from simulation.utils.constants import FR3_DEFAULT_CFG, REALSENSE_IMAGE_HEIGHT, REALSENSE_IMAGE_WIDTH

def design_scene(background_timestamp="asset", intr="examples/franka-track/cam_K.txt"):
    genesis_scene_path_dict = {
        "robot": "assets/fr3/fr3.urdf",
        "background": f"outputs/background-{background_timestamp}/point_cloud/iteration_30000/scene_gs_abs.ply",
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
        show_viewer = False,
    )
    
    background = scene.add_entity(
        gs.morphs.Plane(pos   = (0, 0, 0)),
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
        "background": background,
    }
    
    gs_viewer = GenesisGaussianViewer(genesis_scene_path_dict, 
                                      genesis_scene_dict)
    camera_intr = np.loadtxt(intr)
    camera_intr_dict = dict()
    camera_intr_dict["fx"] = camera_intr[0, 0]
    camera_intr_dict["fy"] = camera_intr[1, 1]
    camera_intr_dict["cx"] = camera_intr[0, 2]
    camera_intr_dict["cy"] = camera_intr[1, 2]
    camera_intr_dict["image_width"] = REALSENSE_IMAGE_WIDTH
    camera_intr_dict["image_height"] = REALSENSE_IMAGE_HEIGHT
    return scene, gs_viewer, genesis_scene_dict, camera_intr_dict

def update_simulator(extr_candidates, render_dir, scene, gs_viewer, genesis_scene_dict, camera_intr):
    scene.step()
    gs_viewer.update()
    robot = genesis_scene_dict["robot"]

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
    default_joint_pos = [FR3_DEFAULT_CFG[name] for name in jnt_names]
    robot.set_dofs_position(torch.tensor(default_joint_pos).to("cuda"))
    # Update buffers
    scene.step()
    gs_viewer.update()
    
    # Simulation loop
    for i in tqdm(range(len(extr_candidates)), desc="rendering views", unit="view"):
        R = extr_candidates[i, :3, :3]
        T = extr_candidates[i, :3, 3]
        render_and_save_specific_view(
            gs_viewer.viewer_renderer,
            torch.device("cuda:0"),
            os.path.join(render_dir,f"1{int(i):04d}.jpg"), # 0 -- collected frame, 1 -- rendered frame
            camera_intr,
            R, T,
            )
        
def initialize_simulator(background_timestamp="asset", intr="examples/franka-track/cam_K.txt"):
    gs.init(backend=gs.gpu, logging_level = 'warning')
    scene, gs_viewer, genesis_scene_dict, camera_intr_dict = design_scene(background_timestamp, intr)
    scene.build()
    return scene, gs_viewer, genesis_scene_dict, camera_intr_dict