import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Any
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import random
# Add current working directory to sys.path to allow local imports
sys.path.insert(0, os.getcwd())
from sam3.model_builder import build_sam3_video_model
from simulation.view_alignment.genesis_joint_rendering_for_opt import initialize_simulator, update_simulator

def load_mask_from_file(file_path):
    """Load a mask image from file and convert to a normalized torch tensor."""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read mask file: {file_path}")
    _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask / 255.0
    mask = torch.tensor(binary_mask, dtype=torch.float32)
    return mask

def loss_function_batch(transform_matrices, args, iteration, pso_log_dir, 
                        sam3_model, genesis_joint_renderer):
    """
    Calculate the loss for a batch of rotation matrices by:
    1. Rendering views using the simulator.
    2. Waiting for rendering to complete.
    3. Running SAM3 to segment the rendered images.
    4. Comparing resulting masks with the target mask.
    """
    iter_dir = os.path.join(pso_log_dir, str(iteration))
    rgb_dir = os.path.join(args.output_dir, "rgb")
    # Ensure the directory is clean before use
    shutil.rmtree(iter_dir, ignore_errors=True)
    os.makedirs(iter_dir, exist_ok=True)

    matrix_dir = os.path.join(iter_dir, "matrix")
    os.makedirs(matrix_dir, exist_ok=True)

    render_dir = os.path.join(iter_dir, "render")
    os.makedirs(render_dir, exist_ok=True)
    
    # Save transformation matrices for this iteration
    for i in range(transform_matrices.shape[0]):
        np.savetxt(
            os.path.join(matrix_dir, f"{int(i):04d}.txt"),
            transform_matrices[i],
            fmt='%.10f'
        )
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        update_simulator(transform_matrices, render_dir, *genesis_joint_renderer)
    
    masks_dir = os.path.join(iter_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    robot_mask = load_mask_from_file(os.path.join(args.output_dir, "masks_robot", "00010.png"))
    table_mask = load_mask_from_file(os.path.join(args.output_dir, "masks_table", "00010.png"))
    gt_mask_all = (robot_mask.bool() | table_mask.bool()).to(torch.float32)
    gt_image_path = os.path.join(args.output_dir, "rgb", "00010.png")
    gt_image = cv2.imread(gt_image_path)
    cv2.imwrite(os.path.join(render_dir, "00000.jpg"), gt_image[:, :, :3])
    cv2.imwrite(os.path.join(render_dir, "00001.jpg"), gt_image[:, :, :3])
    
    sam3_track_predictor = sam3_model.tracker
    sam3_track_predictor.backbone = sam3_model.detector.backbone
    inference_state = sam3_track_predictor.init_state(
        video_path=render_dir
    )
    sam3_track_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=0,
        mask=robot_mask,
    )
    sam3_track_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=table_mask,
    )
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in sam3_track_predictor.propagate_in_video(
        inference_state, 
        start_frame_idx=0, 
        max_frame_num_to_track=300, 
        reverse=False, 
        propagate_preflight=True
    ):
        if frame_idx >= 2:
            robot_mask = (video_res_masks[0] > 0).squeeze().cpu().numpy().astype(np.uint8) * 255
            table_mask = (video_res_masks[1] > 0).squeeze().cpu().numpy().astype(np.uint8) * 255
            all_mask = np.clip(robot_mask + table_mask, 0, 255)
            save_path = os.path.join(masks_dir, f"1{(frame_idx-2):04d}.png")
            cv2.imwrite(save_path, all_mask)

    losses = []
    for i in range(args.num_particles):
        pred_mask = load_mask_from_file(os.path.join(masks_dir, f"1{i:04d}.png"))
        particle_loss = F.binary_cross_entropy(pred_mask, gt_mask_all).item()
        losses.append(particle_loss.item() if torch.is_tensor(particle_loss) else particle_loss)
    return losses

def get_shortest_dist(target, current):
    """
    Calculate the shortest angular distance between two angles in radians.
    Ensures the result is within the range [-pi, pi] to avoid 'long way around' issues.
    """
    return (target - current + np.pi) % (2 * np.pi) - np.pi

def pso_optimization(loss_fn, w_start=0.5, w_end=0.2, c1=0.8, c2=2.5, angle_std=0.2, trans_std=0.05, initial_matrix=None, args=None, pso_log_dir=None,
                     sam3_model=None, genesis_joint_renderer=None):
    """
    Optimized PSO for 6D Pose refinement (20-iteration version).
    Includes Shortest Path logic, Decoupled R/T scales, and Full Logging.
    """
    num_iters = args.num_iterations
    
    # 1. Scale Decoupling: Euler angles are more sensitive than translation
    # 2. Velocity Clamping: Prevents particles from jumping over the 0.4 loss "well"
    v_angle_max = angle_std * 0.5
    v_trans_max = trans_std * 0.5

    # Extract initial pose components
    init_rotation = R.from_matrix(initial_matrix[:3, :3])
    init_angle = init_rotation.as_euler('xyz', degrees=False)
    init_translation = initial_matrix[:3, 3]

    # Initialize positions with Gaussian noise
    particles_angle = np.random.randn(args.num_particles, 3) * angle_std + init_angle
    particles_translation = np.random.randn(args.num_particles, 3) * trans_std + init_translation
    
    # Elite preservation: Particle 0 is the baseline
    particles_angle[0] = init_angle
    particles_translation[0] = init_translation

    # Cold Start: Initialize velocities to zero
    particles_dangle = np.zeros_like(particles_angle)
    particles_dtranslation = np.zeros_like(particles_translation)

    # Pre-allocate matrix buffer for loss_fn
    particles_position = np.zeros((args.num_particles, 4, 4))
    for i in range(args.num_particles):
        particles_position[i, :3, :3] = R.from_euler('xyz', particles_angle[i]).as_matrix()
        particles_position[i, :3, 3] = particles_translation[i]
        particles_position[i, 3, 3] = 1

    # Initial Personal Best (Iteration 0)
    personal_best_angle = particles_angle.copy()
    personal_best_trans = particles_translation.copy()
    
    print("Evaluating Initial Particles...")
    current_losses = np.array(loss_fn(particles_position, args, 0, pso_log_dir, sam3_model, genesis_joint_renderer))
    personal_best_loss = current_losses.copy()

    gb_idx = np.argmin(personal_best_loss)
    global_best_angle = personal_best_angle[gb_idx].copy()
    global_best_trans = personal_best_trans[gb_idx].copy()
    global_best_loss = personal_best_loss[gb_idx]
    global_best_iteration = 0

    # Main Optimization Loop
    for iteration in range(1, num_iters + 1):
        # Linear decay of w for fine-tuning in late stages
        w_current = w_start - (w_start - w_end) * (iteration / num_iters)
        
        iter_dir = os.path.join(pso_log_dir, str(iteration))
        os.makedirs(iter_dir, exist_ok=True)

        for i in range(args.num_particles):
            r1, r2 = np.random.rand(3), np.random.rand(3)
            
            # --- ROTATION UPDATE (Shortest Path Logic) ---
            # Calculate shortest angular distance to avoid +/- PI wrapping issues
            diff_pbest_angle = get_shortest_dist(personal_best_angle[i], particles_angle[i])
            diff_gbest_angle = get_shortest_dist(global_best_angle, particles_angle[i])
            
            d_angle = (w_current * particles_dangle[i] + 
                       c1 * r1 * diff_pbest_angle + 
                       c2 * r2 * diff_gbest_angle)
            
            # Apply Clamping and Update Position
            particles_dangle[i] = np.clip(d_angle, -v_angle_max, v_angle_max)
            particles_angle[i] += particles_dangle[i]

            # --- TRANSLATION UPDATE ---
            diff_pbest_trans = personal_best_trans[i] - particles_translation[i]
            diff_gbest_trans = global_best_trans - particles_translation[i]
            
            d_trans = (w_current * particles_dtranslation[i] + 
                       c1 * r1 * diff_pbest_trans + 
                       c2 * r2 * diff_gbest_trans)
            
            # Apply Clamping and Update Position
            particles_dtranslation[i] = np.clip(d_trans, -v_trans_max, v_trans_max)
            particles_translation[i] += particles_dtranslation[i]

            # Update transformation matrix for the next loss evaluation
            particles_position[i, :3, :3] = R.from_euler('xyz', particles_angle[i]).as_matrix()
            particles_position[i, :3, 3] = particles_translation[i]

        # Batch Evaluation
        current_losses = np.array(loss_fn(particles_position, args, iteration, pso_log_dir, sam3_model, genesis_joint_renderer))

        # Update PBEST
        for i in range(args.num_particles):
            if current_losses[i] < personal_best_loss[i]:
                personal_best_loss[i] = current_losses[i]
                personal_best_angle[i] = particles_angle[i].copy()
                personal_best_trans[i] = particles_translation[i].copy()
        
        # Update GBEST and Log Data
        if np.min(current_losses) < global_best_loss:
            best_idx = np.argmin(current_losses)
            global_best_loss = current_losses[best_idx]
            global_best_angle = particles_angle[best_idx].copy()
            global_best_trans = particles_translation[best_idx].copy()
            global_best_iteration = iteration
            gb_idx = best_idx

        # Construct GBEST matrix for saving
        global_best_position = np.eye(4)
        global_best_position[:3, :3] = R.from_euler('xyz', global_best_angle).as_matrix()
        global_best_position[:3, 3] = global_best_trans

        # --- RESTORED LOGGING LOGIC ---
        with open(os.path.join(iter_dir, "meta.json"), "w") as f:
            json.dump({
                "best_particle_loss": float(global_best_loss), 
                "best_particle_iteration": int(global_best_iteration),
                "best_particle_index": int(gb_idx),
                "w_current": float(w_current)
            }, f)

        # Copy best rendering visuals for debugging
        best_iter_render_dir = os.path.join(pso_log_dir, str(global_best_iteration), "render")
        if os.path.exists(best_iter_render_dir):
            try:
                shutil.copy(os.path.join(best_iter_render_dir, "00000.jpg"), os.path.join(iter_dir, "gt_best.jpg"))
                shutil.copy(os.path.join(best_iter_render_dir, f"1{gb_idx:04d}.jpg"), os.path.join(iter_dir, "render_best.jpg"))
            except Exception as e:
                print(f"Logging Error: {e}")
        
        np.savetxt(os.path.join(iter_dir, "best_w2c.txt"), global_best_position, fmt='%.10f')
        shutil.copy(os.path.join(iter_dir, "best_w2c.txt"), os.path.join(pso_log_dir, "best_w2c.txt"))
        
        print(f"Iteration {iteration}/{num_iters}, Global Best Loss: {global_best_loss:.4f}")

    return global_best_position

def main():
    parser = argparse.ArgumentParser(description="PSO-based viewpoint refiner using Grounded SAM and a simulator.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument('--bg_color', type=str, choices=['white', 'black'], default='black', help='Background color for rendering')
    parser.add_argument("--num_particles", type=int, default=100, help="Number of PSO particles")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of PSO iterations")
    parser.add_argument("--angle_std", type=float, default=0.2, help="Standard deviation for angle perturbation")
    parser.add_argument("--trans_std", type=float, default=0.05, help="Standard deviation for translation perturbation")
    parser.add_argument("--w_start", type=float, default=0.72, help="PSO inertia weight start")
    parser.add_argument("--w_end", type=float, default=0.4, help="PSO inertia weight end")
    parser.add_argument("--c1", type=float, default=1.49, help="PSO cognitive coefficient")
    parser.add_argument("--c2", type=float, default=1.49, help="PSO social coefficient")
    parser.add_argument("--background_timestamp", type=str, default="asset", help="Background timestamp")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sam3_model = build_sam3_video_model(checkpoint_path="checkpoints/sam3/sam3.pt")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    init_trans = os.path.join(args.output_dir, "pose.npy") # world-to-camera
    pso_log_dir = os.path.join(args.output_dir, "pso_logs")
    initial_matrix = np.load(init_trans)[-1]

    # Create logs directory
    os.makedirs(pso_log_dir, exist_ok=True)
    
    genesis_joint_renderer = initialize_simulator(args.background_timestamp, os.path.join(args.output_dir, "cam_K.txt"))
    # Start optimization
    optimized_matrix = pso_optimization(
        loss_function_batch,
        w_start=args.w_start,
        w_end=args.w_end,
        angle_std=args.angle_std,
        trans_std=args.trans_std,
        c1=args.c1,
        c2=args.c2,
        initial_matrix=initial_matrix,
        args=args,
        pso_log_dir=pso_log_dir,
        sam3_model=sam3_model,
        genesis_joint_renderer=genesis_joint_renderer,
    )

    print("Optimized World-to-Camera:")
    print(optimized_matrix)


if __name__ == "__main__":
    main()
