import argparse
import os
import cv2
import numpy as np
import gradio as gr
import traceback
import imageio
import time
import shutil

from sam3.model_builder import build_sam3_video_predictor


def _bgr_to_rgb(image_bgr):
    """Convert BGR (cv2) to RGB (Gradio / SAM3 internal)."""
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _draw_points_on_image(image_bgr, points_px, labels, r=10):
    """Draw positive (green) and negative (red) points on the image. Input/output BGR for cv2."""
    if image_bgr is None or not points_px:
        return image_bgr
    disp = image_bgr.copy()
    for (px, py), lab in zip(points_px, labels):
        # cv2 uses BGR: green=(0,255,0), red=(0,0,255)
        color = (0, 255, 0) if lab == 1 else (0, 0, 255)
        cv2.circle(disp, (int(px), int(py)), r, color, 2)
    return disp


def _write_mask_preview_video(responses, image_dir, image_filelist, args, out_path, fps=15):
    """Write masked RGB frames from propagate responses to an MP4 using imageio for better compatibility."""
    if not responses or not image_filelist:
        return None
    
    # Unique filename to avoid browser caching
    dir_name = os.path.dirname(out_path)
    base_name = os.path.basename(out_path).replace(".mp4", "")
    unique_out_path = os.path.join(dir_name, f"{base_name}_{int(time.time() * 1000)}.mp4")
    
    bg_color = 255 if args.bg_color == "white" else 0
    frames = []
    
    for response_idx, response in enumerate(responses):
        if response_idx >= len(image_filelist):
            break
        image_filename = image_filelist[response_idx]
        ori_image = cv2.imread(os.path.join(image_dir, image_filename))
        if ori_image is None:
            continue
        
        # SAM3 output is often binary masks
        mask_bool = response["outputs"]["out_binary_masks"].sum(0).astype(bool)
        bg_image = np.ones_like(ori_image) * bg_color
        masked = np.where(mask_bool[..., None], ori_image, bg_image)
        
        # Convert BGR (cv2) to RGB (imageio expects RGB)
        frames.append(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    
    if not frames:
        return None

    # Write video using imageio (uses its own ffmpeg)
    try:
        # 'libx264' is standard for web playback
        imageio.mimsave(unique_out_path, frames, fps=fps, codec='libx264', quality=8)
    except Exception as e:
        print(f"Imageio failed with libx264: {e}. Falling back to default.")
        imageio.mimsave(unique_out_path, frames, fps=fps)

    # Cleanup old preview files
    try:
        for f in os.listdir(dir_name):
            if f.startswith(base_name + "_") and f.endswith(".mp4") and f != os.path.basename(unique_out_path):
                os.remove(os.path.join(dir_name, f))
    except Exception:
        pass
        
    return unique_out_path


def _build_gradio_ui(args, video_predictor):
    """Build Gradio UI. Each action (init, add point, clear) runs the full do_sam3 pipeline."""
    image_dir = os.path.join(args.output_dir, args.image_dir_name)
    output_dir = os.path.join(args.output_dir, args.mask_dir_name)
    masked_rgb_dir = os.path.join(args.output_dir, args.masked_rgb_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masked_rgb_dir, exist_ok=True)
    
    image_filelist = sorted(os.listdir(image_dir))
    if not image_filelist:
        raise FileNotFoundError(f"No images found in {image_dir}")
        
    first_path = os.path.join(image_dir, image_filelist[0])
    first_image_bgr = cv2.imread(first_path)
    if first_image_bgr is None:
        raise FileNotFoundError(f"Cannot read the first frame: {first_path}")
    
    first_image_rgb = _bgr_to_rgb(first_image_bgr)
    h, w = first_image_bgr.shape[:2]
    preview_video_path = os.path.join(args.output_dir, "mask_preview.mp4")

    def apply_full_pipeline(points_px=None, labels=None, old_session_id=None):
        """
        Run the full do_sam3 pipeline with user points/labels.
        Returns: last display image, updated points_px, updated labels, video path, status message, new_session_id.
        """
        nonlocal video_predictor
        
        # Cleanup old session if it exists to free GPU memory
        if old_session_id:
            try:
                video_predictor.handle_request(dict(type="close_session", session_id=old_session_id))
            except Exception:
                pass

        # Clear output dirs for full reset
        for d in [output_dir, masked_rgb_dir]:
            if os.path.isdir(d):
                for fname in os.listdir(d):
                    try:
                        os.remove(os.path.join(d, fname))
                    except Exception:
                        pass

        points_px = list(points_px) if points_px else []
        labels = list(labels) if labels else []
        drawn_bgr = _draw_points_on_image(first_image_bgr, points_px, labels)
        out_img = _bgr_to_rgb(drawn_bgr)

        if not points_px:
            return out_img, [], [], None, "Add points on the image to start segmentation.", None

        try:
            # Start a new session
            response = video_predictor.handle_request(dict(type="start_session", resource_path=image_dir))
            session_id = response["session_id"]
            
            # Use points to initialize object (using obj_id=0 as default)
            obj_id = 0
            points_norm = [[x / w, y / h] for (x, y) in points_px]
            
            video_predictor.handle_request(dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text="",
            ))
            
            video_predictor.handle_request(dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                points=points_norm,
                point_labels=labels,
                obj_id=obj_id,
            ))
            
            # Propagate and save masks
            responses = list(video_predictor.handle_stream_request(dict(
                type="propagate_in_video", session_id=session_id
            )))
            
            for response_idx, response in enumerate(responses):
                if response_idx >= len(image_filelist): break
                image_filename = image_filelist[response_idx]
                ori_image = cv2.imread(os.path.join(image_dir, image_filename))
                if ori_image is None: continue
                
                mask_bool = response["outputs"]["out_binary_masks"].sum(0).astype(bool)
                mask_uint8 = mask_bool.astype(np.uint8) * 255
                cv2.imwrite(os.path.join(output_dir, image_filename), mask_uint8)
                
                bg_color = 255 if args.bg_color == "white" else 0
                bg_image = np.ones_like(ori_image) * bg_color
                masked_image = np.where(mask_bool[..., None], ori_image, bg_image)
                cv2.imwrite(os.path.join(masked_rgb_dir, image_filename), masked_image)

            path = _write_mask_preview_video(responses, image_dir, image_filelist, args, preview_video_path)
            return out_img, points_px, labels, path, f"Propagated {len(responses)} frames. Mask video updated.", session_id
        except Exception as e:
            traceback.print_exc()
            return out_img, points_px, labels, gr.update(), f"Pipeline failed: {e}", (session_id if 'session_id' in locals() else old_session_id)

    with gr.Blocks(title="SAM3 Interactive Mask Annotation", css=".point-hint { font-size: 0.9em; color: #666; }") as demo:
        gr.Markdown("## SAM3 Interactive: Add points to define and refine mask; propagation runs automatically.")

        # States
        points_px_state = gr.State([])
        labels_state = gr.State([])
        session_id_state = gr.State(None)

        with gr.Row():
            point_mode = gr.Radio(choices=["Positive Point (inside target)", "Negative Point (outside target)"], value="Positive Point (inside target)", label="Next Click Type")
        with gr.Row():
            img_display = gr.Image(value=first_image_rgb, label="First frame: Click to add points (Green=positive, Red=negative).", type="numpy", height=400)
            mask_video_out = gr.Video(label="Mask video (masked RGB)", height=400, autoplay=True, loop=True)
        with gr.Row():
            clear_btn = gr.Button("Clear points (full reset)")
            quit_btn = gr.Button("Quit & Finish", variant="primary")
        status_out = gr.Textbox(label="Status", interactive=False)

        def on_init():
            return first_image_rgb, [], [], None, "Ready. Click on the image to add your first point.", None

        def on_quit():
            print("Quit button clicked. Shutting down Gradio and exiting...")
            # Use os._exit to force immediate termination of the process and all threads
            import os
            os._exit(0)

        def on_image_select(evt: gr.SelectData, points_px, labels, mode, old_sid):
            if evt is None or not isinstance(evt.index, (list, tuple)) or len(evt.index) < 2:
                return gr.update(), points_px, labels, gr.update(), "Invalid click.", old_sid
            ix, iy = evt.index
            points_px = list(points_px) if points_px else []
            labels = list(labels) if labels else []
            lab = 1 if "Positive" in str(mode) else 0
            points_px.append((ix, iy))
            labels.append(lab)
            return apply_full_pipeline(points_px, labels, old_sid)

        def on_clear(old_sid):
            return apply_full_pipeline([], [], old_sid)

        # Common Outputs for all actions
        common_outputs = [img_display, points_px_state, labels_state, mask_video_out, status_out, session_id_state]

        # Events
        demo.load(on_init, outputs=common_outputs)
        img_display.select(on_image_select, inputs=[points_px_state, labels_state, point_mode, session_id_state], outputs=common_outputs)
        clear_btn.click(on_clear, inputs=[session_id_state], outputs=common_outputs)
        quit_btn.click(on_quit)

    return demo


def do_sam3_interactive_mask(args):
    """Launch Gradio: pre-build predictor once; init session with prompt; each point click auto-propagates and shows mask video."""
    sam3_checkpoint = "checkpoints/sam3/sam3.pt"
    video_predictor = build_sam3_video_predictor(checkpoint_path=sam3_checkpoint)
    demo = _build_gradio_ui(args, video_predictor)
    demo.launch(
        server_name=getattr(args, "gradio_server", "127.0.0.1"),
        server_port=getattr(args, "gradio_port", 7860),
    )


def do_sam3(args, video_predictor=None):
    sam3_checkpoint = "checkpoints/sam3/sam3.pt"
    image_dir = os.path.join(args.output_dir, args.image_dir_name)
    output_dir = os.path.join(args.output_dir, args.mask_dir_name)
    masked_rgb_dir = os.path.join(args.output_dir, args.masked_rgb_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masked_rgb_dir, exist_ok=True)
    if video_predictor is None:
        video_predictor = build_sam3_video_predictor(
            checkpoint_path=sam3_checkpoint
        )
    # Start a session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=image_dir,
        )
    )
    session_id = response["session_id"]
    prompt = args.segment_prompt
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0, # Arbitrary frame index
            text=prompt,
        )
    )

    image_filelist = sorted(os.listdir(image_dir))
    responses = video_predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    )
    for response_idx, response in enumerate(responses):
        image_filename = image_filelist[response_idx]
        ori_image = cv2.imread(os.path.join(image_dir, image_filename))
        mask_bool = response['outputs']['out_binary_masks'].sum(0).astype(bool)
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, image_filename), mask_uint8)
        bg_color = 255 if args.bg_color == "white" else 0
        bg_image = np.ones_like(ori_image) * bg_color
        masked_image = np.where(mask_bool[..., None], ori_image, bg_image)
        cv2.imwrite(os.path.join(masked_rgb_dir, image_filename), masked_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--segment_prompt', type=str)
    parser.add_argument('--image_dir_name', type=str, default="images_ori")
    parser.add_argument('--mask_dir_name', type=str, default="masks")
    parser.add_argument('--masked_rgb_dir_name', type=str, default="images")
    parser.add_argument("--bg_color", type=str, default="white", choices=['white', 'black'])
    parser.add_argument(
        "--interactive_mask",
        action="store_true",
        help="Use Gradio interactive GUI: text prompt + refine mask with positive/negative points, then propagate to the whole video.",
    )
    parser.add_argument(
        "--gradio_server",
        type=str,
        default="0.0.0.0",
        help="Gradio binding address. Use 0.0.0.0 for external access (only applies with --interactive_mask).",
    )
    parser.add_argument(
        "--gradio_port",
        type=int,
        default=7860,
        help="Gradio port (only applies with --interactive_mask).",
    )
    args = parser.parse_args()
    if args.interactive_mask:
        do_sam3_interactive_mask(args)
    else:
        do_sam3(args)
