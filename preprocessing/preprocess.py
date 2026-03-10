from sdfstudio.utils import env
env.set_env_variables()

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
# from grounded_sam_toolkit import do_grounded_sam
from sam3_toolkit import do_sam3
from depth_anything_v2_toolkit import do_depth_anything_v2

import argparse
import numpy as np
from pathlib import Path
from rich.console import Console
from sdfstudio.process_data.process_data_utils import CAMERA_MODELS, convert_video_to_images
from sdfstudio.process_data.process_data_utils import downscale_images as ss_downscale_images
from sdfstudio.utils import install_checks
from scripts.process_data import ProcessImages
import json
from tqdm import tqdm
from PIL import Image
import rich
from functools import partial
import cv2
CONSOLE = Console()

def video2images(args):
    install_checks.check_ffmpeg_installed()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / args.image_ori_name
    image_dir.mkdir(parents=True, exist_ok=True)
    # image_dir_jpg = output_dir / "images_ori_jpg"
    # image_dir_jpg.mkdir(parents=True, exist_ok=True)
    num_frames_all = 0
    if os.path.isdir(args.data):
        video_filenames = sorted(os.listdir(args.data))
        frame_nums = []
        for video_filename in video_filenames:
            cap = cv2.VideoCapture(os.path.join(args.data, video_filename))
            if cap.isOpened():
                frame_num = cap.get(7)
                frame_nums.append(frame_num)

        for video_filename, frame_num in zip(video_filenames, frame_nums):
            CONSOLE.rule(f"[bold green] Processing {video_filename}")
            # Convert video to images
            summary_log, num_extracted_frames = convert_video_to_images(
                os.path.join(args.data, video_filename), 
                image_dir=image_dir, 
                num_frames_target=int(args.num_frames_target * frame_num / sum(frame_nums)), 
                start_number=num_frames_all, 
                verbose=False,
                no_delete=True,
                reverse=args.reverse,
            )
            # if args.track_sam:
            #     summary_log, num_extracted_frames = convert_video_to_images(
            #         os.path.join(args.data, video_filename), 
            #         image_dir=image_dir_jpg, 
            #         num_frames_target=int(args.num_frames_target * frame_num / sum(frame_nums)), 
            #         start_number=num_frames_all, 
            #         verbose=False,
            #         no_delete=True,
            #         fmt="jpg",
            #         template="%05d",
            #         quality=1,
            #         reverse=args.reverse,
            #     )
            num_frames_all += num_extracted_frames
    else:
        video_filename = args.data
        CONSOLE.rule(f"[bold green] Processing {os.path.basename(video_filename)}")
        # Convert video to images
        summary_log, num_extracted_frames = convert_video_to_images(
            video_filename, 
            image_dir=image_dir, 
            num_frames_target=args.num_frames_target, 
            start_number=num_frames_all, 
            verbose=False,
            no_delete=True,
            reverse=args.reverse,
        )
        # if args.track_sam:
        #     summary_log, num_extracted_frames = convert_video_to_images(
        #         video_filename, 
        #         image_dir=image_dir_jpg, 
        #         num_frames_target=args.num_frames_target, 
        #         start_number=num_frames_all, 
        #         verbose=False,
        #         no_delete=True,
        #         fmt="jpg",
        #         template="%05d",
        #         quality=1,
        #         reverse=args.reverse,
        #     )
        num_frames_all += num_extracted_frames
    return num_frames_all

def remove_background(args):
    # if args.segment_type == "grounded_sam":
    #     do_grounded_sam(args)
    if args.segment_type == "sam3":
        do_sam3(args)
    elif args.segment_type == "none":
        copy_images(args)
    else:
        raise RuntimeError(f"Unknown segment type {args.segment_type}")
    
def copy_images(args):
    image_dir = os.path.join(args.output_dir, "images_ori")
    rgb_dir = os.path.join(args.output_dir, "images")
    os.system(f"cp -r {image_dir} {rgb_dir}")

def do_colmap(args, skip_colmap=False):
    processor = ProcessImages(
        data=None,
        output_dir=Path(args.output_dir),
        no_copy=True,
        no_downscale=True,
        skip_colmap=skip_colmap,
        matching_method=args.colmap_method,
        verbose=True,
        use_ori_for_colmap=True,
    )
    processor.main()

def downscale_images(args):
    image_dir = os.path.join(args.output_dir, "images")
    ss_downscale_images(Path(image_dir), args.num_downscales, verbose=True)
    
def downscale_masks(args):
    downscale_factors = [2**(i+1) for i in range(args.num_downscales)]
    mask_dir = os.path.join(args.output_dir, "masks")
    if not os.path.exists(mask_dir):
        return
    if len(os.listdir(mask_dir)) == 0:
        return
        
    for downscale_factor in downscale_factors:
        output_mask_dir = os.path.join(args.output_dir, f"masks_{downscale_factor}")
        os.makedirs(output_mask_dir, exist_ok=True)

    for downscale_factor in downscale_factors:
        for filename in tqdm(os.listdir(mask_dir), desc=f'downscale {downscale_factor}'):
            mask_filepath = os.path.join(mask_dir, filename)
            output_mask_dir = os.path.join(args.output_dir, f"masks_{downscale_factor}")
            pil_mask = Image.open(mask_filepath)
            width, height = pil_mask.size
            newsize = (int(width // downscale_factor), int(height // downscale_factor))
            pil_mask = pil_mask.resize(newsize, resample=Image.NEAREST)
            pil_mask.convert("L").save(os.path.join(output_mask_dir, filename))

def remask(args):
    input_img_dir = os.path.join(args.output_dir, f"images_ori")
    output_mask_dir = os.path.join(args.output_dir, f"masks")
    output_img_dir = os.path.join(args.output_dir, f"images")
    for filename in tqdm(os.listdir(output_mask_dir), desc=f'Remasking ...'):
        input_image_filepath = os.path.join(input_img_dir, filename)
        output_image_filepath = os.path.join(output_img_dir, filename)
        ori_image = cv2.imread(input_image_filepath)
        mask_filepath = os.path.join(output_mask_dir, filename)
        mask_bool = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE).astype(bool)
        bg_color = 255 if args.bg_color == "white" else 0
        bg_image = np.ones_like(ori_image) * bg_color
        masked_image = np.where(mask_bool[..., None], ori_image, bg_image)
        cv2.imwrite(output_image_filepath, masked_image)

def downscale_depths(args):
    downscale_factors = [2**(i+1) for i in range(args.num_downscales)]
    depth_dir = os.path.join(args.output_dir, "mono_depths")
    for downscale_factor in downscale_factors:
        output_depth_dir = os.path.join(args.output_dir, f"mono_depths_{downscale_factor}")
        os.makedirs(output_depth_dir, exist_ok=True)

    for downscale_factor in downscale_factors:
        for filename in tqdm(os.listdir(depth_dir), desc=f'downscale {downscale_factor}'):
            depth_filepath = os.path.join(depth_dir, filename)
            output_depth_dir = os.path.join(args.output_dir, f"mono_depths_{downscale_factor}")
            depth = np.load(depth_filepath)
            height, width = depth.shape
            depth = cv2.resize(depth, (int(width // downscale_factor), int(height // downscale_factor)), interpolation=cv2.INTER_NEAREST)
            np.save(os.path.join(output_depth_dir, filename), depth)

def execute_safely(func, args, do_exit=True):
    try:
        func(args)
        return True
    except Exception as e:
        console = rich.console()
        console.print("[bold red]ERROR:[/bold red]", e)
        if do_exit:
            exit(1)
        return False

execute_safely_noexit = partial(execute_safely, do_exit=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='input videos, separated with comma')
    parser.add_argument('--output-dir', type=str, help='output image directory')
    parser.add_argument('--num-frames-target', type=int, default=300, help='target frame number')
    parser.add_argument('--num-downscales', type=int, default=0, help='downscale numbers')
    parser.add_argument('--redo-colmap', action='store_true')
    parser.add_argument('--redo-segment', action='store_true')
    parser.add_argument('--redo-mono-depth', action='store_true')
    parser.add_argument("--redo-video2images", action="store_true")
    parser.add_argument('--remask', action='store_true')
    parser.add_argument('--segment-type', type=str, default='sam3', choices=['sam3', 'grounded_sam', 'none'])
    parser.add_argument('--segment-prompt', type=str, default=None)
    parser.add_argument('--bg-color', type=str, default='black', choices=['white', 'black'])
    parser.add_argument('--colmap-method', type=str, default='exhaustive', choices=['vocab_tree', 'sequential', 'exhaustive'])
    parser.add_argument("--reverse", action='store_true')
    parser.add_argument("--image-ori-name", type=str, default="images_ori")
    args = parser.parse_args()
    
    if args.remask:
        CONSOLE.rule("[bold green]Remask")
        execute_safely(remask, args)
    else:
        if args.redo_video2images:
            CONSOLE.rule("[bold green]Video2Images")
            execute_safely(video2images, args)
            
        if args.redo_segment:
            CONSOLE.rule("[bold green]Remove background")
            execute_safely(remove_background, args)
        
        if args.redo_mono_depth:
            CONSOLE.rule("[bold green]Mono depths")
            execute_safely(do_depth_anything_v2, args)
                
        if args.redo_colmap:
            CONSOLE.rule("[bold green]Running COLMAP")
            execute_safely(do_colmap, args)
        
        if not (args.redo_segment or args.redo_colmap or args.redo_video2images):
            CONSOLE.rule("[bold green]Video2Images")
            execute_safely(video2images, args)
            CONSOLE.rule("[bold green]Mono depths")
            execute_safely(do_depth_anything_v2, args)
            CONSOLE.rule("[bold green]Remove background")
            execute_safely(remove_background, args)
            CONSOLE.rule("[bold green]Running COLMAP")
            execute_safely(do_colmap, args)
