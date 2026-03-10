import cv2
import torch
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from depth_anything_v2.dpt import DepthAnythingV2
import numpy as np
from sdfstudio.utils import colormaps
import argparse
import os

def do_depth_anything_v2(args):
    image_dir = os.path.join(args.output_dir, "images_ori")
    mono_depth_dir = os.path.join(args.output_dir, "mono_depths")
    mono_depth_render_dir = os.path.join(args.output_dir, "mono_depths_render")
    os.makedirs(mono_depth_dir, exist_ok=True)
    os.makedirs(mono_depth_render_dir, exist_ok=True)
    device = "cuda"
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()
    for image_filename in tqdm(sorted(os.listdir(image_dir)), desc="Doing Depth Anything V2 ..."):
        image_path = os.path.join(image_dir, image_filename)
        mono_depth_path = os.path.join(mono_depth_dir, os.path.splitext(image_filename)[0] + ".png")
        raw_img = cv2.imread(image_path)
        depth = model.infer_image(raw_img)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = 1 - depth
        depth_png = (depth * 65535.0).astype(np.uint16)
        cv2.imwrite(mono_depth_path, depth_png)
        # np.save(mono_depth_path, depth)
        depth_map = colormaps.apply_depth_colormap(torch.tensor(depth).unsqueeze(-1), far_plane=depth.max()).cpu().numpy()
        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = np.flip(depth_map, -1)
        cv2.imwrite(os.path.join(mono_depth_render_dir, image_filename), depth_map)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    do_depth_anything_v2(args)

