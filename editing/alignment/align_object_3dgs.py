'''
Partly from https://github.com/yzslab/gaussian-splatting-lightning/blob/main/internal/utils/gaussian_utils.py
'''

import torch
import numpy as np
import os
import sys
import argparse
from urdfpy import URDF
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from simulation.fast_gaussian_model_manager import construct_from_ply, axis_angle_to_quaternion, save_to_ply

def main(args):
    rel2abs = np.load(os.path.join(os.path.dirname(args.input), "rel2abs.npz"))
    scale, rot, trans = rel2abs["scale"], rel2abs["rot"], rel2abs["trans"]
    
    if args.input.endswith(".urdf"):
        robot = URDF.load(args.input)
        fk = robot.link_fk()
        for urdf_link, transmat in fk.items():
            part_3dgs_path = os.path.join(os.path.dirname(args.input), "parts", f"{urdf_link.name}_3dgs.ply")
            gs_model = construct_from_ply(part_3dgs_path, torch.device(args.device))
            gs_model.translate(torch.from_numpy(trans).to(gs_model._xyz.dtype).to(args.device))
            gs_model.rotate(axis_angle_to_quaternion(torch.from_numpy(rot).to(gs_model._rotation.dtype).to(args.device)))
            gs_model.scale(torch.from_numpy(scale).to(gs_model._scaling.dtype).to(args.device))
            save_to_ply(gs_model, part_3dgs_path.replace(".ply", "_abs.ply"))
    else:
        gs_model = construct_from_ply(args.input, torch.device(args.device))
        gs_model.translate(torch.from_numpy(trans).to(gs_model._xyz.dtype).to(args.device))
        gs_model.rotate(axis_angle_to_quaternion(torch.from_numpy(rot).to(gs_model._rotation.dtype).to(args.device)))
        gs_model.scale(torch.from_numpy(scale).to(gs_model._scaling.dtype).to(args.device))
        save_to_ply(gs_model, args.input.replace(".ply", "_abs.ply"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/home/hwfan/workspace/twinmanip/outputs/component-asset/object_3dgs.ply")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)
    
