'''
Partly from https://github.com/yzslab/gaussian-splatting-lightning/blob/main/internal/utils/gaussian_utils.py
'''
import torch
import numpy as np
import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from simulation.fast_gaussian_model_manager import construct_from_ply, save_to_ply, matrix_to_quaternion

def load_transformation_matrix(txt_file):
    """ Load 4x4 transformation """
    try:
        s2t = np.loadtxt(txt_file)
        if s2t.shape != (4, 4):
            raise ValueError("Must be 4x4")
        print(f"Loaded 4x4 transformation:\n{s2t}")
        return s2t
    except Exception as e:
        print(f"Load 4x4 transformation error: {e}")
        exit()

def main(args):
    s2t = load_transformation_matrix(args.matrix)
    U, S, Vt = np.linalg.svd(s2t[:3, :3])
    scale = np.ones((3,))*S[0]
    rot = U @ Vt
    trans = s2t[:3, 3]
    gs_model = construct_from_ply(args.input, torch.device(args.device))
    gs_model.rotate(matrix_to_quaternion(torch.from_numpy(rot).to(gs_model._rotation.dtype).to(args.device)))
    gs_model.scale(torch.from_numpy(scale).to(gs_model._scaling.dtype).to(args.device))
    gs_model.translate(torch.from_numpy(trans).to(gs_model._xyz.dtype).to(args.device))
    save_to_ply(gs_model, args.output)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input', type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("-o", '--output', type=str, default="", help='Output ply File')
    parser.add_argument('-m', "--matrix", type=str, default="", help='matrix File')
    args = parser.parse_args()
    main(args)
