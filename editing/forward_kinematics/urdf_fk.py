from urdfpy import URDF
import numpy as np
import os
import open3d as o3d
import argparse
import copy

import sys
from simulation.utils.constants import FR3_DEFAULT_CFG

def trimesh_to_open3d(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    return o3d_mesh

def urdf_fk(input_path, cfg=None, save=True, clean_vertex_normals=True):
    urdf_path = os.path.abspath(input_path)
    urdf_robot = URDF.load(urdf_path)
    if cfg is None:
        cfg = FR3_DEFAULT_CFG
    lfk = urdf_robot.link_fk(cfg=cfg)
    combined = None
    urdf_fk_dict = dict()
    for link in lfk:
        visual_meshes = None
        for visual in link.visuals:
            filepath = os.path.join(os.path.dirname(urdf_path), visual.geometry.mesh.filename)
            for mesh in visual.geometry.meshes:
                pose = lfk[link].dot(visual.origin)
                if visual.geometry.mesh is not None:
                    if visual.geometry.mesh.scale is not None:
                        S = np.eye(4, dtype=np.float64)
                        S[:3,:3] = np.diag(visual.geometry.mesh.scale)
                        pose = pose.dot(S)
                o3d_mesh = trimesh_to_open3d(mesh)
                o3d_mesh.transform(pose)
                if combined is None:
                    combined = copy.deepcopy(o3d_mesh)
                else:
                    combined += o3d_mesh
                if visual_meshes is None:
                    visual_meshes = copy.deepcopy(o3d_mesh)
                else:
                    visual_meshes += o3d_mesh
        if visual_meshes is not None:
            urdf_fk_dict[link.name] = visual_meshes
    if save or clean_vertex_normals:
        o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(urdf_path), "all_fk.obj"), combined, write_vertex_normals=not clean_vertex_normals)
        if clean_vertex_normals:
            combined = o3d.io.read_triangle_mesh(os.path.join(os.path.dirname(urdf_path), "all_fk.obj"))
    return urdf_fk_dict, combined

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="",
                        help='Input URDF File')
    parser.add_argument("--save", action="store_true",
                        help="Whether to save the fked parts")
    args = parser.parse_args()
    urdf_fk(args.input, args.save)