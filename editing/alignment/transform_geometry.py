import numpy as np
import open3d as o3d
import argparse

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

def apply_transformation_point_cloud(input_ply, output_ply, s2t):
    pcd = o3d.io.read_point_cloud(input_ply)
    if pcd.is_empty():
        print("Error: pcd is empty!")
        exit()

    points = np.asarray(pcd.points)
    transformed_points = (s2t[:3, :3] @ points.T).T + s2t[:3, 3]
    pcd.points = o3d.utility.Vector3dVector(transformed_points)
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"Saved transformed pcd: {output_ply}")

def apply_transformation_mesh(input_ply, output_ply, s2t):
    mesh = o3d.io.read_triangle_mesh(input_ply)
    if mesh.is_empty():
        print("Error: mesh is empty!")
        exit()

    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None

    U, S, Vt = np.linalg.svd(s2t[:3, :3])
    scale = S[0]
    rotation = U @ Vt
    translation = s2t[:3, 3]

    transformed_vertices = (rotation @ vertices.T).T * scale + translation

    if normals is not None:
        transformed_normals = (rotation @ normals.T).T

    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    if normals is not None:
        mesh.vertex_normals = o3d.utility.Vector3dVector(transformed_normals)

    o3d.io.write_triangle_mesh(output_ply, mesh)
    print(f"Saved mesh: {output_ply}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='transform PLY mesh using transformation matrix')
    parser.add_argument("-i", '--input', type=str, help='Input PLY File')
    parser.add_argument("-o", '--output', type=str, help='Output ply File')
    parser.add_argument('-m', "--matrix", type=str, help='matrix File')
    parser.add_argument("--is-mesh", action="store_true")
    
    args = parser.parse_args()
    transformation_txt =  args.matrix
    input_ply = args.input
    output_ply = args.output

    s2t = load_transformation_matrix(args.matrix)
    if args.is_mesh:
        transformation_func = apply_transformation_mesh
    else:
        transformation_func = apply_transformation_point_cloud
    transformation_func(input_ply, output_ply, s2t)

