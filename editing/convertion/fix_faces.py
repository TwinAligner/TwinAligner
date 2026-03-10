import bpy
import os

def make_mesh_watertight(input_obj_path, output_obj_path):
    # Clear the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    initial_objects = set(bpy.context.scene.objects)

    # Import the OBJ file
    bpy.ops.wm.obj_import(filepath=input_obj_path)
    new_objects = set(bpy.context.scene.objects) - initial_objects
    obj = new_objects.pop()    

    # Check that the object is a mesh
    if obj.type != 'MESH':
        print("Error: imported object is not a mesh.")
        return

    # Switch to edit mode, select holes, and fill them
    try:
        # Set the object as active
        bpy.context.view_layer.objects.active = obj
        
        # Switch to edit mode
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Select all non-manifold edges (hole boundaries)
        bpy.ops.mesh.select_non_manifold(use_boundary=True)
        
        # Fill the selected holes
        bpy.ops.mesh.fill()
        
    except Exception as e:
        print(f"Error processing mesh: {e}")
    finally:
        # Ensure we switch back to object mode for subsequent operations
        bpy.ops.object.mode_set(mode='OBJECT')
    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Export the repaired OBJ file
    bpy.ops.wm.obj_export(filepath=output_obj_path)
    
    print(f"Mesh from '{input_obj_path}' has been processed and saved to '{output_obj_path}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="outputs/carrot-asset/parts/part_0_abs_reduced.obj")
    args = parser.parse_args()
    make_mesh_watertight(args.input_file, args.input_file.replace(".obj", "_watertight.obj"))
