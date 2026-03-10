import bpy

def reduce_obj_faces_by_ratio(input_obj_path, output_obj_path, target_face_count):
    """
    Reads an OBJ file, reduces its face count to a target number using a
    calculated ratio, and saves it as a new OBJ file. This script is
    compatible with Blender 4.1 and newer.

    Args:
        input_obj_path (str): The full path to the original OBJ file.
        output_obj_path (str): The save path for the new OBJ file.
        target_face_count (int): The target number of faces.
    """
    
    # 1. Clear the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # 2. Import the OBJ file
    print(f"Importing OBJ file: {input_obj_path}")
    bpy.ops.wm.obj_import(filepath=input_obj_path)
    
    # Get the imported object and ensure it's active
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    
    # 3. Get the initial face count
    initial_face_count = len(obj.data.polygons)
    print(f"Initial face count: {initial_face_count}")
    
    # Check for invalid target
    if target_face_count >= initial_face_count:
        print("Target face count is greater than or equal to the initial count. Skipping decimation.")
        bpy.ops.wm.obj_export(filepath=output_obj_path)
        print("Operation complete!")
        return
    
    # 4. Calculate the required decimation ratio
    # Ensure float division
    ratio_value = target_face_count / initial_face_count
    
    # 5. Add and configure the Decimate modifier
    print(f"Reducing faces with ratio: {ratio_value:.4f}")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    decimate_modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
    
    # The 'COLLAPSE' type is used for ratio-based decimation
    decimate_modifier.decimate_type = 'COLLAPSE'
    decimate_modifier.ratio = ratio_value
    
    # 6. Apply the modifier
    bpy.ops.object.modifier_apply(modifier="Decimate")
    
    # 7. Export the new OBJ file
    print(f"Exporting reduced mesh to: {output_obj_path}")
    bpy.ops.wm.obj_export(filepath=output_obj_path)
    
    print("Operation complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="outputs/carrot-asset/mesh_w_vertex_color_abs.obj")
    parser.add_argument("--target_faces", type=int, default=700)
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()
    reduce_obj_faces_by_ratio(args.input_file, args.input_file.replace(".obj", "_reduced.obj"), args.target_faces)