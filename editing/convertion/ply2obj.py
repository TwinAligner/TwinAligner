'''
Credit: https://github.com/jiegec/blender-scripts/blob/master/bake_vertex_colors_to_texture_image.py
'''

import bpy
import argparse
import sys
import os

def blender_ply2obj(input_ply, output_obj, texture_size=1024, axes_type="YZ"):
    bpy.ops.wm.read_homefile()
    name, _ = os.path.splitext(output_obj)
    output_png = '{}_texture.png'.format(name)
    output_mtl = '{}_material.mtl'.format(name)
    # https://docs.blender.org/api/current/bpy.data.html
    if "Cube" in bpy.data.meshes:
        mesh = bpy.data.meshes["Cube"]
        bpy.data.meshes.remove(mesh)
    bpy.ops.wm.ply_import(filepath=input_ply, forward_axis="Y", up_axis="Z")
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    # https://blender.stackexchange.com/questions/5668/add-nodes-to-material-with-python
    material = bpy.data.materials.new('SomeMaterial')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bpy.ops.object.editmode_toggle()
    input_node = nodes.new('ShaderNodeVertexColor')
    bsdf_node = nodes.get('Principled BSDF')
    material.node_tree.links.new(bsdf_node.inputs[0], input_node.outputs[0])
    texture_node = nodes.new('ShaderNodeTexImage')
    image = bpy.data.images.new(name='SomeImage', width=texture_size, height=texture_size)
    texture_node.image = image
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.active_object.active_material = material
    bpy.context.view_layer.objects.active = bpy.context.active_object
    bpy.ops.object.bake(type='DIFFUSE',
                        pass_filter={'COLOR'}, use_clear=True)
    image.save_render(output_png)
    # set map_Kd correctly in mtl file
    image.filepath = os.path.basename(output_png)
    material.node_tree.links.new(bsdf_node.inputs[0], texture_node.outputs[0])
    if axes_type == "YZ":
        bpy.ops.wm.obj_export(filepath=output_obj, forward_axis="Y", up_axis="Z")
    elif axes_type == "-XY":
        bpy.ops.wm.obj_export(filepath=output_obj, forward_axis="NEGATIVE_X", up_axis="Y")
    else:
        raise NotImplementedError(f"Unknown axes type {args.axes_type}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Bake vertex colors to texture image')
    parser.add_argument('--input', type=str,
                        help='Input PLY File')
    parser.add_argument('--output', type=str, default=None,
                        help='Output OBJ File')
    parser.add_argument("--axes-type", type=str, choices=["-XY", "YZ"], default="YZ")
    args = parser.parse_args()
    input_ply = args.input
    if args.output is None:
        output_obj = args.input.replace(".PLY", ".OBJ").replace(".ply", ".obj")
    else:
        output_obj = args.output
    blender_ply2obj(input_ply, output_obj, axes_type=args.axes_type)
