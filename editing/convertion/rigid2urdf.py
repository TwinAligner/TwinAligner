import argparse
import sys
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, tostring, SubElement, Comment, ElementTree, XML
import xml.dom.minidom
def export_mesh_urdf(input_path):
    filename = os.path.basename(input_path)
    root = Element('robot', name=os.path.basename(os.path.dirname(input_path)))
    link_list = []
    element = Element('link', name="object")
    visual = SubElement(element, 'visual')
    origin = SubElement(visual, 'origin', rpy="0.0 0.0 0.0", xyz="0.0 0.0 0.0")
    geometry = SubElement(visual, 'geometry')
    mesh = SubElement(geometry, 'mesh', filename=filename, scale="1 1 1")
    collision = SubElement(element, 'collision')
    collision_origin = SubElement(collision, 'origin', rpy="0.0 0.0 0.0", xyz="0.0 0.0 0.0")
    collision_geometry = SubElement(collision, 'geometry')
    if args.coacd:
        collision_mesh = SubElement(collision_geometry, 'mesh', filename=filename.replace(".obj", "_coll.obj"), scale="1 1 1")
    else:
        collision_mesh = SubElement(collision_geometry, 'mesh', filename=filename, scale="1 1 1")
    link_list.append(element)
    root.extend(link_list)
    xml_string = xml.dom.minidom.parseString(tostring(root))
    xml_pretty_str = xml_string.toprettyxml()
    tree = ET.ElementTree(root)
    output_path = os.path.join(os.path.dirname(input_path), "object.urdf")
    with open(output_path, "w") as f:
        f.write(xml_pretty_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate URDF for rigid object')
    parser.add_argument('--input', type=str,
                        help='Input OBJ File')
    parser.add_argument("--coacd", action="store_true")
    args = parser.parse_args()
    export_mesh_urdf(args.input)
