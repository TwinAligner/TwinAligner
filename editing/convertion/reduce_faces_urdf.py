def clean_urdf(urdf_path):
    with open(urdf_path, "r") as f:
        urdf_filecontent = f.read()
    urdf_filecontent = urdf_filecontent.replace(".obj", "_reduced.obj")
    with open(urdf_path, "w") as f:
        f.write(urdf_filecontent)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="outputs/carrot-asset/object.urdf")
    args = parser.parse_args()
    clean_urdf(args.input_file)