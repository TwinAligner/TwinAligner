#!/bin/bash

set -e

# Options and their descriptions
declare -A options
options=(
    ["-i --input"]="Scanned video folder."
    ["-p --prompt"]="Prompt of semantic segmentation."
    ["-t --timestamp"]="Timestamp of starting the pipeline. Keep it empty for a new reconstruction."
    ["-h --help"]="Show help information."
    ["-x --proxy"]="Network proxy for everything."
    ["-s --steps"]="Steps to execute. 1 - measure the background by scanning, 2 - align the scales. 12 as default."
    ["-as --aruco_size"]="aruco size in the real world. 0.035 as default."
    ["-tf --target_face_num"]="target face num of the asset. 700 as default."
)

generate_timestamp() {
    echo $(date +"%Y%m%d_%H%M%S_%3N")
}

# Function: Show help information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    for option in "${!options[@]}"; do
        echo "  $option ${options[$option]}"
    done
}

steps=12
aruco_size=0.035 # meters
target_face_num=700 # target faces of object
# Parse parameters
while [[ $# -gt 0 ]]; do
    case $1 in
        --input | -i)
            data="$2"
            shift 2
            ;;
        --prompt | -p)
            prompt="$2"
            shift 2
            ;;
        --timestamp | -t)
            timestamp="$2"
            shift 2
            ;;
        --help | -h)
            show_help
            exit 0
            ;;
        --proxy | -x)
            http_proxy="$2"
            https_proxy="$2"
            shift 2
            ;;
        --steps | -s)
            steps="$2"
            shift 2
            ;;
        --aruco_size | -as)
            aruco_size="$2"
            shift 2
            ;;
        --target_face_num | -tf)
            target_face_num="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

dataset_cache_dir=datasets
output_cache_dir=outputs
tolerance=0.13
neus_method_fast="neus-facto-monodepth-fast-bg"

## Auto GPU selection
CUDA_VISIBLE_DEVICES=$(
    nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | 
    sort -nr | 
    head -n 1 | 
    awk -F ',' '{print $2}' |
    tr -d '[:space:]'
)

LD_LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:$LD_LIBRARY_PATH

if [[ -z "$timestamp" ]]; then
    timestamp=$(generate_timestamp)
fi

1_measure_recon() {
    ss-train $neus_method_fast --vis tensorboard --output-dir $output_cache_dir/$prompt-$timestamp-measure --specialize-output-dir True \
    nerfstudio-data --data $dataset_cache_dir/nerfstudio-data/$prompt-$timestamp-1 --use-mask False --use-ori-image True --use-mono-depth True
    ss-extract-mesh --load-config $output_cache_dir/$prompt-$timestamp-measure/config.yml --bounding-box-min -0.5 -0.5 -0.5 --bounding-box-max 0.5 0.5 0.5 --simplify-mesh True
    ss-render-mesh --load-config $output_cache_dir/$prompt-$timestamp-measure/config.yml --traj original --first-frame-only --not-use-mask --rendered-output-names ori_world_pointmap
}

2_align() {
    python editing/alignment/align_object_aruco.py --data $dataset_cache_dir/nerfstudio-data/$prompt-$timestamp-1 --output_path $output_cache_dir/$prompt-$timestamp --aruco_size $aruco_size
    python editing/alignment/align_object_3dgs.py --input $output_cache_dir/$prompt-$timestamp/object_3dgs.ply
    python editing/convertion/ply2obj.py --input $output_cache_dir/$prompt-$timestamp/mesh_w_vertex_color_abs.ply
    python editing/convertion/rigid2urdf.py --input $output_cache_dir/$prompt-$timestamp/mesh_w_vertex_color_abs.obj
    python editing/convertion/reduce_faces.py --input_file $output_cache_dir/$prompt-$timestamp/mesh_w_vertex_color_abs.obj  --target_faces $target_face_num
    python editing/convertion/reduce_faces_urdf.py --input_file $output_cache_dir/$prompt-$timestamp/object.urdf
}

# Check if input only contains digits 1-6
if [[ ! "$steps" =~ ^[1-2]+$ ]]; then
    echo "ERROR: Input '$steps' contains invalid characters or step numbers. Please only enter digits between 1-5. Exiting."
    exit 1
fi

for STEP in $(echo "$steps" | grep -o .); do
    echo "" # Print an empty line to separate output for each step

    case $STEP in
        1)
            1_measure_recon
            ;;
        2)
            2_align
            ;;
        *)
            # Although input validation should prevent this, keeping it for robustness
            echo "WARNING: Found unknown step number '$STEP'. Skipping."
            ;;
    esac

    # Check the exit status of the previous step (function)
    if [ $? -ne 0 ]; then
        echo "!!! ERROR: Step $STEP failed to execute. Terminating script !!!"
        exit 1
    fi
done