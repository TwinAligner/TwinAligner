#!/bin/bash

set -e

# Options and their descriptions
declare -A options
options=(
    ["-i --input"]="Scanned video folder."
    ["-t --timestamp"]="Timestamp of starting the pipeline. Keep it empty for a new reconstruction."
    ["-h --help"]="Show help information."
    ["-x --proxy"]="Network proxy for everything."
    ["-c --dataset-cache-dir"]="Directory to cache the processed datasets."
    ["-o --output-cache-dir"]="Directory to store the outputs."
)

## Auto GPU selection
export CUDA_VISIBLE_DEVICES=$(
    nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | 
    sort -nr | 
    head -n 1 | 
    awk -F ',' '{print $2}' |
    tr -d '[:space:]'
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

steps=1234
# Parse parameters
while [[ $# -gt 0 ]]; do
    case $1 in
        --input | -i)
            data="$2"
            shift 2
            ;;
        --dataset-cache-dir | -c)
            dataset_cache_dir="$2"
            shift 2
            ;;
        --output-cache-dir | -o)
            output_cache_dir="$2"
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
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [[ -z "$dataset_cache_dir" ]]; then
    dataset_cache_dir=datasets
fi
if [[ -z "$output_cache_dir" ]]; then
    output_cache_dir=outputs
fi  

# If timestamp is not set, generate a random one
if [[ -z "$timestamp" ]]; then
    timestamp=$(generate_timestamp)
fi

1_align_background() {
    python editing/alignment/align_background.py \
    --gs_points_path $output_cache_dir/background-$timestamp/point_cloud/iteration_30000/point_cloud.ply \
    --colmap_path $dataset_cache_dir/nerfstudio-data/background-$timestamp \
    --expand_factor 0.2 \
    --robot_expand_factor 0.05 \
    --save_txt
}

2_render_pgsr() {
    output_dir=`realpath $output_cache_dir/background-$timestamp`
    pushd reconstruction/pgsr
    echo "===== Extracting robot mesh ... ====="
    python render.py -m $output_dir --load_ply_path $output_dir/point_cloud/iteration_30000/robot_gs.ply --voxel_size 0.01
    echo "===== Extracting scene mesh ... ====="
    python render.py -m $output_dir --load_ply_path $output_dir/point_cloud/iteration_30000/scene_gs.ply --voxel_size 0.01 --num_cluster 2
    popd
}

3_transform_franka_gs_geometry() {
    output_dir=`realpath $output_cache_dir/background-$timestamp`
    python editing/alignment/transform_geometry.py \
    -i $output_dir/point_cloud/iteration_30000/robot_gs_tsdf_fusion_post.ply \
    -o $output_dir/point_cloud/iteration_30000/robot_gs_tsdf_fusion_post_abs.ply \
    -m $output_dir/point_cloud/iteration_30000/background_gs2cad.txt \
    --is-mesh

    python editing/alignment/transform_gs.py \
    -i $output_dir/point_cloud/iteration_30000/robot_gs.ply \
    -o $output_dir/point_cloud/iteration_30000/robot_gs_abs.ply \
    -m $output_dir/point_cloud/iteration_30000/background_gs2cad.txt

    python editing/convertion/ply2obj.py --input $output_dir/point_cloud/iteration_30000/robot_gs_tsdf_fusion_post_abs.ply

    python editing/convertion/reduce_faces.py \
        --input_file $output_dir/point_cloud/iteration_30000/robot_gs_tsdf_fusion_post_abs.obj --target_faces 30000
}

4_transform_scene_gs_geometry() {
    output_dir=`realpath $output_cache_dir/background-$timestamp`
    python editing/alignment/transform_geometry.py \
    -i $output_dir/point_cloud/iteration_30000/scene_gs_tsdf_fusion_post.ply \
    -o $output_dir/point_cloud/iteration_30000/scene_gs_tsdf_fusion_post_abs.ply \
    -m $output_dir/point_cloud/iteration_30000/background_gs2cad.txt \
    --is-mesh

    python editing/alignment/transform_gs.py \
    -i $output_dir/point_cloud/iteration_30000/scene_gs.ply \
    -o $output_dir/point_cloud/iteration_30000/scene_gs_abs.ply \
    -m $output_dir/point_cloud/iteration_30000/background_gs2cad.txt

    python editing/convertion/ply2obj.py --input $output_dir/point_cloud/iteration_30000/scene_gs_tsdf_fusion_post_abs.ply

    python editing/convertion/reduce_faces.py \
        --input_file $output_dir/point_cloud/iteration_30000/scene_gs_tsdf_fusion_post_abs.obj
}

# Check if input only contains digits 1-6
if [[ ! "$steps" =~ ^[1-5]+$ ]]; then
    echo "ERROR: Input '$steps' contains invalid characters or step numbers. Please only enter digits between 1-5. Exiting."
    exit 1
fi

for STEP in $(echo "$steps" | grep -o .); do
    echo "" # Print an empty line to separate output for each step

    case $STEP in
        1)
            1_align_background
            ;;
        2)
            2_render_pgsr
            ;;
        3)
            3_transform_franka_gs_geometry
            ;;
        4)
            4_transform_scene_gs_geometry
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