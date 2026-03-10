join_path() {
    local path="$1"
    shift
    for element in "$@"; do
        path="${path%/}/$element"
    done
    echo "$path"
}
# Options and their descriptions
declare -A options
options=(
    ["-i --input_dir"]="input."
    ["-h --help"]="Show help information."
    ["-t --timestamp"]="timestamp."
)

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    for option in "${!options[@]}"; do
        echo "  $option ${options[$option]}"
    done
}

steps=12

# Parse parameters
while [[ $# -gt 0 ]]; do
    case $1 in
        --input | -i)
            input_dir="$2"
            shift 2
            ;;
        --output-cache-dir | -o)
            output_cache_dir="$2"
            shift 2
            ;;
        --help | -h)
            show_help
            exit 0
            ;;
        --timestamp | -t)
            timestamp="$2"
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

if [[ -z "$output_cache_dir" ]]; then
    output_cache_dir=outputs
fi  

## Auto GPU selection
export CUDA_VISIBLE_DEVICES=$(
    nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | 
    sort -nr | 
    head -n 1 | 
    awk -F ',' '{print $2}' |
    tr -d '[:space:]'
)

output_dir=$input_dir
PROJECT_ROOT=$(pwd)  

1_segment_robot() {
   python preprocessing/sam3_toolkit.py \
    --output_dir $input_dir \
    --image_dir_name rgb \
    --mask_dir_name masks_robot \
    --masked_rgb_dir_name images_robot \
    --bg_color black \
    --segment_prompt robot \
    --interactive_mask
}

2_foundation_pose() {
    python simulation/FoundationPose++/src/obj_pose_track.py \
    --rgb_seq_path $input_dir/rgb \
    --depth_seq_path $input_dir/depth \
    --mesh_path $output_cache_dir/background-$timestamp/point_cloud/iteration_30000/robot_gs_tsdf_fusion_post_abs_reduced.obj \
    --mask_seq_path $input_dir/masks_robot \
    --pose_output_path $output_dir/pose.npy \
    --mask_visualization_path $output_dir/mask_visualization \
    --bbox_visualization_path $output_dir/bbox_visualization \
    --pose_visualization_path $output_dir/pose_visualization \
    --cam_K_txt $input_dir/cam_K.txt \
    --activate_2d_tracker \
    --apply_scale 1 \
    --est_refine_iter 10 \
    --track_refine_iter 3 \
    --debug 2 \
    --debug_dir $input_dir/debug
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
            1_segment_robot
            ;;
        2)
            2_foundation_pose
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


    

