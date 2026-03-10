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

1_segment_table() {
   python preprocessing/sam3_toolkit.py \
    --output_dir $input_dir \
    --image_dir_name rgb \
    --mask_dir_name masks_table \
    --masked_rgb_dir_name images_table \
    --bg_color black \
    --segment_prompt table \
    --interactive_mask
}

2_pso_optimization() {
    python simulation/view_alignment/pso_viewpoint_refiner.py \
    --output_dir $output_dir --background_timestamp $timestamp
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
            1_segment_table
            ;;
        2)
            2_pso_optimization
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