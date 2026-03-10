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

steps=12
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

1_preprocess() {
    python preprocessing/preprocess.py --segment-type none --redo-video2images --image-ori-name input --data $data --output-dir $dataset_cache_dir/nerfstudio-data/background-$timestamp
    python preprocessing/preprocess_pgsr.py --data_path $dataset_cache_dir/nerfstudio-data/background-$timestamp/
    depth-pro-run -i $dataset_cache_dir/nerfstudio-data/background-$timestamp/images -o $dataset_cache_dir/nerfstudio-data/background-$timestamp/depths --skip-display
}

2_pgsr() {
    base_dir=`realpath $dataset_cache_dir/nerfstudio-data/background-$timestamp/`
    depths_dir=`realpath $base_dir/depths`
    output_dir=`realpath $output_cache_dir/background-$timestamp`
    cd reconstruction/pgsr/
    python utils/make_depth_scale.py \
    --base_dir $base_dir \
    --depths_dir $depths_dir \
    --model_type bin
    python train.py -s $base_dir -m $output_dir --max_abs_split_points 0 --opacity_cull_threshold 0.05
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
            1_preprocess
            ;;
        2)
            2_pgsr
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