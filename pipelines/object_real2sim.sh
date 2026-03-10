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
    ["-s --steps"]="Steps to execute. 1 - preprocessing, 2 - prescanning, 3 - merging, 4 - sdf, 5 - gs. 12345 as default."
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

steps=12345

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
        --mono-depth-strong | -monostrong)
            monostrong=true
            shift 1
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

dataset_cache_dir=datasets
output_cache_dir=outputs
tolerance=0.13
neus_method="neus-facto-monodepth-camopt"
if [ "$monostrong" = true ]; then
    neus_method="neus-facto-monodepth-strong-camopt"
fi
neus_method_fast="neus-facto-monodepth-fast"

## Auto GPU selection
export CUDA_VISIBLE_DEVICES=$(
    nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | 
    sort -nr | 
    head -n 1 | 
    awk -F ',' '{print $2}' |
    tr -d '[:space:]'
)

export LD_LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:$LD_LIBRARY_PATH

if [[ -z "$timestamp" ]]; then
    timestamp=$(generate_timestamp)
fi

1_preprocess() {
    scan_cnt=1
    for file in "$data"/*; do
        command="python preprocessing/preprocess.py --segment-prompt $prompt --data $file --output-dir $dataset_cache_dir/nerfstudio-data/$prompt-$timestamp-$scan_cnt"
        eval $command
        ((scan_cnt++)) 
    done
}

2_prescan() {
    FILE_COUNT=0
    FILES_IN_DATA=("$data"/*) 
    if [ -e "${FILES_IN_DATA[0]}" ]; then
        FILE_COUNT=${#FILES_IN_DATA[@]}
    else
        FILE_COUNT=0
    fi
    if [ $FILE_COUNT -gt 1 ]; then
        echo "{$FILE_COUNT} scans found. Prescanning.."

        scan_cnt=1
        for file in "$data"/*; do
            echo "--- Processing Scan #$scan_cnt: $file ---"
            ss-train $neus_method_fast --vis tensorboard --output-dir $output_cache_dir/$prompt-$timestamp-$scan_cnt --specialize-output-dir True nerfstudio-data --data $dataset_cache_dir/nerfstudio-data/$prompt-$timestamp-$scan_cnt
            ss-extract-mesh --load-config $output_cache_dir/$prompt-$timestamp-$scan_cnt/config.yml --bounding-box-min -0.5 -0.5 -0.5 --bounding-box-max 0.5 0.5 0.5
            ss-texture --load-config $output_cache_dir/$prompt-$timestamp-$scan_cnt/config.yml
            ss-dump-opt-camera --load-config $output_cache_dir/$prompt-$timestamp-$scan_cnt/config.yml
            ((scan_cnt++)) 
        done
        echo "Batch processing finished successfully."

    else
        echo "Only single scan found (or less). Skip prescanning."
    fi
}

3_merge() {
    FILE_COUNT=0
    FILES_IN_DATA=("$data"/*) 
    if [ -e "${FILES_IN_DATA[0]}" ]; then
        FILE_COUNT=${#FILES_IN_DATA[@]}
    else
        FILE_COUNT=0
    fi
    if [ $FILE_COUNT -gt 1 ]; then
        echo "${FILE_COUNT} scans found. Merging."
        command="python preprocessing/merge_dataset.py --num-scans $FILE_COUNT --output-dir $dataset_cache_dir/nerfstudio-data/$prompt-$timestamp --scan-recons-dir $output_cache_dir/$prompt-$timestamp-{scan_idx}"
        eval $command
    else
        echo "Only single scan found (or less). Skip merging dataset."
    fi
}

4_sdf() {
    ss-train $neus_method --vis tensorboard --output-dir $output_cache_dir/$prompt-$timestamp --specialize-output-dir True nerfstudio-data --data $dataset_cache_dir/nerfstudio-data/$prompt-$timestamp  --auto-repose False --auto-scale-poses False
    ss-extract-mesh --load-config $output_cache_dir/$prompt-$timestamp/config.yml --bounding-box-min -0.5 -0.5 -0.5 --bounding-box-max 0.5 0.5 0.5
    ss-texture --load-config $output_cache_dir/$prompt-$timestamp/config.yml --target-num-faces 50000
    ss-dump-opt-camera --load-config $output_cache_dir/$prompt-$timestamp/config.yml
}

5_gs() {
    ss-render-mesh --load-config $output_cache_dir/$prompt-$timestamp/config.yml
    ns-train splatfacto-depth-camopt --vis tensorboard --output-dir $output_cache_dir/$prompt-$timestamp-gstrain --specialize-output-dir True nerfstudio-data --data $dataset_cache_dir/nerfstudio-data/$prompt-$timestamp --resume-sdfstudio-dir $output_cache_dir/$prompt-$timestamp
    ns-export gaussian-splat --load-config $output_cache_dir/$prompt-$timestamp-gstrain/config.yml
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
            1_preprocess
            ;;
        2)
            2_prescan
            ;;
        3)
            3_merge
            ;;
        4)
            4_sdf
            ;;
        5)
            5_gs
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