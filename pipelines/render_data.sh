#!/bin/bash

set -e

# Get script dir and project root (parent of pipelines)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Options and their descriptions
declare -A options
options=(
    ["--record_dir DIR"]="Record directory. Default: datasets/records/carrot_plate"
    ["--cfg_path PATH"]="Config file path. Default: simulation/configs/carrot_plate.yaml"
    ["-h --help"]="Show help information."
)

# Function: Show help information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    for option in "${!options[@]}"; do
        echo "  $option"
        echo "      ${options[$option]}"
    done
}

record_dir="datasets/records/carrot_plate"
cfg_path="simulation/configs/carrot_plate.yaml"

# Parse parameters
while [[ $# -gt 0 ]]; do
    case $1 in
        --record_dir)
            record_dir="$2"
            shift 2
            ;;
        --cfg_path)
            cfg_path="$2"
            shift 2
            ;;
        --help | -h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"
python simulation/data_collection/render_pick_and_place.py --record_dir "$record_dir" --cfg_path "$cfg_path"
