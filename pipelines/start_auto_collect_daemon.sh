#!/bin/bash

set -e

# Get script dir and project root (parent of pipelines)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use Ubuntu GNOME desktop display for GUI (e.g. viewer)
if [ -z "${DISPLAY}" ]; then
    export DISPLAY=:0
fi

# Options and their descriptions
declare -A options
options=(
    ["-h --help"]="Show help information."
    ["--ensure-roscore"]="Start roscore in tmux if not running."
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

# Parse parameters
while [[ $# -gt 0 ]]; do
    case $1 in
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

# ensure roscore is running before starting auto collect daemon
source $SCRIPT_DIR/start_roscore.sh

SESSION_NAME="auto_collect"
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT"
tmux send-keys -t "${SESSION_NAME}:0.0" "source /opt/ros/noetic/setup.bash && conda activate twinaligner && python simulation/data_collection/auto_collect_pick_and_place.py --show_viewer" Enter
tmux attach -t "$SESSION_NAME"
