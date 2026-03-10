#!/bin/bash

set -e

# Get script dir and project root (parent of pipelines)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use Ubuntu GNOME desktop display for GUI
if [ -z "${DISPLAY}" ]; then
    export DISPLAY=:0
fi

# Options and their descriptions
declare -A options
options=(
    ["-t --tracker_type"]="Tracker type: keyboard or pico. Default: keyboard."
    ["-h --help"]="Show help information."
    ["--ensure-roscore"]="Start roscore in tmux if not running (for pico/keyboard teleop)."
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

tracker_type=keyboard

# Parse parameters
while [[ $# -gt 0 ]]; do
    case $1 in
        --tracker_type | -t)
            tracker_type="$2"
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

# ensure roscore is running before starting teleop daemon
source $SCRIPT_DIR/start_roscore.sh

if [ "$tracker_type" = "pico" ]; then
    /opt/apps/roboticsservice/runService.sh
fi

SESSION_NAME="teleop"
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT"
tmux send-keys -t "${SESSION_NAME}:0.0" "source /opt/ros/noetic/setup.bash && conda activate twinaligner && tracker_type=$tracker_type python simulation/data_collection/teleop_pick_and_place.py" Enter
sleep 10
tmux split-window -h -t "$SESSION_NAME" -c "$PROJECT_ROOT"
tmux send-keys -t "${SESSION_NAME}:0.1" "source /opt/ros/noetic/setup.bash && conda activate twinaligner && http_proxy= https_proxy= python simulation/data_collection/start_teleop_daemon.py --tracker_type $tracker_type" Enter
tmux select-layout -t "$SESSION_NAME" even-horizontal
tmux attach -t "$SESSION_NAME"
