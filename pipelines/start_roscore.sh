#!/bin/bash
# Start roscore in a tmux session if the ROS master is not already running.

_check_roscore() {
    timeout 3 bash -c 'source /opt/ros/noetic/setup.bash && rosnode list' &>/dev/null
    return $?
}

if ! _check_roscore; then
    echo "roscore not detected, starting in tmux..."
    # Kill existing tmux session named roscore if any (e.g. leftover from previous run)
    tmux kill-session -t roscore 2>/dev/null
    # Start roscore in a new detached tmux session named "roscore"
    tmux new-session -d -s roscore 'source /opt/ros/noetic/setup.bash && roscore'
    for _ in {1..15}; do
        sleep 1
        if _check_roscore; then
            echo "roscore started in tmux session 'roscore'. Attach with: tmux attach -t roscore"
            exit 0
        fi
    done
    tmux kill-session -t roscore 2>/dev/null
    echo "Error: roscore failed to start within timeout" >&2
    exit 1
else
    echo "roscore is already running."
fi
