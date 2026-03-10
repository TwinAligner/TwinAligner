#!/bin/bash
# Install ROS1 Noetic: Use official source for 20.04, Autolabor source for 22.04, and exit with error for other versions
set -e

# Check if the OS is Ubuntu and get its version
if [ ! -f /etc/os-release ]; then
    echo "Error: Cannot detect system type, /etc/os-release not found"
    exit 1
fi

source /etc/os-release
if [ "$ID" != "ubuntu" ]; then
    echo "Error: Current system is $ID, this script only supports Ubuntu"
    exit 1
fi

VERSION_ID="${VERSION_ID:-}"
case "$VERSION_ID" in
    20.04)
        echo "Detected Ubuntu 20.04, installing ROS Noetic from the official source..."
        # Set sources.list
        sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
        # Add the key
        sudo apt install -y curl
        curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
        sudo apt update
        sudo apt install -y ros-noetic-desktop-full
        # Optionally: initialize rosdep
        if ! command -v rosdep &> /dev/null; then
            sudo apt install -y python3-rosdep
            sudo rosdep init 2>/dev/null || true
            rosdep update
        fi
        echo "ROS Noetic (official source) installation completed. Please run: source /opt/ros/noetic/setup.bash"
        ;;
    22.04)
        echo "Detected Ubuntu 22.04, installing ROS Noetic from Autolabor source..."
        echo "deb [trusted=yes arch=amd64] http://deb.repo.autolabor.com.cn jammy main" | sudo tee /etc/apt/sources.list.d/autolabor.list
        sudo apt update
        sudo apt install -y ros-noetic-autolabor
        echo "ROS Noetic (Autolabor) installation completed. Please run: source /opt/ros/noetic/setup.bash"
        ;;
    *)
        echo "Error: Current Ubuntu version is $VERSION_ID, this script only supports 20.04 and 22.04"
        exit 1
        ;;
esac