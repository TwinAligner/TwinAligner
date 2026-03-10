if grep -q "20.04" /etc/os-release; then
    # Installation method for Ubuntu 20.04
    curl -L -f -o /tmp/08evux.zip --create-dirs --retry 3 --retry-delay 2 https://files.catbox.moe/08evux.zip
    unzip -o /tmp/08evux.zip -d /tmp
    sudo dpkg -i /tmp/XRoboToolkit-PC-Service_1.0.0.0_ubuntu20.04_amd64.deb
elif grep -q "22.04" /etc/os-release; then
    # Installation method for Ubuntu 22.04
    curl -L -f -o /tmp/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb --create-dirs --retry 3 --retry-delay 2 https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
    sudo dpkg -i /tmp/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
else
    echo "Unsupported Ubuntu version. Only Ubuntu 20.04 and 22.04 are supported."
    exit 1
fi

mkdir -p simulation/XRoboToolkit-PC-Service-Pybind/lib/
cp /opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so simulation/XRoboToolkit-PC-Service-Pybind/lib/
cd simulation/XRoboToolkit-PC-Service-Pybind
uv pip install -e .