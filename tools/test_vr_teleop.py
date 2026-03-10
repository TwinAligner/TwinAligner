import os
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["no_proxy"] = "localhost,127.0.0.1"
import xrobotoolkit_sdk as xrt
import time

xrt.init()
while True:
    left_pose = xrt.get_left_controller_pose()
    right_pose = xrt.get_right_controller_pose()
    headset_pose = xrt.get_headset_pose()

    print(f"Left Controller Pose: {left_pose}")
    print(f"Right Controller Pose: {right_pose}")
    print(f"Headset Pose: {headset_pose}")
    time.sleep(0.1)
xrt.close()