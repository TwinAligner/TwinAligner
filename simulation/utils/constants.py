FR3_DEFAULT_CFG = {
    "fr3_joint1": 0.03995192,
    "fr3_joint2": -0.00592799,#
    "fr3_joint3": -0.18448183,
    "fr3_joint4": -2.21950033,
    "fr3_joint5": -0.00492253,
    "fr3_joint6": 2.22304146,
    "fr3_joint7": 0.69084185,
    "fr3_finger_joint1": 0.04,
    "fr3_finger_joint2": 0.04,
}
XML_DEFAULT_CFG = {
    "joint1": 0.03995192,
    "joint2": -0.00592799,#
    "joint3": -0.18448183,
    "joint4": -2.21950033,
    "joint5": -0.00492253,
    "joint6": 2.22304146,
    "joint7": 0.69084185,
    "finger_joint1": 0.04,
    "finger_joint2": 0.04,
}

BEST_PARAMS = {
    "kp": [
        2424.657254246728,
        16976.159595959594,
        12101.932123237386,
        10031.517126148701,
        1238.0951824004453,
        69.25030758714965,
        396.801903749272
    ],
    "kv": [
        2003.6817042606517,
        3271.0,
        1537.591478696742,
        1456.187969924812,
        460.44862155388466,
        13.506265664160395,
        80.49874686716791
    ],
}

CAMERA_INTRINSICS_DIR = "assets/realsense/cam_intr.json"
CAMERA_EXTRINSICS_DIR = "assets/realsense/cam_extr_init.txt"

JOINT_NAMES = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]

DEFAULT_JOINT_ANGLES = [XML_DEFAULT_CFG[joint_name] for joint_name in JOINT_NAMES]
REALSENSE_IMAGE_HEIGHT = 720
REALSENSE_IMAGE_WIDTH = 1280