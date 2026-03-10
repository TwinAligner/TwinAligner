'''
Credits: https://github.com/real-dex-suite/REAL-ROBO
'''
from multiprocessing import Process
from termcolor import cprint
from simulation.utils.teleoperation.components.pico import PICOArmTeleOp
from simulation.utils.teleoperation.components.keyboard import KBArmTeleop

def notify_process_start(notification_statement):
    cprint("***************************************************************", "green")
    cprint("     {}".format(notification_statement), "green")
    cprint("***************************************************************", "green")

def pico_teleop():
    notify_process_start("Starting Teleoperation Process")
    teleop = PICOArmTeleOp(simulator="genesis", 
                              gripper="panda",
                              arm_type="franka",
                              gripper_init_state="open",
                              lock_rotation=["pitch", "roll"],
                              lock_z=False)
    teleop.move()

def kb_teleop():
    notify_process_start("Starting Teleoperation Process")
    teleop = KBArmTeleop(simulator="genesis", 
                         gripper="panda",
                         arm_type="franka",
                         gripper_init_state="open",
                         lock_rotation=["pitch", "roll"],
                         lock_z=False)
    teleop.move()

def get_teleop_process(tracker_type="pico"):
    if tracker_type == 'pico': # PICO VR
        teleop_process = Process(target = pico_teleop, args = ())
    elif tracker_type == 'keyboard': # Keyboard
        teleop_process = Process(target = kb_teleop, args = (), daemon=False)
    else:
        raise NotImplementedError(f"Unknown tracker {tracker_type}")
    return teleop_process