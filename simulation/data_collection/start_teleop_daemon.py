import time
import os
import sys
sys.path.insert(0, os.getcwd())
from simulation.utils.teleoperation.processes import get_teleop_process
import multiprocessing
import argparse

def main(args):    
    multiprocessing.set_start_method('spawn')
    
    # Obtaining all the robot streams
    teleop_process = get_teleop_process(args.tracker_type)
    
    # Starting all the processes
    if teleop_process is not None:
        # Teleop process
        time.sleep(2)
        teleop_process.start()

    if teleop_process is not None:
        teleop_process.join()

    # Kill all the sub processes
    teleop_process.kill()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker_type", type=str, default="keyboard")
    args = parser.parse_args()
    main(args)