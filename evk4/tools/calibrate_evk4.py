"""Fit simulator parameters to real EVK4 recordings.  Usage: python tools/calibrate_evk4.py experiments.json"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from evk4_mujoco.calibration import calibrate, to_mjcf   # noqa: E402

if __name__ == "__main__":
    p = calibrate(sys.argv[1])
    print("\nFitted parameters - paste into the <plugin> of your MJCF:\n")
    print(to_mjcf(p))
