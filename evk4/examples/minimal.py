"""Minimal API example: step the UR5e scene and pull EVK4 events (Metavision EventCD layout)."""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import evk4_mujoco as evk                       # noqa: E402
from evk4_mujoco import ur5e_demo as demo       # noqa: E402

scene = os.path.join(os.path.dirname(__file__), "..", "scenes", "scene_ur5e_evk4.xml")
backend = sys.argv[1] if len(sys.argv) > 1 else "cpu"   # "warp" -> GPU
m = evk.load_model(scene, binning=2, backend=backend)    # drop binning for full 1280x720
d = mujoco.MjData(m)
demo.reset(m, d)
cam = evk.Sensor(m, "evk4", backend=backend)
cam.update(d)

chunks = []
for _ in range(200):                            # 200 ms
    d.ctrl[:6] = demo.joint_targets(d.time)
    mujoco.mj_step(m, d)
    cam.update(d)
    chunks.append(cam.labeled_events())         # time-ordered events + ground truth
ev = np.concatenate(chunks)
print(cam.stats())
print(f"{len(ev)} events, ON fraction {ev['p'].mean():.2f}, t = {ev['t'][0]} .. {ev['t'][-1]} us")
np.save("events.npy", ev)
from evk4_mujoco.io import write_evt3
W, H = cam.geometry()
write_evt3("events.raw", ev[["x", "y", "p", "t"]], W, H)       # open in Metavision Studio
