"""
Speed benchmark of the EVK4 sensor on the UR5e scene (full 1280x720 unless --binning).

    python examples/benchmark.py --backend cpu
    python examples/benchmark.py --backend warp --device cuda:0      # e.g. RTX 4070
Reports ms per 1 ms physics step (incl. adaptive sub-steps) and the real-time factor.
"""
import argparse
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import evk4_mujoco as evk                       # noqa: E402
from evk4_mujoco import ur5e_demo as demo       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="cpu", choices=["cpu", "warp"])
ap.add_argument("--device", default=None)
ap.add_argument("--binning", type=int, default=1)
ap.add_argument("--spp", type=int, default=1)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--start", type=float, default=0.0, help="start time in the scripted trajectory")
ap.add_argument("--skip-static-tiles", type=int, default=1)
ap.add_argument("--max-px-motion", type=float, default=1.0)
ap.add_argument("--timestep", type=float, default=0.0, help="override MuJoCo timestep [s]")
ap.add_argument("--shadow-hull", action="store_true", help="warp: convex-hull meshes for shadow rays")
ap.add_argument("--render", default="split", choices=["split", "fused"],
                help="warp: v2.5 split renderer (hit/surface/shadow/light passes) or the old fused kernel")
ap.add_argument("--block", type=int, default=128, help="warp: threads per block of the render kernels")
ap.add_argument("--maxreg", type=int, default=0, help="warp: register cap for the render kernels (0 = none)")
ap.add_argument("--config", nargs="*", default=[], help="extra plugin config, e.g. shadows=0 spp=4")
a = ap.parse_args()
os.environ["EVK4_RENDER"] = a.render                  # read when the GPU backend is imported/created
os.environ["EVK4_BLOCK"] = str(a.block)
os.environ["EVK4_MAXREG"] = str(a.maxreg)
scene = os.path.join(os.path.dirname(__file__), "..", "scenes", "scene_ur5e_evk4.xml")
extra = dict(kv.split("=", 1) for kv in a.config)
m = evk.load_model(scene, binning=a.binning, backend=a.backend, spp=a.spp,
                   skip_static_tiles=a.skip_static_tiles, max_px_motion=a.max_px_motion, **extra)
if a.timestep > 0:
    m.opt.timestep = a.timestep
if a.shadow_hull:
    os.environ["EVK4_SHADOW_MESH"] = "hull"
d = mujoco.MjData(m)
demo.reset(m, d)
m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)     # fast-forward without the sensor
while d.time < a.start:
    d.ctrl[:6] = demo.joint_targets(d.time); mujoco.mj_step(m, d)
m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
s = evk.Sensor(m, "evk4", backend=a.backend, device=a.device)
mujoco.mj_forward(m, d); s.update(d)
for _ in range(5):                                                 # warm-up / JIT
    d.ctrl[:6] = demo.joint_targets(d.time); mujoco.mj_step(m, d); s.update(d)
t0 = time.time(); n = 0; subs = []
for _ in range(a.steps):
    d.ctrl[:6] = demo.joint_targets(d.time); mujoco.mj_step(m, d); s.update(d)
    n += len(s.events()); subs.append(s.stats()["n_substeps"])
el = (time.time() - t0) / a.steps
W, H = s.geometry()
print(f"{a.backend} {W}x{H} spp={a.spp} skip={a.skip_static_tiles} t0={a.start} dt={m.opt.timestep*1e3:g}ms "
      f"hull={a.shadow_hull} render={a.render} block={a.block} maxreg={a.maxreg} {' '.join(a.config)}: {el*1e3:.1f} ms per {m.opt.timestep*1e3:.0f} ms step "
      f"(real-time factor {m.opt.timestep/el:.3f}), mean substeps {np.mean(subs):.2f}, "
      f"{n/(a.steps*m.opt.timestep)/1e6:.2f} Mev/s")
