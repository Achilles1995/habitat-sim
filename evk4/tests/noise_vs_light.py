"""Background-activity (shot-noise) event rate vs. illuminance on a static gray target."""
import sys, os, numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import evk4_mujoco as evk
here = os.path.dirname(os.path.abspath(__file__))

def rate(lux, T=0.4, **cfg):
    m = evk.load_model(os.path.join(here, "calib_scene.xml"), lux_per_unit=lux, **cfg)
    d = mujoco.MjData(m); cam = evk.EVK4(m)
    mujoco.mj_forward(m, d)
    n = 0; steps = int(T / m.opt.timestep)
    for k in range(steps):
        mujoco.mj_step(m, d)
        if k >= 50: n += len(cam.events(d))   # skip settling
    W, H = cam.geometry(d)
    return n / (W * H) / ((steps - 50) * m.opt.timestep)

if __name__ == "__main__":
    gains = [float(g) for g in sys.argv[1:]] or [1.0]
    for g in gains:
        row = [rate(l, noise_gain=g) for l in (1, 10, 100, 1000, 10000)]
        print(f"noise_gain={g:<6}  Hz/pixel @ 1,10,100,1k,10k lux:", " ".join(f"{r:8.3f}" for r in row))
