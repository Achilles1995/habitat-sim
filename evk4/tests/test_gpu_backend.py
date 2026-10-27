"""
Cross-check the Warp (GPU) backend against the C++ plugin.  Runs on CUDA if available,
otherwise on Warp's CPU device (slow, small resolution).

  1. radiometry: photoreceptor ln(I) images agree (textured meshes, primitives, shadows)
  2. light-step event counts agree with theory in both backends
  3. shot-noise rates agree and are time-step invariant
  4. optical-flow labels agree on a spinning textured mesh
Run:  python tests/test_gpu_backend.py [--device cuda:0]
"""
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import evk4_mujoco as evk                       # noqa: E402
from evk4_mujoco.gpu import EVK4Warp            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.join(HERE, "mesh_scene.xml")
CAL = os.path.join(HERE, "calib_scene.xml")
DEV = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else None
RES = []


def check(name, ok, msg):
    RES.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)


def pair(xml, omega=None, **cfg):
    mc = evk.load_model(xml, **cfg)
    mg = evk.load_model(xml, backend="warp", **cfg)
    out = []
    for m, gpu in ((mc, False), (mg, True)):
        d = mujoco.MjData(m)
        if omega is not None:
            d.qvel[m.jnt_dofadr[m.joint("qspin").id]] = omega
        mujoco.mj_forward(m, d)
        s = EVK4Warp(m, "evk4", device=DEV) if gpu else evk.EVK4(m)
        if gpu:
            s.update(d)
        out.append((m, d, s, gpu))
    return out


def step(m, d, s, gpu, n):
    evs = []
    for _ in range(n):
        mujoco.mj_step(m, d)
        if gpu:
            s.update(d)
            evs.append(s.labeled_events().copy())
        else:
            evs.append(s.labeled_events(d))
    return np.concatenate(evs)


def main():
    t0 = time.time()
    (mc, dc, sc, _), (mg, dg, sg, _) = pair(MESH, resolution="320 180", focal_length_mm=1.45)
    Ic, Ig = sc.intensity(dc), sg.intensity()
    diff = np.abs(Ic - Ig)
    check("1 radiometry", np.median(diff) < 1e-3 and np.percentile(diff, 99) < 0.05,
          f"ln(I) |CPU-GPU|: median {np.median(diff):.2e}, 99th pct {np.percentile(diff, 99):.3f}, "
          f"max {diff.max():.3f} (edge pixels)")

    q = dict(shot_noise=0, leak_rate_hz=0, hot_pixel_fraction=0, contrast_sigma=0, latency_jitter_us=0,
             refractory_us=0, lux_per_unit=500, resolution="80 45")
    counts = []
    for gpu in (False, True):
        m = evk.load_model(CAL, backend="warp" if gpu else "cpu", **q)
        m.opt.timestep = 1e-5
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        s = EVK4Warp(m, device=DEV) if gpu else evk.EVK4(m)
        if gpu:
            s.update(d)
        step(m, d, s, gpu, 100)
        m.light_diffuse[0] *= 2.0
        ev = step(m, d, s, gpu, 600)
        counts.append(len(ev) / 3600)
    check("2 light step", abs(counts[0] - counts[1]) < 0.05,
          f"ON ev/px after x2 step: CPU {counts[0]:.3f}, GPU {counts[1]:.3f}")

    rates = []
    for gpu in (False, True):
        for dt in (1e-3, 2e-4):
            m = evk.load_model(CAL, backend="warp" if gpu else "cpu", lux_per_unit=10, leak_rate_hz=0,
                               hot_pixel_fraction=0, resolution="80 45")
            m.opt.timestep = dt
            d = mujoco.MjData(m); mujoco.mj_forward(m, d)
            s = EVK4Warp(m, device=DEV) if gpu else evk.EVK4(m)
            if gpu:
                s.update(d)
            step(m, d, s, gpu, int(0.01 / dt))
            ev = step(m, d, s, gpu, int(0.2 / dt))
            rates.append(len(ev) / 3600 / 0.2)
    ok = max(rates) / min(rates) < 1.2
    check("3 shot noise", ok, "10 lux Hz/px  CPU dt1ms %.2f, dt200us %.2f | GPU dt1ms %.2f, dt200us %.2f" % tuple(rates))

    (mc, dc, sc, _), (mg, dg, sg, _) = pair(MESH, omega=6.0, resolution="320 180", focal_length_mm=1.45)
    ec = step(mc, dc, sc, False, 15); eg = step(mg, dg, sg, True, 15)
    gq = mc.geom("quad").id
    fc = np.median(np.hypot(ec["flow_x"], ec["flow_y"])[ec["geom"] == gq])
    fg = np.median(np.hypot(eg["flow_x"], eg["flow_y"])[eg["geom"] == gq])
    nc, ng = len(ec), len(eg)
    check("4 events + flow labels", abs(fc - fg) / fc < 0.03 and abs(nc - ng) / nc < 0.1,
          f"spinning quad: events CPU {nc}, GPU {ng}; median |flow| CPU {fc:.1f}, GPU {fg:.1f} px/s")
    # 5. dirty-tile skipping on the GPU gives identical events
    res = []
    for skip in (1, 0):
        mg = evk.load_model(MESH, backend="warp", resolution="320 180", focal_length_mm=1.45, skip_static_tiles=skip)
        dg = mujoco.MjData(mg); dg.qvel[mg.jnt_dofadr[mg.joint("qspin").id]] = 6.0
        mujoco.mj_forward(mg, dg); s = EVK4Warp(mg, device=DEV); s.update(dg)
        res.append(step(mg, dg, s, True, 15))
    same = len(res[0]) == len(res[1]) and np.array_equal(res[0][["x", "y", "p", "t"]], res[1][["x", "y", "p", "t"]])
    check("5 GPU dirty tiles", same, f"events with skipping {len(res[0])}, without {len(res[1])}, identical={same}")
    # 6. run-time appearance edits (geom_rgba) reach the GPU renderer, as in the C++ plugin
    (mc, dc, sc, _), (mg, dg, sg, _) = pair(MESH, resolution="320 180", focal_length_mm=1.45, shot_noise=0)
    step(mc, dc, sc, False, 3); step(mg, dg, sg, True, 3)
    gq = mc.geom("quad").id
    for m in (mc, mg):
        m.geom_rgba[gq, :3] = 0.25 * m.geom_rgba[gq, :3] + 0.01
    ec = step(mc, dc, sc, False, 5); eg = step(mg, dg, sg, True, 5)
    diff = np.abs(sc.intensity(dc) - sg.intensity())
    nc, ng = len(ec), len(eg)
    check("6 run-time colour edit", nc > 100 and abs(nc - ng) / nc < 0.1 and np.median(diff) < 1e-3,
          f"OFF burst after darkening a geom: CPU {nc}, GPU {ng} events; median |dlnI| {np.median(diff):.1e}")
    print(f"device {sg.device}, {time.time() - t0:.0f} s")
    print("ALL PASSED" if all(RES) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
