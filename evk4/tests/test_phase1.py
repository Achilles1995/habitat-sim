"""
Phase-1 checks: mounting the EVK4 on any robot + pixel-aligned paired frames.

    python tests/test_phase1.py            (add --device cuda:0 to run check 4 on the GPU)

1  attach_evk4() on the UR5e reproduces the hand-written scene bit-exactly (same events)
2  a model saved with save_xml reloads and gives the same events
3  paired RGB/segmentation land on the right pixels (pinhole AND calibrated lens with distortion)
4  paired depth / segmentation == the event engine's own per-pixel labels (GPU backend)
5  per-event optical-flow labels == dense ground-truth flow of the paired frames
6  two EVK4s on one robot (stereo) work independently
7  Menagerie robots in robots/menagerie (if downloaded) get a working camera
"""
import os
import sys
import tempfile
import time

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco                                   # noqa: E402
import numpy as np                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import evk4_mujoco as evk                        # noqa: E402
from evk4_mujoco import mount, ur5e_demo as demo  # noqa: E402

SCENE = os.path.join(HERE, "..", "scenes", "scene_ur5e_evk4.xml")
DEV = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
RES = []


def check(name, ok, msg):
    RES.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)


def run_events(m, n, sensor="evk4"):
    d = mujoco.MjData(m); demo.reset(m, d)
    s = evk.EVK4(m, sensor)
    out = []
    for _ in range(n):
        d.ctrl[:6] = demo.joint_targets(d.time); mujoco.mj_step(m, d); out.append(s.events(d))
    return np.concatenate(out)


def ur5e_without_evk4():
    """The bundled UR5e scene with its hand-written EVK4 removed (MjSpec)."""
    spec = mujoco.MjSpec.from_file(SCENE)
    cfg = dict(spec.sensors[0].plugin.config)
    spec.delete(spec.sensors[0]); spec.delete(spec.body("evk4"))
    for pl in list(spec.plugins):
        if pl.name == "":
            spec.delete(pl)
    return spec, cfg


def main():
    t0 = time.time()
    evk.load_plugin()
    # 1 ---------------------------------------------------------------------------------------
    ref = evk.load_model(SCENE, binning=4)
    spec, cfg = ur5e_without_evk4()
    cfg["binning"] = "4"
    # hand-written: body (0 0.1 0) quat(-1 1 0 0) + camera 0.069 along its z  ->  optical centre
    mount.add_evk4(spec, "wrist_3_link", pos=(0, 0.169, 0), quat=(-1, 1, 0, 0), config=cfg)
    m1 = spec.compile()
    e0, e1 = run_events(ref, 120), run_events(m1, 120)
    check("1 attach == hand-written XML", len(e0) > 1000 and np.array_equal(e0, e1),
          f"{len(e0)} vs {len(e1)} events, identical={np.array_equal(e0, e1)}")
    # 2 ---------------------------------------------------------------------------------------
    fd, path = tempfile.mkstemp(suffix=".xml", dir=os.path.dirname(SCENE)); os.close(fd)
    try:
        with open(path, "w") as f:
            f.write(spec.to_xml())
        m2 = mujoco.MjModel.from_xml_path(path)
        e2 = run_events(m2, 60)
    finally:
        os.remove(path)
    e1b = run_events(m1, 60)
    common = len(np.intersect1d(e1b.view(np.void), e2.view(np.void))) / max(1, len(e1b))
    # MJCF text stores quaternions to ~7 digits, so tiny differences are expected
    check("2 save_xml round trip", len(e2) > 500 and abs(len(e2) - len(e1b)) <= 0.01 * len(e1b) and common > 0.95,
          f"{len(e1b)} vs {len(e2)} events, {100 * common:.1f}% identical (XML stores ~7 digits)")
    # 3 ---------------------------------------------------------------------------------------
    pts = [(0, 0), (0.35, 0.2), (-0.4, -0.25), (0.5, -0.3), (-0.2, 0.3)]
    xml = ('<mujoco><worldbody><light pos="0 0 3" dir="0 0 -1" directional="true"/><body name="rig" pos="0 0 1"/>'
           + "".join(f'<geom name="s{i}" type="sphere" size="0.012" pos="{x} {y} 0"/>' for i, (x, y) in enumerate(pts))
           + "</worldbody></mujoco>")
    fd, path = tempfile.mkstemp(suffix=".xml"); os.close(fd)
    open(path, "w").write(xml)
    worst = []
    try:
        for cfg3 in (dict(resolution="320 240", focal_length_mm=1.2),
                     dict(resolution="320 240", fx=250, fy=265, cx=171.3, cy=109.8,
                          distortion="-0.28 0.09 0.001 -0.0015 -0.012")):
            m3 = evk.attach_evk4(path, "rig", pos=(0.02, -0.01, 0), euler_deg=(180, 3, -4), config=cfg3)
            d3 = mujoco.MjData(m3); mujoco.mj_forward(m3, d3)
            pc = evk.PairedCamera(m3)
            fr = pc.capture(d3)
            T = fr.T_world_cam
            for i in range(len(pts)):
                g = m3.geom(f"s{i}").id
                vv, uu = np.nonzero(fr.seg_geom == g)
                p, _ = pc.cm.project(((d3.geom_xpos[g] - T[:3, 3]) @ T[:3, :3])[None])
                worst.append(np.hypot(uu.mean() - p[0, 0], vv.mean() - p[0, 1]) if len(uu) else 99)
            pc.close()
    finally:
        os.remove(path)
    check("3 frame/pixel alignment", max(worst) < 0.5,
          f"sphere centroid vs projection: max {max(worst):.2f} px (pinhole + distorted lens)")
    # 4 ---------------------------------------------------------------------------------------
    try:
        import warp  # noqa: F401  (GPU backend available?)
        del warp
        m4 = evk.load_model(SCENE, backend="warp", binning=4)
        d4 = mujoco.MjData(m4); demo.reset(m4, d4)
        cam = evk.Sensor(m4, "evk4", backend="warp", device=DEV); cam.update(d4)
        pc = evk.PairedCamera(m4)
        for _ in range(30):
            d4.ctrl[:6] = demo.joint_targets(d4.time); mujoco.mj_step(m4, d4); cam.update(d4)
        fr = pc.capture(d4, t=cam.time())
        lg = cam._s.lab_g.numpy().reshape(fr.seg_geom.shape); ld = cam._s.lab_d.numpy().reshape(lg.shape)
        agree = np.mean(lg == fr.seg_geom)
        both = (lg >= 0) & (lg == fr.seg_geom)
        dd = np.abs(ld - pc.range(fr))[both]
        pc.close()
        check("4 paired depth/seg == event engine", agree > 0.99 and np.median(dd) < 5e-4,
              f"segmentation agreement {agree:.4f}, range |diff| median {np.median(dd) * 1e3:.2f} mm")
    except ImportError:
        print("[SKIP] 4: warp not installed")
    # 5 ---------------------------------------------------------------------------------------
    m5 = evk.load_model(SCENE, binning=4)
    d5 = mujoco.MjData(m5); demo.reset(m5, d5)
    cam = evk.Sensor(m5, "evk4"); cam.update(d5)
    pc = evk.PairedCamera(m5)
    mujoco.mj_step(m5, d5); cam.update(d5); prev = pc.capture(d5, cam.time())
    rel = []
    for k in range(30):
        d5.ctrl[:6] = demo.joint_targets(d5.time); mujoco.mj_step(m5, d5); cam.update(d5)
        fr = pc.capture(d5, cam.time()); ev = cam.labeled_events()
        fl, ok = pc.flow(prev, fr); dt = fr.t - prev.t
        if len(ev) and k > 5:
            x, y = ev["x"].astype(int), ev["y"].astype(int)
            good = ok[y, x] & (fr.seg_geom[y, x] == ev["geom"])
            gt = fl[y[good], x[good]] / dt
            lab = np.stack([ev["flow_x"][good], ev["flow_y"][good]], 1)
            n = np.linalg.norm(gt, axis=1); sel = n > 50
            rel.append(np.linalg.norm(lab[sel] - gt[sel], axis=1) / n[sel])
        prev = fr
    pc.close()
    rel = np.concatenate(rel)
    check("5 event flow labels == dense GT flow", len(rel) > 1000 and np.median(rel) < 0.03,
          f"{len(rel)} events, median relative difference {np.median(rel):.3f} (finite-difference over 1 ms)")
    # 6 ---------------------------------------------------------------------------------------
    spec, cfg = ur5e_without_evk4()
    cfg["binning"] = "8"
    mount.add_evk4(spec, "wrist_3_link", pos=(-0.03, 0.169, 0), quat=(-1, 1, 0, 0), name="left", config=cfg)
    mount.add_evk4(spec, "wrist_3_link", pos=(0.03, 0.169, 0), quat=(-1, 1, 0, 0), name="right", config=cfg)
    m6 = spec.compile()
    el, er = run_events(m6, 60, "left"), run_events(m6, 60, "right")
    ok = len(el) > 200 and len(er) > 200 and not np.array_equal(el, er)
    check("6 stereo pair", ok, f"left {len(el)}, right {len(er)} events")
    # 7 ---------------------------------------------------------------------------------------
    men = os.path.join(HERE, "..", "robots", "menagerie")
    for folder, body in (("franka_emika_panda", "hand"), ("kinova_gen3", "bracelet_link"), ("skydio_x2", "x2")):
        xml = os.path.join(men, folder, "scene.xml")
        if not os.path.exists(xml):
            print(f"[SKIP] 7 {folder}: not downloaded (python tools/fetch_robots.py)")
            continue
        m7 = evk.attach_evk4(xml, body, pos=(0.07, 0, 0.02), config=dict(binning=8),
                             edit=lambda s: evk.add_props(s, center=(0.5, 0, 0)))
        d7 = mujoco.MjData(m7)
        s7 = evk.EVK4(m7)
        n = 0
        for k in range(60):
            if m7.nu:
                d7.ctrl[:] = 0.3 * np.sin(20 * d7.time)
            mujoco.mj_step(m7, d7); n += len(s7.events(d7))
        check(f"7 {folder}", n > 0, f"{n} events in 60 steps")
    print(f"{time.time() - t0:.0f} s")
    print("ALL PASSED" if all(RES) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
