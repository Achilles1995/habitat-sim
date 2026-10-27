"""
Checks of the ESD simulation (UR5e + stereo DAVIS346 + D435, ESD objects and scenarios).

    python tests/test_esd.py [--backend warp --device cuda:0]

1  rig calibration: cameras look straight down, ~5-8 deg toe-in, ~0.21 m stereo baseline (ESD values)
2  UR5e IK follows every trajectory type at both heights (< 0.05 mm), arm never in the images
3  trajectory speeds: 0.15 / 0.3 m/s reached exactly, 1 m/s reached on long moves
4  a generated sequence has the ESD layout, variables and value conventions
5  per-event labels agree with the dense masks (slow motion)
6  the loader reads simulated folders and a real-format (MATLAB v7.3) folder identically; metrics
"""
import glob
import json
import os
import shutil
import sys
import tempfile
import time

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco                                    # noqa: E402
import numpy as np                               # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from evk4_mujoco.esd import data, generate as G, motion, scene  # noqa: E402

BACKEND = sys.argv[sys.argv.index("--backend") + 1] if "--backend" in sys.argv else "cpu"
DEV = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else None
RES = []


def check(name, ok, msg):
    RES.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)


def main():
    t_start = time.time()
    tmp = tempfile.mkdtemp(prefix="esd_test_")
    try:
        # 1 ------------------------------------------------------------------------------------
        m, info = scene.build(scene.SceneConfig(objects=[], tex_cache=os.path.join(tmp, "tex")))
        d = mujoco.MjData(m)
        drv = motion.RigDriver(m, d, info, 0.82, motion.ESDTrajectory("linear", 0.15, 1, np.random.default_rng(0)))
        mujoco.mj_forward(m, d)
        ax = {c: -d.cam_xmat[m.camera(f"{c}_cam").id].reshape(3, 3)[:, 2] for c in ("left", "right", "d435")}
        tilt = [np.degrees(np.arccos(np.clip(ax[c] @ ax["d435"], -1, 1))) for c in ("left", "right")]
        base = np.linalg.norm(d.cam_xpos[m.camera("left_cam").id] - d.cam_xpos[m.camera("right_cam").id])
        down = np.degrees(np.arccos(-ax["d435"][2]))
        check("1 rig calibration", down < 0.1 and all(3 < t < 10 for t in tilt) and 0.18 < base < 0.24,
              f"D435 {down:.2f} deg from vertical, toe-in L {tilt[0]:.1f} / R {tilt[1]:.1f} deg, baseline {base:.3f} m")
        # 2 ------------------------------------------------------------------------------------
        errs, robot_px = [], 0
        from evk4_mujoco import PairedCamera
        for h in (0.82, 0.62):
            for tr in motion.TRAJS:
                d = mujoco.MjData(m)
                drv = motion.RigDriver(m, d, info, h, motion.ESDTrajectory(tr, 1.0, 2.0, np.random.default_rng(1),
                                                                           extent=(0.15, 0.11), acc=6.0))
                pc = PairedCamera(m, "d435", rgb=False, depth=False)
                rb = G._robot_bodies(m)
                for t in np.arange(0, 2.0, 0.004):
                    drv.apply(d, t)
                    if abs(t % 0.25) < 1e-6:
                        mujoco.mj_forward(m, d)
                        fr = pc.capture(d)
                        robot_px += int(np.isin(m.geom_bodyid[np.maximum(fr.seg_geom, 0)], rb)[fr.seg_geom >= 0].sum())
                pc.close()
                errs.append(drv.max_err)
        check("2 UR5e IK + arm out of view", max(errs) < 5e-5 and robot_px == 0,
              f"max tracking error {max(errs) * 1e3:.4f} mm over 6 trajectories, robot pixels in D435: {robot_px}")
        # 3 ------------------------------------------------------------------------------------
        sp = []
        for v in (0.15, 0.3, 1.0):
            tr = motion.ESDTrajectory("linear", v, 4.0, np.random.default_rng(2), extent=(0.15, 0.11),
                                      acc=6.0 if v > 0.8 else 1.5)
            ts = np.arange(0, 4.0, 0.001)
            xy = np.array([tr(t)[:2] for t in ts])
            sp.append(float(np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1) / 0.001)))
        check("3 trajectory speeds", abs(sp[0] - 0.15) < 0.005 and abs(sp[1] - 0.3) < 0.01 and sp[2] > 0.9,
              f"peak TCP speed {sp[0]:.3f} / {sp[1]:.3f} / {sp[2]:.3f} m/s for S015 / S03 / S1")
        # 4 ------------------------------------------------------------------------------------
        from scipy.io import loadmat
        p = G.SeqParams(object_set="esd1", n_obj=6, speed=0.15, traj="linear", height=0.82, occlusion=True,
                        duration=0.25, seed=3, backend=BACKEND, device=DEV, tex_cache=os.path.join(tmp, "tex"))
        root = G.generate(p, tmp, "train", verbose=False)
        L = loadmat(os.path.join(root, "events", "left.mat"))
        E = L["event_labeled"]
        meta = json.load(open(os.path.join(root, "meta.json")))
        lab_ok = set(np.unique(E[:, 5]).astype(int)) <= set(range(0, 11))
        ok = (E.shape[1] == 6 and len(E) > 1000 and set(np.unique(E[:, 2])) <= {-1.0, 1.0}
              and np.all(np.diff(E[:, 3]) >= 0) and E[:, 3].max() < 0.3e9 and lab_ok
              and len(glob.glob(os.path.join(root, "RGB", "images", "*.png"))) == meta["n_frames"]
              and all(os.path.exists(os.path.join(root, "events", f)) for f in
                      ("right.mat", "events_frame.mat", "mask_events_frame.mat")))
        check("4 ESD layout + conventions", ok, f"{len(E)} left events, cols x y p(+-1) t(ns) depth(mm) label, "
              f"labels {sorted(set(np.unique(E[:, 5]).astype(int)))}, {meta['n_frames']} RGB-D frames")
        # 5 ------------------------------------------------------------------------------------
        mk = loadmat(os.path.join(root, "events", "mask_events_frame.mat"))
        masks, mts = mk["left"], mk["ts"].ravel()
        k = np.clip(np.searchsorted(mts, E[:, 3]), 0, len(mts) - 1)
        x, y = E[:, 0].astype(int), E[:, 1].astype(int)
        lm = masks[k, y, x]
        sel = E[:, 5] > 0
        agree = float(np.mean(lm[sel] == E[sel, 5]))
        dz = E[sel, 4]
        check("5 event labels vs dense masks", agree > 0.9 and np.all((dz > 500) & (dz < 900)),
              f"{100 * agree:.1f} % of object events carry the mask label at their pixel; object depth "
              f"{dz.min():.0f}..{dz.max():.0f} mm (camera at 820 mm)")
        # 6 ------------------------------------------------------------------------------------
        seq = data.Sequence(root)
        ev = seq.events("left")
        smp = list(seq.samples("left", window_us=60000, repr="voxel"))
        fr = seq.rgb_frames()
        x0, y0 = smp[0]
        acc = data.pixel_accuracy(y0, y0); mi = data.miou(y0, y0)
        fake = os.path.join(tmp, "real", "conditions_1")              # real-ESD-like folder: v7.3 mat + colour masks
        os.makedirs(os.path.join(fake, "events")); os.makedirs(os.path.join(fake, "RGB", "images"))
        os.makedirs(os.path.join(fake, "RGB", "masks"))
        import h5py
        with h5py.File(os.path.join(fake, "events", "left.mat"), "w") as f:
            f["event_labeled"] = E.T                                   # MATLAB column-major
        shutil.copy(os.path.join(root, "RGB", "images", "0.png"), os.path.join(fake, "RGB", "images", "14.png"))
        shutil.copy(os.path.join(root, "RGB", "masks", "0.png"), os.path.join(fake, "RGB", "masks", "14.png"))
        real = data.Sequence(fake)
        ev2 = real.events("left"); fr2 = real.rgb_frames()
        same = np.array_equal(ev, ev2) and np.array_equal(fr2[0][1], fr[0][1])
        check("6 loader sim == real format, metrics", same and len(smp) >= 3 and x0.shape == (5, 260, 346)
              and acc == 1.0 and mi == 1.0, f"{len(ev)} events, {len(smp)} 60 ms samples, identical after "
              f"v7.3 round trip={same}, Acc/mIoU self-check {acc:.2f}/{mi:.2f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{time.time() - t_start:.0f} s")
    print("ALL PASSED" if all(RES) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
