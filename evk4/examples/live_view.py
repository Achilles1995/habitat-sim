"""
LIVE view of the UR5e + EVK4 simulation.

The MuJoCo viewer shows the robot (with the EVK4 frustum) and a picture-in-picture panel with the
live event stream.  If your OpenCV has GUI support, a separate full-size event window opens too.

    python examples/live_view.py                    # full EVK4 resolution (1280x720)
    python examples/live_view.py --binning 2        # 640x360 virtual pixels, ~4x faster
    python examples/live_view.py --manual           # drive the arm yourself: viewer Control panel
    python examples/live_view.py --backend warp --resolution 320 320     # 320x320 sensor, same vertical FOV
    python examples/live_view.py --resolution 320 320 --focal-length-mm 4  # = real EVK4 320x320 ROI, 4 mm lens
    mjpython examples/live_view.py ...              # macOS

Keys (viewer window or event window):
    SPACE  pause        + / -  accumulation window        I  toggle photoreceptor ln(I) view
    R      reset scene  S      save current event window to .npy       Q/ESC (event window) quit

The simulation runs as fast as the CPU ray-casting pixel model allows (slower than real time at full
resolution); the overlay shows the real-time factor.  Events are shown in simulated time, exactly
as an EVK4 would timestamp them.
"""
import argparse
import os
import re
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import evk4_mujoco as evk                       # noqa: E402
from evk4_mujoco import ur5e_demo as demo       # noqa: E402

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenes", "scene_ur5e_evk4.xml")


def resize_nn(img, w, h):
    ys = (np.arange(h) * img.shape[0] / h).astype(int)
    xs = (np.arange(w) * img.shape[1] / w).astype(int)
    return img[ys][:, xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binning", type=int, default=1)
    ap.add_argument("--window-ms", type=float, default=20.0)
    ap.add_argument("--display-ms", type=float, default=10.0, help="sim time between display updates")
    ap.add_argument("--manual", action="store_true", help="do not run the scripted trajectory")
    ap.add_argument("--loop", type=float, default=4.2, help="scripted trajectory loop length [s]")
    ap.add_argument("--max-px-motion", type=float, default=1.0)
    ap.add_argument("--pip-width", type=int, default=640, help="event panel width inside the viewer")
    ap.add_argument("--no-cv-window", action="store_true")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "warp"], help="warp = GPU (NVIDIA Warp)")
    ap.add_argument("--device", default=None, help="warp device, e.g. cuda:0")
    ap.add_argument("--resolution", type=int, nargs=2, metavar=("W", "H"), default=None,
                    help="sensor array size, e.g. 320 320 (EVK4 pixels, 4.86 um)")
    ap.add_argument("--focal-length-mm", type=float, default=None,
                    help="lens focal length; default with --resolution: scaled so the vertical field of view "
                         "stays the same as the scene's 1280x720 setup")
    ap.add_argument("--config", nargs="*", default=[], help="any other plugin config, e.g. shadows=0 bias_diff_on=20")
    ap.add_argument("--dry-run", type=int, default=0, metavar="STEPS",
                    help="no window: step N times and print sensor info (quick check)")
    args = ap.parse_args()

    over = dict(kv.split("=", 1) for kv in args.config)
    if args.resolution:
        over["resolution"] = f"{args.resolution[0]} {args.resolution[1]}"
        if args.focal_length_mm is None:                      # keep the scene's vertical field of view
            xml = open(SCENE).read()
            f0 = float(re.search(r'key="focal_length_mm"\s+value="([^"]+)"', xml).group(1))
            h0 = int(re.search(r'key="resolution"\s+value="\s*\d+\s+(\d+)', xml).group(1))
            args.focal_length_mm = f0 * args.resolution[1] / h0
    if args.focal_length_mm is not None:
        over["focal_length_mm"] = args.focal_length_mm
    m = evk.load_model(SCENE, binning=args.binning, max_px_motion=args.max_px_motion, backend=args.backend,
                       **over)
    d = mujoco.MjData(m)
    demo.reset(m, d)
    cam = evk.Sensor(m, "evk4", backend=args.backend, device=args.device)
    cam.update(d)
    W, H = cam.geometry()
    frame = evk.EventFrame(W, H, "dark")
    fx = cam.intrinsics()[0]
    print(f"EVK4 sim: {W}x{H} px, fx {fx:.1f} px, FOV {2 * np.degrees(np.arctan(W / 2 / fx)):.1f} x "
          f"{2 * np.degrees(np.arctan(H / 2 / fx)):.1f} deg, backend {args.backend}")
    if args.dry_run:
        t0 = time.time(); n = 0
        for _ in range(args.dry_run):
            d.ctrl[:6] = demo.joint_targets(d.time); mujoco.mj_step(m, d); cam.update(d); n += len(cam.events())
        el = time.time() - t0
        print(f"{args.dry_run} steps: {n} events, {1e3 * el / args.dry_run:.2f} ms per {m.opt.timestep * 1e3:g} ms "
              f"step (real-time factor {args.dry_run * m.opt.timestep / el:.3f})")
        return
    scale = max(1, 720 // max(1, H * args.binning)) * args.binning     # small arrays: enlarge the window

    # optional OpenCV window (needs opencv-python with GUI, not -headless)
    win = "EVK4 events (prophesee.sensor.evk4)"
    use_cv = False
    if cv2 is not None and not args.no_cv_window:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, W * scale, H * scale)
            use_cv = True
        except cv2.error:
            print("OpenCV has no GUI support -> events are shown inside the MuJoCo viewer only")

    state = dict(paused=False, show_I=False, window_ms=args.window_ms, reset=False, save=False)

    def on_key(key):
        c = chr(key) if 0 <= key < 256 else ""
        if c == " ":
            state["paused"] = not state["paused"]
        elif c in "+=":
            state["window_ms"] = min(500.0, state["window_ms"] * 1.5)
        elif c == "-":
            state["window_ms"] = max(1.0, state["window_ms"] / 1.5)
        elif c in "iI":
            state["show_I"] = not state["show_I"]
        elif c in "rR":
            state["reset"] = True
        elif c in "sS":
            state["save"] = True

    buf, evw = [], np.zeros(0, evk.EVENT_DTYPE)
    t0_loop, next_disp = 0.0, 0.0
    wall0, sim0 = time.time(), d.time
    pw = args.pip_width

    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = [-0.12, 0.50, 0.22]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.45, 205, -28
        while viewer.is_running():
            if state["reset"]:
                state["reset"] = False
                with viewer.lock():
                    demo.reset(m, d)
                    cam.update(d)
                buf, t0_loop, next_disp = [], 0.0, 0.0
                wall0, sim0 = time.time(), d.time

            if not state["paused"]:
                with viewer.lock():
                    if not args.manual:
                        tl = d.time - t0_loop
                        if tl > args.loop:
                            t0_loop, tl = d.time, 0.0
                        d.ctrl[:6] = demo.joint_targets(tl)
                    mujoco.mj_step(m, d)
                    cam.update(d)
                    ev = cam.events()
                if len(ev):
                    buf.append(ev)
            else:
                time.sleep(0.02)

            if d.time >= next_disp or state["paused"]:
                next_disp = d.time + args.display_ms * 1e-3
                wms = state["window_ms"]
                t_us = d.time * 1e6
                evw = np.concatenate(buf) if buf else np.zeros(0, evk.EVENT_DTYPE)
                evw = evw[evw["t"] > t_us - wms * 1e3]
                buf = [evw]
                if state["show_I"]:
                    L = cam.intensity()
                    g = np.clip((L - np.percentile(L, 0.5)) / (np.ptp(L) + 1e-9), 0, 1)
                    rgb = np.repeat((g * 255).astype(np.uint8)[:, :, None], 3, 2)
                else:
                    frame.reset(); frame.add(evw)
                    rgb = frame.rgb()
                rtf = (d.time - sim0) / max(1e-6, time.time() - wall0)
                rate = len(evw) / (wms * 1e-3) / 1e6

                # picture-in-picture inside the MuJoCo viewer (bottom-right)
                vp = viewer.viewport
                if vp is not None and vp.width > 64 and vp.height > 64:
                    w = int(min(pw, 0.55 * vp.width, 0.55 * vp.height * W / H))
                    h = int(round(w * H / W))
                    rect = mujoco.MjrRect(vp.left + vp.width - w - 8, vp.bottom + 8, w, h)
                    viewer.set_images((rect, resize_nn(rgb, w, h)))
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                                  "EVK4 (IMX636)\nsim time\nevent rate\nwindow\nreal-time factor\nstatus",
                                  f"{W}x{H}\n{d.time:.3f} s\n{rate:.2f} Mev/s\n{wms:.0f} ms\n{rtf:.3f}\n"
                                  + ("PAUSED" if state["paused"] else "running")))
                viewer.sync()

                if use_cv:
                    img = np.ascontiguousarray(rgb[:, :, ::-1])
                    cv2.putText(img, f"t={d.time:6.3f}s  {rate:6.2f} Mev/s  win {wms:.0f}ms  RTF {rtf:.3f}",
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5 if W >= 640 else 0.35,
                                (230, 230, 230), 1, cv2.LINE_AA)
                    cv2.imshow(win, img)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 27):
                        break
                    if k != 255:
                        on_key(ord(chr(k).upper()) if chr(k).isalpha() else k)
            elif not state["paused"] and int(d.time * 1000) % 5 == 0:
                viewer.sync()

            if state["save"]:
                state["save"] = False
                fn = f"evk4_snapshot_{d.time:.3f}s.npy"
                np.save(fn, evw)
                print("saved", fn, len(evw), "events")
    if use_cv:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
