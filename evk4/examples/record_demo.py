"""
Headless recording of the UR5e + EVK4 demo: event view (Metavision palette), third-person view,
photoreceptor log-intensity view and live event-rate plot, written to an MP4.

    MUJOCO_GL=egl python examples/record_demo.py --duration 4.2 --out demo.mp4

Each video frame shows the events of the last `--window-ms` of simulated time. With the default
`--frame-ms 10` and 30 fps the video plays at 1/3.3 of real time (slow motion).
"""
import argparse
import os
import sys
import time

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import evk4_mujoco as evk                       # noqa: E402
from evk4_mujoco import ur5e_demo as demo       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "..", "scenes", "scene_ur5e_evk4.xml")

PHASES = [(0.6, "static camera: only moving objects (disk, pendulum, ball) fire"),
          (2.1, "pan / tilt: ego-motion over the textured table"),
          (2.9, "fast wrist roll: rotational optical flow"),
          (3.6, "approach towards the spinning disk"),
          (1e9, "robot stopped: back to scene-motion only")]


def phase(t):
    for t1, s in PHASES:
        if t < t1:
            return s


def put(img, txt, org, scale=0.55, color=(235, 235, 235), thick=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.2)
    ap.add_argument("--frame-ms", type=float, default=10.0)
    ap.add_argument("--window-ms", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="evk4_ur5e_demo.mp4")
    ap.add_argument("--save-events", default="", help="optional .npy file for (a slice of) the event stream")
    ap.add_argument("--save-from", type=float, default=0.0, help="sim time [s] where saving starts")
    ap.add_argument("--save-to", type=float, default=1e9, help="sim time [s] where saving stops")
    ap.add_argument("--binning", type=int, default=1, help="2/4 = fast preview (larger virtual pixels)")
    ap.add_argument("--max-px-motion", type=float, default=1.0,
                    help="adaptive sampling: max image motion [px] between pixel-circuit samples")
    ap.add_argument("--disk-hz", type=float, default=2.0)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "warp"], help="warp = GPU (NVIDIA Warp)")
    ap.add_argument("--device", default=None, help="warp device, e.g. cuda:0")
    ap.add_argument("--raw", default="", help="also write the event stream as an EVT 3.0 .raw file")
    ap.add_argument("--panel", default="intensity", choices=["intensity", "flow", "objects"],
                    help="lower-right panel: photoreceptor image, GT flow of events, or GT object ids")
    args = ap.parse_args()

    m = evk.load_model(SCENE, binning=args.binning, max_px_motion=args.max_px_motion, backend=args.backend)
    d = mujoco.MjData(m)
    demo.reset(m, d, disk_hz=args.disk_hz)
    cam = evk.Sensor(m, "evk4", backend=args.backend, device=args.device)
    cam.update(d)
    W, H = cam.geometry()
    raw = None
    if args.raw:
        from evk4_mujoco.io import EVT3Writer
        raw = EVT3Writer(args.raw, W, H)
    print(f"EVK4 sensor {W}x{H}, timestep {m.opt.timestep*1e3:.1f} ms, backend {args.backend}")

    third = mujoco.Renderer(m, 360, 640)
    tcam = mujoco.MjvCamera()
    tcam.lookat[:] = [-0.12, 0.50, 0.22]; tcam.distance = 1.45; tcam.azimuth = 205; tcam.elevation = -28
    vopt = mujoco.MjvOption()

    frame = evk.EventFrame(W, H, "dark")
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
    buf, all_ev = [], []
    hist_t, hist_on, hist_off, hist_sub = [], [], [], []
    frame_dt = args.frame_ms * 1e-3
    next_frame = frame_dt
    t_wall = time.time()
    total = 0
    Lmin, Lmax = None, None

    while d.time < args.duration - 1e-9:
        d.ctrl[:6] = demo.joint_targets(d.time)
        mujoco.mj_step(m, d)
        cam.update(d)
        ev = cam.labeled_events()
        st = cam.stats()
        if raw is not None and len(ev):
            raw.write(ev[["x", "y", "p", "t"]])
        total += len(ev)
        if len(ev):
            buf.append(ev)
            if args.save_events and args.save_from <= d.time <= args.save_to:
                all_ev.append(ev.copy())
        hist_t.append(d.time); hist_on.append(st["n_on"]); hist_off.append(st["n_off"])
        hist_sub.append(st["n_substeps"])

        if d.time >= next_frame - 1e-9:
            next_frame += frame_dt
            t_us = d.time * 1e6
            evw = np.concatenate(buf) if buf else np.zeros(0, evk.LABELED_DTYPE)
            evw = evw[evw["t"] > t_us - args.window_ms * 1e3]
            buf = [evw]
            frame.reset(); frame.add(evw)
            evimg = frame.rgb()[:, :, ::-1].copy()          # BGR for cv2 drawing
            if (W, H) != (1280, 720):
                evimg = cv2.resize(evimg, (1280, 720), interpolation=cv2.INTER_NEAREST)

            if args.panel == "intensity":          # photoreceptor view
                L = cam.intensity()
                if Lmin is None:
                    Lmin, Lmax = np.percentile(L, 0.5), np.percentile(L, 99.8)
                g = np.clip((L - Lmin) / (Lmax - Lmin), 0, 1)
                gimg = cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:                                   # ground-truth labels carried by each event
                gimg = np.zeros((H, W, 3), np.uint8)
                if len(evw):
                    if args.panel == "flow":
                        ang = (np.degrees(np.arctan2(evw["flow_y"], evw["flow_x"])) % 360) / 2
                        mag = np.clip(np.hypot(evw["flow_x"], evw["flow_y"]) / 3000.0, 0, 1) * 255
                        hsv = np.stack([ang, np.full_like(ang, 255), np.maximum(mag, 60)], 1).astype(np.uint8)
                    else:
                        gid = evw["geom"].astype(np.int64)
                        hsv = np.stack([(gid * 37) % 180, np.full(len(gid), 200), np.where(gid >= 0, 255, 60)], 1).astype(np.uint8)
                    col = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]
                    gimg[evw["y"], evw["x"]] = col
            gimg = cv2.resize(gimg, (640, 360), interpolation=cv2.INTER_AREA if args.panel == "intensity" else cv2.INTER_NEAREST)

            third.update_scene(d, tcam, vopt)
            timg = third.render()[:, :, ::-1].copy()

            # rate plot (last 1.5 s)
            plot = np.full((180, 1920, 3), (22, 22, 26), np.uint8)
            ht = np.array(hist_t); sel = ht > d.time - 1.5
            if sel.sum() > 2:
                dt = m.opt.timestep
                on = np.array(hist_on)[sel] / dt / 1e6; off = np.array(hist_off)[sel] / dt / 1e6
                ymax = max(0.5, np.percentile(np.maximum(on, off), 99.5) * 1.15)
                xs = ((ht[sel] - (d.time - 1.5)) / 1.5 * 1840 + 60).astype(int)
                for arr, col in ((on, (236, 223, 216)), (off, (201, 126, 64))):
                    ys = (170 - np.clip(arr / ymax, 0, 1) * 140).astype(int)
                    cv2.polylines(plot, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
                put(plot, f"{ymax:.1f}", (8, 36), 0.45)
                put(plot, "0", (8, 172), 0.45)
            put(plot, "event rate [Mev/s]  ON = white, OFF = blue   (last 1.5 s)", (70, 22), 0.5)

            canvas = np.zeros((900, 1920, 3), np.uint8)
            canvas[0:720, 0:1280] = evimg
            canvas[0:360, 1280:1920] = timg
            canvas[360:720, 1280:1920] = gimg
            canvas[720:900] = plot
            rate = len(evw) / (args.window_ms * 1e-3) / 1e6
            hdr = canvas[0:66, 0:1280]
            hdr[:] = (hdr * 0.35).astype(np.uint8)
            put(canvas, f"Prophesee EVK4 HD (IMX636) - MuJoCo 'prophesee.sensor.evk4' v2 [{args.backend}]", (14, 28), 0.7)
            put(canvas, f"t = {d.time:5.3f} s   window {args.window_ms:.0f} ms   {rate:6.2f} Mev/s   "
                        f"substeps {int(hist_sub[-1])}   ({W}x{H})", (14, 56))
            put(canvas, phase(d.time), (14, 704), 0.65, (120, 220, 255), 2)
            put(canvas, "third-person (MuJoCo GL)", (1290, 22))
            put(canvas, {"intensity": "photoreceptor ln(I) inside the plugin",
                         "flow": "ground truth per event: optical flow (hue = direction)",
                         "objects": "ground truth per event: object id"}[args.panel], (1290, 382))
            speed = frame_dt * args.fps
            put(canvas, f"playback {speed:.2f}x real time", (1290, 350), 0.5)
            cv2.line(canvas, (1280, 0), (1280, 720), (80, 80, 80), 1)
            cv2.line(canvas, (1280, 360), (1920, 360), (80, 80, 80), 1)
            writer.append_data(canvas[:, :, ::-1])

            el = time.time() - t_wall
            print(f"sim {d.time:6.3f}s | wall {el:7.1f}s | events so far {total/1e6:8.2f} M | "
                  f"window rate {rate:6.2f} Mev/s | substeps {int(hist_sub[-1])}", flush=True)

    writer.close()
    if raw is not None:
        raw.close()
        print("EVT 3.0 file ->", args.raw)
    if args.save_events and all_ev:
        np.save(args.save_events, np.concatenate(all_ev))
    np.savez(os.path.splitext(args.out)[0] + "_rates.npz", t=np.array(hist_t), on=np.array(hist_on),
             off=np.array(hist_off), substeps=np.array(hist_sub))
    print(f"done: {total/1e6:.2f} M events, video -> {args.out}")


if __name__ == "__main__":
    main()
