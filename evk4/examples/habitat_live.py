#!/usr/bin/env python3
"""
Live view of the EVK4 in Habitat: a humanoid walks between random places while Spot drives
(I/K forward/back, J/L turn).  The window shows, for the selected camera, the event stream
(33 ms per frame; human events orange, everything else white/blue) next to the paired
grey-level luminance preview.

  python evk4/examples/habitat_live.py --data data [--device cuda:0] [--binning 2]
  python evk4/examples/habitat_live.py --dataset data/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json --scene 102816009

Keys: 1 Spot camera, 2 wall camera, I/K/J/L drive Spot, N new human goal, Esc quit.
"""
import argparse
import glob
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import magnum as mn  # noqa: E402

import evk4_habitat as evkh  # noqa: E402
from hrc_generate import event_rgb, nav_point, wall_camera_pose  # noqa: E402

TICK = 1 / 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--binning", type=int, default=2)
    ap.add_argument("--bias", default="", help='JSON EVK4 config overrides, e.g. \'{"bias_diff_on": 10}\'')
    args = ap.parse_args()
    scene = args.scene or os.path.join(args.data, "scene_datasets/habitat-test-scenes/van-gogh-room.glb")
    if args.dataset and not os.path.exists(scene):
        hits = glob.glob(os.path.join(os.path.dirname(os.path.abspath(args.dataset)), "**",
                                      f"{scene}.scene_instance.json"), recursive=True)
        scene = hits[0] if hits else scene
    sim = evkh.make_simulator(scene, args.dataset)
    rng = np.random.default_rng(0)
    motion = evkh.SceneMotion(sim)
    rp = nav_point(sim, rng)
    spot = evkh.Robot(sim, motion, "spot", rp, data_dir=args.data)
    hp = nav_point(sim, rng, near=rp, radius=4.0)
    human = evkh.Humanoid(sim, motion, *evkh.humanoid_files(args.data), feet_pos=hp)
    cfg = dict(resolution="1280 720", binning=args.binning)
    if args.bias:
        import json
        cfg.update(json.loads(args.bias))
    cams = [evkh.EVK4Habitat(sim, evkh.spot_mount(spot.ao, "front"), motion, name="spot_front",
                             config=dict(cfg, hfov_deg=90), device=args.device)]
    pose = wall_camera_pose(sim, np.array(rp), rng)
    if pose:
        cams.append(evkh.EVK4Habitat(sim, evkh.wall_mount(*pose), motion, name="wall",
                                     config=dict(cfg, hfov_deg=80), device=args.device))
    sel, t = 0, 0.0
    win = "EVK4 in Habitat (1 Spot cam, 2 wall cam, IJKL drive, N new goal, Esc quit)"
    while True:
        t0 = time.time()
        if not human.path:
            human.walk_to(nav_point(sim, rng, near=spot.pos, radius=5.0))
        human.follow_path()
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("1"), ord("2")) and int(chr(key)) <= len(cams):
            sel = int(chr(key)) - 1
        if key == ord("n"):
            human.path = None
        spot.speed = 0.0
        if key in (ord("i"), ord("k")):
            spot.speed = 0.8 if key == ord("i") else -0.5
            spot.drive(spot.speed * TICK * 3)
        if key in (ord("j"), ord("l")):
            spot.yaw += math.radians(10) * (1 if key == ord("j") else -1)
        spot.apply()
        t += TICK
        for c in cams:
            c.update(t)
        c = cams[sel]
        ev = c.labeled_events()
        img = event_rgb(ev, c.W, c.H, [human.instance_id])
        fr = c.frame()
        lum = cv2.cvtColor(cv2.cvtColor(fr["rgb"], cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2BGR)
        lum = cv2.resize(lum, (c.W, c.H))
        view = np.concatenate([img, lum], 1)
        scale = max(1, 960 // view.shape[1])
        view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        st = c.stats()
        txt = f"{c.name}  {st['n_events']} ev/33ms  substeps {st['n_substeps']}  {1 / max(1e-3, time.time() - t0):.1f} FPS"
        cv2.putText(view, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, view)
    sim.close()


if __name__ == "__main__":
    main()
