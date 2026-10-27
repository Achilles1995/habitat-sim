"""
EVK4 in a real Habitat scene: a Habitat 3.0 humanoid walks through the van Gogh room, watched by
an EVK4 on Spot's body and an EVK4 on the wall; then Spot turns on the spot.

Checks
  1  HDR radiance sensor returns linear float data (values not clamped to 8 bit)
  2  static scene + static cameras -> only sensor noise (low event rate)
  3  walking human -> events, most of them on the human for the wall camera, labelled with the
     human's instance id, with range labels close to the camera-human distance
  4  flow labels of human events point the way the human moves in the image
  5  Spot turning left -> scene events on its camera with flow_x > 0 (image content moves right)
Writes event-frame / luminance previews and an event video per camera to --out.

    python tests/test_habitat_scene.py --data data --out evk4-out [--device cuda:0] [--binning 4]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import magnum as mn  # noqa: E402

import evk4_habitat as evkh  # noqa: E402

TICK = 1 / 30.0


def event_image(ev, W, H, ids=None, human_id=None):
    """Metavision-like event frame: dark background, ON blue-white, OFF dark blue; human events
    tinted orange (if human_id given)."""
    img = np.full((H, W, 3), (30, 37, 52), np.uint8)
    if not len(ev):
        return img
    on = ev["p"] == 1
    img[ev["y"][~on], ev["x"][~on]] = (200, 126, 64)
    img[ev["y"][on], ev["x"][on]] = (255, 255, 255)
    if human_id is not None:
        h = ev["geom"] == human_id
        img[ev["y"][h & on], ev["x"][h & on]] = (40, 170, 255)
        img[ev["y"][h & ~on], ev["x"][h & ~on]] = (0, 90, 200)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="evk4-out")
    ap.add_argument("--device", default=None)
    ap.add_argument("--binning", type=int, default=4)
    ap.add_argument("--scene", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import cv2

    scene = args.scene or os.path.join(args.data, "scene_datasets/habitat-test-scenes/van-gogh-room.glb")
    sim = evkh.make_simulator(scene)
    pf = sim.pathfinder
    print("navmesh:", pf.is_loaded)
    lo, hi = (np.array(v) for v in pf.get_bounds())
    floor_y = float(lo[1])
    ctr = (lo + hi) / 2
    motion = evkh.SceneMotion(sim)

    # human walks across the room along its longer axis, Spot stands at the side
    ax = 0 if (hi[0] - lo[0]) >= (hi[2] - lo[2]) else 2
    side = 2 - ax
    a = ctr.copy(); a[ax] = lo[ax] + 0.25 * (hi[ax] - lo[ax])
    b = ctr.copy(); b[ax] = lo[ax] + 0.75 * (hi[ax] - lo[ax])
    a[1] = b[1] = floor_y
    pa, pb = evkh.snap(sim, a), evkh.snap(sim, b)
    urdf, pkl = evkh.humanoid_files(args.data)
    human = evkh.Humanoid(sim, motion, urdf, pkl, pa)
    human.place(pa, heading=pb - pa)
    sp = ctr.copy(); sp[1] = floor_y; sp[side] = lo[side] + 0.2 * (hi[side] - lo[side])
    sp = evkh.snap(sim, sp)
    look = mn.Vector3(ctr[0], floor_y, ctr[2]) - sp
    spot = evkh.Robot(sim, motion, "spot", sp, data_dir=args.data, yaw=math.atan2(-look.z, look.x))
    print("human at", pa, "->", pb, "spot at", sp)

    cfg = dict(resolution="1280 720", binning=args.binning)
    eye = [lo[0] + 0.15, floor_y + 2.1, lo[2] + 0.15]       # high up in a corner
    tgt = [ctr[0], floor_y + 0.8, ctr[2]]
    cams = {
        "wall": evkh.EVK4Habitat(sim, evkh.wall_mount(eye, tgt), motion, name="evk4_wall",
                                 config=dict(cfg, hfov_deg=80), device=args.device),
        "spot_front": evkh.EVK4Habitat(sim, evkh.spot_mount(spot.ao, "front"), motion, name="evk4_spot",
                                       config=dict(cfg, hfov_deg=90), device=args.device),
    }
    results = []

    def check(name, ok, info):
        print(("[PASS] " if ok else "[FAIL] ") + f"{name}: {info}", flush=True)
        results.append((name, bool(ok), info))

    # ---- 1 HDR radiance
    t = 0.0
    for c in cams.values():
        c.update(t)
    rad = cams["wall"]._last_obs["rad"]
    vals = rad[..., :3][cams["wall"]._last_obs["depth"] > 0]
    q = np.quantile(vals, [0.01, 0.5, 0.99]) if len(vals) else [0, 0, 0]
    distinct = len(np.unique(np.round(vals.ravel()[:200000] * 255 * 64)))
    check("1 HDR radiance", rad.dtype == np.float32 and distinct > 4 * 256 / 4,
          f"dtype {rad.dtype}, radiance quantiles 1/50/99% {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}, "
          f"{distinct} distinct levels at 1/64 of an 8-bit step; lux_per_unit {cams['wall'].lux_per_unit:.1f}")

    log = {k: [] for k in cams}
    frames = {k: [] for k in cams}
    phases = []
    human_id = human.instance_id

    def step_all(phase):
        nonlocal t
        t += TICK
        for k, c in cams.items():
            c.update(t)
            log[k].append((phase, c.labeled_events().copy(), c.nsub))
            ev = c.labeled_events()
            img = event_image(ev, c.W, c.H, human_id=human_id)
            frames[k].append(img)
        phases.append(phase)

    t0 = time.time()
    for _ in range(9):                       # 0.3 s static
        step_all("static")
    human.walk_to(pb)
    h_start = np.array(human.feet)
    walk_steps = 0
    while human.path and walk_steps < 75:
        human.follow_path()
        step_all("walk")
        walk_steps += 1
    h_end = np.array(human.feet)
    for _ in range(15):                      # Spot turns left at 60 deg/s
        spot.yaw += math.radians(60) * TICK
        spot.apply()
        step_all("turn")
    wall_time = time.time() - t0
    print(f"human walked {np.linalg.norm(h_end - h_start):.2f} m in {walk_steps} steps; "
          f"{len(phases)} updates in {wall_time:.1f} s")

    # ---- 2 static noise
    for k, c in cams.items():
        n = sum(len(e) for p, e, _ in log[k] if p == "static")
        rate = n / c.N / (9 * TICK)
        check(f"2 static scene ({k})", rate < 2.0, f"{rate:.3f} events/px/s with nothing moving")

    # ---- 3/4 human events (wall camera)
    c = cams["wall"]
    ev = np.concatenate([e for p, e, _ in log["wall"] if p == "walk"])
    hev = ev[ev["geom"] == human_id]
    frac = len(hev) / max(1, len(ev))
    Tcw = np.linalg.inv(c.T_world_cam())
    mid = (h_start + h_end) / 2 + np.array([0, 0.9, 0])
    dist = float(np.linalg.norm((Tcw @ np.r_[mid, 1])[:3]))
    med_d = float(np.median(hev["depth"])) if len(hev) else 0
    check("3 human events (wall)", len(hev) > 500 and frac > 0.5 and abs(med_d - dist) < 1.0,
          f"{len(ev)} events while walking, {len(hev)} ({frac:.0%}) on the human; median range "
          f"{med_d:.2f} m vs camera-human {dist:.2f} m")
    # expected image motion direction of the human
    p0, _ = c._project(c.T0, np.array([h_start + [0, 0.9, 0]]))
    p1, _ = c._project(c.T0, np.array([h_end + [0, 0.9, 0]]))
    dimg = (p1 - p0)[0]
    fmed = np.array([np.median(hev["flow_x"]), np.median(hev["flow_y"])]) if len(hev) else np.zeros(2)
    cosang = float(np.dot(dimg, fmed) / (np.linalg.norm(dimg) * np.linalg.norm(fmed) + 1e-9))
    check("4 human flow direction (wall)", cosang > 0.7,
          f"image displacement of the pelvis {dimg.round(1)} px, median event flow {fmed.round(1)} px/s, "
          f"cos {cosang:.2f}")
    # spot camera sees the human too
    evs = np.concatenate([e for p, e, _ in log["spot_front"] if p == "walk"])
    print(f"spot_front while walking: {len(evs)} events, {(evs['geom'] == human_id).mean() if len(evs) else 0:.0%} on the human")

    # ---- 5 ego-motion
    evt = np.concatenate([e for p, e, _ in log["spot_front"] if p == "turn"])
    st = evt[(evt["geom"] >= 0) & (evt["geom"] < evkh.HUMAN_ID0)]
    fx = float(np.median(st["flow_x"])) if len(st) else 0
    exp_fx = math.radians(60) * cams["spot_front"].fx
    nsubs = [n for p, _, n in log["spot_front"] if p == "turn"]
    check("5 Spot turning left (spot_front)", len(st) > 1000 and fx > 0.5 * exp_fx,
          f"{len(evt)} events, median scene flow_x {fx:.0f} px/s (rotation alone: {exp_fx:.0f}); "
          f"sub-steps {min(nsubs)}..{max(nsubs)}")

    # ---- outputs
    inst = motion.instances()
    for k, c in cams.items():
        fr = c.frame()
        cv2.imwrite(os.path.join(args.out, f"{k}_rgb.png"), cv2.cvtColor(fr["rgb"], cv2.COLOR_RGB2BGR))
        idx = [i for i, p in enumerate(phases) if p == "walk"]
        picks = [idx[len(idx) // 4], idx[len(idx) // 2], idx[3 * len(idx) // 4]] if idx else []
        picks += [len(phases) - 3]
        strip = np.concatenate([frames[k][i] for i in picks], 1) if picks else frames[k][-1]
        cv2.imwrite(os.path.join(args.out, f"{k}_events.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        try:
            import imageio
            imageio.mimsave(os.path.join(args.out, f"{k}_events.mp4"), frames[k], fps=15)
        except Exception as e:  # noqa: BLE001
            print("video not written:", e)
        tm = c.timing()
        print(f"{k}: {c.W}x{c.H}, intrinsics {np.round(c.intrinsics(), 1)}, timing ms/update {tm}")
    with open(os.path.join(args.out, "instances.json"), "w") as f:
        json.dump({str(k): v for k, v in inst.items()}, f, indent=1)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        for name, ok, info in results:
            f.write(f"{'PASS' if ok else 'FAIL'}  {name}  {info}\n")
    ok = all(r[1] for r in results)
    print("ALL PASSED" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
