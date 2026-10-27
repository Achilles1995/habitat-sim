"""
EVK4 event camera on ANY MuJoCo robot + paired RGB / depth / segmentation / optical-flow frames.

    python tools/fetch_robots.py panda gen3 x2 ur5e                 # once: Menagerie robots
    python examples/any_robot.py --robot panda --backend warp --device cuda:0          # live viewer
    python examples/any_robot.py --robot x2 --record x2.mp4 --seconds 3                # video panels
    python examples/any_robot.py --robot ur5e_scene --frames-dir out --seconds 1       # dataset-style dump
    python examples/any_robot.py --robot my --xml path/to/scene.xml --body tool0 --pos 0 0 0.05

Robots: panda (hand camera), gen3 (wrist camera), x2 (drone, forward-down camera), ur5e (Menagerie
UR5e, flange camera), ur5e_scene (the bundled UR5e table scene, no download needed), or any MJCF
with --xml/--body.  A texture-rich prop set is placed where the camera looks at the start pose.

--record : MP4 with 4 panels  events | RGB  /  depth | ground-truth flow   (all pixel-aligned)
--frames-dir : events.npy (labelled: x y p t geom depth flow_x flow_y) + frame_XXXXX.npz with
               rgb, depth, seg_geom, seg_body, flow_to_next, T_world_cam, K, dist, t
"""
import argparse
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import evk4_mujoco as evk                        # noqa: E402

MENAGERIE = os.path.join(HERE, "..", "robots", "menagerie")
# mount poses: OpenCV camera frame (z = view direction, y = image down) in the parent body frame
PRESETS = {
    "panda": dict(xml="franka_emika_panda/scene.xml", body="hand", pos=(0.065, 0.0, 0.02),
                  quat=(1, 0, 0, 0), key=0),
    "gen3": dict(xml="kinova_gen3/scene.xml", body="bracelet_link", pos=(0.0, -0.06, -0.075),
                 look=((0.0, 0.64, -0.77), (0, -1, 0)), key=0),                  # 40 deg down
    "x2": dict(xml="skydio_x2/scene.xml", body="x2", pos=(0.12, 0.0, -0.02),
               look=((1.0, 0.0, -1.0), (0, 0, 1)), key=0, flying=True),          # 45 deg down
    "ur5e": dict(xml="universal_robots_ur5e/scene.xml", body="wrist_3_link", pos=(0.0, 0.169, 0.0),
                 quat=(-0.7071068, 0.7071068, 0, 0), key=0),
}


def build(a):
    cfg = dict(kv.split("=", 1) for kv in a.config)
    if a.resolution:
        cfg["resolution"] = f"{a.resolution[0]} {a.resolution[1]}"
    if a.focal_length_mm:
        cfg["focal_length_mm"] = a.focal_length_mm
    if a.binning > 1:
        cfg["binning"] = a.binning
    if a.robot == "ur5e_scene":
        m = evk.load_model(os.path.join(HERE, "..", "scenes", "scene_ur5e_evk4.xml"), backend=a.backend, **cfg)
        return m, dict(ur5e_scene=True), None
    if a.robot in PRESETS:
        p = dict(PRESETS[a.robot])
        p["xml"] = os.path.join(a.menagerie or MENAGERIE, p["xml"])
        if not os.path.exists(p["xml"]):
            sys.exit(f"{p['xml']} not found - run:  python tools/fetch_robots.py {a.robot}")
    else:
        if not a.xml or not a.body:
            sys.exit("custom robot: give --xml and --body")
        p = dict(xml=a.xml, body=a.body, pos=tuple(a.pos), quat=(1, 0, 0, 0), key=0)
    if "look" in p:
        d_, up = p.pop("look")
        p["quat"] = evk.look_at_quat(p["pos"], np.add(p["pos"], d_), up)
    props = {}

    def edit(spec):
        spec.option.timestep = a.timestep
        if p.get("flying"):
            spec.option.gravity = [0, 0, 0]                  # the flight path is kinematic
        # a prop set where the camera looks (computed below from the start pose) - placeholder here
        props.update(evk.add_props(spec, center=a._props_center, size=0.25))

    # 1st pass: find where the optical axis hits the floor at the start pose
    m0 = evk.attach_evk4(p["xml"], p["body"], pos=p["pos"], quat=p["quat"], backend=a.backend, config=cfg)
    d0 = mujoco.MjData(m0)
    if m0.nkey:
        mujoco.mj_resetDataKeyframe(m0, d0, p.get("key", 0))
    if p.get("flying"):
        _fly(m0, d0, 0.0)
    mujoco.mj_forward(m0, d0)
    cid = m0.camera("evk4_cam").id
    c, fwd = d0.cam_xpos[cid], -d0.cam_xmat[cid].reshape(3, 3)[:, 2]
    s = (0.0 - c[2]) / fwd[2] if fwd[2] < -1e-3 else 0.6
    a._props_center = (0.0, 0.0, 0.0) if p.get("flying") else (c + max(0.2, min(2.0, s)) * fwd) * [1, 1, 0]
    m = evk.attach_evk4(p["xml"], p["body"], pos=p["pos"], quat=p["quat"], backend=a.backend, config=cfg, edit=edit)
    return m, p, props


def _fly(m, d, t, center=(0.0, 0.0), radius=0.7, alt=0.75):
    """Kinematic drone path: orbit around the prop set, nose (camera) towards its centre."""
    w = 2 * np.pi * 0.25
    ang = w * t
    pos = np.array([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang),
                    alt + 0.08 * np.sin(3 * w * t)])
    yaw = ang + np.pi                                           # look at the centre
    roll = 0.12 * np.sin(2 * w * t)
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, [roll, 0.0, yaw], "xyz")
    d.qpos[:3], d.qpos[3:7] = pos, q
    d.qvel[:3] = [-radius * w * np.sin(ang), radius * w * np.cos(ang), 0.24 * w * np.cos(3 * w * t)]
    d.qvel[3:6] = [0, 0, w]


class Driver:
    """Smooth motion for any position-controlled arm (around its keyframe) or a flying base."""

    def __init__(self, m, d, p, props):
        self.m, self.p = m, p
        if p.get("ur5e_scene"):
            from evk4_mujoco import ur5e_demo as demo
            self.demo = demo
            demo.reset(m, d)
            return
        self.demo = None
        mujoco.mj_resetData(m, d)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, p.get("key", 0))
        self.flying = p.get("flying", False)
        self.center = tuple(self._props_xy(m)) if props else (0.0, 0.0)
        rng = np.random.default_rng(3)
        self.base = d.ctrl.copy()
        from_qpos = not np.any(self.base)                       # keyframe without ctrl
        self.jnt = []
        for i in range(m.nu):
            if m.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT or m.actuator_biastype[i] == 0:
                continue                                        # only position servos
            j = m.actuator_trnid[i, 0]
            if from_qpos:
                self.base[i] = d.qpos[m.jnt_qposadr[j]]
            self.jnt.append((i, rng.uniform(0.12, 0.3), rng.uniform(0.25, 0.6), rng.uniform(0, 2 * np.pi)))
        if props:
            evk.start_props(m, d, props)
        if self.flying:
            _fly(m, d, 0.0, self.center)
            mujoco.mj_forward(m, d)

    @staticmethod
    def _props_xy(m):
        return m.geom("evk4_props_board").pos[:2] if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                                                      "evk4_props_board") >= 0 else (0, 0)

    def __call__(self, d):
        if self.demo is not None:
            d.ctrl[:6] = self.demo.joint_targets(d.time)
            return
        if self.flying:
            _fly(self.m, d, d.time, self.center)
            return
        t = d.time
        for i, amp, f, ph in self.jnt:
            lo, hi = self.m.actuator_ctrlrange[i]
            v = self.base[i] + amp * np.sin(2 * np.pi * f * t + ph) * min(1.0, t / 0.5)
            d.ctrl[i] = np.clip(v, lo, hi) if self.m.actuator_ctrllimited[i] else v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="panda", help="panda | gen3 | x2 | ur5e | ur5e_scene | custom")
    ap.add_argument("--xml"); ap.add_argument("--body")
    ap.add_argument("--pos", type=float, nargs=3, default=(0, 0, 0))
    ap.add_argument("--menagerie", default=None, help="folder with Menagerie robots (default robots/menagerie)")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "warp"]); ap.add_argument("--device", default=None)
    ap.add_argument("--resolution", type=int, nargs=2, default=None)
    ap.add_argument("--focal-length-mm", type=float, default=None)
    ap.add_argument("--binning", type=int, default=1)
    ap.add_argument("--timestep", type=float, default=0.001)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--record", default=None, help="write an MP4 (headless)")
    ap.add_argument("--frames-dir", default=None, help="write events + paired frames (headless)")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--frame-ms", type=float, default=20.0, help="paired-frame period")
    a = ap.parse_args()
    a._props_center = (0.5, 0, 0)

    m, p, props = build(a)
    d = mujoco.MjData(m)
    drive = Driver(m, d, p, props)
    cam = evk.Sensor(m, "evk4", backend=a.backend, device=a.device)
    mujoco.mj_forward(m, d)
    cam.update(d)
    W, H = cam.geometry()
    fx = cam.intrinsics()[0]
    print(f"{a.robot}: EVK4 {W}x{H}, fx {fx:.0f}px, FOV {np.degrees(2 * np.arctan(W / 2 / fx)):.0f}x"
          f"{np.degrees(2 * np.arctan(H / 2 / fx)):.0f} deg, backend {a.backend}, dt {m.opt.timestep * 1e3:g} ms")
    if a.record or a.frames_dir:
        headless(a, m, d, cam, drive, W, H)
    else:
        live(a, m, d, cam, drive, W, H)


def headless(a, m, d, cam, drive, W, H):
    pair = evk.PairedCamera(m, "evk4")
    writer = None
    if a.record:
        import imageio.v2 as imageio
        writer = imageio.get_writer(a.record, fps=25, quality=8, macro_block_size=1)
    if a.frames_dir:
        os.makedirs(a.frames_dir, exist_ok=True)
    every = max(1, int(round(a.frame_ms * 1e-3 / m.opt.timestep)))
    nsteps = int(round(a.seconds / m.opt.timestep))
    ef = evk.EventFrame(W, H, "dark")
    all_ev, prev, k, t0 = [], None, 0, time.time()
    for step in range(1, nsteps + 1):
        drive(d)
        mujoco.mj_step(m, d)
        cam.update(d)
        ev = cam.labeled_events()
        all_ev.append(ev.copy())
        ef.add(ev)
        if step % every:
            continue
        fr = pair.capture(d, t=cam.time())
        if prev is not None:
            flow, ok = pair.flow(prev, fr)
            if a.frames_dir:                               # flow belongs to the earlier frame
                _save(a.frames_dir, k - 1, prev, flow)
            if writer is not None:
                writer.append_data(panels(ef.rgb(), fr, flow, W, H))
        ef.reset()
        prev, k = fr, k + 1
        if k % 10 == 0:
            print(f"  t={d.time:.3f}s  frames {k}  {time.time() - t0:.1f}s wall")
    if a.frames_dir and prev is not None:
        _save(a.frames_dir, k - 1, prev, None)
        ev = np.concatenate(all_ev)
        np.save(os.path.join(a.frames_dir, "events.npy"), ev)
        print(f"wrote {len(ev)} events + {k} frames to {a.frames_dir}")
    if writer is not None:
        writer.close()
        print("wrote", a.record)
    pair.close()


def _save(folder, k, fr, flow):
    np.savez_compressed(os.path.join(folder, f"frame_{k:05d}.npz"), t=fr.t, rgb=fr.rgb, depth=fr.depth,
                        seg_geom=fr.seg_geom, seg_body=fr.seg_body, T_world_cam=fr.T_world_cam, K=fr.K,
                        dist=fr.dist, flow_to_next=flow if flow is not None else np.zeros(0))


def panels(ev_rgb, fr, flow, W, H):
    dep = fr.depth.copy()
    v = dep[dep > 0]
    lo, hi = (np.percentile(v, 1), np.percentile(v, 99)) if len(v) else (0, 1)
    g = np.clip((dep - lo) / max(1e-6, hi - lo), 0, 1)
    dep_rgb = np.stack([255 * (1 - g), 180 * np.sqrt(g * (1 - g)) * 2, 255 * g], -1).astype(np.uint8)
    dep_rgb[fr.depth <= 0] = 0
    top = np.concatenate([ev_rgb, fr.rgb], 1)
    bot = np.concatenate([dep_rgb, evk.flow_to_rgb(flow)], 1)
    img = np.concatenate([top, bot], 0)
    return img if img.shape[1] <= 1920 else img[::2, ::2]


def live(a, m, d, cam, drive, W, H):
    import mujoco.viewer
    ef = evk.EventFrame(W, H, "dark")
    win_ms, buf, next_disp = 20.0, [], 0.0
    wall0, sim0 = time.time(), d.time

    def nn(img, w, h):
        return img[(np.arange(h) * img.shape[0] // h)[:, None], (np.arange(w) * img.shape[1] // w)[None, :]]

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            with viewer.lock():
                drive(d)
                mujoco.mj_step(m, d)
                cam.update(d)
                ev = cam.events()
            if len(ev):
                buf.append(ev)
            if d.time >= next_disp:
                next_disp = d.time + 0.01
                evw = np.concatenate(buf) if buf else np.zeros(0, evk.EVENT_DTYPE)
                evw = evw[evw["t"] > d.time * 1e6 - win_ms * 1e3]
                buf = [evw]
                ef.reset(); ef.add(evw)
                vp = viewer.viewport
                if vp is not None and vp.width > 64:
                    w = int(min(640, 0.5 * vp.width, 0.5 * vp.height * W / H)); h = int(w * H / W)
                    viewer.set_images((mujoco.MjrRect(vp.left + vp.width - w - 8, vp.bottom + 8, w, h),
                                       np.ascontiguousarray(nn(ef.rgb(), w, h))))
                rtf = (d.time - sim0) / max(1e-6, time.time() - wall0)
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                                  "EVK4\nsim time\nevents\nreal-time factor",
                                  f"{W}x{H}\n{d.time:.2f} s\n{len(evw) / win_ms / 1e3:.2f} Mev/s\n{rtf:.3f}"))
                viewer.sync()


if __name__ == "__main__":
    main()
