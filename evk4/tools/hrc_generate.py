#!/usr/bin/env python3
"""
Event-camera datasets for human-robot collaboration, simulated in Habitat with the physics-based
EVK4 (evk4_habitat).  Two applications:

  safety   Human-aware robot safety.  Spot drives between goals while 1-2 people cross its path,
           approach it, walk alongside, wander or stand nearby.  Ground truth per control step:
           human skeletons and velocities, robot pose and velocity, human-robot clearance,
           closest-point-of-approach, time-to-collision and a risk class (safe / caution / danger),
           so the data serves proximity estimation, intent / trajectory prediction and
           collision-risk classification.

  actions  Collaborative action recognition.  Spot waits in a room; a person performs a sequence
           of actions around it: approach, handover (reach to the robot), point, wave, pick and
           place an object, sit down / stand up on a stool, walk past, idle, and sometimes fall.
           Ground truth: frame-accurate action segments (temporal segmentation, early recognition)
           plus the same skeleton / robot streams.

Every episode is recorded by EVK4s on Spot (front) and on the wall(s); each event carries
x, y, polarity, t [us] and the simulator ground truth: instance id (human / robot link / object /
scene), range [m] and optical flow [px/s].

Output (one folder per episode):
  meta.json            scene, seed, cameras (resolution, K, distortion, mount, sensor config),
                       instance ids, class names, joint names
  events_<cam>.h5      /events/{x,y,p,t,instance,depth,flow_x,flow_y}, /pose/{t,T_world_cam}
  gt.npz               t, human_joints (T,H,J,3), human_vel (T,H,3), robot_T (T,4,4), robot_vel,
                       clearance, cpa_dist, cpa_time, ttc, risk (safety) | segments (actions)
  frames_<cam>.h5      synchronised RGB video of the same camera (default 30 fps, --rgb-every): /rgb, /t_us
                       (end of exposure, same clock as the events), /depth, /instance, /T_world_cam;
                       pixel-aligned with the events (shared optics), 8 ms exposure with motion blur
  frames_<cam>/        optional PNGs (--png-every N): rgb/*.png, depth/*.png (mm), instance/*.png
  preview_<cam>.mp4    optional (--preview): event visualisation

Usage:
  python evk4/tools/hrc_generate.py safety  --data data --episodes 10 --seconds 12 --out datasets
  python evk4/tools/hrc_generate.py actions --data data --episodes 10 --out datasets \\
      --dataset data/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json --scene 102816009
"""
from __future__ import annotations

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
from evk4_habitat import scene as evscene  # noqa: E402
from evk4_habitat import sim as evsim  # noqa: E402
from evk4_habitat.poses import Actor  # noqa: E402

TICK = 1.0 / 30.0
RISK_NAMES = ["safe", "caution", "danger"]
ACTION_NAMES = ["idle", "walk", "approach", "walk_past", "handover", "point", "wave", "pick", "place", "sit",
                "stand_up", "fall"]
# coarse classes of the application spec (point and wave share a class; walk / stand_up are transitions)
ACTION_COARSE = {"idle": "idle", "walk": "transition", "approach": "approach", "walk_past": "walk_past",
                 "handover": "handover", "point": "point_wave", "wave": "point_wave", "pick": "pick",
                 "place": "place", "sit": "sit", "stand_up": "transition", "fall": "fall"}


# ------------------------------------------------------------------------------------ output
class EventWriter:
    """Chunked, compressed HDF5 event file (appends one update at a time)."""

    FIELDS = [("x", "u2", "x"), ("y", "u2", "y"), ("p", "i1", "p"), ("t", "i8", "t"), ("instance", "i4", "geom"),
              ("depth", "f4", "depth"), ("flow_x", "f4", "flow_x"), ("flow_y", "f4", "flow_y")]

    def __init__(self, path, cam):
        import h5py
        self.f = h5py.File(path, "w")
        g = self.f.create_group("events")
        self.ds = {name: g.create_dataset(name, (0,), dtype=dt, maxshape=(None,), chunks=(1 << 16,),
                                          compression="lzf") for name, dt, _ in self.FIELDS}
        p = self.f.create_group("pose")
        self.pt = p.create_dataset("t", (0,), dtype="f8", maxshape=(None,), chunks=(1024,))
        self.pT = p.create_dataset("T_world_cam", (0, 4, 4), dtype="f8", maxshape=(None, 4, 4), chunks=(1024, 4, 4))
        a = self.f.attrs
        a["width"], a["height"] = cam.W, cam.H
        a["K"] = cam.K
        a["dist"] = cam.dist
        a["convention"] = "pixel centres at integer coordinates; camera frame OpenCV (x right, y down, z fwd)"
        self.n = 0
        self.buf = []

    def add(self, ev, t, T):
        if len(ev):
            self.buf.append(ev.copy())
        k = self.pt.shape[0]
        self.pt.resize((k + 1,))
        self.pT.resize((k + 1, 4, 4))
        self.pt[k] = t
        self.pT[k] = T
        if sum(len(b) for b in self.buf) > (1 << 20):
            self.flush()

    def flush(self):
        if not self.buf:
            return
        ev = np.concatenate(self.buf)
        self.buf = []
        m = len(ev)
        for name, _, src in self.FIELDS:
            d = self.ds[name]
            d.resize((self.n + m,))
            d[self.n:] = ev[src]
        self.n += m

    def close(self):
        self.flush()
        self.f.attrs["n_events"] = self.n
        self.f.close()


class FrameWriter:
    """Synchronised RGB (+ depth + instance) frames of one EVK4, in one HDF5 file:
    /rgb (N,H,W,3) uint8 sRGB, /t_us (N,) frame time = end of exposure, on the event clock,
    /depth (N,H,W) float16 [m, z], /instance (N,H,W) int32, /T_world_cam (N,4,4)."""

    def __init__(self, path, cam, depth=True, instance=True):
        import h5py
        self.f = h5py.File(path, "w")
        H, W = cam.Hr, cam.Wr
        mk = lambda name, shape, dt, chunk: self.f.create_dataset(
            name, (0,) + shape, dtype=dt, maxshape=(None,) + shape, chunks=(1,) + shape if chunk else None,
            compression="lzf" if chunk else None)
        self.ds = {"rgb": mk("rgb", (H, W, 3), "u1", True), "t_us": mk("t_us", (), "i8", False),
                   "T_world_cam": mk("T_world_cam", (4, 4), "f8", False)}
        if depth:
            self.ds["depth"] = mk("depth", (H, W), "f2", True)
        if instance:
            self.ds["instance"] = mk("instance", (H, W), "i4", True)
        a = self.f.attrs
        a["width"], a["height"] = W, H
        a["K"] = np.array([[cam.fr, 0, cam.cxr], [0, cam.fr, cam.cyr], [0, 0, 1.0]])
        a["exposure_s"] = cam.rgb_exposure_s
        a["pixel_aligned_with_events"] = bool(cam.direct)
        a["note"] = ("global-shutter RGB through the EVK4's optics (beam-splitter / APS-style); t_us is the end "
                     "of the exposure on the same clock as the events; motion blur from the sub-step renders")
        self.n = 0

    def add(self, fr):
        vals = {"rgb": fr["rgb"], "t_us": int(round(fr["t"] * 1e6)), "T_world_cam": fr["T_world_cam"],
                "depth": fr["depth"].astype(np.float16), "instance": fr["instance"]}
        for k, d in self.ds.items():
            d.resize((self.n + 1,) + d.shape[1:])
            d[self.n] = vals[k]
        self.n += 1

    def close(self):
        self.f.attrs["n_frames"] = self.n
        self.f.close()


def event_rgb(ev, W, H, human_ids=()):
    img = np.full((H, W, 3), (52, 37, 30), np.uint8)               # BGR
    if len(ev):
        on = ev["p"] == 1
        img[ev["y"][~on], ev["x"][~on]] = (200, 126, 64)
        img[ev["y"][on], ev["x"][on]] = (255, 255, 255)
        if len(human_ids):
            h = np.isin(ev["geom"], list(human_ids))
            img[ev["y"][h & on], ev["x"][h & on]] = (40, 170, 255)
            img[ev["y"][h & ~on], ev["x"][h & ~on]] = (0, 90, 200)
    return img


# ------------------------------------------------------------------------------------ scene helpers
def nav_point(sim, rng, near=None, radius=None):
    """Random navigable point (within `radius` of `near`, if given; same island as `near`)."""
    pf = sim.pathfinder
    pts = []
    for _ in range(64):
        pf.seed(int(rng.integers(1 << 30)))
        p = pf.get_random_navigable_point()
        if evsim.finite(p):
            pts.append(mn.Vector3(p))
    if near is None:
        return pts[0] if pts else mn.Vector3(0, 0, 0)
    c = evkh.snap(sim, near)
    ok = [p for p in pts if (p - c).length() <= radius and abs(p.y - c.y) < 0.5 and evkh.plan(sim, c, p)]
    if ok:
        return ok[int(rng.integers(len(ok)))]
    return c


def path_len(sim, a, b):
    pts = evkh.plan(sim, a, b)
    if not pts:
        return math.inf
    return sum((pts[i + 1] - pts[i]).length() for i in range(len(pts) - 1))


def wall_camera_pose(sim, center, rng, height=2.2, dmin=2.5, dmax=6.0, tries=200):
    """A ceiling-height viewpoint near a wall with a clear line of sight to `center`."""
    import habitat_sim
    pf = sim.pathfinder
    floor_y = float(center[1])
    best = None
    for _ in range(tries):
        p = nav_point(sim, rng, near=center, radius=dmax)
        d = math.hypot(p.x - center[0], p.z - center[2])
        if d < dmin:
            continue
        # push the camera to the wall behind the sampled point (cameras hang on walls)
        away = mn.Vector3(p.x - center[0], 0, p.z - center[2]).normalized()
        eye = mn.Vector3(p.x, floor_y + height, p.z)
        res = sim.cast_ray(habitat_sim.geo.Ray(np.array(eye), np.array(away)), 3.0)
        if res.has_hits():
            eye = eye + away * max(0.0, res.hits[0].ray_distance - 0.15)
        tgt = mn.Vector3(center[0], floor_y + 0.9, center[2])
        dirv = tgt - eye
        res = sim.cast_ray(habitat_sim.geo.Ray(np.array(eye), np.array(dirv.normalized())), dirv.length())
        clear = (not res.has_hits()) or res.hits[0].ray_distance > dirv.length() - 0.3
        if clear:
            return [eye.x, eye.y, eye.z], [tgt.x, tgt.y, tgt.z]
        best = best or ([eye.x, eye.y, eye.z], [tgt.x, tgt.y, tgt.z])
    return best


def spawn_box(sim, motion, pos, size, name):
    """Kinematic primitive cube of `size` (x, y, z) [m] standing on the floor at pos."""
    import habitat_sim
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    base = [h for h in otm.get_template_handles("cubeSolid")][0]
    t = otm.get_template_by_handle(base)
    t.scale = mn.Vector3(*size)
    h = f"evk4_box_{name}_{size[0]:.2f}_{size[1]:.2f}_{size[2]:.2f}"
    otm.register_template(t, h)
    obj = rom.add_object_by_template_handle(h)
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obj.translation = mn.Vector3(pos[0], pos[1] + size[1] / 2.0, pos[2])
    motion.add_object(obj, name=name)
    return obj


# ------------------------------------------------------------------------------------ episode base
class Episode:
    def __init__(self, args, app, idx, rng):
        self.args, self.app, self.idx, self.rng = args, app, idx, rng
        self.default_cams = ["spot_front", "wall0"] if app == "safety" else ["spot_head", "wall0"]
        self.dir = os.path.join(args.out, app, f"ep{idx:04d}")
        os.makedirs(self.dir, exist_ok=True)
        scene = args.scene or os.path.join(args.data, "scene_datasets/habitat-test-scenes/van-gogh-room.glb")
        if args.dataset and not os.path.exists(scene):
            import glob
            root = os.path.dirname(os.path.abspath(args.dataset))
            hits = glob.glob(os.path.join(root, "**", f"{scene}.scene_instance.json"), recursive=True)
            scene = hits[0] if hits else scene
        self.scene = scene
        self.sim = evkh.make_simulator(scene, args.dataset)
        self.motion = evkh.SceneMotion(self.sim)
        self.humans, self.actors = [], []
        self.t = 0.0
        self.gt = {k: [] for k in ("t", "human_joints", "human_root", "human_feet", "robot_T", "robot_vel")}

    def add_human(self, pos, heading, name, speed):
        names = self.args.humanoids
        who = names[int(self.rng.integers(len(names)))]
        urdf, pkl = evkh.humanoid_files(self.args.data, who)
        if not (os.path.exists(urdf) and os.path.exists(pkl)):
            urdf, pkl = evkh.humanoid_files(self.args.data, "female_2")
        h = evkh.Humanoid(self.sim, self.motion, urdf, pkl, pos, name=name, lin_speed=speed)
        h.place(pos, heading=heading)
        self.humans.append(h)
        self.actors.append(Actor(h))
        return h

    def make_cameras(self, center):
        a = self.args
        cfg = dict(resolution=a.resolution, binning=a.binning)
        if a.bias:
            cfg.update(json.loads(a.bias))
        cams = {}
        for name in a.cams or self.default_cams:
            if name.startswith("spot_"):
                cams[name] = evkh.EVK4Habitat(self.sim, evkh.spot_mount(self.robot.ao, name[5:]), self.motion,
                                              name=f"evk4_{name}", config=dict(cfg, hfov_deg=a.hfov_robot),
                                              device=a.device)
        nwall = sum(c.startswith("wall") for c in (a.cams or self.default_cams))
        for k in range(nwall):
            pose = wall_camera_pose(self.sim, center, self.rng)
            if pose is None:
                continue
            cams[f"wall{k}"] = evkh.EVK4Habitat(self.sim, evkh.wall_mount(*pose), self.motion, name=f"evk4_wall{k}",
                                                config=dict(cfg, hfov_deg=a.hfov_wall), device=a.device)
            self.wall_poses = getattr(self, "wall_poses", {})
            self.wall_poses[f"wall{k}"] = pose
        self.cams = cams
        self.writers = {k: EventWriter(os.path.join(self.dir, f"events_{k}.h5"), c) for k, c in cams.items()}
        self.frame_writers = {}
        if a.rgb_every > 0:
            for k, c in cams.items():
                c.rgb_exposure_s = a.rgb_exposure_ms * 1e-3
                self.frame_writers[k] = FrameWriter(os.path.join(self.dir, f"frames_{k}.h5"), c,
                                                    depth=not a.no_depth_frames, instance=not a.no_depth_frames)
        self.preview = {k: [] for k in cams}

    def record(self):
        """Advance all cameras to self.t and append events + ground truth."""
        a = self.args
        hid = [h.instance_id for h in self.humans]
        for k, c in self.cams.items():
            c.update(self.t)
            ev = c.labeled_events()
            self.writers[k].add(ev, self.t, c.T_world_cam())
            if a.preview:
                self.preview[k].append(event_rgb(ev, c.W, c.H, hid))
            if k in self.frame_writers and self.tick % a.rgb_every == 0:
                self.frame_writers[k].add(c.frame())
            if a.png_every and self.tick % a.png_every == 0:
                self.save_frame(k, c)
        self.gt["t"].append(self.t)
        J = []
        R = []
        for h in self.humans:
            J.append([np.array(h.ao.get_link_scene_node(l).absolute_translation) for l in h.ao.get_link_ids()])
            R.append(evscene.m4(h.ao.transformation))
        self.gt["human_joints"].append(np.array(J))
        self.gt["human_feet"].append(np.array([np.array(h.feet) for h in self.humans]))
        self.gt["human_root"].append(np.array(R))
        self.gt["robot_T"].append(evscene.m4(self.robot.ao.transformation))
        self.gt["robot_vel"].append(np.array(self.robot.forward) * self.robot.speed)

    def save_frame(self, k, c):
        import cv2
        fr = c.frame()
        d = os.path.join(self.dir, f"frames_{k}")
        for sub in ("rgb", "depth", "instance"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        name = f"{int(round(self.t * 1e6)):012d}.png"
        cv2.imwrite(os.path.join(d, "rgb", name), cv2.cvtColor(fr["rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(d, "depth", name), np.clip(fr["depth"] * 1000, 0, 65535).astype(np.uint16))
        cv2.imwrite(os.path.join(d, "instance", name), np.clip(fr["instance"], 0, 65535).astype(np.uint16))

    def finish(self, extra_gt, extra_meta):
        for w in list(self.writers.values()) + list(self.frame_writers.values()):
            w.close()
        g = {k: np.array(v) for k, v in self.gt.items()}
        g["human_vel"] = np.gradient(g["human_feet"], TICK, axis=0) if len(g["t"]) > 1 else \
            np.zeros_like(g["human_feet"])
        g.update(extra_gt)
        np.savez_compressed(os.path.join(self.dir, "gt.npz"), **g)
        h0 = self.humans[0]
        meta = dict(
            app=self.app, scene=self.scene, dataset=self.args.dataset, episode=self.idx, seed=self.args.seed,
            tick_s=TICK, duration_s=self.t,
            instances={str(k): v for k, v in self.motion.instances().items()},
            humans=[dict(name=h.name, instance_id=h.instance_id) for h in self.humans],
            joint_names=[h0.ao.get_link_name(l) for l in h0.ao.get_link_ids()],
            cameras={k: dict(width=c.W, height=c.H, K=c.K.tolist(), dist=c.dist.tolist(), mount=c.mount.name,
                             wall_pose=getattr(self, "wall_poses", {}).get(k),
                             lux_per_unit=c.lux_per_unit, n_events=self.writers[k].n,
                             config={kk: (vv if isinstance(vv, (int, float, str)) else str(vv))
                                     for kk, vv in c.cfg.items()})
                     for k, c in self.cams.items()},
        )
        meta.update(extra_meta)
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        if self.args.preview:
            try:
                import imageio
                for k, frames in self.preview.items():
                    imageio.mimsave(os.path.join(self.dir, f"preview_{k}.mp4"),
                                    [f[..., ::-1] for f in frames], fps=30, macro_block_size=1)
            except Exception as e:  # noqa: BLE001
                print("preview not written:", e)
        for c in self.cams.values():
            c.close()
        self.sim.close()


# ------------------------------------------------------------------------------------ app 1: safety
def risk_terms(hum_xz, hum_v, rob_xz, rob_v, clearance, radius=0.9):
    dp = hum_xz - rob_xz
    dv = hum_v - rob_v
    vv = float(dv @ dv)
    t_cpa = 0.0 if vv < 1e-9 else max(0.0, -float(dp @ dv) / vv)
    d_cpa = float(np.linalg.norm(dp + dv * t_cpa))
    ttc = math.inf
    if d_cpa < radius:
        # first time |dp + dv t| = radius
        b = float(dp @ dv)
        c = float(dp @ dp) - radius * radius
        disc = b * b - vv * c
        if c <= 0:
            ttc = 0.0
        elif vv > 1e-9 and disc >= 0:
            t1 = (-b - math.sqrt(disc)) / vv
            ttc = t1 if t1 >= 0 else math.inf
    if clearance < 0.5 or ttc < 1.0:
        risk = 2
    elif clearance < 1.2 or ttc < 3.0:
        risk = 1
    else:
        risk = 0
    return t_cpa, d_cpa, ttc, risk


def robot_points(robot):
    """Points on the robot body: link origins + corners of Spot's body box."""
    T = evscene.m4(robot.ao.transformation)
    pts = [np.array(robot.ao.get_link_scene_node(l).absolute_translation) for l in robot.ao.get_link_ids()]
    for x in (-0.45, 0.45):
        for y in (-0.1, 0.1):
            for z in (-0.25, 0.25):
                pts.append((T @ np.array([x, y, z, 1.0]))[:3])
    return np.array(pts)


def run_safety(args, idx, rng):
    ep = Episode(args, "safety", idx, rng)
    sim = ep.sim
    start = nav_point(sim, rng)
    goal = nav_point(sim, rng, near=start, radius=8.0)
    ep.robot = evkh.Robot(sim, ep.motion, "spot", start, data_dir=args.data, yaw=float(rng.uniform(-math.pi, math.pi)))
    ep.robot.go_to(goal)
    robot_speed = float(rng.uniform(0.5, 1.0))
    policy = str(rng.choice(["oblivious", "stop_if_close"]))
    nh = int(rng.integers(1, args.max_humans + 1))
    behaviours = ["cross", "approach", "alongside", "wander", "stand"]
    brains = []
    for i in range(nh):
        p = nav_point(sim, rng, near=start, radius=6.0)
        h = ep.add_human(p, mn.Vector3(start - p), f"human{i}", float(rng.uniform(0.7, 1.4)))
        brains.append(dict(beh=None, until=0.0))
    ep.make_cameras(np.array(start))

    def new_behaviour(i):
        h = ep.humans[i]
        b = str(rng.choice(behaviours, p=[0.35, 0.25, 0.15, 0.15, 0.10]))
        br = brains[i]
        br["beh"] = b
        rp = ep.robot.pos
        if b == "cross":
            # a point the robot will pass in ~2-4 s, crossed perpendicular to its heading
            ahead = rp + ep.robot.forward * robot_speed * float(rng.uniform(2.0, 4.0))
            side = mn.math.cross(ep.robot.forward, mn.Vector3(0, 1, 0)).normalized()
            s = float(rng.choice([-1, 1]))
            a_ = evkh.snap(sim, ahead + side * s * float(rng.uniform(2.0, 3.5)))
            b_ = evkh.snap(sim, ahead - side * s * float(rng.uniform(2.0, 3.5)))
            # walk to whichever side of the robot's path is farther away, crossing it
            h.walk_to(b_ if (b_ - h.feet).length() > (a_ - h.feet).length() else a_)
        elif b == "approach":
            h.walk_to(evkh.snap(sim, rp + (h.feet - rp).normalized() * float(rng.uniform(0.3, 1.5))))
        elif b == "alongside":
            h.walk_to(evkh.snap(sim, rp + ep.robot.forward * float(rng.uniform(3.0, 6.0))
                                + mn.math.cross(ep.robot.forward, mn.Vector3(0, 1, 0)) * float(rng.uniform(-1.2, 1.2))))
        elif b == "wander":
            h.walk_to(nav_point(sim, rng, near=h.feet, radius=5.0))
        else:
            h.path = None
        br["until"] = ep.t + float(rng.uniform(2.0, 4.0))
        br["label"] = b

    for i in range(nh):
        new_behaviour(i)
    n_ticks = int(args.seconds / TICK)
    ep.tick = 0
    beh_log, terms = [], []
    ep.record()
    for ep.tick in range(1, n_ticks + 1):
        ep.t = ep.tick * TICK
        for i, h in enumerate(ep.humans):
            if h.path:
                h.follow_path()
            else:
                h.step(walk=False)
                if ep.t > brains[i]["until"]:
                    new_behaviour(i)
            ep.actors[i].update(TICK)
        # robot: drive, optionally stop for close humans
        close = min((np.linalg.norm(np.array(h.feet - ep.robot.pos)[[0, 2]]) for h in ep.humans), default=9)
        if policy == "stop_if_close" and close < 1.0:
            ep.robot.speed = 0.0
        elif not ep.robot.follow_path(max_speed=robot_speed, dt=TICK):
            ep.robot.go_to(nav_point(sim, rng, near=ep.robot.pos, radius=8.0))
        ep.record()
        rp = robot_points(ep.robot)
        rxz = np.array(ep.robot.pos)[[0, 2]]
        rv = np.array(ep.robot.forward)[[0, 2]] * ep.robot.speed
        row = []
        for i, h in enumerate(ep.humans):
            J = ep.gt["human_joints"][-1][i]
            clr = float(np.min(np.linalg.norm(J[:, None, :] - rp[None, :, :], axis=2)))
            f1, f0 = ep.gt["human_feet"][-1][i], ep.gt["human_feet"][-2][i]
            hv = (f1 - f0) / TICK
            row.append((clr,) + risk_terms(f1[[0, 2]], hv[[0, 2]], rxz, rv, clr))
        terms.append(row)
        beh_log.append([brains[i]["label"] for i in range(nh)])
    terms = np.array(terms)                  # (T-1, H, 5)
    pad = lambda x: np.concatenate([x[:1], x], 0)
    extra = dict(clearance=pad(terms[..., 0]), cpa_time=pad(terms[..., 1]), cpa_dist=pad(terms[..., 2]),
                 ttc=pad(terms[..., 3]), risk=pad(terms[..., 4]).astype(np.int8),
                 behaviour=np.array([beh_log[0]] + beh_log, dtype="U12"))
    counts = np.bincount(extra["risk"].ravel(), minlength=3)
    ep.finish(extra, dict(risk_names=RISK_NAMES, risk_rule="danger: clearance<0.5 m or TTC<1 s; caution: "
                          "clearance<1.2 m or TTC<3 s (TTC: constant-velocity time until centre distance < 0.9 m)",
                          robot_policy=policy, robot_speed=robot_speed, behaviours=behaviours))
    return f"{nh} human(s), robot {policy}, risk safe/caution/danger ticks {counts.tolist()}"


# ------------------------------------------------------------------------------------ app 2: actions
def run_actions(args, idx, rng):
    ep = Episode(args, "actions", idx, rng)
    sim = ep.sim
    # Spot waits facing an open, walled part of the room: free line of sight over a +-25 deg cone
    # and navigable floor 1.6 m and 2.5 m in front of it (where the person will interact)
    import habitat_sim
    rpos, yaw, best = nav_point(sim, rng), 0.0, -1.0

    def free_dist(p, a):
        d = mn.Vector3(math.cos(a), 0, -math.sin(a))
        res = sim.cast_ray(habitat_sim.geo.Ray(np.array(p + mn.Vector3(0, 0.6, 0)), np.array(d)), 20.0)
        return res.hits[0].ray_distance if res.has_hits() else -1.0   # no hit: gap in the scene mesh

    for _ in range(80):
        p = nav_point(sim, rng)
        a = float(rng.uniform(-math.pi, math.pi))
        cone = [free_dist(p, a + math.radians(da)) for da in (-25, 0, 25)]
        if min(cone) < 0:
            continue
        d = mn.Vector3(math.cos(a), 0, -math.sin(a))
        on_floor = all((evkh.snap(sim, p + d * r) - (p + d * r)).length() < 0.2 for r in (1.6, 2.5))
        score = min(min(cone), 6.0) + (3.0 if on_floor else 0.0)
        if score > best:
            rpos, yaw, best = p, a, score
    ep.robot = evkh.Robot(sim, ep.motion, "spot", rpos, data_dir=args.data, yaw=yaw)
    rf = ep.robot.forward
    side = mn.math.cross(rf, mn.Vector3(0, 1, 0)).normalized()
    stage = rpos + rf * 1.8                     # where most of the interaction happens
    # props in the robot's view: an object on the floor to pick and a stool to sit on
    sgn = float(rng.choice([-1.0, 1.0]))
    box_p = evkh.snap(sim, stage + rf * 0.6 + side * sgn * float(rng.uniform(0.6, 1.2)))
    stool_p = evkh.snap(sim, stage + side * -sgn * float(rng.uniform(0.9, 1.4)))
    box = spawn_box(sim, ep.motion, box_p, (0.22, 0.22, 0.22), "box")
    stool = spawn_box(sim, ep.motion, stool_p, (0.45, 0.45, 0.45), "stool")
    hp = nav_point(sim, rng, near=rpos + rf * 3.5, radius=1.5)          # starts in view
    h = ep.add_human(hp, mn.Vector3(rpos - hp), "human0", float(rng.uniform(0.8, 1.3)))
    actor = ep.actors[0]
    ep.make_cameras(np.array(stage))
    front = lambda: evkh.snap(sim, ep.robot.pos + ep.robot.forward * float(rng.uniform(1.4, 1.9)))

    # action program: (label, kind, params)
    prog = [("idle", None)]
    pool = ["handover", "point", "wave", "pickplace", "sit", "walk_past", "idle"]
    rng.shuffle(pool)
    prog += [("approach", None)] + [(p, None) for p in pool[: int(rng.integers(3, 6))]]
    if rng.random() < args.fall_prob:
        prog.append(("fall", None))
    segments = []
    ep.tick = 0
    ep.record()

    def seg_begin(label):
        segments.append([label, ep.t, None])

    def seg_end():
        if segments and segments[-1][2] is None:
            segments[-1][2] = ep.t

    def tick(update_walk=True):
        ep.tick += 1
        ep.t = ep.tick * TICK
        if update_walk:
            if h.path:
                h.follow_path()
            else:
                h.step(walk=False)
        actor.update(TICK)
        ep.record()

    def walk(goal, label):
        if (goal - h.feet).length() < 0.3 or not h.walk_to(goal):
            return
        seg_begin(label)
        n = 0
        while h.path and n < 400:
            tick()
            n += 1
        seg_end()

    def face(p):
        d = mn.Vector3(p - h.feet)
        if d.length() > 1e-3:
            for _ in range(12):
                h.turn_to(d)
                tick(update_walk=False)

    def play(label, kind, dur, **params):
        seg_begin(label)
        actor.start(kind, dur, **params)
        while not actor.done:
            tick()
        actor.action = None
        seg_end()

    for label, _ in prog:
        if label == "idle":
            play("idle", "idle", float(rng.uniform(1.0, 2.5)))
        elif label == "approach":
            walk(front(), "approach")
            face(ep.robot.pos)
        elif label == "handover":
            walk(front(), "walk")
            face(ep.robot.pos)
            hand_pt = np.array(ep.robot.pos + ep.robot.forward * 0.45) + np.array([0, float(rng.uniform(0.7, 1.0)), 0])
            play("handover", "reach", float(rng.uniform(1.6, 2.4)), target=hand_pt, side=str(rng.choice(["right", "left"])))
        elif label == "point":
            tgt = np.array(nav_point(sim, rng, near=h.feet, radius=5.0)) + np.array([0, float(rng.uniform(0.0, 1.5)), 0])
            play("point", "point", float(rng.uniform(1.5, 2.5)), target=tgt, side=str(rng.choice(["right", "left"])))
        elif label == "wave":
            face(ep.robot.pos)
            play("wave", "wave", float(rng.uniform(1.5, 2.5)), side=str(rng.choice(["right", "left"])))
        elif label == "pickplace":
            bp = box.translation
            walk(evkh.snap(sim, bp + (h.feet - bp).normalized() * 0.45), "walk")
            face(bp)
            play("pick", "pick", 2.0, target=np.array(box.translation), obj=box)
            dest = nav_point(sim, rng, near=stage, radius=1.5)
            walk(dest, "walk")
            drop = np.array(h.feet + h.forward * 0.45) + np.array([0, 0.11, 0])
            play("place", "place", 2.0, target=drop)
            box.translation = mn.Vector3(*drop)
        elif label == "sit":
            sp = stool.translation
            stand = evkh.snap(sim, sp + (h.feet - sp).normalized() * 0.5)
            walk(stand, "walk")
            face(stand + (stand - sp))                        # back to the stool
            play("sit", "sit", 1.6)
            actor.action = "sit"
            actor.params = {}
            for _ in range(int(rng.integers(20, 60))):         # stay seated (labelled sit)
                actor.t = actor.dur
                tick(update_walk=False)
            segments[-1][2] = ep.t
            play("stand_up", "sit", 1.4, up=True)
        elif label == "walk_past":
            side = mn.math.cross(ep.robot.forward, mn.Vector3(0, 1, 0)).normalized()
            off = ep.robot.forward * float(rng.uniform(1.5, 3.0))
            a_ = evkh.snap(sim, ep.robot.pos + off + side * 3.0)
            b_ = evkh.snap(sim, ep.robot.pos + off - side * 3.0)
            walk(a_, "walk")
            walk(b_, "walk_past")
        elif label == "fall":
            play("fall", "fall", 2.5, direction=str(rng.choice(["forward", "left", "right"])))
            actor.action = "fall"
            for _ in range(30):                                # lies on the floor
                actor.t = actor.dur
                tick(update_walk=False)
            segments[-1][2] = ep.t
    seg = [(s[0], round(s[1], 4), round(s[2], 4)) for s in segments if s[2] is not None]
    # per-tick label (index into ACTION_NAMES, -1 = none)
    tl = np.array(ep.gt["t"])
    lab = np.full(len(tl), ACTION_NAMES.index("idle"), np.int16)
    for name, a0, a1 in seg:
        lab[(tl > a0) & (tl <= a1)] = ACTION_NAMES.index(name)
    extra = dict(action=lab, segments=np.array([(ACTION_NAMES.index(n), a0, a1) for n, a0, a1 in seg], np.float64))
    ep.finish(extra, dict(action_names=ACTION_NAMES, action_coarse=ACTION_COARSE,
                          segments=[dict(label=n, t0=a0, t1=a1) for n, a0, a1 in seg]))
    return f"{len(seg)} segments: " + " ".join(n for n, _, _ in seg)


# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("app", choices=["safety", "actions"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--dataset", default=None, help="scene dataset config (e.g. HSSD)")
    ap.add_argument("--scene", default=None, help="scene file / scene name in --dataset")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--first", type=int, default=0, help="index of the first episode (for parallel runs)")
    ap.add_argument("--seconds", type=float, default=12.0, help="safety episode length")
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="Warp device, e.g. cuda:0 (default: cuda if available)")
    ap.add_argument("--resolution", default="1280 720")
    ap.add_argument("--binning", type=int, default=2)
    ap.add_argument("--hfov-robot", type=float, default=90.0)
    ap.add_argument("--hfov-wall", type=float, default=80.0)
    ap.add_argument("--cams", nargs="+", default=None,
                    help="spot_front | spot_head | spot_top | spot_rear, wall0 [wall1 ...] "
                         "(default: safety spot_front wall0, actions spot_head wall0)")
    ap.add_argument("--bias", default="", help='JSON dict of EVK4 config overrides, e.g. \'{"bias_diff_on": 10}\'')
    ap.add_argument("--humanoids", nargs="+", default=["female_2", "male_1", "female_0", "male_0"])
    ap.add_argument("--max-humans", type=int, default=2)
    ap.add_argument("--fall-prob", type=float, default=0.3)
    ap.add_argument("--rgb-every", type=int, default=1,
                    help="synchronised RGB frame every N control ticks (1 = 30 fps; 0 = no RGB)")
    ap.add_argument("--rgb-exposure-ms", type=float, default=8.0, help="RGB exposure time (motion blur)")
    ap.add_argument("--no-depth-frames", action="store_true", help="RGB only in frames_<cam>.h5 (no depth / instance)")
    ap.add_argument("--png-every", type=int, default=0, help="also save RGB/depth/instance PNGs every N ticks")
    ap.add_argument("--preview", action="store_true", help="write an event-visualisation mp4 per camera")
    args = ap.parse_args()
    run = run_safety if args.app == "safety" else run_actions
    for i in range(args.first, args.first + args.episodes):
        rng = np.random.default_rng([args.seed, i])
        t0 = time.time()
        info = run(args, i, rng)
        print(f"[{args.app} ep{i:04d}] {info}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
