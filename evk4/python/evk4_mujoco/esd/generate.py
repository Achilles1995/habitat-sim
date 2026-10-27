"""
Generate ESD-style sequences (same folder layout, label ids, mask colours and .mat variable names as
the real ESD, so the same training / evaluation code reads both).

<root>/<split>/<n>_obj_<speed>speed_<traj>_<light>light_<height>height[_occ]_<seed>/
    RGB/images/<k>.png        D435 colour (640x480), auto-exposed
    RGB/masks/<k>.png         ESD colour-coded instance masks (palette of read_label.m)
    RGB/labels/<k>.png        same as label ids (uint8)
    RGB/depth/<k>.png         D435 depth, uint16 mm
    events/left.mat, right.mat
        event_labeled  N x 6 double  [x  y  p(+1/-1)  t(ns)  depth(mm, z)  label]   (as ESD)
        image_raw_ts, depth_ts, tcp_pose_ts / _position / _orientation (xyzw), tcp_vel_ts / _linear / _angular,
        K, dist, T_tcp_cam, resolution
    events/events_frame.mat       left / right: RGB frames in event-camera coordinates (K x 260 x 346 x 3)
    events/mask_events_frame.mat  left / right: label masks in event-camera coordinates (K x 260 x 346)
    events/event_count_frames.npz left / right: ON/OFF event counts per RGB interval (K x 260 x 346 x 2)
    meta.json                     attributes, objects, calibration, sensor parameters, statistics
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .. import Sensor, PairedCamera
from . import catalog, motion, rig, scene

SPEED_NAMES = {0.15: "015", 0.3: "03", 1.0: "1"}


@dataclass
class SeqParams:
    object_set: str = "esd1"           # esd1 | esd2 | random | mixed
    n_obj: int = 10
    speed: float = 0.15                # m/s
    traj: str = "linear"               # linear | rotation | linearrotation
    light: str = "normal"              # normal | low
    height: float = 0.82               # m, cameras above the table
    occlusion: bool = False
    duration: float = 3.0              # s
    seed: int = 0
    randomize_sensor: bool = False     # DAVIS346 circuit parameters from rig.DAVIS346_RANDOM
    randomize_scene: bool = False      # lux, table look, light directions, start yaw, object textures
    rgb_period: float = 0.06           # s (ESD RGB-D period)
    backend: str = "cpu"
    device: str | None = None
    robot: bool = True
    settle_s: float = 1.2
    tex_cache: str = ""
    tex_override: dict | None = None
    sensor_overrides: dict | None = None  # e.g. fitted by tools/esd_match_rate.py
    measured: str = ""                     # objects.json from tools/esd_measure_objects.py


def seq_name(p: SeqParams) -> str:
    spd = SPEED_NAMES.get(round(p.speed, 2), f"{p.speed:g}".replace(".", ""))
    light = {"normal": "good", "low": "low"}.get(p.light, str(p.light))
    return (f"{p.n_obj}_obj_{spd}speed_{p.traj}_{light}light_{int(round(p.height * 100)):03d}height"
            f"{'_occ' if p.occlusion else ''}_{p.object_set}_s{p.seed}")


def choose_objects(p: SeqParams, rng):
    if p.object_set == "esd1":
        return [catalog.CATALOG[n] for n in catalog.ESD1_ORDER[:p.n_obj]]
    if p.object_set == "esd2":
        return list(catalog.ESD2[:p.n_obj])
    if p.object_set == "random":
        return [catalog.random_object(rng, i) for i in range(p.n_obj)]
    if p.object_set == "mixed":                       # ESD-1 replicas + random distractors
        k = int(rng.integers(1, p.n_obj + 1))
        names = list(rng.choice(catalog.ESD1_ORDER, size=min(k, 10), replace=False))
        return [catalog.CATALOG[n] for n in names] + [catalog.random_object(rng, i) for i in range(p.n_obj - len(names))]
    raise ValueError(p.object_set)


def auto_expose(rgb, target, rng, noise=2.0):
    """D435-like auto exposure: gain so the median (the table) reaches `target`, soft highlight
    roll-off (Reinhard) instead of hard clipping, plus sensor noise."""
    x = rgb.astype(np.float32) / 255.0
    g = float(np.clip((target / 255.0) / max(1e-3, float(np.median(x))), 0.5, 12.0))
    v = x * g
    w2 = 2.5 ** 2
    y = v * (1 + v / w2) / (1 + v) * 1.25                  # ~ identity in the mid-tones, rolls off highlights
    y = y * 255 + rng.normal(0, noise * np.sqrt(g), y.shape)
    return np.clip(y, 0, 255).astype(np.uint8), g


def _label_image(seg, geom_label):
    lab = np.zeros(seg.shape, np.uint8)
    ok = seg >= 0
    lab[ok] = geom_label[seg[ok]]
    return lab


def _colorize(lab, palette):
    out = np.zeros(lab.shape + (3,), np.uint8)
    for k, c in palette.items():
        out[lab == k] = c
    return out


def setup(p: SeqParams, rng, tex_cache: str):
    """Build the scene of a sequence, settle the objects (sensors off), put the rig at its start pose."""
    over = dict(p.sensor_overrides or {})
    if p.randomize_sensor:                             # randomise AROUND fitted values (esd_match_rate.py)
        sensor_extra = rig.sample_sensor_params(rng)
        for k in ("contrast_on", "contrast_off"):
            if k in over:
                sensor_extra[k] *= float(over.pop(k)) / rig.DAVIS346[k]
    else:
        sensor_extra = {}
    sensor_extra.update(over)
    table = {}
    lux = p.light
    if p.randomize_scene:
        table = dict(mat_gray=float(rng.uniform(0.10, 0.35)), mat_noise=float(rng.uniform(0.0, 0.05)),
                     board_gray=float(rng.uniform(0.3, 0.6)), mat_offset=tuple(rng.uniform(-0.06, 0.06, 2)))
        lux = float(np.exp(rng.uniform(np.log(250), np.log(900)))) if p.light == "normal" else \
            float(np.exp(rng.uniform(np.log(8), np.log(60))))
    objs = choose_objects(p, rng)
    tex_override = dict(p.tex_override or {})
    if p.measured:
        objs, t = catalog.apply_measured(objs, p.measured, tex_cache)
        tex_override.update(t)
    for attempt in range(4):                           # re-place if something falls off the table
        placed = scene.place_objects(objs, rng, p.occlusion)
        cfg = scene.SceneConfig(objects=placed, light=lux, robot=p.robot, sensor_extra=sensor_extra,
                                backend=p.backend, table=table, tex_cache=tex_cache,
                                tex_override=tex_override, seed=p.seed + 1000 * attempt)
        m, info = scene.build(cfg)
        d = mujoco.MjData(m)
        yaw0 = float(rng.uniform(-np.pi, np.pi)) if p.randomize_scene else 0.0
        acc = 6.0 if p.speed >= 0.8 else 1.5
        traj = motion.ESDTrajectory(p.traj, p.speed, p.duration, rng, extent=(0.15, 0.11), acc=acc)
        drv = motion.RigDriver(m, d, info, p.height, traj, yaw0=yaw0)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        for _ in range(int(p.settle_s / m.opt.timestep)):
            drv.apply(d, 0.0)
            mujoco.mj_step(m, d)
        m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        zs = np.array([d.xpos[m.body(n).id] for n in info["objects"]])
        if np.all(zs[:, 2] > -0.005) and np.all(np.abs(zs[:, :2]) < 0.5):
            break
    drv.t0 = d.time
    drv.apply(d, 0.0)
    mujoco.mj_forward(m, d)
    return m, d, info, drv, placed, sensor_extra


def event_rate(p: SeqParams, window_s: float = 0.1, side: str = "left"):
    """Simulate a sequence WITHOUT writing anything; returns per-window event rates [ev/s]."""
    rng = np.random.default_rng(p.seed)
    m, d, info, drv, _, _ = setup(p, rng, p.tex_cache or "esd_textures")
    t0 = d.time
    sen = Sensor(m, side, backend=p.backend, device=p.device)
    sen.update(d)
    counts = np.zeros(int(np.ceil(p.duration / window_s)) + 1)
    for _ in range(int(round(p.duration / m.opt.timestep))):
        drv.apply(d, d.time - t0)
        mujoco.mj_step(m, d)
        sen.update(d)
        e = sen.events()
        if len(e):
            k = ((e["t"] * 1e-6 - t0) / window_s).astype(int)
            np.add.at(counts, np.clip(k, 0, len(counts) - 1), 1)
    return counts[:-1] / window_s


def generate(p: SeqParams, out_root: str, split: str = "train", verbose: bool = True) -> str:
    from PIL import Image
    rng = np.random.default_rng(p.seed)
    name = seq_name(p)
    root = os.path.join(out_root, split, name)
    for sub in ("RGB/images", "RGB/masks", "RGB/labels", "RGB/depth", "events"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    t_wall = time.time()
    m, d, info, drv, placed, sensor_extra = setup(p, rng, p.tex_cache or os.path.join(out_root, "_textures"))
    t0 = d.time
    sensors = {s: Sensor(m, s, backend=p.backend, device=p.device) for s in ("left", "right")}
    for s in sensors.values():
        s.update(d)
    pcs = {"d435": PairedCamera(m, "d435"), "left": PairedCamera(m, "left"), "right": PairedCamera(m, "right")}
    rays = {s: np.sqrt(1 + (pcs[s].rays ** 2).sum(-1)) for s in ("left", "right")}
    geom_label = info["geom_label"]
    H, W = pcs["left"].cm.H, pcs["left"].cm.W
    ev = {"left": [], "right": []}
    frames = {k: [] for k in ("rgb_l", "rgb_r", "mask_l", "mask_r", "cnt_l", "cnt_r")}
    img_ts, tcp = [], []
    cnt = {s: np.zeros((H, W, 2), np.int32) for s in ("left", "right")}
    target = 92.0 if p.light == "normal" else 55.0          # table brightness in the real ESD frames
    next_frame, k, rgb_gain = 0.0, 0, []
    robot_px = 0
    n_steps = int(round(p.duration / m.opt.timestep))
    for step in range(n_steps + 1):
        tau = d.time - t0
        if tau >= next_frame - 1e-9:                    # ---- RGB-D frame + aligned masks
            fr = pcs["d435"].capture(d)
            lab = _label_image(fr.seg_geom, geom_label)
            robot_px += int(np.sum((fr.seg_geom >= 0) & np.isin(m.geom_bodyid[np.maximum(fr.seg_geom, 0)],
                                                                 _robot_bodies(m))))
            img, g = auto_expose(fr.rgb, target, rng, noise=2.0 if p.light == "normal" else 5.0)
            rgb_gain.append(g)
            Image.fromarray(img).save(os.path.join(root, "RGB/images", f"{k}.png"))
            pal = catalog.ESD2_PALETTE if p.object_set == "esd2" else catalog.ESD1_PALETTE
            Image.fromarray(_colorize(lab, pal)).save(os.path.join(root, "RGB/masks", f"{k}.png"))
            Image.fromarray(lab).save(os.path.join(root, "RGB/labels", f"{k}.png"))
            Image.fromarray(np.clip(fr.depth * 1000, 0, 65535).astype(np.uint16)).save(
                os.path.join(root, "RGB/depth", f"{k}.png"))
            for s, tag in (("left", "l"), ("right", "r")):
                fe = pcs[s].capture(d)
                frames[f"rgb_{tag}"].append(auto_expose(fe.rgb, target, rng, 0.0)[0])
                frames[f"mask_{tag}"].append(_label_image(fe.seg_geom, geom_label))
                if k > 0:
                    frames[f"cnt_{tag}"].append(np.clip(cnt[s], 0, 255).astype(np.uint8))
                cnt[s][:] = 0
            img_ts.append(sensors["left"].time() - t0)        # the instant these poses belong to
            next_frame += p.rgb_period
            k += 1
        if step == n_steps:
            break
        pt, Rt = drv.apply(d, tau)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, Rt.ravel())
        tcp.append(np.concatenate([[tau], pt, q[[1, 2, 3, 0]]]))
        mujoco.mj_step(m, d)
        for s, sen in sensors.items():
            sen.update(d)
            e = sen.labeled_events()
            if len(e):
                x, y = e["x"].astype(np.int64), e["y"].astype(np.int64)
                gid = e["geom"].astype(np.int64)
                lab = np.where(gid >= 0, geom_label[np.clip(gid, 0, len(geom_label) - 1)], 0)
                z_mm = e["depth"] / rays[s][y, x] * 1000.0
                t_ns = (e["t"].astype(np.float64) * 1e-6 - t0) * 1e9
                pol = np.where(e["p"] > 0, 1.0, -1.0)
                ev[s].append(np.stack([x, y, pol, t_ns, np.where(gid >= 0, z_mm, 0.0), lab], 1))
                np.add.at(cnt[s], (y, x, (e["p"] <= 0).astype(np.int64)), 1)
    for pc in pcs.values():
        pc.close()
    # ---------------------------------------------------------------- write
    tcp = np.array(tcp)
    vel = np.gradient(tcp[:, 1:4], tcp[:, 0], axis=0) if len(tcp) > 2 else np.zeros((len(tcp), 3))
    rot = np.zeros((len(tcp), 3))
    for i in range(1, len(tcp)):
        qa, qb = tcp[i - 1, [7, 4, 5, 6]], tcp[i, [7, 4, 5, 6]]
        mujoco.mju_subQuat(rot[i], qb, qa)
        rot[i] /= max(1e-9, tcp[i, 0] - tcp[i - 1, 0])
    stats = {}
    from scipy.io import savemat
    for s in ("left", "right"):
        E = np.concatenate(ev[s]) if ev[s] else np.zeros((0, 6))
        E = E[np.argsort(E[:, 3], kind="stable")]
        c = rig.ESD_CALIB[s]
        savemat(os.path.join(root, "events", f"{s}.mat"), dict(
            event_labeled=E, image_raw_ts=np.array(img_ts) * 1e9, depth_ts=np.array(img_ts) * 1e9,
            tcp_pose_ts=tcp[:, 0] * 1e9, tcp_pose_position=tcp[:, 1:4], tcp_pose_orientation=tcp[:, 4:8],
            tcp_vel_ts=tcp[:, 0] * 1e9, tcp_vel_linear=vel, tcp_vel_angular=rot,
            K=c["K"], dist=c["dist"], T_tcp_cam=c["T"], resolution=np.array(c["res"]), time_unit="ns"),
            do_compression=True)
        stats[s] = dict(events=int(len(E)), rate_kevps=float(len(E) / p.duration / 1e3),
                        labeled_fraction=float(np.mean(E[:, 5] > 0)) if len(E) else 0.0,
                        on_fraction=float(np.mean(E[:, 2] > 0)) if len(E) else 0.0)
    savemat(os.path.join(root, "events", "events_frame.mat"),
            dict(left=np.array(frames["rgb_l"]), right=np.array(frames["rgb_r"]), ts=np.array(img_ts) * 1e9),
            do_compression=True)
    savemat(os.path.join(root, "events", "mask_events_frame.mat"),
            dict(left=np.array(frames["mask_l"]), right=np.array(frames["mask_r"]), ts=np.array(img_ts) * 1e9),
            do_compression=True)
    np.savez_compressed(os.path.join(root, "events", "event_count_frames.npz"),
                        left=np.array(frames["cnt_l"]), right=np.array(frames["cnt_r"]),
                        t_start=np.array(img_ts[:-1]), t_end=np.array(img_ts[1:]))
    meta = dict(name=name, params=asdict(p), lux=info["lux"], sensor_params=dict(rig.DAVIS346, **sensor_extra),
                objects=[dict(body=nm, name=o.name, label=info["label_of"][nm], shape=o.shape, size=o.size,
                              pose=pose, xy=[float(x), float(y)], yaw=float(yaw))
                         for nm, (o, x, y, yaw, pose, _) in zip(info["objects"], placed)],
                n_frames=k, stats=stats, robot_pixels_in_d435=robot_px, ik_max_err_mm=drv.max_err * 1e3,
                rgb_gain=float(np.mean(rgb_gain)) if rgb_gain else 1.0,
                calibration={s: dict(K=rig.ESD_CALIB[s]["K"].tolist(), dist=rig.ESD_CALIB[s]["dist"].tolist(),
                                     T_tcp_cam=rig.ESD_CALIB[s]["T"].tolist(), res=rig.ESD_CALIB[s]["res"])
                             for s in rig.ESD_CALIB},
                wall_s=time.time() - t_wall)
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    if verbose:
        print(f"{name}: {k} frames, events L {stats['left']['events']} R {stats['right']['events']} "
              f"({stats['left']['rate_kevps']:.0f} kev/s), {meta['wall_s']:.0f} s")
    return root


def _robot_bodies(m):
    names = ("base", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")
    return [m.body(n).id for n in names if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0]


# ---------------------------------------------------------------- attribute grids
def esd1_grid(object_set="esd1"):
    """All ESD-1 attribute combinations: O{2,4,6,8,10} x S{015,03,1} x M{L,R,LR} x L{NL,LL} x H{062,082},
    occlusion for > 2 objects (O2 = separated, as in the paper)."""
    out = []
    for n in (2, 4, 6, 8, 10):
        for v in (0.15, 0.3, 1.0):
            for tr in motion.TRAJS:
                for li in ("normal", "low"):
                    for h in (0.82, 0.62):
                        for occ in ((False,) if n == 2 else (False, True)):
                            out.append(SeqParams(object_set=object_set, n_obj=n, speed=v, traj=tr, light=li,
                                                 height=h, occlusion=occ))
    return out


def esd2_grid(object_set="esd2"):
    """ESD-2 attributes: O{2,5} x S{015,1} x M{L,LR} x L{NL,LL} x H{062,082}."""
    out = []
    for n in (2, 5):
        for v in (0.15, 1.0):
            for tr in ("linear", "linearrotation"):
                for li in ("normal", "low"):
                    for h in (0.82, 0.62):
                        out.append(SeqParams(object_set=object_set, n_obj=n, speed=v, traj=tr, light=li, height=h,
                                             occlusion=n > 2))
    return out
