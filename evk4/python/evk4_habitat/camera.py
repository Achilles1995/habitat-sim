"""
Physics-based Prophesee EVK4 event camera for Habitat-Sim.

This is the GPU backend of `evk4_mujoco` with a different photon source.  What reaches the pixel
array is computed from Habitat's own renderer instead of the MuJoCo ray caster; everything from
the photocurrent on is the original, unmodified model (the same Warp `pixel_kernel`, the same
host-side readout / ESP / ERC code and the same configuration and bias mapping):

  Habitat HDR radiance render (linear, float, no tonemap / sRGB, linearised textures)
    -> luminance  L = lux_per_unit * (w . rgb) / pi            [cd/m^2]   (w = rgb_weights)
    -> pixel area sampling + lens distortion (resampling) + cos^4 vignetting + optional flicker
    -> photocurrent  rate = k_rate * L,  k_rate = pi/(4 N^2) * 4.09e15 * pitch^2 * QE   [e-/s]
    -> PSF blur
    -> evk4_mujoco.gpu.pixel_kernel: log photoreceptor with light-dependent bandwidth (1-2
       poles), Rice-hazard shot noise, ON/OFF thresholds with mismatch, refractory, leak, hot
       pixels, latency + jitter
    -> EVK4Warp._readout / _esp / _erc: ROI, arbiter, AFK, STC/trail, ERC

Time is continuous: between two `update(t)` calls the scene (camera, robots, humans, objects)
is rendered at as many interpolated states as needed to keep the image motion below
`max_px_motion` pixels per sub-step (`max_substeps` at most), exactly like the MuJoCo backend.

Every event carries ground truth: instance id (`geom` field; Habitat semantic id - see
`SceneMotion.instances()`), range [m] and optical flow [px/s] (exact for the static scene,
robot links and rigid objects; nearest-bone rigid approximation on skinned humans).

    motion = SceneMotion(sim); motion.add_human(ao_h, urdf_path=...); motion.add_robot(spot)
    cam = EVK4Habitat(sim, spot_mount(spot, "front"), motion, name="evk4_front",
                      config=dict(resolution="640 480", bias_diff_on=10))
    while ...:
        ... move humans / robot to the state at time t ...
        cam.update(t)
        ev = cam.labeled_events()      # x, y, p, t[us], geom(instance id), depth, flow_x, flow_y
"""
from __future__ import annotations

import math
import os
import time
import weakref

import magnum as mn
import numpy as np
import warp as wp

import habitat_sim

from evk4_mujoco import EVENT_DTYPE, LABELED_DTYPE
from evk4_mujoco.gpu import (Cam, EVK4Warp, PixParams, blur_h, blur_v, gather_events, make_keys,
                             np_gauss, np_key, np_u01, pixel_kernel, project_px)

from .config import intrinsics, make_config
from .mounts import CV2GL, Mount, world_pose
from .scene import SceneMotion

wp.config.quiet = True

_EXPOSURE = weakref.WeakKeyDictionary()      # sim -> lux_per_unit shared by all cameras of a scene


# ---------------------------------------------------------------------------------- kernels
@wp.kernel
def radiance_direct_kernel(W: int, rad: wp.array(dtype=wp.vec4), depth: wp.array(dtype=float),
                           sem: wp.array(dtype=wp.int32), ctr: wp.array(dtype=wp.vec2),
                           cos4: wp.array(dtype=float), rgbw: wp.vec3, wn: float, lux: float,
                           bg_lv: float, k_rate: float, flick: float, far: float,
                           out_raw: wp.array(dtype=float), lab_g: wp.array(dtype=int),
                           lab_d: wp.array(dtype=float)):
    """Render grid == pixel grid: radiance -> photocurrent, plus centre labels."""
    i = wp.tid()
    z = depth[i]
    xy = ctr[i]
    Lv = bg_lv
    g = int(-1)
    rng = float(0.0)
    if z > 0.0 and z < far:
        c = rad[i]
        Lv = lux * wp.max(0.0, wp.dot(rgbw, wp.vec3(c[0], c[1], c[2]))) / wn / wp.pi * flick
        g = int(sem[i])
        rng = z * wp.sqrt(1.0 + xy[0] * xy[0] + xy[1] * xy[1])
    out_raw[i] = k_rate * Lv * cos4[i]
    lab_g[i] = g
    lab_d[i] = rng


@wp.func
def _bilerp_lv(rad: wp.array(dtype=wp.vec4), depth: wp.array(dtype=float), Wr: int, Hr: int,
               ur: float, vr: float, rgbw: wp.vec3, wn: float, lux: float, bg_lv: float,
               flick: float, far: float):
    u0 = int(wp.floor(ur))
    v0 = int(wp.floor(vr))
    a = ur - float(u0)
    b = vr - float(v0)
    acc = float(0.0)
    for k in range(4):
        uu = wp.clamp(u0 + k % 2, 0, Wr - 1)
        vv = wp.clamp(v0 + k / 2, 0, Hr - 1)
        wa = 1.0 - a
        if k % 2 == 1:
            wa = a
        wb = 1.0 - b
        if k / 2 == 1:
            wb = b
        w = wa * wb
        j = vv * Wr + uu
        z = depth[j]
        lv = bg_lv
        if z > 0.0 and z < far:
            c = rad[j]
            lv = lux * wp.max(0.0, wp.dot(rgbw, wp.vec3(c[0], c[1], c[2]))) / wn / wp.pi * flick
        acc = acc + w * lv
    return acc


@wp.kernel
def radiance_resample_kernel(W: int, ns: int, Wr: int, Hr: int, fr: float, cxr: float, cyr: float,
                             rad: wp.array(dtype=wp.vec4), depth: wp.array(dtype=float),
                             sem: wp.array(dtype=wp.int32), corner: wp.array(dtype=wp.vec2),
                             ctr: wp.array(dtype=wp.vec2), cos4: wp.array(dtype=float), rgbw: wp.vec3,
                             wn: float, lux: float, bg_lv: float, k_rate: float, flick: float,
                             far: float, out_raw: wp.array(dtype=float), lab_g: wp.array(dtype=int),
                             lab_d: wp.array(dtype=float)):
    """Pinhole render -> distorted / area-sampled pixel array (ns x ns stratified samples per
    pixel, bilinear in the render), plus nearest-sample labels at the pixel centre."""
    i = wp.tid()
    u = i % W
    v = i / W
    acc = float(0.0)
    for s in range(ns * ns):
        a = (float(s % ns) + 0.5) / float(ns)
        b = (float(s / ns) + 0.5) / float(ns)
        c00 = corner[v * (W + 1) + u]
        c10 = corner[v * (W + 1) + u + 1]
        c01 = corner[(v + 1) * (W + 1) + u]
        c11 = corner[(v + 1) * (W + 1) + u + 1]
        xy = (1.0 - a) * (1.0 - b) * c00 + a * (1.0 - b) * c10 + (1.0 - a) * b * c01 + a * b * c11
        acc = acc + _bilerp_lv(rad, depth, Wr, Hr, cxr + fr * xy[0], cyr + fr * xy[1], rgbw, wn, lux,
                               bg_lv, flick, far)
    out_raw[i] = k_rate * acc / float(ns * ns) * cos4[i]
    xc = ctr[i]
    uc = wp.clamp(int(wp.round(cxr + fr * xc[0])), 0, Wr - 1)
    vc = wp.clamp(int(wp.round(cyr + fr * xc[1])), 0, Hr - 1)
    j = vc * Wr + uc
    z = depth[j]
    g = int(-1)
    rng = float(0.0)
    if z > 0.0 and z < far:
        g = int(sem[j])
        rng = z * wp.sqrt(1.0 + xc[0] * xc[0] + xc[1] * xc[1])
    lab_g[i] = g
    lab_d[i] = rng


@wp.kernel
def flow_label_kernel(W: int, ctr: wp.array(dtype=wp.vec2), cam: Cam, pcam: Cam, dt: float,
                      lab_g: wp.array(dtype=int), lab_d: wp.array(dtype=float),
                      idmap: wp.array(dtype=int), rig_D: wp.array(dtype=wp.mat44),
                      hmap: wp.array(dtype=int), h_lo: wp.array(dtype=int), h_hi: wp.array(dtype=int),
                      seg_a: wp.array(dtype=wp.vec3), seg_b: wp.array(dtype=wp.vec3),
                      bone_D: wp.array(dtype=wp.mat44), out_f: wp.array(dtype=wp.vec2)):
    """Optical-flow label [px/s] of the surface point seen at each pixel centre: the point is
    carried back to the previous sub-step with the motion of the part it belongs to (static
    scene, robot link / rigid object, or the nearest bone of a human) and projected with the
    previous camera pose."""
    i = wp.tid()
    fl = wp.vec2(0.0, 0.0)
    g = lab_g[i]
    if g >= 0:
        xy = ctr[i]
        d = wp.normalize(cam.R * wp.vec3(xy[0], -xy[1], -1.0))
        P = cam.pos + lab_d[i] * d
        Q = P
        n = idmap.shape[0]
        if g < n:
            k = idmap[g]
            if k >= 0:
                Q = wp.transform_point(rig_D[k], P)
            h = hmap[g]
            if h >= 0:
                best = float(1.0e30)
                bi = int(-1)
                for bb in range(h_lo[h], h_hi[h]):
                    a = seg_a[bb]
                    ab = seg_b[bb] - a
                    t = wp.clamp(wp.dot(P - a, ab) / wp.max(1.0e-12, wp.dot(ab, ab)), 0.0, 1.0)
                    dd = wp.length(P - (a + t * ab))
                    if dd < best:
                        best = dd
                        bi = bb
                if bi >= 0:
                    Q = wp.transform_point(bone_D[bi], P)
        pa = project_px(pcam, Q)
        pb = project_px(cam, P)
        if pa[2] > 0.0 and pb[2] > 0.0:
            fl = wp.vec2((pb[0] - pa[0]) / dt, (pb[1] - pa[1]) / dt)
    out_f[i] = fl


# ---------------------------------------------------------------------------------- sensor
class EVK4Habitat:
    """One EVK4 in a Habitat scene (any number can be created; each has its own mount)."""

    # the original host-side readout / ESP / ERC / download code of the MuJoCo GPU backend
    _readout = EVK4Warp._readout
    _esp = EVK4Warp._esp
    _erc = EVK4Warp._erc
    _download = EVK4Warp._download

    def __init__(self, sim, mount: Mount, motion: SceneMotion | None = None, name: str = "evk4",
                 config: dict | None = None, device: str | None = None):
        wp.init()
        self.sim = sim
        self.mount = mount
        self.motion = motion if motion is not None else SceneMotion(sim)
        self.name = name
        self.device = dev = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
        self.cfg, self.hab = c, hab = make_config(**(config or {}))
        self.W, self.H, self.fx, self.fy, self.cx, self.cy, self.dist = intrinsics(c, hab)
        W, H = self.W, self.H
        self.N = N = W * H
        b = int(c["binning"])
        pitch = c["pixel_pitch_um"] * 1e-6 * b
        self.k_rate = math.pi / (4 * c["f_number"] ** 2) * 4.09e15 * pitch * pitch * c["qe"]
        self.r_half = self.k_rate * c["bw_half_lux"] / math.pi
        self.ns = int(c["spp_n"])
        self.rgbw = np.array([float(x) for x in str(c["rgb_weights"]).split()][:3])

        # pixel geometry (undistorted normalised coordinates, OpenCV convention) - as EVK4Warp
        uu, vv = np.meshgrid(np.arange(W + 1) - 0.5, np.arange(H + 1) - 0.5)
        corner = self._undistort((uu - self.cx) / self.fx, (vv - self.cy) / self.fy)
        uc, vc = np.meshgrid(np.arange(W), np.arange(H))
        ctr = self._undistort((uc - self.cx) / self.fx, (vc - self.cy) / self.fy)
        c2 = 1.0 / (1.0 + ctr[..., 0] ** 2 + ctr[..., 1] ** 2)
        self.ctr_np = ctr.reshape(-1, 2)
        self.corner = wp.array(corner.reshape(-1, 2).astype(np.float32), dtype=wp.vec2, device=dev)
        self.ctr = wp.array(self.ctr_np.astype(np.float32), dtype=wp.vec2, device=dev)
        self.cos4 = wp.array((c2 * c2).reshape(-1).astype(np.float32), device=dev)

        # render camera: identical to the pixel grid when possible, else a covering pinhole render
        self.direct = (self.ns == 1 and not np.any(self.dist) and abs(self.fx - self.fy) < 1e-9
                       and abs(self.cx - 0.5 * (W - 1)) < 1e-9 and abs(self.cy - 0.5 * (H - 1)) < 1e-9)
        if self.direct:
            self.Wr, self.Hr, self.fr = W, H, self.fx
        else:
            cx_ = corner[..., 0]
            cy_ = corner[..., 1]
            X = float(np.max(np.abs(cx_)))
            Y = float(np.max(np.abs(cy_)))
            self.fr = max(self.fx, self.fy) * self.ns
            self.Wr = int(math.ceil(2 * X * self.fr)) + 2
            self.Hr = int(math.ceil(2 * Y * self.fr)) + 2
        self.cxr, self.cyr = 0.5 * (self.Wr - 1), 0.5 * (self.Hr - 1)
        self.hfov_r = 2 * math.degrees(math.atan(0.5 * self.Wr / self.fr))
        self._make_sensors()

        # fixed-pattern noise (same hashes as the C++ plugin / EVK4Warp)
        seed = int(c["seed"])
        pix = np.arange(N, dtype=np.uint64)
        thon = np.maximum(0.03, c["eff_con"] + c["contrast_sigma"] * np_gauss(np_key(seed, pix, 0, 11)))
        thoff = np.maximum(0.03, c["eff_coff"] + c["contrast_sigma"] * np_gauss(np_key(seed, pix, 0, 12)))
        s = c["leak_sigma"]
        lk = c["eff_leak"] * np.exp(s * np_gauss(np_key(seed, pix, 0, 13)) - 0.5 * s * s)
        lk = np.where(np_u01(np_key(seed, pix, 0, 14)) < c["hot_pixel_fraction"], c["hot_pixel_rate_hz"], lk)
        self.thon_np, self.thoff_np = thon, thoff
        self.thon = wp.array(thon.astype(np.float32), device=dev)
        self.thoff = wp.array(thoff.astype(np.float32), device=dev)
        self.leak = wp.array(lk.astype(np.float32), device=dev)

        # pixel state + labels + event buffers (same layout as EVK4Warp)
        self.xin = wp.zeros(N, dtype=float, device=dev)
        self.L1 = wp.zeros_like(self.xin)
        self.L2 = wp.zeros_like(self.xin)
        self.ref = wp.zeros_like(self.xin)
        self.rfr = wp.zeros(N, dtype=wp.float64, device=dev)
        self.raw = wp.zeros_like(self.xin)
        self.tmp = wp.zeros_like(self.xin)
        self.rate = wp.zeros_like(self.xin)
        self.lab_g = wp.zeros(N, dtype=int, device=dev)
        self.lab_d = wp.zeros_like(self.xin)
        self.lab_f = wp.zeros(N, dtype=wp.vec2, device=dev)
        self.cap = max(4 * N, 1 << 20)
        self.counter = wp.zeros(1, dtype=int, device=dev)
        self.evf = wp.zeros((self.cap, 6), dtype=float, device=dev)
        self.evt = wp.zeros((self.cap, 2), dtype=wp.float64, device=dev)
        self.sort_k = wp.zeros(2 * self.cap, dtype=wp.int64, device=dev)
        self.sort_v = wp.zeros(2 * self.cap, dtype=wp.int32, device=dev)
        self.evs = wp.zeros((self.cap, 6), dtype=float, device=dev)
        self._pinned = False
        r = max(1, int(math.ceil(3 * c["psf_sigma_px"])))
        kk = np.exp(-0.5 * np.arange(-r, r + 1) ** 2 / max(1e-9, c["psf_sigma_px"]) ** 2)
        self.blur_r = r
        self.blur_k = wp.array((kk / kk.sum()).astype(np.float32), device=dev)
        # host-side readout state
        self.last_t = np.full(N, -1.0)
        self.last_p = np.full(N, -1, np.int8)
        self.afk_t = np.zeros(N)
        self.afk_per = np.zeros(N)
        self.afk_until = np.zeros(N)
        self.initialized = False
        self.t_prev = 0.0
        self.sample = 1
        self.pending = np.zeros(0, LABELED_DTYPE)
        self.arb_busy = 0.0
        self.erc = dict(wstart=0.0, count=0.0, keep=1.0, acc=0.0, out=0.0)
        self._events = np.zeros(0, LABELED_DTYPE)
        self.dropped_total = 0
        self._dropped_step = 0
        self.nsub = 0
        self._pp = None
        self._last_obs = None
        self.probe_np = np.zeros((0, 3))
        self.lux_per_unit = float(c["lux_per_unit"]) if hab["lux_per_unit_given"] else None
        self._timing = dict(render=0.0, readback=0.0, pixel=0.0, readout=0.0, n=0)

    # ------------------------------------------------------------------ setup
    _undistort = EVK4Warp._undistort

    def _make_sensors(self):
        self.node = self.mount.create_node(self.sim)
        self.sensors = {}
        for kind, stype in (("rad", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH),
                            ("sem", habitat_sim.SensorType.SEMANTIC)):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = f"{self.name}_{kind}"
            spec.sensor_type = stype
            spec.resolution = [self.Hr, self.Wr]
            spec.hfov = self.hfov_r
            spec.near = float(self.hab["near"])
            spec.far = float(self.hab["far"])
            spec.position = [0.0, 0.0, 0.0]
            if kind == "rad":
                spec.hdr = True
                spec.clear_color = mn.Color4(0.0, 0.0, 0.0, 0.0)
            self.sensors[kind] = self.sim.create_sensor(spec, self.node)

    def close(self):
        for k in list(self.sensors):
            try:
                self.sim.remove_sensor(self.sensors[k].uuid)
            except Exception:  # noqa: BLE001
                pass
        self.sensors = {}

    # ------------------------------------------------------------------ geometry
    def T_world_cam(self) -> np.ndarray:
        """(4,4) world pose of the camera, OpenCV convention (x right, y down, z forward)."""
        return world_pose(self.node) @ CV2GL

    def _cam_struct(self, T_gl):
        cam = Cam()
        cam.pos = wp.vec3(*T_gl[:3, 3].astype(np.float32))
        cam.R = wp.mat33(*T_gl[:3, :3].reshape(9).astype(np.float32))
        cam.fx, cam.fy, cam.cx, cam.cy = float(self.fx), float(self.fy), float(self.cx), float(self.cy)
        cam.k1, cam.k2, cam.p1, cam.p2, cam.k3 = [float(x) for x in self.dist]
        return cam

    def _project(self, T_gl, w):
        """world points (n,3) -> pixels (n,2), in-front mask; same model as the kernels."""
        R, p = T_gl[:3, :3], T_gl[:3, 3]
        c = (w - p) @ R
        z = -c[:, 2]
        ok = z > 1e-4
        z = np.where(ok, z, 1.0)
        x, y = c[:, 0] / z, -c[:, 1] / z
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([self.cx + self.fx * xd, self.cy + self.fy * yd], 1), ok

    def intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy

    def geometry(self):
        return self.W, self.H

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    # ------------------------------------------------------------------ rendering
    def _draw(self):
        t0 = time.perf_counter()
        for s in self.sensors.values():
            s.draw_observation()
        t1 = time.perf_counter()
        obs = {k: s.get_observation() for k, s in self.sensors.items()}
        self._timing["render"] += t1 - t0
        self._timing["readback"] += time.perf_counter() - t1
        return obs

    def _exposure(self, obs):
        """lux_per_unit: explicit, shared by the scene's cameras, or from the first render."""
        if self.lux_per_unit is not None:
            return self.lux_per_unit
        if self.sim in _EXPOSURE:
            self.lux_per_unit = _EXPOSURE[self.sim]
            return self.lux_per_unit
        d = obs["depth"]
        geo = (d > 0) & (d < self.hab["far"])
        Y = (obs["rad"][..., :3] @ self.rgbw) / self.rgbw.sum()
        m = float(np.mean(Y[geo])) if np.any(geo) else 0.0
        self.lux_per_unit = 0.18 * self.hab["scene_lux"] / m if m > 1e-9 else float(self.cfg["lux_per_unit"])
        _EXPOSURE[self.sim] = self.lux_per_unit
        return self.lux_per_unit

    def _render(self, t, flow_ctx=None):
        """Draw at the current scene state; photocurrent -> self.rate, labels -> lab_g/d(/f)."""
        c, dev = self.cfg, self.device
        obs = self._draw()
        lux = self._exposure(obs)
        rad = np.ascontiguousarray(obs["rad"], dtype=np.float32).reshape(-1, 4)
        depth = np.ascontiguousarray(obs["depth"], dtype=np.float32).reshape(-1)
        sem = np.ascontiguousarray(obs["sem"]).view(np.int32).reshape(-1)
        self._last_obs = obs
        fl = 1.0
        if c["flicker_hz"] > 0 and c["flicker_depth"] > 0:
            fl = 1.0 - c["flicker_depth"] * 0.5 * (1.0 + math.cos(2 * math.pi * c["flicker_hz"] * t))
        rad_w = wp.array(rad, dtype=wp.vec4, device=dev)
        dep_w = wp.array(depth, dtype=float, device=dev)
        sem_w = wp.array(sem, dtype=wp.int32, device=dev)
        rgbw = wp.vec3(*self.rgbw.astype(np.float32))
        wn = float(self.rgbw.sum())
        bg_lv = float(c["background_lux"] / math.pi)
        far = float(self.hab["far"]) * 0.999
        if self.direct:
            wp.launch(radiance_direct_kernel, dim=self.N, device=dev, inputs=[
                self.W, rad_w, dep_w, sem_w, self.ctr, self.cos4, rgbw, wn, float(lux), bg_lv,
                float(self.k_rate), float(fl), far, self.raw, self.lab_g, self.lab_d])
        else:
            wp.launch(radiance_resample_kernel, dim=self.N, device=dev, inputs=[
                self.W, self.ns, self.Wr, self.Hr, float(self.fr), float(self.cxr), float(self.cyr),
                rad_w, dep_w, sem_w, self.corner, self.ctr, self.cos4, rgbw, wn, float(lux), bg_lv,
                float(self.k_rate), float(fl), far, self.raw, self.lab_g, self.lab_d])
        if c["psf_sigma_px"] > 0.05:
            wp.launch(blur_h, dim=self.N, inputs=[self.raw, self.tmp, self.blur_k, self.blur_r, self.W], device=dev)
            wp.launch(blur_v, dim=self.N, inputs=[self.tmp, self.rate, self.blur_k, self.blur_r, self.W, self.H],
                      device=dev)
        else:
            wp.copy(self.rate, self.raw)
        if flow_ctx is not None and int(self.hab["flow_labels"]):
            self._flow(*flow_ctx)
        else:
            self.lab_f.zero_()

    def _flow(self, T_now, T_prev, parts_now, parts_prev, dt):
        dev = self.device
        rig_now, hum_now = parts_now
        rig_prev, hum_prev = parts_prev
        ids = [i for i in rig_now if i in rig_prev]
        hids = [i for i in hum_now if i in hum_prev]
        n = max([0] + ids + hids) + 1
        idmap = np.full(n, -1, np.int32)
        hmap = np.full(n, -1, np.int32)
        for k, i in enumerate(ids):
            idmap[i] = k
        D = [rig_prev[i] @ np.linalg.inv(rig_now[i]) for i in ids] or [np.eye(4)]
        lo, hi, sa, sb, BD = [], [], [], [], []
        for h, i in enumerate(hids):
            Ts, a, bb = hum_now[i]
            Tp = hum_prev[i][0]
            hmap[i] = h
            lo.append(len(sa))
            for j in range(len(Ts)):
                sa.append(a[j])
                sb.append(bb[j])
                BD.append(Tp[j] @ np.linalg.inv(Ts[j]))
            hi.append(len(sa))
        if not sa:
            sa, sb, BD = [np.zeros(3)], [np.zeros(3)], [np.eye(4)]
        if not lo:
            lo, hi = [0], [0]
        f32 = lambda x: np.ascontiguousarray(np.array(x, dtype=np.float32))
        wp.launch(flow_label_kernel, dim=self.N, device=dev, inputs=[
            self.W, self.ctr, self._cam_struct(T_now), self._cam_struct(T_prev), float(max(dt, 1e-12)),
            self.lab_g, self.lab_d, wp.array(idmap, dtype=int, device=dev),
            wp.array(f32(D), dtype=wp.mat44, device=dev), wp.array(hmap, dtype=int, device=dev),
            wp.array(np.array(lo, np.int32), dtype=int, device=dev), wp.array(np.array(hi, np.int32), dtype=int, device=dev),
            wp.array(f32(sa), dtype=wp.vec3, device=dev), wp.array(f32(sb), dtype=wp.vec3, device=dev),
            wp.array(f32(BD), dtype=wp.mat44, device=dev), self.lab_f])

    def _probe(self, T_gl):
        """Sparse static-scene world points from the last depth image (sub-step criterion)."""
        obs = self._last_obs
        d = obs["depth"]
        Hr, Wr = d.shape
        vs, us = np.meshgrid(np.linspace(0, Hr - 1, 18).astype(int), np.linspace(0, Wr - 1, 32).astype(int),
                             indexing="ij")
        z = d[vs, us].reshape(-1)
        sem = np.asarray(obs["sem"])[vs, us].reshape(-1)
        moving = set(self.motion.instances().keys()) - {0}
        ok = (z > 0) & (z < self.hab["far"]) & ~np.isin(sem, list(moving))
        x = (us.reshape(-1) - self.cxr) / self.fr
        y = (vs.reshape(-1) - self.cyr) / self.fr
        Pc = np.stack([x * z, -y * z, -z], 1)[ok]           # GL camera frame
        return Pc @ T_gl[:3, :3].T + T_gl[:3, 3]

    def _substeps(self, kp0, kp1, T0, T1, dt):
        maxd = 0.0
        if len(kp0) and len(kp0) == len(kp1):
            p0, ok0 = self._project(T0, kp0)
            p1, ok1 = self._project(T1, kp1)
            inb = lambda p: (p[:, 0] > -50) & (p[:, 0] < self.W + 50) & (p[:, 1] > -50) & (p[:, 1] < self.H + 50)
            sel = ok0 & ok1 & (inb(p0) | inb(p1))
            if sel.any():
                maxd = max(maxd, float(np.max(np.linalg.norm(p1[sel] - p0[sel], axis=1))))
        pr = self.probe_np
        if len(pr):
            p0, ok0 = self._project(T0, pr)
            p1, ok1 = self._project(T1, pr)
            sel = ok0 & ok1
            if sel.any():
                maxd = max(maxd, float(np.max(np.linalg.norm(p1[sel] - p0[sel], axis=1))))
        n = int(math.ceil(maxd / max(0.05, self.cfg["max_px_motion"])))
        if self.cfg["flicker_hz"] > 0 and self.cfg["flicker_depth"] > 0:
            n = max(n, int(math.ceil(dt * self.cfg["flicker_hz"] * 16)))
        return max(1, min(int(self.cfg["max_substeps"]), n))

    # ------------------------------------------------------------------ main update
    def update(self, t: float):
        """Advance the sensor to time `t` [s]; the scene must already be in its state at `t`."""
        c, dev = self.cfg, self.device
        S1 = self.motion.capture()
        T1 = world_pose(self.node)
        kp1 = self.motion.key_points()
        t1 = float(t)
        if (not self.initialized) or t1 < self.t_prev - 1e-12 or t1 - self.t_prev > 0.05:
            self._render(t1)
            rate = self.rate.numpy().astype(np.float64)
            x = np.log(np.maximum(0, rate) + c["dark_current"])
            u = np_u01(np_key(int(c["seed"]), np.arange(self.N, dtype=np.uint64), 0, 22))
            ref = x + self.thoff_np - (self.thon_np + self.thoff_np) * (0.05 + 0.9 * u)
            for a in (self.xin, self.L1, self.L2):
                a.assign(x.astype(np.float32))
            self.ref.assign(ref.astype(np.float32))
            self.rfr.fill_(-1.0)
            self.initialized = True
            self.t_prev = t1
            self.sample = 1
            self.pending = np.zeros(0, LABELED_DTYPE)
            self._events = np.zeros(0, LABELED_DTYPE)
            self.arb_busy = t1
            w = c["erc_window_us"] * 1e-6
            self.erc = dict(wstart=math.floor(t1 / w) * w, count=0.0, keep=1.0, acc=0.0, out=0.0)
            self.last_t[:] = -1
            self.last_p[:] = -1
            self.afk_t[:] = 0
            self.afk_per[:] = 0
            self.afk_until[:] = 0
            self.S0, self.T0, self.kp0 = S1, T1, kp1
            self.parts0 = self.motion.part_transforms()
            self.probe_np = self._probe(T1)
            self.nsub = 0
            return
        if t1 - self.t_prev < 1e-9:
            self._events = np.zeros(0, LABELED_DTYPE)
            return
        t0, DT = self.t_prev, t1 - self.t_prev
        nsub = self._substeps(self.kp0, kp1, self.T0, T1, DT)
        self.nsub = nsub
        pp = self._pp
        if pp is None:
            pp = PixParams()
            pp.dark = float(c["dark_current"])
            pp.fmax = float(c["eff_cutoff"])
            pp.r_half = float(self.r_half)
            pp.tau2 = float(1 / (2 * math.pi * c["sf_cutoff_hz"])) if c["sf_cutoff_hz"] > 0 else 0.0
            pp.sf = float(c["sf_cutoff_hz"])
            pp.gain = float(c["noise_gain"])
            pp.shot = int(c["shot_noise"])
            pp.trefr = c["eff_refr_us"] * 1e-6
            pp.lat = float(c["latency_us"])
            pp.jit = float(c["latency_jitter_us"])
            pp.seed = wp.uint64(int(c["seed"]))
            pp.W = self.W
            pp.cap = self.cap
            self._pp = pp
        self.counter.zero_()
        ta = t0
        T_a, parts_a = self.T0, self.parts0
        want_flow = bool(int(self.hab["flow_labels"]))
        for k in range(1, nsub + 1):
            f = k / nsub
            tb = t0 + f * DT
            if nsub > 1:
                if k < nsub:
                    self.motion.apply_interp(self.S0, S1, f)
                else:
                    self.motion.apply(S1)
            T_b = world_pose(self.node)
            parts_b = self.motion.part_transforms() if want_flow else ({}, {})
            self._render(tb, (T_b, T_a, parts_b, parts_a, tb - ta))
            tp = time.perf_counter()
            pp.hpf = float(1.0 - math.exp(-2 * math.pi * c["eff_hpf"] * (tb - ta))) if c["eff_hpf"] > 0 else 0.0
            pp.sample = wp.uint64(self.sample)
            wp.launch(pixel_kernel, dim=self.N, device=dev, inputs=[
                pp, wp.float64(ta), wp.float64(tb), self.rate, self.xin, self.L1, self.L2, self.ref, self.rfr,
                self.thon, self.thoff, self.leak, self.lab_g, self.lab_d, self.lab_f, self.counter,
                self.evf, self.evt])
            self._timing["pixel"] += time.perf_counter() - tp
            self.sample += 1
            ta, T_a, parts_a = tb, T_b, parts_b
        # ---- one synchronisation + packed downloads per update (as EVK4Warp.update)
        tro = time.perf_counter()
        n = int(self.counter.numpy()[0])
        if n > self.cap:
            self.dropped_total += n - self.cap
            n = self.cap
        ev = np.empty(n, LABELED_DTYPE)
        te = np.zeros(n)
        if n:
            wp.launch(make_keys, dim=n, inputs=[self.evf, self.evt, self.sort_k, self.sort_v], device=dev)
            wp.utils.radix_sort_pairs(self.sort_k, self.sort_v, n)
            wp.launch(gather_events, dim=n, inputs=[self.sort_v, self.evf, self.evs], device=dev)
            key, ef = self._download(n)
            pix = key & ((1 << 21) - 1)
            ev["t"] = key >> 21
            ev["x"] = pix % self.W
            ev["y"] = pix // self.W
            ev["p"] = ef[:, 1]
            ev["geom"] = ef[:, 2]
            ev["depth"] = ef[:, 3]
            ev["flow_x"] = ef[:, 4]
            ev["flow_y"] = ef[:, 5]
        self._events = self._readout(ev, te, t1)
        self._timing["readout"] += time.perf_counter() - tro
        self._timing["n"] += 1
        self.S0, self.T0, self.kp0, self.parts0 = S1, T1, kp1, parts_a
        self.probe_np = self._probe(T1)
        self.t_prev = t1

    # ------------------------------------------------------------------ API (same as evk4_mujoco)
    def events(self):
        e = self._events
        out = np.empty(len(e), EVENT_DTYPE)
        for k in EVENT_DTYPE.names:
            out[k] = e[k]
        return out

    def labeled_events(self):
        return self._events

    def intensity(self):
        """Photoreceptor state ln(photocurrent) (H, W)."""
        return self.L2.numpy().reshape(self.H, self.W).astype(np.float64)

    def time(self):
        return self.t_prev

    def stats(self):
        e = self._events
        return dict(n_events=len(e), n_on=int((e["p"] == 1).sum()), n_off=int((e["p"] == 0).sum()),
                    n_substeps=self.nsub, n_dropped_total=self.dropped_total, lux_per_unit=self.lux_per_unit)

    def timing(self, reset=True):
        """Mean ms per update spent in rendering, GPU->CPU readback, pixel kernel and readout."""
        n = max(1, self._timing["n"])
        out = {k: 1e3 * v / n for k, v in self._timing.items() if k != "n"}
        if reset:
            self._timing = dict(render=0.0, readback=0.0, pixel=0.0, readout=0.0, n=0)
        return out

    def frame(self):
        """Paired ground-truth frame of the last render, on the render grid (Hr, Wr):
        luminance [cd/m^2], 8-bit sRGB preview, z-depth [m], instance ids, K of the render and the
        OpenCV world pose of the camera.  On the pixel grid when `direct` (no distortion, spp 1)."""
        obs = self._last_obs
        lux = self.lux_per_unit or float(self.cfg["lux_per_unit"])
        rgb_lin = np.maximum(obs["rad"][..., :3], 0) * lux / math.pi       # cd/m^2 per channel
        lum = rgb_lin @ self.rgbw / self.rgbw.sum()
        expo = 0.18 / max(1e-6, float(np.percentile(lum, 50)) if np.any(lum > 0) else 1.0)
        v = np.clip(rgb_lin * expo, 0, None)
        v = v / (1.0 + v)                                                    # Reinhard
        srgb = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)
        Kr = np.array([[self.fr, 0, self.cxr], [0, self.fr, self.cyr], [0, 0, 1.0]])
        return dict(luminance=lum.astype(np.float32), rgb=(np.clip(srgb, 0, 1) * 255).astype(np.uint8),
                    depth=np.asarray(obs["depth"], np.float32), instance=np.asarray(obs["sem"]).astype(np.int32),
                    K=Kr, T_world_cam=self.T_world_cam(), t=self.t_prev)
