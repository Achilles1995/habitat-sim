"""
Paired frames for the EVK4 sensor: RGB, depth, segmentation and dense optical flow rendered from
EXACTLY the event camera's viewpoint and pixel grid (same intrinsics, principal point and lens
distortion), so frame pixel (u, v) and event pixel (u, v) look at the same ray.

    cam  = evk.Sensor(m, "evk4", backend="warp")
    pair = evk.PairedCamera(m, "evk4")
    ...
    mujoco.mj_step(m, d); cam.update(d)
    fr = pair.capture(d, t=cam.time())       # RGB / depth / segmentation of the current pose
    flow, ok = pair.flow(prev_fr, fr)         # forward flow [px] prev -> current (dense GT)

How it works: MuJoCo's OpenGL renderer draws an ideal pinhole image (square pixels, centred
principal point, a little larger than needed); every sensor pixel's ray (undistorted with the same
Brown-Conrady model the sensor uses) is then looked up in it (bilinear for RGB, nearest for depth
and labels).  Ground-truth flow is computed from depth + segmentation + exact MuJoCo poses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .config import MJ2CV, camera_model, read_config, sensor_instance

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class Frame:
    t: float                    # [s] sensor time the frame belongs to
    rgb: np.ndarray | None      # (H, W, 3) uint8
    depth: np.ndarray | None    # (H, W) float32, z-depth along the optical axis [m], 0 = no surface
    seg_geom: np.ndarray | None  # (H, W) int32 MuJoCo geom id, -1 = background
    seg_body: np.ndarray | None  # (H, W) int32 body id, -1 = background
    T_world_cam: np.ndarray     # (4, 4) camera pose, OpenCV axes (x right, y down, z forward)
    geom_xpos: np.ndarray       # (ngeom, 3)   poses at capture time (for flow)
    geom_xmat: np.ndarray       # (ngeom, 3, 3)
    K: np.ndarray               # (3, 3)
    dist: np.ndarray            # (5,) k1 k2 p1 p2 k3


def _remap(img, mu, mv, nearest):
    """Sample img at float pixel coords (mu, mv); out-of-image -> 0."""
    if cv2 is not None:
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        if img.dtype == np.int32:                        # cv2.remap has no int32 -> go via float64
            return cv2.remap(img.astype(np.float64), mu, mv, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=-1).astype(np.int32)
        return cv2.remap(img, mu, mv, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    Hs, Ws = img.shape[:2]
    if nearest:
        iu = np.clip(np.rint(mu).astype(int), 0, Ws - 1); iv = np.clip(np.rint(mv).astype(int), 0, Hs - 1)
        return img[iv, iu]
    u0 = np.clip(np.floor(mu).astype(int), 0, Ws - 2); v0 = np.clip(np.floor(mv).astype(int), 0, Hs - 2)
    a = np.clip(mu - u0, 0, 1)[..., None]; b = np.clip(mv - v0, 0, 1)[..., None]
    f = img.astype(np.float32)
    if f.ndim == 2:
        f = f[..., None]
    out = ((1 - a) * (1 - b) * f[v0, u0] + a * (1 - b) * f[v0, u0 + 1] + (1 - a) * b * f[v0 + 1, u0]
           + a * b * f[v0 + 1, u0 + 1])
    out = out[..., 0] if img.ndim == 2 else out
    return out.astype(img.dtype)


class PairedCamera:
    """RGB / depth / segmentation / flow aligned pixel-for-pixel with an EVK4 sensor."""

    def __init__(self, model: mujoco.MjModel, sensor_name: str = "evk4", rgb: bool = True, depth: bool = True,
                 segmentation: bool = True, margin_px: int = 3, max_render_px: int = 4096,
                 reflections: bool = False):
        self.m = model
        self.cm = camera_model(model, sensor_name)
        self.want = dict(rgb=rgb, depth=depth, seg=segmentation)
        cm = self.cm
        self.rays = cm.pixel_rays()                                  # (H, W, 2) ideal normalised coords
        f = max(cm.fx, cm.fy)
        ex, ey = float(np.abs(self.rays[..., 0]).max()), float(np.abs(self.rays[..., 1]).max())
        Wr = 2 * int(math.ceil(f * ex + margin_px)); Hr = 2 * int(math.ceil(f * ey + margin_px))
        s = min(1.0, max_render_px / max(Wr, Hr))                  # extreme wide-angle: cap the render size
        self.f_r = f * s
        self.Wr, self.Hr = max(8, int(round(Wr * s))), max(8, int(round(Hr * s)))
        self.map_u = (0.5 * self.Wr + self.f_r * self.rays[..., 0] - 0.5).astype(np.float32)
        self.map_v = (0.5 * self.Hr + self.f_r * self.rays[..., 1] - 0.5).astype(np.float32)
        g = model.vis.global_
        g.offwidth, g.offheight = max(g.offwidth, self.Wr), max(g.offheight, self.Hr)
        self.renderer = mujoco.Renderer(model, self.Hr, self.Wr)
        # the event sensor has no mirror reflections -> keep the RGB consistent with it by default
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = bool(reflections)
        self.opt = mujoco.MjvOption()
        cfg = read_config(model, sensor_instance(model, sensor_name))
        gg = [int(float(x)) for x in str(cfg["geomgroup"]).split()] + [0] * 6
        for i in range(6):
            self.opt.geomgroup[i] = gg[i]
        self.opt.sitegroup[:] = 0                                   # sites are invisible to the sensor too
        self._range_scale = np.sqrt(1.0 + self.rays[..., 0] ** 2 + self.rays[..., 1] ** 2).astype(np.float32)

    # ------------------------------------------------------------------ rendering
    def _render(self, d, mode):
        r = self.renderer
        r.disable_depth_rendering(); r.disable_segmentation_rendering()
        if mode == "depth":
            r.enable_depth_rendering()
        elif mode == "seg":
            r.enable_segmentation_rendering()
        r.update_scene(d, camera=self.cm.cam_id, scene_option=self.opt)
        self._drop_decor(r.scene)
        for c in r.scene.camera:                                    # exact pinhole frustum (square pixels)
            n = c.frustum_near
            c.frustum_center = 0.0
            c.frustum_width = n * 0.5 * self.Wr / self.f_r
            c.frustum_bottom = -n * 0.5 * self.Hr / self.f_r
            c.frustum_top = n * 0.5 * self.Hr / self.f_r
        return r.render()

    @staticmethod
    def _drop_decor(scn):
        """Neutralise decoration drawn by plugins' visualize callbacks (the EVK4 frustum lines):
        not scene content, and older plugin builds leave their segmentation ids uninitialised."""
        cam = scn.camera[0]
        behind = np.asarray(cam.pos) - 1e3 * np.asarray(cam.forward)     # far behind the camera
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.type == int(mujoco.mjtGeom.mjGEOM_LINE) and g.category == int(mujoco.mjtCatBit.mjCAT_DECOR):
                g.pos[:] = behind; g.size[:] = 0.0; g.rgba[3] = 0.0
                g.segid = -1; g.objtype = int(mujoco.mjtObj.mjOBJ_UNKNOWN); g.objid = -1

    def camera_pose(self, d):
        cid = self.cm.cam_id
        T = np.eye(4)
        T[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ MJ2CV
        T[:3, 3] = d.cam_xpos[cid]
        return T

    def capture(self, d: mujoco.MjData, t: float | None = None) -> Frame:
        """Render the current pose.  `t`: timestamp to store (use the sensor's time: Sensor.time())."""
        mu, mv = self.map_u, self.map_v
        rgb = depth = sg = sb = None
        if self.want["seg"] or self.want["depth"]:
            seg = self._render(d, "seg")
            obj = seg[..., 0].astype(np.int32); typ = seg[..., 1]
            geom = np.where((typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (obj >= 0), obj, -1).astype(np.int32)
            sg = _remap(geom, mu, mv, nearest=True)
            sb = np.where(sg >= 0, self.m.geom_bodyid[np.maximum(sg, 0)], -1).astype(np.int32)
        if self.want["depth"]:
            z = self._render(d, "depth").astype(np.float32)
            depth = _remap(z, mu, mv, nearest=True)
            depth[sg < 0] = 0.0
        if self.want["rgb"]:
            rgb = _remap(np.ascontiguousarray(self._render(d, "rgb")), mu, mv, nearest=False)
        return Frame(t=float(d.time if t is None else t), rgb=rgb, depth=depth,
                     seg_geom=sg if self.want["seg"] else None, seg_body=sb if self.want["seg"] else None,
                     T_world_cam=self.camera_pose(d), geom_xpos=d.geom_xpos.copy(),
                     geom_xmat=d.geom_xmat.reshape(-1, 3, 3).copy(), K=self.cm.K, dist=self.cm.dist.copy())

    def range(self, fr: Frame):
        """Euclidean camera-to-surface distance [m] (the per-event 'depth' label convention)."""
        return fr.depth * self._range_scale

    # ------------------------------------------------------------------ ground-truth flow
    def points_world(self, fr: Frame):
        z = fr.depth
        Xc = np.stack([self.rays[..., 0] * z, self.rays[..., 1] * z, z], -1)
        R, p = fr.T_world_cam[:3, :3], fr.T_world_cam[:3, 3]
        return Xc @ R.T + p

    def flow(self, fa: Frame, fb: Frame):
        """Dense forward flow [px] from frame a to frame b: where the surface point seen at pixel
        (u, v) in `fa` is imaged at the time of `fb` (object + camera motion, rigid geoms, lens
        distortion included; occlusions at b are not tested).  Returns (flow (H,W,2), valid (H,W))."""
        if fa.depth is None or fa.seg_geom is None:
            raise ValueError("flow needs depth and segmentation in the first frame")
        H, W = fa.depth.shape
        ok = (fa.seg_geom >= 0) & (fa.depth > 0)
        g = fa.seg_geom[ok]
        Xw = self.points_world(fa)[ok]
        loc = np.einsum("nji,nj->ni", fa.geom_xmat[g], Xw - fa.geom_xpos[g])     # R_a^T (X - p_a)
        Xb = np.einsum("nij,nj->ni", fb.geom_xmat[g], loc) + fb.geom_xpos[g]
        Rb, pb = fb.T_world_cam[:3, :3], fb.T_world_cam[:3, 3]
        pix, front = self.cm.project((Xb - pb) @ Rb)
        vv, uu = np.nonzero(ok)
        flow = np.full((H, W, 2), np.nan, np.float32)
        flow[vv, uu, 0] = pix[:, 0] - uu
        flow[vv, uu, 1] = pix[:, 1] - vv
        valid = np.zeros((H, W), bool)
        valid[vv, uu] = front
        flow[~valid] = np.nan
        return flow, valid

    def close(self):
        self.renderer.close()


def flow_to_rgb(flow, max_mag=None):
    """Middlebury-style colour coding (hue = direction, saturation = magnitude); NaN -> black."""
    fx, fy = flow[..., 0], flow[..., 1]
    bad = ~np.isfinite(fx) | ~np.isfinite(fy)
    fx = np.where(bad, 0, fx); fy = np.where(bad, 0, fy)
    mag = np.hypot(fx, fy)
    mm = max_mag or max(1e-6, np.percentile(mag[~bad], 99) if np.any(~bad) else 1.0)
    ang = (np.arctan2(-fy, -fx) / np.pi + 1) / 2                     # 0..1
    s = np.clip(mag / mm, 0, 1)
    h6 = ang * 6
    i = np.floor(h6).astype(int) % 6
    fr = h6 - np.floor(h6)
    q, t = 1 - fr, fr
    rgb = np.select([i[..., None] == k for k in range(6)],
                    [np.stack(c, -1) for c in ((np.ones_like(t), t, 0 * t), (q, np.ones_like(t), 0 * t),
                                               (0 * t, np.ones_like(t), t), (0 * t, q, np.ones_like(t)),
                                               (t, 0 * t, np.ones_like(t)), (np.ones_like(t), 0 * t, q))])
    rgb = 1 - s[..., None] * (1 - rgb)
    rgb[bad] = 0
    return (rgb * 255).astype(np.uint8)
