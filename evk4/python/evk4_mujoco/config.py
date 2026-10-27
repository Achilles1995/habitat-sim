"""
Plugin configuration + camera model, shared by the CPU plugin accessors, the GPU backend and the
paired RGB / depth / segmentation / flow renderer.  No Warp dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

# must match the attribute list registered by the C++ plugin (same order)
ATTRS = [
    "resolution", "binning", "pixel_pitch_um", "focal_length_mm", "f_number", "qe", "fx", "fy",
    "cx", "cy", "distortion", "spp", "psf_sigma_px", "specular_scale", "rgb_weights",
    "lux_per_unit", "background_lux", "dark_current", "shadows", "cutoff_hz", "bw_half_lux",
    "sf_cutoff_hz", "shot_noise", "noise_gain", "contrast_on", "contrast_off", "contrast_sigma",
    "refractory_us", "hpf_hz", "leak_rate_hz", "leak_sigma", "temperature_c", "leak_doubling_c",
    "hot_pixel_fraction", "hot_pixel_rate_hz", "bias_file", "bias_diff_on", "bias_diff_off",
    "bias_fo", "bias_hpf", "bias_refr", "bias_diff_scale", "bias_fo_scale",
    "bias_hpf_hz_per_unit", "bias_refr_scale", "latency_us", "latency_jitter_us",
    "readout_mevps", "erc_mevps", "erc_window_us", "afk", "afk_min_hz", "afk_max_hz", "stc_us",
    "trail_us", "roi", "max_events", "flicker_hz", "flicker_depth", "max_px_motion",
    "max_substeps", "export_intensity", "show_frustum", "nthread", "skip_static_tiles", "seed",
    "geomgroup", "backend"]

DEFAULTS = dict(
    resolution="1280 720", binning=1, pixel_pitch_um=4.86, focal_length_mm=0, f_number=2.0, qe=0.5,
    fx=0, fy=0, cx=-1, cy=-1, distortion="0 0 0 0 0", spp=1, psf_sigma_px=0.5, specular_scale=1.0,
    rgb_weights="0.2126 0.7152 0.0722", lux_per_unit=1000.0, background_lux=50.0, dark_current=30.0,
    shadows=1, cutoff_hz=2000.0, bw_half_lux=20.0, sf_cutoff_hz=0.0, shot_noise=1, noise_gain=0.05,
    contrast_on=0.22, contrast_off=0.22, contrast_sigma=0.02, refractory_us=50.0, hpf_hz=0.0,
    leak_rate_hz=0.05, leak_sigma=0.5, temperature_c=25.0, leak_doubling_c=7.0,
    hot_pixel_fraction=5e-5, hot_pixel_rate_hz=20.0, bias_file="", bias_diff_on=0, bias_diff_off=0,
    bias_fo=0, bias_hpf=0, bias_refr=0, bias_diff_scale=100.0, bias_fo_scale=40.0,
    bias_hpf_hz_per_unit=0.5, bias_refr_scale=80.0, latency_us=20.0, latency_jitter_us=5.0,
    readout_mevps=0.0, erc_mevps=0.0, erc_window_us=200.0, afk=0, afk_min_hz=40, afk_max_hz=500,
    stc_us=0, trail_us=0, roi="0 0 0 0", max_events=400000, flicker_hz=0.0, flicker_depth=0.0,
    max_px_motion=1.0, max_substeps=12, export_intensity=1, show_frustum=1, nthread=0,
    skip_static_tiles=1, seed=1, geomgroup="1 1 1 0 0 0", backend="cpu")

_STR = {"resolution", "distortion", "rgb_weights", "bias_file", "roi", "geomgroup", "backend"}

def read_config(model: mujoco.MjModel, instance: int) -> dict:
    """Plugin <config> of an instance, with defaults, bias file and bias mapping applied."""
    start = int(model.plugin_attradr[instance])
    end = int(model.plugin_attradr[instance + 1]) if instance + 1 < model.nplugin else len(model.plugin_attr)
    vals = bytes(model.plugin_attr[start:end]).split(b"\0")
    c = dict(DEFAULTS)
    for k, v in zip(ATTRS, vals):
        v = v.decode()
        if v:
            c[k] = v if k in _STR else float(v)
    if c["bias_file"]:
        with open(c["bias_file"]) as f:
            for line in f:
                if "%" in line:
                    val, name = line.split("%", 1)
                    name = name.strip()
                    if name in ("bias_diff_on", "bias_diff_off", "bias_fo", "bias_hpf", "bias_refr"):
                        c[name] = float(val)
        for k in ("bias_diff_on", "bias_diff_off", "bias_fo", "bias_hpf", "bias_refr"):  # explicit keys win
            idx = ATTRS.index(k)
            if idx < len(vals) and vals[idx]:
                c[k] = float(vals[idx])
    c["eff_con"] = c["contrast_on"] * math.exp(c["bias_diff_on"] / c["bias_diff_scale"])
    c["eff_coff"] = c["contrast_off"] * math.exp(c["bias_diff_off"] / c["bias_diff_scale"])
    c["eff_cutoff"] = c["cutoff_hz"] * math.exp(c["bias_fo"] / c["bias_fo_scale"])
    c["eff_hpf"] = max(0.0, c["hpf_hz"] + c["bias_hpf_hz_per_unit"] * c["bias_hpf"])
    c["eff_refr_us"] = c["refractory_us"] * math.exp(-c["bias_refr"] / c["bias_refr_scale"])
    c["eff_leak"] = c["leak_rate_hz"] * 2.0 ** ((c["temperature_c"] - 25.0) / c["leak_doubling_c"])
    n = int(round(math.sqrt(max(1, int(c["spp"])))))
    c["spp_n"] = max(1, min(4, n))
    return c


def sensor_instance(model: mujoco.MjModel, sensor_name: str = "evk4") -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid < 0:
        raise KeyError(f"sensor '{sensor_name}' not found")
    return int(model.sensor_plugin[sid])


@dataclass
class CameraModel:
    """Pinhole + Brown-Conrady (OpenCV) model of the simulated array, exactly as the sensor uses it.
    Pixel centres are integer coordinates; camera frame: x right, y down, z forward (OpenCV)."""
    W: int
    H: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray            # k1 k2 p1 p2 k3
    cam_id: int
    cam_body: int

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def undistort(self, xd, yd):
        """distorted normalised coords -> ideal normalised coords (same iteration as the sensor)."""
        k1, k2, p1, p2, k3 = self.dist
        x, y = np.array(xd, float).copy(), np.array(yd, float).copy()
        if np.any(self.dist != 0):
            for _ in range(20):
                r2 = x * x + y * y
                rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
                dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
                dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
                x, y = (xd - dx) / rad, (yd - dy) / rad
        return x, y

    def pixel_rays(self):
        """(H, W, 2) ideal normalised coordinates (x/z, y/z) of every pixel centre."""
        uu, vv = np.meshgrid(np.arange(self.W, dtype=float), np.arange(self.H, dtype=float))
        x, y = self.undistort((uu - self.cx) / self.fx, (vv - self.cy) / self.fy)
        return np.stack([x, y], -1)

    def project(self, Xc):
        """(N,3) points in the OpenCV camera frame -> (N,2) pixels, (N,) in-front mask."""
        Xc = np.asarray(Xc, float)
        z = Xc[:, 2]
        ok = z > 1e-6
        zs = np.where(ok, z, 1.0)
        x, y = Xc[:, 0] / zs, Xc[:, 1] / zs
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([self.cx + self.fx * xd, self.cy + self.fy * yd], 1), ok


# MuJoCo camera frame (x right, y up, looking along -z)  ->  OpenCV frame (x right, y down, +z)
MJ2CV = np.diag([1.0, -1.0, -1.0])


def camera_model(model: mujoco.MjModel, sensor_name: str = "evk4") -> CameraModel:
    """Intrinsics of an EVK4 sensor from its MJCF config (same rules as the C++ plugin)."""
    inst = sensor_instance(model, sensor_name)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    c = read_config(model, inst)
    cam = int(model.sensor_objid[sid])
    b = int(c["binning"])
    res = [int(float(x)) for x in str(c["resolution"]).split()]
    W, H = res[0] // b, res[1] // b
    pitch = c["pixel_pitch_um"] * 1e-6 * b
    if c["fx"] > 0:
        fx = c["fx"] / b; fy = (c["fy"] if c["fy"] > 0 else c["fx"]) / b
    elif c["focal_length_mm"] > 0:
        fx = fy = c["focal_length_mm"] * 1e-3 / pitch
    else:
        fx = fy = 0.5 * H / math.tan(0.5 * math.radians(model.cam_fovy[cam]))
    cx = (c["cx"] + 0.5) / b - 0.5 if c["cx"] >= 0 else 0.5 * (W - 1)
    cy = (c["cy"] + 0.5) / b - 0.5 if c["cy"] >= 0 else 0.5 * (H - 1)
    dist = np.array([float(x) for x in str(c["distortion"]).split()] + [0] * 5)[:5]
    return CameraModel(W, H, fx, fy, cx, cy, dist, cam, int(model.cam_bodyid[cam]))
