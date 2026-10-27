"""
Fit the simulator's pixel parameters to recordings from a real EVK4.

Recommended bench protocol (uniform, matte target filling the ROI; lens and f-number as in the
MJCF; illuminance measured at the target with a lux meter):

  ramp   : slow LED brightness ramp by a known factor k (e.g. x2 over >= 0.3 s) up, then down
           -> contrast thresholds ON / OFF and their pixel-to-pixel spread
  step   : fast LED step by a known factor at several illuminance levels (e.g. 10 / 100 / 1000 lux)
           -> photoreceptor bandwidth: cutoff_hz and bw_half_lux (latency vs light)
  static : constant illumination for a few seconds at several levels
           -> background activity: noise_gain and leak_rate_hz

A JSON file lists the recordings:
  {"reflectance": 0.5, "roi": [x0, y0, x1, y1],
   "experiments": [
     {"type": "ramp",   "file": "ramp.raw",  "lux": 500,  "factor": 2.0, "t0_us": 0, "t1_us": 400000, "direction": "up"},
     {"type": "step",   "file": "s10.raw",   "lux": 10,   "factor": 2.0, "t_step_us": 100000},
     {"type": "static", "file": "d10.raw",   "lux": 10,   "t0_us": 0, "t1_us": 2000000}]}
Run  python tools/calibrate_evk4.py experiments.json  ->  fitted <config> lines.
"""
from __future__ import annotations

import json
import math
import os

import mujoco
import numpy as np

from . import load_model, EVK4
from .io import read_evt3

HERE = os.path.dirname(os.path.abspath(__file__))
CAL_XML = """<mujoco model="evk4 calibration target">
  <option timestep="0.001"/>
  <extension><plugin plugin="prophesee.sensor.evk4"/></extension>
  <visual><global bvactive="false"/></visual>
  <worldbody>
    <light name="L" type="directional" dir="0 0 -1" diffuse="1 1 1" specular="0 0 0" ambient="0 0 0" castshadow="false"/>
    <geom type="plane" size="2 2 0.1" rgba="{rho} {rho} {rho} 1"/>
    <body name="cam" pos="0 0 0.5"><camera name="c" fovy="40"/></body>
  </worldbody>
  <sensor>
    <plugin name="evk4" plugin="prophesee.sensor.evk4" objtype="camera" objname="c">
      <config key="resolution" value="{w} {h}"/>
      <config key="focal_length_mm" value="4.0"/>
      <config key="hot_pixel_fraction" value="0"/>
    </plugin>
  </sensor>
</mujoco>
"""


# ------------------------------------------------------------------ measurements
def _roi(ev, roi):
    if roi is None:
        return ev
    x0, y0, x1, y1 = roi
    k = (ev["x"] >= x0) & (ev["x"] < x1) & (ev["y"] >= y0) & (ev["y"] < y1)
    return ev[k]


def measure(ev, exp, npix, roi=None):
    ev = _roi(ev, roi)
    t = ev["t"].astype(np.int64)
    if exp["type"] == "ramp":
        sel = (t >= exp["t0_us"]) & (t < exp["t1_us"])
        e = ev[sel]
        pol = 1 if exp.get("direction", "up") == "up" else 0
        e = e[e["p"] == pol]
        pix = e["y"].astype(np.int64) * 100000 + e["x"]
        _, cnt = np.unique(pix, return_counts=True)
        cnt = np.concatenate([cnt, np.zeros(max(0, npix - len(cnt)))])
        return dict(mean=float(cnt.mean()), std=float(cnt.std()))
    if exp["type"] == "step":
        e = ev[t >= exp["t_step_us"]]
        pix = e["y"].astype(np.int64) * 100000 + e["x"]
        _, first = np.unique(pix, return_index=True)          # first event of each pixel
        lat = e["t"][first] - exp["t_step_us"]
        return dict(latency_us=float(np.median(lat)) if len(lat) else float("nan"))
    if exp["type"] == "static":
        sel = (t >= exp["t0_us"]) & (t < exp["t1_us"])
        dur = (exp["t1_us"] - exp["t0_us"]) * 1e-6
        return dict(rate_hz=float(sel.sum()) / npix / dur)
    raise ValueError(exp["type"])


# ------------------------------------------------------------------ simulation of the protocol
def simulate(exp, params, reflectance=0.5, res=(64, 36), seed=1):
    xml = CAL_XML.format(rho=reflectance, w=res[0], h=res[1])
    path = os.path.join(HERE, f".cal_{os.getpid()}.xml")
    with open(path, "w") as f:
        f.write(xml)
    try:
        cfg = dict(params)
        cfg["lux_per_unit"] = exp["lux"]
        cfg["seed"] = seed
        m = load_model(path, **cfg)
    finally:
        os.remove(path)
    d = mujoco.MjData(m)
    cam = EVK4(m)
    base = float(m.light_diffuse[0, 0])
    out = []
    if exp["type"] == "ramp":
        m.opt.timestep = 1e-3
        T0, T1 = exp["t0_us"] * 1e-6, exp["t1_us"] * 1e-6
        k = exp["factor"] if exp.get("direction", "up") == "up" else 1.0 / exp["factor"]
        mujoco.mj_forward(m, d)
        while d.time < T1 + 0.02:
            f = min(1.0, max(0.0, (d.time - T0) / (T1 - T0)))
            m.light_diffuse[0] = base * k ** f
            mujoco.mj_step(m, d); out.append(cam.events(d))
    elif exp["type"] == "step":
        m.opt.timestep = 1e-5
        mujoco.mj_forward(m, d)
        ts = exp["t_step_us"] * 1e-6
        while d.time < ts + 0.006:
            if d.time >= ts:
                m.light_diffuse[0] = base * exp["factor"]
            mujoco.mj_step(m, d); out.append(cam.events(d))
    else:
        m.opt.timestep = 1e-3
        mujoco.mj_forward(m, d)
        while d.time < exp["t1_us"] * 1e-6:
            mujoco.mj_step(m, d); out.append(cam.events(d))
    ev = np.concatenate(out) if out else np.zeros(0)
    return ev, res[0] * res[1]


# ------------------------------------------------------------------ fitting
def _nelder_mead(f, x0, step, iters=40):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
        c = np.mean(pts[:-1], 0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fcv = f(xc)
            if fcv < vals[-1]:
                pts[-1], vals[-1] = xc, fcv
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0]); vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit(measurements, experiments, reflectance=0.5, verbose=True):
    """measurements[i] = measure(...) of experiments[i] on the REAL recordings."""
    p = {}
    # 1) thresholds from slow ramps: mean count = ln(k) / theta ; spread -> mismatch
    for direction, key in (("up", "contrast_on"), ("down", "contrast_off")):
        r = [(e, mm) for e, mm in zip(experiments, measurements) if e["type"] == "ramp" and e.get("direction", "up") == direction]
        if r:
            e, mm = r[0]
            th = math.log(e["factor"]) / max(1e-6, mm["mean"])
            p[key] = th
            rel = math.sqrt(max(0.0, (mm["std"] ** 2 - 1 / 12.0))) / max(1e-6, mm["mean"])
            p["contrast_sigma"] = max(p.get("contrast_sigma", 0.0), rel * th)
    fixed = dict(p, refractory_us=50)
    # 2) bandwidth from step latencies
    steps = [(e, mm) for e, mm in zip(experiments, measurements) if e["type"] == "step"]
    if steps:
        def cost_bw(x):
            q = dict(fixed, cutoff_hz=math.exp(x[0]), bw_half_lux=math.exp(x[1]), shot_noise=0)
            c = 0.0
            for e, mm in steps:
                ev, npx = simulate(e, q, reflectance)
                lat = measure(ev, e, npx)["latency_us"]
                c += (math.log(max(lat, 1)) - math.log(max(mm["latency_us"], 1))) ** 2
            return c
        x, c = _nelder_mead(cost_bw, [math.log(2000), math.log(20)], [0.5, 0.7], iters=25)
        p["cutoff_hz"], p["bw_half_lux"] = math.exp(x[0]), math.exp(x[1])
        if verbose:
            print(f"  bandwidth fit: cutoff_hz={p['cutoff_hz']:.0f}, bw_half_lux={p['bw_half_lux']:.1f} (cost {c:.4f})")
    # 3) background activity from static recordings
    stat = [(e, mm) for e, mm in zip(experiments, measurements) if e["type"] == "static"]
    if stat:
        base = dict(p)
        def cost_noise(x):
            q = dict(base, noise_gain=math.exp(x[0]), leak_rate_hz=math.exp(x[1]))
            c = 0.0
            for e, mm in stat:
                ev, npx = simulate(e, q, reflectance)
                r = measure(ev, e, npx)["rate_hz"]
                c += (math.log(max(r, 1e-4)) - math.log(max(mm["rate_hz"], 1e-4))) ** 2
            return c
        x, c = _nelder_mead(cost_noise, [math.log(0.05), math.log(0.05)], [0.7, 0.7], iters=25)
        p["noise_gain"], p["leak_rate_hz"] = math.exp(x[0]), math.exp(x[1])
        if verbose:
            print(f"  noise fit: noise_gain={p['noise_gain']:.4f}, leak_rate_hz={p['leak_rate_hz']:.4f} (cost {c:.4f})")
    # 4) joint refinement: thresholds and noise interact (noise adds events during ramps)
    ramps = [(e, mm) for e, mm in zip(experiments, measurements) if e["type"] == "ramp"]
    if ramps and stat:
        keys = [k for k in ("contrast_on", "contrast_off", "noise_gain", "leak_rate_hz") if k in p]
        base = dict(p)
        def cost_joint(x):
            q = dict(base, **{k: math.exp(v) for k, v in zip(keys, x)})
            c = 0.0
            for e, mm in ramps:
                ev, npx = simulate(e, q, reflectance)
                c += (math.log(max(measure(ev, e, npx)["mean"], 1e-3)) - math.log(max(mm["mean"], 1e-3))) ** 2
            for e, mm in stat:
                ev, npx = simulate(e, q, reflectance)
                r = measure(ev, e, npx)["rate_hz"]
                c += (math.log(max(r, 1e-4)) - math.log(max(mm["rate_hz"], 1e-4))) ** 2
            return c
        x0 = [math.log(p[k]) for k in keys]
        x, c = _nelder_mead(cost_joint, x0, [0.15] * len(keys), iters=45)
        for k, v in zip(keys, x):
            p[k] = math.exp(v)
        if verbose:
            print("  joint refinement: " + ", ".join(f"{k}={p[k]:.4g}" for k in keys) + f" (cost {c:.5f})")
    return p


def calibrate(json_path, verbose=True):
    cfg = json.load(open(json_path))
    root = os.path.dirname(os.path.abspath(json_path))
    meas = []
    for e in cfg["experiments"]:
        ev, hdr = read_evt3(os.path.join(root, e["file"]))
        roi = cfg.get("roi")
        if roi:
            npix = (roi[2] - roi[0]) * (roi[3] - roi[1])
        else:
            w, h = [int(v) for v in hdr.get("geometry", "1280x720").split("x")]
            npix = w * h
        meas.append(measure(ev, e, npix, roi))
        if verbose:
            print(f"  {e['type']:6s} {e['file']}: {meas[-1]}")
    return fit(meas, cfg["experiments"], cfg.get("reflectance", 0.5), verbose)


def to_mjcf(params):
    return "\n".join(f'      <config key="{k}" value="{v:.6g}"/>' for k, v in params.items())
