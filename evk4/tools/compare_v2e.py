#!/usr/bin/env python3
"""
Physics-based EVK4 (evk4_habitat) vs. a video-to-event converter (v2e) on the *same* episode.

The episode's synchronised RGB video (frames_<cam>.h5: 30 fps, 8-bit sRGB, the input a video
converter gets) is converted to events with v2e, using the same contrast thresholds, threshold
mismatch, photoreceptor cutoff, refractory period and leak as our sensor.  Both event streams are
then compared on the same pixels, time window and ground truth:

  figures
    fig1_event_frames.png    RGB frame | our events | v2e events, in short windows at the
                             fastest moments of the episode (e.g. a wave, a fall)
    fig2_space_time.png      x-t slice through the person: our events form continuous streaks
                             in time, converted events form steps at the video frame times
    fig3_xyt_cloud.png       3-D (x, y, t) event cloud of the person over ~200 ms
    fig4_timestamp_phase.png where event times fall inside the video frame interval
    fig5_rate_vs_motion.png  event rate on the person vs. the person's true image speed
  metrics.json / summary.md
    event counts and rates; timestamp-phase uniformity (KS distance to uniform; 0 = events
    spread evenly in time); fraction of events sharing a timestamp; fraction of events near
    each frame time that lie on the true person mask; correlation of the person's event rate
    with its true image speed

Set-up (once):
  git clone https://github.com/SensorsINI/v2e
  pip install matplotlib            (torch, opencv and h5py are already installed)
  optional, v2e's recommended mode with SuperSloMo frame interpolation: download the
  SuperSloMo39.ckpt checkpoint linked in the v2e README and pass --slomo-model

Usage:
  python evk4/tools/compare_v2e.py datasets/actions/ep0000 --cam wall0 --v2e-root v2e --out cmp_wall0
  python evk4/tools/compare_v2e.py ... --t0 2 --t1 8                 (time window [s] to compare)
  python evk4/tools/compare_v2e.py ... --slomo-model SuperSloMo39.ckpt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i1"), ("t", "<i8"), ("instance", "<i4")])


# ------------------------------------------------------------------------------------ loading
class EventFile:
    """Time-window reads from an events_<cam>.h5 file (events are stored in time order)."""

    def __init__(self, path):
        import h5py
        self.f = h5py.File(path, "r")
        self.W, self.H = int(self.f.attrs["width"]), int(self.f.attrs["height"])
        self.t = self.f["events"]["t"]
        self.n = self.t.shape[0]

    def find(self, tq):
        lo, hi = 0, self.n
        while hi - lo > 4096:
            mid = (lo + hi) // 2
            if self.t[mid] <= tq:
                lo = mid
            else:
                hi = mid
        return lo + int(np.searchsorted(self.t[lo:hi], tq, side="right"))

    def window(self, t0_us, t1_us):
        a, b = self.find(t0_us), self.find(t1_us)
        e = self.f["events"]
        out = np.empty(b - a, EV_DTYPE)
        for k in ("x", "y", "p", "t", "instance"):
            out[k] = e[k][a:b]
        return out


def load_frames(path, t0, t1):
    """RGB / instance frames with exposure end in [t0 - 1 s, t1] (1 s of warm-up for v2e)."""
    import h5py
    with h5py.File(path, "r") as f:
        t = f["t_us"][:] * 1e-6
        sel = np.nonzero((t >= t0 - 1.0) & (t <= t1 + 1e-6))[0]
        rgb = f["rgb"][sel[0]:sel[-1] + 1]
        inst = f["instance"][sel[0]:sel[-1] + 1] if "instance" in f else None
        aligned = bool(f.attrs.get("pixel_aligned_with_events", True))
        exposure = float(f.attrs.get("exposure_s", 0.0))
    return t[sel], rgb, inst, aligned, exposure


# ------------------------------------------------------------------------------------ v2e
def _stub_optional_modules():
    """v2e's emulator module imports GUI / file-format helpers it does not need for emulation;
    stand-ins are installed only for those that are missing."""
    def stub(name, **attrs):
        if name in sys.modules:
            return
        try:
            __import__(name)
            return
        except Exception:  # noqa: BLE001
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m

    class EngNumber(float):
        def __new__(cls, v, *a, **k):
            return float.__new__(cls, float(v))

    stub("screeninfo", get_monitors=lambda: [])
    stub("easygui")
    stub("dv_processing")
    stub("engineering_notation", EngNumber=EngNumber)
    stub("plyer")
    stub("numba", njit=lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f)))
    try:
        import tkinter  # noqa: F401
    except Exception:  # noqa: BLE001
        tk = types.ModuleType("tkinter")
        tk.filedialog = types.ModuleType("tkinter.filedialog")
        sys.modules["tkinter"] = tk
        sys.modules["tkinter.filedialog"] = tk.filedialog


def v2e_params(cfg, shot_noise_hz):
    """v2e emulator parameters equal to our sensor's (effective values after the bias mapping)."""
    return dict(pos_thres=float(cfg["eff_con"]), neg_thres=float(cfg["eff_coff"]),
                sigma_thres=float(cfg["contrast_sigma"]), cutoff_hz=float(cfg["eff_cutoff"]),
                leak_rate_hz=float(cfg["eff_leak"]), refractory_period_s=float(cfg["eff_refr_us"]) * 1e-6,
                shot_noise_rate_hz=float(shot_noise_hz))


def run_v2e(times, rgb, params, v2e_root, device, slomo_model=None, workdir=None):
    """Events (EV_DTYPE, instance = -2: unknown) from the RGB video with v2e's EventEmulator."""
    import cv2
    sys.path.insert(0, os.path.abspath(v2e_root))
    _stub_optional_modules()
    import torch
    from v2ecore.emulator import EventEmulator
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = rgb.shape[1:3]
    gray = [cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY) for f in rgb]
    t_in = np.asarray(times, np.float64)
    if slomo_model:
        # v2e's recommended mode: SuperSloMo interpolation (automatic up-sampling to <= 1 px motion)
        import tempfile
        from v2ecore.slomo import SuperSloMo
        src = tempfile.mkdtemp(dir=workdir)
        dst = tempfile.mkdtemp(dir=workdir)
        for i, g in enumerate(gray):
            cv2.imwrite(os.path.join(src, f"{i:06d}.png"), g)
        slomo = SuperSloMo(model=slomo_model, auto_upsample=True, upsampling_factor=1, video_path=None,
                           vid_orig=None, vid_slomo=None, preview=False, batch_size=4)
        interp, _ = slomo.interpolate(src, dst, (W, H))
        files = sorted(f for f in os.listdir(dst) if f.endswith(".png"))
        frames = [cv2.imread(os.path.join(dst, f), cv2.IMREAD_GRAYSCALE) for f in files]
        interp = np.asarray(interp, np.float64)
        f = (t_in[-1] - t_in[0]) / max(1e-9, interp.max() - interp.min())
        t_frames = t_in[0] + (interp - interp.min()) * f
        frames, t_frames = frames[:len(t_frames)], t_frames[:len(frames)]
    else:
        frames, t_frames = gray, t_in
    emu = EventEmulator(output_width=W, output_height=H, device=device, seed=1, **params)
    out = []
    for g, t in zip(frames, t_frames):
        ev = emu.generate_events(g.astype(np.float32), float(t))
        if ev is not None and len(ev):
            out.append(np.asarray(ev, np.float64))
    if not out:
        return np.zeros(0, EV_DTYPE)
    ev = np.concatenate(out)                    # [t, x, y, p(+1/-1)]
    res = np.empty(len(ev), EV_DTYPE)
    res["t"] = np.rint(ev[:, 0] * 1e6).astype(np.int64)
    res["x"] = ev[:, 1].astype(np.int64)
    res["y"] = ev[:, 2].astype(np.int64)
    res["p"] = (ev[:, 3] > 0).astype(np.int8)
    res["instance"] = -2
    return res[np.argsort(res["t"], kind="stable")]


# ------------------------------------------------------------------------------------ metrics
def phase_of(t_us, frame_t_us):
    """Position of each event inside its video frame interval, 0..1."""
    k = np.clip(np.searchsorted(frame_t_us, t_us, side="right") - 1, 0, len(frame_t_us) - 2)
    T = frame_t_us[k + 1] - frame_t_us[k]
    return np.clip((t_us - frame_t_us[k]) / np.maximum(T, 1), 0, 1)


def ks_uniform(ph):
    if len(ph) == 0:
        return float("nan")
    s = np.sort(ph)
    n = len(s)
    return float(max(np.max(np.arange(1, n + 1) / n - s), np.max(s - np.arange(n) / n)))


def near_frame_fraction(t_us, frame_t_us, half_us=500):
    """Fraction of events within +-0.5 ms of a video frame time (uniform in time: 1 ms / frame interval)."""
    if len(t_us) == 0:
        return float("nan")
    k = np.clip(np.searchsorted(frame_t_us, t_us), 1, len(frame_t_us) - 1)
    d = np.minimum(np.abs(t_us - frame_t_us[k - 1]), np.abs(frame_t_us[k] - t_us))
    return float(np.mean(d <= half_us))


def on_mask_fraction(ev, frame_t_us, inst, ids, half_ms=4.0, dilate=2):
    """Events within +-half_ms of a frame time that fall on the (dilated) person mask of that frame,
    among all events in those windows that fall inside the person's bounding box (+20 px)."""
    import cv2
    hit = tot = 0
    k_ = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    for i, tf in enumerate(frame_t_us):
        m = np.isin(inst[i], ids)
        if not m.any():
            continue
        sel = np.abs(ev["t"] - tf) <= half_ms * 1e3
        if not sel.any():
            continue
        ys, xs = np.nonzero(m)
        x0, x1 = max(0, xs.min() - 20), xs.max() + 20
        y0, y1 = max(0, ys.min() - 20), ys.max() + 20
        x, y = ev["x"][sel].astype(np.int64), ev["y"][sel].astype(np.int64)
        box = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        md = cv2.dilate(m.astype(np.uint8), k_) > 0
        hit += int(md[y[box], x[box]].sum())
        tot += int(box.sum())
    return hit / tot if tot else float("nan")


def person_speed(ep_dir, cam_meta, frames_t, gt):
    """True image speed [px/s] of the first person's pelvis/root, at the frame times."""
    K = np.array(cam_meta["K"])
    import h5py
    with h5py.File(os.path.join(ep_dir, f"events_{cam_meta['_name']}.h5"), "r") as f:
        pt, pT = f["pose"]["t"][:], f["pose"]["T_world_cam"][:]
    tt = gt["t"]
    root = gt["human_root"][:, 0, :3, 3]                         # (T,3) world
    out = []
    for t in frames_t:
        k = int(np.clip(np.searchsorted(tt, t), 1, len(tt) - 1))
        j = int(np.clip(np.searchsorted(pt, t), 0, len(pt) - 1))
        Tcw = np.linalg.inv(pT[j])
        uv = []
        for kk in (k - 1, k):
            pc = Tcw @ np.r_[root[kk], 1.0]
            uv.append(K @ (pc[:3] / max(pc[2], 1e-3)))
        out.append(np.linalg.norm(uv[1][:2] - uv[0][:2]) / max(1e-6, tt[k] - tt[k - 1]))
    return np.array(out)


# ------------------------------------------------------------------------------------ figures
def ev_image(ev, W, H, ids=None):
    img = np.full((H, W, 3), (30, 37, 52), np.uint8)
    if len(ev):
        x, y, p = ev["x"].astype(int), ev["y"].astype(int), ev["p"]
        img[y[p == 0], x[p == 0]] = (64, 126, 200)
        img[y[p == 1], x[p == 1]] = (255, 255, 255)
    return img


def figures(out, ours, v2e, frames_t_us, rgb, inst, ids, W, H, speed, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- 1 event frames at the fastest moments
    order = np.argsort(-np.nan_to_num(speed))
    picks = []
    for i in order:
        if all(abs(i - j) > 10 for j in picks) and 1 <= i < len(frames_t_us):
            picks.append(int(i))
        if len(picks) == 3:
            break
    picks = sorted(picks) or [len(frames_t_us) // 2]
    win_us = 10_000
    fig, ax = plt.subplots(len(picks), 3, figsize=(12, 3.0 * len(picks)), squeeze=False)
    for r, i in enumerate(picks):
        tf = frames_t_us[i]
        for c, (name, ev) in enumerate((("our EVK4", ours), (labels["v2e"], v2e))):
            sel = (ev["t"] > tf - win_us) & (ev["t"] <= tf)
            ax[r, c + 1].imshow(ev_image(ev[sel], W, H))
            ax[r, c + 1].set_title(f"{name}: {int(sel.sum())} events in the last {win_us // 1000} ms")
        ax[r, 0].imshow(rgb[i])
        ax[r, 0].set_title(f"RGB frame t = {tf / 1e6:.3f} s (person {speed[i]:.0f} px/s)")
        for a in ax[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_event_frames.png"), dpi=130)
    plt.close(fig)

    # --- 2 x-t slice through the person at its fastest moment (1 s window)
    i = picks[len(picks) // 2]
    m = np.isin(inst[i], ids) if inst is not None else None
    if m is not None and m.any():
        ys, xs = np.nonzero(m)
        row = int(np.median(ys))
        x0, x1 = max(0, xs.min() - 40), min(W - 1, xs.max() + 40)
    else:
        row, x0, x1 = H // 2, 0, W - 1
    t_c = frames_t_us[i]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for a, (name, ev) in zip(ax, (("our EVK4", ours), (labels["v2e"], v2e))):
        sel = (np.abs(ev["y"].astype(int) - row) <= 1) & (ev["x"] >= x0) & (ev["x"] <= x1) & \
              (np.abs(ev["t"] - t_c) <= 500_000)
        e = ev[sel]
        a.scatter((e["t"] - t_c) / 1e3, e["x"], s=1.5, c=np.where(e["p"] == 1, "tab:red", "tab:blue"),
                  linewidths=0)
        for tf in frames_t_us[(frames_t_us > t_c - 500_000) & (frames_t_us < t_c + 500_000)]:
            a.axvline((tf - t_c) / 1e3, color="0.85", lw=0.5, zorder=0)
        a.set_title(f"{name}: image row {row} +-1, {len(e)} events")
        a.set_xlabel("time [ms] (grey lines: RGB frame times)")
    ax[0].set_ylabel("x [px]")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_space_time.png"), dpi=150)
    plt.close(fig)

    # --- 3 x-y-t cloud of the person, 200 ms
    fig = plt.figure(figsize=(13, 5.5))
    for c, (name, ev) in enumerate((("our EVK4", ours), (labels["v2e"], v2e))):
        a = fig.add_subplot(1, 2, c + 1, projection="3d")
        sel = (ev["t"] > t_c - 200_000) & (ev["t"] <= t_c)
        if m is not None and m.any():
            sel &= (ev["x"] >= x0) & (ev["x"] <= x1)
        e = ev[sel]
        if len(e) > 25000:
            e = e[np.random.default_rng(0).choice(len(e), 25000, replace=False)]
        a.scatter((e["t"] - t_c) / 1e3, e["x"], e["y"], s=0.6, c=np.where(e["p"] == 1, "tab:red", "tab:blue"),
                  linewidths=0, alpha=0.6)
        a.set_xlabel("t [ms]")
        a.set_ylabel("x")
        a.set_zlabel("y")
        a.invert_zaxis()
        a.set_title(f"{name}: (x, y, t) events, last 200 ms")
        a.view_init(elev=18, azim=-60)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_xyt_cloud.png"), dpi=150)
    plt.close(fig)

    # --- 4 timestamp phase histogram
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, ev, col in (("our EVK4", ours, "tab:green"), (labels["v2e"], v2e, "tab:purple")):
        ph = phase_of(ev["t"], frames_t_us)
        ax.hist(ph, bins=50, range=(0, 1), density=True, histtype="step", lw=1.8, color=col,
                label=f"{name} (KS {ks_uniform(ph):.2f})")
    ax.axhline(1.0, color="0.6", ls="--", lw=1, label="uniform in time")
    ax.set_xlabel("position of the event time inside the RGB frame interval")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_timestamp_phase.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode")
    ap.add_argument("--cam", default=None)
    ap.add_argument("--v2e-root", required=True, help="folder of a clone of github.com/SensorsINI/v2e")
    ap.add_argument("--out", default="cmp_v2e")
    ap.add_argument("--t0", type=float, default=1.0, help="start of the compared window [s]")
    ap.add_argument("--t1", type=float, default=None, help="end of the window [s] (default t0 + 4 s)")
    ap.add_argument("--slomo-model", default=None, help="SuperSloMo39.ckpt for v2e frame interpolation")
    ap.add_argument("--shot-noise-hz", default="match",
                    help="v2e shot-noise rate per pixel; 'match' = our measured background rate")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    import h5py  # noqa: F401
    os.makedirs(args.out, exist_ok=True)
    ep = args.episode
    meta = json.load(open(os.path.join(ep, "meta.json")))
    cam = args.cam or sorted(meta["cameras"])[0]
    cm = dict(meta["cameras"][cam])
    cm["_name"] = cam
    cfg = {k: (float(v) if isinstance(v, (int, float)) or _isnum(v) else v) for k, v in cm["config"].items()}
    gt = np.load(os.path.join(ep, "gt.npz"))
    ids = [h["instance_id"] for h in meta["humans"]]
    t0 = args.t0
    t1 = args.t1 if args.t1 is not None else t0 + 4.0
    t1 = min(t1, float(gt["t"][-1]))

    # ---- ours
    ef = EventFile(os.path.join(ep, f"events_{cam}.h5"))
    W, H = ef.W, ef.H
    ours = ef.window(int(t0 * 1e6), int(t1 * 1e6))
    frames_t, rgb, inst, aligned, exposure = load_frames(os.path.join(ep, f"frames_{cam}.h5"), t0, t1)
    if not aligned or rgb.shape[1:3] != (H, W):
        sys.exit("frames are not pixel-aligned with the events (distortion / spp > 1): not supported here")
    in_win = (frames_t >= t0) & (frames_t <= t1)
    # background rate of our sensor: events on the static scene (instance 0 / -1) per pixel and second
    static = np.isin(ours["instance"], [0, -1])
    bg_rate = float(static.sum() / (W * H) / max(1e-6, t1 - t0))
    shot = bg_rate if args.shot_noise_hz == "match" else float(args.shot_noise_hz)
    params = v2e_params(cfg, min(shot, 5.0))
    print(f"{cam}: {W}x{H}, window {t0:.2f}-{t1:.2f} s, our events {len(ours)}; v2e params {params}", flush=True)

    # ---- v2e on the RGB video of the same camera
    v2e = run_v2e(frames_t, rgb, params, args.v2e_root, args.device, args.slomo_model, workdir=args.out)
    v2e = v2e[(v2e["t"] > t0 * 1e6) & (v2e["t"] <= t1 * 1e6)]
    label = "v2e + SuperSloMo" if args.slomo_model else "v2e (30 fps video)"
    print(f"{label}: {len(v2e)} events", flush=True)

    # ---- metrics
    ft_us = np.rint(frames_t * 1e6).astype(np.int64)
    ft_win = ft_us[in_win]
    speed = person_speed(ep, cm, frames_t[in_win], gt) if ids else np.zeros(in_win.sum())
    res = {"camera": cam, "window_s": [t0, t1], "resolution": [W, H], "v2e_mode": label,
           "v2e_params": params, "rgb_fps": float(1.0 / np.median(np.diff(frames_t))),
           "rgb_exposure_s": exposure}
    rate_series = {}
    for name, ev in (("ours", ours), ("v2e", v2e)):
        ph = phase_of(ev["t"], ft_us)
        r = dict(n_events=int(len(ev)), events_per_s=float(len(ev) / (t1 - t0)),
                 events_per_px_per_s=float(len(ev) / (t1 - t0) / (W * H)),
                 timestamp_phase_ks_to_uniform=ks_uniform(ph),
                 near_frame_time_fraction=near_frame_fraction(ev["t"], ft_us),
                 unique_timestamps_per_s=float(len(np.unique(ev["t"])) / (t1 - t0)))
        if inst is not None and ids:
            r["on_person_mask_near_frame_times"] = on_mask_fraction(ev, ft_win, inst[in_win], ids)
            # person event rate per frame interval (events inside the dilated person mask)
            import cv2
            cnt = []
            for k in range(len(ft_win)):
                sel = (ev["t"] > ft_win[k] - 33_333) & (ev["t"] <= ft_win[k])
                mk = cv2.dilate(np.isin(inst[in_win][k], ids).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                cnt.append(int(mk[ev["y"][sel].astype(int), ev["x"][sel].astype(int)].sum()))
            cnt = np.array(cnt, float)
            rate_series[name] = cnt
            ok = np.isfinite(speed)
            r["person_rate_vs_speed_corr"] = float(np.corrcoef(cnt[ok], speed[ok])[0, 1]) \
                if ok.sum() > 3 and cnt[ok].std() > 0 and speed[ok].std() > 0 else float("nan")
        res[name] = r
    res["our_background_rate_hz_per_px"] = bg_rate
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)

    # ---- figures
    figures(args.out, ours, v2e, ft_win, rgb[in_win], inst[in_win] if inst is not None else None,
            ids, W, H, speed, {"v2e": label})
    if rate_series:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3.6))
        tt = frames_t[in_win] - frames_t[in_win][0]
        ax.plot(tt, rate_series["ours"], label="our EVK4: events on the person / 33 ms", color="tab:green")
        ax.plot(tt, rate_series["v2e"], label=f"{label}: events on the person / 33 ms", color="tab:purple")
        ax2 = ax.twinx()
        ax2.plot(tt, speed, color="0.4", ls="--", lw=1, label="person image speed [px/s]")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("events")
        ax2.set_ylabel("px/s")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fig5_rate_vs_motion.png"), dpi=150)
        plt.close(fig)

    # ---- summary
    o, v = res["ours"], res["v2e"]
    lines = [f"# Physics-based EVK4 vs {label} - {os.path.basename(os.path.normpath(ep))} / {cam}", "",
             f"Window {t0:.2f}-{t1:.2f} s, {W}x{H} px, RGB input {res['rgb_fps']:.0f} fps "
             f"(exposure {exposure * 1e3:.0f} ms). Same contrast thresholds, mismatch, cutoff, refractory and "
             "leak for both.", "",
             "| metric | our EVK4 | " + label + " |", "|---|---|---|"]
    for k, desc in (("events_per_px_per_s", "events / pixel / s"),
                    ("timestamp_phase_ks_to_uniform", "timestamp clustering at frame times (KS to uniform; 0 = even)"),
                    ("near_frame_time_fraction", "events within 0.5 ms of a video frame time (uniform: 3 %)"),
                    ("unique_timestamps_per_s", "distinct timestamps / s"),
                    ("on_person_mask_near_frame_times", "events within 4 ms of a frame on the true person mask"),
                    ("person_rate_vs_speed_corr", "person event rate vs true image speed (corr.)")):
        if k in o:
            lines.append(f"| {desc} | {o[k]:.3g} | {v[k]:.3g} |")
    lines += ["", "Figures: fig1_event_frames.png, fig2_space_time.png, fig3_xyt_cloud.png, "
              "fig4_timestamp_phase.png, fig5_rate_vs_motion.png"]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
