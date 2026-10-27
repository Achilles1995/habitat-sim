"""
Read simulated AND real ESD sequences with the same code, build event representations with masks,
and score segmentations with the paper's metrics (pixel accuracy, Eq. 6; mIoU, Eq. 7).

    seq = load_sequence("ESD-2/conditions_3")            # real ESD  (or a simulated folder)
    ev  = seq.events("left")                              # structured: x y p t_us depth_mm label
    for x, y in seq.samples("left", window_us=60000, repr="frame2"):   # (C,H,W) float32, (H,W) uint8
        ...
"""
from __future__ import annotations

import glob
import os

import numpy as np

from .catalog import ESD1_PALETTE, ESD2_PALETTE

EV_DTYPE = np.dtype([("x", "<i2"), ("y", "<i2"), ("p", "<i1"), ("t", "<i8"), ("depth", "<f4"), ("label", "<u1")])


def _loadmat(path, variables=None):
    """MATLAB v5 (scipy) or v7.3 (HDF5 / h5py) -> dict of numpy arrays.  `variables`: read only these
    (the real ESD left/right.mat also hold the RGB-D images and MATLAB objects - skipping them is
    faster and avoids touching damaged or unsupported parts of a file)."""
    from scipy.io import loadmat
    clean = lambda d: {k: v for k, v in d.items() if not k.startswith("__")}
    try:
        if variables:
            try:                                       # only the wanted variables (others are skipped unread)
                d = clean(loadmat(path, variable_names=list(variables)))
                if d:
                    return d
            except (NotImplementedError, ValueError):
                raise
            except Exception:                          # odd variable types elsewhere in the file
                pass
        return clean(loadmat(path))
    except (NotImplementedError, ValueError):          # v7.3 (HDF5)
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            for k, v in f.items():
                if variables and k not in variables:
                    continue
                if hasattr(v, "shape"):
                    out[k] = np.array(v).T                 # MATLAB column-major
        return out


def _time_unit(t):
    """(origin, scale to microseconds) of a time array; unit detected from the span (ns | us | s)."""
    t = np.asarray(t, np.float64)
    if len(t) == 0:
        return 0.0, 1.0
    span = t.max() - t.min()
    return float(t.min()), (1e-3 if span > 5e7 else (1.0 if span > 5e3 else 1e6))


def _to_us(t, origin=None, scale=None):
    """Times -> microseconds from `origin` (default: the first sample)."""
    t = np.asarray(t, np.float64)
    if len(t) == 0:
        return t.astype(np.int64)
    o, sc = _time_unit(t)
    o = o if origin is None else origin
    sc = sc if scale is None else scale
    return np.round((t - o) * sc).astype(np.int64)


def events_from_mat(path, var="event_labeled"):
    d = _loadmat(path, [var, "image_raw_ts", "intel_color_ts", "image_ts"])
    if var not in d:
        raise KeyError(f"{path}: no '{var}' (variables: {list(d)})")
    E = np.asarray(d[var], np.float64)
    if E.shape[0] == 6 and E.shape[1] != 6:
        E = E.T
    ev = np.empty(len(E), EV_DTYPE)
    ev["x"], ev["y"] = E[:, 0], E[:, 1]
    ev["p"] = np.where(E[:, 2] > 0, 1, -1)
    origin, scale = _time_unit(E[:, 3])
    ev["t"] = _to_us(E[:, 3], origin, scale)
    ev["depth"] = E[:, 4]
    ev["label"] = np.clip(E[:, 5], 0, 255)
    ev = ev[np.argsort(ev["t"], kind="stable")]
    img_ts = None
    for k in ("image_raw_ts", "intel_color_ts", "image_ts"):
        if k in d:
            img_ts = _to_us(np.asarray(d[k], np.float64).ravel(), origin, scale)
            break
    return ev, img_ts


def mask_from_color(img, palette=None):
    """ESD colour mask (H,W,3) -> label ids; tries the ESD-1 then the ESD-2 palette."""
    img = np.asarray(img)[..., :3]
    best = None
    for pal in ([palette] if palette else [ESD1_PALETTE, ESD2_PALETTE]):
        lab = np.zeros(img.shape[:2], np.uint8)
        for k, c in pal.items():
            lab[np.all(img == c, -1)] = k
        if best is None or (lab > 0).sum() > (best > 0).sum():
            best = lab
    return best


class Sequence:
    def __init__(self, root):
        self.root = root
        self._ev = {}

    @property
    def is_sim(self):
        return os.path.exists(os.path.join(self.root, "meta.json"))

    def events(self, side="left"):
        if side not in self._ev:
            cands = glob.glob(os.path.join(self.root, "events", f"{side}.mat")) or \
                glob.glob(os.path.join(self.root, f"{side}.mat")) or \
                glob.glob(os.path.join(self.root, "**", f"{side}.mat"), recursive=True)
            if not cands:
                raise FileNotFoundError(f"{side}.mat under {self.root}")
            self._ev[side] = events_from_mat(cands[0])
        return self._ev[side][0]

    def image_times(self, side="left"):
        """RGB-D frame times [us] on the event clock (None if the .mat does not store them)."""
        self.events(side)
        return self._ev[side][1]

    def rgb_frames(self):
        """[(rgb, label)] of the RGB-D camera (images + colour masks, or label pngs if present)."""
        from PIL import Image
        imgs = glob.glob(os.path.join(self.root, "RGB", "images", "*"))
        key = lambda f: int("".join(c for c in os.path.splitext(os.path.basename(f))[0] if c.isdigit()) or 0)
        out = []
        for f in sorted(imgs, key=key):
            b = os.path.splitext(os.path.basename(f))[0]
            lab_f = os.path.join(self.root, "RGB", "labels", b + ".png")
            if os.path.exists(lab_f):
                lab = np.array(Image.open(lab_f))
            else:
                mf = glob.glob(os.path.join(self.root, "RGB", "masks", b + ".*"))
                lab = mask_from_color(np.array(Image.open(mf[0]).convert("RGB"))) if mf else None
            out.append((np.array(Image.open(f).convert("RGB")), lab))
        return out

    def samples(self, side="left", window_us=None, n_events=None, repr="frame2", H=260, W=346,
                min_events=200, label_from="events"):
        """Yield (x, y) per time window (window_us, default 60 ms) or per n_events.
        y = per-pixel label of the events in the window (majority, 0 = background, 255 = no event)
        - the ESD evaluation unit; label_from='mask' uses mask_events_frame.mat (dense) instead."""
        ev = self.events(side)
        if len(ev) == 0:
            return
        masks = None
        if label_from == "mask":
            mf = os.path.join(self.root, "events", "mask_events_frame.mat")
            d = _loadmat(mf)
            masks, mts = np.asarray(d[side]), _to_us(np.asarray(d["ts"]).ravel())
        if n_events:
            bounds = list(range(0, len(ev), n_events))
            idx = [(a, min(a + n_events, len(ev))) for a in bounds]
        else:
            w = window_us or 60000
            edges = np.arange(ev["t"][0], ev["t"][-1] + w, w)
            pos = np.searchsorted(ev["t"], edges)
            idx = list(zip(pos[:-1], pos[1:]))
        for a, b in idx:
            e = ev[a:b]
            if len(e) < min_events:
                continue
            x = event_repr(e, repr, H, W)
            if masks is not None:
                k = int(np.argmin(np.abs(mts - e["t"][len(e) // 2])))
                y = masks[k].astype(np.uint8)
            else:
                y = event_label_image(e, H, W)
            yield x, y


def event_repr(e, repr="frame2", H=260, W=346, bins=5):
    """frame2: ON/OFF counts (2,H,W) | frame1: signed sum | voxel: (bins,H,W) bilinear in time |
    timesurface: (2,H,W) latest-timestamp decay."""
    x, y = e["x"].astype(np.int64), e["y"].astype(np.int64)
    on = e["p"] > 0
    if repr == "frame2":
        out = np.zeros((2, H, W), np.float32)
        np.add.at(out, (np.where(on, 0, 1), y, x), 1)
        return out
    if repr == "frame1":
        out = np.zeros((1, H, W), np.float32)
        np.add.at(out, (0, y, x), np.where(on, 1.0, -1.0))
        return out
    if repr == "voxel":
        out = np.zeros((bins, H, W), np.float32)
        t = e["t"].astype(np.float64)
        tn = (bins - 1) * (t - t[0]) / max(1.0, t[-1] - t[0])
        b0 = np.floor(tn).astype(int); f = tn - b0
        pol = np.where(on, 1.0, -1.0)
        np.add.at(out, (b0, y, x), pol * (1 - f))
        ok = b0 + 1 < bins
        np.add.at(out, (b0[ok] + 1, y[ok], x[ok]), (pol * f)[ok])
        return out
    if repr == "timesurface":
        out = np.zeros((2, H, W), np.float32)
        t = e["t"].astype(np.float64)
        tau = max(1.0, (t[-1] - t[0]) / 3)
        out[np.where(on, 0, 1), y, x] = np.exp(-(t[-1] - t) / tau)
        return out
    raise ValueError(repr)


def event_label_image(e, H=260, W=346, ignore=255):
    """Majority label of the events per pixel; pixels without events = `ignore`."""
    n_lab = 256
    acc = np.zeros((H * W, n_lab), np.int32)
    np.add.at(acc, (e["y"].astype(np.int64) * W + e["x"], e["label"].astype(np.int64)), 1)
    lab = acc.argmax(1).astype(np.uint8)
    lab[acc.sum(1) == 0] = ignore
    return lab.reshape(H, W)


# ---------------------------------------------------------------- metrics (paper Eq. 6, 7)
def pixel_accuracy(gt, pred, ignore=255):
    ok = gt != ignore
    return float(np.mean(gt[ok] == pred[ok])) if ok.any() else float("nan")


def miou(gt, pred, n_classes=None, ignore=255):
    ok = gt != ignore
    g, p = gt[ok].astype(int), pred[ok].astype(int)
    classes = range(n_classes) if n_classes else np.union1d(np.unique(g), np.unique(p))
    ious = []
    for c in classes:
        inter = np.sum((g == c) & (p == c)); union = np.sum((g == c) | (p == c))
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else float("nan")


def load_sequence(root):
    return Sequence(root)


def find_sequences(root):
    """All sequence folders (real ESD or simulated) at or below `root`: any folder holding left.mat or
    right.mat (directly or in an 'events' sub-folder)."""
    root = os.path.abspath(root)
    hits = set()
    for pat in ("left.mat", "right.mat"):
        for p in glob.glob(os.path.join(root, "**", pat), recursive=True):
            d = os.path.dirname(p)
            hits.add(os.path.dirname(d) if os.path.basename(d).lower() == "events" else d)
    return [Sequence(h) for h in sorted(hits)]


# ---------------------------------------------------------------- optional PyTorch dataset
def torch_dataset(roots, side="left", window_us=60000, n_events=None, repr="frame2", label_from="events",
                  class_agnostic=False, max_per_seq=None):
    """torch.utils.data.Dataset over sim and/or real ESD folders.  class_agnostic=True maps every
    object label to 1 (foreground) - the setting for unseen objects (train on sim, test on ESD-2)."""
    import torch
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __init__(self):
            self.items = []
            for r in ([roots] if isinstance(roots, str) else roots):
                for s in find_sequences(r):
                    for i, (x, y) in enumerate(s.samples(side, window_us, n_events, repr, label_from=label_from)):
                        if max_per_seq and i >= max_per_seq:
                            break
                        if class_agnostic:
                            y = np.where((y > 0) & (y != 255), 1, y).astype(np.uint8)
                        self.items.append((x, y))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            x, y = self.items[i]
            return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))

    return _DS()
