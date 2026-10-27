#!/usr/bin/env python3
"""
Train / test baseline models on the HRC event datasets written by hrc_generate.py.

  safety   per camera and control tick: collision-risk class (safe / caution / danger),
           human-robot clearance [m], and the future path of the nearest person in the robot
           frame (+0.5 / +1.0 / +1.5 s) - i.e. proximity, intent and trajectory prediction.
           Metrics: accuracy, macro-F1 and confusion matrix of the risk class, clearance MAE,
           ADE / FDE of the predicted path.
  actions  online action recognition from the last 1 s of events (one decision per tick);
           temporal segmentation of whole test episodes from those decisions; early recognition
           (accuracy when only the first 10 %, 20 %, ... of an action has been observed).
           Metrics: window accuracy, frame accuracy, segmental edit score, F1@{10,25,50},
           early-recognition curve.

Input representation: event-count frames per control tick (33 ms), ON / OFF channels, at
--size (default 160x90), stacked over the last --window ticks -> a (2 * window)-channel image.
They are computed once per episode / camera and cached as <episode>/cache_<cam>_<w>x<h>.npy.

  python evk4/tools/hrc_train.py safety  --root datasets/safety  --epochs 15 --out runs/safety
  python evk4/tools/hrc_train.py actions --root datasets/actions --epochs 25 --out runs/actions
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

TICK = 1.0 / 30.0


# ------------------------------------------------------------------------------------ data
def tick_frames(ep_dir, cam, size):
    """(T, 2, h, w) uint8 event counts per control tick (ON, OFF), cached."""
    w, h = size
    cache = os.path.join(ep_dir, f"cache_{cam}_{w}x{h}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import h5py
    gt = np.load(os.path.join(ep_dir, "gt.npz"))
    T = len(gt["t"])
    t_ticks = gt["t"]
    with h5py.File(os.path.join(ep_dir, f"events_{cam}.h5"), "r") as f:
        W, H = int(f.attrs["width"]), int(f.attrs["height"])
        e = f["events"]
        x = e["x"][:].astype(np.int64) * w // W
        y = e["y"][:].astype(np.int64) * h // H
        p = e["p"][:].astype(np.int64)
        t = e["t"][:] * 1e-6
    k = np.searchsorted(t_ticks, t, side="left")          # event in (t_{k-1}, t_k] -> tick k
    ok = (k >= 0) & (k < T)
    out = np.zeros((T, 2, h, w), np.int32)
    np.add.at(out, (k[ok], 1 - p[ok], y[ok], x[ok]), 1)
    out = np.minimum(out, 255).astype(np.uint8)
    np.save(cache, out)
    return out


def tick_rgb(ep_dir, cam, size):
    """(T, 3, h, w) uint8 synchronised RGB frame of every control tick (the latest frame whose
    exposure ended at or before the tick), from frames_<cam>.h5; cached."""
    w, h = size
    cache = os.path.join(ep_dir, f"cache_rgb_{cam}_{w}x{h}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import cv2
    import h5py
    t_ticks = np.load(os.path.join(ep_dir, "gt.npz"))["t"]
    with h5py.File(os.path.join(ep_dir, f"frames_{cam}.h5"), "r") as f:
        tf = f["t_us"][:] * 1e-6
        rgb = f["rgb"]
        idx = np.clip(np.searchsorted(tf, t_ticks + 1e-7, side="right") - 1, 0, len(tf) - 1)
        small = np.stack([cv2.resize(rgb[i], (w, h), interpolation=cv2.INTER_AREA) for i in range(len(tf))])
    out = np.ascontiguousarray(small[idx].transpose(0, 3, 1, 2))
    np.save(cache, out)
    return out


def episodes(root):
    return sorted(d for d in glob.glob(os.path.join(root, "ep*")) if os.path.exists(os.path.join(d, "gt.npz")))


def split(eps, test_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eps))
    n_test = max(1, int(round(test_frac * len(eps)))) if len(eps) > 1 else 0
    test = [eps[i] for i in idx[:n_test]]
    train = [eps[i] for i in idx[n_test:]] or test
    return train, test


def window(frames, k, n):
    """Stack ticks k-n+1..k (zero-padded at the start) -> (2n, h, w) float32 (log counts)."""
    lo = k - n + 1
    if lo >= 0:
        w = frames[lo:k + 1]
    else:
        w = np.concatenate([np.zeros((-lo,) + frames.shape[1:], frames.dtype), frames[:k + 1]])
    return np.log1p(w.reshape(-1, *frames.shape[2:]).astype(np.float32))


def safety_targets(ep_dir, horizons=(15, 30, 45)):
    """Per tick: risk (max over humans), clearance (min), future feet of the nearest human in the
    robot frame (x forward, z right) at the given tick horizons (NaN beyond the episode)."""
    g = np.load(os.path.join(ep_dir, "gt.npz"))
    risk = g["risk"].max(1).astype(np.int64)
    clr = g["clearance"].min(1)
    near = g["clearance"].argmin(1)
    T = len(risk)
    fut = np.full((T, len(horizons), 2), np.nan, np.float32)
    Rinv = np.linalg.inv(g["robot_T"])
    for k in range(T):
        for j, hz in enumerate(horizons):
            if k + hz < T:
                p = g["human_feet"][k + hz, near[k]]
                q = Rinv[k] @ np.r_[p, 1.0]
                fut[k, j] = (q[0], q[2])
    return risk, clr.astype(np.float32), fut


# ------------------------------------------------------------------------------------ model
def build_model(in_ch, n_cls, n_reg, n_cams):
    import torch
    import torch.nn as nn

    def block(i, o, s):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(block(in_ch, 32, 2), block(32, 64, 2), block(64, 128, 2), block(128, 192, 2),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.cam = nn.Embedding(n_cams, 16)
            self.cls = nn.Sequential(nn.Linear(192 + 16, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                     nn.Linear(128, n_cls))
            self.reg = nn.Sequential(nn.Linear(192 + 16, 128), nn.ReLU(inplace=True), nn.Linear(128, max(1, n_reg)))

        def forward(self, x, cam):
            f = torch.cat([self.enc(x), self.cam(cam)], 1)
            return self.cls(f), self.reg(f)

    return Net()


class Samples:
    """(episode, camera, tick) samples with lazily loaded tick frames."""

    def __init__(self, eps, cams, size, win, app, stride=1, modality="events"):
        self.items, self.frames, self.rgb, self.targets = [], {}, {}, {}
        self.cams = cams
        self.win = win
        self.modality = modality
        for e in eps:
            meta = json.load(open(os.path.join(e, "meta.json")))
            for ci, c in enumerate(cams):
                if c not in meta["cameras"]:
                    continue
                fr = tick_frames(e, c, size)
                self.frames[(e, c)] = fr
                if modality != "events":
                    self.rgb[(e, c)] = tick_rgb(e, c, size)
                for k in range(win, len(fr), stride):
                    self.items.append((e, c, ci, k))
            if app == "safety":
                self.targets[e] = safety_targets(e)
            else:
                self.targets[e] = np.load(os.path.join(e, "gt.npz"))["action"].astype(np.int64)

    def __len__(self):
        return len(self.items)

    def batch(self, idx, app):
        import torch
        X, C, Y, R = [], [], [], []
        for i in idx:
            e, c, ci, k = self.items[i]
            parts = []
            if self.modality in ("events", "both"):
                parts.append(window(self.frames[(e, c)], k, self.win))
            if self.modality in ("rgb", "both"):
                parts.append(self.rgb[(e, c)][k].astype(np.float32) / 255.0)    # frame synchronised with tick k
            X.append(np.concatenate(parts, 0))
            C.append(ci)
            if app == "safety":
                risk, clr, fut = self.targets[e]
                Y.append(risk[k])
                R.append(np.r_[clr[k], fut[k].reshape(-1)])
            else:
                Y.append(self.targets[e][k])
                R.append([0.0])
        R = np.array(R, np.float32)
        return (torch.from_numpy(np.stack(X)), torch.tensor(C), torch.tensor(Y), torch.from_numpy(R))


# ------------------------------------------------------------------------------------ metrics
def macro_f1(cm):
    f = []
    for i in range(len(cm)):
        tp = cm[i, i]
        p = tp / max(1, cm[:, i].sum())
        r = tp / max(1, cm[i, :].sum())
        if cm[i, :].sum() > 0:
            f.append(0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f)) if f else 0.0


def segments_of(seq):
    out, s = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[s]:
            out.append((int(seq[s]), s, i))
            s = i
    return out


def edit_score(pred, gt):
    a = [s[0] for s in segments_of(pred)]
    b = [s[0] for s in segments_of(gt)]
    D = np.zeros((len(a) + 1, len(b) + 1))
    D[:, 0] = np.arange(len(a) + 1)
    D[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return 100 * (1 - D[-1, -1] / max(len(a), len(b), 1))


def f1_at(pred, gt, thr):
    P, G = segments_of(pred), segments_of(gt)
    used = np.zeros(len(G), bool)
    tp = 0
    for c, s, e in P:
        best, bi = 0, -1
        for j, (c2, s2, e2) in enumerate(G):
            if c2 != c or used[j]:
                continue
            inter = max(0, min(e, e2) - max(s, s2))
            iou = inter / (max(e, e2) - min(s, s2))
            if iou > best:
                best, bi = iou, j
        if best >= thr:
            tp += 1
            used[bi] = True
    prec = tp / max(1, len(P))
    rec = tp / max(1, len(G))
    return 100 * (0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))


# ------------------------------------------------------------------------------------ train / test
def run(args):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    eps = episodes(args.root)
    if not eps:
        sys.exit(f"no episodes in {args.root}")
    train_eps, test_eps = split(eps, args.test_frac, args.seed)
    meta = json.load(open(os.path.join(eps[0], "meta.json")))
    cams = args.cams or sorted(meta["cameras"])
    size = tuple(int(v) for v in args.size.split("x"))
    win = args.window or (15 if args.app == "safety" else 30)
    names = meta.get("risk_names") if args.app == "safety" else meta.get("action_names")
    n_cls = len(names)
    n_reg = 7 if args.app == "safety" else 0
    tr = Samples(train_eps, cams, size, win, args.app, stride=args.stride, modality=args.modality)
    te = Samples(test_eps, cams, size, win, args.app, stride=1, modality=args.modality)
    in_ch = {"events": 2 * win, "rgb": 3, "both": 2 * win + 3}[args.modality]
    print(f"{len(train_eps)} train / {len(test_eps)} test episodes, cameras {cams}, "
          f"{len(tr)} train / {len(te)} test samples, modality {args.modality}, input {in_ch}x{size[1]}x{size[0]}", flush=True)
    # class weights against imbalance
    ys = np.array([tr.batch([i], args.app)[2].item() for i in range(0, len(tr), max(1, len(tr) // 2000))])
    cnt = np.bincount(ys, minlength=n_cls).astype(np.float64)
    wts = torch.tensor(np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0.0), dtype=torch.float32, device=dev)
    model = build_model(in_ch, n_cls, n_reg, len(cams)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * max(1, math.ceil(len(tr) / args.batch)))
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(tr))
        tot = n = 0
        for b in range(0, len(perm), args.batch):
            X, C, Y, R = (v.to(dev) for v in tr.batch(perm[b:b + args.batch], args.app))
            if args.augment and args.app == "actions" and rng.random() < 0.5:
                X = torch.flip(X, [3])                 # left-right mirror (action labels are symmetric)
            lc, lr = model(X, C)
            loss = F.cross_entropy(lc, Y, weight=wts)
            if args.app == "safety":
                m = torch.isfinite(R)
                loss = loss + F.smooth_l1_loss(lr[m], R[m]) if m.any() else loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(Y)
            n += len(Y)
        print(f"epoch {epoch + 1}/{args.epochs}  train loss {tot / max(1, n):.4f}", flush=True)
    torch.save(dict(model=model.state_dict(), cams=cams, size=size, window=win, names=names, app=args.app,
                    modality=args.modality),
               os.path.join(args.out, "model.pt"))
    # ---------------- test
    model.eval()
    preds, gts, regs, regt, keys = [], [], [], [], []
    with torch.no_grad():
        for b in range(0, len(te), args.batch):
            idx = list(range(b, min(len(te), b + args.batch)))
            X, C, Y, R = (v.to(dev) for v in te.batch(idx, args.app))
            lc, lr = model(X, C)
            preds.append(lc.argmax(1).cpu().numpy())
            gts.append(Y.cpu().numpy())
            regs.append(lr.cpu().numpy())
            regt.append(R.cpu().numpy())
            keys += [te.items[i] for i in idx]
    P, G = np.concatenate(preds), np.concatenate(gts)
    cm = np.zeros((n_cls, n_cls), np.int64)
    np.add.at(cm, (G, P), 1)
    res = dict(app=args.app, modality=args.modality, n_test=int(len(G)), accuracy=float((P == G).mean()), macro_f1=macro_f1(cm),
               confusion=cm.tolist(), classes=names)
    if args.app == "safety":
        RP, RT = np.concatenate(regs), np.concatenate(regt)
        res["clearance_mae_m"] = float(np.nanmean(np.abs(RP[:, 0] - RT[:, 0])))
        fp, ft = RP[:, 1:].reshape(-1, 3, 2), RT[:, 1:].reshape(-1, 3, 2)
        err = np.linalg.norm(fp - ft, axis=2)
        res["ADE_m"] = float(np.nanmean(err))
        res["FDE_1.5s_m"] = float(np.nanmean(err[:, -1]))
        for c in cams:      # per camera
            m = np.array([k[1] == c for k in keys])
            if m.any():
                res[f"accuracy_{c}"] = float((P[m] == G[m]).mean())
    else:
        # temporal segmentation per test episode and camera from the online decisions
        seg = dict(frame_acc=[], edit=[], f1_10=[], f1_25=[], f1_50=[])
        early = {f: [] for f in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)}
        for e in test_eps:
            gt_full = np.load(os.path.join(e, "gt.npz"))["action"]
            for ci, c in enumerate(cams):
                m = np.array([(k[0] == e and k[1] == c) for k in keys])
                if not m.any():
                    continue
                ks = np.array([k[3] for k in np.array(keys, dtype=object)[m]])
                p = P[m][np.argsort(ks)]
                g = gt_full[np.sort(ks)]
                # light temporal smoothing of the online decisions (majority over 5 ticks)
                ps = p.copy()
                for i in range(len(p)):
                    vals = p[max(0, i - 4):i + 1]
                    ps[i] = np.bincount(vals).argmax()
                seg["frame_acc"].append(100 * float((ps == g).mean()))
                seg["edit"].append(edit_score(ps, g))
                for thr, kk in ((0.1, "f1_10"), (0.25, "f1_25"), (0.5, "f1_50")):
                    seg[kk].append(f1_at(ps, g, thr))
                for cls, s0, s1 in segments_of(g):
                    for frac in early:
                        j = s0 + max(0, int(math.ceil(frac * (s1 - s0))) - 1)
                        early[frac].append(int(ps[j] == cls))
        res["segmentation"] = {k: float(np.mean(v)) if v else None for k, v in seg.items()}
        res["early_recognition_accuracy"] = {f"{int(100 * f)}%": float(np.mean(v)) if v else None
                                             for f, v in early.items()}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "confusion"}, indent=1))
    print("confusion (rows = truth):")
    for name, row in zip(names, cm):
        print(f"  {name:>10s} " + " ".join(f"{v:6d}" for v in row))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("app", choices=["safety", "actions"])
    ap.add_argument("--root", required=True, help="folder with ep*/ episodes of this app")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--size", default="160x90")
    ap.add_argument("--window", type=int, default=0, help="ticks of history (default 15 safety / 30 actions)")
    ap.add_argument("--stride", type=int, default=2, help="training sample stride in ticks")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--modality", choices=["events", "rgb", "both"], default="events",
                    help="events only, the synchronised RGB frame only, or both (early fusion)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
