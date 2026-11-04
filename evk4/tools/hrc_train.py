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

Input representations (--modality), all over the SAME last --window control ticks (33 ms each):
  events  event-count frames per tick, ON / OFF channels                 -> 2 * window channels
  rgb     the synchronised RGB frame of each of those ticks               -> 3 * window channels
  both    the two stacked (early fusion)                                  -> 5 * window channels
at --size (default 160x90), cached as <episode>/cache_[rgb_]<cam>_<w>x<h>.npy.

Fair comparison: every modality uses the same episodes, cameras and samples (an episode/camera is
used only if both its events and frames exist), the same train / val / test split, the same network
(an input BatchNorm, then identical conv blocks; only the input width differs), the same loss,
class weights, augmentation, optimiser, schedule, early stopping on validation macro-F1 and seed.
--modality all runs the three in a row and writes comparison.md / comparison.json.

  python evk4/tools/hrc_train.py safety  --root datasets/safety  --epochs 30 --modality all --out runs/safety
  python evk4/tools/hrc_train.py actions --root datasets/actions --epochs 30 --modality all --augment --out runs/actions
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
def tick_frames(ep_dir, cam, size, chunk=1 << 22):
    """(T, 2, h, w) uint8 event counts per control tick (ON, OFF), cached.

    Events are read from the HDF5 file in chunks (4 M events at a time), so episodes with
    hundreds of millions of events need only ~100 MB of RAM."""
    w, h = size
    cache = os.path.join(ep_dir, f"cache_{cam}_{w}x{h}.npy")
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")       # memory-mapped: RAM stays small for big datasets
    import h5py
    t_ticks = np.load(os.path.join(ep_dir, "gt.npz"))["t"]
    T = len(t_ticks)
    counts = np.zeros(T * 2 * h * w, np.int64)
    with h5py.File(os.path.join(ep_dir, f"events_{cam}.h5"), "r") as f:
        W, H = int(f.attrs["width"]), int(f.attrs["height"])
        e = f["events"]
        n = e["t"].shape[0]
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            t = e["t"][a:b] * 1e-6
            k = np.searchsorted(t_ticks, t, side="left")      # event in (t_{k-1}, t_k] -> tick k
            ok = k < T
            x = e["x"][a:b][ok].astype(np.int64) * w // W
            y = e["y"][a:b][ok].astype(np.int64) * h // H
            ch = 1 - e["p"][a:b][ok].astype(np.int64)            # channel 0 = ON, 1 = OFF
            flat = ((k[ok] * 2 + ch) * h + y) * w + x
            counts += np.bincount(flat, minlength=counts.size)
            del t, k, ok, x, y, ch, flat
    out = np.minimum(counts.reshape(T, 2, h, w), 255).astype(np.uint8)
    np.save(cache, out)
    del out, counts
    return np.load(cache, mmap_mode="r")


def tick_rgb(ep_dir, cam, size):
    """(T, 3, h, w) uint8 synchronised RGB frame of every control tick (the latest frame whose
    exposure ended at or before the tick), from frames_<cam>.h5; cached."""
    w, h = size
    cache = os.path.join(ep_dir, f"cache_rgb_{cam}_{w}x{h}.npy")
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    import cv2
    import h5py
    t_ticks = np.load(os.path.join(ep_dir, "gt.npz"))["t"]
    with h5py.File(os.path.join(ep_dir, f"frames_{cam}.h5"), "r") as f:
        tf = f["t_us"][:] * 1e-6
        rgb = f["rgb"]
        idx = np.clip(np.searchsorted(tf, t_ticks + 1e-7, side="right") - 1, 0, len(tf) - 1)
        small = np.stack([cv2.resize(rgb[i], (w, h), interpolation=cv2.INTER_AREA) for i in range(len(tf))])
    np.save(cache, np.ascontiguousarray(small[idx].transpose(0, 3, 1, 2)))
    del small
    return np.load(cache, mmap_mode="r")


def episodes(root):
    return sorted(d for d in glob.glob(os.path.join(root, "ep*")) if os.path.exists(os.path.join(d, "gt.npz")))


def split(eps, test_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eps))
    n_test = max(1, int(round(test_frac * len(eps)))) if len(eps) > 1 else 0
    test = [eps[i] for i in idx[:n_test]]
    train = [eps[i] for i in idx[n_test:]] or test
    return train, test


def _stack(frames, k, n):
    lo = k - n + 1
    if lo >= 0:
        return frames[lo:k + 1]
    return np.concatenate([np.zeros((-lo,) + frames.shape[1:], frames.dtype), frames[:k + 1]])


def window(frames, k, n):
    """Event-count frames of ticks k-n+1..k (zero-padded at the start) -> (2n, h, w) float32 (log counts)."""
    w = _stack(frames, k, n)
    return np.log1p(w.reshape(-1, *frames.shape[2:]).astype(np.float32))


def window_rgb(frames, k, n):
    """The synchronised RGB frames of the *same* ticks k-n+1..k -> (3n, h, w) float32 in [0, 1]."""
    w = _stack(frames, k, n)
    return w.reshape(-1, *frames.shape[2:]).astype(np.float32) / 255.0


def safety_targets(ep_dir, horizons=(15, 30, 45), hist=15):
    """Per tick: risk (max over humans), clearance (min), future feet of the nearest human in the
    robot frame (x forward, z right) at the given tick horizons (NaN beyond the episode), and the
    same quantity predicted by a constant-velocity model from the true positions of the last
    `hist` ticks (a state-based reference that does not use the camera at all)."""
    g = np.load(os.path.join(ep_dir, "gt.npz"))
    risk = g["risk"].max(1).astype(np.int64)
    clr = g["clearance"].min(1)
    near = g["clearance"].argmin(1)
    T = len(risk)
    fut = np.full((T, len(horizons), 2), np.nan, np.float32)
    cv = np.full((T, len(horizons), 2), np.nan, np.float32)
    Rinv = np.linalg.inv(g["robot_T"])
    feet = g["human_feet"]
    for k in range(T):
        h = near[k]
        v = (feet[k, h] - feet[max(0, k - hist), h]) / max(1, min(hist, k))      # per tick
        for j, hz in enumerate(horizons):
            q = Rinv[k] @ np.r_[feet[k, h] + v * hz, 1.0]
            cv[k, j] = (q[0], q[2])
            if k + hz < T:
                q = Rinv[k] @ np.r_[feet[k + hz, h], 1.0]
                fut[k, j] = (q[0], q[2])
    return risk, clr.astype(np.float32), fut, cv


def robot_camera(name):
    """Trajectory targets are in the robot frame: only meaningful for cameras that move with it."""
    return not name.startswith("wall")


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
            # the same network for every modality; only the number of input channels differs and an
            # input BatchNorm puts event counts and RGB values on the same scale
            self.enc = nn.Sequential(nn.BatchNorm2d(in_ch), block(in_ch, 32, 2), block(32, 64, 2),
                                     block(64, 128, 2), block(128, 192, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.cam = nn.Embedding(n_cams, 16)
            self.cls = nn.Sequential(nn.Linear(192 + 16, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                     nn.Linear(128, n_cls))
            self.reg = nn.Sequential(nn.Linear(192 + 16, 128), nn.ReLU(inplace=True), nn.Linear(128, max(1, n_reg)))

        def forward(self, x, cam):
            f = torch.cat([self.enc(x), self.cam(cam)], 1)
            return self.cls(f), self.reg(f)

    return Net()


class Samples:
    """(episode, camera, tick) samples.  The sample set does not depend on the modality: an
    episode/camera is used only when both its events and its synchronised RGB were recorded, so
    event, RGB and fusion models are trained and tested on exactly the same samples."""

    def __init__(self, eps, cams, size, win, app, stride=1, modality="events"):
        self.items, self.frames, self.rgb, self.targets = [], {}, {}, {}
        self.cams = cams
        self.win = win
        self.modality = modality
        self.skipped = 0
        for e in eps:
            meta = json.load(open(os.path.join(e, "meta.json")))
            used = False
            for ci, c in enumerate(cams):
                if c not in meta["cameras"]:
                    continue
                if not (os.path.exists(os.path.join(e, f"events_{c}.h5")) and
                        os.path.exists(os.path.join(e, f"frames_{c}.h5"))):
                    self.skipped += 1
                    continue
                n_ticks = len(np.load(os.path.join(e, "gt.npz"))["t"])
                if modality in ("events", "both"):
                    self.frames[(e, c)] = tick_frames(e, c, size)
                if modality in ("rgb", "both"):
                    self.rgb[(e, c)] = tick_rgb(e, c, size)
                for k in range(win, n_ticks, stride):
                    self.items.append((e, c, ci, k))
                used = True
            if not used:
                continue
            if app == "safety":
                self.targets[e] = safety_targets(e)
            else:
                self.targets[e] = np.load(os.path.join(e, "gt.npz"))["action"].astype(np.int64)

    def __len__(self):
        return len(self.items)

    def labels(self, app):
        if app == "safety":
            return np.array([self.targets[e][0][k] for e, c, ci, k in self.items])
        return np.array([self.targets[e][k] for e, c, ci, k in self.items])

    def batch(self, idx, app):
        import torch
        X, C, Y, R = [], [], [], []
        for i in idx:
            e, c, ci, k = self.items[i]
            parts = []
            if self.modality in ("events", "both"):
                parts.append(window(self.frames[(e, c)], k, self.win))
            if self.modality in ("rgb", "both"):
                parts.append(window_rgb(self.rgb[(e, c)], k, self.win))
            X.append(np.concatenate(parts, 0))
            C.append(ci)
            if app == "safety":
                risk, clr, fut, _ = self.targets[e]
                Y.append(risk[k])
                f = fut[k].reshape(-1) if robot_camera(c) else np.full(fut[k].size, np.nan, np.float32)
                R.append(np.r_[clr[k], f])
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


def class_report(cm, names):
    rec = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i in range(len(cm))]
    valid = [r for r in rec if r is not None]
    return dict(balanced_accuracy=float(np.mean(valid)) if valid else 0.0,
                recall_per_class={n: r for n, r in zip(names, rec)})


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
MODALITIES = ("events", "rgb", "both")


def split3(eps, test_frac, val_frac, seed):
    """Episode-level train / val / test split; identical for every modality (it depends only on the
    episode list and the seed)."""
    train, test = split(eps, test_frac, seed)
    if val_frac <= 0 or len(train) < 3:
        return train, [], test
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(train))
    n_val = max(1, int(round(val_frac * len(train))))
    return [train[i] for i in idx[n_val:]], [train[i] for i in idx[:n_val]], test


def predict(model, ds, app, batch, dev, idx=None):
    import torch
    idx = list(range(len(ds))) if idx is None else list(idx)
    P, G, RP, RT = [], [], [], []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(idx), batch):
            X, C, Y, R = (v.to(dev) for v in ds.batch(idx[b:b + batch], app))
            lc, lr = model(X, C)
            P.append(lc.argmax(1).cpu().numpy())
            G.append(Y.cpu().numpy())
            RP.append(lr.cpu().numpy())
            RT.append(R.cpu().numpy())
    return np.concatenate(P), np.concatenate(G), np.concatenate(RP), np.concatenate(RT), [ds.items[i] for i in idx]


def confusion(P, G, n_cls):
    cm = np.zeros((n_cls, n_cls), np.int64)
    np.add.at(cm, (G, P), 1)
    return cm


def run(args, modality=None):
    import torch
    import torch.nn.functional as F

    modality = modality or args.modality
    torch.manual_seed(args.seed)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    eps = episodes(args.root)
    if not eps:
        sys.exit(f"no episodes in {args.root}")
    train_eps, val_eps, test_eps = split3(eps, args.test_frac, args.val_frac, args.seed)
    meta = json.load(open(os.path.join(eps[0], "meta.json")))
    cams = args.cams or sorted(meta["cameras"])
    size = tuple(int(v) for v in args.size.split("x"))
    win = args.window or (15 if args.app == "safety" else 30)
    names = meta.get("risk_names") if args.app == "safety" else meta.get("action_names")
    n_cls = len(names)
    n_reg = 7 if args.app == "safety" else 0
    tr = Samples(train_eps, cams, size, win, args.app, stride=args.stride, modality=modality)
    va = Samples(val_eps, cams, size, win, args.app, stride=args.val_stride, modality=modality) if val_eps else None
    te = Samples(test_eps, cams, size, win, args.app, stride=1, modality=modality)
    in_ch = {"events": 2 * win, "rgb": 3 * win, "both": 5 * win}[modality]
    out = os.path.join(args.out, modality) if args.modality == "all" else args.out
    os.makedirs(out, exist_ok=True)
    print(f"[{modality}] {len(train_eps)} train / {len(val_eps)} val / {len(test_eps)} test episodes, cameras {cams}, "
          f"{len(tr)} train / {len(va) if va else 0} val / {len(te)} test samples, "
          f"input {in_ch}x{size[1]}x{size[0]} ({win} ticks = {win * TICK:.2f} s)", flush=True)
    if tr.skipped or te.skipped:
        print(f"  note: {tr.skipped + te.skipped} episode/camera pairs without events or frames were left out "
              f"(they are left out for every modality)", flush=True)
    # class weights against imbalance (from the training labels; same for every modality)
    y_tr = tr.labels(args.app)
    cnt = np.bincount(y_tr, minlength=n_cls).astype(np.float64)
    wts = torch.tensor(np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0.0), dtype=torch.float32, device=dev)
    majority = int(cnt.argmax())
    # regression targets are standardised with training statistics
    if n_reg:
        Rall = np.array([np.r_[tr.targets[e][1][k],
                               tr.targets[e][2][k].reshape(-1) if robot_camera(c) else np.full(6, np.nan)]
                         for e, c, ci, k in tr.items], np.float32)
        r_mu = np.nan_to_num(np.nanmean(Rall, 0)).astype(np.float32)
        r_sd = np.nan_to_num(np.nanstd(Rall, 0), nan=1.0).astype(np.float32)
        r_sd[r_sd < 1e-3] = 1.0
    else:
        r_mu, r_sd = np.zeros(1, np.float32), np.ones(1, np.float32)
    mu_t, sd_t = torch.tensor(r_mu, device=dev), torch.tensor(r_sd, device=dev)

    model = build_model(in_ch, n_cls, n_reg, len(cams)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * max(1, math.ceil(len(tr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    rng = np.random.default_rng(args.seed)
    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(tr))
        tot = n = 0
        for b in range(0, len(perm), args.batch):
            X, C, Y, R = (v.to(dev) for v in tr.batch(perm[b:b + args.batch], args.app))
            if args.augment and args.app == "actions" and rng.random() < 0.5:
                X = torch.flip(X, [3])                 # left-right mirror (action labels are symmetric)
            lc, lr = model(X, C)
            loss = F.cross_entropy(lc, Y, weight=wts, label_smoothing=args.label_smoothing)
            if n_reg:
                Rn = (R - mu_t) / sd_t
                m = torch.isfinite(Rn)
                if m.any():
                    loss = loss + args.reg_weight * F.smooth_l1_loss(lr[m], Rn[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(Y)
            n += len(Y)
        line = f"[{modality}] epoch {epoch + 1}/{args.epochs}  train loss {tot / max(1, n):.4f}"
        if va is not None and len(va):
            P, G, _, _, _ = predict(model, va, args.app, args.batch, dev)
            f1 = macro_f1(confusion(P, G, n_cls))
            history.append(dict(epoch=epoch + 1, train_loss=tot / max(1, n), val_macro_f1=f1,
                                val_accuracy=float((P == G).mean())))
            line += f"  val acc {history[-1]['val_accuracy']:.3f}  val macro-F1 {f1:.3f}"
            if f1 > best:
                best, best_epoch, bad = f1, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                line += "  *"
            else:
                bad += 1
        print(line, flush=True)
        if va is not None and args.patience and bad >= args.patience:
            print(f"[{modality}] early stop: no val improvement for {args.patience} epochs", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(dict(model=model.state_dict(), cams=cams, size=size, window=win, names=names, app=args.app,
                    modality=modality, reg_mean=r_mu, reg_std=r_sd, best_epoch=best_epoch),
               os.path.join(out, "model.pt"))

    # ---------------- test (best-on-validation checkpoint)
    P, G, RP, RT, keys = predict(model, te, args.app, args.batch, dev)
    cm = confusion(P, G, n_cls)
    maj = np.full_like(G, majority)
    res = dict(app=args.app, modality=modality, input_channels=in_ch, window_ticks=win,
               window_s=win * TICK, epochs_run=len(history) or args.epochs, best_epoch=best_epoch or None,
               n_train=len(tr), n_val=len(va) if va else 0, n_test=int(len(G)),
               test_episodes=[os.path.basename(e) for e in test_eps],
               accuracy=float((P == G).mean()), macro_f1=macro_f1(cm), **class_report(cm, names),
               majority_baseline=dict(cls=names[majority], accuracy=float((maj == G).mean()),
                                      macro_f1=macro_f1(confusion(maj, G, n_cls))),
               confusion=cm.tolist(), classes=names, history=history)
    if args.app == "safety":
        RP = RP * r_sd + r_mu                        # back to metres
        res["clearance_mae_m"] = float(np.nanmean(np.abs(RP[:, 0] - RT[:, 0])))
        res["clearance_mae_mean_baseline_m"] = float(np.nanmean(np.abs(r_mu[0] - RT[:, 0])))
        robot = np.array([robot_camera(k[1]) for k in keys])
        fp, ft = RP[:, 1:].reshape(-1, 3, 2), RT[:, 1:].reshape(-1, 3, 2)
        if robot.any():
            err = np.linalg.norm(fp[robot] - ft[robot], axis=2)
            cv = np.stack([te.targets[k[0]][3][k[3]] for k in keys])[robot]
            cerr = np.linalg.norm(cv - ft[robot], axis=2)
            merr = np.linalg.norm(r_mu[1:].reshape(3, 2) - ft[robot], axis=2)
            res["trajectory_robot_cams"] = dict(
                ADE_m=float(np.nanmean(err)), FDE_1p5s_m=float(np.nanmean(err[:, -1])),
                const_velocity_from_true_state=dict(ADE_m=float(np.nanmean(cerr)), FDE_1p5s_m=float(np.nanmean(cerr[:, -1]))),
                mean_position_baseline=dict(ADE_m=float(np.nanmean(merr)), FDE_1p5s_m=float(np.nanmean(merr[:, -1]))))
        res["per_camera"] = {}
        for c in cams:
            m = np.array([k[1] == c for k in keys])
            if m.any():
                cmc = confusion(P[m], G[m], n_cls)
                res["per_camera"][c] = dict(accuracy=float((P[m] == G[m]).mean()), macro_f1=macro_f1(cmc),
                                            clearance_mae_m=float(np.nanmean(np.abs(RP[m, 0] - RT[m, 0]))))
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
        res["per_camera"] = {}
        for c in cams:
            m = np.array([k[1] == c for k in keys])
            if m.any():
                res["per_camera"][c] = dict(accuracy=float((P[m] == G[m]).mean()),
                                            macro_f1=macro_f1(confusion(P[m], G[m], n_cls)))
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("confusion", "history", "test_episodes")}, indent=1))
    print("confusion (rows = truth):")
    for name, row in zip(names, cm):
        print(f"  {name:>10s} " + " ".join(f"{v:6d}" for v in row))
    return res


def comparison_table(results, app):
    """Markdown table of the modalities side by side."""
    def g(r, *path):
        for p in path:
            r = r.get(p) if isinstance(r, dict) else None
        return r

    def f(v, pct=False):
        return "-" if v is None else (f"{100 * v:.1f}" if pct else f"{v:.3f}" if isinstance(v, float) else str(v))

    mods = [m for m in MODALITIES if m in results]
    rows = [("input channels", lambda r: f(r["input_channels"])),
            ("window [s]", lambda r: f"{r['window_s']:.2f}"),
            ("best epoch", lambda r: f(r["best_epoch"])),
            ("accuracy [%]", lambda r: f(r["accuracy"], True)),
            ("macro-F1 [%]", lambda r: f(r["macro_f1"], True)),
            ("balanced accuracy [%]", lambda r: f(r["balanced_accuracy"], True))]
    names = results[mods[0]]["classes"]
    rows += [(f"recall {n} [%]", (lambda n: lambda r: f(r["recall_per_class"][n], True))(n)) for n in names]
    if app == "safety":
        rows += [("clearance MAE [m]", lambda r: f(r["clearance_mae_m"])),
                 ("ADE robot cams [m]", lambda r: f(g(r, "trajectory_robot_cams", "ADE_m"))),
                 ("FDE 1.5 s robot cams [m]", lambda r: f(g(r, "trajectory_robot_cams", "FDE_1p5s_m")))]
    else:
        p1 = lambda v: "-" if v is None else f"{v:.1f}"          # already in percent
        rows += [("frame acc (smoothed) [%]", lambda r: p1(g(r, "segmentation", "frame_acc"))),
                 ("edit", lambda r: p1(g(r, "segmentation", "edit"))),
                 ("F1@10 / 25 / 50", lambda r: " / ".join(p1(g(r, "segmentation", k)) for k in ("f1_10", "f1_25", "f1_50")))]
        rows += [(f"early {k}", (lambda k: lambda r: f(g(r, "early_recognition_accuracy", k), True))(k))
                 for k in ("10%", "20%", "30%", "50%", "75%", "100%")]
    r0 = results[mods[0]]
    lines = [f"# {app}: events vs RGB vs fusion (same samples, model, window, schedule, seed)", "",
             "| metric | " + " | ".join(mods) + " |", "|---|" + "---|" * len(mods)]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(results[m]) for m in mods) + " |")
    mb = r0["majority_baseline"]
    lines += ["", f"Majority-class baseline ({mb['cls']}): accuracy {100 * mb['accuracy']:.1f} %, "
              f"macro-F1 {100 * mb['macro_f1']:.1f} %."]
    if app == "safety":
        lines.append(f"Mean-clearance baseline: MAE {r0['clearance_mae_mean_baseline_m']:.3f} m.")
        t = r0.get("trajectory_robot_cams")
        if t:
            cv, mp = t["const_velocity_from_true_state"], t["mean_position_baseline"]
            lines.append(f"Trajectory references (robot cams): constant velocity from the TRUE positions "
                         f"ADE {cv['ADE_m']:.3f} / FDE {cv['FDE_1p5s_m']:.3f} m; mean position "
                         f"ADE {mp['ADE_m']:.3f} / FDE {mp['FDE_1p5s_m']:.3f} m.")
    lines += ["", f"Test episodes: {', '.join(r0['test_episodes'])} ({r0['n_test']} samples per modality); "
              f"{r0['n_train']} train / {r0['n_val']} val samples.", ""]
    lines.append("Per camera (accuracy / macro-F1 %):")
    lines.append("")
    lines.append("| camera | " + " | ".join(mods) + " |")
    lines.append("|---|" + "---|" * len(mods))
    for c in r0["per_camera"]:
        lines.append(f"| {c} | " + " | ".join(
            f"{100 * results[m]['per_camera'][c]['accuracy']:.1f} / {100 * results[m]['per_camera'][c]['macro_f1']:.1f}"
            for m in mods) + " |")
    return "\n".join(lines) + "\n"


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
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of the training episodes held out for model selection / early stopping")
    ap.add_argument("--val-stride", type=int, default=3)
    ap.add_argument("--patience", type=int, default=6, help="early-stopping patience in epochs (0 = off)")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--reg-weight", type=float, default=1.0)
    ap.add_argument("--modality", choices=list(MODALITIES) + ["all"], default="events",
                    help="events, rgb (the synchronised RGB frames of the same ticks), both (early fusion), "
                         "or all = train and test the three one after the other and write comparison.md")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.modality != "all":
        run(args)
        return
    results = {m: run(args, m) for m in MODALITIES}
    table = comparison_table(results, args.app)
    with open(os.path.join(args.out, "comparison.md"), "w") as f:
        f.write(table)
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump({m: {k: v for k, v in r.items() if k != "history"} for m, r in results.items()}, f, indent=1)
    print()
    print(table)


if __name__ == "__main__":
    main()
