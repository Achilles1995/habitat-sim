"""
Train a U-Net for event-based tabletop segmentation on SIMULATED ESD data and test it on the REAL
ESD dataset (PyTorch, GPU).  Inputs are the data.npz files of tools/esd_events_to_frames.py, made
with the SAME options for sim and real.

    # 1. frames (once):
    python tools/esd_events_to_frames.py sim_esd/train_esd1 --out frames_sim --side left right --no-png
    python tools/esd_events_to_frames.py D:/ESD-1          --out frames_real --side left right --no-png
    # 2. train on sim, test on real ESD-1:
    python tools/esd_train_unet.py --train frames_sim --test frames_real --epochs 40 --out runs/unet_sim2real
    # test an existing checkpoint only:
    python tools/esd_train_unet.py --test frames_real --ckpt runs/unet_sim2real/best.pt --out runs/eval

Labels: 0 = background, 1..10 = ESD-1 objects (same ids in sim and real), 255 = pixel without
events (ignored in loss and metrics).  --class-agnostic maps all objects to 1 (unseen objects).
Metrics as in the paper: pixel accuracy (Eq. 6) and mIoU (Eq. 7), overall and per condition
(objects, speed, trajectory, light, height).  Model selection uses a held-out SIMULATED split only -
the real test data is never used for training or tuning.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

IGNORE = 255


# ---------------------------------------------------------------- data
def _cache(npz_path):
    """data.npz -> x.npy / y.npy next to it (uncompressed, memory-mapped afterwards)."""
    d = os.path.dirname(npz_path)
    xp, yp = os.path.join(d, "x.npy"), os.path.join(d, "y.npy")
    if not (os.path.exists(xp) and os.path.exists(yp)):
        z = np.load(npz_path)
        np.save(xp, z["x"].astype(np.float16)); np.save(yp, z["y"].astype(np.uint8))
    return xp, yp


def find_items(roots, exclude=()):
    """data.npz files under roots; a root may also be a text file listing sequence folders (one per line).
    Files whose path contains any `exclude` string are skipped."""
    files = []
    for r in roots:
        if os.path.isfile(r) and not r.endswith(".npz"):
            for line in open(r):
                line = line.strip()
                if line and not line.startswith("#"):
                    files += sorted(glob.glob(os.path.join(line, "**", "data.npz"), recursive=True))
        else:
            files += sorted(glob.glob(os.path.join(r, "**", "data.npz"), recursive=True))
    return [f for f in files if not any(e in f.replace("\\", "/") for e in exclude)]


class FrameSet(Dataset):
    def __init__(self, files, class_agnostic=False, augment=False, pad_to=16):
        self.files = [_cache(f) for f in files]
        self.names = [os.path.relpath(os.path.dirname(f)) for f in files]
        self.index = []
        for i, (xp, yp) in enumerate(self.files):
            n = np.load(yp, mmap_mode="r").shape[0]
            self.index += [(i, k) for k in range(n)]
        self.ca, self.aug, self.pad_to = class_agnostic, augment, pad_to
        self._mm = {}

    def __getstate__(self):                                         # DataLoader workers re-open the files
        st = dict(self.__dict__); st["_mm"] = {}
        return st

    def _arrays(self, i):
        if i not in self._mm:
            xp, yp = self.files[i]
            self._mm[i] = (np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r"))
        return self._mm[i]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        i, k = self.index[j]
        X, Y = self._arrays(i)
        x = np.asarray(X[k], np.float32)
        y = np.asarray(Y[k], np.int64)
        if self.ca:
            y = np.where((y > 0) & (y != IGNORE), 1, y)
        x = np.sign(x) * np.log1p(np.abs(x))                     # compress event counts
        if self.aug:                                                # sim -> real robustness
            rng = np.random
            if rng.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if rng.rand() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
            x = x * rng.uniform(0.7, 1.3)                           # event-rate variation
            if rng.rand() < 0.5:                                    # background activity noise
                noise = (rng.rand(*x.shape[1:]) < rng.uniform(0, 0.01)).astype(np.float32)
                ch = rng.randint(x.shape[0])
                x = x.copy(); x[ch] += noise * np.log1p(1.0)
            if rng.rand() < 0.5:                                    # missing events (dead / slow pixels)
                keep = rng.rand(*x.shape[1:]) > rng.uniform(0, 0.15)
                x = x * keep
                y = np.where(keep, y, IGNORE)                       # no event left -> no label
        x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        H, W = y.shape
        ph, pw = (-H) % self.pad_to, (-W) % self.pad_to
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
        y = np.pad(y, ((0, ph), (0, pw)), constant_values=IGNORE)
        return torch.from_numpy(x), torch.from_numpy(y), j


# ---------------------------------------------------------------- model
def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin, ncls, base=32, depth=4, dropout=0.1):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.inc = block(cin, ch[0])
        self.down = nn.ModuleList([block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([block(2 * ch[i], ch[i]) for i in reversed(range(depth))])
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(ch[0], ncls, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        h = self.drop(skips.pop())
        for up, dec in zip(self.up, self.dec):
            h = up(h)
            h = dec(torch.cat([skips.pop(), h], 1))
        return self.head(h)


# ---------------------------------------------------------------- metrics (paper Eq. 6 / 7)
class Confusion:
    def __init__(self, n):
        self.n = n
        self.m = torch.zeros(n, n, dtype=torch.int64)
        self.frame_miou = []

    def add(self, pred, gt):
        ok = gt != IGNORE
        for p, g, o in zip(pred, gt, ok):
            if o.sum() == 0:
                continue
            idx = g[o] * self.n + p[o]
            cm = torch.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n).cpu()
            self.m += cm
            inter = cm.diag().float(); union = cm.sum(0).float() + cm.sum(1).float() - inter
            present = union > 0
            self.frame_miou.append(float((inter[present] / union[present]).mean()))

    def scores(self):
        m = self.m.float()
        inter = m.diag(); union = m.sum(0) + m.sum(1) - inter
        present = union > 0
        iou = torch.where(present, inter / union.clamp(min=1), torch.full_like(inter, float("nan")))
        return dict(acc=float(inter.sum() / m.sum().clamp(min=1)), miou=float(iou[present].mean()),
                    miou_per_frame=float(np.mean(self.frame_miou)) if self.frame_miou else float("nan"),
                    iou_per_class=[None if np.isnan(v) else round(float(v), 4) for v in iou.tolist()],
                    pixels=int(m.sum()))


COND = re.compile(r"(\d+)_obj_(\d+)speed_([a-z]+)_([a-z]+)light_(\d+)height")


def conditions(name):
    mm = COND.search(name.replace("\\", "/"))
    if not mm:
        return {}
    n, s, tr, li, h = mm.groups()
    return {"objects": f"O{n}", "speed": f"S{s}", "trajectory": tr, "light": li, "height": f"H{h}"}


# ---------------------------------------------------------------- train / eval
def class_weights(ds, ncls, n_samples=400):
    cnt = np.zeros(ncls)
    for j in np.random.default_rng(0).choice(len(ds), min(n_samples, len(ds)), replace=False):
        y = ds[int(j)][1].numpy()
        cnt += np.bincount(y[y != IGNORE], minlength=ncls)[:ncls]
    f = cnt / max(1, cnt.sum())
    w = np.where(f > 0, np.median(f[f > 0]) / np.maximum(f, 1e-12), 0.0)    # median-frequency balancing
    return torch.tensor(np.clip(w, 0.05, 20.0), dtype=torch.float32)


def dice_loss(logits, y, ncls):
    ok = (y != IGNORE).unsqueeze(1).float()
    p = logits.float().softmax(1) * ok
    t = F.one_hot(torch.where(y == IGNORE, 0, y), ncls).permute(0, 3, 1, 2).float() * ok
    inter = (p * t).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    present = t.sum((0, 2, 3)) > 0
    return 1 - (2 * inter[present] / den[present].clamp(min=1e-6)).mean()


@torch.no_grad()
def evaluate(model, ds, dev, ncls, bs, workers, per_condition=True, vis_dir=None, n_vis=12):
    model.eval()
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    total = Confusion(ncls)
    groups = {}
    vis_left = n_vis
    for x, y, j in dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            pred = model(x).argmax(1)
        total.add(pred, y)
        if per_condition:
            for b in range(len(j)):
                c = conditions(ds.names[ds.index[int(j[b])][0]])
                for k, v in c.items():
                    groups.setdefault(f"{k}={v}", Confusion(ncls)).add(pred[b:b + 1], y[b:b + 1])
        if vis_dir and vis_left > 0:
            save_vis(vis_dir, x, y, pred, j, vis_left)
            vis_left -= len(j)
    return total.scores(), {k: v.scores() for k, v in sorted(groups.items())}


PAL = np.array([(40, 40, 40), (51, 221, 255), (250, 50, 83), (52, 209, 183), (255, 0, 124), (255, 96, 55),
                (221, 255, 51), (36, 179, 83), (184, 61, 245), (102, 255, 102), (255, 204, 51)] + [(255, 255, 255)] * 245,
               np.uint8)


def save_vis(d, x, y, pred, j, n):
    from PIL import Image
    os.makedirs(d, exist_ok=True)
    for b in range(min(n, len(j))):
        ev = x[b].float().cpu().numpy()
        img = np.zeros(ev.shape[1:] + (3,), np.uint8)
        img[..., 0] = np.clip(np.abs(ev[0]) * 120, 0, 255)
        img[..., 1] = np.clip(np.abs(ev[1 if ev.shape[0] > 1 else 0]) * 120, 0, 255)
        g = y[b].cpu().numpy(); p = pred[b].cpu().numpy().copy()
        p[g == IGNORE] = 255                                         # show predictions where events are
        gi, pi = PAL[np.minimum(g, 255)], PAL[np.minimum(p, 255)]
        gi[g == IGNORE] = 0; pi[g == IGNORE] = 0
        Image.fromarray(np.concatenate([img, gi, pi], 1)).save(os.path.join(d, f"{int(j[b]):06d}.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="*", default=[], help="frame folders of SIMULATED data")
    ap.add_argument("--test", nargs="*", default=[], help="frame folders of REAL ESD data")
    ap.add_argument("--out", default="runs/unet")
    ap.add_argument("--ckpt", default=None, help="evaluate this checkpoint (skip training if no --train)")
    ap.add_argument("--classes", type=int, default=11, help="11 = background + 10 ESD-1 objects")
    ap.add_argument("--class-agnostic", action="store_true", help="2 classes: background / object")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out SIMULATED sequences for model selection")
    ap.add_argument("--dice", type=float, default=0.5, help="weight of the Dice loss term")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="skip sequences whose path contains any of these strings (e.g. the ones used for tuning)")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA not available to PyTorch - install the CUDA build of torch (pytorch.org) or use --device cpu")
    dev = torch.device(a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    ncls = 2 if a.class_agnostic else a.classes
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("GPU:", torch.cuda.get_device_name(dev))

    model = None
    if a.train:
        files = find_items(a.train, a.exclude)
        if not files:
            sys.exit(f"no data.npz under {a.train} - run tools/esd_events_to_frames.py first")
        seqs = sorted({os.path.dirname(os.path.dirname(f)) for f in files})     # split by SEQUENCE
        rng = np.random.default_rng(a.seed)
        val_seqs = set(rng.choice(seqs, max(1, int(round(a.val_frac * len(seqs)))), replace=False)) if len(seqs) > 1 else set()
        tr_files = [f for f in files if os.path.dirname(os.path.dirname(f)) not in val_seqs]
        va_files = [f for f in files if os.path.dirname(os.path.dirname(f)) in val_seqs]
        print(f"preparing {len(files)} files (first run caches them as .npy) ...")
        tr = FrameSet(tr_files, a.class_agnostic, augment=True)
        va = FrameSet(va_files, a.class_agnostic) if va_files else None
        cin = tr[0][0].shape[0]
        print(f"train: {len(tr)} frames from {len(seqs) - len(val_seqs)} sim sequences | "
              f"val: {len(va) if va else 0} frames from {len(val_seqs)} | input channels {cin}, classes {ncls}")
        w = class_weights(tr, ncls).to(dev)
        print("class weights:", [round(float(v), 2) for v in w])
        model = UNet(cin, ncls, a.base).to(dev)
        if a.ckpt:
            model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True,
                        drop_last=True, persistent_workers=a.workers > 0)
        total_steps, warm = a.epochs * len(dl), max(1, int(0.05 * a.epochs * len(dl)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / warm) *
                                                  0.5 * (1 + np.cos(np.pi * min(1.0, k / max(1, total_steps)))))
        scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and not a.no_amp)
        best, log = -1.0, []
        for ep in range(a.epochs):
            model.train(); t0 = time.time(); tot = 0.0
            for x, y, _ in dl:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda" and not a.no_amp):
                    out = model(x)
                    loss = F.cross_entropy(out.float(), y, weight=w, ignore_index=IGNORE)
                    if a.dice > 0:
                        loss = loss + a.dice * dice_loss(out, y, ncls)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                tot += float(loss.detach())
            msg = f"epoch {ep + 1:3d}/{a.epochs}  loss {tot / len(dl):.4f}  {time.time() - t0:.0f}s"
            score = None
            if va:
                s, _ = evaluate(model, va, dev, ncls, a.batch, a.workers, per_condition=False)
                score = s["miou"]
                msg += f"  | sim-val Acc {100 * s['acc']:.2f}  mIoU {100 * s['miou']:.2f}"
            print(msg, flush=True)
            log.append(dict(epoch=ep + 1, loss=tot / len(dl), val=score))
            state = dict(model=model.state_dict(), cin=cin, ncls=ncls, base=a.base, args=vars(a))
            torch.save(state, os.path.join(a.out, "last.pt"))
            if score is None or score > best:
                best = score if score is not None else best
                torch.save(state, os.path.join(a.out, "best.pt"))
        json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
        a.ckpt = os.path.join(a.out, "best.pt")

    if a.test:
        if not a.ckpt:
            sys.exit("--test needs --train or --ckpt")
        st = torch.load(a.ckpt, map_location=dev)
        model = UNet(st["cin"], st["ncls"], st["base"]).to(dev)
        model.load_state_dict(st["model"])
        files = find_items(a.test, a.exclude)
        if not files:
            sys.exit(f"no data.npz under {a.test}")
        te = FrameSet(files, st["ncls"] == 2)
        if te[0][0].shape[0] != st["cin"]:
            sys.exit(f"test frames have {te[0][0].shape[0]} channels, model expects {st['cin']} - use the same --repr")
        print(f"testing {len(te)} REAL frames from {len(files)} files with {a.ckpt}")
        json.dump(sorted({os.path.dirname(os.path.dirname(f)) for f in files}),
                  open(os.path.join(a.out, "test_sequences.json"), "w"), indent=1)
        total, groups = evaluate(model, te, dev, st["ncls"], a.batch, a.workers,
                                 vis_dir=os.path.join(a.out, "test_vis"))
        print(f"\nREAL test:  Acc {100 * total['acc']:.2f} %   mIoU {100 * total['miou']:.2f} %   "
              f"(per-frame mIoU {100 * total['miou_per_frame']:.2f} %, {total['pixels']} labelled pixels)")
        print("IoU per class:", total["iou_per_class"])
        if groups:
            print("\nper condition (as Fig. 12 of the paper):")
            for k, s in groups.items():
                print(f"  {k:24s} Acc {100 * s['acc']:6.2f}   mIoU {100 * s['miou']:6.2f}")
        json.dump(dict(total=total, per_condition=groups, ckpt=a.ckpt, test=a.test),
                  open(os.path.join(a.out, "test_results.json"), "w"), indent=1)
        print(f"\nresults -> {os.path.join(a.out, 'test_results.json')}, examples -> {os.path.join(a.out, 'test_vis')}"
              "  (event frame | ground truth | prediction)")


if __name__ == "__main__":
    main()
