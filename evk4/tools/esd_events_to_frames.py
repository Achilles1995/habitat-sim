"""
Convert ESD event streams into frames + segmentation masks - the SAME code for simulated sequences
(tools/esd_generate.py) and the ORIGINAL ESD dataset, so training and test data are built identically.

    python tools/esd_events_to_frames.py D:\\ESD-1 --out frames_real_esd1
    python tools/esd_events_to_frames.py sim_esd\\train_esd1 --out frames_sim_esd1
    python tools/esd_events_to_frames.py D:\\ESD-1\\10_obj_015speed_linear_goodlight_082height --side left right

Input: one sequence folder or any folder containing many (searched recursively for left.mat / right.mat).

Per sequence and camera:  <out>/<sequence>/<side>/
    frames/<k>.png        event frame: ON = red, OFF = green on black (as ESD's generate_event_frame.m)
    labels/<k>.png        label ids of the events at each pixel (uint8; 0 background, 255 = no event)
    labels_color/<k>.png  Fig. 6 style: object events in their ESD colour, background events blue
    data.npz              x (N,C,H,W) float16 network input, y (N,H,W) uint8 labels, t_start/t_end [us]

Windows: fixed time slices (--window-ms, default 60 = the RGB-D period), fixed event counts
(--n-events, e.g. 3000 as in the paper's figures), or slices ending at the RGB frames (--align rgb).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from evk4_mujoco.esd import data                     # noqa: E402
from evk4_mujoco.esd.catalog import ESD1_PALETTE, ESD2_PALETTE  # noqa: E402

H, W = 260, 346


def event_frame_png(e):
    img = np.zeros((H, W, 3), np.uint8)
    on = e["p"] > 0
    img[e["y"][on], e["x"][on]] = (255, 0, 0)
    img[e["y"][~on], e["x"][~on]] = (0, 255, 0)
    return img


def label_color_png(e, palette):
    img = np.full((H, W, 3), 240, np.uint8)
    bg = e["label"] == 0
    img[e["y"][bg], e["x"][bg]] = (40, 110, 200)
    for k, c in palette.items():
        s = e["label"] == k
        img[e["y"][s], e["x"][s]] = c
    return img


def windows(seq, side, a):
    ev = seq.events(side)
    if len(ev) == 0:
        return ev, []
    if a.n_events:
        starts = range(0, len(ev), a.n_events)
        return ev, [(i, min(i + a.n_events, len(ev))) for i in starts]
    w = int(a.window_ms * 1000)
    if a.align == "rgb":
        ts = seq.image_times(side)
        if ts is None or len(ts) == 0:
            print("   (no RGB timestamps in the .mat - using fixed windows)")
        else:
            ends = np.asarray(ts, np.int64)
            a_ = np.searchsorted(ev["t"], ends - w, side="right")
            b_ = np.searchsorted(ev["t"], ends, side="right")
            return ev, list(zip(a_, b_))
    edges = np.arange(int(ev["t"][0]), int(ev["t"][-1]) + w, w)
    pos = np.searchsorted(ev["t"], edges)
    return ev, list(zip(pos[:-1], pos[1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="sequence folder(s) or a folder containing sequences")
    ap.add_argument("--out", default="esd_frames")
    ap.add_argument("--side", nargs="+", default=["left"], choices=["left", "right"])
    ap.add_argument("--window-ms", type=float, default=60.0)
    ap.add_argument("--n-events", type=int, default=0, help="fixed number of events per frame instead")
    ap.add_argument("--align", default="fixed", choices=["fixed", "rgb"], help="rgb: windows end at RGB frames")
    ap.add_argument("--repr", default="frame2", choices=["frame2", "frame1", "voxel", "timesurface"],
                    help="network input stored in data.npz")
    ap.add_argument("--min-events", type=int, default=100, help="skip nearly empty windows")
    ap.add_argument("--max-frames", type=int, default=0, help="per sequence (0 = all)")
    ap.add_argument("--no-png", action="store_true", help="only data.npz")
    ap.add_argument("--class-agnostic", action="store_true", help="all objects -> label 1 (unseen-object task)")
    ap.add_argument("--skip-existing", action="store_true", help="do not redo sequences already converted")
    a = ap.parse_args()
    from PIL import Image

    seqs = []
    for inp in a.inputs:
        found = data.find_sequences(inp)
        if not found:
            print(f"no left.mat / right.mat found under {inp}")
        seqs += found
    print(f"{len(seqs)} sequence(s)")
    bad = []
    for seq in seqs:
        name = os.path.basename(os.path.normpath(seq.root))
        pal = ESD2_PALETTE if ("esd2" in seq.root.lower() or "ESD-2" in seq.root) else ESD1_PALETTE
        for side in a.side:
            od = os.path.join(a.out, name, side)
            if a.skip_existing and os.path.exists(os.path.join(od, "data.npz")):
                print(f"{name} [{side}]: already converted - skipped")
                continue
            try:
                ev, wins = windows(seq, side, a)
            except FileNotFoundError:
                print(f"{name}: no {side}.mat - skipped")
                continue
            except Exception as ex:                          # damaged / truncated file: report and go on
                import traceback
                tb = traceback.extract_tb(ex.__traceback__)[-1]
                print(f"{name} [{side}]: CANNOT READ {side}.mat ({type(ex).__name__}: {ex}; at "
                      f"{os.path.basename(tb.filename)}:{tb.lineno}) - skipped")
                bad.append(f"{seq.root} [{side}]")
                continue
            for sub in ("frames", "labels", "labels_color"):
                if not a.no_png:
                    os.makedirs(os.path.join(od, sub), exist_ok=True)
            os.makedirs(od, exist_ok=True)
            X, Y, T0, T1 = [], [], [], []
            k = 0
            for i0, i1 in wins:
                e = ev[i0:i1]
                if len(e) < a.min_events:
                    continue
                if a.class_agnostic:
                    e = e.copy()
                    e["label"] = np.where(e["label"] > 0, 1, 0)
                y = data.event_label_image(e, H, W)
                X.append(data.event_repr(e, a.repr, H, W).astype(np.float16))
                Y.append(y)
                T0.append(int(e["t"][0])); T1.append(int(e["t"][-1]))
                if not a.no_png:
                    Image.fromarray(event_frame_png(e)).save(os.path.join(od, "frames", f"{k}.png"))
                    Image.fromarray(y).save(os.path.join(od, "labels", f"{k}.png"))
                    Image.fromarray(label_color_png(e, pal)).save(os.path.join(od, "labels_color", f"{k}.png"))
                k += 1
                if a.max_frames and k >= a.max_frames:
                    break
            if k:
                np.savez_compressed(os.path.join(od, "data.npz"), x=np.stack(X), y=np.stack(Y),
                                    t_start=np.array(T0), t_end=np.array(T1), repr=a.repr)
            n_obj = int(np.mean([np.sum(np.isin(y, [0, 255]) == 0) for y in Y])) if Y else 0
            print(f"{name} [{side}]: {len(ev)} events -> {k} frames  (avg {n_obj} object pixels/frame)  -> {od}")
    if bad:
        print(f"\n{len(bad)} file(s) could not be read (damaged or incomplete download / extraction):")
        for b in bad:
            print("   ", b)
        print("re-extract or re-download those sequences, then run again with --skip-existing")


if __name__ == "__main__":
    main()
