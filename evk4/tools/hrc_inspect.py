#!/usr/bin/env python3
"""
Check / visualise the synchronised RGB + event data of an episode written by hrc_generate.py.

For a few frames it overlays the events of the 33 ms before the frame time on the RGB frame
(ON white, OFF blue, human events orange) and measures alignment: the fraction of events labelled
as a person (instance id) that fall on that person's pixels in the RGB frame's instance mask.

  python evk4/tools/hrc_inspect.py datasets/actions/ep0000 --cam spot_head --out inspect.png
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--cam", default=None)
    ap.add_argument("--n", type=int, default=6, help="frames in the overlay strip")
    ap.add_argument("--window-ms", type=float, default=33.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import cv2
    import h5py

    meta = json.load(open(os.path.join(args.episode, "meta.json")))
    cam = args.cam or sorted(meta["cameras"])[0]
    if cam not in meta["cameras"]:
        print(f"{cam}: not recorded in this episode (cameras: {sorted(meta['cameras'])})")
        return float("nan")
    human_ids = [h["instance_id"] for h in meta["humans"]]
    with h5py.File(os.path.join(args.episode, f"events_{cam}.h5"), "r") as f:
        e = {k: f["events"][k][:] for k in ("x", "y", "p", "t", "instance")}
        W, H = int(f.attrs["width"]), int(f.attrs["height"])
    with h5py.File(os.path.join(args.episode, f"frames_{cam}.h5"), "r") as f:
        tf = f["t_us"][:]
        aligned = bool(f.attrs["pixel_aligned_with_events"])
        Wr, Hr = int(f.attrs["width"]), int(f.attrs["height"])
        picks = np.linspace(len(tf) * 0.15, len(tf) - 1, args.n).astype(int)
        tiles, hits, tot = [], 0, 0
        for i in picks:
            rgb = f["rgb"][i]
            inst = f["instance"][i] if "instance" in f else None
            t1 = tf[i]
            sel = (e["t"] > t1 - args.window_ms * 1e3) & (e["t"] <= t1)
            x, y, p, g = e["x"][sel], e["y"][sel], e["p"][sel], e["instance"][sel]
            if (Wr, Hr) != (W, H):            # different grids: scale event pixels for the overlay
                x = (x * Wr // W).astype(np.int64)
                y = (y * Hr // H).astype(np.int64)
            img = (rgb.astype(np.float32) * 0.55).astype(np.uint8)
            hum = np.isin(g, human_ids)
            img[y[(p == 0) & ~hum], x[(p == 0) & ~hum]] = (64, 126, 200)
            img[y[(p == 1) & ~hum], x[(p == 1) & ~hum]] = (255, 255, 255)
            img[y[hum], x[hum]] = (255, 160, 30)
            if inst is not None and aligned and hum.any():
                # an event at the edge of a moving person may land just outside the end-of-window mask
                m = np.isin(inst, human_ids)
                m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                hits += int(m[y[hum], x[hum]].sum())
                tot += int(hum.sum())
            cv2.putText(img, f"t={t1 / 1e6:.3f}s  {int(sel.sum())} ev", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(img)
    frac = hits / tot if tot else float("nan")
    dt = np.diff(tf) / 1e3
    print(f"{cam}: {len(tf)} RGB frames {Wr}x{Hr} every {np.median(dt):.1f} ms, events {W}x{H} "
          f"({len(e['t'])} events), pixel-aligned {aligned}; human events on the human mask (2 px): {frac:.1%}")
    if args.out:
        strip = np.concatenate(tiles, 1)
        cv2.imwrite(args.out, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        print("wrote", args.out)
    return frac


if __name__ == "__main__":
    main()
