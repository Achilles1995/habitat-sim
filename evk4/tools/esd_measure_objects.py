"""
Measure the real ESD objects from dataset frames and extract their top-face textures, so the MuJoCo
replicas match the real ones (sizes + appearance).

    python tools/esd_measure_objects.py --rgb ESD/.../RGB/images/14.png --mask ESD/.../RGB/masks/14.png \
        --height 0.82 [--depth depth_14.png (uint16 mm, aligned to colour)] --out esd_measured
    -> esd_measured/objects.json + esd_measured/<object>.png
    python tools/esd_generate.py ... --measured esd_measured/objects.json

Use frames where objects are unoccluded and the camera is slow (sharp image).  Without a depth map the
object top is assumed at (camera height - catalogue height of that object).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from evk4_mujoco.esd import catalog, data            # noqa: E402
from evk4_mujoco.esd.rig import ESD_CALIB           # noqa: E402


def main():
    import cv2
    from PIL import Image
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgb", required=True); ap.add_argument("--mask", required=True)
    ap.add_argument("--depth", default=None); ap.add_argument("--height", type=float, default=0.82)
    ap.add_argument("--set", default="auto", choices=["auto", "esd1", "esd2"])
    ap.add_argument("--out", default="esd_measured")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rgb = np.array(Image.open(a.rgb).convert("RGB"))
    mimg = np.array(Image.open(a.mask).convert("RGB"))
    pal = None if a.set == "auto" else (catalog.ESD1_PALETTE if a.set == "esd1" else catalog.ESD2_PALETTE)
    lab = data.mask_from_color(mimg, pal)
    is2 = pal is catalog.ESD2_PALETTE or (pal is None and all(
        np.any(np.all(mimg == c, -1)) is False for c in catalog.ESD1_PALETTE.values()) and lab.max() <= 5)
    objs = {o.label: o for o in (catalog.ESD2 if is2 else catalog.ESD1)}
    K = ESD_CALIB["d435"]["K"]
    f = 0.5 * (K[0, 0] + K[1, 1])
    depth = np.array(Image.open(a.depth)).astype(float) / 1000.0 if a.depth else None
    if depth is not None and depth.shape != lab.shape:
        depth = cv2.resize(depth, lab.shape[::-1], interpolation=cv2.INTER_NEAREST)
    out = {}
    if os.path.exists(os.path.join(a.out, "objects.json")):
        out = json.load(open(os.path.join(a.out, "objects.json")))
    for k in sorted(set(np.unique(lab)) - {0}):
        if k not in objs:
            continue
        o = objs[k]
        bm = (lab == k).astype(np.uint8)
        cnts, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
        cat_h = o.size[2] if o.shape in ("box", "pouch") else (o.size[0] if "lying" in o.shape else o.size[1])
        if depth is not None and np.any((depth > 0) & (bm > 0)):
            z_top = float(np.median(depth[(bm > 0) & (depth > 0)]))
            ring = cv2.dilate(bm, np.ones((25, 25), np.uint8)) - cv2.dilate(bm, np.ones((9, 9), np.uint8))
            tz = depth[(ring > 0) & (lab == 0) & (depth > 0)]
            obj_h = float(np.median(tz) - z_top) if len(tz) else cat_h
        else:
            z_top, obj_h = a.height - cat_h, cat_h
        s = z_top / f                                                # m per pixel at the object top
        L, W = max(w, h) * s, min(w, h) * s
        circ = 4 * np.pi * cv2.contourArea(cnt) / cv2.arcLength(cnt, True) ** 2
        if o.shape == "box":
            size = [L, W, obj_h]
        elif o.shape in ("cylinder_lying", "bottle_lying"):
            size = [W, L]
        elif o.shape in ("bottle", "cup", "disc", "cylinder"):
            size = [0.5 * (L + W), obj_h if o.shape != "bottle" else max(obj_h, o.size[1])]
        elif o.shape == "pouch":
            size = [L, W, obj_h]
        else:
            size = list(o.size)
        box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.float32)
        if w < h:                                                    # long side horizontal in the texture
            box = np.roll(box, 1, 0)
        tw, th = 256, max(16, int(256 * min(w, h) / max(w, h)))
        M = cv2.getPerspectiveTransform(box, np.float32([[0, th], [0, 0], [tw, 0], [tw, th]]))
        tex = cv2.warpPerspective(rgb, M, (tw, th))
        tpath = os.path.abspath(os.path.join(a.out, f"{o.name}.png"))
        Image.fromarray(cv2.resize(tex, (256, 256))).save(tpath)
        out[o.name] = dict(label=int(k), size=[round(float(x), 4) for x in size], texture=tpath,
                           circularity=round(float(circ), 2), catalog_size=list(o.size))
        print(f"{k:2d} {o.name:14s} {' x '.join(f'{x * 100:.1f}' for x in size)} cm   (catalogue "
              f"{' x '.join(f'{x * 100:.1f}' for x in o.size)})")
    json.dump(out, open(os.path.join(a.out, "objects.json"), "w"), indent=1)
    print("->", os.path.join(a.out, "objects.json"))


if __name__ == "__main__":
    main()
