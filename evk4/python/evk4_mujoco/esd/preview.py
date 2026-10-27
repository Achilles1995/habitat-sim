"""Preview video of an ESD sequence (sim or real): RGB | mask  /  left events (labelled) | right events."""
from __future__ import annotations

import numpy as np

from .catalog import ESD1_PALETTE, ESD2_PALETTE
from .data import Sequence


def _label_rgb(lab, palette):
    out = np.zeros(lab.shape + (3,), np.uint8)
    for k, c in palette.items():
        out[lab == k] = c
    return out


def labeled_event_image(e, H, W, palette, bg=(240, 240, 240), table=(40, 110, 200)):
    """Fig. 6 style: background events blue, object events in their label colour."""
    img = np.full((H, W, 3), bg, np.uint8)
    lab = e["label"]
    img[e["y"][lab == 0], e["x"][lab == 0]] = table
    for k, c in palette.items():
        s = lab == k
        img[e["y"][s], e["x"][s]] = c
    return img


def preview(seq_dir, out_mp4, fps=10, window_us=60000):
    import imageio.v2 as imageio
    seq = Sequence(seq_dir)
    frames = seq.rgb_frames()
    evl, evr = seq.events("left"), seq.events("right")
    pal = ESD2_PALETTE if "esd2" in seq_dir else ESD1_PALETTE
    W = 346; H = 260
    w = imageio.get_writer(out_mp4, fps=fps, quality=8, macro_block_size=1)
    t_end = max(int(evl["t"][-1]) if len(evl) else 0, 1)
    n = len(frames)
    ts = seq.image_times("left")
    for k, (rgb, lab) in enumerate(frames):
        t = int(ts[k]) if ts is not None and k < len(ts) else int(k * t_end / max(1, n - 1))
        el = evl[(evl["t"] > t - window_us) & (evl["t"] <= t)] if len(evl) else evl
        er = evr[(evr["t"] > t - window_us) & (evr["t"] <= t)] if len(evr) else evr
        top = np.concatenate([rgb, _label_rgb(lab if lab is not None else np.zeros(rgb.shape[:2], np.uint8), pal)], 1)
        bl = labeled_event_image(el, H, W, pal); br = labeled_event_image(er, H, W, pal)
        bot = np.concatenate([bl, br], 1)
        bot = np.kron(bot, np.ones((1, 1, 1), np.uint8))
        pad = np.zeros((bot.shape[0], top.shape[1] - bot.shape[1], 3), np.uint8)
        w.append_data(np.concatenate([top, np.concatenate([bot, pad], 1)], 0))
    w.close()
    return out_mp4
