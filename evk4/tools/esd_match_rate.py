"""
Fit the simulated DAVIS346 contrast threshold so that simulated event rates match REAL ESD sequences
of the same conditions (objects, speed, trajectory, light, height).  This is the most important
sim-to-real knob for the event stream; run it once on a few real sequences.

    python tools/esd_match_rate.py --real ESD/ESD-1/10_obj_015speed_linear_goodlight_082height \
                                   --real ESD/ESD-1/2_obj_1speed_linear_lowlight_062height --backend warp
    -> writes esd_sensor_fit.json  ->  python tools/esd_generate.py ... --sensor-json esd_sensor_fit.json

If a folder name does not follow the ESD pattern, pass the attributes: --n-obj --speed --traj --light --height.
"""
import argparse
import json
import os
import re
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np                                    # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from evk4_mujoco.esd import data, generate as G      # noqa: E402

PAT = re.compile(r"(\d+)_obj_(\d+)speed_([a-z]+)_([a-z]+)light_(\d+)height")


def parse(name):
    mm = PAT.search(name)
    if not mm:
        return None
    n, spd, traj, light, h = mm.groups()
    speed = {"015": 0.15, "03": 0.3, "1": 1.0}.get(spd, float(spd))
    traj = {"rotary": "rotation", "general": "linearrotation", "rotational": "rotation"}.get(traj, traj)
    return dict(n_obj=int(n), speed=speed, traj=traj, light="low" if light.startswith("low") else "normal",
                height=int(h) / 100.0)


def real_rate(seq_dir, side, window_s=0.1):
    ev = data.Sequence(seq_dir).events(side)
    t = ev["t"] * 1e-6
    counts = np.histogram(t, bins=np.arange(0, t.max() + window_s, window_s))[0] / window_s
    return float(np.median(counts[counts > np.percentile(counts, 20)]))     # while moving


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="append", required=True, help="real ESD sequence folder (repeatable)")
    ap.add_argument("--side", default="left")
    ap.add_argument("--n-obj", type=int); ap.add_argument("--speed", type=float); ap.add_argument("--traj")
    ap.add_argument("--light"); ap.add_argument("--height", type=float)
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--backend", default="cpu"); ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", default="esd_sensor_fit.json")
    a = ap.parse_args()
    cases = []
    for r in a.real:
        attrs = parse(os.path.basename(os.path.normpath(r))) or {}
        for k in ("n_obj", "speed", "traj", "light", "height"):
            if getattr(a, k) is not None:
                attrs[k] = getattr(a, k)
        if len(attrs) < 5:
            sys.exit(f"cannot parse conditions of {r} - pass --n-obj --speed --traj --light --height")
        rr = real_rate(r, a.side)
        print(f"real {os.path.basename(r)}: {rr / 1e3:.1f} kev/s while moving  {attrs}")
        cases.append((attrs, rr))

    def sim_ratio(c):
        ratios = []
        for attrs, rr in cases:
            p = G.SeqParams(object_set="esd1", duration=a.duration, backend=a.backend, device=a.device,
                            occlusion=attrs["n_obj"] > 2, sensor_overrides=dict(contrast_on=c, contrast_off=c),
                            **attrs)
            rates = G.event_rate(p, side=a.side)
            sr = float(np.median(rates[rates > np.percentile(rates, 20)]))
            ratios.append(sr / rr)
            print(f"   contrast {c:.3f}: sim {sr / 1e3:.1f} kev/s  (x{sr / rr:.2f} of real)", flush=True)
        return float(np.exp(np.mean(np.log(ratios))))

    lo, hi = np.log(0.08), np.log(1.5)                # bisection on log(contrast); rate falls with contrast
    for _ in range(a.iters):
        mid = 0.5 * (lo + hi)
        if sim_ratio(float(np.exp(mid))) > 1.0:
            lo = mid
        else:
            hi = mid
    c = float(np.exp(0.5 * (lo + hi)))
    json.dump(dict(contrast_on=c, contrast_off=c), open(a.out, "w"), indent=1)
    print(f"best contrast threshold {c:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
