"""
Closed-loop check of the calibration tool: synthesise "real" EVT3 recordings with KNOWN pixel
parameters, run tools/calibrate_evk4.py on them, and compare the recovered parameters.
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from evk4_mujoco.calibration import simulate, calibrate      # noqa: E402
from evk4_mujoco.io import write_evt3                        # noqa: E402

TRUE = dict(contrast_on=0.28, contrast_off=0.18, contrast_sigma=0.03, cutoff_hz=1200, bw_half_lux=60,
            noise_gain=0.10, leak_rate_hz=0.2)
EXPS = [
    dict(type="ramp", file="ramp_up.raw", lux=500, factor=3.0, t0_us=20000, t1_us=320000, direction="up"),
    dict(type="ramp", file="ramp_dn.raw", lux=500, factor=3.0, t0_us=20000, t1_us=320000, direction="down"),
    dict(type="step", file="step10.raw", lux=10, factor=2.0, t_step_us=2000),
    dict(type="step", file="step1000.raw", lux=1000, factor=2.0, t_step_us=2000),
    dict(type="static", file="static5.raw", lux=5, t0_us=50000, t1_us=1050000),
    dict(type="static", file="static2000.raw", lux=2000, t0_us=50000, t1_us=2050000),
]

with tempfile.TemporaryDirectory() as d:
    for e in EXPS:
        ev, _ = simulate(e, TRUE, seed=777)            # different seed = different "device"
        write_evt3(os.path.join(d, e["file"]), ev, 64, 36)
    json.dump(dict(reflectance=0.5, experiments=EXPS), open(os.path.join(d, "exp.json"), "w"))
    fitted = calibrate(os.path.join(d, "exp.json"))
ok = True
print("\nparameter        true      fitted    rel.err")
for k, v in TRUE.items():
    f = fitted.get(k, float("nan"))
    err = abs(f - v) / v
    tol = 0.5 if k in ("contrast_sigma", "leak_rate_hz", "bw_half_lux") else 0.2
    ok &= err < tol
    print(f"{k:15s} {v:8.3f}  {f:8.3f}   {100*err:5.1f}%  {'ok' if err < tol else 'X'}")
print("ALL PASSED" if ok else "SOME CHECKS FAILED")
