"""
Physics sanity checks of the EVK4 plugin on a static, uniform gray target (160x90 crop of the array).

  1. no light change, no noise sources      -> zero events
  2. light step x2                          -> ~ln(2)/theta ON events per pixel, no OFF events
  3. light step x0.5                        -> OFF events only
  4. latency of the first event after a step grows in low light (photoreceptor bandwidth ~ I)
  5. shot-noise background rate falls with illuminance
  6. 100 Hz flicker                         -> ON/OFF bursts at 100 Hz
Run:  python tests/test_physics.py
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import evk4_mujoco as evk  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "calib_scene.xml")
QUIET = dict(shot_noise=0, leak_rate_hz=0, hot_pixel_fraction=0, contrast_sigma=0,
             latency_jitter_us=0)


def make(dt=1e-3, **cfg):
    m = evk.load_model(SCENE, **cfg)
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, evk.EVK4(m)


def run(m, d, cam, T):
    out = []
    for _ in range(int(round(T / m.opt.timestep))):
        mujoco.mj_step(m, d)
        out.append(cam.events(d))
    return np.concatenate(out) if out else np.zeros(0, evk.EVENT_DTYPE)


def step_test(factor, lux, dt=10e-6, **cfg):
    m, d, cam = make(dt, lux_per_unit=lux, **QUIET, **cfg)
    run(m, d, cam, 2e-3)                           # settle
    t_step = d.time
    m.light_diffuse[0] *= factor                    # instantaneous illuminance change
    ev = run(m, d, cam, 8e-3)
    W, H = cam.geometry(d)
    lat = (np.median(ev["t"][:W * H]) - t_step * 1e6) if len(ev) else np.nan
    return ev, W * H, lat


def main():
    ok = True
    theta = 0.22

    m, d, cam = make(**QUIET, lux_per_unit=500)
    n = len(run(m, d, cam, 0.2))
    print(f"[1] static target, noise off          : {n} events  (expect 0)")
    ok &= n == 0

    # the memorised reference starts with a random phase D0 ~ U(-th, th) (5% margins); the expected
    # number of events for a ln(k) step is E[floor((D0 + ln k) / th)]
    u = (np.arange(100000) + 0.5) / 100000
    D0 = -theta + 2 * theta * (0.05 + 0.9 * u)
    exp_on = np.mean(np.maximum(0, np.floor((D0 + np.log(2)) / theta)))
    exp_off = np.mean(np.maximum(0, np.floor((np.log(2) - D0) / theta)))
    ev, npx, _ = step_test(2.0, 500, refractory_us=0)
    per_px = len(ev) / npx
    print(f"[2] light x2 step, no refractory      : {per_px:.2f} ON ev/px, OFF={np.sum(ev['p']==0)} "
          f"(expect {exp_on:.2f} = E[floor((D0+ln2)/theta)], 0 OFF)")
    ok &= abs(per_px - exp_on) < 0.1 and np.sum(ev["p"] == 0) == 0
    ev2, _, _ = step_test(2.0, 500, refractory_us=100)
    print(f"    same with 100 us refractory       : {len(ev2)/npx:.2f} ON ev/px (expect fewer: the reset "
          f"switch swallows part of the change)")
    ok &= len(ev2) < len(ev)

    ev, npx, _ = step_test(0.5, 500, refractory_us=0)
    print(f"[3] light x0.5 step, no refractory    : {len(ev)/npx:.2f} OFF ev/px, ON={np.sum(ev['p']==1)} "
          f"(expect {exp_off:.2f}, 0 ON)")
    ok &= np.sum(ev["p"] == 1) == 0 and abs(len(ev) / npx - exp_off) < 0.1

    lats = {}
    for lux in (10, 100, 1000):
        _, _, lats[lux] = step_test(2.0, lux)
    print("[4] latency after x2 step (median)   : " +
          ", ".join(f"{k} lux: {v:.0f} us" for k, v in lats.items()) + "  (expect decreasing)")
    ok &= lats[10] > lats[100] > lats[1000]

    rates = {}
    for lux in (10, 1000):
        m, d, cam = make(lux_per_unit=lux, leak_rate_hz=0, hot_pixel_fraction=0)
        run(m, d, cam, 0.05)
        W, H = cam.geometry(d)
        rates[lux] = len(run(m, d, cam, 0.3)) / (W * H) / 0.3
    print("[5] shot-noise background activity   : " +
          ", ".join(f"{k} lux: {v:.3f} Hz/px" for k, v in rates.items()) + "  (expect decreasing)")
    ok &= rates[10] > rates[1000]

    m, d, cam = make(dt=1e-3, lux_per_unit=500, flicker_hz=100, flicker_depth=0.8, **QUIET)
    run(m, d, cam, 0.05)
    ev = run(m, d, cam, 0.2)
    if len(ev) > 10:
        t = ev["t"][ev["p"] == 1] * 1e-6          # ON bursts once per flicker period
        hist, _ = np.histogram(t, bins=np.arange(t.min(), t.max(), 1e-3))
        spec = np.abs(np.fft.rfft(hist - hist.mean()))
        f = np.fft.rfftfreq(len(hist), 1e-3)
        fpk = f[np.argmax(spec[1:]) + 1]
    else:
        fpk = 0
    print(f"[6] 100 Hz flicker                    : {len(ev)} events, ON-event rate peaks at "
          f"{fpk:.1f} Hz (expect 100; ON+OFF together give 200 Hz)")
    ok &= abs(fpk - 100) < 5

    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
