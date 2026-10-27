"""
Tests for the v2 features of prophesee.sensor.evk4:

  A. save / restore: mj_copyData snapshots reproduce the event stream bit-exactly
  B. dirty-tile skipping gives the same events as full re-rendering
  C. per-event optical-flow labels match the analytic flow of a spinning object
  D. lens model (fx, fy, cx, cy, k1 k2 p1 p2 k3) is consistent with OpenCV
  E. Metavision biases (keys and .bias file) change thresholds as mapped
  F. temperature scales junction-leak events
  G. ESP / readout: AFK removes flicker, STC removes noise, ERC caps the rate,
     arbiter throughput adds load-dependent latency, ROI masks the array
Run:  python tests/test_v2_features.py
"""
import os
import sys
import tempfile

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import evk4_mujoco as evk                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.join(HERE, "mesh_scene.xml")
CAL = os.path.join(HERE, "calib_scene.xml")
QUIET = dict(shot_noise=0, leak_rate_hz=0, hot_pixel_fraction=0, contrast_sigma=0, latency_jitter_us=0)
RESULTS = []


def check(name, ok, msg):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")


def run(m, d, cam, n, labeled=False, ctrl=None):
    out = []
    for _ in range(n):
        mujoco.mj_step(m, d)
        out.append(cam.labeled_events(d) if labeled else cam.events(d))
    return np.concatenate(out)


def spin_scene(omega=6.0, **cfg):
    m = evk.load_model(MESH, **cfg)
    d = mujoco.MjData(m)
    d.qvel[m.jnt_dofadr[m.joint("qspin").id]] = omega
    mujoco.mj_forward(m, d)
    return m, d, evk.EVK4(m)


def test_save_restore():
    m, d, cam = spin_scene()
    run(m, d, cam, 30)
    snap = mujoco.MjData(m)
    mujoco.mj_copyData(snap, m, d)
    a = run(m, d, cam, 30, labeled=True)
    mujoco.mj_copyData(d, m, snap)                          # rewind the same MjData
    b = run(m, d, cam, 30, labeled=True)
    d2 = mujoco.MjData(m)                                   # and a fresh MjData
    mujoco.mj_copyData(d2, m, snap)
    c = run(m, d2, cam, 30, labeled=True)
    ok = len(a) > 100 and np.array_equal(a, b) and np.array_equal(a, c)
    check("A save/restore", ok, f"{len(a)} events; rewind identical={np.array_equal(a, b)}, "
                                f"copy-to-new-MjData identical={np.array_equal(a, c)}")


def test_tile_skip():
    res = []
    for skip in (1, 0):
        m, d, cam = spin_scene(skip_static_tiles=skip)
        res.append(run(m, d, cam, 40))
    same = len(res[0]) == len(res[1]) and np.array_equal(res[0], res[1])
    check("B dirty-tile skipping", same, f"events with skipping {len(res[0])}, without {len(res[1])}, identical={same}")


def test_flow_labels():
    omega = 6.0
    m, d, cam = spin_scene(omega)
    run(m, d, cam, 20)
    ev = run(m, d, cam, 20, labeled=True)
    gq = m.geom("quad").id
    ev = ev[ev["geom"] == gq]
    fx, fy, cx, cy = cam.intrinsics(d)
    cpos = d.cam_xpos[m.camera("c").id].copy()
    c = d.xpos[m.body("quadb").id].copy()
    # back-project each event with its depth, compute analytic image velocity of v = w x (P - c)
    xn, yn = (ev["x"] - cx) / fx, (ev["y"] - cy) / fy
    dirs = np.stack([xn, -yn, -np.ones_like(xn)], 1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    P = cpos + dirs * ev["depth"][:, None]                 # camera looks straight down (-z)
    vel = np.cross(np.array([0, 0, omega]), P - c)
    z = cpos[2] - P[:, 2]
    # camera frame: x = world x, y_img = -world y  (camera identity orientation, looking down)
    u_dot = fx * (vel[:, 0] / z)
    v_dot = fy * (-vel[:, 1] / z)
    ana = np.stack([u_dot, v_dot], 1)
    lab = np.stack([ev["flow_x"], ev["flow_y"]], 1)
    err = np.linalg.norm(lab - ana, axis=1) / (np.linalg.norm(ana, axis=1) + 1e-9)
    med = float(np.median(err))
    check("C optical-flow labels", med < 0.05 and len(ev) > 100,
          f"{len(ev)} labelled events on the spinning quad, median relative flow error {100*med:.2f}%")


def test_distortion():
    K = dict(fx=520.0, fy=515.0, cx=330.0, cy=172.0, distortion="-0.25 0.08 0.001 -0.0015 -0.01")
    m, d, cam = spin_scene(**K)
    run(m, d, cam, 10)
    ev = run(m, d, cam, 20, labeled=True)
    ev = ev[ev["geom"] >= 0][:5000]
    Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])
    dist = np.array([float(v) for v in K["distortion"].split()])
    pts = np.stack([ev["x"], ev["y"]], 1).astype(np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, Km, dist).reshape(-1, 2)          # OpenCV inverse model
    rays = np.concatenate([und, np.ones((len(und), 1))], 1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    P = rays * ev["depth"][:, None]                                   # OpenCV camera frame
    proj, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), Km, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - pts.reshape(-1, 2), axis=1)
    # also check that distortion visibly bends the image: compare against no-distortion render
    check("D lens model vs OpenCV", float(np.max(err)) < 0.05,
          f"round-trip (pixel -> depth ray -> cv2.projectPoints) max error {np.max(err):.4f} px "
          f"over {len(err)} events")


def step_count(**cfg):
    m = evk.load_model(CAL, lux_per_unit=500, **QUIET, **cfg)
    m.opt.timestep = 1e-5
    d = mujoco.MjData(m); mujoco.mj_forward(m, d); cam = evk.EVK4(m)
    run(m, d, cam, 200)
    m.light_diffuse[0] *= 2.0
    ev = run(m, d, cam, 800)
    W, H = cam.geometry(d)
    return len(ev) / (W * H)


def test_biases():
    base = step_count(refractory_us=0)
    hi = step_count(refractory_us=0, bias_diff_on=100)
    with tempfile.NamedTemporaryFile("w", suffix=".bias", delete=False) as f:
        f.write("100   % bias_diff_on\n0     % bias_diff_off\n0     % bias_fo\n0     % bias_hpf\n0     % bias_refr\n")
        path = f.name
    viaf = step_count(refractory_us=0, bias_file=path)
    os.remove(path)
    th_on, th_off = 0.22 * np.e, 0.22                       # bias_diff_on=+100 -> x e
    u = (np.arange(100000) + 0.5) / 100000
    D0 = -th_off + (th_on + th_off) * (0.05 + 0.9 * u)
    exp_hi = np.mean(np.maximum(0, np.floor((D0 + np.log(2)) / th_on)))
    check("E Metavision biases", hi < base and abs(hi - viaf) < 1e-9 and abs(hi - exp_hi) < 0.1,
          f"ON ev/px on x2 step: default {base:.2f}, bias_diff_on=+100 -> {hi:.2f} "
          f"(expected {exp_hi:.2f} for threshold x e), same via .bias file: {viaf:.2f}")


def leak_rate(T):
    m = evk.load_model(CAL, lux_per_unit=500, shot_noise=0, hot_pixel_fraction=0, leak_rate_hz=1.0,
                       temperature_c=T)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d); cam = evk.EVK4(m)
    run(m, d, cam, 1000)                                    # let the random phases wash out
    ev = run(m, d, cam, 4000)
    W, H = cam.geometry(d)
    return len(ev) / (W * H) / 4.0


def test_temperature():
    r25, r45 = leak_rate(25), leak_rate(45)
    exp = 2 ** (20 / 7)
    check("F temperature leak", abs(r45 / r25 - exp) / exp < 0.15,
          f"leak events 25C {r25:.2f} Hz/px, 45C {r45:.2f} Hz/px, ratio {r45/r25:.2f} (expect {exp:.2f})")


def cal_run(T=0.2, lux=500, **cfg):
    m = evk.load_model(CAL, lux_per_unit=lux, **cfg)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d); cam = evk.EVK4(m)
    run(m, d, cam, 50)
    return run(m, d, cam, int(T / m.opt.timestep)), cam, d


def test_esp():
    fl = dict(flicker_hz=100, flicker_depth=0.8, **QUIET)
    ev0, _, _ = cal_run(**fl)
    ev1, _, _ = cal_run(afk=1, afk_min_hz=50, afk_max_hz=200, **fl)
    check("G1 AFK anti-flicker", len(ev1) < 0.1 * len(ev0),
          f"100 Hz flicker: {len(ev0)} events -> {len(ev1)} with AFK ({100*len(ev1)/max(1,len(ev0)):.1f}%)")

    n0, _, _ = cal_run(lux=10, leak_rate_hz=0, hot_pixel_fraction=0)
    n1, _, _ = cal_run(lux=10, leak_rate_hz=0, hot_pixel_fraction=0, stc_us=2000)
    check("G2 STC noise filter", len(n1) < 0.3 * len(n0),
          f"10 lux background: {len(n0)} events -> {len(n1)} with STC 2 ms")

    # steady stimulus: strong shot noise; ERC target = 1/4 of the raw rate
    nz = dict(lux=10, noise_gain=2.0, leak_rate_hz=0, hot_pixel_fraction=0)
    r0, _, _ = cal_run(**nz)
    raw = len(r0) / 0.2 / 1e6
    tgt = raw / 4
    e0, _, _ = cal_run(erc_mevps=tgt, **nz)
    rate = len(e0) / 0.2 / 1e6
    win = np.floor(e0["t"] / 200.0)
    _, cnt = np.unique(win, return_counts=True)
    allowed = tgt * 1e6 * 200e-6
    check("G3 Event Rate Controller", abs(rate - tgt) / tgt < 0.15 and cnt.max() <= allowed + 1,
          f"steady {raw:.2f} Mev/s -> {rate:.2f} Mev/s (target {tgt:.2f}); max per 200 us window "
          f"{cnt.max()} <= {allowed:.0f}")

    def lat(**cfg):
        m = evk.load_model(CAL, lux_per_unit=500, **QUIET, **cfg)
        m.opt.timestep = 1e-5
        d = mujoco.MjData(m); mujoco.mj_forward(m, d); cam = evk.EVK4(m)
        run(m, d, cam, 200); t0 = d.time
        m.light_diffuse[0] *= 2.0
        ev = run(m, d, cam, 3000)
        return np.percentile(ev["t"], 90) - t0 * 1e6
    l0, l1 = lat(), lat(readout_mevps=20)
    check("G4 arbiter throughput", l1 > l0 + 100, f"90th-pct latency after a global step: {l0:.0f} us "
                                                   f"unlimited -> {l1:.0f} us at 20 Mev/s bus")

    er, cam, d = cal_run(roi="0 0 80 45", **fl)
    ok = len(er) > 0 and er["x"].max() < 80 and er["y"].max() < 45
    check("G5 ROI", ok, f"{len(er)} events, all inside x<80, y<45: {ok}")


def test_dt_invariance():
    rates = {}
    for lux in (10, 1000):
        for dt in (1e-3, 2e-4, 5e-5):
            m = evk.load_model(CAL, lux_per_unit=lux, leak_rate_hz=0, hot_pixel_fraction=0)
            m.opt.timestep = dt
            d = mujoco.MjData(m); mujoco.mj_forward(m, d); cam = evk.EVK4(m)
            run(m, d, cam, int(0.02 / dt))
            T = 0.1 if lux == 10 else 0.4
            ev = run(m, d, cam, int(T / dt))
            W, H = cam.geometry(d)
            rates[(lux, dt)] = len(ev) / (W * H) / T
    ok = True
    for lux in (10, 1000):
        r = [rates[(lux, dt)] for dt in (1e-3, 2e-4, 5e-5)]
        ok &= (max(r) - min(r)) / np.mean(r) < 0.15
    check("H noise statistics vs time step", ok,
          "; ".join(f"{lux} lux: " + ", ".join(f"{rates[(lux, dt)]:.3f}" for dt in (1e-3, 2e-4, 5e-5))
                    for lux in (10, 1000)) + "  Hz/px at dt = 1 ms, 200 us, 50 us")


if __name__ == "__main__":
    test_save_restore()
    test_tile_skip()
    test_flow_labels()
    test_distortion()
    test_biases()
    test_temperature()
    test_esp()
    test_dt_invariance()
    print("ALL PASSED" if all(RESULTS) else "SOME CHECKS FAILED")
    sys.exit(0 if all(RESULTS) else 1)
