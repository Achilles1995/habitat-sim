"""
Checks of the Habitat EVK4 photon chain that need no Habitat renderer: the Habitat sensors are
replaced by an analytic scene (a textured wall 2 m in front of the camera with a bright square
object moving across it), so the radiometry, the pixel physics hand-over, the sub-stepping and
the flow / instance / range labels can be compared with closed-form expectations.

    python tests/test_habitat_chain.py [--device cuda:0]

Runs with or without habitat_sim installed (stubs are used if it is missing).
"""
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

try:
    import habitat_sim  # noqa: F401
    import magnum  # noqa: F401
except ImportError:                     # minimal stand-ins, only for module import
    hs = types.ModuleType("habitat_sim")
    hs.SensorType = types.SimpleNamespace(COLOR=0, DEPTH=1, SEMANTIC=2)
    hs.physics = types.SimpleNamespace(JointType=types.SimpleNamespace(Spherical=3))
    sys.modules["habitat_sim"] = hs
    mn = types.ModuleType("magnum")
    mn.Color4 = lambda *a: a
    mn.Matrix4 = object
    sys.modules["magnum"] = mn

import evk4_habitat.camera as camera_mod  # noqa: E402
from evk4_habitat.camera import EVK4Habitat  # noqa: E402
from evk4_habitat.mounts import Mount  # noqa: E402

DEV = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
OBJ_ID = 63000


class World:
    """Analytic scene + SceneMotion stand-in.  Camera at the origin looking along -z (GL),
    wall at z = -2 with a sinusoidal albedo, square object (0.4 m) at z = -1.5 moving in x."""

    def __init__(self):
        self.obj_x = -0.6
        self.gain = 1.0          # global illumination multiplier

    # SceneMotion API
    def capture(self):
        return [self.obj_x]

    def apply(self, s):
        self.obj_x = s[0]

    def apply_interp(self, a, b, f):
        self.obj_x = a[0] + f * (b[0] - a[0])

    def part_transforms(self):
        T = np.eye(4)
        T[0, 3] = self.obj_x
        return {OBJ_ID: T}, {}

    def key_points(self):
        return np.array([[self.obj_x + dx, dy, -1.5] for dx in (-0.2, 0.2) for dy in (-0.2, 0.2)])

    def instances(self):
        return {0: dict(kind="stage"), OBJ_ID: dict(kind="object")}

    # renderer: per render pixel, GL camera at the origin, hfov from the sensor
    def render(self, W, H, f, cx, cy):
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x, y = (u - cx) / f, (v - cy) / f                    # OpenCV normalised coords
        wall_X = x * 2.0
        alb = 0.4 + 0.3 * np.sin(2 * math.pi * wall_X / 0.5)
        depth = np.full((H, W), 2.0, np.float32)
        sem = np.zeros((H, W), np.uint32)
        X, Y = x * 1.5, y * 1.5
        inside = (np.abs(X - self.obj_x) < 0.2) & (np.abs(Y) < 0.2)
        alb = np.where(inside, 0.9, alb)
        depth = np.where(inside, 1.5, depth).astype(np.float32)
        sem = np.where(inside, OBJ_ID, sem).astype(np.uint32)
        rad = np.zeros((H, W, 4), np.float32)
        rad[..., :3] = (alb * self.gain)[..., None]
        rad[..., 3] = 1
        return dict(rad=rad, depth=depth, sem=sem)


class FakeSensor:
    def __init__(self, cam, world, kind):
        self.cam, self.world, self.kind, self.uuid = cam, world, kind, kind

    def draw_observation(self):
        pass

    def get_observation(self):
        c = self.cam
        return self.world.render(c.Wr, c.Hr, c.fr, c.cxr, c.cyr)[self.kind]


class FakeNode:
    def absolute_transformation(self):
        return np.eye(4)


def make_cam(world, **cfg):
    def _make_sensors(self):
        self.node = FakeNode()
        self.sensors = {k: FakeSensor(self, world, k) for k in ("rad", "depth", "sem")}

    camera_mod.EVK4Habitat._make_sensors = _make_sensors
    base = dict(resolution="160 120", hfov_deg=60, lux_per_unit=1000.0)
    base.update(cfg)
    return EVK4Habitat(sim=object(), mount=Mount(np.eye(4)), motion=world, name="t", config=base, device=DEV)


def check(name, ok, info):
    print(("[PASS] " if ok else "[FAIL] ") + f"{name}: {info}", flush=True)
    return ok


def main():
    results = []
    # 1 radiometry: photocurrent = k_rate * lux * albedo / pi * cos^4
    w = World()
    cam = make_cam(w, shot_noise=0, psf_sigma_px=0)
    cam.update(0.0)
    rate = cam.rate.numpy().reshape(cam.H, cam.W)
    alb = w.render(cam.W, cam.H, cam.fx, cam.cx, cam.cy)["rad"][..., 0]
    cos4 = cam.cos4.numpy().reshape(cam.H, cam.W)
    want = cam.k_rate * 1000.0 * alb / math.pi * cos4
    err = np.max(np.abs(rate / want - 1))
    results.append(check("1 radiometry", err < 1e-4, f"max |rate/expected - 1| = {err:.2e}, "
                                                      f"k_rate {cam.k_rate:.3g} e-/s per cd/m^2"))

    # 2 light step x2 -> ON events per pixel ~ E[floor((D0 + ln 2) / theta)] (MuJoCo test 2)
    w = World()
    w.obj_x = 10.0                       # object out of view
    cam = make_cam(w, shot_noise=0, refractory_us=0, psf_sigma_px=0, contrast_sigma=0, leak_rate_hz=0,
                   hot_pixel_fraction=0)
    cam.update(0.0)
    on = off = 0
    for k in range(1, 40):
        if k == 2:
            w.gain = 2.0
        cam.update(k * 1e-3)
        e = cam.events()
        on += int((e["p"] == 1).sum())
        off += int((e["p"] == 0).sum())
    th, thoff = cam.cfg["eff_con"], cam.cfg["eff_coff"]
    D0 = (th + thoff) * (0.05 + 0.9 * np.linspace(0, 1, 20001)) - thoff     # initial x - ref
    expect = float(np.mean(np.floor((D0 + math.log(2)) / th)))
    per = on / cam.N
    results.append(check("2 light step x2", abs(per - expect) < 0.08 and off == 0,
                         f"ON ev/px {per:.3f} (expect {expect:.3f}), OFF {off}"))

    # 3 moving object: events on the object carry its id, range and flow = fx * v / z
    w = World()
    cam = make_cam(w, shot_noise=0)
    v = 1.0                                    # m/s
    cam.update(0.0)
    evs, nsubs = [], []
    for k in range(1, 31):
        w.obj_x = -0.6 + v * k / 30.0 * 0.5     # 0.5 m in 0.5 s ... (1 m/s over 30 steps of 1/60 s)
        cam.update(k / 60.0)
        evs.append(cam.labeled_events())
        nsubs.append(cam.nsub)
    ev = np.concatenate(evs)
    on_obj = ev[ev["geom"] == OBJ_ID]
    fx_exp = cam.fx * v / 1.5
    med_fx = float(np.median(on_obj["flow_x"])) if len(on_obj) else 0.0
    med_fy = float(np.median(np.abs(on_obj["flow_y"]))) if len(on_obj) else 0.0
    rng = on_obj["depth"]
    xs = (on_obj["x"] - cam.cx) / cam.fx
    ys = (on_obj["y"] - cam.cy) / cam.fy
    zr = rng / np.sqrt(1 + xs ** 2 + ys ** 2)
    ok = len(on_obj) > 100 and abs(med_fx / fx_exp - 1) < 0.05 and med_fy < 0.05 * fx_exp \
        and np.max(np.abs(zr - 1.5)) < 1e-3
    results.append(check("3 moving object labels", ok,
                         f"{len(ev)} events, {len(on_obj)} on the object; median flow_x {med_fx:.1f} px/s "
                         f"(expect {fx_exp:.1f}), |flow_y| {med_fy:.2f}; z from range in [{zr.min():.4f}, "
                         f"{zr.max():.4f}] (expect 1.5); sub-steps per update {min(nsubs)}..{max(nsubs)} "
                         f"(motion {fx_exp / 60:.1f} px per update)"))
    want_sub = int(math.ceil(fx_exp / 60.0 / cam.cfg["max_px_motion"]))
    results.append(check("4 adaptive sub-steps", max(nsubs) == min(int(cam.cfg["max_substeps"]), want_sub),
                         f"max {max(nsubs)}, expect {min(int(cam.cfg['max_substeps']), want_sub)}"))

    # 5 static scene, shot noise on: only background activity, rate falls with light (MuJoCo test 5)
    rates = []
    for lux in (10.0, 1000.0):
        w = World()
        w.obj_x = 10.0
        cam = make_cam(w, lux_per_unit=lux / 0.55, resolution="80 60")
        cam.update(0.0)
        n = 0
        for k in range(1, 201):
            cam.update(k * 1e-3)
            n += len(cam.events())
        rates.append(n / cam.N / 0.2)
    results.append(check("5 shot-noise background", rates[0] > 3 * rates[1] > 0,
                         f"{rates[0]:.3f} Hz/px at ~10 lux, {rates[1]:.3f} Hz/px at ~1000 lux"))

    # 6 distortion + area sampling path (resampling kernel) keeps the radiometry
    w = World()
    w.obj_x = 10.0
    cam = make_cam(w, shot_noise=0, psf_sigma_px=0, spp=4, distortion="-0.1 0.01 0 0 0")
    cam.update(0.0)
    rate = cam.rate.numpy().reshape(cam.H, cam.W)
    cos4 = cam.cos4.numpy().reshape(cam.H, cam.W)
    ctr = cam.ctr_np.reshape(cam.H, cam.W, 2)
    alb = 0.4 + 0.3 * np.sin(2 * math.pi * ctr[..., 0] * 2.0 / 0.5)
    want = cam.k_rate * 1000.0 * alb / math.pi * cos4
    err = np.median(np.abs(rate / want - 1))
    results.append(check("6 resampling (distortion, spp 4)", (not cam.direct) and err < 0.02,
                         f"render {cam.Wr}x{cam.Hr} for {cam.W}x{cam.H}, median |rate/expected - 1| = {err:.4f}"))

    print("ALL PASSED" if all(results) else "SOME FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
