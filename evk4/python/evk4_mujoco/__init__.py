"""
evk4_mujoco — Python side of the `prophesee.sensor.evk4` MuJoCo event-camera plugin.

    import evk4_mujoco as evk
    evk.load_plugin()                                   # registers the native plugin
    m = evk.load_model("scenes/scene_ur5e_evk4.xml")    # optional config overrides as kwargs
    d = mujoco.MjData(m)
    cam = evk.EVK4(m, "evk4")
    mujoco.mj_step(m, d)
    ev = cam.events(d)          # numpy structured array, Metavision EventCD layout (x, y, p, t[us])
    img = cam.intensity(d)      # photoreceptor log-intensity image (H x W), for debugging
"""
from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass

import numpy as np
import mujoco

__all__ = ["load_plugin", "load_model", "EVK4", "Sensor", "EventFrame", "EVENT_DTYPE", "LABELED_DTYPE", "plugin_path",
           "attach_evk4", "add_evk4", "add_props", "start_props", "look_at_quat", "euler_to_quat",
           "PairedCamera", "Frame", "flow_to_rgb", "camera_model"]

# Same memory layout as metavision_sdk_base.EventCD
EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])
# EventCD + per-event ground truth produced by the simulator
LABELED_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8"),
                          ("geom", "<i4"), ("depth", "<f4"), ("flow_x", "<f4"), ("flow_y", "<f4")])
_HEADER = 64
# header slots (must match the C++ plugin)
H_NEV, H_CAP, H_W, H_H, H_DROPTOT, H_TLAST, H_FX, H_NSUB = 0, 1, 2, 3, 4, 5, 6, 7
H_FY, H_CX, H_CY, H_STRIDE, H_VERSION = 13, 14, 15, 16, 17
H_OFF_EVENTS, H_OFF_PEND, H_OFF_PIX, H_OFF_POSE, H_OFF_PROBE, H_OFF_IMG = 30, 31, 32, 33, 34, 35
H_BACKEND_EXTERNAL = 37
_loaded = False


def plugin_path() -> str:
    """Locate the plugin library built for THIS operating system (.dll / .dylib / .so)."""
    import sys as _sys
    ext = "dll" if _sys.platform.startswith("win") else ("dylib" if _sys.platform == "darwin" else "so")
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.get("EVK4_PLUGIN")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"EVK4_PLUGIN={env} does not exist")
        return os.path.abspath(env)
    top = os.path.join(here, "..", "..")
    for root in (here, os.path.join(top, "build"), os.path.join(top, "build", "Release"), top):
        c = glob.glob(os.path.join(root, f"*evk4_event_camera*.{ext}"))
        if c:
            return os.path.abspath(c[0])
    raise FileNotFoundError(f"EVK4 plugin (*.{ext}) not found - build it first (build.sh / CMake) "
                            "or set EVK4_PLUGIN")


def load_plugin(path: str | None = None) -> None:
    """Register the native plugin with MuJoCo (idempotent)."""
    global _loaded
    if _loaded:
        return
    mujoco.mj_loadPluginLibrary(path or plugin_path())
    _loaded = True


def load_model(xml_path: str, **config) -> mujoco.MjModel:
    """Load an MJCF file; keyword args override/add <config key=... value=...> of the EVK4 plugin.

    e.g. load_model("scene.xml", binning=2, shot_noise=0, bias_diff_on=20)
    """
    load_plugin()
    if not config:
        return mujoco.MjModel.from_xml_path(xml_path)
    xml = open(xml_path).read()
    for k, v in config.items():
        if k == "bias_file":
            v = os.path.abspath(v)
        v = " ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
        pat = re.compile(r'(<config\s+key="%s"\s+value=")[^"]*(")' % re.escape(k))
        if pat.search(xml):
            xml = pat.sub(lambda mm: mm.group(1) + v + mm.group(2), xml)
        else:
            xml = re.sub(r'(<plugin[^>]*plugin="prophesee\.sensor\.evk4"[^>]*objtype[^>]*>)',
                         lambda mm: mm.group(1) + '\n      <config key="%s" value="%s"/>' % (k, v), xml)
    tmp = os.path.join(os.path.dirname(os.path.abspath(xml_path)), ".evk4_override_%d.xml" % os.getpid())
    with open(tmp, "w") as f:
        f.write(xml)
    try:
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)


def plugin_config(model: mujoco.MjModel, instance: int) -> dict:
    """Parse the <config> key/values of a plugin instance from the compiled model."""
    start = int(model.plugin_attradr[instance])
    end = int(model.plugin_attradr[instance + 1]) if instance + 1 < model.nplugin else len(model.plugin_attr)
    raw = bytes(model.plugin_attr[start:end]).split(b"\0")
    # attribute values are stored in declaration order of the plugin's attribute list
    return [r.decode() for r in raw]


class EVK4:
    """Accessor for one EVK4 sensor instance inside an MjModel/MjData."""

    def __init__(self, model: mujoco.MjModel, sensor_name: str = "evk4"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sid < 0:
            raise KeyError(f"sensor '{sensor_name}' not found")
        self.model = model
        self.sensor_id = sid
        self.instance = int(model.sensor_plugin[sid])
        self.adr = int(model.plugin_stateadr[self.instance])
        self.num = int(model.plugin_statenum[self.instance])
        self.sadr = int(model.sensor_adr[sid])
        self.camera_id = int(model.sensor_objid[sid])

    def _state(self, data):
        return data.plugin_state[self.adr:self.adr + self.num]

    def geometry(self, data):
        s = self._state(data)
        return int(s[H_W]), int(s[H_H])

    def intrinsics(self, data):
        """fx, fy, cx, cy of the (possibly binned) simulated array, OpenCV convention."""
        s = self._state(data)
        return float(s[H_FX]), float(s[H_FY]), float(s[H_CX]), float(s[H_CY])

    def _raw(self, data):
        s = self._state(data)
        n, off = int(s[H_NEV]), int(s[H_OFF_EVENTS])
        return s[off:off + 8 * n].reshape(n, 8)

    def events(self, data) -> np.ndarray:
        """Events of the last mj_step (time-ordered), Metavision EventCD dtype."""
        raw = self._raw(data)
        ev = np.empty(len(raw), EVENT_DTYPE)
        for j, k in enumerate(("x", "y", "p", "t")):
            ev[k] = raw[:, j]
        return ev

    def labeled_events(self, data) -> np.ndarray:
        """Events of the last mj_step with ground truth: geom id, depth [m], optical flow [px/s]."""
        raw = self._raw(data)
        ev = np.empty(len(raw), LABELED_DTYPE)
        for j, k in enumerate(LABELED_DTYPE.names):
            ev[k] = raw[:, j]
        return ev

    def intensity(self, data) -> np.ndarray:
        """Photoreceptor output ln(I) incl. noise, H x W (debug / visualisation)."""
        s = self._state(data)
        off, W, H = int(s[H_OFF_IMG]), int(s[H_W]), int(s[H_H])
        if off < 0:
            raise RuntimeError("export_intensity=0")
        return s[off:off + W * H].reshape(H, W)

    def stats(self, data) -> dict:
        sd = data.sensordata[self.sadr:self.sadr + 8]
        keys = ["n_events", "n_on", "n_off", "n_dropped", "n_substeps", "rate_mevps",
                "mean_log_intensity", "n_dropped_total"]
        return dict(zip(keys, map(float, sd)))


@dataclass
class EventFrame:
    """Accumulates events into an image, Metavision-viewer style.

    palette 'dark'  : background (30,37,52), ON (216,223,236), OFF (64,126,201)  [RGB]
    palette 'gray'  : ON white, OFF black on mid-gray
    """
    width: int
    height: int
    palette: str = "dark"

    def __post_init__(self):
        if self.palette == "dark":
            self.bg = np.array([30, 37, 52], np.uint8)
            self.on = np.array([216, 223, 236], np.uint8)
            self.off = np.array([64, 126, 201], np.uint8)
        else:
            self.bg = np.array([128, 128, 128], np.uint8)
            self.on = np.array([255, 255, 255], np.uint8)
            self.off = np.array([0, 0, 0], np.uint8)
        self.reset()

    def reset(self):
        self.img = np.empty((self.height, self.width, 3), np.uint8)
        self.img[:] = self.bg

    def add(self, ev: np.ndarray):
        if len(ev) == 0:
            return
        on = ev["p"] > 0
        self.img[ev["y"][on], ev["x"][on]] = self.on
        self.img[ev["y"][~on], ev["x"][~on]] = self.off

    def rgb(self) -> np.ndarray:
        return self.img


class Sensor:
    """Backend-independent handle:  s = Sensor(m, "evk4", backend="cpu"|"warp");
    after every mj_step call s.update(d); then s.events(), s.labeled_events(), s.intensity(), ...
    With backend="warp" load the model with load_model(..., backend="warp") so the C++ plugin
    does not also render."""

    def __init__(self, model, sensor_name="evk4", backend="cpu", device=None):
        self.backend = backend
        if backend == "cpu":
            self._s = EVK4(model, sensor_name)
        else:
            from .gpu import EVK4Warp
            self._s = EVK4Warp(model, sensor_name, device=device)
        self._d = None

    def update(self, d):
        self._d = d
        if self.backend != "cpu":
            self._s.update(d)

    def events(self):
        return self._s.events(self._d) if self.backend == "cpu" else self._s.events()

    def labeled_events(self):
        return self._s.labeled_events(self._d) if self.backend == "cpu" else self._s.labeled_events()

    def intensity(self):
        return self._s.intensity(self._d) if self.backend == "cpu" else self._s.intensity()

    def geometry(self):
        return self._s.geometry(self._d) if self.backend == "cpu" else self._s.geometry()

    def time(self):
        """Sensor time [s] of the latest sample - the instant the current MjData poses belong to
        for this backend (timestamp paired frames with it)."""
        if self.backend == "cpu":
            return float(self._s._state(self._d)[H_TLAST])
        return float(self._s.t_prev)

    def intrinsics(self):
        """fx, fy, cx, cy [px] of the simulated array (OpenCV convention)."""
        return self._s.intrinsics(self._d) if self.backend == "cpu" else self._s.intrinsics()

    def stats(self):
        return self._s.stats(self._d) if self.backend == "cpu" else self._s.stats()


# Phase-1 helpers (imported last: they use load_plugin from this module)
from .config import camera_model                                                # noqa: E402
from .mount import add_evk4, add_props, attach_evk4, euler_to_quat, look_at_quat, start_props  # noqa: E402
from .frames import Frame, PairedCamera, flow_to_rgb                            # noqa: E402
