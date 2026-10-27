"""
Sensor configuration for the Habitat EVK4.

The pixel / readout configuration is *exactly* the MuJoCo plugin's: the same keys, defaults,
`.bias` file handling and Metavision bias -> physics mapping (`evk4_mujoco.config.read_config`
is called on the given values, so derived quantities such as `eff_con` or `eff_cutoff` can
never drift between the two simulators).

Habitat-only keys (rendering of the photon input) are kept separately in `HABITAT_DEFAULTS`.
"""
from __future__ import annotations

import math

import numpy as np

from evk4_mujoco.config import ATTRS, DEFAULTS, _STR, read_config

# Keys that only exist on the Habitat side (how the irradiance image is produced).
HABITAT_DEFAULTS = dict(
    hfov_deg=70.0,       # horizontal field of view when neither fx nor focal_length_mm is given
    near=0.05,           # render clipping planes [m]
    far=50.0,
    scene_lux=300.0,     # auto exposure: mean illuminance [lux] of the scene, assuming 18 % grey
                         # reflectance; used only when lux_per_unit is not given explicitly
    flow_labels=1,       # per-event optical-flow ground truth
)


class _PluginAttrs:
    """Minimal stand-in for an MjModel with one EVK4 plugin instance, so the original
    `read_config` (defaults, bias file, bias mapping, derived values) is reused verbatim."""

    def __init__(self, values: dict):
        blob = b"\0".join(str(values.get(k, "")).encode() for k in ATTRS) + b"\0"
        self.plugin_attr = np.frombuffer(blob, dtype=np.uint8)
        self.plugin_attradr = np.array([0])
        self.nplugin = 1


def _fmt(k, v):
    if k in _STR:
        if isinstance(v, (list, tuple, np.ndarray)):
            return " ".join(str(x) for x in v)
        return str(v)
    return repr(float(v))


def make_config(**overrides) -> tuple[dict, dict]:
    """(pixel_config, habitat_config) from keyword overrides.

    Pixel keys are any of `evk4_mujoco.config.ATTRS` (resolution, contrast_on, bias_diff_on, ...),
    Habitat keys are those of `HABITAT_DEFAULTS`.  Unknown keys raise KeyError."""
    hab = dict(HABITAT_DEFAULTS)
    pix = {}
    for k, v in overrides.items():
        if k in HABITAT_DEFAULTS:
            hab[k] = float(v)
        elif k in ATTRS:
            pix[k] = _fmt(k, v)
        else:
            raise KeyError(f"unknown EVK4 config key '{k}'")
    hab["lux_per_unit_given"] = "lux_per_unit" in pix
    c = read_config(_PluginAttrs(pix), 0)
    return c, hab


def intrinsics(c: dict, hab: dict):
    """W, H, fx, fy, cx, cy, dist of the (binned) pixel array, same rules as the plugin;
    without fx / focal_length_mm the focal length follows from hfov_deg."""
    b = int(c["binning"])
    res = [int(float(x)) for x in str(c["resolution"]).split()]
    W, H = res[0] // b, res[1] // b
    pitch = c["pixel_pitch_um"] * 1e-6 * b
    if c["fx"] > 0:
        fx = c["fx"] / b
        fy = (c["fy"] if c["fy"] > 0 else c["fx"]) / b
    elif c["focal_length_mm"] > 0:
        fx = fy = c["focal_length_mm"] * 1e-3 / pitch
    else:
        fx = fy = 0.5 * W / math.tan(0.5 * math.radians(hab["hfov_deg"]))
    cx = (c["cx"] + 0.5) / b - 0.5 if c["cx"] >= 0 else 0.5 * (W - 1)
    cy = (c["cy"] + 0.5) / b - 0.5 if c["cy"] >= 0 else 0.5 * (H - 1)
    dist = np.array([float(x) for x in str(c["distortion"]).split()] + [0] * 5)[:5]
    return W, H, fx, fy, cx, cy, dist


__all__ = ["make_config", "intrinsics", "HABITAT_DEFAULTS", "DEFAULTS"]
