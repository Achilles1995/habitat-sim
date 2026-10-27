"""
The ESD camera rig on the robot flange: two DAVIS346c event cameras (left / right, tilted ~5 deg
towards the middle) and an Intel D435 RGB-D camera in the middle, all looking down.

Calibration = the authors' own values from ESD_labeling_tool/labeling_auto_clean_2.m:
    k_el, d_el, m_el   left  DAVIS346   (K, OpenCV distortion k1 k2 p1 p2 k3, T_tcp_cam)
    k_er, d_er, m_er   right DAVIS346
    k_d,  d_d,  m_d    D435 colour stream (640 x 480)
The 4x4 m_* matrices map camera coordinates into the robot TCP frame (their code computes
p_out = inv(out_pose) * in_pose * p_in), i.e. they are T_tcp_cam in OpenCV axes - exactly the
mount-frame convention of evk4_mujoco.add_evk4, so they are used verbatim.
"""
from __future__ import annotations

import numpy as np
import mujoco

from ..mount import add_evk4

ESD_CALIB = {
    "left": dict(
        K=np.array([[349.60817472, 0.0, 179.96134154], [0.0, 349.94740878, 125.95477917], [0, 0, 1.0]]),
        dist=np.array([-3.40016570e-01, -4.16759407e-03, 3.74480929e-05, -4.14482894e-04, 5.43477741e-01]),
        T=np.array([[0.6308511, -0.04194963, 0.77476907, 0.0041838],
                    [-0.77565985, -0.05914015, 0.62837428, 0.15558023],
                    [0.01945989, -0.99736787, -0.06984727, 0.10039507], [0, 0, 0, 1.0]]),
        res=(346, 260)),
    "right": dict(
        K=np.array([[344.4324997, 0.0, 161.91358016], [0.0, 344.63168706, 124.92903904], [0, 0, 1.0]]),
        dist=np.array([-0.36338287, 0.30069459, -0.00040538, -0.00137928, -0.39233062]),
        T=np.array([[0.77356643, -0.02081131, 0.63337341, 0.1565858],
                    [-0.63353662, -0.04912395, 0.77215166, 0.00904179],
                    [0.01504432, -0.99857585, -0.05118536, 0.10344005], [0, 0, 0, 1.0]]),
        res=(346, 260)),
    "d435": dict(
        K=np.array([[613.087646484375, 0.0, 324.5477600097656], [0.0, 613.378173828125, 233.75576782226562],
                    [0, 0, 1.0]]),
        dist=np.zeros(5),
        T=np.array([[0.71816448, 0.00808576, 0.69582641, 0.03926356],
                    [-0.69540667, -0.02827723, 0.71805986, 0.08717657],
                    [0.0254821, -0.99956742, -0.01468481, 0.09887745], [0, 0, 0, 1.0]]),
        res=(640, 480)),
}

# DAVIS346 pixel model.  Optics/geometry are real (18.5 um pixels, ESD calibration); the circuit
# values are typical DAVIS346 figures used as the CENTRE of the randomisation ranges below - tune
# them so simulated event rates / noise match real ESD sequences (tools/esd_generate.py --stats).
DAVIS346 = dict(pixel_pitch_um=18.5, f_number=1.8, qe=0.25, psf_sigma_px=0.6,
                contrast_on=0.25, contrast_off=0.25, contrast_sigma=0.03, cutoff_hz=800, bw_half_lux=30,
                refractory_us=100, hpf_hz=0, leak_rate_hz=0.1, leak_sigma=0.5, shot_noise=1, noise_gain=0.08,
                hot_pixel_fraction=5e-4, hot_pixel_rate_hz=30, latency_us=15, latency_jitter_us=5,
                readout_mevps=12, erc_mevps=0, max_events=3000000, max_px_motion=1.0, max_substeps=16,
                bias_diff_on=0, bias_diff_off=0, bias_fo=0, bias_hpf=0, bias_refr=0)
DAVIS346_RANDOM = dict(contrast_on=(0.15, 0.45), contrast_off=(0.15, 0.45), contrast_sigma=(0.01, 0.06),
                       cutoff_hz=(200, 3000), refractory_us=(10, 500), leak_rate_hz=(0.02, 0.5),
                       noise_gain=(0.02, 0.3), hot_pixel_fraction=(0, 2e-3), latency_us=(5, 50))


def sample_sensor_params(rng: np.random.Generator) -> dict:
    """DAVIS346 circuit parameters drawn from DAVIS346_RANDOM (log-uniform for the wide ranges)."""
    out = {}
    for k, (lo, hi) in DAVIS346_RANDOM.items():
        if lo > 0 and hi / lo > 10:
            out[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            out[k] = float(rng.uniform(lo, hi))
    return out


def T_to_posquat(T):
    U, _, Vt = np.linalg.svd(T[:3, :3])                    # re-orthonormalise the calibration
    R = U @ Vt
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return T[:3, 3].copy(), q


def sensor_config(cam: str, extra: dict | None = None, calib: dict | None = None) -> dict:
    c = (calib or ESD_CALIB)[cam]
    K, d, (W, H) = c["K"], c["dist"], c["res"]
    cfg = dict(DAVIS346) if cam != "d435" else {}
    cfg.update(resolution=f"{W} {H}", fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
               distortion=" ".join(f"{x:.9g}" for x in d))
    if cam == "d435":
        cfg["backend"] = "warp"            # frames only: the C++ plugin just reserves a tiny slot
    cfg.update(extra or {})
    return cfg


def add_rig(spec: mujoco.MjSpec, parent: str = "wrist_3_link", tool0_pos=(0, 0.1, 0),
            tool0_quat=(-0.7071068, 0.7071068, 0, 0), sensor_extra: dict | None = None,
            backend: str = "cpu", calib: dict | None = None, visuals: bool = True):
    """Add tool0 (UR flange frame, z out of the flange) + the three cameras.  Sensors are named
    'left', 'right' (DAVIS346 event cameras) and 'd435' (frames only)."""
    par = spec.body(parent) if parent not in ("world", None) else spec.worldbody
    tool0 = par.add_body(name="tool0", pos=list(tool0_pos), quat=list(tool0_quat))
    tool0.add_site(name="tool0", size=[0.005, 0, 0], group=4)
    calib = calib or ESD_CALIB
    for cam in ("left", "right", "d435"):
        pos, q = T_to_posquat(calib[cam]["T"])
        extra = dict(sensor_extra or {}) if cam != "d435" else {}
        if cam != "d435" and backend != "cpu":
            extra["backend"] = backend
        K, H = calib[cam]["K"], calib[cam]["res"][1]
        fovy = float(np.degrees(2 * np.arctan(0.5 * H / K[1, 1])))
        mnt = add_evk4(spec, "tool0", pos=pos, quat=q, name=cam, config=sensor_config(cam, extra, calib),
                       housing=False, mass=0.0, fovy=fovy)
        if visuals:
            C = dict(contype=0, conaffinity=0, group=1)
            if cam == "d435":
                mnt.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.0125, 0.0125], pos=[0.02, 0, -0.014],
                             rgba=[0.75, 0.75, 0.78, 1], **C)
            else:
                mnt.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.016, 0.018, 0], pos=[0, 0, -0.019],
                             rgba=[0.05, 0.05, 0.05, 1], **C)
                mnt.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.020, 0.0225, 0.020], pos=[0, 0, -0.058],
                             rgba=[0.80, 0.20, 0.20, 1], **C)
    if visuals:                                             # blue holder: bar + flange plate
        pl, _ = T_to_posquat(calib["left"]["T"]); pr, _ = T_to_posquat(calib["right"]["T"])
        zl, zr = calib["left"]["T"][:3, 2], calib["right"]["T"][:3, 2]
        a, b = pl - 0.085 * zl, pr - 0.085 * zr
        C = dict(contype=0, conaffinity=0, group=1, rgba=[0.15, 0.35, 0.85, 1])
        tool0.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=list(a) + list(b), size=[0.012, 0, 0], **C)
        mid = 0.5 * (a + b)
        tool0.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=list(mid) + [0, 0, 0.01], size=[0.012, 0, 0], **C)
        tool0.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.04, 0.006, 0], pos=[0, 0, 0.006], **C)
    return tool0
