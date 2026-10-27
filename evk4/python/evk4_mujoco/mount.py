"""
Mount a Prophesee EVK4 event camera on ANY MuJoCo robot / body, without editing XML by hand.

    import evk4_mujoco as evk
    m = evk.attach_evk4("mujoco_menagerie/franka_emika_panda/scene.xml", body="hand",
                        pos=(0.05, 0, 0.06), euler_deg=(0, -30, 0),
                        config=dict(resolution="640 480", focal_length_mm=3.0))

Mount frame convention (== OpenCV camera frame, so hand-eye calibration results drop straight in):
    origin  = optical centre of the lens,
    +z      = viewing direction,  +x = image right,  +y = image down.
`pos` / `quat` / `euler_deg` give this frame relative to the parent body.  The housing (adapter,
body, lens) is drawn behind the optical centre and is invisible to its own sensor.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from . import load_plugin

PLUGIN = "prophesee.sensor.evk4"

# Same defaults as scenes/scene_ur5e_evk4.xml (full-resolution EVK4 HD, 4 mm f/2 lens)
DEFAULT_CONFIG = {
    "resolution": "1280 720", "binning": "1", "pixel_pitch_um": "4.86", "focal_length_mm": "4.0",
    "f_number": "2.0", "qe": "0.5", "psf_sigma_px": "0.5", "lux_per_unit": "1000", "background_lux": "60",
    "contrast_on": "0.22", "contrast_off": "0.22", "contrast_sigma": "0.02", "cutoff_hz": "2000",
    "refractory_us": "50", "hpf_hz": "0", "leak_rate_hz": "0.05", "shot_noise": "1", "latency_us": "20",
    "erc_mevps": "0", "max_events": "400000", "max_px_motion": "1.0", "max_substeps": "12",
}


def _fmt(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(str(x) for x in v)
    return str(v)


def euler_to_quat(euler_deg, seq="xyz"):
    """Euler angles [deg] -> MuJoCo quaternion (w x y z); seq as in MJCF 'eulerseq'."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.radians(np.asarray(euler_deg, float)), seq)
    return q


def look_at_quat(eye, target, up=(0, 0, 1)):
    """Quaternion of an OpenCV-convention mount frame at `eye` looking at `target` (parent frame).
    Useful for fixed cameras:  attach_evk4(xml, "world", pos=eye, quat=look_at_quat(eye, target))."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, (1, 0, 0) if abs(z[0]) < 0.9 else (0, 1, 0))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).ravel())
    return q


def _find_body(spec, name):
    if name in ("world", "worldbody", None):
        return spec.worldbody
    b = spec.body(name)
    if b is None:
        names = [x.name for x in spec.bodies if x.name]
        raise KeyError(f"body '{name}' not found. Bodies in this model: {', '.join(names)}")
    return b


def add_evk4(spec: mujoco.MjSpec, body: str, pos=(0, 0, 0), quat=None, euler_deg=None, name: str = "evk4",
             config: dict | None = None, housing: bool = True, mass: float = 0.075, geom_group: int = 1,
             fovy: float | None = None):
    """Add an EVK4 (housing + camera + sensor plugin) to an MjSpec.  Returns the mount MjsBody."""
    load_plugin()
    parent = _find_body(spec, body)
    if quat is None:
        quat = euler_to_quat(euler_deg) if euler_deg is not None else (1, 0, 0, 0)
    q = np.asarray(quat, float)
    q = q / np.linalg.norm(q)
    if spec.body(name) is not None or any(s.name == name for s in spec.sensors):
        raise ValueError(f"an element named '{name}' already exists - pass name=...")
    mnt = parent.add_body(name=name, pos=list(map(float, pos)), quat=list(q))
    if mass > 0:
        mnt.explicitinertial = True
        mnt.mass = float(mass)
        mnt.ipos = [0, 0, -0.035]
        mnt.inertia = [2e-5, 2e-5, 1.5e-5]
    try:                                                   # follow the parent's gravity compensation
        mnt.gravcomp = float(parent.gravcomp)
    except Exception:
        pass
    if housing:
        common = dict(contype=0, conaffinity=0, group=geom_group)
        # lens (front face at the optical centre), camera body, mounting adapter - all behind z=0
        mnt.add_geom(name=f"{name}_lens", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.014, 0.012, 0],
                     pos=[0, 0, -0.013], rgba=[0.02, 0.02, 0.02, 1], **common)
        mnt.add_geom(name=f"{name}_housing", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.015, 0.018],
                     pos=[0, 0, -0.043], rgba=[0.09, 0.09, 0.10, 1], **common)
        mnt.add_geom(name=f"{name}_adapter", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.032, 0.004, 0],
                     pos=[0, 0, -0.065], rgba=[0.15, 0.15, 0.16, 1], **common)
    cfg = dict(DEFAULT_CONFIG)
    for k, v in (config or {}).items():
        cfg[k] = _fmt(v)
    if fovy is None:                                       # viewer frustum / fallback only
        res = [float(x) for x in cfg["resolution"].split()]
        f = float(cfg.get("focal_length_mm", 0) or 0)
        pitch = float(cfg.get("pixel_pitch_um", 4.86)) * 1e-3
        fovy = float(np.degrees(2 * np.arctan(0.5 * res[1] * pitch / f))) if f > 0 else 45.0
    # MuJoCo cameras look along their local -z: rotate 180 deg about x -> looks along mount +z
    mnt.add_camera(name=f"{name}_cam", pos=[0, 0, 0], quat=[0, 1, 0, 0], fovy=fovy)
    # named plugin instance (<extension><plugin><instance>) + sensor referencing it
    inst = spec.add_plugin(name=f"{name}_plugin", plugin_name=PLUGIN, active=True)
    inst.config = cfg
    sen = spec.add_sensor(name=name, type=mujoco.mjtSensor.mjSENS_PLUGIN, objtype=mujoco.mjtObj.mjOBJ_CAMERA,
                          objname=f"{name}_cam")
    sen.plugin = inst
    spec.activate_plugin(PLUGIN)                           # <extension><plugin plugin=.../> (idempotent)
    # mesh ray casting from several threads must not write BVH debug flags
    spec.visual.global_.bvactive = False
    return mnt


def attach_evk4(robot_xml: str, body: str, pos=(0, 0, 0), quat=None, euler_deg=None, name: str = "evk4",
                config: dict | None = None, housing: bool = True, mass: float = 0.075, geom_group: int = 1,
                backend: str = "cpu", offscreen=(1920, 1080), save_xml: str | None = None,
                edit=None, return_spec: bool = False):
    """Load `robot_xml` (a robot or a whole scene, <include>s allowed), mount an EVK4 on `body` and
    compile.  `edit(spec)` may add more things (props, a second camera, ...) before compiling.
    backend="warp" -> the C++ plugin only reserves the slot, the GPU engine renders.
    save_xml: write the combined model next to `robot_xml` (relative asset paths keep working)."""
    load_plugin()
    spec = mujoco.MjSpec.from_file(os.path.abspath(robot_xml))
    cfg = dict(config or {})
    if backend != "cpu":
        cfg["backend"] = backend
    add_evk4(spec, body, pos, quat, euler_deg, name, cfg, housing, mass, geom_group)
    g = spec.visual.global_
    g.offwidth, g.offheight = max(g.offwidth, offscreen[0]), max(g.offheight, offscreen[1])
    if edit is not None:
        edit(spec)
    m = spec.compile()
    if save_xml:
        out = save_xml if os.path.isabs(save_xml) else os.path.join(os.path.dirname(os.path.abspath(robot_xml)), save_xml)
        with open(out, "w") as f:
            f.write(spec.to_xml())
    return (m, spec) if return_spec else m


def add_props(spec: mujoco.MjSpec, center=(0.5, 0, 0), size: float = 0.25, spin_hz: float = 2.0, seed: int = 0):
    """A small, texture-rich workspace for event cameras (no texture files needed): a checkerboard
    target, a free-spinning striped disk, coloured blocks and a ball.  Returns names of movers."""
    rng = np.random.default_rng(seed)
    cx, cy, cz = map(float, center)
    t1 = spec.add_texture(name="evk4_props_checker", type=mujoco.mjtTexture.mjTEXTURE_2D,
                          builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=256, height=256,
                          rgb1=[0.9, 0.9, 0.88], rgb2=[0.08, 0.08, 0.09])
    t2 = spec.add_texture(name="evk4_props_disk", type=mujoco.mjtTexture.mjTEXTURE_2D,
                          builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=64, height=512,
                          rgb1=[0.95, 0.75, 0.2], rgb2=[0.1, 0.1, 0.12])
    m1 = spec.add_material(name="evk4_props_checker", texrepeat=[4, 4], specular=0.05)
    m1.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t1.name
    m2 = spec.add_material(name="evk4_props_disk", texrepeat=[1, 1], specular=0.2)
    m2.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t2.name
    wb = spec.worldbody
    s = size
    wb.add_geom(name="evk4_props_board", type=mujoco.mjtGeom.mjGEOM_BOX, size=[s, s, 0.004],
                pos=[cx, cy, cz + 0.004], material=m1.name, contype=1, conaffinity=1)
    disk = wb.add_body(name="evk4_props_disk", pos=[cx + 0.45 * s, cy - 0.45 * s, cz + 0.03])
    disk.add_joint(name="evk4_props_spin", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0)
    disk.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.35 * s, 0.01, 0], material=m2.name,
                  contype=0, conaffinity=0, mass=0.2)
    for i in range(3):
        col = rng.uniform(0.15, 0.95, 3)
        wb.add_geom(name=f"evk4_props_block{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.03, 0.03, 0.03 + 0.02 * i], rgba=list(col) + [1],
                    pos=[cx - 0.5 * s + 0.2 * s * i, cy + 0.55 * s, cz + 0.038 + 0.02 * i])
    ball = wb.add_body(name="evk4_props_ball", pos=[cx - 0.4 * s, cy - 0.3 * s, cz + 0.04])
    ball.add_freejoint(name="evk4_props_ball")
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0], rgba=[0.85, 0.2, 0.15, 1], mass=0.05)
    return dict(spin_joint="evk4_props_spin", spin_hz=spin_hz, ball="evk4_props_ball")


def start_props(m: mujoco.MjModel, d: mujoco.MjData, props: dict, ball_vel=(0.4, 0.25, 0)):
    """Initial motion for add_props(): spinning disk + rolling ball."""
    j = m.joint(props["spin_joint"]).id
    d.qvel[m.jnt_dofadr[j]] = 2 * np.pi * props["spin_hz"]
    jb = m.joint(props["ball"]).id
    d.qvel[m.jnt_dofadr[jb]:m.jnt_dofadr[jb] + 3] = ball_vel
    mujoco.mj_forward(m, d)
