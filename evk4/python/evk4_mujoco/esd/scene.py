"""
ESD tabletop scene on the UR5e: dark tabletop mat on a perforated table, overhead lights (normal /
low), the stereo-DAVIS + D435 rig on the flange, and the objects of one sequence.

World frame: table top at z = 0, centre of the object area at (0, 0).  The UR5e stands on a pedestal
at (0, -base_dist) so that the rig can hover 0.62 / 0.82 m above the objects (the paper's two heights)
with the flange horizontal, as on the authors' UR10.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .. import load_plugin
from . import catalog
from .rig import add_rig

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_XML = os.path.join(HERE, "..", "..", "..", "scenes", "ur5e_evk4_robot.xml")


@dataclass
class SceneConfig:
    objects: list = field(default_factory=list)       # [(ObjSpec, x, y, yaw, rest_pose, z_drop)]
    light: str = "normal"                             # normal | low   (or a float = lux on the table)
    robot: bool = True                                # False: rig on a free-flying mocap body
    base_pos: tuple = (0.0, -0.48, 0.45)              # UR5e base on a pedestal (table top = 0)
    sensor_extra: dict = field(default_factory=dict)  # DAVIS346 overrides (randomised parameters)
    backend: str = "cpu"
    timestep: float = 0.001
    table: dict = field(default_factory=dict)         # mat colour / noise / show holes
    tex_cache: str = "esd_textures"
    tex_override: dict = field(default_factory=dict)  # object name -> top-face photo (PNG)
    lux_per_unit: float = 1000.0
    seed: int = 0


def _tex(spec, name, img, cache):
    from PIL import Image
    os.makedirs(cache, exist_ok=True)
    path = os.path.abspath(os.path.join(cache, f"{name}.png"))
    Image.fromarray(img).save(path)
    t = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=path)
    return t.name


def _table_textures(rng, cfg):
    t = cfg.table
    res = 512
    mat_v = t.get("mat_gray", 0.20)
    mat = np.full((res, res, 3), mat_v) * np.array(t.get("mat_tint", (1.0, 1.0, 1.04)))
    n = rng.normal(0, 1, (res // 4, res // 4))
    mat *= 1 + t.get("mat_noise", 0.02) * np.kron(n, np.ones((4, 4)))[..., None]
    board = np.full((res, res, 3), t.get("board_gray", 0.42))   # optical breadboard, 25 mm hole grid
    yy, xx = np.mgrid[0:res, 0:res]
    pitch = res / 16
    holes = np.hypot((xx % pitch) - pitch / 2, (yy % pitch) - pitch / 2) < pitch * 0.18
    board[holes] = t.get("hole_gray", 0.75)                        # holes look bright in the ESD frames
    return (np.clip(mat, 0, 1) * 255).astype(np.uint8), (np.clip(board, 0, 1) * 255).astype(np.uint8)


def build(cfg: SceneConfig):
    """Returns (MjModel, info) with info['labels'] = geom id -> object label, objects, names."""
    load_plugin()
    rng = np.random.default_rng(cfg.seed)
    spec = mujoco.MjSpec.from_file(os.path.abspath(ROBOT_XML))
    old = spec.body("evk4")
    if old is not None:
        spec.delete(old)
    spec.option.timestep = cfg.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    g = spec.visual.global_
    g.offwidth, g.offheight = max(g.offwidth, 1280), max(g.offheight, 960)
    spec.visual.headlight.ambient = [0.05, 0.05, 0.05]
    spec.visual.headlight.diffuse = [0.05, 0.05, 0.05]
    spec.visual.headlight.specular = [0, 0, 0]
    wb = spec.worldbody
    # -------- robot on its pedestal (or no robot: mocap rig)
    base = spec.body("base")
    if cfg.robot:
        base.pos = list(cfg.base_pos)
        wb.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.08, cfg.base_pos[2] / 2 + 0.375, 0],
                    pos=[cfg.base_pos[0], cfg.base_pos[1], (cfg.base_pos[2] - 0.75) / 2], rgba=[0.3, 0.3, 0.32, 1],
                    group=1)
        parent = "wrist_3_link"
    else:
        spec.delete(base)
        wb.add_body(name="rig", mocap=True, pos=[0, 0, 0.9])
        parent = "rig"
    add_rig(spec, parent=parent, tool0_pos=(0, 0.1, 0) if cfg.robot else (0, 0, 0),
            tool0_quat=(-0.7071068, 0.7071068, 0, 0) if cfg.robot else (1, 0, 0, 0),
            sensor_extra=dict(cfg.sensor_extra, lux_per_unit=cfg.lux_per_unit), backend=cfg.backend)
    # -------- table, mat, floor
    mat_img, board_img = _table_textures(rng, cfg)
    tm = spec.add_material(name="esd_mat", specular=0.05, shininess=0.1, texrepeat=[1, 1])
    tm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _tex(spec, "esd_mat_tex", mat_img, cfg.tex_cache)
    tb = spec.add_material(name="esd_board", specular=0.3, shininess=0.4, texrepeat=[1, 1])
    tb.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _tex(spec, "esd_board_tex", board_img, cfg.tex_cache)
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.60, 0.60, 0.02], pos=[0, 0, -0.02],
                material="esd_board", rgba=[1, 1, 1, 1], group=1, friction=[0.8, 0.01, 0.001])
    off = cfg.table.get("mat_offset", (0.07, 0.0))        # a strip of the breadboard shows at -x (as in ESD)
    wb.add_geom(name="mat", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.48, 0.46, 0.0015],
                pos=[off[0], off[1], 0.0015], material="esd_mat", rgba=[1, 1, 1, 1], group=1,
                friction=[0.9, 0.01, 0.001])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75],
                rgba=[0.35, 0.35, 0.36, 1], group=1)
    # -------- lights: two overhead fluorescent-like directional lights + one shadowing spot
    lux = cfg.light if isinstance(cfg.light, (int, float)) else {"normal": 450.0, "low": 25.0}[cfg.light]
    k = lux / cfg.lux_per_unit / 1.25                       # the 3 lights + ambient sum to ~1.25 k at the table
    jit = rng.uniform(-0.25, 0.25, 4) if cfg.table.get("light_jitter", True) else np.zeros(4)
    wb.add_light(name="ceiling_a", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0.3, 0.2, 2.0],
                 dir=[-0.15 + 0.1 * jit[0], -0.1 + 0.1 * jit[1], -1], diffuse=[0.40 * k] * 3,
                 specular=[0.1 * k] * 3, ambient=[0.10 * k] * 3, castshadow=False)
    wb.add_light(name="ceiling_b", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[-0.3, -0.2, 2.0],
                 dir=[0.12 + 0.1 * jit[2], 0.08 + 0.1 * jit[3], -1], diffuse=[0.30 * k] * 3,
                 specular=[0.1 * k] * 3, ambient=[0.05 * k] * 3, castshadow=False)
    wb.add_light(name="ceiling_spot", type=mujoco.mjtLightType.mjLIGHT_SPOT, pos=[0.1, 0.3, 1.6],
                 dir=[-0.05, -0.15, -1], cutoff=70, exponent=1, diffuse=[0.25 * k / 0.6] * 3,
                 specular=[0.05 * k] * 3, ambient=[0, 0, 0], castshadow=True)
    # -------- objects
    names, label_of = [], {}
    for i, (obj, x, y, yaw, pose, zdrop) in enumerate(cfg.objects):
        nm = f"obj{i}_{obj.name}"
        z = 0.003 + catalog.rest_height(obj, pose) + zdrop
        catalog.add_object(spec, obj, nm, (x, y, z), yaw, pose, tex_cache=cfg.tex_cache,
                           tex_seed=cfg.seed * 100 + i if obj.label == 0 else 0,
                           tex_override=cfg.tex_override.get(obj.name))
        names.append(nm)
        label_of[nm] = obj.label if obj.label > 0 else i + 1
    m = spec.compile()
    geom_label = np.zeros(m.ngeom, np.int32)
    geom_obj = np.full(m.ngeom, -1, np.int32)
    for i, nm in enumerate(names):
        b = m.body(nm).id
        sel = m.geom_bodyid == b
        geom_label[sel] = label_of[nm]
        geom_obj[sel] = i
    info = dict(objects=names, label_of=label_of, geom_label=geom_label, geom_obj=geom_obj, lux=lux,
                spec=spec, robot=cfg.robot)
    return m, info


# ---------------------------------------------------------------- object placement
def place_objects(objs, rng, occlusion: bool, area=(0.30, 0.22), margin=0.012, max_tries=400):
    """Non-overlapping footprints (occlusion=False) or deliberately stacked / overlapping ones
    (occlusion=True: ~40 % of the objects are dropped onto others).  Returns cfg.objects entries."""
    placed = []
    order = sorted(range(len(objs)), key=lambda i: -catalog.footprint_radius(objs[i]))
    for i in order:
        o = objs[i]
        pose = str(rng.choice(o.rest))
        r = catalog.footprint_radius(o)
        stack = occlusion and placed and rng.random() < 0.45
        for _ in range(max_tries):
            yaw = rng.uniform(-np.pi, np.pi)
            if stack:                                          # on / leaning on a bigger object
                j = rng.integers(len(placed))
                _, xj, yj, _, _, _ = placed[j]
                x, y = xj + rng.normal(0, 0.03), yj + rng.normal(0, 0.03)
                zdrop = 0.02 + max(0.12, 2 * catalog.rest_height(placed[j][0], placed[j][4]))
                placed.append((o, x, y, yaw, pose, zdrop))
                break
            x, y = rng.uniform(-area[0] + r, area[0] - r), rng.uniform(-area[1] + r, area[1] - r)
            ok = all(np.hypot(x - xj, y - yj) > r + catalog.footprint_radius(oj) + margin
                     for oj, xj, yj, _, _, zj in placed if zj == 0)
            if ok or (occlusion and rng.random() < 0.15):      # partial overlaps also occlude
                placed.append((o, x, y, yaw, pose, 0.0 if ok else 0.05))
                break
        else:
            placed.append((o, rng.uniform(-area[0], area[0]), rng.uniform(-area[1], area[1]), 0.0, pose, 0.1))
    return placed
