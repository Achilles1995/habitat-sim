"""
Objects of the ESD dataset (Huang et al., "A neuromorphic dataset for tabletop object segmentation in
indoor cluttered environment", Scientific Data 11:127, 2024) as MuJoCo replicas, plus a procedural
object generator for domain randomisation.

The paper does not list its objects by name.  The replicas below are identified from the paper's
figures and the example frame of the authors' labelling tool (github.com/yellow07200/ESD_labeling_tool,
examples/RGB.png + Mask.png, 10-object scene at 0.82 m).  Footprints were measured from that mask
(0.82 m / fx 613 px -> 1.34 mm/px, corrected for object height); heights are typical values for such
products.  Label ids and mask colours are the ones used by the authors' code (read_label.m).
Refine any entry with tools/esd_measure_objects.py on real ESD frames (sizes + top-face textures).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

# ---------------------------------------------------------------- ESD label palettes (read_label.m)
ESD1_PALETTE = {1: (51, 221, 255), 2: (250, 50, 83), 3: (52, 209, 183), 4: (255, 0, 124), 5: (255, 96, 55),
                6: (221, 255, 51), 7: (36, 179, 83), 8: (184, 61, 245), 9: (102, 255, 102), 10: (255, 204, 51)}
ESD2_PALETTE = {1: (240, 120, 240), 2: (42, 125, 209), 3: (178, 80, 80), 4: (204, 51, 102), 5: (204, 153, 51)}


@dataclass
class ObjSpec:
    name: str
    shape: str                   # box | cylinder | cylinder_lying | bottle | cup | pouch | disc | sphere | ellipsoid
    size: tuple                  # full dimensions [m]: box (L, W, H); cylinder (diameter, height/length); ...
    rgb: tuple                   # base colour 0..1
    texture: dict = field(default_factory=dict)   # procedural texture recipe (see make_texture)
    label: int = 0               # ESD label id (0 = not an ESD object)
    mask_rgb: tuple = (0, 0, 0)  # ESD mask colour
    density: float = 400.0       # kg/m^3 for the physics
    rest: tuple = ("flat",)      # allowed resting poses
    note: str = ""


# ---------------------------------------------------------------- ESD-1 (training, "known" objects)
ESD1 = [
    ObjSpec("book", "box", (0.230, 0.155, 0.030), (0.93, 0.89, 0.74), dict(kind="book", ink=(0.10, 0.55, 0.62)),
            1, ESD1_PALETTE[1], 600, note="paperback 'Optimal Control and Estimation' (cream cover, teal title)"),
    ObjSpec("white_box", "box", (0.120, 0.080, 0.075), (0.95, 0.94, 0.90),
            dict(kind="label", colors=[(0.85, 0.70, 0.15)], text=True), 2, ESD1_PALETTE[2], 150,
            note="small white product box with yellow logo"),
    ObjSpec("blue_pack", "box", (0.100, 0.055, 0.030), (0.10, 0.25, 0.75),
            dict(kind="label", colors=[(0.95, 0.95, 0.97), (0.90, 0.25, 0.55)], text=True), 3, ESD1_PALETTE[3], 300,
            note="blue/white biscuit or toothpaste pack"),
    ObjSpec("tennis_tube", "cylinder_lying", (0.075, 0.215), (0.75, 0.80, 0.10),
            dict(kind="tube", colors=[(0.05, 0.35, 0.10), (0.95, 0.85, 0.10)], cap=(0.05, 0.05, 0.05)),
            4, ESD1_PALETTE[4], 250, rest=("lying",), note="Dunlop Fort tennis-ball tube, black cap"),
    ObjSpec("milk_carton", "box", (0.095, 0.055, 0.040), (0.85, 0.10, 0.12),
            dict(kind="label", colors=[(0.97, 0.97, 0.97)], text=True), 5, ESD1_PALETTE[5], 500,
            rest=("flat", "side"), note="small red/white milk or juice carton"),
    ObjSpec("tea_box", "box", (0.160, 0.085, 0.070), (0.08, 0.10, 0.30),
            dict(kind="label", colors=[(0.75, 0.12, 0.10), (0.85, 0.65, 0.35)], text=True), 6, ESD1_PALETTE[6], 250,
            note="dark-blue tea box with red picture"),
    ObjSpec("green_bottle", "bottle", (0.066, 0.200), (0.10, 0.35, 0.12),
            dict(kind="bottle", label=(0.85, 0.85, 0.30)), 7, ESD1_PALETTE[7], 500, rest=("upright",),
            note="small green glass bottle, standing"),
    ObjSpec("blue_glass", "cup", (0.080, 0.100), (0.10, 0.55, 0.65), dict(kind="cup", inside=(0.03, 0.25, 0.33)),
            8, ESD1_PALETTE[8], 400, rest=("upright",), note="teal glass tumbler (rendered opaque)"),
    ObjSpec("pencil_case", "pouch", (0.215, 0.068, 0.035), (0.05, 0.05, 0.06),
            dict(kind="pouch", rim=(0.92, 0.92, 0.90)), 9, ESD1_PALETTE[9], 300,
            note="glossy black zip pouch with white rim"),
    ObjSpec("orange_cap", "disc", (0.045, 0.014), (0.98, 0.72, 0.12), dict(kind="solid"), 10, ESD1_PALETTE[10], 400,
            note="small orange disc (cap), label 10 - identity uncertain"),
]
# nested object sets used for O2..O10 (Fig. 6: 2 = book + box, 10 = all)
ESD1_ORDER = ["book", "white_box", "tennis_tube", "blue_pack", "tea_box", "milk_carton", "green_bottle",
              "blue_glass", "pencil_case", "orange_cap"]

# ---------------------------------------------------------------- ESD-2 (testing, "unknown" objects)
# !! keep these OUT of training data if you evaluate unseen-object segmentation on real ESD-2 !!
ESD2 = [
    ObjSpec("pink_box", "box", (0.135, 0.100, 0.060), (0.93, 0.55, 0.80),
            dict(kind="label", colors=[(0.97, 0.97, 0.97), (0.30, 0.70, 0.30)], text=True), 1, ESD2_PALETTE[1], 200),
    ObjSpec("white_roll", "cup", (0.085, 0.050), (0.95, 0.95, 0.95), dict(kind="cup", inside=(0.70, 0.70, 0.72)),
            2, ESD2_PALETTE[2], 300, rest=("upright",), note="white round object (roll / lid)"),
    ObjSpec("water_bottle", "bottle_lying", (0.065, 0.210), (0.80, 0.88, 0.95),
            dict(kind="bottle", label=(0.15, 0.35, 0.85)), 3, ESD2_PALETTE[3], 150, rest=("lying",),
            note="plastic water bottle (rendered opaque)"),
    ObjSpec("yellow_cup", "cup", (0.070, 0.085), (0.97, 0.82, 0.15), dict(kind="cup", inside=(0.85, 0.20, 0.15)),
            4, ESD2_PALETTE[4], 300, rest=("upright",)),
    ObjSpec("snack_pack", "box", (0.110, 0.070, 0.030), (0.90, 0.60, 0.15),
            dict(kind="label", colors=[(0.70, 0.15, 0.10)], text=True), 5, ESD2_PALETTE[5], 300),
]
# extra objects visible in Fig. 1 of the paper (not labelled in ESD)
EXTRA = [
    ObjSpec("cardboard_box", "box", (0.130, 0.120, 0.080), (0.70, 0.52, 0.32), dict(kind="noise", amp=0.06)),
    ObjSpec("black_box", "box", (0.150, 0.080, 0.060), (0.06, 0.06, 0.07), dict(kind="label", colors=[(0.8, 0.8, 0.8)],
                                                                             text=True)),
]
CATALOG = {o.name: o for o in ESD1 + ESD2 + EXTRA}


# ---------------------------------------------------------------- procedural textures
def _rng(seed):
    return np.random.default_rng(seed)


def make_texture(spec: ObjSpec, seed: int = 0, res: int = 256) -> np.ndarray:
    """(res, res, 3) uint8 texture from the object's recipe."""
    r = _rng(seed)
    base = np.array(spec.rgb, float)
    t = spec.texture or {}
    kind = t.get("kind", "solid")
    img = np.ones((res, res, 3)) * base
    yy, xx = np.mgrid[0:res, 0:res] / res
    if kind in ("label", "book"):
        cols = t.get("colors", [])
        if kind == "book":
            ink = np.array(t.get("ink", (0.1, 0.5, 0.6)))
            for i in range(4):                                  # title lines
                y0 = 0.12 + 0.09 * i
                img[(yy > y0) & (yy < y0 + 0.055) & (xx > 0.1) & (xx < 0.1 + r.uniform(0.5, 0.8))] = ink
            for i in range(10):                                 # pencil-drawing strokes
                x0, y0 = r.uniform(0.2, 0.7), r.uniform(0.55, 0.9)
                ang = r.uniform(0, np.pi)
                d = np.abs((xx - x0) * np.sin(ang) - (yy - y0) * np.cos(ang))
                img[(d < 0.004) & (np.hypot(xx - x0, yy - y0) < 0.12)] *= 0.55
        for c in cols:                                          # colour blocks / logo
            c = np.array(c)
            if r.random() < 0.5:
                y0, h = r.uniform(0.1, 0.7), r.uniform(0.1, 0.3)
                img[(yy > y0) & (yy < y0 + h)] = c
            else:
                cx, cy, rad = r.uniform(0.25, 0.75), r.uniform(0.25, 0.75), r.uniform(0.08, 0.2)
                img[np.hypot(xx - cx, yy - cy) < rad] = c
        if t.get("text", False):                                # text-like dashes
            dark = base.mean() > 0.45
            ink = np.array((0.08, 0.08, 0.1)) if dark else np.array((0.95, 0.95, 0.95))
            for _ in range(r.integers(4, 10)):
                x0, y0 = r.uniform(0.05, 0.7), r.uniform(0.05, 0.9)
                w = r.uniform(0.1, 0.3)
                img[(yy > y0) & (yy < y0 + 0.025) & (xx > x0) & (xx < x0 + w) & ((xx * res // 7) % 3 != 0)] = ink
    elif kind == "tube":
        c1, c2 = (np.array(c) for c in t.get("colors", [(0.1, 0.4, 0.1), (0.9, 0.85, 0.1)]))
        g = np.clip((xx - 0.3) / 0.5, 0, 1)[..., None]
        img = c1 * (1 - g) + c2 * g
        img[(yy > 0.40) & (yy < 0.60) & ((xx * 14).astype(int) % 2 == 0) & (xx > 0.35) & (xx < 0.9)] = (0.05, 0.05, 0.05)
    elif kind == "bottle":
        lab = np.array(t.get("label", (0.9, 0.9, 0.3)))
        img[(yy > 0.35) & (yy < 0.7)] = lab
    elif kind == "cup":
        img[np.hypot(xx - 0.5, yy - 0.5) < 0.42] = np.array(t.get("inside", base * 0.4))
    elif kind == "pouch":
        img[np.abs(yy - 0.5) > 0.40] = np.array(t.get("rim", (0.9, 0.9, 0.9)))
        for _ in range(12):                                     # glossy highlights
            cx, cy = r.uniform(0.1, 0.9), r.uniform(0.2, 0.8)
            img[np.hypot((xx - cx) * 0.5, yy - cy) < r.uniform(0.02, 0.05)] = (0.55, 0.55, 0.6)
    elif kind == "stripes":
        c = np.array(t.get("color2", r.uniform(0, 1, 3)))
        k = t.get("freq", r.integers(3, 12))
        img[((xx * k).astype(int) % 2) == 0] = c
    elif kind == "checker":
        c = np.array(t.get("color2", r.uniform(0, 1, 3)))
        k = t.get("freq", r.integers(2, 8))
        img[(((xx * k).astype(int) + (yy * k).astype(int)) % 2) == 0] = c
    elif kind == "noise":
        amp = t.get("amp", 0.1)
        n = r.normal(0, 1, (res // 8, res // 8))
        n = np.kron(n, np.ones((8, 8)))[:res, :res]
        img = img * (1 + amp * n[..., None])
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def texture_file(spec: ObjSpec, cache_dir: str, seed: int = 0, override: str | None = None) -> str:
    """Write (or reuse) the texture PNG; `override` = a real top-face photo (esd_measure_objects.py)."""
    if override and os.path.exists(override):
        return os.path.abspath(override)
    from PIL import Image
    key = hashlib.md5(repr((spec.name, spec.rgb, sorted(spec.texture.items()), seed)).encode()).hexdigest()[:10]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{spec.name}_{key}.png")
    if not os.path.exists(path):
        Image.fromarray(make_texture(spec, seed)).save(path)
    return os.path.abspath(path)


# ---------------------------------------------------------------- MJCF construction
def footprint_radius(spec: ObjSpec) -> float:
    s = spec.size
    if spec.shape in ("box", "pouch"):
        return 0.5 * float(np.hypot(s[0], s[1]))
    if spec.shape in ("cylinder_lying", "bottle_lying"):
        return 0.5 * float(np.hypot(s[0], s[1]))
    return 0.5 * float(s[0])


def rest_height(spec: ObjSpec, pose: str) -> float:
    s = spec.size
    if spec.shape == "box":
        return {"flat": s[2], "side": s[1], "end": s[0]}.get(pose, s[2]) / 2
    if spec.shape in ("cylinder_lying", "bottle_lying"):
        return s[0] / 2
    if spec.shape in ("cylinder", "bottle", "cup", "disc"):
        return s[1] / 2 if spec.shape != "bottle" else 0.0
    if spec.shape == "pouch":
        return s[2] / 2
    return s[0] / 2


def add_object(spec_: mujoco.MjSpec, obj: ObjSpec, name: str, pos, yaw: float, pose: str = "flat",
               tex_cache: str = ".", tex_seed: int = 0, tex_override: str | None = None, group: int = 1):
    """Add `obj` as a free body `name`; returns the body.  Geoms are named '<name>/<part>'."""
    tex = spec_.add_texture(name=f"{name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                            file=texture_file(obj, tex_cache, tex_seed, tex_override))
    mat = spec_.add_material(name=f"{name}_mat", specular=0.15, shininess=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, [0, 0, yaw], "xyz")
    body = spec_.worldbody.add_body(name=name, pos=list(map(float, pos)), quat=list(q))
    body.add_freejoint(name=f"{name}_free")
    G = mujoco.mjtGeom
    s = obj.size
    common = dict(group=group, density=obj.density, friction=[0.9, 0.01, 0.001], condim=4)
    plain = lambda c: dict(rgba=list(c) + [1.0], **common)
    textured = dict(material=mat.name, rgba=[1, 1, 1, 1], **common)
    rot_x90 = [0.7071068, 0.7071068, 0, 0]                          # cylinder axis z -> y (lying)
    if obj.shape == "box":
        dims = {"flat": (s[0], s[1], s[2]), "side": (s[0], s[2], s[1]), "end": (s[2], s[1], s[0])}[pose]
        body.add_geom(name=f"{name}/body", type=G.mjGEOM_BOX, size=[d / 2 for d in dims], **textured)
    elif obj.shape == "cylinder_lying":
        body.add_geom(name=f"{name}/body", type=G.mjGEOM_CYLINDER, size=[s[0] / 2, s[1] / 2 - 0.006, 0],
                      quat=rot_x90, **textured)
        cap = obj.texture.get("cap", (0.05, 0.05, 0.05))
        body.add_geom(name=f"{name}/cap", type=G.mjGEOM_CYLINDER, size=[s[0] / 2 + 0.002, 0.006, 0],
                      pos=[0, s[1] / 2 - 0.006, 0], quat=rot_x90, **plain(cap))
    elif obj.shape in ("cylinder", "cup", "disc"):
        body.add_geom(name=f"{name}/body", type=G.mjGEOM_CYLINDER, size=[s[0] / 2, s[1] / 2, 0], **textured)
    elif obj.shape in ("bottle", "bottle_lying"):
        rb, hb = s[0] / 2, 0.68 * s[1]
        rn, hn = 0.4 * rb, 0.32 * s[1]
        if obj.shape == "bottle":                                    # body origin at the table
            body.add_geom(name=f"{name}/body", type=G.mjGEOM_CYLINDER, size=[rb, hb / 2, 0], pos=[0, 0, hb / 2],
                          **textured)
            body.add_geom(name=f"{name}/shoulder", type=G.mjGEOM_SPHERE, size=[rb * 0.98, 0, 0], pos=[0, 0, hb],
                          **plain(obj.rgb))
            body.add_geom(name=f"{name}/neck", type=G.mjGEOM_CYLINDER, size=[rn, hn / 2, 0],
                          pos=[0, 0, hb + hn / 2 + 0.3 * rb], **plain(obj.rgb))
        else:
            body.add_geom(name=f"{name}/body", type=G.mjGEOM_CYLINDER, size=[rb, hb / 2, 0], pos=[0, -hn / 2, 0],
                          quat=rot_x90, **textured)
            body.add_geom(name=f"{name}/neck", type=G.mjGEOM_CYLINDER, size=[rn, hn / 2, 0],
                          pos=[0, hb / 2, 0], quat=rot_x90, **plain(obj.rgb))
    elif obj.shape == "pouch":
        rim = obj.texture.get("rim", (0.9, 0.9, 0.9))
        body.add_geom(name=f"{name}/rim", type=G.mjGEOM_ELLIPSOID, size=[s[0] / 2, s[1] / 2, s[2] * 0.3],
                      pos=[0, 0, -s[2] * 0.2], **plain(rim))
        body.add_geom(name=f"{name}/body", type=G.mjGEOM_ELLIPSOID, size=[s[0] / 2 - 0.004, s[1] / 2 - 0.004, s[2] / 2],
                      **textured)
    elif obj.shape == "sphere":
        body.add_geom(name=f"{name}/body", type=G.mjGEOM_SPHERE, size=[s[0] / 2, 0, 0], **textured)
    elif obj.shape == "ellipsoid":
        body.add_geom(name=f"{name}/body", type=G.mjGEOM_ELLIPSOID, size=[d / 2 for d in s], **textured)
    else:
        raise ValueError(obj.shape)
    return body


# ---------------------------------------------------------------- procedural random objects
def random_object(rng: np.random.Generator, i: int) -> ObjSpec:
    """A random household-like object (domain randomisation for unseen-object segmentation)."""
    shape = rng.choice(["box", "box", "box", "cylinder", "cylinder_lying", "bottle", "bottle_lying", "cup",
                        "pouch", "sphere", "ellipsoid", "disc"])
    if shape == "box":
        L = rng.uniform(0.04, 0.24); W = rng.uniform(0.03, min(L, 0.16)); H = rng.uniform(0.01, 0.10)
        size = (L, W, H)
    elif shape in ("cylinder", "cup"):
        size = (rng.uniform(0.03, 0.10), rng.uniform(0.02, 0.14))
    elif shape in ("cylinder_lying", "bottle_lying"):
        size = (rng.uniform(0.03, 0.09), rng.uniform(0.08, 0.26))
    elif shape == "bottle":
        size = (rng.uniform(0.04, 0.09), rng.uniform(0.12, 0.26))
    elif shape == "pouch":
        size = (rng.uniform(0.10, 0.25), rng.uniform(0.04, 0.10), rng.uniform(0.015, 0.05))
    elif shape == "sphere":
        size = (rng.uniform(0.03, 0.09),)
    elif shape == "ellipsoid":
        a = rng.uniform(0.03, 0.10)
        size = (a, rng.uniform(0.6, 1.0) * a, rng.uniform(0.4, 0.9) * a)
    else:
        size = (rng.uniform(0.02, 0.08), rng.uniform(0.005, 0.02))
    base = rng.uniform(0.02, 1.0, 3) ** rng.uniform(0.6, 1.5)
    kind = rng.choice(["solid", "label", "label", "stripes", "checker", "noise", "book"])
    tex = dict(kind=str(kind))
    if kind in ("label", "book"):
        tex["colors"] = [tuple(rng.uniform(0, 1, 3)) for _ in range(rng.integers(1, 4))]
        tex["text"] = bool(rng.random() < 0.7)
        tex["ink"] = tuple(rng.uniform(0, 1, 3))
    if kind in ("stripes", "checker"):
        tex["color2"] = tuple(rng.uniform(0, 1, 3)); tex["freq"] = int(rng.integers(2, 12))
    if kind == "noise":
        tex["amp"] = float(rng.uniform(0.05, 0.3))
    if shape == "cup":
        tex = dict(kind="cup", inside=tuple(base * rng.uniform(0.2, 0.6)))
    rest = {"box": ("flat", "flat", "side"), "cylinder_lying": ("lying",), "bottle_lying": ("lying",),
            "bottle": ("upright",), "cup": ("upright",)}.get(shape, ("flat",))
    return ObjSpec(f"rand{i}", shape, tuple(float(x) for x in size), tuple(float(x) for x in base), tex,
                   label=0, density=float(rng.uniform(150, 800)), rest=rest)


# ---------------------------------------------------------------- real measurements (esd_measure_objects.py)
def apply_measured(objs, measured_json: str, cache_dir: str):
    """Replace catalogue sizes by measured ones and use the real top-face photos as textures
    (re-mapped to how each primitive is textured).  Returns (objects, {name: texture png})."""
    import dataclasses
    import json
    from PIL import Image
    meas = json.load(open(measured_json))
    out, tex = [], {}
    os.makedirs(cache_dir, exist_ok=True)
    for o in objs:
        mm = meas.get(o.name)
        if not mm:
            out.append(o)
            continue
        o = dataclasses.replace(o, size=tuple(float(x) for x in mm["size"]))
        out.append(o)
        src = mm.get("texture")
        if not src or not os.path.exists(src) or o.shape in ("bottle", "pouch"):
            continue
        img = np.array(Image.open(src).convert("RGB"))
        if o.shape in ("cylinder_lying", "bottle_lying"):       # top view = half the circumference
            img = np.rot90(img)
            img = np.concatenate([img, img[:, ::-1]], 1)
        path = os.path.join(cache_dir, f"measured_{o.name}.png")
        Image.fromarray(np.ascontiguousarray(img)).save(path)
        tex[o.name] = os.path.abspath(path)
    return out, tex
