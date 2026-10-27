#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Live, keyboard-driven viewer for habitat-sim that works on Windows (and Linux).

Shows the simulator's camera in a window in real time and lets you move the
camera, walk a Habitat 3.0 SMPL-X humanoid (with its motion-capture walk
cycle), drive a robot (Spot or Fetch), drop physics objects and save
snapshots (RGB, depth, human mask).

It renders offscreen with habitat-sim and displays the frames with OpenCV, so
it doesn't need habitat-sim's native GUI viewer.

Setup (once, in the folder that holds data/):
  pip install opencv-python
  python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes habitat_humanoids --data-path data/
  # optional robots:
  python -m habitat_sim.utils.datasets_download --uids hab_spot_arm hab_fetch --data-path data/

Run:
  python interactive_viewer.py                      # van Gogh room + humanoid
  python interactive_viewer.py --robot spot         # + Spot robot
  python interactive_viewer.py --scene path\\to\\scene.glb --humanoid male_1
  # HSSD (Habitat 3.0 houses):
  python interactive_viewer.py --dataset data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json --list-scenes
  python interactive_viewer.py --dataset data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json --scene 102344049

Controls (HITL-style; click the window once so it gets keyboard focus):
  Mouse      Right-click floor: humanoid walks there   Middle-click / Shift+Right-click: robot goes there
             Left-drag: rotate camera   Wheel: zoom
  Humanoid   W/S walk forward/back   A/D turn   (also Arrow keys)
  Robot      I/K forward/back   J/L turn
  Camera     C cycle: orbit humanoid / top-down / free / follow robot
             in free mode W/S/A/D/Q/E/R/F move the camera instead of the humanoid
  View       1 RGB  2 depth  3 RGB + human/robot masks  4 all side by side
  Other      T bring humanoid+robot to the view   O drop a box   P snapshot   H help   Esc quit
"""

import argparse
import math
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV is needed for the window: pip install opencv-python")
    sys.exit(1)

import magnum as mn

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from humanoid_controller import HumanoidRearrangeController  # noqa: E402

HUMAN_SEMANTIC_ID = 100
ROBOT_SEMANTIC_ID = 200
TICK_HZ = 30.0

ROBOTS = {
    # urdf, height of base above the floor, rest joint positions {link_id: [values]}
    "spot": (
        "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf",
        0.48,
        {
            **{i: [v] for i, v in enumerate([0.0, -3.14, 0.0, 3.0, 0.0, 0.0, 0.0])},
            7: [-1.56],
            **{8 + i: [v] for i, v in enumerate([0.0, 0.7, -1.5] * 4)},
        },
    ),
    "fetch": (
        "data/robots/hab_fetch/robots/hab_fetch.urdf",
        0.0,
        {15 + i: [v] for i, v in enumerate([-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005])},
    ),
}

# Arrow key codes from cv2.waitKeyEx (Windows, Linux/GTK)
KEY_UP = {2490368, 65362}
KEY_DOWN = {2621440, 65364}
KEY_LEFT = {2424832, 65361}
KEY_RIGHT = {2555904, 65363}

HELP = [
    "Right-click: walk human there   Middle/Shift+Right-click: send robot",
    "Left-drag: rotate camera   Wheel: zoom   C: orbit/top-down/free/robot",
    "W/S/A/D: walk/turn human (free cam: move camera)   I/K/J/L: robot",
    "1 RGB  2 depth  3 masks  4 all   T bring here  O box  P snapshot  H help",
]


# ----------------------------------------------------------------- helpers
def finite(v):
    return v is not None and bool(np.all(np.isfinite(np.array(v))))


def yaw_quat(yaw):
    return quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))


def yaw_from_forward(fx, fz):
    """Agent yaw whose camera (-Z) looks along (fx, fz)."""
    return math.atan2(-fx, -fz)


def set_link_joints(ao, values_by_link):
    pos = list(ao.joint_positions)
    for link_id, vals in values_by_link.items():
        try:
            off = ao.get_link_joint_pos_offset(link_id)
        except Exception:  # noqa: BLE001
            continue
        for k, v in enumerate(vals):
            if 0 <= off + k < len(pos):
                pos[off + k] = v
    ao.joint_positions = pos


def add_ao_with_semantic_id(sim, urdf, semantic_id, fixed_base=True):
    """Add a URDF articulated object whose meshes (incl. a skinned humanoid
    mesh) render with the given semantic id. The id has to be on the template
    before the object is created."""
    aom = sim.get_articulated_object_manager()
    aotm = sim.metadata_mediator.ao_template_manager
    probe = aom.add_articulated_object_from_urdf(
        urdf, fixed_base=fixed_base, maintain_link_order=True
    )
    if probe is None:
        raise RuntimeError(f"failed to load {urdf}")
    attrs = probe.creation_attributes
    aom.remove_object_by_handle(probe.handle)
    attrs.semantic_id = semantic_id
    for name, value in (("link_order", "URDF_ORDER"), ("base_type", "FIXED" if fixed_base else "FREE")):
        try:
            setattr(attrs, name, value)
        except Exception:  # noqa: BLE001
            pass
    handle = f"{os.path.basename(urdf)}_viewer_{semantic_id}"
    aotm.register_template(attrs, handle)
    ao = aom.add_articulated_object_by_template_handle(handle)
    if ao is None:
        raise RuntimeError(f"failed to instance {handle}")
    ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    for node in ao.visual_scene_nodes:
        node.semantic_id = semantic_id
    return ao


class Humanoid:
    """SMPL-X humanoid driven by habitat-lab's walking controller."""

    # Same fixed rotation habitat-lab's KinematicHumanoid applies
    OFFSET_BASE = mn.Matrix4.rotation(mn.Rad(-math.pi / 2), mn.Vector3(0, 0, 1)) @ mn.Matrix4.rotation(
        mn.Rad(-math.pi / 2), mn.Vector3(0, 1, 0)
    )
    HIP = mn.Vector3(0, 0.9, 0)

    def __init__(self, sim, urdf, motion_pkl, feet_pos):
        self.sim = sim
        self.ao = add_ao_with_semantic_id(sim, urdf, HUMAN_SEMANTIC_ID)
        self.ctrl = HumanoidRearrangeController(motion_pkl)
        self.ctrl.set_framerate_for_linspeed(lin_speed=1.1, ang_speed=2.5, ctrl_freq=TICK_HZ)
        self.heading = mn.Vector3(1, 0, 0)
        self.place(feet_pos)

    def place(self, feet_pos, heading=None):
        if heading is not None:
            self.heading = heading.normalized()
        base = mn.Matrix4.look_at(
            mn.Vector3(feet_pos) + self.HIP,
            mn.Vector3(feet_pos) + self.HIP + self.heading,
            mn.Vector3(0, 1, 0),
        )
        self.ctrl.reset(mn.Matrix4.translation(mn.Vector3(feet_pos) + self.HIP))
        self.ctrl.obj_transform_base = mn.Matrix4.translation(base.translation)
        for _ in range(20):  # turns are rate-limited; converge to the heading
            self.ctrl.calculate_turn_pose(self.heading)
        self.ctrl.calculate_stop_pose()
        self.apply()

    @property
    def feet(self):
        return self.ctrl.obj_transform_base.translation - self.HIP

    @property
    def forward(self):
        # the controller tracks the walking direction in world space
        f = self.ctrl.prev_orientation
        if f is None:
            return self.heading
        f = mn.Vector3(f[0], 0, f[2])
        return f.normalized() if f.length() > 1e-6 else self.heading

    path = None  # remaining waypoints when walking to a clicked goal

    def walk_to(self, points):
        self.path = [mn.Vector3(p) for p in points[1:]] or None

    def follow_path(self):
        """Walk along self.path; returns False when there's nothing to follow."""
        while self.path:
            d = self.path[0] - self.feet
            d = mn.Vector3(d.x, 0, d.z)
            if d.length() > 0.25:
                self.heading = d.normalized()
                self.step(walk=True, turn=0, target=d)
                return True
            self.path.pop(0)
        self.path = None
        return False

    def step(self, walk, turn, target=None):
        if turn:
            self.heading = mn.Quaternion.rotation(mn.Deg(12.0 * turn), mn.Vector3(0, 1, 0)).transform_vector(
                self.heading
            )
        if walk:
            before = self.feet
            self.ctrl.calculate_walk_pose(target if target is not None else self.heading)
            after = self.feet
            # keep the humanoid on walkable floor
            pf = self.sim.pathfinder
            if pf.is_loaded:
                ok = pf.try_step(np.array(before), np.array(after))
                ok = mn.Vector3(ok) if finite(ok) else before
                self.ctrl.obj_transform_base.translation = ok + self.HIP
        elif turn:
            self.ctrl.calculate_turn_pose(self.heading)
        else:
            self.ctrl.calculate_stop_pose()
        self.apply()

    def apply(self):
        self.ao.joint_positions = self.ctrl.joint_pose
        self.ao.transformation = (
            self.ctrl.obj_transform_base @ self.OFFSET_BASE.inverted()
        ) @ self.ctrl.obj_transform_offset


class Robot:
    def __init__(self, sim, kind, floor_pos):
        urdf, self.height, rest = ROBOTS[kind]
        if not os.path.exists(urdf):
            raise FileNotFoundError(
                f"{urdf} not found. Download it with: python -m habitat_sim.utils.datasets_download "
                f"--uids hab_{'spot_arm' if kind == 'spot' else 'fetch'} --data-path data/"
            )
        self.sim = sim
        self.ao = add_ao_with_semantic_id(sim, urdf, ROBOT_SEMANTIC_ID)
        set_link_joints(self.ao, rest)
        self.yaw = 0.0
        self.pos = mn.Vector3(floor_pos)
        self.apply()

    @property
    def forward(self):  # robots drive along their local +X
        return mn.Vector3(math.cos(self.yaw), 0, -math.sin(self.yaw))

    def step(self, move, turn):
        if turn:
            self.yaw += math.radians(6.0) * turn
        if move:
            target = self.pos + self.forward * (0.06 * move)
            pf = self.sim.pathfinder
            new = pf.try_step(np.array(self.pos), np.array(target)) if pf.is_loaded else target
            if finite(new):
                self.pos = mn.Vector3(new)
        self.apply()

    path = None

    def go_to(self, points):
        self.path = [mn.Vector3(p) for p in points[1:]] or None

    def follow_path(self):
        while self.path:
            d = self.path[0] - self.pos
            d = mn.Vector3(d.x, 0, d.z)
            if d.length() > 0.15:
                want = math.atan2(-d.z, d.x)  # yaw whose +X points along d
                diff = (want - self.yaw + math.pi) % (2 * math.pi) - math.pi
                step_turn = max(-math.radians(8), min(math.radians(8), diff))
                self.yaw += step_turn
                if abs(diff) < math.radians(25):
                    target = self.pos + self.forward * min(0.06, d.length())
                    pf = self.sim.pathfinder
                    new = pf.try_step(np.array(self.pos), np.array(target)) if pf.is_loaded else target
                    if finite(new):
                        self.pos = mn.Vector3(new)
                self.apply()
                return True
            self.path.pop(0)
        self.path = None
        return False

    def apply(self):
        self.ao.rotation = mn.Quaternion.rotation(mn.Rad(self.yaw), mn.Vector3(0, 1, 0))
        self.ao.translation = self.pos + mn.Vector3(0, self.height, 0)


# ----------------------------------------------------------------- simulator
def make_sim(scene, width, height, dataset=None):
    backend = habitat_sim.SimulatorConfiguration()
    if dataset:
        backend.scene_dataset_config_file = dataset
    backend.scene_id = scene
    backend.enable_physics = True
    sensors = []
    for uuid, stype in (
        ("color", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
        ("semantic", habitat_sim.SensorType.SEMANTIC),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = stype
        spec.resolution = [height, width]
        spec.position = [0.0, 0.0, 0.0]
        spec.hfov = HFOV_DEG
        sensors.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


def set_sensor_pitch(agent, pitch):
    q = mn.Quaternion.rotation(mn.Rad(pitch), mn.Vector3(1, 0, 0))
    for sensor in agent.scene_node.subtree_sensors.values():
        sensor.node.rotation = q


HFOV_DEG = 90.0


def camera_rotation(yaw, pitch):
    return mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0, 1, 0)) * mn.Quaternion.rotation(
        mn.Rad(pitch), mn.Vector3(1, 0, 0)
    )


def pixel_ray(u, v, w, h, cam_pos, yaw, pitch):
    """World-space ray through pixel (u, v) of the pinhole camera used here."""
    t = math.tan(math.radians(HFOV_DEG) / 2)
    x = (2 * (u + 0.5) / w - 1) * t
    y = (1 - 2 * (v + 0.5) / h) * t * h / w
    d = camera_rotation(yaw, pitch).transform_vector(mn.Vector3(x, y, -1)).normalized()
    return mn.Vector3(cam_pos), d


def project(p, w, h, cam_pos, yaw, pitch):
    """World point -> pixel (or None if behind the camera)."""
    q = camera_rotation(yaw, pitch).inverted()
    c = q.transform_vector(mn.Vector3(p) - mn.Vector3(cam_pos))
    if c.z > -0.05:
        return None
    t = math.tan(math.radians(HFOV_DEG) / 2)
    u = ((c.x / -c.z) / t + 1) / 2 * w
    v = (1 - (c.y / -c.z) / (t * h / w)) / 2 * h
    return int(u), int(v)


def colorize_depth(depth, max_m=None):
    """Near = warm, far = cool. Scales to the visible depth range unless max_m is given."""
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0)
    if max_m is None:
        valid = d[d > 0]
        max_m = float(np.percentile(valid, 98)) if valid.size else 1.0
    d = np.clip(d / max(max_m, 0.1), 0, 1)
    img = cv2.applyColorMap((255 - d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth <= 0] = 0
    return img


def compose(obs, mode):
    rgb = cv2.cvtColor(obs["color"][..., :3], cv2.COLOR_RGB2BGR)
    depth = colorize_depth(obs["depth"])
    sem = obs["semantic"]
    overlay = rgb.copy()
    overlay[sem == HUMAN_SEMANTIC_ID] = (0.4 * overlay[sem == HUMAN_SEMANTIC_ID] + (0, 0.6 * 255, 0)).astype(np.uint8)
    overlay[sem == ROBOT_SEMANTIC_ID] = (0.4 * overlay[sem == ROBOT_SEMANTIC_ID] + (0.6 * 255, 0.3 * 255, 0)).astype(
        np.uint8
    )
    if mode == 1:
        return rgb
    if mode == 2:
        return depth
    if mode == 3:
        return overlay
    h, w = rgb.shape[:2]
    small = (w // 2, h // 2)
    top = np.hstack([cv2.resize(rgb, small), cv2.resize(depth, small)])
    mask = np.zeros_like(rgb)
    mask[sem == HUMAN_SEMANTIC_ID] = (255, 255, 255)
    mask[sem == ROBOT_SEMANTIC_ID] = (255, 160, 60)
    bottom = np.hstack([cv2.resize(overlay, small), cv2.resize(mask, small)])
    return np.vstack([top, bottom])


def draw_text(img, lines, y0=8):
    """White text on a translucent dark panel (top-left)."""
    font, scale, thick, line_h = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1, 22
    width = max(cv2.getTextSize(line, font, scale, thick)[0][0] for line in lines) + 16
    height = line_h * len(lines) + 10
    h, w = img.shape[:2]
    x1, y1 = min(width, w), min(y0 + height, h)
    img[y0:y1, 0:x1] = (img[y0:y1, 0:x1] * 0.35).astype(img.dtype)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (8, y0 + 20 + line_h * i), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


# ----------------------------------------------------------------- main
def main():
    here = os.getcwd()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=os.path.join(here, "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"))
    ap.add_argument("--dataset", default=None,
                    help="scene_dataset_config.json (e.g. HSSD); then --scene is a scene name from --list-scenes")
    ap.add_argument("--list-scenes", action="store_true", help="print the scenes in --dataset and exit")
    ap.add_argument("--humanoid", default="female_0", help="avatar folder in data/humanoids/humanoid_data, or 'none'")
    ap.add_argument("--robot", default="none", choices=["none", "spot", "fetch"])
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--out", default="captures", help="folder for P snapshots")
    ap.add_argument("--selftest", type=int, default=0, help="run N scripted steps headless (no window) and exit")
    args = ap.parse_args()

    if args.list_scenes:
        if not args.dataset:
            print("--list-scenes needs --dataset path/to/*.scene_dataset_config.json")
            sys.exit(1)
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_dataset_config_file = args.dataset
        mm = habitat_sim.metadata.MetadataMediator(cfg)
        names = sorted(mm.get_scene_handles())
        for n in names:
            print(os.path.basename(n).replace(".scene_instance.json", ""))
        print(f"{len(names)} scenes")
        sys.exit(0)

    if args.dataset and not os.path.exists(args.scene):
        # resolve a short scene name to its scene instance file, as habitat-lab
        # episodes do (some datasets only resolve their semantics that way)
        import glob

        root = os.path.dirname(os.path.abspath(args.dataset))
        hits = glob.glob(os.path.join(root, "**", f"{args.scene}.scene_instance.json"), recursive=True)
        if hits:
            args.scene = hits[0]
            print(f"scene: {args.scene}")

    if not args.dataset and not os.path.exists(args.scene):
        print(f"Scene not found: {args.scene}\nDownload test scenes with:\n"
              "  python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/")
        sys.exit(1)

    sim = make_sim(args.scene, args.width, args.height, args.dataset)
    agent = sim.get_agent(0)
    pf = sim.pathfinder
    if not pf.is_loaded:
        # scene came without a navmesh: compute one so humans/robots stay on the floor
        print("No navmesh in this scene; computing one ...")
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius = 0.25
        settings.agent_height = 1.8
        try:
            settings.include_static_objects = True
        except Exception:  # noqa: BLE001
            pass
        sim.recompute_navmesh(pf, settings)
        print("navmesh ready" if pf.is_loaded else "navmesh failed; movement is unconstrained")
    rng = np.random.default_rng(3)

    def nav_point():
        if pf.is_loaded:
            pf.seed(int(rng.integers(1 << 30)))
            return mn.Vector3(pf.get_random_navigable_point())
        return mn.Vector3(0, 0, 0)

    # camera starts on the navmesh at eye height
    cam_pos = nav_point() + mn.Vector3(0, 1.5, 0)
    cam_yaw, cam_pitch = 0.0, -0.15

    def cam_forward():
        return mn.Vector3(-math.sin(cam_yaw), 0, -math.cos(cam_yaw))

    def in_front(dist, side=0.0):
        """Walkable floor point ~dist ahead of the camera (and `side` m to the right)."""
        p = cam_pos + cam_forward() * dist + mn.math.cross(cam_forward(), mn.Vector3(0, 1, 0)) * side
        floor = mn.Vector3(p.x, cam_pos.y - 1.5, p.z)
        if not pf.is_loaded:
            return floor
        snapped = pf.snap_point(np.array(floor))
        if finite(snapped):
            return mn.Vector3(snapped)
        near = pf.get_random_navigable_point_near(np.array(floor), 3.0)
        if finite(near):
            return mn.Vector3(near)
        return nav_point()

    humanoid = robot = None
    if args.humanoid.lower() != "none":
        hdir = os.path.join(here, "data/humanoids/humanoid_data", args.humanoid)
        urdf = os.path.join(hdir, f"{args.humanoid}.urdf")
        pkl = os.path.join(hdir, f"{args.humanoid}_motion_data_smplx.pkl")
        if os.path.exists(urdf) and os.path.exists(pkl):
            humanoid = Humanoid(sim, urdf, pkl, in_front(2.5))
            humanoid.place(humanoid.feet, heading=-cam_forward())
        else:
            print(f"Humanoid files not found in {hdir}; download with:\n"
                  "  python -m habitat_sim.utils.datasets_download --uids habitat_humanoids --data-path data/")
    if args.robot != "none":
        robot = Robot(sim, args.robot, in_front(2.0, side=1.0))

    mode, show_help = 1, True
    cam_modes = ["orbit humanoid", "top-down", "free", "follow robot"]
    cam_mode = 0 if humanoid else (1 if robot else 2)
    # orbit / top-down camera parameters (HITL-style), controlled by mouse
    start_yaw = yaw_from_forward(humanoid.forward.x, humanoid.forward.z) if humanoid else cam_yaw
    orbit = {"yaw": start_yaw, "pitch": -0.6, "dist": 4.0}
    walk_until = 0.0
    turn_h = 0
    last_turnaround = 0.0
    boxes = []
    shots = 0
    window = "Habitat-Sim live viewer"
    clicks = []  # (kind, u, v) from the mouse callback: kind "human" | "robot"
    mouse = {"drag": None}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_RBUTTONDOWN:
            clicks.append(("robot" if flags & cv2.EVENT_FLAG_SHIFTKEY else "human", x, y))
        elif event == cv2.EVENT_MBUTTONDOWN:
            clicks.append(("robot", x, y))
        elif event == cv2.EVENT_LBUTTONDOWN:
            mouse["drag"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["drag"] = None
        elif event == cv2.EVENT_MOUSEMOVE and mouse["drag"] and (flags & cv2.EVENT_FLAG_LBUTTON):
            dx, dy = x - mouse["drag"][0], y - mouse["drag"][1]
            mouse["drag"] = (x, y)
            clicks.append(("drag", dx, dy))
        elif event == cv2.EVENT_MOUSEWHEEL:
            clicks.append(("zoom", 1 if cv2.getMouseWheelDelta(flags) > 0 else -1, 0))

    if not args.selftest:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, on_mouse)

    script = []  # selftest: keys, or ("click", kind, fu, fv) with fractions of the view
    if args.selftest:
        script = (["w"] * 25 + ["a"] * 6 + ["w"] * 15 + ["i"] * 20 + ["j"] * 8
                  + ["c"] + [("click", "human", 0.5, 0.62)] + [""] * 80          # top-down, right-click walk
                  + [("click", "robot", 0.42, 0.6)] + [""] * 80                   # middle-click robot
                  + [("drag", 60, -20), ("zoom", -1, 0)] + [""] * 3 + ["p"]
                  + ["c", "c"] + ["o"] + [""] * 10 + ["4", "p", "1"]              # orbit humanoid
                  + ["c"] + [""] * 3 + ["p"]                                       # top-down
                  + ["c", "c", "c"] + [""] * 3 + ["p"])                            # follow robot
        script = script[: args.selftest] if args.selftest < len(script) else script

    fps, t_last, step_i = 0.0, time.time(), 0
    start_h = humanoid.feet if humanoid else None
    start_r = robot.pos if robot else None
    last_h, last_r = start_h, start_r
    travel = {"human": 0.0, "robot": 0.0}  # path length walked / driven (self-test)
    click_travel = {"human": 0.0, "robot": 0.0}  # ... of which while following a clicked path
    view = {"pos": cam_pos, "yaw": cam_yaw, "pitch": cam_pitch}  # camera used for the last frame

    def focus():
        if humanoid:
            return humanoid.feet
        if robot:
            return robot.pos
        return cam_pos - mn.Vector3(0, 1.5, 0)

    def click_to_floor(u, v):
        """Map a click to a navigable floor point using a physics ray cast."""
        w, h = args.width, args.height
        if mode == 4:  # 2x2 tiles: use the click's position within its tile
            u, v = (u % (w // 2)) * 2, (v % (h // 2)) * 2
        origin, d = pixel_ray(u, v, w, h, view["pos"], view["yaw"], view["pitch"])
        hit = None
        try:
            res = sim.cast_ray(habitat_sim.geo.Ray(np.array(origin), np.array(d)), 200.0)
            if res.hits:
                hit = mn.Vector3(res.hits[0].point)
        except Exception:  # noqa: BLE001
            hit = None
        if hit is None and abs(d.y) > 1e-4:  # fall back to the floor plane under the focus
            t = (focus().y - origin.y) / d.y
            if t > 0:
                hit = origin + d * t
        if hit is None or not pf.is_loaded:
            return hit
        snapped = pf.snap_point(np.array(hit))
        return mn.Vector3(snapped) if finite(snapped) else None

    def unblocked(look, want):
        """Pull a follow camera in front of the first wall between it and its target, so it
        never ends up inside or behind a wall (only the static stage counts as a wall)."""
        d = want - look
        dist = d.length()
        if dist < 1e-6:
            return want
        try:
            res = sim.cast_ray(habitat_sim.geo.Ray(np.array(look), np.array(d / dist)), dist)
            for hit in res.hits:
                if hit.object_id == habitat_sim.stage_id and hit.ray_distance < dist:
                    return look + d / dist * max(0.3, hit.ray_distance - 0.25)
        except Exception:  # noqa: BLE001
            pass
        return want

    def plan(start, goal):
        if not pf.is_loaded:
            return [start, goal]
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(start)
        path.requested_end = np.array(goal)
        if pf.find_path(path) and len(path.points) > 0:
            return [mn.Vector3(p) for p in path.points]
        return None

    while True:
        t0 = time.time()
        # ---------------- input
        if args.selftest:
            if step_i >= len(script):
                break
            k = script[step_i]
            if isinstance(k, tuple):
                if k[0] == "click":
                    clicks.append((k[1], int(k[2] * args.width), int(k[3] * args.height)))
                else:
                    clicks.append(k)
                key = -1
            else:
                special = {"up": 2490368, "left": 2424832, "": -1}
                key = special[k] if k in special else ord(k)
        else:
            key = cv2.waitKeyEx(1)
        step_i += 1
        now = time.time()
        if key == 27:
            break
        ch = chr(key).lower() if 0 <= key < 256 else ""
        free_cam = cam_modes[cam_mode] == "free"

        move_r = turn_r = 0
        if free_cam and ch in ("w", "s", "a", "d", "q", "e", "r", "f"):
            if ch == "w":
                cam_pos += cam_forward() * 0.15
            elif ch == "s":
                cam_pos -= cam_forward() * 0.15
            elif ch == "a":
                cam_yaw += math.radians(5)
            elif ch == "d":
                cam_yaw -= math.radians(5)
            elif ch == "q":
                cam_pos.y -= 0.1
            elif ch == "e":
                cam_pos.y += 0.1
            elif ch == "r":
                cam_pitch = min(cam_pitch + math.radians(4), 1.4)
            elif ch == "f":
                cam_pitch = max(cam_pitch - math.radians(4), -1.4)
        elif ch == "w":
            walk_until = now + 0.15
        elif ch == "s":
            # turn around once (key repeat keeps it walking, not flipping)
            if humanoid and now - last_turnaround > 0.5:
                humanoid.heading = -humanoid.heading
            last_turnaround = now
            walk_until = now + 0.15
        elif ch == "a":
            turn_h = 1
        elif ch == "d":
            turn_h = -1
        elif ch == "i":
            move_r = 1
        elif ch == "k":
            move_r = -1
        elif ch == "j":
            turn_r = 1
        elif ch == "l":
            turn_r = -1
        elif ch in ("1", "2", "3", "4"):
            mode = int(ch)
        elif ch == "c":
            for _ in range(len(cam_modes)):
                cam_mode = (cam_mode + 1) % len(cam_modes)
                name = cam_modes[cam_mode]
                if (name != "orbit humanoid" or humanoid) and (name != "follow robot" or robot):
                    break
            if cam_modes[cam_mode] == "free":
                cam_pos, cam_yaw, cam_pitch = mn.Vector3(view["pos"]), view["yaw"], max(view["pitch"], -1.4)
        elif ch == "h":
            show_help = not show_help
        elif ch == "t":
            if humanoid:
                humanoid.path = None
                humanoid.place(in_front(2.5), heading=-cam_forward())
            if robot:
                robot.path = None
                robot.pos = in_front(2.0, side=1.0)
                robot.apply()
        elif ch == "o":
            otm = sim.get_object_template_manager()
            h = otm.get_template_handles("cubeSolid")
            if h:
                box = sim.get_rigid_object_manager().add_object_by_template_handle(h[0])
                fwd = camera_rotation(view["yaw"], 0).transform_vector(mn.Vector3(0, 0, -1))
                drop = focus() + fwd * 1.0 + mn.Vector3(0, 1.5, 0) if not free_cam else cam_pos + cam_forward() * 1.2
                box.translation = drop
                box.motion_type = habitat_sim.physics.MotionType.DYNAMIC
                boxes.append(box)

        if key in KEY_UP:
            walk_until = now + 0.15
        elif key in KEY_DOWN:
            walk_until = 0.0
        if key in KEY_LEFT:
            turn_h = 1
        elif key in KEY_RIGHT:
            turn_h = -1

        # mouse: clicks set goals, drag/wheel move the orbit camera
        while clicks:
            kind, a, b = clicks.pop(0)
            if kind == "drag":
                if free_cam:
                    cam_yaw -= a * 0.005
                    cam_pitch = max(-1.4, min(1.4, cam_pitch - b * 0.005))
                else:
                    orbit["yaw"] -= a * 0.005
                    orbit["pitch"] = max(-1.45, min(-0.05, orbit["pitch"] - b * 0.005))
            elif kind == "zoom":
                orbit["dist"] = max(1.2, min(25.0, orbit["dist"] * (0.85 if a > 0 else 1.18)))
            else:
                goal = click_to_floor(a, b)
                agent_obj = humanoid if kind == "human" else robot
                if goal is not None and agent_obj is not None:
                    start = agent_obj.feet if kind == "human" else agent_obj.pos
                    pts = plan(start, goal)
                    if pts:
                        (agent_obj.walk_to if kind == "human" else agent_obj.go_to)(pts)

        # ---------------- simulate
        humanoid_on_path = bool(humanoid and humanoid.path)
        robot_on_path = bool(robot and robot.path)
        if humanoid:
            manual = (now < walk_until) or (args.selftest and ch in ("w", "s") and not free_cam) or turn_h
            if manual:
                humanoid.path = None
                humanoid.step(walk=(now < walk_until) or (args.selftest and ch in ("w", "s")), turn=turn_h)
            elif not humanoid.follow_path():
                humanoid.step(walk=False, turn=0)
        turn_h = 0
        if robot:
            if move_r or turn_r:
                robot.path = None
                robot.step(move_r, turn_r)
            else:
                robot.follow_path()
        sim.step_physics(1.0 / TICK_HZ)
        if humanoid:
            step_len = (humanoid.feet - last_h).length()
            travel["human"] += step_len
            if humanoid_on_path:
                click_travel["human"] += step_len
            last_h = humanoid.feet
        if robot:
            step_len = (robot.pos - last_r).length()
            travel["robot"] += step_len
            if robot_on_path:
                click_travel["robot"] += step_len
            last_r = robot.pos

        # ---------------- camera
        name = cam_modes[cam_mode]
        tgt = focus()
        if name == "orbit humanoid":
            look = tgt + mn.Vector3(0, 1.0, 0)
            d = camera_rotation(orbit["yaw"], orbit["pitch"]).transform_vector(mn.Vector3(0, 0, -1))
            pos, yaw, pitch = look - d * orbit["dist"], orbit["yaw"], orbit["pitch"]
        elif name == "top-down":
            pos, yaw, pitch = tgt + mn.Vector3(0, max(3.0, orbit["dist"] * 2.5), 0), orbit["yaw"], -math.pi / 2 + 1e-3
        elif name == "follow robot" and robot:
            f = robot.forward
            pos = unblocked(robot.pos + mn.Vector3(0, 0.9, 0), robot.pos - f * 2.2 + mn.Vector3(0, 1.4, 0))
            yaw, pitch = yaw_from_forward(f.x, f.z), -0.3
        else:
            pos, yaw, pitch = cam_pos, cam_yaw, cam_pitch
        if not finite(pos):
            pos, yaw, pitch = cam_pos, cam_yaw, cam_pitch
        st = habitat_sim.AgentState()
        st.position = np.array(pos)
        st.rotation = yaw_quat(yaw)
        agent.set_state(st, reset_sensors=False)
        set_sensor_pitch(agent, pitch)
        view.update(pos=pos, yaw=yaw, pitch=pitch)

        obs = sim.get_sensor_observations()

        # ---------------- display
        frame = compose(obs, mode)
        if mode in (1, 2, 3):  # draw planned paths and goals
            for agent_obj, color in ((humanoid, (60, 220, 60)), (robot, (255, 160, 40))):
                if agent_obj is None or not agent_obj.path:
                    continue
                start = agent_obj.feet if agent_obj is humanoid else agent_obj.pos
                pts = [project(p + mn.Vector3(0, 0.05, 0), args.width, args.height, pos, yaw, pitch)
                       for p in [start] + agent_obj.path]
                pts = [q for q in pts if q is not None]
                for a, b in zip(pts, pts[1:]):
                    cv2.line(frame, a, b, color, 2, cv2.LINE_AA)
                if pts:
                    cv2.circle(frame, pts[-1], 8, color, 2, cv2.LINE_AA)
        dt = time.time() - t_last
        t_last = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
        status = [f"{fps:5.1f} FPS   camera: {name}   view: {['', 'RGB', 'depth', 'masks', 'all'][mode]}"]
        if show_help:
            status += HELP
        draw_text(frame, status)

        if ch == "p":
            os.makedirs(args.out, exist_ok=True)
            base = os.path.join(args.out, f"shot_{shots:03d}")
            cv2.imwrite(base + "_rgb.png", cv2.cvtColor(obs["color"][..., :3], cv2.COLOR_RGB2BGR))
            np.save(base + "_depth.npy", obs["depth"])
            cv2.imwrite(base + "_depth.png", colorize_depth(obs["depth"]))
            cv2.imwrite(base + "_human_mask.png", ((obs["semantic"] == HUMAN_SEMANTIC_ID) * 255).astype(np.uint8))
            cv2.imwrite(base + "_view.png", frame)
            print(f"saved {base}_*")
            shots += 1

        if args.selftest:
            continue
        cv2.imshow(window, frame)
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        # keep a steady tick rate
        rest = (1.0 / TICK_HZ) - (time.time() - t0)
        if rest > 0:
            time.sleep(rest)

    if args.selftest:
        report = []
        if humanoid:
            report.append(f"humanoid walked {travel['human']:.2f} m "
                          f"(click-walk {click_travel['human']:.2f} m)")
        if robot:
            report.append(f"robot drove {travel['robot']:.2f} m (click-drive {click_travel['robot']:.2f} m)")
        report.append(f"{len(boxes)} box(es), {shots} snapshot(s), last FPS {fps:.0f}")
        print("SELFTEST: " + "; ".join(report))
        ok = (humanoid is None or click_travel["human"] > 0.3) and (
            robot is None or click_travel["robot"] > 0.3
        )
        sim.close()
        sys.exit(0 if ok else 1)
    cv2.destroyAllWindows()
    sim.close()


if __name__ == "__main__":
    main()
