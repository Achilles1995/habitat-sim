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

Keys (click the window first so it has keyboard focus):
  Camera     W/S forward/back   A/D turn   Q/E down/up   R/F look up/down
  Humanoid   Arrow Up walk   Arrow Left/Right turn   (release = stand)
  Robot      I/K forward/back   J/L turn
  View       1 RGB  2 depth  3 human mask overlay  4 all side by side
  Camera     C cycle: free / follow humanoid / follow robot
  Other      T bring humanoid+robot in front of camera   O drop a box
             P save snapshot   H toggle help   Esc quit
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
    "Camera: W/S move  A/D turn  Q/E down/up  R/F look",
    "Human: Arrows walk/turn   Robot: I/K move  J/L turn",
    "1 RGB  2 depth  3 mask  4 all   C camera mode",
    "T bring to camera  O drop box  P snapshot  H help  Esc quit",
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

    def step(self, walk, turn):
        if turn:
            self.heading = mn.Quaternion.rotation(mn.Deg(12.0 * turn), mn.Vector3(0, 1, 0)).transform_vector(
                self.heading
            )
        if walk:
            before = self.feet
            self.ctrl.calculate_walk_pose(self.heading)
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
        spec.hfov = 90
        sensors.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


def set_sensor_pitch(agent, pitch):
    q = mn.Quaternion.rotation(mn.Rad(pitch), mn.Vector3(1, 0, 0))
    for sensor in agent.scene_node.subtree_sensors.values():
        sensor.node.rotation = q


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

    mode, cam_mode, show_help = 1, 0, True
    cam_modes = ["free", "follow humanoid", "follow robot"]
    walk_until = 0.0
    turn_h = 0
    boxes = []
    shots = 0
    window = "Habitat-Sim live viewer"
    if not args.selftest:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    script = []  # selftest key script
    if args.selftest:
        script = (["up"] * 30 + ["left"] * 6 + ["up"] * 20 + ["i"] * 20 + ["j"] * 8 + ["s"] * 6
                  + ["o"] + [""] * 10 + ["4", "p", "1", "p"]          # free camera: all views, RGB
                  + ["c"] + [""] * 3 + ["p"]                           # follow humanoid
                  + ["c"] + [""] * 3 + ["p"])                          # follow robot
        script = script[: args.selftest] if args.selftest < len(script) else script

    fps, t_last, step_i = 0.0, time.time(), 0
    start_h = humanoid.feet if humanoid else None
    start_r = robot.pos if robot else None

    while True:
        t0 = time.time()
        # ---------------- input
        if args.selftest:
            if step_i >= len(script):
                break
            k = script[step_i]
            special = {"up": 2490368, "left": 2424832, "": -1}
            key = special[k] if k in special else ord(k)
        else:
            key = cv2.waitKeyEx(1)
        step_i += 1
        now = time.time()
        if key == 27:
            break
        ch = chr(key).lower() if 0 <= key < 256 else ""

        move_r = turn_r = 0
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
        elif ch == "i":
            move_r = 1
        elif ch == "k":
            move_r = -1
        elif ch == "j":
            turn_r = 1
        elif ch == "l":
            turn_r = -1
        elif ch in "1234" and ch:
            mode = int(ch)
        elif ch == "c":
            cam_mode = (cam_mode + 1) % 3
            if cam_mode == 1 and humanoid is None:
                cam_mode = 2
            if cam_mode == 2 and robot is None:
                cam_mode = 0
        elif ch == "h":
            show_help = not show_help
        elif ch == "t":
            if humanoid:
                humanoid.place(in_front(2.5), heading=-cam_forward())
            if robot:
                robot.pos = in_front(2.0, side=1.0)
                robot.apply()
        elif ch == "o":
            otm = sim.get_object_template_manager()
            h = otm.get_template_handles("cubeSolid")
            if h:
                box = sim.get_rigid_object_manager().add_object_by_template_handle(h[0])
                box.translation = cam_pos + cam_forward() * 1.2 + mn.Vector3(0, 0.3, 0)
                box.motion_type = habitat_sim.physics.MotionType.DYNAMIC
                boxes.append(box)

        if key in KEY_UP:
            walk_until = now + 0.15
        if key in KEY_LEFT:
            turn_h = 1
        elif key in KEY_RIGHT:
            turn_h = -1
        elif key in KEY_DOWN:
            walk_until = 0.0

        # ---------------- simulate
        if humanoid:
            humanoid.step(walk=now < walk_until or (args.selftest and key in KEY_UP), turn=turn_h)
        turn_h = 0
        if robot and (move_r or turn_r):
            robot.step(move_r, turn_r)
        sim.step_physics(1.0 / TICK_HZ)

        # ---------------- camera
        if cam_mode == 1 and humanoid:
            f = humanoid.forward
            eye = humanoid.feet - f * 2.6 + mn.Vector3(0, 1.7, 0)
            pos, yaw, pitch = eye, yaw_from_forward(f.x, f.z), -0.25
        elif cam_mode == 2 and robot:
            f = robot.forward
            pos = robot.pos - f * 2.2 + mn.Vector3(0, 1.4, 0)
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

        obs = sim.get_sensor_observations()

        # ---------------- display
        frame = compose(obs, mode)
        dt = time.time() - t_last
        t_last = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
        status = [f"{fps:5.1f} FPS   camera: {cam_modes[cam_mode]}   view: {['', 'RGB', 'depth', 'mask', 'all'][mode]}"]
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
            report.append(f"humanoid walked {(humanoid.feet - start_h).length():.2f} m")
        if robot:
            report.append(f"robot drove {(robot.pos - start_r).length():.2f} m")
        report.append(f"{len(boxes)} box(es), {shots} snapshot(s), last FPS {fps:.0f}")
        print("SELFTEST: " + "; ".join(report))
        ok = (humanoid is None or (humanoid.feet - start_h).length() > 0.3) and (
            robot is None or (robot.pos - start_r).length() > 0.3
        )
        sim.close()
        sys.exit(0 if ok else 1)
    cv2.destroyAllWindows()
    sim.close()


if __name__ == "__main__":
    main()
