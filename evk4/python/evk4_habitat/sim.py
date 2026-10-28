"""
Habitat simulator set-up for physics-based event simulation.

`make_simulator` creates a habitat_sim.Simulator whose PBR shaders output *linear* radiance:
material textures are converted from sRGB to linear reflectance, and tonemapping and the sRGB
output encoding are switched off.  Together with the HDR (float) colour sensors of the EVK4 this
makes the rendered value proportional to the luminance of the surface, which is what a
photoreceptor integrates.  (Ordinary 8-bit RGB previews are produced from the same radiance with
`EVK4Habitat.frame()`.)
"""
from __future__ import annotations

import math
import os

import magnum as mn
import numpy as np

import habitat_sim

from .humanoid_controller import HumanoidRearrangeController
from .scene import SceneMotion, add_articulated

LINEAR_PBR = dict(map_mat_txtr_to_linear=True, map_ibl_txtr_to_linear=True, map_output_to_srgb=False,
                  use_direct_tonemap=False, use_ibl_tonemap=False)


def linear_pbr(mm) -> int:
    """Switch every PBR shader configuration of a MetadataMediator to linear radiance output.
    Must run before the scene is loaded (drawables copy the configuration when created)."""
    pm = mm.pbr_template_manager
    n = 0
    for h in pm.get_template_handles():
        a = pm.get_template_by_handle(h)
        for k, v in LINEAR_PBR.items():
            setattr(a, k, v)
        pm.register_template(a, h)
        n += 1
    return n


def make_simulator(scene: str, dataset: str | None = None, sensors=(), physics=True, navmesh=True,
                   agent_radius=0.25, gpu_device_id=0):
    """habitat_sim.Simulator for `scene` (optionally from a scene dataset config) with linear PBR
    output, semantic rendering enabled and a navmesh (computed if the scene has none).
    `sensors` are optional extra sensor specs for agent 0."""
    backend = habitat_sim.SimulatorConfiguration()
    if dataset:
        backend.scene_dataset_config_file = dataset
    backend.scene_id = scene
    backend.enable_physics = physics
    backend.load_semantic_mesh = True
    backend.gpu_device_id = gpu_device_id
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sensors = list(sensors)
    if not any(sp.sensor_type == habitat_sim.SensorType.COLOR for sp in sensors):
        # habitat_sim only creates a renderer and loads textures when an agent has a colour
        # sensor at start-up; a tiny one keeps that true for the EVK4 sensors added later
        sp = habitat_sim.CameraSensorSpec()
        sp.uuid = "evk4_keepalive_rgb"
        sp.sensor_type = habitat_sim.SensorType.COLOR
        sp.resolution = [16, 16]
        sensors.append(sp)
    agent_cfg.sensor_specifications = sensors
    mm = habitat_sim.metadata.MetadataMediator(backend)
    linear_pbr(mm)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg], metadata_mediator=mm))
    pf = sim.pathfinder
    if navmesh and not pf.is_loaded:
        s = habitat_sim.NavMeshSettings()
        s.set_defaults()
        s.agent_radius = agent_radius
        s.agent_height = 1.8
        try:
            s.include_static_objects = True
        except Exception:  # noqa: BLE001
            pass
        sim.recompute_navmesh(pf, s)
    return sim


def finite(v):
    return v is not None and bool(np.all(np.isfinite(np.array(v))))


def snap(sim, p, radius=3.0, pf=None):
    """Nearest navigable floor point to p (or p itself without a navmesh)."""
    pf = pf or sim.pathfinder
    if not pf.is_loaded:
        return mn.Vector3(p)
    s = pf.snap_point(np.array(p))
    if finite(s):
        return mn.Vector3(s)
    s = pf.get_random_navigable_point_near(np.array(p), radius)
    return mn.Vector3(s) if finite(s) else mn.Vector3(p)


def plan(sim, start, goal, pf=None):
    """Navmesh shortest path start -> goal as a list of points (None if unreachable)."""
    pf = pf or sim.pathfinder
    if not pf.is_loaded:
        return [mn.Vector3(start), mn.Vector3(goal)]
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(start)
    path.requested_end = np.array(goal)
    if pf.find_path(path) and len(path.points) > 0:
        return [mn.Vector3(p) for p in path.points]
    return None


def navmesh_settings(radius, height=1.8, static_objects=True):
    s = habitat_sim.NavMeshSettings()
    s.set_defaults()
    s.agent_radius = radius
    s.agent_height = height
    try:
        s.include_static_objects = static_objects
    except Exception:  # noqa: BLE001
        pass
    return s


def robot_pathfinder(sim, radius):
    """A separate navmesh for a robot body of the given radius (kept in clearance of walls and
    furniture); cached per radius on the simulator.  Falls back to the scene's navmesh."""
    cache = getattr(sim, "_evk4_robot_pf", None)
    if cache is None:
        cache = {}
        try:
            sim._evk4_robot_pf = cache
        except Exception:  # noqa: BLE001
            pass
    key = round(radius, 3)
    if key not in cache:
        pf = habitat_sim.nav.PathFinder()
        try:
            sim.recompute_navmesh(pf, navmesh_settings(radius))
        except Exception:  # noqa: BLE001
            pf = None
        cache[key] = pf if (pf is not None and pf.is_loaded and pf.navigable_area > 0.5) else sim.pathfinder
    return cache[key]


def bake_obstacles(sim, rects, radius=0.25):
    """Rebuild the people's navmesh (sim.pathfinder) with box obstacles, e.g. a parked robot or a
    stool, so that paths go around them.  rects: (centre xyz, (half x, half z), yaw)."""
    import habitat_sim as hs
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    base = otm.get_template_handles("cubeSolid")[0]
    tmp = []
    for k, (c, (hx, hz), yaw) in enumerate(rects):
        t = otm.get_template_by_handle(base)
        t.scale = mn.Vector3(2 * hx, 1.0, 2 * hz)
        h = f"evk4_navobstacle_{k}_{hx:.2f}_{hz:.2f}"
        otm.register_template(t, h)
        o = rom.add_object_by_template_handle(h)
        o.motion_type = hs.physics.MotionType.STATIC
        o.rotation = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0, 1, 0))
        o.translation = mn.Vector3(c[0], c[1] + 0.5, c[2])
        tmp.append(o)
    ok = sim.recompute_navmesh(sim.pathfinder, navmesh_settings(radius))
    for o in tmp:
        rom.remove_object_by_handle(o.handle)
    return ok


# ----------------------------------------------------------------------------- humans
def humanoid_files(data_dir, name="female_2"):
    d = os.path.join(data_dir, "humanoids", "humanoid_data", name)
    return os.path.join(d, f"{name}.urdf"), os.path.join(d, f"{name}_motion_data_smplx.pkl")


class Humanoid:
    """Habitat 3.0 SMPL-X humanoid (skinned) driven by the motion-capture walking controller.

    Extra joint-level poses (reach, point, wave, sit, fall, ...) can be layered on top through
    `pose_override`, a {link name: quaternion (x, y, z, w)} applied after the controller."""

    OFFSET_BASE = mn.Matrix4.rotation(mn.Rad(-math.pi / 2), mn.Vector3(0, 0, 1)) @ mn.Matrix4.rotation(
        mn.Rad(-math.pi / 2), mn.Vector3(0, 1, 0))
    HIP = mn.Vector3(0, 0.9, 0)

    def __init__(self, sim, motion: SceneMotion, urdf, motion_pkl, feet_pos, name="human", instance_id=None,
                 ctrl_hz=30.0, lin_speed=1.1):
        self.sim = sim
        n = sum(1 for e in motion.entities if e.kind == 1)
        iid = instance_id if instance_id is not None else 60000 + n
        self.urdf = urdf
        self.ao = add_articulated(sim, urdf, iid)
        self.instance_id = motion.add_human(self.ao, name=name, urdf_path=urdf, instance_id=iid)
        self.name = name
        self.ctrl = HumanoidRearrangeController(motion_pkl)
        self.ctrl.set_framerate_for_linspeed(lin_speed=lin_speed, ang_speed=2.5, ctrl_freq=ctrl_hz)
        self.heading = mn.Vector3(1, 0, 0)
        self.pose_override = {}
        self.root_override = None          # optional (4,4) extra transform applied to the base (falls)
        self.links = {self.ao.get_link_name(l): l for l in self.ao.get_link_ids()}
        self.step_len = lin_speed / ctrl_hz
        self.guard = None          # optional callable(next feet position) -> bool (False: wait)
        self.blocked = 0           # control ticks spent waiting for the way to clear
        self.place(feet_pos)

    def place(self, feet_pos, heading=None):
        if heading is not None:
            self.heading = mn.Vector3(heading).normalized()
        self.ctrl.reset(mn.Matrix4.translation(mn.Vector3(feet_pos) + self.HIP))
        for _ in range(20):
            self.ctrl.calculate_turn_pose(self.heading)
        self.ctrl.calculate_stop_pose()
        self.apply()

    @property
    def feet(self):
        return self.ctrl.obj_transform_base.translation - self.HIP

    @property
    def forward(self):
        f = self.ctrl.prev_orientation
        if f is None:
            return self.heading
        f = mn.Vector3(f[0], 0, f[2])
        return f.normalized() if f.length() > 1e-6 else self.heading

    path = None

    def walk_to(self, goal):
        pts = plan(self.sim, self.feet, goal)
        self.path = [mn.Vector3(p) for p in pts[1:]] if pts else None
        return self.path is not None

    def follow_path(self):
        while self.path:
            d = self.path[0] - self.feet
            d = mn.Vector3(d.x, 0, d.z)
            if d.length() > 0.25:
                self.heading = d.normalized()
                nxt = self.feet + self.heading * (2.0 * self.step_len)
                if self.guard is not None and not self.guard(nxt):
                    self.blocked += 1          # someone / something is in the way: wait
                    self.turn_to(self.heading)
                    return True
                self.blocked = 0
                self.step(walk=True, target=d)
                return True
            self.path.pop(0)
        self.path = None
        self.step(walk=False)
        return False

    def turn_to(self, direction):
        self.heading = mn.Vector3(direction.x, 0, direction.z).normalized()
        self.ctrl.calculate_turn_pose(self.heading)
        self.apply()

    def step(self, walk, target=None):
        if walk:
            before = self.feet
            self.ctrl.calculate_walk_pose(target if target is not None else self.heading)
            pf = self.sim.pathfinder
            if pf.is_loaded:
                ok = pf.try_step(np.array(before), np.array(self.feet))
                ok = mn.Vector3(ok) if finite(ok) else before
                self.ctrl.obj_transform_base.translation = ok + self.HIP
        else:
            self.ctrl.calculate_stop_pose()
        self.apply()

    def reach(self, target, hand=1):
        """Reaching pose toward a world point (controller IK from the motion-capture reach data)."""
        try:
            self.ctrl.calculate_reach_pose(mn.Vector3(target), index_hand=hand)
        except Exception:  # noqa: BLE001  (motion file without hand data)
            self.ctrl.calculate_stop_pose()
        self.apply()

    def apply(self):
        q = list(self.ctrl.joint_pose)
        for lname, quat in self.pose_override.items():
            l = self.links.get(lname)
            if l is None:
                continue
            off = self.ao.get_link_joint_pos_offset(l)
            q[off:off + 4] = list(quat)
        self.ao.joint_positions = q
        self.apply_root()

    def apply_root(self):
        """Base transform from the controller, with the optional world-space root_override."""
        T = (self.ctrl.obj_transform_base @ self.OFFSET_BASE.inverted()) @ self.ctrl.obj_transform_offset
        if self.root_override is not None:
            T = self.root_override @ T
        self.ao.transformation = T


# ----------------------------------------------------------------------------- robots
ROBOTS = {
    # urdf (relative to data/), base height above the floor, rest joints {link: [values]}
    "spot": ("robots/hab_spot_arm/urdf/hab_spot_arm.urdf", 0.48,
             {**{i: [v] for i, v in enumerate([0.0, -3.14, 0.0, 3.0, 0.0, 0.0, 0.0])}, 7: [-1.56],
              **{8 + i: [v] for i, v in enumerate([0.0, 0.7, -1.5] * 4)}}),
    "fetch": ("robots/hab_fetch/robots/hab_fetch.urdf", 0.0,
              {15 + i: [v] for i, v in enumerate([-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005])}),
}


FOOTPRINT = {"spot": (0.55, 0.27), "fetch": (0.32, 0.32)}    # half length, half width [m]


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


class Robot:
    """Kinematic mobile robot (Spot / Fetch) that drives along navmesh paths; local +X forward."""

    def __init__(self, sim, motion: SceneMotion, kind, floor_pos, data_dir="data", name=None, yaw=0.0):
        urdf, self.height, rest = ROBOTS[kind]
        urdf = os.path.join(data_dir, urdf)
        if not os.path.exists(urdf):
            raise FileNotFoundError(f"{urdf} not found (python -m habitat_sim.utils.datasets_download "
                                    f"--uids hab_{'spot_arm' if kind == 'spot' else 'fetch'} --data-path {data_dir})")
        self.sim = sim
        self.kind = kind
        self.name = name or kind
        self.ao = add_articulated(sim, urdf, 61000)
        self.ids = motion.add_robot(self.ao, name=self.name, urdf_path=urdf)
        self.rest = rest
        set_link_joints(self.ao, rest)
        # body footprint (half length, half width) and a navmesh that keeps it off walls/furniture
        self.half = FOOTPRINT[kind]
        self.pf = robot_pathfinder(sim, self.half[1] + 0.22)
        self.yaw = yaw
        self.pos = snap(sim, floor_pos, pf=self.pf)
        self.speed = 0.0
        self.guard = None          # optional callable(pos, yaw) -> bool: False = pose not allowed
        self.blocked = 0
        self.apply()

    @property
    def forward(self):
        return mn.Vector3(math.cos(self.yaw), 0, -math.sin(self.yaw))

    def rect_dist(self, p, pos=None, yaw=None):
        """Planar distance from point p to the robot's body footprint (0 inside)."""
        pos = self.pos if pos is None else pos
        yaw = self.yaw if yaw is None else yaw
        dx, dz = p[0] - pos[0], p[2] - pos[2]
        c, s_ = math.cos(yaw), math.sin(yaw)
        lx = dx * c - dz * s_            # along forward = (cos, 0, -sin)
        lz = dx * s_ + dz * c            # along the right side
        ex = max(0.0, abs(lx) - self.half[0])
        ez = max(0.0, abs(lz) - self.half[1])
        return math.hypot(ex, ez)

    def _pose_ok(self, pos, yaw):
        """Body clear of walls (nose and tail on the robot navmesh) and of the guard's objects."""
        f = mn.Vector3(math.cos(yaw), 0, -math.sin(yaw))
        if self.pf.is_loaded:
            for e in (0.0, 1.0, -1.0):
                q = pos + f * (e * max(0.0, self.half[0] - self.half[1] - 0.05))
                if not self.pf.is_navigable(np.array(q), 0.5):
                    return False
        return self.guard is None or self.guard(pos, yaw)

    path = None

    def go_to(self, goal):
        pts = plan(self.sim, self.pos, snap(self.sim, goal, pf=self.pf), pf=self.pf)
        self.path = [mn.Vector3(p) for p in pts[1:]] if pts else None
        return self.path is not None

    def follow_path(self, max_speed=0.8, dt=1 / 30.0, turn_rate=math.radians(120)):
        """One control step along the path; returns False when the path is finished.  The robot
        never moves into a pose that collides with walls or with the guard's objects (people)."""
        while self.path:
            d = self.path[0] - self.pos
            d = mn.Vector3(d.x, 0, d.z)
            if d.length() > 0.12:
                want = math.atan2(-d.z, d.x)
                diff = (want - self.yaw + math.pi) % (2 * math.pi) - math.pi
                yaw = self.yaw + max(-turn_rate * dt, min(turn_rate * dt, diff))
                if self._pose_ok(self.pos, yaw):
                    self.yaw = yaw
                self.speed = 0.0
                if abs(diff) < math.radians(25) and max_speed > 0:
                    v = min(max_speed, d.length() / dt)
                    if self.drive(v * dt):
                        self.speed = v
                        self.blocked = 0
                    else:
                        self.blocked += 1
                self.apply()
                return True
            self.path.pop(0)
        self.path = None
        self.speed = 0.0
        return False

    def drive(self, dist):
        """Move forward by dist if the new pose is free; returns whether it moved."""
        target = self.pos + self.forward * dist
        pf = self.pf
        new = pf.try_step(np.array(self.pos), np.array(target)) if pf.is_loaded else target
        if not finite(new):
            return False
        new = mn.Vector3(new)
        if (new - self.pos).length() < 1e-4 or not self._pose_ok(new, self.yaw):
            return False
        self.pos = new
        return True

    def apply(self):
        self.ao.rotation = mn.Quaternion.rotation(mn.Rad(self.yaw), mn.Vector3(0, 1, 0))
        self.ao.translation = self.pos + mn.Vector3(0, self.height, 0)
