"""
Where an EVK4 sits: on a wall / tripod (static) or on any link of a robot.

A mount is a parent scene node plus the camera pose relative to it.  Poses are given with
`look_at` helpers or in the OpenCV camera convention used by the MuJoCo package (origin at the
optical centre, +z viewing direction, +x image right, +y image down), so hand-eye calibrations
`T_parent_cam` can be used directly.  Internally Habitat cameras look along -z with +y up
(OpenGL convention); `CV2GL` converts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import magnum as mn
import numpy as np

from .scene import m4, to_mn

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])     # OpenCV camera frame <-> Habitat (OpenGL) camera frame


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """(4,4) pose of a Habitat (OpenGL) camera at `eye` looking at `target`."""
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    if abs(np.dot(f, up / np.linalg.norm(up))) > 0.999:      # looking straight up/down
        up = np.array([0.0, 0.0, -1.0]) if abs(f[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, eye
    return T


def pose_cv(pos, R_cv=None, quat_wxyz=None) -> np.ndarray:
    """(4,4) Habitat camera pose from an OpenCV-convention pose (rotation matrix or w,x,y,z quat)."""
    T = np.eye(4)
    if quat_wxyz is not None:
        w, x, y, z = quat_wxyz
        R_cv = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T[:3, :3] = np.eye(3) if R_cv is None else np.asarray(R_cv, np.float64)
    T[:3, 3] = pos
    return T @ CV2GL


@dataclass
class Mount:
    """Camera pose `T_parent_cam` (Habitat/OpenGL camera convention) relative to `parent`.

    parent: None (scene root = world, i.e. a wall/tripod mount), or (articulated object, link id)
            with link -1 = the base."""
    T_parent_cam: np.ndarray
    parent: tuple | None = None
    name: str = "mount"

    def parent_node(self, sim):
        if self.parent is None:
            return sim.get_active_scene_graph().get_root_node()
        ao, link = self.parent
        return ao.root_scene_node if link < 0 else ao.get_link_scene_node(link)

    def create_node(self, sim):
        node = self.parent_node(sim).create_child()
        node.transformation = to_mn(self.T_parent_cam)
        return node


def wall_mount(eye, target, up=(0.0, 1.0, 0.0), name="wall") -> Mount:
    """Static camera at `eye` [m, world] looking at `target`."""
    return Mount(look_at(eye, target, up), None, name)


def link_mount(ao, link: int, pos, look_dir=(1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), pitch_deg=0.0,
               name="link") -> Mount:
    """Camera rigidly attached to link `link` (-1 = base) of an articulated object, at `pos` in the
    link frame, looking along `look_dir` (link frame), tilted down by `pitch_deg`."""
    pos = np.asarray(pos, np.float64)
    d = np.asarray(look_dir, np.float64)
    d /= np.linalg.norm(d)
    up = np.asarray(up, np.float64)
    if pitch_deg:
        axis = np.cross(d, up)
        axis /= np.linalg.norm(axis)
        a = math.radians(-pitch_deg)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        d = (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K) @ d
    return Mount(look_at(pos, pos + d, up), (ao, link), name)


# Spot (data/robots/hab_spot_arm): base frame +x forward, +y up, +z right in Habitat.
SPOT_MOUNTS = {
    # body front, below the arm, slightly tilted down: human-aware navigation / safety
    "front": dict(link=-1, pos=(0.47, 0.02, 0.0), look_dir=(1, 0, 0), pitch_deg=12.0),
    # on top of the body, looking forward over the arm: wider view of a shared workspace
    "top": dict(link=-1, pos=(0.20, 0.22, 0.0), look_dir=(1, 0, 0), pitch_deg=8.0),
    # rear bumper, looking backward
    "rear": dict(link=-1, pos=(-0.47, 0.02, 0.0), look_dir=(-1, 0, 0), pitch_deg=12.0),
}


def spot_mount(ao, where="front", name=None) -> Mount:
    p = SPOT_MOUNTS[where]
    return link_mount(ao, p["link"], p["pos"], p["look_dir"], pitch_deg=p["pitch_deg"], name=name or f"spot_{where}")


def world_pose(node) -> np.ndarray:
    """(4,4) world pose of a scene node."""
    return m4(node.absolute_transformation())
