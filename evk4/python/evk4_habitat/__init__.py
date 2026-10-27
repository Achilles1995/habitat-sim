"""
evk4_habitat - the physics-based Prophesee EVK4 event camera of `evk4_mujoco`, in Habitat-Sim.

    import evk4_habitat as evkh
    sim = evkh.make_simulator("data/scene_datasets/habitat-test-scenes/van-gogh-room.glb")
    motion = evkh.SceneMotion(sim)
    human = evkh.Humanoid(sim, motion, *evkh.humanoid_files("data"), feet_pos=...)
    spot = evkh.Robot(sim, motion, "spot", floor_pos=...)
    cams = [evkh.EVK4Habitat(sim, evkh.spot_mount(spot.ao, "front"), motion, name="spot_front"),
            evkh.EVK4Habitat(sim, evkh.wall_mount(eye, target), motion, name="wall")]
    t = 0.0
    while ...:
        human.follow_path(); spot.follow_path(); t += 1 / 30
        for c in cams:
            c.update(t)
            ev = c.labeled_events()     # x, y, p, t[us], geom (instance id), depth [m], flow_x/y [px/s]
"""
from evk4_mujoco import EVENT_DTYPE, LABELED_DTYPE  # noqa: F401

from .camera import EVK4Habitat  # noqa: F401
from .config import HABITAT_DEFAULTS, intrinsics, make_config  # noqa: F401
from .mounts import CV2GL, Mount, link_mount, look_at, pose_cv, spot_mount, wall_mount  # noqa: F401
from .scene import (HUMAN_ID0, OBJECT_ID0, ROBOT_ID0, SceneMotion, add_articulated)  # noqa: F401

_SIM_NAMES = ("Humanoid", "Robot", "humanoid_files", "linear_pbr", "make_simulator", "plan", "set_link_joints",
              "snap", "ROBOTS")


def __getattr__(name):          # simulator / agent helpers are imported on first use
    if name in _SIM_NAMES:
        from . import sim
        return getattr(sim, name)
    raise AttributeError(name)
