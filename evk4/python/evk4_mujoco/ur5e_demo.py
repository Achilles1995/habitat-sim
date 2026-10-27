"""Initial conditions and a scripted UR5e trajectory for scenes/scene_ur5e_evk4.xml."""
import numpy as np
import mujoco

HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

# (time [s], joint offsets from HOME) — smooth min-jerk interpolation between waypoints
WAYPOINTS = [
    (0.00, [0, 0, 0, 0, 0, 0]),
    (0.60, [0, 0, 0, 0, 0, 0]),            # static camera: only moving objects fire
    (1.35, [0.22, 0, 0, 0.12, 0, 0]),      # pan + tilt: ego-motion over textured table
    (2.10, [-0.16, 0, 0, -0.10, 0.10, 0]),
    (2.90, [-0.16, 0, 0, -0.10, 0.10, 1.6]),  # fast wrist roll (rotational flow)
    (3.60, [0.05, 0.22, -0.30, 0.10, 0, 1.6]),  # dive towards the spinning disk
    (9.99, [0.05, 0.22, -0.30, 0.10, 0, 1.6]),
]


def _minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def joint_targets(t: float) -> np.ndarray:
    for (t0, q0), (t1, q1) in zip(WAYPOINTS[:-1], WAYPOINTS[1:]):
        if t <= t1:
            a = _minjerk((t - t0) / max(1e-9, t1 - t0))
            return HOME + (1 - a) * np.asarray(q0) + a * np.asarray(q1)
    return HOME + np.asarray(WAYPOINTS[-1][1])


def reset(m: mujoco.MjModel, d: mujoco.MjData, disk_hz: float = 2.0, pend_angle: float = 0.55,
          ball_vel=(0.30, -0.05)) -> None:
    """Robot at HOME, disk spinning, pendulum released, ball rolling."""
    mujoco.mj_resetData(m, d)
    d.qpos[:6] = HOME
    d.ctrl[:6] = HOME
    j = m.joint("disk_spin"); d.qvel[m.jnt_dofadr[j.id]] = 2 * np.pi * disk_hz
    d.ctrl[m.actuator("disk_motor").id] = 2 * np.pi * disk_hz
    j = m.joint("pend"); d.qpos[m.jnt_qposadr[j.id]] = pend_angle
    j = m.joint("ball"); a = m.jnt_dofadr[j.id]
    d.qvel[a:a + 2] = ball_vel
    d.qvel[a + 3:a + 5] = [-ball_vel[1] / 0.02, ball_vel[0] / 0.02]   # rolling without slip
    mujoco.mj_forward(m, d)
