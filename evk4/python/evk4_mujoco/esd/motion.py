"""
ESD camera motions (Fig. 2 of the paper) in x-y-r space - x-y = plane the cameras move in, r = rotation
about the vertical axis through the TCP - at the paper's speeds, executed by the UR5e through
differential inverse kinematics (or by a mocap rig without robot).

    ML  linear  : point-to-point straight moves (moveL-like trapezoidal profiles), r constant
    MR  rotary  : back-and-forth rotation about the vertical TCP axis, x-y constant
    MLR general : both at the same time (curved, rotating paths)

Speeds S015 / S03 / S1 = 0.15 / 0.3 / 1.0 m/s TCP speed; rotary speed = v / rot_radius (default 0.3 m
-> 0.5 / 1.0 / 3.3 rad/s), which keeps camera-plane speeds comparable.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .rig import ESD_CALIB, T_to_posquat

SPEEDS = {"015": 0.15, "03": 0.30, "1": 1.0}
TRAJS = ("linear", "rotation", "linearrotation")


def _trapezoid(dist, vmax, acc):
    """duration and s(t) in [0, dist] for a rest-to-rest move."""
    if dist <= 0:
        return 0.0, lambda t: 0.0
    ta = vmax / acc
    if acc * ta * ta >= dist:                              # triangular profile
        ta = np.sqrt(dist / acc); vmax = acc * ta; tc = 0.0
    else:
        tc = (dist - acc * ta * ta) / vmax
    T = 2 * ta + tc

    def s(t):
        t = min(max(t, 0.0), T)
        if t < ta:
            return 0.5 * acc * t * t
        if t < ta + tc:
            return 0.5 * acc * ta * ta + vmax * (t - ta)
        td = T - t
        return dist - 0.5 * acc * td * td
    return T, s


class Schedule:
    """Piecewise point-to-point schedule of a vector quantity."""

    def __init__(self, points, vmax, acc, dwell=0.05):
        self.segs, t = [], 0.0
        for a, b in zip(points[:-1], points[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            L = float(np.linalg.norm(b - a))
            T, s = _trapezoid(L, vmax, acc)
            self.segs.append((t, t + T, a, b, L, s))
            t += T + dwell
        self.T = t
        self.end = np.asarray(points[-1], float)

    def __call__(self, t):
        for t0, t1, a, b, L, s in self.segs:
            if t < t0:
                return a
            if t <= t1:
                return a + (b - a) * (s(t - t0) / L if L > 0 else 0.0)
        return self.end


class ESDTrajectory:
    """Offsets (dx, dy, dr) of the TCP from its start pose as functions of time."""

    def __init__(self, kind: str, speed: float, duration: float, rng: np.random.Generator,
                 extent=(0.12, 0.09), rot_amp=(0.35, 0.8), rot_radius=0.3, acc=1.5):
        self.kind, self.duration = kind, duration
        n = 64
        if kind in ("linear", "linearrotation"):
            pts = [np.zeros(2)]
            while len(pts) < n:
                p = rng.uniform(-1, 1, 2) * extent
                if np.linalg.norm(p - pts[-1]) > 0.06:
                    pts.append(p)
            self.xy = Schedule(pts, speed, acc)
        else:
            self.xy = lambda t: np.zeros(2)
        if kind in ("rotation", "linearrotation"):
            amp = rng.uniform(*rot_amp)
            rs = [np.zeros(1)] + [np.array([amp * (1 if i % 2 == 0 else -1) * rng.uniform(0.6, 1.0)])
                                  for i in range(n)]
            w = speed / rot_radius
            self.r = Schedule(rs, w, acc / rot_radius)
        else:
            self.r = lambda t: np.zeros(1)

    def __call__(self, t):
        xy = self.xy(t)
        return float(xy[0]), float(xy[1]), float(self.r(t)[0])


def start_tool_pose(height: float, center=(0.0, 0.0), table_z: float = 0.0, cam: str = "d435", yaw: float = 0.0):
    """tool0 pose (pos, 3x3 R) that puts `cam` straight above `center` at `height`, looking down."""
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R_wc = Rz @ np.diag([1.0, -1.0, -1.0])                 # OpenCV cam: x = world x, y = -world y, z down
    T_tc = ESD_CALIB[cam]["T"]
    p, q = T_to_posquat(T_tc)
    R_tc = np.zeros(9); mujoco.mju_quat2Mat(R_tc, q); R_tc = R_tc.reshape(3, 3)
    R_wt = R_wc @ R_tc.T
    p_wc = np.array([center[0], center[1], table_z + height])
    p_wt = p_wc - R_wt @ p
    return p_wt, R_wt


def apply_offset(p0, R0, dx, dy, dr):
    """Translate in x-y and rotate by dr about the vertical axis through the TCP."""
    Rz = np.array([[np.cos(dr), -np.sin(dr), 0], [np.sin(dr), np.cos(dr), 0], [0, 0, 1]])
    return p0 + np.array([dx, dy, 0.0]), Rz @ R0


class UR5eIK:
    """Damped-least-squares IK of the 6 UR5e joints for the 'tool0' site."""

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.d = mujoco.MjData(m)
        self.sid = m.site("tool0").id
        names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                 "wrist_3_joint"]
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
        self.dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in names])
        self.jp = np.zeros((3, m.nv)); self.jr = np.zeros((3, m.nv))

    def solve(self, p, R, q0, iters=60, tol=(1e-5, 1e-4), damping=1e-3):
        m, d = self.m, self.d
        q = np.array(q0, float)
        qt = np.zeros(4); mujoco.mju_mat2Quat(qt, R.ravel())
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            ep = p - d.site_xpos[self.sid]
            qc = np.zeros(4); mujoco.mju_mat2Quat(qc, d.site_xmat[self.sid])
            er = np.zeros(3); mujoco.mju_subQuat(er, qt, qc)
            er = d.site_xmat[self.sid].reshape(3, 3) @ er          # local -> world angular error
            if np.linalg.norm(ep) < tol[0] and np.linalg.norm(er) < tol[1]:
                break
            mujoco.mj_jacSite(m, d, self.jp, self.jr, self.sid)
            J = np.vstack([self.jp[:, self.dadr], self.jr[:, self.dadr]])
            e = np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            n = np.linalg.norm(dq)
            q = q + (dq if n < 0.5 else dq * 0.5 / n)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        ep = np.linalg.norm(p - d.site_xpos[self.sid])
        qc = np.zeros(4); mujoco.mju_mat2Quat(qc, d.site_xmat[self.sid])
        er = np.zeros(3); mujoco.mju_subQuat(er, qt, qc)
        return q, ep, float(np.linalg.norm(er))

    def _arm_in_view(self, q):
        """How far arm links reach into the cameras' view cone (0 = arm stays out of the images)."""
        m, d = self.m, self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)                                 # camera poses
        cams = [m.camera(f"{c}_cam").id for c in ("left", "right", "d435")]
        pen = 0.0
        for bname in ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link"):
            b = m.body(bname).id
            pts = [d.xpos[b]]
            if bname in ("upper_arm_link", "forearm_link"):           # sample along the link
                ch = [c for c in range(m.nbody) if m.body_parentid[c] == b]
                if ch:
                    pts += [d.xpos[b] + f * (d.xpos[ch[0]] - d.xpos[b]) for f in (0.25, 0.5, 0.75, 1.0)]
            for P in pts:
                for c in cams:
                    v = P - d.cam_xpos[c]
                    fwd = -d.cam_xmat[c].reshape(3, 3)[:, 2]
                    depth = v @ fwd
                    if depth > 0.0:
                        lateral = np.linalg.norm(v - depth * fwd)
                        if lateral < depth * np.tan(np.radians(42)) + 0.08:   # in (or near) the view cone
                            pen += depth + 0.1
        return pen

    def initial(self, p, R, n_random=400, seed=0):
        """IK solution that reaches the start pose with the whole arm OUT of the camera views."""
        rng = np.random.default_rng(seed)
        seeds = [[np.pi / 2, -1.2, 1.6, -0.4, 1.57, 0], [np.pi / 2, -1.6, 1.9, -0.3, 1.57, np.pi],
                 [np.pi / 2, -0.9, 1.2, -0.3, -1.57, 0], [np.pi / 2, -1.4, 1.2, 0.2, 1.57, 0]]
        seeds += list(rng.uniform(-np.pi, np.pi, (n_random, 6)))
        best = None
        for s in seeds:
            q, ep, er = self.solve(p, R, s, iters=150)
            if ep > 1e-3 or er > 1e-2:
                continue
            q = (q + np.pi) % (2 * np.pi) - np.pi                     # wrap to [-pi, pi)
            score = self._arm_in_view(q) + 0.05 * float(np.abs(q).sum())
            if best is None or score < best[0]:
                best = (score, q)
        if best is None:
            return np.zeros(6), 1.0, 1.0
        q, ep, er = self.solve(p, R, best[1], iters=50)
        self.view_penalty = best[0]
        return q, ep, er


class RigDriver:
    """Moves the rig along an ESDTrajectory: sets UR5e joints (kinematically, exact) or the mocap rig
    before every mj_step, so the sensors see exactly the commanded camera motion."""

    def __init__(self, m, d, info, height, traj: ESDTrajectory, yaw0: float = 0.0, center=(0.0, 0.0)):
        self.m, self.traj = m, traj
        self.p0, self.R0 = start_tool_pose(height, center=center, yaw=yaw0)
        self.robot = info["robot"]
        self.max_err = 0.0
        if self.robot:
            self.ik = UR5eIK(m)
            self.q, ep, er = self.ik.initial(self.p0, self.R0)
            if ep > 1e-3 or er > 1e-2:
                raise RuntimeError(f"start pose not reachable (pos err {ep * 1e3:.1f} mm, rot err {er:.3f} rad)")
            self.qadr, self.dadr = self.ik.qadr, self.ik.dadr
        self.apply(d, 0.0, init=True)

    def tool_pose(self, t):
        return apply_offset(self.p0, self.R0, *self.traj(t))

    def apply(self, d, t, init=False):
        p, R = self.tool_pose(t)
        if self.robot:
            q, ep, er = self.ik.solve(p, R, self.q, iters=20)
            self.max_err = max(self.max_err, ep)
            dt = self.m.opt.timestep
            d.qvel[self.dadr] = 0.0 if init else (q - self.q) / dt
            d.qpos[self.qadr] = q
            if self.m.nu >= 6:
                d.ctrl[:6] = q
            self.q = q
        else:
            qq = np.zeros(4); mujoco.mju_mat2Quat(qq, R.ravel())
            d.mocap_pos[0], d.mocap_quat[0] = p, qq
        return p, R
