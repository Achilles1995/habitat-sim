"""
Procedural full-body actions for the Habitat 3.0 SMPL-X humanoid, layered on top of the motion-
capture walking / reaching controller.

Limb poses are set by *aiming bones*: "make the right upper arm point along d" is turned into the
rotation of the right shoulder joint that achieves it with the smallest change from the current
pose (twist is preserved).  Only the skeleton topology from the URDF and the current link frames
are needed, so the same code works for every Habitat humanoid (female_*, male_*, neutral_*).

Each action is a function of the action's normalised time s in [0, 1] that returns joint aims
(and optionally a root transform); `Actor` blends them in and out smoothly.
"""
from __future__ import annotations

import math

import magnum as mn
import numpy as np

from .scene import _polar, _rot2quat, _urdf_parents, m4, to_mn


def _quat_from_two(a, b):
    """Shortest rotation (x, y, z, w) taking unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -0.999999:
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    q = np.array([c[0], c[1], c[2], 1.0 + d])
    return q / np.linalg.norm(q)


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def _qmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _mat2q(R):
    """(x, y, z, w) of a (nearly) rotation matrix; robust to scale / round-off."""
    Rp, _ = _polar(np.asarray(R, np.float64))
    return _rot2quat(Rp)


def _slerp(a, b, f):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + f * (b - a)
    else:
        th = math.acos(min(1.0, d))
        q = (math.sin((1 - f) * th) * a + math.sin(f * th) * b) / math.sin(th)
    return q / np.linalg.norm(q)


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Skeleton:
    """Named joints of a humanoid articulated object (SMPL-X naming, matched by substring)."""

    NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
             "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
             "left_middle1", "right_middle1"]

    def __init__(self, ao, urdf_path):
        self.ao = ao
        self.link_of_name = {ao.get_link_name(l): l for l in ao.get_link_ids()}
        par = _urdf_parents(urdf_path)
        self.parent = {l: self.link_of_name.get(par.get(n, ""), -1) for n, l in self.link_of_name.items()}
        self.joint = {}
        for key in self.NAMES:
            hits = [n for n in self.link_of_name if n.lower().endswith(key) or n.lower() == key
                    or f"{key}" in n.lower().split("/")[-1]]
            hits = sorted(hits, key=len)
            if hits:
                self.joint[key] = self.link_of_name[hits[0]]

    def world(self, l):
        node = self.ao.root_scene_node if l < 0 else self.ao.get_link_scene_node(l)
        return m4(node.absolute_transformation())

    def origin(self, key):
        return self.world(self.joint[key])[:3, 3]

    def aim(self, q, key, child_key, d_world, weight=1.0):
        """Rotate joint `key` (in the joint vector q, xyzw per joint) so its bone toward `child_key`
        points along d_world; the result is blended with the current rotation by `weight`.
        Writes q to the articulated object (forward kinematics) and returns it."""
        if key not in self.joint or child_key not in self.joint or weight <= 0:
            return q
        l = self.joint[key]
        Tj = self.world(l)
        v = self.world(self.joint[child_key])[:3, 3] - Tj[:3, 3]
        if np.linalg.norm(v) < 1e-6:
            return q
        R = Tj[:3, :3]
        delta = _quat_from_two(v, np.asarray(d_world, np.float64))
        local = _mat2q(R.T @ _qmat(delta) @ R)                # delta expressed in the joint frame
        off = self.ao.get_link_joint_pos_offset(l)
        q0 = np.array(q[off:off + 4], np.float64)
        q1 = _qmul(q0, local)
        q1 /= np.linalg.norm(q1)
        q[off:off + 4] = list(_slerp(q0, q1, weight))
        self.ao.joint_positions = q
        return q


# ----------------------------------------------------------------------------------- actions
class Actor:
    """Plays procedural actions on a `sim.Humanoid`.

    Call `start(action, duration, **params)` then `update(dt)` every control step (after the
    walking controller has produced its pose).  Actions: idle, point, wave, reach, pick, place,
    sit, fall, stand (walk/approach/walk_past are plain controller walking)."""

    BLEND = 0.35        # s to blend an action in / out

    def __init__(self, human):
        self.h = human
        self.sk = Skeleton(human.ao, human.urdf)
        self.action = None
        self.t = 0.0
        self.dur = 1.0
        self.params = {}
        self.carry = None            # rigid object held in the right hand
        self.rest_after = False      # stay down after a fall / sit

    def start(self, action, duration, **params):
        self.action, self.t, self.dur, self.params = action, 0.0, float(duration), params

    @property
    def done(self):
        return self.action is None or self.t >= self.dur

    def _env(self, hold_end=False):
        """Blend weight: ramps in over BLEND s, out over the last BLEND s (unless hold_end)."""
        w = smooth(self.t / self.BLEND)
        if not hold_end:
            w = min(w, smooth((self.dur - self.t) / self.BLEND))
        return w

    def _legs(self, q, a_deg, b_deg, w):
        """Thighs a_deg forward of vertical, shins b_deg leaning back (knees forward), feet kept in
        place by shifting the pelvis.  Returns (q, pelvis shift)."""
        sk, fwd, up = self.sk, self._fwd, self._up
        if "left_hip" not in sk.joint or w <= 0:
            return q, np.zeros(3)
        a, b = math.radians(a_deg * w), math.radians(b_deg * w)
        thigh = math.sin(a) * fwd - math.cos(a) * up
        shin = -math.sin(b) * fwd - math.cos(b) * up
        for sd in ("left", "right"):
            q = sk.aim(q, f"{sd}_hip", f"{sd}_knee", thigh, 1.0)
            q = sk.aim(q, f"{sd}_knee", f"{sd}_ankle", shin, 1.0)
        L1 = float(np.linalg.norm(sk.origin("left_knee") - sk.origin("left_hip")))
        L2 = float(np.linalg.norm(sk.origin("left_ankle") - sk.origin("left_knee")))
        shift = -(L1 * math.sin(a) - L2 * math.sin(b)) * fwd - (L1 * (1 - math.cos(a)) + L2 * (1 - math.cos(b))) * up
        return q, shift

    def update(self, dt):
        """Apply the current action on top of the controller pose (already on the AO)."""
        h, sk = self.h, self.sk
        self.t += dt
        a = self.action
        fwd = np.array(h.forward)
        self._fwd = fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        self._up = up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
        s = min(1.0, self.t / max(1e-6, self.dur))
        # 1) whole-body root motion first (aims below are computed in the final body frame)
        h.root_override = None
        if a == "fall":
            direction = self.params.get("direction", "forward")
            axis = right if direction == "forward" else (-fwd if direction == "left" else fwd)
            x = min(1.0, (self.t / 0.8) ** 2)                 # gravity-like: ~0.8 s to the ground
            ang = -math.radians(84) * x
            feet = np.array(h.feet)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            Rm = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
            T = np.eye(4)
            T[:3, :3] = Rm
            T[:3, 3] = feet - Rm @ feet + up * 0.10 * x
            h.root_override = to_mn(T)
        h.apply_root()
        q = list(h.ao.joint_positions)
        shift = np.zeros(3)
        # 2) limbs
        if a in ("point", "wave", "reach", "pick", "place") and "right_shoulder" in sk.joint:
            w = self._env()
            side = self.params.get("side", "right")
            sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
            sgn = 1.0 if side == "right" else -1.0
            if a == "point":
                tgt = np.asarray(self.params.get("target", sk.origin(sh) + 2 * fwd), np.float64)
                d = tgt - sk.origin(sh)
                d /= np.linalg.norm(d)
                q = sk.aim(q, sh, el, d, w)
                q = sk.aim(q, el, wr, d, w)
            elif a == "wave":
                q = sk.aim(q, sh, el, 0.75 * sgn * right + 0.66 * up, w)
                ang = math.radians(30) * math.sin(2 * math.pi * 2.0 * self.t)
                d_up = 0.35 * sgn * right + 0.94 * up
                fore = math.cos(ang) * d_up + math.sin(ang) * sgn * right
                q = sk.aim(q, el, wr, fore / np.linalg.norm(fore), w)
            else:   # reach / pick / place: the body crouches for low targets, the arm aims at it
                tgt = np.asarray(self.params["target"], np.float64)
                obj = self.params.get("obj")
                if a == "pick" and obj is not None and self.carry is None:
                    # grasp the top of the object: the hand goes to its top face
                    tgt = np.array(obj.translation) + np.array([0, self.params.get("half", 0.11) + 0.02, 0])
                low = max(0.0, min(1.0, (0.9 - (tgt[1] - np.array(h.feet)[1])) / 0.7))  # 0 above 0.9 m, 1 at 0.2 m
                bend = w * low
                if bend > 0:
                    q, shift = self._legs(q, 80.0, 50.0, bend)
                    if "spine1" in sk.joint and "spine2" in sk.joint:
                        trunk = math.cos(math.radians(50 * bend)) * up + math.sin(math.radians(50 * bend)) * fwd
                        q = sk.aim(q, "spine1", "spine2", trunk, 1.0)
                    if np.any(shift):
                        h.root_override = to_mn(np.array([[1, 0, 0, shift[0]], [0, 1, 0, shift[1]],
                                                          [0, 0, 1, shift[2]], [0, 0, 0, 1.0]]))
                        h.apply_root()
                d = tgt - sk.origin(sh)
                d /= np.linalg.norm(d)
                q = sk.aim(q, sh, el, d, w)
                d2 = tgt - sk.origin(el)
                d2 /= np.linalg.norm(d2)
                q = sk.aim(q, el, wr, d2, w)
                if a == "pick" and s > 0.5 and obj is not None and self.carry is None:
                    self._grasp(obj, side, self.params.get("half", 0.11))
                if a == "place" and s > 0.5 and self.carry is not None:
                    self._release(np.array(h.feet)[1])
        elif a == "sit":            # params: up=False sits down (and stays), up=True stands up
            w = self._env(hold_end=True)
            if self.params.get("up", False):
                w = 1.0 - smooth(self.t / max(1e-6, self.dur))
            q, shift = self._legs(q, 90.0, 0.0, w)
            T = np.eye(4)
            T[:3, 3] = shift
            h.root_override = to_mn(T)
            h.apply_root()
        elif a == "fall":
            x = min(1.0, (self.t / 0.8) ** 2)
            brace = -up * 0.6 + fwd * 0.8
            for sd in ("left", "right"):
                q = sk.aim(q, f"{sd}_shoulder", f"{sd}_elbow", brace / np.linalg.norm(brace), min(1.0, 2 * x))
        elif a == "idle":
            if "spine3" in sk.joint and "neck" in sk.joint:
                sway = 0.04 * math.sin(2 * math.pi * 0.3 * self.t)
                q = sk.aim(q, "spine3", "neck", up + sway * right, 0.5)
        if a is not None:
            h.ao.joint_positions = q
        self._carry_update()

    # ------------------------------------------------------------------ holding objects
    def _hand(self, side):
        """Wrist frame and pointing direction of the hand (wrist -> middle finger)."""
        sk = self.sk
        T = sk.world(sk.joint[f"{side}_wrist"])
        tip = f"{side}_middle1"
        if tip in sk.joint:
            d = sk.origin(tip) - T[:3, 3]
        else:
            d = T[:3, 3] - sk.origin(f"{side}_elbow")
        return T, d / (np.linalg.norm(d) + 1e-9)

    def _grasp(self, obj, side, half):
        """Start holding `obj`: from now on it moves rigidly with the hand, held at the fingertips
        (its centre one half-size beyond the fingers); it glides there over 0.2 s instead of jumping."""
        T, d = self._hand(side)
        self.carry, self.carry_side, self.carry_half = obj, side, half
        self.carry_from = np.array(obj.translation)
        self.carry_t0 = self.t
        R_box = _qmat(_mat2q_from_mn(obj.rotation))
        self.carry_Rrel = T[:3, :3].T @ R_box

    def _release(self, floor_y):
        """Put the held object down: upright, resting on the floor under its current position."""
        o = self.carry
        p = np.array(o.translation)
        o.rotation = mn.Quaternion()
        o.translation = mn.Vector3(p[0], floor_y + self.carry_half, p[2])
        self.carry = None

    def _carry_update(self):
        if self.carry is None:
            return
        side = getattr(self, "carry_side", "right")
        if f"{side}_wrist" not in self.sk.joint:
            return
        T, d = self._hand(side)
        p = T[:3, 3] + d * (0.09 + self.carry_half)
        f = smooth((self.t - self.carry_t0) / 0.2)
        p = (1 - f) * self.carry_from + f * p
        R = T[:3, :3] @ self.carry_Rrel
        self.carry.rotation = _mn_quat(_mat2q(R))
        self.carry.translation = mn.Vector3(*p)


def _mat2q_from_mn(q):
    v = q.vector
    return np.array([v[0], v[1], v[2], q.scalar])


def _mn_quat(q):
    return mn.Quaternion(mn.Vector3(q[0], q[1], q[2]), q[3])
