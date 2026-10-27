"""
Moving things of a Habitat scene, as the event camera needs them.

`SceneMotion` keeps a registry of tracked entities - humans (skinned SMPL-X articulated
objects), robots (articulated objects, one instance id per link) and rigid objects - and can

  * capture / restore their complete kinematic state (base transform + joint positions),
  * set any intermediate state between two captured ones (slerp / lerp), which is how the
    camera renders the continuous motion between two control steps (adaptive sub-steps),
  * report the world transform of every rigid part at the current state, which gives exact
    optical-flow labels for rigid parts and a nearest-bone (rigid skinning) flow for humans.

Instance ids are the semantic ids rendered by Habitat's semantic sensor.  The event labels
(`geom` field of the labelled events) carry these ids; `SceneMotion.instances()` maps them to
names / kinds / links.
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import magnum as mn
import numpy as np

import habitat_sim

# id ranges (kept far above the ids of semantic scene annotations such as HSSD's)
HUMAN_ID0 = 60000
ROBOT_ID0 = 61000          # robot r, link l -> ROBOT_ID0 + 100 r + (l + 1); base = +0
OBJECT_ID0 = 63000
ROBOT_STRIDE = 100

KIND_STAGE, KIND_HUMAN, KIND_ROBOT, KIND_OBJECT = 0, 1, 2, 3


def m4(T) -> np.ndarray:
    """magnum Matrix4 (column-major, T[c] is column c) -> (4, 4) numpy with p_world = M @ p."""
    if isinstance(T, np.ndarray):
        return T
    return np.array([[T[c][r] for c in range(4)] for r in range(4)], dtype=np.float64)


def to_mn(M: np.ndarray) -> mn.Matrix4:
    """(4, 4) numpy -> magnum Matrix4."""
    M = np.asarray(M, dtype=np.float64)
    return mn.Matrix4([mn.Vector4(*M[:, c]) for c in range(4)])


def _slerp_quat(a, b, f):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + f * (b - a)
    else:
        th = math.acos(min(1.0, d))
        s = math.sin(th)
        q = (math.sin((1 - f) * th) * a + math.sin(f * th) * b) / s
    n = np.linalg.norm(q)
    return q / n if n > 0 else a


def _interp_transform(A: mn.Matrix4, B: mn.Matrix4, f: float) -> mn.Matrix4:
    if f <= 0:
        return A
    if f >= 1:
        return B
    qa = mn.Quaternion.from_matrix(A.rotation_normalized())
    qb = mn.Quaternion.from_matrix(B.rotation_normalized())
    q = mn.math.slerp_shortest_path(qa, qb, f)
    t = A.translation * (1 - f) + B.translation * f
    return mn.Matrix4.from_(q.to_matrix(), t)


def _urdf_parents(urdf_path):
    """{child link name: parent link name} from a URDF file."""
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for j in root.findall("joint"):
        p, c = j.find("parent"), j.find("child")
        if p is not None and c is not None:
            out[c.get("link")] = p.get("link")
    return out


@dataclass
class Entity:
    kind: int
    name: str
    obj: object                       # ManagedArticulatedObject / ManagedRigidObject
    ids: dict = field(default_factory=dict)        # link id (-1 = base) -> instance id
    joint_slices: list = field(default_factory=list)   # (offset, n, spherical) per link
    parent: dict = field(default_factory=dict)     # link -> parent link (humans)


class SceneMotion:
    """Registry + state interpolation of the moving parts of a Habitat scene."""

    def __init__(self, sim):
        self.sim = sim
        self.entities: list[Entity] = []
        self._next_robot = 0
        self._next_object = 0

    # ------------------------------------------------------------------ registration
    def add_human(self, ao, name="human", urdf_path=None, instance_id=None):
        """Track a (kinematic, skinned) humanoid articulated object.

        Its skinned mesh must have been created with `instance_id` as semantic id - use
        `add_articulated(sim, urdf, instance_id)` to create it that way."""
        n = sum(e.kind == KIND_HUMAN for e in self.entities)
        iid = int(instance_id if instance_id is not None else HUMAN_ID0 + n)
        e = Entity(KIND_HUMAN, name, ao, ids={-1: iid})
        e.joint_slices = self._joint_slices(ao)
        # skeleton topology for the nearest-bone flow labels
        names = {ao.get_link_name(l): l for l in ao.get_link_ids()}
        par = _urdf_parents(urdf_path) if urdf_path and os.path.exists(urdf_path) else {}
        e.parent = {l: names.get(par.get(nm, ""), -1) for nm, l in names.items()}
        self.entities.append(e)
        return iid

    def add_robot(self, ao, name="robot", urdf_path=None):
        """Track a robot articulated object; every link gets its own instance id
        (ROBOT_ID0 + 100 r + link + 1, base link -> +0) that is written to its visual nodes."""
        r = self._next_robot
        self._next_robot += 1
        e = Entity(KIND_ROBOT, name, ao)
        base = ROBOT_ID0 + ROBOT_STRIDE * r
        for node in ao.visual_scene_nodes:      # everything not claimed by a link -> base id
            node.semantic_id = base
        for l in [-1] + list(ao.get_link_ids()):
            iid = base + (l + 1)
            e.ids[l] = iid
            try:
                for node in ao.get_link_visual_nodes(l):
                    node.semantic_id = iid
            except Exception:  # noqa: BLE001
                pass
        e.joint_slices = self._joint_slices(ao)
        self.entities.append(e)
        return e.ids

    def add_object(self, obj, name="object", instance_id=None):
        """Track a rigid object (ManagedRigidObject); its semantic id is set to the instance id."""
        iid = int(instance_id if instance_id is not None else OBJECT_ID0 + self._next_object)
        self._next_object += 1
        try:
            obj.semantic_id = iid
        except Exception:  # noqa: BLE001
            pass
        for node in getattr(obj, "visual_scene_nodes", []):
            node.semantic_id = iid
        self.entities.append(Entity(KIND_OBJECT, name, obj, ids={-1: iid}))
        return iid

    @staticmethod
    def _joint_slices(ao):
        sl = []
        sph = habitat_sim.physics.JointType.Spherical
        for l in ao.get_link_ids():
            n = ao.get_link_num_joint_pos(l)
            if n > 0:
                sl.append((ao.get_link_joint_pos_offset(l), n, ao.get_link_joint_type(l) == sph))
        return sl

    def instances(self) -> dict:
        """{instance id: dict(kind, name, link)} for every tracked part (id 0 = static scene)."""
        kinds = {KIND_HUMAN: "human", KIND_ROBOT: "robot", KIND_OBJECT: "object"}
        out = {0: dict(kind="stage", name="stage", link=-1)}
        for e in self.entities:
            for l, iid in e.ids.items():
                link = e.obj.get_link_name(l) if (e.kind == KIND_ROBOT and l >= 0) else ""
                out[iid] = dict(kind=kinds[e.kind], name=e.name, link=l, link_name=link)
        return out

    # ------------------------------------------------------------------ state
    def capture(self):
        st = []
        for e in self.entities:
            if e.kind == KIND_OBJECT:
                st.append((mn.Matrix4(e.obj.transformation), None))
            else:
                st.append((mn.Matrix4(e.obj.transformation), np.array(e.obj.joint_positions, np.float64)))
        return st

    def apply(self, state):
        for e, (T, q) in zip(self.entities, state):
            e.obj.transformation = T
            if q is not None and len(q):
                e.obj.joint_positions = q.tolist()

    def apply_interp(self, a, b, f):
        """Set the state a + f (b - a): slerp of base rotations and spherical joints, lerp of
        positions and 1-dof joints."""
        if f >= 1.0:
            return self.apply(b)
        for e, (Ta, qa), (Tb, qb) in zip(self.entities, a, b):
            e.obj.transformation = _interp_transform(Ta, Tb, f)
            if qa is None or not len(qa):
                continue
            q = qa + f * (qb - qa)
            for off, n, sph in e.joint_slices:
                if sph and n == 4:
                    q[off:off + 4] = _slerp_quat(qa[off:off + 4], qb[off:off + 4], f)
            e.obj.joint_positions = q.tolist()

    # ------------------------------------------------------------------ geometry at the current state
    def part_transforms(self):
        """World transforms of the rigid parts at the current state.

        Returns (rigid, humans):
          rigid  : {instance id: (4,4) world_T_part} for robot links and rigid objects
          humans : {instance id: (Ts (n,4,4), seg_a (n,3), seg_b (n,3))} bone frames and the
                   bone segments used to assign surface points to bones."""
        rigid, humans = {}, {}
        for e in self.entities:
            if e.kind == KIND_OBJECT:
                rigid[e.ids[-1]] = m4(e.obj.transformation)
            elif e.kind == KIND_ROBOT:
                for l, iid in e.ids.items():
                    node = e.obj.get_link_scene_node(l) if l >= 0 else e.obj.root_scene_node
                    rigid[iid] = m4(node.absolute_transformation())
            else:
                links = list(e.obj.get_link_ids())
                Ts = np.stack([m4(e.obj.get_link_scene_node(l).absolute_transformation()) for l in links])
                org = Ts[:, :3, 3]
                idx = {l: i for i, l in enumerate(links)}
                children = {l: [] for l in links}
                for l in links:
                    p = e.parent.get(l, -1)
                    if p in children:
                        children[p].append(l)
                seg_b = org.copy()
                for i, l in enumerate(links):
                    ch = children[l]
                    if ch:
                        seg_b[i] = np.mean([org[idx[c]] for c in ch], axis=0)
                    else:
                        p = e.parent.get(l, -1)
                        if p in idx:   # leaf: extend half a bone beyond the joint
                            seg_b[i] = org[i] + 0.5 * (org[i] - org[idx[p]])
                humans[e.ids[-1]] = (Ts, org, seg_b)
        return rigid, humans

    def key_points(self):
        """World points that move with the tracked parts (for the sub-step criterion)."""
        pts = []
        for e in self.entities:
            if e.kind == KIND_OBJECT:
                T = m4(e.obj.transformation)
                try:
                    bb = e.obj.root_scene_node.cumulative_bb
                    lo, hi = np.array(bb.min), np.array(bb.max)
                    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                except Exception:  # noqa: BLE001
                    c = np.zeros((1, 3))
                pts.append(c @ T[:3, :3].T + T[:3, 3])
            else:
                nodes = [e.obj.root_scene_node] + [e.obj.get_link_scene_node(l) for l in e.obj.get_link_ids()]
                pts.append(np.array([np.array(n.absolute_translation) for n in nodes]))
        return np.concatenate(pts) if pts else np.zeros((0, 3))


def add_articulated(sim, urdf, instance_id, fixed_base=True, kinematic=True):
    """Create a URDF articulated object whose meshes (including a skinned humanoid mesh) are
    rendered with `instance_id` by the semantic sensor.  The id has to be on the template before
    the object is instanced, because a skinned mesh takes its id at creation."""
    aom = sim.get_articulated_object_manager()
    aotm = sim.metadata_mediator.ao_template_manager
    probe = aom.add_articulated_object_from_urdf(urdf, fixed_base=fixed_base, maintain_link_order=True)
    if probe is None:
        raise RuntimeError(f"failed to load {urdf}")
    attrs = probe.creation_attributes
    aom.remove_object_by_handle(probe.handle)
    attrs.semantic_id = int(instance_id)
    for name, value in (("link_order", "URDF_ORDER"), ("base_type", "FIXED" if fixed_base else "FREE")):
        try:
            setattr(attrs, name, value)
        except Exception:  # noqa: BLE001
            pass
    handle = f"{os.path.basename(urdf)}_evk4_{instance_id}"
    aotm.register_template(attrs, handle)
    ao = aom.add_articulated_object_by_template_handle(handle)
    if ao is None:
        raise RuntimeError(f"failed to instance {handle}")
    if kinematic:
        ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    for node in ao.visual_scene_nodes:
        node.semantic_id = int(instance_id)
    return ao
