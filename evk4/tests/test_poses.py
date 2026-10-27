"""
Contact sheet of the procedural humanoid actions (for visual checks), rendered with an ordinary
RGB camera from two viewpoints (front-side and side) at the key moment of every action.

    python tests/test_poses.py --data data --out poses
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import magnum as mn  # noqa: E402

import habitat_sim  # noqa: E402
import evk4_habitat as evkh  # noqa: E402
from evk4_habitat.mounts import look_at  # noqa: E402
from evk4_habitat.poses import Actor  # noqa: E402
from evk4_habitat.scene import to_mn  # noqa: E402

TICK = 1 / 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="poses")
    ap.add_argument("--humanoid", default="female_2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import cv2

    sim = evkh.make_simulator(os.path.join(args.data, "scene_datasets/habitat-test-scenes/van-gogh-room.glb"))
    pf = sim.pathfinder
    lo, hi = (np.array(v) for v in pf.get_bounds())
    c = evkh.snap(sim, (lo + hi) / 2)
    motion = evkh.SceneMotion(sim)
    stool = None
    human = evkh.Humanoid(sim, motion, *evkh.humanoid_files(args.data, args.humanoid), feet_pos=c)
    human.place(c, heading=mn.Vector3(1, 0, 0))
    actor = Actor(human)
    print("skeleton joints found:", sorted(actor.sk.joint))
    # props
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    base = otm.get_template_handles("cubeSolid")[0]

    def box(pos, size, name):
        t = otm.get_template_by_handle(base)
        t.scale = mn.Vector3(*size)
        otm.register_template(t, name)
        o = rom.add_object_by_template_handle(name)
        o.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        o.translation = mn.Vector3(pos[0], pos[1] + size[1] / 2, pos[2])
        return o

    cube = box(np.array(c) + np.array([0.5, 0, 0.0]), (0.2, 0.2, 0.2), "pose_box")
    stool = box(np.array(c) + np.array([-0.45, 0, 0.0]), (0.42, 0.45, 0.42), "pose_stool")
    # two cameras
    cams = []
    for k, (off, h) in enumerate((((2.4, 0, 1.6), 1.3), ((0.0, 0, 2.8), 1.0))):
        node = sim.get_active_scene_graph().get_root_node().create_child()
        eye = np.array(c) + np.array(off) + np.array([0, h, 0])
        node.transformation = to_mn(look_at(eye, np.array(c) + np.array([0, 0.75, 0])))
        sp = habitat_sim.CameraSensorSpec()
        sp.uuid = f"pose_cam{k}"
        sp.sensor_type = habitat_sim.SensorType.COLOR
        sp.resolution = [360, 360]
        sp.hfov = 70
        cams.append(sim.create_sensor(sp, node))

    def shot(label):
        imgs = []
        for s in cams:
            s.draw_observation()
            im = np.array(s.get_observation())[..., :3].astype(np.float32) / 255.0
            im = np.clip(im ** (1 / 2.2) * 1.3, 0, 1)         # linear PBR output -> display
            imgs.append((im * 255).astype(np.uint8))
        im = np.concatenate(imgs, 1)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.putText(im, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return im

    def play(kind, dur, at, **params):
        actor.start(kind, dur, **params)
        out = None
        while not actor.done:
            human.step(walk=False)
            actor.update(TICK)
            if out is None and actor.t >= at:
                out = shot(f"{kind} {params.get('side', '')} {params.get('direction', '')}".strip())
        actor.action = None
        human.step(walk=False)
        actor.update(TICK)
        return out

    sheet = [shot("stand")]
    fwd = np.array([1.0, 0, 0])
    sheet.append(play("point", 2.0, 1.0, side="right", target=np.array(c) + np.array([2.0, 1.2, 1.5])))
    sheet.append(play("wave", 2.0, 1.0, side="right"))
    sheet.append(play("wave", 2.0, 1.15, side="left"))
    sheet.append(play("reach", 2.0, 1.0, side="right", target=np.array(c) + fwd * 0.55 + np.array([0, 1.0, 0])))
    sheet.append(play("pick", 2.0, 1.0, target=np.array(cube.translation), obj=cube))
    sheet.append(play("place", 2.0, 0.9, target=np.array(c) + fwd * 0.5 + np.array([0, 0.11, 0])))
    human.turn_to(mn.Vector3(1, 0, 0))
    actor.start("sit", 1.6)
    while not actor.done:
        human.step(walk=False)
        actor.update(TICK)
    sheet.append(shot("sit (stool behind)"))
    actor.start("sit", 1.4, up=True)
    while not actor.done:
        human.step(walk=False)
        actor.update(TICK)
    actor.action = None
    for d in ("forward", "left"):
        human.place(c, heading=mn.Vector3(1, 0, 0) if d == "forward" else mn.Vector3(0, 0, 1))
        actor.start("fall", 2.0, direction=d)
        while not actor.done:
            human.step(walk=False)
            actor.update(TICK)
        sheet.append(shot(f"fall {d}"))
        actor.action = None
    rows = [np.concatenate(sheet[i:i + 2], 1) for i in range(0, len(sheet) - 1, 2)]
    if len(sheet) % 2:
        rows.append(np.concatenate([sheet[-1], np.zeros_like(sheet[-1])], 1))
    cv2.imwrite(os.path.join(args.out, "poses.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", os.path.join(args.out, "poses.jpg"))


if __name__ == "__main__":
    main()
