#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Habitat 3.0 humanoid check: load an SMPL-X humanoid (articulated object from
URDF + skinned mesh) into the van Gogh test room, render it, and verify it
shows up in the RGB, depth and semantic images.

Needs:
  python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes habitat_humanoids --data-path data/

Run:  python tools/windows/humanoid_test.py --out humanoid
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENE = os.path.join(REPO, "data", "scene_datasets", "habitat-test-scenes", "van-gogh-room.glb")
HUMANOIDS = os.path.join(REPO, "data", "humanoids", "humanoid_data")
HUMANOID_SEMANTIC_ID = 77


def make_sim(scene, res):
    backend = habitat_sim.SimulatorConfiguration()
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
        spec.resolution = [res, res]
        spec.position = [0.0, 0.0, 0.0]
        sensors.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


def best_viewpoint(sim, eye=1.3, tries=24):
    """Pick a navigable point and heading with the most open space ahead."""
    sim.pathfinder.seed(7)
    best = None
    agent = sim.get_agent(0)
    for _ in range(tries):
        floor = np.array(sim.pathfinder.get_random_navigable_point())
        for k in range(8):
            yaw = k * np.pi / 4
            st = habitat_sim.AgentState()
            st.position = floor + np.array([0.0, eye, 0.0])
            st.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
            agent.set_state(st)
            d = sim.get_sensor_observations()["depth"]
            center = d[d.shape[0] // 3 : 2 * d.shape[0] // 3, d.shape[1] // 3 : 2 * d.shape[1] // 3]
            score = float(np.median(center))
            if best is None or score > best[0]:
                best = (score, floor, yaw)
    return best


def save(obs, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(obs["color"][..., :3]).save(os.path.join(out_dir, f"{tag}_rgb.png"))
    d = obs["depth"]
    d = np.clip(d / max(float(np.nanmax(d)), 1e-6), 0, 1)
    Image.fromarray((d * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{tag}_depth.png"))
    sem = (obs["semantic"] == HUMANOID_SEMANTIC_ID).astype(np.uint8) * 255
    Image.fromarray(sem).save(os.path.join(out_dir, f"{tag}_humanoid_mask.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="humanoid")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--avatar", default="female_0")
    args = ap.parse_args()

    urdf = os.path.join(HUMANOIDS, args.avatar, f"{args.avatar}.urdf")
    for p in (SCENE, urdf):
        if not os.path.exists(p):
            print(f"[FAIL] missing data: {p}")
            sys.exit(1)

    with make_sim(SCENE, args.res) as sim:
        score, floor, yaw = best_viewpoint(sim)
        print(f"viewpoint: floor={floor.round(2)} yaw={np.degrees(yaw):.0f} deg, open depth ahead {score:.2f} m")

        agent = sim.get_agent(0)
        st = habitat_sim.AgentState()
        st.position = floor + np.array([0.0, 1.3, 0.0])
        st.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
        agent.set_state(st)
        before = sim.get_sensor_observations()
        save(before, args.out, "before")

        # Load the humanoid as an articulated object (Habitat 3.0 style)
        aom = sim.get_articulated_object_manager()
        t0 = time.time()
        humanoid = aom.add_articulated_object_from_urdf(
            urdf, fixed_base=True, maintain_link_order=True
        )
        assert humanoid is not None, f"failed to load {urdf}"
        load_s = time.time() - t0
        humanoid.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        nodes = list(humanoid.visual_scene_nodes)
        for link_id in [-1] + list(humanoid.get_link_ids()):
            nodes += list(humanoid.get_link_visual_nodes(link_id))
        for node in nodes:
            node.semantic_id = HUMANOID_SEMANTIC_ID

        # Stand it ~2 m in front of the camera, facing the camera
        dist = float(min(2.0, max(score * 0.6, 1.0)))
        forward = np.array([-np.sin(yaw), 0.0, -np.cos(yaw)])
        humanoid.translation = floor + forward * dist + np.array([0.0, 0.9, 0.0])
        humanoid.rotation = quat_to_magnum_yaw(yaw + np.pi)

        after = sim.get_sensor_observations()
        save(after, args.out, "after")

        changed = float(
            (np.abs(after["color"][..., :3].astype(int) - before["color"][..., :3].astype(int)).max(axis=-1) > 20).mean()
        )
        mask = float((after["semantic"] == HUMANOID_SEMANTIC_ID).mean())
        n_joints = len(humanoid.joint_positions)

        # Pose the joints a bit and make sure rendering follows (skinning)
        pose = np.array(humanoid.joint_positions)
        pose[: min(len(pose), 12)] += 0.3
        humanoid.joint_positions = pose
        posed = sim.get_sensor_observations()
        save(posed, args.out, "posed")
        moved = float(
            (np.abs(posed["color"][..., :3].astype(int) - after["color"][..., :3].astype(int)).max(axis=-1) > 20).mean()
        )

        n = 50
        t0 = time.time()
        for _ in range(n):
            sim.get_sensor_observations()
        fps = n / (time.time() - t0)

    report = (
        f"avatar={args.avatar} links={humanoid.num_links if hasattr(humanoid, 'num_links') else '?'} "
        f"joint_dofs={n_joints} load={load_s:.2f}s | pixels changed by humanoid: {changed:.1%}, "
        f"semantic mask: {mask:.1%}, pixels changed by posing: {moved:.1%} | {fps:.1f} FPS at {args.res}px"
    )
    ok = changed > 0.01
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write(("PASS  " if ok else "FAIL  ") + report + "\n")
    print(("[PASS] " if ok else "[FAIL] ") + report)
    sys.exit(0 if ok else 1)


def quat_to_magnum_yaw(yaw):
    import magnum as mn

    return mn.Quaternion.rotation(mn.Rad(float(yaw)), mn.Vector3.y_axis())


if __name__ == "__main__":
    main()
