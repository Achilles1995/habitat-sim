#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Windows smoke test for habitat-sim.

Uses only assets tracked in the repo (no downloads), and checks:
  1. Dataset config loading, including glob patterns (io::globDirs).
  2. Offscreen rendering (WGL on Windows): RGB, depth and semantic sensors.
  3. Bullet physics: a primitive cube falls under gravity.
  4. Rendering throughput (frames per second).

Run:  python tools/windows/smoke_test.py --out smoke
Exit code is non-zero if any check fails.
"""

import argparse
import os
import platform
import sys
import time
import traceback

import numpy as np

import habitat_sim

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(REPO, "data", "test_assets", "dataset_tests", "dataset_0")
DATASET_CFG = os.path.join(DATASET_DIR, "test_dataset_0.scene_dataset_config.json")
STAGE_GLB = os.path.join(DATASET_DIR, "stages", "dataset_test_stage.glb")
# Real indoor scene with a navmesh, if downloaded via
#   python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
VAN_GOGH = os.path.join(REPO, "data", "scene_datasets", "habitat-test-scenes", "van-gogh-room.glb")

RESULTS = []


def check(name):
    def deco(fn):
        def run(*a, **kw):
            t0 = time.time()
            try:
                info = fn(*a, **kw)
                RESULTS.append((name, True, info or ""))
                print(f"[PASS] {name} ({time.time() - t0:.1f}s) {info or ''}")
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, False, repr(e)))
                print(f"[FAIL] {name}: {e!r}")
                traceback.print_exc()

        return run

    return deco


def make_sim(scene_id, dataset_cfg=None, res=256):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_id
    if dataset_cfg:
        backend.scene_dataset_config_file = dataset_cfg
    backend.enable_physics = True
    backend.gpu_device_id = 0

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


def place_agent(sim):
    """Stand the agent on the navmesh if there is one, else back off from the
    scene's bounding box so it's in view."""
    state = habitat_sim.AgentState()
    if sim.pathfinder.is_loaded:
        sim.pathfinder.seed(1)
        state.position = sim.pathfinder.get_random_navigable_point()
        state.position[1] += 1.5  # eye height
        where = "navmesh"
    else:
        root = sim.get_active_scene_graph().get_root_node()
        root.compute_cumulative_bb()
        bb = root.cumulative_bb
        center = np.array(bb.center())
        size = float(max(np.array(bb.size()).max(), 1.0))
        state.position = center + np.array([0.0, 0.0, 1.5 * size], dtype=np.float32)
        where = "bbox"
    sim.get_agent(0).set_state(state)
    return where


@check("import and build info")
def test_import():
    feats = []
    for flag in ("built_with_bullet", "cuda_enabled", "audio_enabled"):
        feats.append(f"{flag}={getattr(habitat_sim, flag, '?')}")
    return f"version={habitat_sim.__version__} python={sys.version.split()[0]} {platform.platform()} " + " ".join(feats)


@check("dataset config + glob patterns")
def test_dataset():
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = DATASET_CFG
    mm = habitat_sim.metadata.MetadataMediator(cfg)
    stages = mm.stage_template_manager.get_template_handles()
    objects = mm.object_template_manager.get_template_handles()
    globbed = [h for h in stages if "glbload_test_stage" in h]
    assert len(globbed) >= 3, f"glob found {len(globbed)} of 3 stage glbs: {stages}"
    return f"{len(stages)} stage templates ({len(globbed)} from glob), {len(objects)} object templates"


@check("offscreen render RGB/depth/semantic")
def test_render(out_dir, scene):
    from PIL import Image

    with make_sim(scene) as sim:
        where = place_agent(sim)
        obs = sim.get_sensor_observations()
        rgb, depth, sem = obs["color"], obs["depth"], obs["semantic"]
        finite = depth[np.isfinite(depth)]
        dmax = float(finite.max()) if finite.size else 0.0

        # Save before asserting so failures can be inspected
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(rgb[..., :3]).save(os.path.join(out_dir, "rgb.png"))
        d = depth / max(dmax, 1e-6)
        Image.fromarray((np.clip(d, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(out_dir, "depth.png")
        )
        Image.fromarray((sem % 256).astype(np.uint8)).save(
            os.path.join(out_dir, "semantic.png")
        )

        assert rgb.shape[:2] == (256, 256), rgb.shape
        nonblank = float((rgb[..., :3].max(axis=-1) > 0).mean())
        assert nonblank > 0.01, f"RGB looks empty ({nonblank:.1%} non-black pixels, agent via {where})"
        assert dmax > 0, "depth is empty"
        return f"{os.path.basename(scene)}: {nonblank:.0%} non-black pixels, depth max {dmax:.2f} m (agent via {where})"


@check("bullet physics: falling cube")
def test_physics():
    with make_sim(STAGE_GLB) as sim:
        otm = sim.get_object_template_manager()
        rom = sim.get_rigid_object_manager()
        handles = otm.get_template_handles("cubeSolid")
        assert handles, "no cubeSolid primitive template"
        obj = rom.add_object_by_template_handle(handles[0])
        start = np.array([0.0, 50.0, 0.0])
        obj.translation = start
        sim.step_physics(1.0)
        moved = start[1] - float(obj.translation[1])
        assert moved > 1.0, f"cube only fell {moved:.3f} m in 1 s"
        return f"cube fell {moved:.2f} m in 1 s"


@check("render throughput")
def test_fps(scene):
    with make_sim(scene, res=640) as sim:
        place_agent(sim)
        sim.get_sensor_observations()
        n = 100
        t0 = time.time()
        for _ in range(n):
            sim.get_sensor_observations()
        fps = n / (time.time() - t0)
        return f"{fps:.1f} FPS at 640x640 (RGB+depth+semantic) in {os.path.basename(scene)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="smoke")
    ap.add_argument("--scene", default=None, help="scene for render/FPS tests")
    args = ap.parse_args()
    scene = args.scene or (VAN_GOGH if os.path.exists(VAN_GOGH) else STAGE_GLB)
    print(f"Render scene: {scene}")

    test_import()
    test_dataset()
    test_render(args.out, scene)
    test_physics()
    test_fps(scene)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        for name, ok, info in RESULTS:
            f.write(f"{'PASS' if ok else 'FAIL'}  {name}  {info}\n")
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
