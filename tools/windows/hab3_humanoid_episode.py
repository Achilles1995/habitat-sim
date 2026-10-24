#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Habitat-Lab / Habitat 3.0 check on Windows.

Runs the official Habitat 3.0 benchmark task with a walking KinematicHumanoid
(SMPL-X, skinned) driven by the oracle navigation action, in an HSSD scene
from hab3_bench_assets. Saves third-person frames and a short video, then
steps the same task in 2 parallel worker processes through habitat.VectorEnv
(multiprocessing 'spawn' on Windows).

Run from the habitat-lab checkout root, after:
  python -m habitat_sim.utils.datasets_download --uids hab3_bench_assets ycb --data-path data/

  python <habitat-sim>/tools/windows/hab3_humanoid_episode.py --out hab3
"""

import argparse
import os
import random
import sys
import time

import numpy as np

import habitat

CFG = "benchmark/rearrange/hab3_bench/humanoid_oracle.yaml"
OVERRIDES = [
    "habitat.task.task_spec=tidy_house_2obj",
    "habitat.task.pddl_domain_def=fp",
    "habitat.dataset.data_path=data/hab3_bench_assets/episode_datasets/small_small.json.gz",
]


def make_env(overrides=()):
    config = habitat.get_config(CFG, OVERRIDES + list(overrides))
    return habitat.Env(config=config)


def sample_action(env, num_targets):
    """Same sampling as scripts/hab3_bench/hab3_benchmark.py: pick a random
    target entity for the oracle navigation action."""
    names, args = [], {}
    for action_name, action_args in env.action_space.spaces.items():
        names.append(action_name)
        for arg_name, space in action_args.items():
            if "oracle_nav" in arg_name:
                args[arg_name] = np.array([random.randint(0, num_targets)]) + 1
            else:
                args[arg_name] = space.sample()
    return {"action": tuple(names), "action_args": args}


def run_single(out_dir, steps):
    from PIL import Image

    env = make_env()
    obs = env.reset()
    keys = sorted(obs.keys())
    print("observation keys:", keys)
    rgb_key = next((k for k in keys if "third_rgb" in k), None)
    assert rgb_key is not None, f"no third-person RGB sensor in {keys}"
    num_targets = len(env.task.pddl_problem.get_ordered_entities_list())

    humanoid = env.sim.agents_mgr[0].articulated_agent
    start_pos = np.array(humanoid.base_pos)
    frames = [obs[rgb_key][..., :3]]
    t0 = time.time()
    for _ in range(steps):
        if env.episode_over:
            obs = env.reset()
        obs = env.step(sample_action(env, num_targets))
        frames.append(obs[rgb_key][..., :3])
    sps = steps / (time.time() - t0)
    moved = float(np.linalg.norm(np.array(humanoid.base_pos) - start_pos))
    env.close()

    os.makedirs(out_dir, exist_ok=True)
    for i in (0, len(frames) // 2, len(frames) - 1):
        Image.fromarray(np.ascontiguousarray(frames[i])).save(
            os.path.join(out_dir, f"frame_{i:03d}.png")
        )
    try:
        import imageio

        imageio.mimsave(os.path.join(out_dir, "humanoid.mp4"), frames, fps=15)
    except Exception as e:  # noqa: BLE001
        print("video not written:", e)

    nonblank = float((frames[-1].max(axis=-1) > 0).mean())
    return (
        f"type={type(humanoid).__name__} moved {moved:.2f} m over {steps} steps, "
        f"{sps:.1f} steps/s single process, last frame {nonblank:.0%} non-black"
    ), moved > 0.1 and nonblank > 0.05


def _vec_make_env(overrides):
    return make_env(overrides)


def run_vector(n_procs, steps):
    env_args = [((f"habitat.seed={100 + i}",),) for i in range(n_procs)]
    envs = habitat.VectorEnv(make_env_fn=_vec_make_env, env_fn_args=env_args)
    try:
        envs.reset()
        num_targets = 2
        spaces = envs.action_spaces[0].spaces
        t0 = time.time()
        for _ in range(steps):
            actions = []
            for _ in range(n_procs):
                names, args = [], {}
                for action_name, action_args in spaces.items():
                    names.append(action_name)
                    for arg_name, space in action_args.items():
                        if "oracle_nav" in arg_name:
                            args[arg_name] = np.array([random.randint(0, num_targets)]) + 1
                        else:
                            args[arg_name] = space.sample()
                actions.append({"action": tuple(names), "action_args": args})
            envs.step(actions)
        sps = n_procs * steps / (time.time() - t0)
        method = envs._mp_ctx.get_start_method()
    finally:
        envs.close()
    return f"{n_procs} worker processes ({method}): {sps:.1f} total steps/s", True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hab3")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--vec-procs", type=int, default=2)
    args = ap.parse_args()
    random.seed(0)

    results = []
    for name, fn in (
        ("Habitat 3.0 humanoid episode", lambda: run_single(args.out, args.steps)),
        ("VectorEnv multiprocessing", lambda: run_vector(args.vec_procs, 40)),
    ):
        try:
            info, ok = fn()
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            info, ok = repr(e), False
        results.append((name, ok, info))
        print(("[PASS] " if ok else "[FAIL] ") + f"{name}: {info}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        for name, ok, info in results:
            f.write(f"{'PASS' if ok else 'FAIL'}  {name}  {info}\n")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
