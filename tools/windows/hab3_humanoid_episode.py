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
    # the inherited default (data/replica_cad/) is only used for an existence
    # check here; the episodes point at the HSSD scenes in hab3_bench_assets
    "habitat.dataset.scenes_dir=data/hab3_bench_assets",
]


def make_env(overrides=()):
    config = habitat.get_config(CFG, OVERRIDES + list(overrides))
    return habitat.Env(config=config)


def nav_action(env, target):
    """Oracle navigation toward one target entity (kept fixed so the
    humanoid actually walks there instead of turning toward a new random
    target every step)."""
    names, args = [], {}
    for action_name, action_args in env.action_space.spaces.items():
        names.append(action_name)
        for arg_name, space in action_args.items():
            if "oracle_nav" in arg_name:
                args[arg_name] = np.array([target])
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
    target = 1
    moved = 0.0
    prev = start_pos
    t0 = time.time()
    for _ in range(steps):
        if env.episode_over:
            obs = env.reset()
            prev = np.array(humanoid.base_pos)
        if obs.get("has_finished_oracle_nav", [0])[0] > 0.5:
            # reached this target: walk to the next entity
            target = target % num_targets + 1
        obs = env.step(nav_action(env, target))
        pos = np.array(humanoid.base_pos)
        moved += float(np.linalg.norm(pos - prev))
        prev = pos
        frames.append(obs[rgb_key][..., :3])
    sps = steps / (time.time() - t0)
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
        f"type={type(humanoid).__name__} walked {moved:.2f} m over {steps} steps, "
        f"{sps:.1f} steps/s single process, last frame {nonblank:.0%} non-black"
    ), moved > 0.1 and nonblank > 0.05


def _vec_make_env(overrides):
    return make_env(overrides)


def _spawn_probe(conn, overrides):
    """Runs in a 'spawn'ed child: build the env, step it, report back."""
    try:
        import faulthandler

        faulthandler.enable()
        env = make_env(overrides)
        env.reset()
        for _ in range(5):
            env.step(nav_action(env, 1))
        env.close()
        conn.send(("ok", None))
    except BaseException:  # noqa: BLE001
        import traceback

        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def run_spawn_probe():
    """One child process via multiprocessing 'spawn', with its traceback (or
    exit code, for a native crash) reported to the parent."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    p = ctx.Process(target=_spawn_probe, args=(child, ("habitat.seed=101",)))
    p.start()
    child.close()
    msg = None
    try:
        if parent.poll(600):
            msg = parent.recv()
    except EOFError:
        pass
    p.join(60)
    if msg is None:
        return f"child died without reporting, exitcode={p.exitcode}", False
    if msg[0] != "ok":
        print(msg[1])
        return "child raised: " + msg[1].strip().splitlines()[-1], False
    return f"spawned child built and stepped the env (exitcode={p.exitcode})", True


def run_vector(n_procs, steps):
    env_args = [((f"habitat.seed={100 + i}",),) for i in range(n_procs)]
    envs = habitat.VectorEnv(make_env_fn=_vec_make_env, env_fn_args=env_args)
    try:
        envs.reset()
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
                            args[arg_name] = np.array([1])
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
        ("spawned worker process", run_spawn_probe),
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
