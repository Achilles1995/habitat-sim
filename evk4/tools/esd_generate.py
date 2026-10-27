"""
Generate simulated ESD sequences with the UR5e + stereo DAVIS346 + D435 rig.

    # ESD-1 replica set, every ESD-1 attribute combination (5 x 3 x 3 x 2 x 2 + occlusion variants)
    python tools/esd_generate.py --grid esd1 --out sim_esd --split train --backend warp --device cuda:0

    # domain-randomised random objects (for UNSEEN-object segmentation -> test on real ESD-2)
    python tools/esd_generate.py --random 500 --object-set random --randomize-sensor --randomize-scene \
        --out sim_esd --split train_random --backend warp --device cuda:0

    # one sequence, chosen attributes, + preview video
    python tools/esd_generate.py --one --object-set esd1 --n-obj 10 --speed 1 --traj linear --light normal \
        --height 0.82 --occlusion --preview

Conditions follow the paper: O{2,4,6,8,10} objects, S{0.15,0.3,1} m/s, M{linear,rotation,linearrotation},
L{normal,low}, H{0.62,0.82} m, occlusion yes/no.  ESD-2 replicas (--object-set esd2) are for
sim-only checks; keep them out of training data when you test unseen objects on real ESD-2.
"""
import argparse
import os
import sys
import time

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np                                   # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from evk4_mujoco.esd import generate as G, motion  # noqa: E402


def _run(args_tuple):
    p, out, split = args_tuple
    try:
        return G.generate(p, out, split)
    except Exception as ex:                          # keep the batch going
        print(f"FAILED {G.seq_name(p)}: {ex}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--grid", choices=["esd1", "esd2"])
    mode.add_argument("--random", type=int, metavar="N")
    mode.add_argument("--one", action="store_true")
    ap.add_argument("--out", default="sim_esd"); ap.add_argument("--split", default="train")
    ap.add_argument("--object-set", default=None, help="esd1 | esd2 | random | mixed")
    ap.add_argument("--n-obj", type=int, default=10); ap.add_argument("--speed", type=float, default=0.15)
    ap.add_argument("--traj", default="linear", choices=list(motion.TRAJS))
    ap.add_argument("--light", default="normal", choices=["normal", "low"])
    ap.add_argument("--height", type=float, default=0.82); ap.add_argument("--occlusion", action="store_true")
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--repeat", type=int, default=1, help="grid: seeds per combination")
    ap.add_argument("--limit", type=int, default=0, help="grid: only the first N (after --shuffle)")
    ap.add_argument("--shuffle", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize-sensor", action="store_true"); ap.add_argument("--randomize-scene", action="store_true")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "warp"]); ap.add_argument("--device", default=None)
    ap.add_argument("--no-robot", action="store_true", help="mocap rig instead of the UR5e (faster)")
    ap.add_argument("--rgb-period", type=float, default=0.06)
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (CPU backend)")
    ap.add_argument("--preview", action="store_true", help="also write preview.mp4 per sequence")
    ap.add_argument("--sensor-json", default=None, help="DAVIS346 overrides, e.g. from tools/esd_match_rate.py")
    ap.add_argument("--contrast", type=float, default=None, help="ON/OFF contrast threshold override")
    ap.add_argument("--measured", default="", help="objects.json from tools/esd_measure_objects.py")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    over = {}
    if a.sensor_json:
        import json
        over.update(json.load(open(a.sensor_json)))
    if a.contrast:
        over.update(contrast_on=a.contrast, contrast_off=a.contrast)
    common = dict(duration=a.duration, randomize_sensor=a.randomize_sensor, randomize_scene=a.randomize_scene,
                  backend=a.backend, device=a.device, robot=not a.no_robot, rgb_period=a.rgb_period,
                  sensor_overrides=over or None, measured=os.path.abspath(a.measured) if a.measured else "")
    jobs = []
    if a.grid:
        base = G.esd1_grid(a.object_set or "esd1") if a.grid == "esd1" else G.esd2_grid(a.object_set or "esd2")
        for r in range(a.repeat):
            for p in base:
                q = G.SeqParams(**{**p.__dict__, **common, "seed": a.seed + r * 100000 + len(jobs)})
                jobs.append(q)
        if a.shuffle:
            rng.shuffle(jobs)
        if a.limit:
            jobs = jobs[:a.limit]
    elif a.random:
        for i in range(a.random):
            jobs.append(G.SeqParams(object_set=a.object_set or "random", n_obj=int(rng.integers(2, 11)),
                                    speed=float(rng.choice([0.15, 0.3, 1.0])), traj=str(rng.choice(motion.TRAJS)),
                                    light=str(rng.choice(["normal", "normal", "low"])),
                                    height=float(rng.choice([0.62, 0.82])), occlusion=bool(rng.random() < 0.6),
                                    seed=a.seed + i, **common))
    else:
        jobs.append(G.SeqParams(object_set=a.object_set or "esd1", n_obj=a.n_obj, speed=a.speed, traj=a.traj,
                                light=a.light, height=a.height, occlusion=a.occlusion, seed=a.seed, **common))
    print(f"{len(jobs)} sequences -> {os.path.join(a.out, a.split)}")
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            done = pool.map(_run, [(p, a.out, a.split) for p in jobs])
    else:
        done = [_run((p, a.out, a.split)) for p in jobs]
    if a.preview:
        from evk4_mujoco.esd.preview import preview
        for r in done:
            if r:
                preview(r, os.path.join(r, "preview.mp4"))
    print(f"done: {sum(r is not None for r in done)}/{len(jobs)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
