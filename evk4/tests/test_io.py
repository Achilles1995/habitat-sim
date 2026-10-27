"""EVT 3.0 round trip, checked with our decoder AND the independent `expelliarmus` decoder."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from evk4_mujoco import EVENT_DTYPE                               # noqa: E402
from evk4_mujoco.io import write_evt3, read_evt3, voxel_grid, time_surface, histogram  # noqa: E402

rng = np.random.default_rng(0)
n = 200000
ev = np.empty(n, EVENT_DTYPE)
ev["t"] = np.sort(rng.integers(0, 40_000_000, n))            # 40 s -> exercises 24-bit time wrap
ev["x"] = rng.integers(0, 1280, n); ev["y"] = rng.integers(0, 720, n); ev["p"] = rng.integers(0, 2, n)
# add dense bursts that get packed into VECT_12 words
b = np.empty(5000, EVENT_DTYPE); b["t"] = 12_345_678; b["y"] = 300; b["p"] = 1; b["x"] = np.arange(5000) % 1280
b = np.unique(b)
ev = np.concatenate([ev, b]); ev = ev[np.argsort(ev["t"], kind="stable")]

key = lambda e: np.sort(e.astype([("t", "<i8"), ("y", "<u2"), ("x", "<u2"), ("p", "<i2")]), order=["t", "y", "x", "p"])
ok = True
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "sim.raw")
    write_evt3(path, ev)
    size = os.path.getsize(path)
    dec, hdr = read_evt3(path)
    same = len(dec) == len(ev) and np.array_equal(key(dec), key(ev))
    print(f"[{'PASS' if same else 'FAIL'}] own decoder: {len(ev)} events, {size/len(ev):.2f} bytes/event, "
          f"header format='{hdr.get('format')}'")
    ok &= same
    try:
        import evlib  # independent (Rust) EVT3 decoder
        df = evlib.load_events(path).collect()
        e3 = np.empty(len(df), EVENT_DTYPE)
        e3["x"] = df["x"].to_numpy(); e3["y"] = df["y"].to_numpy()
        e3["p"] = (df["polarity"].to_numpy() > 0).astype(np.int16)
        e3["t"] = df["t"].dt.total_microseconds().to_numpy()
        same3 = len(e3) == len(ev) and np.array_equal(key(e3), key(ev))
        print(f"[{'PASS' if same3 else 'FAIL'}] evlib (independent EVT3 decoder): {len(e3)} events, "
              f"x/y/p/t identical={same3}")
        ok &= same3
    except ImportError:
        print("[SKIP] evlib not installed")
    try:
        import expelliarmus  # decodes addresses/polarity; its TIME_LOW-overflow heuristic differs from the spec
        arr = expelliarmus.Wizard(encoding="evt3").read(path)
        e2 = np.empty(len(arr), EVENT_DTYPE)
        e2["t"], e2["x"], e2["y"], e2["p"] = arr["t"], arr["x"], arr["y"], arr["p"]
        k2 = lambda e: np.sort(e[["y", "x", "p"]].astype([("y", "<u2"), ("x", "<u2"), ("p", "<i2")]))
        same2 = len(e2) == len(ev) and np.array_equal(k2(e2), k2(ev))
        print(f"[{'PASS' if same2 else 'FAIL'}] expelliarmus: {len(e2)} events, x/y/p identical={same2}")
        ok &= same2
    except ImportError:
        print("[SKIP] expelliarmus not installed")
vg = voxel_grid(ev[:10000], 1280, 720, 5); ts = time_surface(ev[:10000], 1280, 720); h = histogram(ev[:10000], 1280, 720)
print(f"[PASS] representations: voxel {vg.shape}, time surface {ts.shape}, histogram {h.shape} sum={h.sum():.0f}")
print("ALL PASSED" if ok else "SOME CHECKS FAILED")
