"""
I/O and event representations.

* EVT 3.0 ``.raw`` writer / reader (the format recorded by the EVK4 and read by Metavision
  Studio / SDK / OpenEB).  Vectorised encoding with VECT_12 / VECT_8 words.
* Metavision ``.bias`` files.
* Event representations for learning: histograms, time surfaces, voxel grids.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from . import EVENT_DTYPE

# EVT 3.0 word types (bits 15..12)
EVT_ADDR_Y, EVT_ADDR_X, VECT_BASE_X, VECT_12, VECT_8 = 0x0, 0x2, 0x3, 0x4, 0x5
EVT_TIME_LOW, CONTINUED_4, EVT_TIME_HIGH, EXT_TRIGGER, OTHERS, CONTINUED_12 = 0x6, 0x7, 0x8, 0xA, 0xE, 0xF


def _header(width, height):
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "% camera_integrator_name Prophesee",
        f"% date {now}",
        "% evt 3.0",
        f"% format EVT3;height={height};width={width}",
        "% generation 4.2",
        f"% geometry {width}x{height}",
        "% integrator_name Prophesee",
        "% plugin_integrator_name Prophesee",
        "% plugin_name hal_plugin_imx636_evk4",
        "% sensor_generation 4.2",
        "% serial_number SIMULATED-MUJOCO",
        "% system_ID 49",
        "% simulator prophesee.sensor.evk4 (MuJoCo)",
        "% end",
    ]
    return ("\n".join(lines) + "\n").encode()


class EVT3Writer:
    """Streaming EVT 3.0 encoder.  Feed time-ordered EventCD arrays with ``write``."""

    def __init__(self, path, width=1280, height=720, vectorize=True):
        self.f = open(path, "wb")
        self.f.write(_header(width, height))
        self.vectorize = vectorize
        self.last_th = -1
        self.last_tl = -1
        self.last_y = -1
        self.last_t = -1
        self.n = 0

    def write(self, ev: np.ndarray):
        if len(ev) == 0:
            return
        t = ev["t"].astype(np.int64)
        if self.last_t >= 0 and t[0] < self.last_t:
            raise ValueError("events must be time-ordered")
        # group runs of (t, y, p) so that nearby x can be packed in vectors
        o = np.lexsort((ev["x"], ev["p"], ev["y"], t))
        x, y, p, t = ev["x"][o].astype(np.int64), ev["y"][o].astype(np.int64), ev["p"][o].astype(np.int64), t[o]
        words = []
        i, n = 0, len(t)
        while i < n:
            ti = int(t[i])
            th, tl = (ti >> 12) & 0xFFF, ti & 0xFFF
            if (ti >> 12) != self.last_th:
                # like the sensor, emit a TIME_HIGH word for every 4096 us tick that elapsed, so
                # decoders can always unwrap the 24-bit timestamp
                start = ti >> 12 if self.last_th < 0 else self.last_th + 1
                for h in range(start, (ti >> 12) + 1):
                    words.append((EVT_TIME_HIGH << 12) | (h & 0xFFF))
                self.last_th = ti >> 12
                self.last_tl = -1
            if tl != self.last_tl:
                words.append((EVT_TIME_LOW << 12) | tl)
                self.last_tl = tl
            yi = int(y[i])
            if yi != self.last_y:
                words.append((EVT_ADDR_Y << 12) | (yi & 0x7FF))
                self.last_y = yi
            # run with identical t, y, p
            j = i + 1
            while j < n and t[j] == ti and y[j] == yi and p[j] == p[i]:
                j += 1
            xs = x[i:j]
            pol = int(p[i]) & 1
            k = 0
            while k < len(xs):
                base = int(xs[k])
                # how many of the next events fall within [base, base+12)
                m = k
                while m < len(xs) and xs[m] < base + 12:
                    m += 1
                if self.vectorize and m - k >= 2:
                    words.append((VECT_BASE_X << 12) | (pol << 11) | (base & 0x7FF))
                    mask = 0
                    for q in xs[k:m]:
                        mask |= 1 << int(q - base)
                    words.append((VECT_12 << 12) | mask)
                    k = m
                else:
                    words.append((EVT_ADDR_X << 12) | (pol << 11) | (base & 0x7FF))
                    k += 1
            i = j
        self.last_t = int(t[-1])
        self.f.write(np.asarray(words, dtype="<u2").tobytes())
        self.n += len(ev)

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def write_evt3(path, events, width=1280, height=720):
    with EVT3Writer(path, width, height) as w:
        w.write(events)


def read_evt3(path) -> tuple[np.ndarray, dict]:
    """Reference EVT 3.0 decoder (handles 24-bit time wrap, vectors). Returns (EventCD, header)."""
    with open(path, "rb") as f:
        data = f.read()
    hdr, pos = {}, 0
    while data[pos:pos + 1] == b"%":
        end = data.index(b"\n", pos)
        line = data[pos + 2:end].decode(errors="ignore")
        if line.strip() == "end":
            pos = end + 1
            break
        k, _, v = line.partition(" ")
        hdr[k] = v
        pos = end + 1
    w = np.frombuffer(data[pos:pos + ((len(data) - pos) // 2) * 2], dtype="<u2")
    out_x, out_y, out_p, out_t = [], [], [], []
    th = tl = 0
    loops = 0
    last_th = 0
    y = 0
    base = 0
    pol = 0
    for word in w.tolist():
        typ = word >> 12
        if typ == EVT_TIME_HIGH:
            v = word & 0xFFF
            if v < last_th:
                loops += 1
            last_th = v
            th = v
        elif typ == EVT_TIME_LOW:
            tl = word & 0xFFF
        elif typ == EVT_ADDR_Y:
            y = word & 0x7FF
        elif typ in (EVT_ADDR_X, VECT_BASE_X, VECT_12, VECT_8):
            t = (loops << 24) | (th << 12) | tl
            if typ == EVT_ADDR_X:
                out_x.append(word & 0x7FF); out_y.append(y); out_p.append((word >> 11) & 1); out_t.append(t)
            elif typ == VECT_BASE_X:
                base = word & 0x7FF; pol = (word >> 11) & 1
            else:
                nb = 12 if typ == VECT_12 else 8
                mask = word & ((1 << nb) - 1)
                for b in range(nb):
                    if mask >> b & 1:
                        out_x.append(base + b); out_y.append(y); out_p.append(pol); out_t.append(t)
                base += nb
    ev = np.empty(len(out_x), EVENT_DTYPE)
    ev["x"], ev["y"], ev["p"], ev["t"] = out_x, out_y, out_p, out_t
    return ev, hdr


# ---------------------------------------------------------------- bias files
BIAS_NAMES = ("bias_diff_on", "bias_diff_off", "bias_fo", "bias_hpf", "bias_refr")


def read_bias_file(path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            if "%" in line:
                v, name = line.split("%", 1)
                out[name.strip()] = int(float(v))
    return out


def write_bias_file(path, biases: dict):
    with open(path, "w") as f:
        for k in BIAS_NAMES:
            f.write(f"{int(biases.get(k, 0)):<6d}% {k}\n")


# ---------------------------------------------------------------- representations
def histogram(ev, width, height):
    """2 x H x W event counts (OFF, ON)."""
    h = np.zeros((2, height, width), np.float32)
    np.add.at(h, (ev["p"].astype(int), ev["y"].astype(int), ev["x"].astype(int)), 1)
    return h


def time_surface(ev, width, height, t_now=None, tau_us=50000.0):
    """2 x H x W exponentially decaying time surface."""
    t_now = ev["t"].max() if t_now is None and len(ev) else (t_now or 0)
    last = np.full((2, height, width), -np.inf)
    np.maximum.at(last, (ev["p"].astype(int), ev["y"].astype(int), ev["x"].astype(int)), ev["t"].astype(float))
    return np.exp((last - t_now) / tau_us).astype(np.float32)


def voxel_grid(ev, width, height, bins=5):
    """bins x H x W voxel grid with bilinear temporal interpolation (E-RAFT / E2VID style)."""
    vg = np.zeros((bins, height, width), np.float32)
    if len(ev) == 0:
        return vg
    t = ev["t"].astype(np.float64)
    tn = (bins - 1) * (t - t[0]) / max(1.0, t[-1] - t[0])
    pol = np.where(ev["p"] > 0, 1.0, -1.0)
    t0 = np.floor(tn).astype(int)
    for dtb in (0, 1):
        tb = t0 + dtb
        w = pol * np.maximum(0.0, 1.0 - np.abs(tn - tb))
        ok = (tb >= 0) & (tb < bins)
        np.add.at(vg, (tb[ok], ev["y"][ok].astype(int), ev["x"][ok].astype(int)), w[ok])
    return vg
