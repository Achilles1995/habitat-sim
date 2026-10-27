"""
GPU backend (NVIDIA Warp) for the prophesee.sensor.evk4 event camera.

Same physics as the C++ plugin (radiometric ray casting with smooth-normal / UV meshes, cast
shadows, stratified pixel-area sampling, lens distortion, 2-pole log photoreceptor with
light-dependent bandwidth, Rice-hazard shot noise, mismatch, refractory, leak, hot pixels,
latency) + the same readout / ESP (ROI, arbiter, AFK, STC/trail, ERC) implemented with
vectorised numpy.  Rendering and the pixel circuit run as Warp kernels on CUDA (or on the CPU
for testing).  Sensor configuration is read from the <plugin> config in the MJCF, so the same
model file drives both backends:

    m = evk.load_model("scene.xml", backend="warp")   # C++ plugin then only reserves the slot
    d = mujoco.MjData(m)
    cam = EVK4Warp(m, "evk4", device="cuda:0")
    while ...:
        mujoco.mj_step(m, d)
        cam.update(d)                  # after every mj_step
        ev = cam.labeled_events()      # x, y, p, t[us], geom, depth, flow_x, flow_y
"""
from __future__ import annotations

import math
import os
import time

import mujoco
import numpy as np
import warp as wp

from . import EVENT_DTYPE, LABELED_DTYPE

wp.config.quiet = True
# forward-only kernels: no adjoint code is generated -> about half the first-run JIT time
wp.set_module_options({"enable_backward": False})
_MAXREG = int(os.environ.get("EVK4_MAXREG", "0") or 0)   # optional register cap for the render kernels

from .config import ATTRS, DEFAULTS, _STR, read_config  # noqa: F401  (re-exported)

PGX, PGY = 32, 18


# ---------------------------------------------------------------- hashing (same as C++)
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def np_mix64(z):
    with np.errstate(over="ignore"):
        z = (z + np.uint64(0x9E3779B97F4A7C15))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def np_key(seed, pix, sample, stream):
    with np.errstate(over="ignore"):
        a = np_mix64(np.uint64(sample) * np.uint64(0x9FB21C651E98DF25) ^ np.uint64(stream))
        b = np_mix64(pix.astype(np.uint64) * np.uint64(0x8CB92BA72F3D8DD7) ^ a)
        return np_mix64(np.uint64(seed) * np.uint64(0x632BE59BD9B4E019) ^ b)


def np_u01(key):
    return ((np_mix64(key) >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / 9007199254740992.0)


def np_gauss(key):
    a = np_u01(key)
    b = np_u01(key ^ np.uint64(0xDA942042E4DD58B5))
    return np.sqrt(-2.0 * np.log(a)) * np.cos(2.0 * np.pi * b)


@wp.func
def mix64(z: wp.uint64):
    z = z + wp.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
    return z ^ (z >> wp.uint64(31))


@wp.func
def key_of(seed: wp.uint64, pix: wp.uint64, sample: wp.uint64, stream: wp.uint64):
    a = mix64(sample * wp.uint64(0x9FB21C651E98DF25) ^ stream)
    b = mix64(pix * wp.uint64(0x8CB92BA72F3D8DD7) ^ a)
    return mix64(seed * wp.uint64(0x632BE59BD9B4E019) ^ b)


@wp.func
def u01(key: wp.uint64):
    return (wp.float64(mix64(key) >> wp.uint64(11)) + wp.float64(0.5)) * wp.float64(1.0 / 9007199254740992.0)


@wp.func
def u01f(key: wp.uint64):
    # 24-bit uniform in float32 (avoids slow fp64 on consumer GPUs)
    return (float(mix64(key) >> wp.uint64(40)) + 0.5) * (1.0 / 16777216.0)


@wp.func
def gauss(key: wp.uint64):
    a = u01(key)
    b = u01(key ^ wp.uint64(0xDA942042E4DD58B5))
    return wp.float32(wp.sqrt(wp.float64(-2.0) * wp.log(a)) * wp.cos(wp.float64(2.0 * math.pi) * b))


# ---------------------------------------------------------------- geometry
@wp.struct
class Cam:
    pos: wp.vec3
    R: wp.mat33
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float


@wp.func
def project_px(cam: Cam, w: wp.vec3):
    c = wp.transpose(cam.R) * (w - cam.pos)
    z = -c[2]
    if z < 1.0e-4:
        return wp.vec3(0.0, 0.0, 0.0)
    x = c[0] / z
    y = -c[1] / z
    r2 = x * x + y * y
    rad = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2 + cam.k3 * r2 * r2 * r2
    xd = x * rad + 2.0 * cam.p1 * x * y + cam.p2 * (r2 + 2.0 * x * x)
    yd = y * rad + cam.p1 * (r2 + 2.0 * y * y) + 2.0 * cam.p2 * x * y
    return wp.vec3(cam.cx + cam.fx * xd, cam.cy + cam.fy * yd, 1.0)


@wp.func
def solve_quad(a: float, b: float, c: float):
    # smallest non-negative root of a t^2 + 2 b t + c = 0, -1 if none
    det = b * b - a * c
    if det < 0.0 or a < 1.0e-12:
        return -1.0
    s = wp.sqrt(det)
    t0 = (-b - s) / a
    t1 = (-b + s) / a
    if t0 >= 0.0:
        return t0
    if t1 >= 0.0:
        return t1
    return -1.0


@wp.func
def ray_prim(gtype: int, size: wp.vec3, lo: wp.vec3, ld: wp.vec3):
    """Local-frame ray vs primitive: returns (t, nx, ny, nz); t < 0 = miss."""
    miss = wp.vec4(-1.0, 0.0, 0.0, 1.0)
    if gtype == 0:  # plane
        if wp.abs(ld[2]) < 1.0e-12:
            return miss
        t = -lo[2] / ld[2]
        if t < 0.0:
            return miss
        p = lo + t * ld
        if size[0] > 0.0 and wp.abs(p[0]) > size[0]:
            return miss
        if size[1] > 0.0 and wp.abs(p[1]) > size[1]:
            return miss
        return wp.vec4(t, 0.0, 0.0, 1.0)
    if gtype == 2:  # sphere
        t = solve_quad(wp.dot(ld, ld), wp.dot(lo, ld), wp.dot(lo, lo) - size[0] * size[0])
        if t < 0.0:
            return miss
        n = wp.normalize(lo + t * ld)
        return wp.vec4(t, n[0], n[1], n[2])
    if gtype == 4:  # ellipsoid
        s = size
        o2 = wp.vec3(lo[0] / s[0], lo[1] / s[1], lo[2] / s[2])
        d2 = wp.vec3(ld[0] / s[0], ld[1] / s[1], ld[2] / s[2])
        t = solve_quad(wp.dot(d2, d2), wp.dot(o2, d2), wp.dot(o2, o2) - 1.0)
        if t < 0.0:
            return miss
        p = lo + t * ld
        n = wp.normalize(wp.vec3(p[0] / (s[0] * s[0]), p[1] / (s[1] * s[1]), p[2] / (s[2] * s[2])))
        return wp.vec4(t, n[0], n[1], n[2])
    if gtype == 3 or gtype == 5:  # capsule / cylinder: side
        r = size[0]
        h = size[1]
        best = -1.0
        bn = wp.vec3(0.0, 0.0, 1.0)
        a = ld[0] * ld[0] + ld[1] * ld[1]
        t = solve_quad(a, lo[0] * ld[0] + lo[1] * ld[1], lo[0] * lo[0] + lo[1] * lo[1] - r * r)
        if t >= 0.0:
            p = lo + t * ld
            if wp.abs(p[2]) <= h:
                best = t
                bn = wp.normalize(wp.vec3(p[0], p[1], 0.0))
            elif gtype == 5:
                # entering through the side outside the height: try the far root (inside caps handled below)
                best = -1.0
        if gtype == 3:  # caps are spheres
            for sgn in range(2):
                zc = h
                if sgn == 1:
                    zc = -h
                oc = lo - wp.vec3(0.0, 0.0, zc)
                ts = solve_quad(wp.dot(ld, ld), wp.dot(oc, ld), wp.dot(oc, oc) - r * r)
                if ts >= 0.0 and (best < 0.0 or ts < best):
                    best = ts
                    bn = wp.normalize(oc + ts * ld)
        else:  # flat caps
            if wp.abs(ld[2]) > 1.0e-12:
                for sgn in range(2):
                    zc = h
                    if sgn == 1:
                        zc = -h
                    tc = (zc - lo[2]) / ld[2]
                    if tc >= 0.0:
                        p = lo + tc * ld
                        if p[0] * p[0] + p[1] * p[1] <= r * r and (best < 0.0 or tc < best):
                            best = tc
                            bn = wp.vec3(0.0, 0.0, wp.sign(zc))
        if best < 0.0:
            return miss
        return wp.vec4(best, bn[0], bn[1], bn[2])
    if gtype == 6:  # box
        tmin = float(-1.0e30)
        tmax = float(1.0e30)
        ax = int(0)
        sg = float(1.0)
        for k in range(3):
            if wp.abs(ld[k]) < 1.0e-12:
                if wp.abs(lo[k]) > size[k]:
                    return miss
            else:
                t1 = (-size[k] - lo[k]) / ld[k]
                t2 = (size[k] - lo[k]) / ld[k]
                s1 = float(-1.0)
                if t1 > t2:
                    tmp = t1
                    t1 = t2
                    t2 = tmp
                    s1 = 1.0
                if t1 > tmin:
                    tmin = t1
                    ax = k
                    sg = s1
                if t2 < tmax:
                    tmax = t2
        if tmin > tmax or tmax < 0.0:
            return miss
        t = tmin
        if t < 0.0:
            return miss
        n = wp.vec3(0.0, 0.0, 0.0)
        n[ax] = sg
        return wp.vec4(t, n[0], n[1], n[2])
    return miss


@wp.struct
class Scene:
    nvis: int
    nlight: int
    lux: float
    flick: float
    bg_lv: float
    k_rate: float
    shadows: int
    spec_scale: float
    wn: float


@wp.func
def tex_sample(texdata: wp.array(dtype=wp.vec3), off: int, w: int, h: int, u: float, v: float):
    u = u - wp.floor(u)
    v = v - wp.floor(v)
    fu = u * float(w) - 0.5
    fv = v * float(h) - 0.5
    u0 = int(wp.floor(fu))
    v0 = int(wp.floor(fv))
    au = fu - float(u0)
    av = fv - float(v0)
    acc = wp.vec3(0.0, 0.0, 0.0)
    for k in range(4):
        uu = u0 + (k % 2)
        vv = v0 + (k / 2)
        uu = ((uu % w) + w) % w
        vv = ((vv % h) + h) % h
        wu = au
        if k % 2 == 0:
            wu = 1.0 - au
        wv = av
        if k / 2 == 0:
            wv = 1.0 - av
        acc = acc + (wu * wv) * texdata[off + vv * w + uu]
    return acc


@wp.kernel(cuda_max_registers=(_MAXREG or None))
def render_kernel(
    W: int, H: int, ns: int,
    corner: wp.array(dtype=wp.vec2), ctr: wp.array(dtype=wp.vec2), cos4: wp.array(dtype=float),
    cam: Cam, pcam: Cam, dtflow: float, do_flow: int,
    sc: Scene, TW: int, G: int, tile_cnt: wp.array(dtype=int), tile_list: wp.array(dtype=int),
    tile_dirty: wp.array(dtype=int), tile_hit: wp.array2d(dtype=float),
    vis: wp.array(dtype=int), gtype: wp.array(dtype=int), gsize: wp.array(dtype=wp.vec3),
    grb: wp.array(dtype=float), gpos: wp.array(dtype=wp.vec3), gmat: wp.array(dtype=wp.mat33),
    ppos: wp.array(dtype=wp.vec3), pmat: wp.array(dtype=wp.mat33),
    gmesh: wp.array(dtype=int), mesh_ids: wp.array(dtype=wp.uint64),
    mfn_off: wp.array(dtype=int), mfacenrm: wp.array(dtype=wp.vec3i), mnrm_off: wp.array(dtype=int),
    mnrm: wp.array(dtype=wp.vec3), mft_off: wp.array(dtype=int), mfacetex: wp.array(dtype=wp.vec3i),
    mtex_off: wp.array(dtype=int), mtex: wp.array(dtype=wp.vec2),
    grgb: wp.array(dtype=wp.vec3), gspec: wp.array(dtype=float), gshin: wp.array(dtype=float),
    gemis: wp.array(dtype=float), gtex: wp.array(dtype=int), grep: wp.array(dtype=wp.vec2),
    guni: wp.array(dtype=int), rgbw: wp.vec3,
    tex_off: wp.array(dtype=int), tex_w: wp.array(dtype=int), tex_h: wp.array(dtype=int),
    texdata: wp.array(dtype=wp.vec3),
    ltype: wp.array(dtype=int), lpos: wp.array(dtype=wp.vec3), ldir: wp.array(dtype=wp.vec3),
    lamb: wp.array(dtype=float), ldif: wp.array(dtype=float), lspe: wp.array(dtype=float),
    latt: wp.array(dtype=wp.vec3), lcut: wp.array(dtype=float), lexp: wp.array(dtype=float),
    lshadow: wp.array(dtype=int), lactive: wp.array(dtype=int), shadow_ids: wp.array(dtype=wp.uint64),
    out_raw: wp.array(dtype=float), out_g: wp.array(dtype=int), out_d: wp.array(dtype=float),
    out_f: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    u = i % W
    v = i / W
    tl = (v / 16) * TW + (u / 16)
    if tile_dirty[tl] == 0:                 # nothing that affects this tile moved: keep cached result
        out_f[i] = wp.vec2(0.0, 0.0)
        return
    acc = float(0.0)
    nss = ns * ns
    lab_s = nss / 2
    for s in range(nss):
        a = (float(s % ns) + 0.5) / float(ns)
        b = (float(s / ns) + 0.5) / float(ns)
        xy = ctr[i]
        if ns > 1:
            c00 = corner[v * (W + 1) + u]
            c10 = corner[v * (W + 1) + u + 1]
            c01 = corner[(v + 1) * (W + 1) + u]
            c11 = corner[(v + 1) * (W + 1) + u + 1]
            xy = (1.0 - a) * (1.0 - b) * c00 + a * (1.0 - b) * c10 + (1.0 - a) * b * c01 + a * b * c11
        d = wp.normalize(cam.R * wp.vec3(xy[0], -xy[1], -1.0))
        best = float(-1.0)
        bg = int(-1)
        bn = wp.vec3(0.0, 0.0, 1.0)
        buv = wp.vec2(0.0, 0.0)
        hasuv = int(0)
        ntl = tile_cnt[tl]
        for k in range(ntl):
            g = tile_list[tl * G + k]
            t = gtype[g]
            p = gpos[g]
            if t != 0:  # bounding-sphere reject
                oc = p - cam.pos
                tp = wp.dot(oc, d)
                rb = grb[g]
                if tp < -rb:
                    continue
                if wp.dot(oc, oc) - tp * tp > rb * rb:
                    continue
                if best > 0.0 and tp - rb > best:
                    continue
            R = gmat[g]
            lo = wp.transpose(R) * (cam.pos - p)
            ld = wp.transpose(R) * d
            if t == 7:  # mesh
                tmax = float(1.0e30)
                if best > 0.0:
                    tmax = best
                q = wp.mesh_query_ray(mesh_ids[gmesh[g]], lo, ld, tmax)
                if q.result and q.t > 1.0e-4:
                    if best < 0.0 or q.t < best:
                        best = q.t
                        bg = g
                        mi = gmesh[g]
                        f = q.face
                        fnv = mfacenrm[mfn_off[mi] + f]
                        w0 = q.u
                        w1 = q.v
                        w2 = 1.0 - q.u - q.v
                        no = mnrm_off[mi]
                        nl = w0 * mnrm[no + fnv[0]] + w1 * mnrm[no + fnv[1]] + w2 * mnrm[no + fnv[2]]
                        if wp.length(nl) < 1.0e-9:
                            nl = q.normal
                        bn = R * wp.normalize(nl)
                        hasuv = 0
                        if mtex_off[mi] >= 0:
                            ftv = mfacetex[mft_off[mi] + f]
                            to = mtex_off[mi]
                            buv = w0 * mtex[to + ftv[0]] + w1 * mtex[to + ftv[1]] + w2 * mtex[to + ftv[2]]
                            hasuv = 1
            else:
                r = ray_prim(t, gsize[g], lo, ld)
                if r[0] > 1.0e-4 and (best < 0.0 or r[0] < best):
                    best = r[0]
                    bg = g
                    bn = R * wp.vec3(r[1], r[2], r[3])
                    hasuv = 0
        Lv = sc.bg_lv
        if bg >= 0:
            P = cam.pos + best * d
            n = bn
            if wp.dot(n, d) > 0.0:
                n = -n
            # ---- albedo
            rgb = grgb[bg]
            texid = gtex[bg]
            if texid >= 0:
                t = gtype[bg]
                sz = gsize[bg]
                rep = grep[bg]
                s0 = rep[0]
                s1 = rep[1]
                uu = float(0.0)
                vv = float(0.0)
                ok = int(1)
                if t == 7:
                    if hasuv == 1:
                        uu = buv[0]
                        vv = buv[1]
                    else:
                        ok = 0
                else:
                    if guni[bg] == 1:
                        if sz[0] > 0.0:
                            s0 = s0 * sz[0]
                        if sz[1] > 0.0:
                            s1 = s1 * sz[1]
                    lp3 = wp.transpose(gmat[bg]) * (P - gpos[bg])
                    if t == 0:
                        if sz[0] > 0.0:
                            uu = (lp3[0] + sz[0]) / (2.0 * sz[0])
                        else:
                            uu = 0.5 * lp3[0]
                        if sz[1] > 0.0:
                            vv = 1.0 - (lp3[1] + sz[1]) / (2.0 * sz[1])
                        else:
                            vv = -0.5 * lp3[1]
                    elif t == 6:
                        a0 = lp3[0] / sz[0]
                        a1 = lp3[1] / sz[1]
                        a2 = lp3[2] / sz[2]
                        pp = a0
                        qq = a1
                        if wp.abs(a0) >= wp.abs(a1) and wp.abs(a0) >= wp.abs(a2):
                            pp = a1
                            qq = a2
                        elif wp.abs(a1) >= wp.abs(a2):
                            pp = a0
                            qq = a2
                        uu = (pp + 1.0) / 2.0
                        vv = 1.0 - (qq + 1.0) / 2.0
                    elif t == 5:
                        if wp.abs(lp3[2]) > sz[1] * 0.999:
                            uu = 0.5 + 0.5 * lp3[0] / sz[0]
                            vv = 0.5 + 0.5 * lp3[1] / sz[0]
                        else:
                            uu = wp.atan2(lp3[1], lp3[0]) / (2.0 * wp.pi) + 0.5
                            vv = 0.5 + 0.5 * lp3[2] / sz[1]
                    elif t == 2 or t == 4:
                        rr = wp.length(lp3) + 1.0e-12
                        uu = wp.atan2(lp3[1], lp3[0]) / (2.0 * wp.pi) + 0.5
                        vv = wp.acos(wp.clamp(lp3[2] / rr, -1.0, 1.0)) / wp.pi
                    else:
                        ok = 0
                if ok == 1:
                    tc = tex_sample(texdata, tex_off[texid], tex_w[texid], tex_h[texid], uu * s0, vv * s1)
                    rgb = wp.cw_mul(rgb, tc)
            rho = wp.dot(rgbw, rgb) / sc.wn
            # ---- lights
            Po = P + 2.0e-4 * n
            Ed = float(0.0)
            Es = float(0.0)
            for l in range(sc.nlight):
                if lactive[l] == 0:
                    continue
                Ed = Ed + lamb[l]
                dif = ldif[l] * sc.flick
                spl = lspe[l] * sc.flick
                if dif <= 0.0 and spl <= 0.0:
                    continue
                L = -ldir[l]
                dist = float(1.0e9)
                att = float(1.0)
                spot = float(1.0)
                if ltype[l] != 1:
                    Lv3 = lpos[l] - P
                    dist = wp.length(Lv3)
                    L = Lv3 / dist
                    at = latt[l]
                    att = 1.0 / wp.max(1.0e-6, at[0] + at[1] * dist + at[2] * dist * dist)
                    if ltype[l] == 0:
                        ca = -wp.dot(L, ldir[l])
                        if ca < lcut[l]:
                            spot = 0.0
                        else:
                            spot = wp.pow(wp.max(0.0, ca), lexp[l])
                ndl = wp.dot(n, L)
                if ndl <= 0.0 or spot <= 0.0:
                    continue
                if sc.shadows == 1 and lshadow[l] == 1:
                    occl = int(0)
                    for k in range(sc.nvis):
                        g = vis[k]
                        t = gtype[g]
                        p = gpos[g]
                        if t != 0:
                            oc = p - Po
                            tp = wp.dot(oc, L)
                            rb = grb[g]
                            if tp < -rb or tp - rb > dist:
                                continue
                            if wp.dot(oc, oc) - tp * tp > rb * rb:
                                continue
                        R = gmat[g]
                        lo = wp.transpose(R) * (Po - p)
                        ld = wp.transpose(R) * L
                        if t == 7:
                            if wp.mesh_query_ray_anyhit(shadow_ids[gmesh[g]], lo, ld, dist):
                                occl = 1
                                break
                        else:
                            th = ray_prim(t, gsize[g], lo, ld)[0]
                            if th > 0.0 and th < dist:
                                occl = 1
                                break
                    if occl == 1:
                        continue
                Ed = Ed + dif * att * spot * ndl
                hv = wp.normalize(L - d)
                ndh = wp.max(0.0, wp.dot(n, hv))
                Es = Es + spl * att * spot * wp.pow(ndh, wp.max(1.0, gshin[bg] * 128.0)) * ndl
            Lv = (rho * Ed + sc.spec_scale * gspec[bg] * Es + gemis[bg] * rho * sc.lux) / wp.pi
        acc = acc + Lv
        if s == lab_s:
            out_g[i] = bg
            out_d[i] = best
            if bg >= 0:                              # tile hit bounds (for shadow-dependency tests)
                Ph = cam.pos + best * d
                for j in range(3):
                    wp.atomic_min(tile_hit, tl, j, Ph[j])
                    wp.atomic_max(tile_hit, tl, j + 3, Ph[j])
            fl = wp.vec2(0.0, 0.0)
            if bg >= 0 and do_flow == 1:
                P = cam.pos + best * d
                R = gmat[bg]
                loc = wp.transpose(R) * (P - gpos[bg])
                wprev = pmat[bg] * loc + ppos[bg]
                pa = project_px(pcam, wprev)
                pb = project_px(cam, P)
                if pa[2] > 0.0 and pb[2] > 0.0:
                    fl = wp.vec2((pb[0] - pa[0]) / dtflow, (pb[1] - pa[1]) / dtflow)
            out_f[i] = fl
    out_raw[i] = sc.k_rate * acc / float(nss) * cos4[i]


# ---------------------------------------------------------------- split renderer (v2.5)
# The fused render_kernel above does primary visibility, texturing, lighting and shadow rays in
# one very large kernel: high register pressure (low occupancy), heavy divergence (mesh BVH vs.
# primitives vs. background inside one warp), a shadow loop over EVERY visible geom, and six
# contended atomics per pixel for the tile hit bounds.  The split path:
#   1. hit_kernel       primary rays only -> compact hit buffer (t, geom, normal, uv)
#   2. tilebound_kernel per-tile world-space AABB of the hits (no atomics)
#   3. shadowcull_kernel per (tile, light): the few geoms that can possibly shadow that tile
#   4. shade_kernel     albedo + lights, shadow rays only against the tile's occluder list
# Threads are mapped tile-major (a warp covers a 16x2 pixel block, not a 32x1 row), so rays in a
# warp are coherent and walk the same BVH nodes.  Output is identical to render_kernel.
@wp.func
def pix_dir(cam: Cam, ns: int, s: int, i: int, u: int, v: int, W: int,
            corner: wp.array(dtype=wp.vec2), ctr: wp.array(dtype=wp.vec2)):
    xy = ctr[i]
    if ns > 1:
        a = (float(s % ns) + 0.5) / float(ns)
        b = (float(s / ns) + 0.5) / float(ns)
        c00 = corner[v * (W + 1) + u]
        c10 = corner[v * (W + 1) + u + 1]
        c01 = corner[(v + 1) * (W + 1) + u]
        c11 = corner[(v + 1) * (W + 1) + u + 1]
        xy = (1.0 - a) * (1.0 - b) * c00 + a * (1.0 - b) * c10 + (1.0 - a) * b * c01 + a * b * c11
    return wp.normalize(cam.R * wp.vec3(xy[0], -xy[1], -1.0))


@wp.kernel(cuda_max_registers=(_MAXREG or None))
def hit_kernel(
    W: int, H: int, ns: int, TW: int,
    corner: wp.array(dtype=wp.vec2), ctr: wp.array(dtype=wp.vec2),
    cam: Cam,
    G: int, tile_cnt: wp.array(dtype=int), tile_list: wp.array(dtype=int), tile_dirty: wp.array(dtype=int),
    gtype: wp.array(dtype=int), gsize: wp.array(dtype=wp.vec3), grb: wp.array(dtype=float),
    gpos: wp.array(dtype=wp.vec3), gmat: wp.array(dtype=wp.mat33),
    gmesh: wp.array(dtype=int), mesh_ids: wp.array(dtype=wp.uint64),
    hit_t: wp.array(dtype=float), hit_g: wp.array(dtype=int), hit_n: wp.array(dtype=wp.vec3),
    hit_h: wp.array(dtype=int),
    out_g: wp.array(dtype=int), out_d: wp.array(dtype=float)):
    j = wp.tid()
    tl = j / 256
    lj = j % 256
    u = (tl % TW) * 16 + lj % 16
    v = (tl / TW) * 16 + lj / 16
    if u >= W or v >= H:
        return
    i = v * W + u
    if tile_dirty[tl] == 0:
        return
    nss = ns * ns
    lab_s = nss / 2
    for s in range(nss):
        d = pix_dir(cam, ns, s, i, u, v, W, corner, ctr)
        best = float(-1.0)
        bg = int(-1)
        bn = wp.vec3(0.0, 0.0, 1.0)
        face = int(0)
        ntl = tile_cnt[tl]
        for k in range(ntl):
            g = tile_list[tl * G + k]
            t = gtype[g]
            p = gpos[g]
            if t != 0:
                oc = p - cam.pos
                tp = wp.dot(oc, d)
                rb = grb[g]
                if tp < -rb:
                    continue
                if wp.dot(oc, oc) - tp * tp > rb * rb:
                    continue
                if best > 0.0 and tp - rb > best:
                    continue
            R = gmat[g]
            lo = wp.transpose(R) * (cam.pos - p)
            ld = wp.transpose(R) * d
            if t == 7:
                tmax = float(1.0e30)
                if best > 0.0:
                    tmax = best
                q = wp.mesh_query_ray(mesh_ids[gmesh[g]], lo, ld, tmax)
                if q.result and q.t > 1.0e-4:
                    if best < 0.0 or q.t < best:
                        best = q.t
                        bg = g
                        bn = wp.vec3(q.u, q.v, 0.0)       # barycentrics; normal/uv in shade_kernel
                        face = q.face
            else:
                r = ray_prim(t, gsize[g], lo, ld)
                if r[0] > 1.0e-4 and (best < 0.0 or r[0] < best):
                    best = r[0]
                    bg = g
                    bn = wp.vec3(r[1], r[2], r[3])       # local-frame normal
        hs = i * nss + s
        hit_t[hs] = best
        hit_g[hs] = bg
        hit_n[hs] = bn
        hit_h[hs] = face
        if s == lab_s:
            out_g[i] = bg
            out_d[i] = best


@wp.kernel
def flow_kernel(W: int, ns: int, TW: int, corner: wp.array(dtype=wp.vec2), ctr: wp.array(dtype=wp.vec2),
                cam: Cam, pcam: Cam, dtflow: float, do_flow: int, tile_dirty: wp.array(dtype=int),
                gpos: wp.array(dtype=wp.vec3), gmat: wp.array(dtype=wp.mat33),
                ppos: wp.array(dtype=wp.vec3), pmat: wp.array(dtype=wp.mat33),
                lab_g: wp.array(dtype=int), lab_d: wp.array(dtype=float), out_f: wp.array(dtype=wp.vec2)):
    """Exact optical-flow label of the labelled sample (kept out of hit_kernel: fewer registers)."""
    i = wp.tid()
    u = i % W
    v = i / W
    tl = (v / 16) * TW + (u / 16)
    fl = wp.vec2(0.0, 0.0)
    bg = lab_g[i]
    if do_flow == 1 and tile_dirty[tl] == 1 and bg >= 0:
        d = pix_dir(cam, ns, (ns * ns) / 2, i, u, v, W, corner, ctr)
        P = cam.pos + lab_d[i] * d
        R = gmat[bg]
        loc = wp.transpose(R) * (P - gpos[bg])
        wprev = pmat[bg] * loc + ppos[bg]
        pa = project_px(pcam, wprev)
        pb = project_px(cam, P)
        if pa[2] > 0.0 and pb[2] > 0.0:
            fl = wp.vec2((pb[0] - pa[0]) / dtflow, (pb[1] - pa[1]) / dtflow)
    out_f[i] = fl


@wp.kernel
def tilebound_kernel(W: int, H: int, ns: int, TW: int, cam: Cam,
                     corner: wp.array(dtype=wp.vec2), ctr: wp.array(dtype=wp.vec2),
                     tile_dirty: wp.array(dtype=int), hit_t: wp.array(dtype=float), hit_g: wp.array(dtype=int),
                     tile_hit: wp.array2d(dtype=float)):
    """World-space AABB of all primary hits of a (re-rendered) tile - used by the shadow culling of
    this render and by the shadow-dependency dirty test of the next one."""
    tl = wp.tid()
    if tile_dirty[tl] == 0:
        return
    x0 = (tl % TW) * 16
    y0 = (tl / TW) * 16
    nss = ns * ns
    lo = wp.vec3(1.0e30, 1.0e30, 1.0e30)
    hi = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
    for lj in range(256):
        u = x0 + lj % 16
        v = y0 + lj / 16
        if u < W and v < H:
            i = v * W + u
            for s in range(nss):
                hs = i * nss + s
                if hit_g[hs] >= 0:
                    P = cam.pos + hit_t[hs] * pix_dir(cam, ns, s, i, u, v, W, corner, ctr)
                    lo = wp.min(lo, P)
                    hi = wp.max(hi, P)
    for k in range(3):
        tile_hit[tl, k] = lo[k]
        tile_hit[tl, k + 3] = hi[k]


@wp.kernel
def shadowcull_kernel(nlight: int, nvis: int, shadows: int, tile_dirty: wp.array(dtype=int),
                      tile_hit: wp.array2d(dtype=float), vis: wp.array(dtype=int), gtype: wp.array(dtype=int),
                      gpos: wp.array(dtype=wp.vec3), grb: wp.array(dtype=float),
                      ltype: wp.array(dtype=int), lpos: wp.array(dtype=wp.vec3), ldir: wp.array(dtype=wp.vec3),
                      lshadow: wp.array(dtype=int), lactive: wp.array(dtype=int),
                      sh_cnt: wp.array(dtype=int), sh_list: wp.array(dtype=int)):
    """Per (tile, light): every shadow ray of the tile starts inside the tile's hit AABB (radius R
    around C) and ends at the light, so it lies inside the capsule of radius R around the segment
    C -> light.  Only geoms whose bounding sphere touches that capsule can occlude it (planes are
    always kept).  Conservative => identical shading to testing every geom."""
    idx = wp.tid()
    tl = idx / nlight
    l = idx % nlight
    if tile_dirty[tl] == 0:
        return
    n = int(0)
    if shadows == 1 and lactive[l] == 1 and lshadow[l] == 1 and tile_hit[tl, 0] <= tile_hit[tl, 3]:
        C = wp.vec3(0.5 * (tile_hit[tl, 0] + tile_hit[tl, 3]), 0.5 * (tile_hit[tl, 1] + tile_hit[tl, 4]),
                    0.5 * (tile_hit[tl, 2] + tile_hit[tl, 5]))
        ext = wp.vec3(tile_hit[tl, 3] - tile_hit[tl, 0], tile_hit[tl, 4] - tile_hit[tl, 1],
                      tile_hit[tl, 5] - tile_hit[tl, 2])
        R = 0.5 * wp.length(ext) + 1.0e-3            # + ray-origin offset and float slack
        E = lpos[l]
        if ltype[l] == 1:
            E = C - 1.0e3 * ldir[l]
        seg = E - C
        sl2 = wp.dot(seg, seg)
        for k in range(nvis):
            g = vis[k]
            ok = int(1)
            if gtype[g] != 0:
                w = gpos[g] - C
                tt = float(0.0)
                if sl2 > 0.0:
                    tt = wp.clamp(wp.dot(w, seg) / sl2, 0.0, 1.0)
                q = w - tt * seg
                rr = grb[g] * 1.0001 + R
                if wp.dot(q, q) > rr * rr:
                    ok = 0
            if ok == 1:
                sh_list[idx * nvis + n] = g
                n = n + 1
    sh_cnt[idx] = n


@wp.func
def tile_pix(j: int, TW: int, W: int, H: int):
    """tile-major thread -> (pixel index or -1, tile)"""
    tl = j / 256
    lj = j % 256
    u = (tl % TW) * 16 + lj % 16
    v = (tl / TW) * 16 + lj / 16
    if u >= W or v >= H:
        return wp.vec3i(-1, tl, 0)
    return wp.vec3i(v * W + u, tl, 0)


@wp.kernel(cuda_max_registers=(_MAXREG or None))
def surf_kernel(
    W: int, H: int, ns: int, TW: int,
    corner: wp.array(dtype=wp.vec2), ctr: wp.array(dtype=wp.vec2),
    cam: Cam, tile_dirty: wp.array(dtype=int),
    hit_t: wp.array(dtype=float), hit_g: wp.array(dtype=int), hit_n: wp.array(dtype=wp.vec3),
    hit_h: wp.array(dtype=int), hit_rho: wp.array(dtype=float),
    gtype: wp.array(dtype=int), gsize: wp.array(dtype=wp.vec3),
    gpos: wp.array(dtype=wp.vec3), gmat: wp.array(dtype=wp.mat33),
    gmesh: wp.array(dtype=int), mesh_ids: wp.array(dtype=wp.uint64),
    mfn_off: wp.array(dtype=int), mfacenrm: wp.array(dtype=wp.vec3i), mnrm_off: wp.array(dtype=int),
    mnrm: wp.array(dtype=wp.vec3), mft_off: wp.array(dtype=int), mfacetex: wp.array(dtype=wp.vec3i),
    mtex_off: wp.array(dtype=int), mtex: wp.array(dtype=wp.vec2),
    grgb: wp.array(dtype=wp.vec3), gtex: wp.array(dtype=int), grep: wp.array(dtype=wp.vec2),
    guni: wp.array(dtype=int), rgbw: wp.vec3, wn: float,
    tex_off: wp.array(dtype=int), tex_w: wp.array(dtype=int), tex_h: wp.array(dtype=int),
    texdata: wp.array(dtype=wp.vec3)):
    """Surface pass: world-space shading normal (smooth, facing the camera) -> hit_n (in place),
    luminance albedo incl. texture -> hit_rho."""
    tp_ = tile_pix(wp.tid(), TW, W, H)
    i = tp_[0]
    tl = tp_[1]
    if i < 0:
        return
    if tile_dirty[tl] == 0:
        return
    u = i % W
    v = i / W
    nss = ns * ns
    for s in range(nss):
        hs = i * nss + s
        bg = hit_g[hs]
        if bg >= 0:
            d = pix_dir(cam, ns, s, i, u, v, W, corner, ctr)
            P = cam.pos + hit_t[hs] * d
            hn = hit_n[hs]
            Rg = gmat[bg]
            hasuv = int(0)
            buv = wp.vec2(0.0, 0.0)
            n = Rg * hn
            t = gtype[bg]
            if t == 7:                               # smooth normal + uv from the barycentrics
                mi = gmesh[bg]
                f = hit_h[hs]
                fnv = mfacenrm[mfn_off[mi] + f]
                w0 = hn[0]
                w1 = hn[1]
                w2 = 1.0 - hn[0] - hn[1]
                no = mnrm_off[mi]
                nl = w0 * mnrm[no + fnv[0]] + w1 * mnrm[no + fnv[1]] + w2 * mnrm[no + fnv[2]]
                if wp.length(nl) < 1.0e-9:
                    nl = wp.mesh_eval_face_normal(mesh_ids[mi], f)
                n = Rg * wp.normalize(nl)
                if mtex_off[mi] >= 0:
                    ftv = mfacetex[mft_off[mi] + f]
                    to = mtex_off[mi]
                    buv = w0 * mtex[to + ftv[0]] + w1 * mtex[to + ftv[1]] + w2 * mtex[to + ftv[2]]
                    hasuv = 1
            if wp.dot(n, d) > 0.0:
                n = -n
            hit_n[hs] = n
            rgb = grgb[bg]
            texid = gtex[bg]
            if texid >= 0:
                sz = gsize[bg]
                rep = grep[bg]
                s0 = rep[0]
                s1 = rep[1]
                uu = float(0.0)
                vv = float(0.0)
                ok = int(1)
                if t == 7:
                    if hasuv == 1:
                        uu = buv[0]
                        vv = buv[1]
                    else:
                        ok = 0
                else:
                    if guni[bg] == 1:
                        if sz[0] > 0.0:
                            s0 = s0 * sz[0]
                        if sz[1] > 0.0:
                            s1 = s1 * sz[1]
                    lp3 = wp.transpose(Rg) * (P - gpos[bg])
                    if t == 0:
                        if sz[0] > 0.0:
                            uu = (lp3[0] + sz[0]) / (2.0 * sz[0])
                        else:
                            uu = 0.5 * lp3[0]
                        if sz[1] > 0.0:
                            vv = 1.0 - (lp3[1] + sz[1]) / (2.0 * sz[1])
                        else:
                            vv = -0.5 * lp3[1]
                    elif t == 6:
                        a0 = lp3[0] / sz[0]
                        a1 = lp3[1] / sz[1]
                        a2 = lp3[2] / sz[2]
                        pp = a0
                        qq = a1
                        if wp.abs(a0) >= wp.abs(a1) and wp.abs(a0) >= wp.abs(a2):
                            pp = a1
                            qq = a2
                        elif wp.abs(a1) >= wp.abs(a2):
                            pp = a0
                            qq = a2
                        uu = (pp + 1.0) / 2.0
                        vv = 1.0 - (qq + 1.0) / 2.0
                    elif t == 5:
                        if wp.abs(lp3[2]) > sz[1] * 0.999:
                            uu = 0.5 + 0.5 * lp3[0] / sz[0]
                            vv = 0.5 + 0.5 * lp3[1] / sz[0]
                        else:
                            uu = wp.atan2(lp3[1], lp3[0]) / (2.0 * wp.pi) + 0.5
                            vv = 0.5 + 0.5 * lp3[2] / sz[1]
                    elif t == 2 or t == 4:
                        rr = wp.length(lp3) + 1.0e-12
                        uu = wp.atan2(lp3[1], lp3[0]) / (2.0 * wp.pi) + 0.5
                        vv = wp.acos(wp.clamp(lp3[2] / rr, -1.0, 1.0)) / wp.pi
                    else:
                        ok = 0
                if ok == 1:
                    tc = tex_sample(texdata, tex_off[texid], tex_w[texid], tex_h[texid], uu * s0, vv * s1)
                    rgb = wp.cw_mul(rgb, tc)
            hit_rho[hs] = wp.dot(rgbw, rgb) / wn


vec7 = wp.types.vector(length=7, dtype=float)


@wp.func
def light_geom(l: int, P: wp.vec3, n: wp.vec3, flick: float,
               ltype: wp.array(dtype=int), lpos: wp.array(dtype=wp.vec3), ldir: wp.array(dtype=wp.vec3),
               ldif: wp.array(dtype=float), lspe: wp.array(dtype=float), latt: wp.array(dtype=wp.vec3),
               lcut: wp.array(dtype=float), lexp: wp.array(dtype=float)):
    """-> (L.x, L.y, L.z, dist, att, spot, ndl); ndl <= 0 means the light does not reach P."""
    dif = ldif[l] * flick
    spl = lspe[l] * flick
    if dif <= 0.0 and spl <= 0.0:
        return vec7(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    L = -ldir[l]
    dist = float(1.0e9)
    att = float(1.0)
    spot = float(1.0)
    if ltype[l] != 1:
        Lv3 = lpos[l] - P
        dist = wp.length(Lv3)
        L = Lv3 / dist
        at = latt[l]
        att = 1.0 / wp.max(1.0e-6, at[0] + at[1] * dist + at[2] * dist * dist)
        if ltype[l] == 0:
            ca = -wp.dot(L, ldir[l])
            if ca < lcut[l]:
                spot = 0.0
            else:
                spot = wp.pow(wp.max(0.0, ca), lexp[l])
    ndl = wp.dot(n, L)
    if spot <= 0.0:
        ndl = 0.0
    return vec7(L[0], L[1], L[2], dist, att, spot, ndl)


@wp.kernel(cuda_max_registers=(_MAXREG or None))
def shadow_kernel(
    W: int, H: int, ns: int, TW: int, nlw: int,
    corner: wp.array(dtype=wp.vec2), ctr: wp.array(dtype=wp.vec2),
    cam: Cam, sc: Scene, tile_dirty: wp.array(dtype=int),
    hit_t: wp.array(dtype=float), hit_g: wp.array(dtype=int), hit_n: wp.array(dtype=wp.vec3),
    sh_cnt: wp.array(dtype=int), sh_list: wp.array(dtype=int),
    gtype: wp.array(dtype=int), gsize: wp.array(dtype=wp.vec3), grb: wp.array(dtype=float),
    gpos: wp.array(dtype=wp.vec3), gmat: wp.array(dtype=wp.mat33),
    gmesh: wp.array(dtype=int), shadow_ids: wp.array(dtype=wp.uint64),
    ltype: wp.array(dtype=int), lpos: wp.array(dtype=wp.vec3), ldir: wp.array(dtype=wp.vec3),
    ldif: wp.array(dtype=float), lspe: wp.array(dtype=float),
    latt: wp.array(dtype=wp.vec3), lcut: wp.array(dtype=float), lexp: wp.array(dtype=float),
    lshadow: wp.array(dtype=int), lactive: wp.array(dtype=int),
    occ: wp.array2d(dtype=int)):
    """Shadow rays only (any-hit), against the tile's culled occluder list -> bit per light."""
    tp_ = tile_pix(wp.tid(), TW, W, H)
    i = tp_[0]
    tl = tp_[1]
    if i < 0:
        return
    if tile_dirty[tl] == 0:
        return
    u = i % W
    v = i / W
    nss = ns * ns
    for s in range(nss):
        hs = i * nss + s
        for w in range(nlw):
            occ[hs, w] = 0
        if hit_g[hs] >= 0:
            d = pix_dir(cam, ns, s, i, u, v, W, corner, ctr)
            P = cam.pos + hit_t[hs] * d
            n = hit_n[hs]
            Po = P + 2.0e-4 * n
            for l in range(sc.nlight):
                if lactive[l] == 0 or lshadow[l] == 0:
                    continue
                lg = light_geom(l, P, n, sc.flick, ltype, lpos, ldir, ldif, lspe, latt, lcut, lexp)
                if lg[6] <= 0.0:
                    continue
                L = wp.vec3(lg[0], lg[1], lg[2])
                dist = lg[3]
                occl = int(0)
                base = (tl * sc.nlight + l) * sc.nvis
                for k in range(sh_cnt[tl * sc.nlight + l]):
                    g = sh_list[base + k]
                    t = gtype[g]
                    p = gpos[g]
                    if t != 0:
                        oc = p - Po
                        tq = wp.dot(oc, L)
                        rb = grb[g]
                        if tq < -rb or tq - rb > dist:
                            continue
                        if wp.dot(oc, oc) - tq * tq > rb * rb:
                            continue
                    R = gmat[g]
                    lo = wp.transpose(R) * (Po - p)
                    ld = wp.transpose(R) * L
                    if t == 7:
                        if wp.mesh_query_ray_anyhit(shadow_ids[gmesh[g]], lo, ld, dist):
                            occl = 1
                            break
                    else:
                        th = ray_prim(t, gsize[g], lo, ld)[0]
                        if th > 0.0 and th < dist:
                            occl = 1
                            break
                if occl == 1:
                    occ[hs, l / 32] = occ[hs, l / 32] | (1 << (l % 32))


@wp.kernel
def light_kernel(
    W: int, H: int, ns: int, TW: int, shadowed: int,
    corner: wp.array(dtype=wp.vec2), ctr: wp.array(dtype=wp.vec2), cos4: wp.array(dtype=float),
    cam: Cam, sc: Scene, tile_dirty: wp.array(dtype=int),
    hit_t: wp.array(dtype=float), hit_g: wp.array(dtype=int), hit_n: wp.array(dtype=wp.vec3),
    hit_rho: wp.array(dtype=float), occ: wp.array2d(dtype=int),
    gspec: wp.array(dtype=float), gshin: wp.array(dtype=float), gemis: wp.array(dtype=float),
    ltype: wp.array(dtype=int), lpos: wp.array(dtype=wp.vec3), ldir: wp.array(dtype=wp.vec3),
    lamb: wp.array(dtype=float), ldif: wp.array(dtype=float), lspe: wp.array(dtype=float),
    latt: wp.array(dtype=wp.vec3), lcut: wp.array(dtype=float), lexp: wp.array(dtype=float),
    lactive: wp.array(dtype=int), out_raw: wp.array(dtype=float)):
    """Direct lighting (ambient + diffuse + Blinn-Phong specular + emission) -> radiance -> e-/s."""
    tp_ = tile_pix(wp.tid(), TW, W, H)
    i = tp_[0]
    tl = tp_[1]
    if i < 0:
        return
    if tile_dirty[tl] == 0:
        return
    u = i % W
    v = i / W
    nss = ns * ns
    acc = float(0.0)
    for s in range(nss):
        hs = i * nss + s
        bg = hit_g[hs]
        Lv = sc.bg_lv
        if bg >= 0:
            d = pix_dir(cam, ns, s, i, u, v, W, corner, ctr)
            P = cam.pos + hit_t[hs] * d
            n = hit_n[hs]
            rho = hit_rho[hs]
            Ed = float(0.0)
            Es = float(0.0)
            for l in range(sc.nlight):
                if lactive[l] == 0:
                    continue
                Ed = Ed + lamb[l]
                lg = light_geom(l, P, n, sc.flick, ltype, lpos, ldir, ldif, lspe, latt, lcut, lexp)
                if lg[6] <= 0.0:
                    continue
                if shadowed == 1:
                    if (occ[hs, l / 32] >> (l % 32)) & 1 == 1:
                        continue
                L = wp.vec3(lg[0], lg[1], lg[2])
                att = lg[4]
                spot = lg[5]
                ndl = lg[6]
                dif = ldif[l] * sc.flick
                spl = lspe[l] * sc.flick
                Ed = Ed + dif * att * spot * ndl
                hv = wp.normalize(L - d)
                ndh = wp.max(0.0, wp.dot(n, hv))
                Es = Es + spl * att * spot * wp.pow(ndh, wp.max(1.0, gshin[bg] * 128.0)) * ndl
            Lv = (rho * Ed + sc.spec_scale * gspec[bg] * Es + gemis[bg] * rho * sc.lux) / wp.pi
        acc = acc + Lv
    out_raw[i] = sc.k_rate * acc / float(nss) * cos4[i]


@wp.kernel
def tile_kernel(G: int, ngeo: int, tb: wp.array(dtype=wp.vec4), gidx: wp.array(dtype=int),
                gstat: wp.array(dtype=float), grect: wp.array(dtype=wp.vec4),
                all_dirty: int, valid: wp.array(dtype=int),
                nmov: int, mrect: wp.array(dtype=wp.vec4), msph: wp.array(dtype=wp.vec4),
                nlight: int, ltype: wp.array(dtype=int), lpos: wp.array(dtype=wp.vec3),
                ldir: wp.array(dtype=wp.vec3), lshadow: wp.array(dtype=int), lactive: wp.array(dtype=int),
                shadows: int, tile_hit: wp.array2d(dtype=float),
                tile_cnt: wp.array(dtype=int), tile_list: wp.array(dtype=int), tile_dirty: wp.array(dtype=int)):
    """Per 16x16 tile: (1) candidate geoms whose projected bounds overlap the tile, (2) whether
    anything the tile depends on (visible geoms, their shadows) moved since its last render."""
    tl = wp.tid()
    b = tb[tl]
    n = int(0)
    for k in range(ngeo):
        st = gstat[k]
        ok = int(0)
        if st > 1.5:
            ok = 1
        elif st > 0.5:
            r = grect[k]
            if not (r[2] < b[0] or r[0] > b[2] or r[3] < b[1] or r[1] > b[3]):
                ok = 1
        if ok == 1:
            tile_list[tl * G + n] = gidx[k]
            n = n + 1
    tile_cnt[tl] = n
    dirty = int(0)
    if all_dirty == 1 or valid[tl] == 0:
        dirty = 1
    else:
        for k in range(nmov):                   # moved geoms: previous and current screen bounds
            r = mrect[k]
            if r[0] < 1.0e20:
                if not (r[2] < b[0] or r[0] > b[2] or r[3] < b[1] or r[1] > b[3]):
                    dirty = 1
        if dirty == 0 and shadows == 1 and tile_hit[tl, 0] <= tile_hit[tl, 3]:
            C = wp.vec3(0.5 * (tile_hit[tl, 0] + tile_hit[tl, 3]), 0.5 * (tile_hit[tl, 1] + tile_hit[tl, 4]),
                        0.5 * (tile_hit[tl, 2] + tile_hit[tl, 5]))
            ext = wp.vec3(tile_hit[tl, 3] - tile_hit[tl, 0], tile_hit[tl, 4] - tile_hit[tl, 1],
                          tile_hit[tl, 5] - tile_hit[tl, 2])
            R = 0.5 * wp.length(ext)
            for l in range(nlight):
                if lactive[l] == 0 or lshadow[l] == 0:
                    continue
                E = lpos[l]
                if ltype[l] == 1:
                    E = C - 20.0 * ldir[l]
                seg = E - C
                sl2 = wp.dot(seg, seg)
                for k in range(nmov):
                    sp = msph[k]
                    w = wp.vec3(sp[0], sp[1], sp[2]) - C
                    tt = float(0.0)
                    if sl2 > 0.0:
                        tt = wp.clamp(wp.dot(w, seg) / sl2, 0.0, 1.0)
                    q = w - tt * seg
                    rr = sp[3] + R
                    if wp.dot(q, q) <= rr * rr:
                        dirty = 1
    tile_dirty[tl] = dirty
    if dirty == 1:
        valid[tl] = 1
        for j in range(3):
            tile_hit[tl, j] = 1.0e30
            tile_hit[tl, j + 3] = -1.0e30


@wp.kernel
def make_keys(evf: wp.array2d(dtype=float), evt: wp.array2d(dtype=wp.float64),
              keys: wp.array(dtype=wp.int64), vals: wp.array(dtype=wp.int32)):
    """sort key = t_us * 2^21 + pixel  (time order, deterministic tie-break)"""
    i = wp.tid()
    tus = wp.int64(wp.floor(evt[i, 0] * wp.float64(1.0e6) + wp.float64(0.5)))
    keys[i] = tus * wp.int64(2097152) + wp.int64(evf[i, 0])
    vals[i] = i


@wp.kernel
def gather_events(vals: wp.array(dtype=wp.int32), evf: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    i = wp.tid()
    j = vals[i]
    for c in range(6):
        out[i, c] = evf[j, c]


@wp.kernel
def blur_h(src: wp.array(dtype=float), dst: wp.array(dtype=float), k: wp.array(dtype=float), r: int, W: int):
    i = wp.tid()
    u = i % W
    row = i - u
    acc = float(0.0)
    for j in range(-r, r + 1):
        uu = wp.clamp(u + j, 0, W - 1)
        acc = acc + k[j + r] * src[row + uu]
    dst[i] = acc


@wp.kernel
def blur_v(src: wp.array(dtype=float), dst: wp.array(dtype=float), k: wp.array(dtype=float), r: int, W: int, H: int):
    i = wp.tid()
    u = i % W
    v = i / W
    acc = float(0.0)
    for j in range(-r, r + 1):
        vv = wp.clamp(v + j, 0, H - 1)
        acc = acc + k[j + r] * src[vv * W + u]
    dst[i] = acc


@wp.kernel
def probe_kernel(W: int, H: int, sx: int, sy: int, ctr: wp.array(dtype=wp.vec2), cam: Cam,
                 lab_g: wp.array(dtype=int), lab_d: wp.array(dtype=float), static_g: wp.array(dtype=int),
                 out: wp.array(dtype=wp.vec4)):
    k = wp.tid()
    pu = k % 32
    pv = k / 32
    u = pu * sx + sx / 2
    v = pv * sy + sy / 2
    res = wp.vec4(0.0, 0.0, 0.0, 0.0)
    if u < W and v < H:
        i = v * W + u
        g = lab_g[i]
        if g >= 0:
            if static_g[g] == 1:
                xy = ctr[i]
                d = wp.normalize(cam.R * wp.vec3(xy[0], -xy[1], -1.0))
                P = cam.pos + lab_d[i] * d
                res = wp.vec4(P[0], P[1], P[2], 1.0)
    out[k] = res


@wp.struct
class PixParams:
    dark: float
    fmax: float
    r_half: float
    tau2: float
    sf: float
    gain: float
    shot: int
    trefr: wp.float64
    hpf: float
    lat: float
    jit: float
    seed: wp.uint64
    sample: wp.uint64
    W: int
    cap: int


@wp.func
def cum_haz(c0: float, sc: float, T: float, nu0: float, sig: float):
    if T <= 0.0:
        return float(0.0)
    if wp.abs(sc) * T < 1.0e-6 * sig:
        return nu0 * T * wp.exp(-c0 * c0 / (2.0 * sig * sig))
    x0 = c0 / (sig * 1.41421356)
    x1 = (c0 - sc * T) / (sig * 1.41421356)
    diff = float(0.0)
    if x0 > 0.0 and x1 > 0.0:
        diff = wp.erfc(x1) - wp.erfc(x0)
    elif x0 < 0.0 and x1 < 0.0:
        diff = wp.erfc(-x0) - wp.erfc(-x1)
    else:
        diff = wp.erf(x0) - wp.erf(x1)
    return wp.max(0.0, nu0 * sig * 1.25331414 / sc * diff)


@wp.func
def ramp(y0: float, x0: float, x1: float, tau: float, dt: float):
    if tau <= 0.0:
        return x1
    s = (x1 - x0) / dt
    a = wp.exp(-dt / tau)
    return x1 - s * tau + (y0 - x0 + s * tau) * a


@wp.kernel
def pixel_kernel(pp: PixParams, ta: wp.float64, tb: wp.float64,
                 rate: wp.array(dtype=float), xin: wp.array(dtype=float), L1: wp.array(dtype=float),
                 L2: wp.array(dtype=float), ref: wp.array(dtype=float), rfr: wp.array(dtype=wp.float64),
                 thon_a: wp.array(dtype=float), thoff_a: wp.array(dtype=float), leak_a: wp.array(dtype=float),
                 lab_g: wp.array(dtype=int), lab_d: wp.array(dtype=float), lab_f: wp.array(dtype=wp.vec2),
                 counter: wp.array(dtype=int), evf: wp.array2d(dtype=float), evt: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    dtd = tb - ta
    dt = float(dtd)
    r = wp.max(0.0, rate[i]) + pp.dark
    x1 = wp.log(r)
    x0 = xin[i]
    fc = pp.fmax * r / (r + pp.r_half)
    tau = 1.0 / (2.0 * wp.pi * fc)
    l1a = L1[i]
    l1b = ramp(l1a, x0, x1, tau, dt)
    l2a = l1a
    l2b = l1b
    if pp.tau2 > 0.0:
        l2a = L2[i]
        l2b = ramp(l2a, l1a, l1b, pp.tau2, dt)
    fb = 3.0 * fc
    if pp.sf > 0.0:
        fb = pp.sf
    sig = wp.sqrt(pp.gain * wp.pi * fc * fb / (r * (fc + fb)))
    nu0 = wp.sqrt(fc * fb)
    noisy = pp.shot == 1 and sig > 0.0
    V0 = l2a
    V1 = l2b
    slopeV = (V1 - V0) / dt
    thon = thon_a[i]
    thoff = thoff_a[i]
    lk = leak_a[i] * thon
    tcd = ta                      # absolute times in float64, offsets in float32
    Vc = V0
    refc = ref[i]
    ru = rfr[i]
    pix = wp.uint64(i)
    for guard in range(256):
        if tcd >= tb:
            break
        gu = wp.uint64(guard + 1)
        if ru > tcd:
            if ru >= tb:
                refc = V1
                tcd = tb
                break
            Vc = V0 + slopeV * float(ru - ta)
            tcd = ru
            refc = Vc
            if noisy:
                refc = Vc + sig * gauss(key_of(pp.seed, pix, pp.sample, wp.uint64(300) + gu))
        D0 = Vc - refc
        sD = slopeV + lk
        rem = float(tb - tcd)
        Dend = D0 + sD * rem
        te = rem
        pol = int(-1)
        if Dend >= thon:
            te = float(0.0)
            if D0 < thon and sD > 0.0:
                te = (thon - D0) / sD
            pol = 1
        elif Dend <= -thoff:
            te = float(0.0)
            if D0 > -thoff and sD < 0.0:
                te = (-thoff - D0) / sD
            pol = 0
        te = wp.clamp(te, 0.0, rem)
        noise_ev = int(0)
        if noisy:
            con = thon - D0
            coff = thoff + D0
            Lon = cum_haz(con, sD, te, nu0, sig)
            Loff = cum_haz(coff, -sD, te, nu0, sig)
            E = -wp.log(u01f(key_of(pp.seed, pix, pp.sample, wp.uint64(400) + gu)))
            if E < Lon + Loff:
                lo = float(0.0)
                hi = te
                for it in range(30):
                    mid = 0.5 * (lo + hi)
                    if cum_haz(con, sD, mid, nu0, sig) + cum_haz(coff, -sD, mid, nu0, sig) < E:
                        lo = mid
                    else:
                        hi = mid
                tn = 0.5 * (lo + hi)
                c1 = con - sD * tn
                c2 = coff + sD * tn
                hon = wp.exp(-c1 * c1 / (2.0 * sig * sig))
                hoff = wp.exp(-c2 * c2 / (2.0 * sig * sig))
                pol = 0
                if u01f(key_of(pp.seed, pix, pp.sample, wp.uint64(500) + gu)) * (hon + hoff) < hon:
                    pol = 1
                te = tn
                noise_ev = 1
        if pol < 0:
            refc = refc - lk * rem
            tcd = tb
            break
        ted = tcd + wp.float64(te)
        lat = pp.lat + pp.jit * gauss(key_of(pp.seed, pix, pp.sample, wp.uint64(100) + gu))
        tout = ted + wp.float64(1.0e-6) * wp.float64(wp.max(0.0, lat))
        slot = wp.atomic_add(counter, 0, 1)
        if slot < pp.cap:
            fl = lab_f[i]
            evf[slot, 0] = float(i)          # exact for i < 2^24 (1280x720 = 921600)
            evf[slot, 1] = float(pol)
            evf[slot, 2] = float(lab_g[i])
            evf[slot, 3] = lab_d[i]
            evf[slot, 4] = fl[0]
            evf[slot, 5] = fl[1]
            evt[slot, 0] = tout
            evt[slot, 1] = ted
        ref_at = refc - lk * te
        Vc = V0 + slopeV * float(ted - ta)
        if noise_ev == 1:
            if pol == 1:
                refc = ref_at + thon
            else:
                refc = ref_at - thoff
        else:
            refc = Vc
        tcd = ted
        ru = ted + pp.trefr
        if pp.trefr <= wp.float64(0.0) and sD == 0.0 and noise_ev == 0:
            break
    if pp.hpf > 0.0:
        refc = refc + (V1 - refc) * pp.hpf
    xin[i] = x1
    L1[i] = l1b
    L2[i] = l2b
    ref[i] = refc
    rfr[i] = ru


# ---------------------------------------------------------------- host-side helpers
def _mats2quats(R):
    """Batch rotation matrix (n,3,3) -> unit quaternion (n,4) [w,x,y,z]."""
    R = np.asarray(R, np.float64)
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = np.empty((len(R), 4))
    c0 = t > 0
    s0 = np.sqrt(np.maximum(t + 1.0, 1e-300)) * 2
    q[:, 0] = np.where(c0, 0.25 * s0, 0); q[:, 1] = np.where(c0, (R[:, 2, 1] - R[:, 1, 2]) / s0, 0)
    q[:, 2] = np.where(c0, (R[:, 0, 2] - R[:, 2, 0]) / s0, 0); q[:, 3] = np.where(c0, (R[:, 1, 0] - R[:, 0, 1]) / s0, 0)
    for k in np.nonzero(~c0)[0]:                       # rare branch, per element
        m = R[k]
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s_ = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[k] = [(m[2, 1] - m[1, 2]) / s_, 0.25 * s_, (m[0, 1] + m[1, 0]) / s_, (m[0, 2] + m[2, 0]) / s_]
        elif m[1, 1] > m[2, 2]:
            s_ = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[k] = [(m[0, 2] - m[2, 0]) / s_, (m[0, 1] + m[1, 0]) / s_, 0.25 * s_, (m[1, 2] + m[2, 1]) / s_]
        else:
            s_ = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[k] = [(m[1, 0] - m[0, 1]) / s_, (m[0, 2] + m[2, 0]) / s_, (m[1, 2] + m[2, 1]) / s_, 0.25 * s_]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _quats2mats(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _slerp_mats_multi(A, B, fs):
    """Slerp for several fractions at once: returns (len(fs), n, 3, 3)."""
    qa, qb = _mats2quats(A), _mats2quats(B)
    dot = np.sum(qa * qb, 1)
    qb = np.where(dot[:, None] < 0, -qb, qb); dot = np.abs(dot)
    th = np.arccos(np.clip(dot, -1, 1)); s = np.sin(th)
    lin = dot > 0.9995
    ss = np.where(lin, 1.0, s)
    out = []
    for f in fs:
        wa = np.where(lin, 1 - f, np.sin((1 - f) * th) / ss)
        wb = np.where(lin, f, np.sin(f * th) / ss)
        q = wa[:, None] * qa + wb[:, None] * qb
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        out.append(_quats2mats(q))
    return out


def _slerp_mats(A, B, f):
    """Batch slerp between rotation matrices (n,3,3), fully vectorised."""
    qa, qb = _mats2quats(A), _mats2quats(B)
    dot = np.sum(qa * qb, 1)
    qb = np.where(dot[:, None] < 0, -qb, qb); dot = np.abs(dot)
    th = np.arccos(np.clip(dot, -1, 1)); s = np.sin(th)
    lin = dot > 0.9995
    ss = np.where(lin, 1.0, s)
    wa = np.where(lin, 1 - f, np.sin((1 - f) * th) / ss)
    wb = np.where(lin, f, np.sin(f * th) / ss)
    q = wa[:, None] * qa + wb[:, None] * qb
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return _quats2mats(q)


class EVK4Warp:
    """GPU event camera driven by the same MJCF <plugin> configuration."""

    def __init__(self, model: mujoco.MjModel, sensor_name: str = "evk4", device: str | None = None,
                 tile_cull: bool | None = None):
        """Per-tile ray culling and dirty-tile skipping run on the GPU.  Options via environment:
        EVK4_SHADOW_MESH=hull (convex-hull shadow casters), EVK4_PROFILE=1|2 (timing)."""
        wp.init()
        self.device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
        m = model
        self.m = m
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sid < 0:
            raise KeyError(sensor_name)
        inst = int(m.sensor_plugin[sid])
        self.cfg = c = read_config(m, inst)
        self.cam_id = int(m.sensor_objid[sid])
        self.cambody = int(m.cam_bodyid[self.cam_id])
        b = int(c["binning"])
        res = [int(float(x)) for x in str(c["resolution"]).split()]
        self.W, self.H = res[0] // b, res[1] // b
        self.N = self.W * self.H
        W, H, N = self.W, self.H, self.N
        pitch = c["pixel_pitch_um"] * 1e-6 * b
        if c["fx"] > 0:
            self.fx = c["fx"] / b; self.fy = (c["fy"] if c["fy"] > 0 else c["fx"]) / b
        elif c["focal_length_mm"] > 0:
            self.fx = self.fy = c["focal_length_mm"] * 1e-3 / pitch
        else:
            self.fx = self.fy = 0.5 * H / math.tan(0.5 * math.radians(m.cam_fovy[self.cam_id]))
        self.cx = (c["cx"] + 0.5) / b - 0.5 if c["cx"] >= 0 else 0.5 * (W - 1)
        self.cy = (c["cy"] + 0.5) / b - 0.5 if c["cy"] >= 0 else 0.5 * (H - 1)
        self.dist = np.array([float(x) for x in str(c["distortion"]).split()] + [0] * 5)[:5]
        self.k_rate = math.pi / (4 * c["f_number"] ** 2) * 4.09e15 * pitch * pitch * c["qe"]
        self.r_half = self.k_rate * c["bw_half_lux"] / math.pi
        self.ns = c["spp_n"]
        self.rgbw = np.array([float(x) for x in str(c["rgb_weights"]).split()][:3])
        dev = self.device

        # pixel geometry (undistorted normalised coordinates, OpenCV convention)
        uu, vv = np.meshgrid(np.arange(W + 1) - 0.5, np.arange(H + 1) - 0.5)
        corner = self._undistort((uu - self.cx) / self.fx, (vv - self.cy) / self.fy)
        uc, vc = np.meshgrid(np.arange(W), np.arange(H))
        ctr = self._undistort((uc - self.cx) / self.fx, (vc - self.cy) / self.fy)
        c2 = 1.0 / (1.0 + ctr[..., 0] ** 2 + ctr[..., 1] ** 2)
        self.ctr_np = ctr.reshape(-1, 2)
        self.corner = wp.array(corner.reshape(-1, 2).astype(np.float32), dtype=wp.vec2, device=dev)
        self.ctr = wp.array(self.ctr_np.astype(np.float32), dtype=wp.vec2, device=dev)
        self.cos4 = wp.array((c2 * c2).reshape(-1).astype(np.float32), device=dev)

        # fixed-pattern noise (same hashes as the C++ plugin)
        seed = int(c["seed"]); pix = np.arange(N, dtype=np.uint64)
        thon = np.maximum(0.03, c["eff_con"] + c["contrast_sigma"] * np_gauss(np_key(seed, pix, 0, 11)))
        thoff = np.maximum(0.03, c["eff_coff"] + c["contrast_sigma"] * np_gauss(np_key(seed, pix, 0, 12)))
        s = c["leak_sigma"]
        lk = c["eff_leak"] * np.exp(s * np_gauss(np_key(seed, pix, 0, 13)) - 0.5 * s * s)
        lk = np.where(np_u01(np_key(seed, pix, 0, 14)) < c["hot_pixel_fraction"], c["hot_pixel_rate_hz"], lk)
        self.thon_np, self.thoff_np = thon, thoff
        self.thon = wp.array(thon.astype(np.float32), device=dev)
        self.thoff = wp.array(thoff.astype(np.float32), device=dev)
        self.leak = wp.array(lk.astype(np.float32), device=dev)

        self._build_scene()
        # state
        self.xin = wp.zeros(N, dtype=float, device=dev); self.L1 = wp.zeros_like(self.xin)
        self.L2 = wp.zeros_like(self.xin); self.ref = wp.zeros_like(self.xin)
        self.rfr = wp.zeros(N, dtype=wp.float64, device=dev)
        self.raw = wp.zeros_like(self.xin); self.tmp = wp.zeros_like(self.xin); self.rate = wp.zeros_like(self.xin)
        self.lab_g = wp.zeros(N, dtype=int, device=dev); self.lab_d = wp.zeros_like(self.xin)
        self.lab_f = wp.zeros(N, dtype=wp.vec2, device=dev)
        self.cap = max(4 * N, 1 << 20)
        self.counter = wp.zeros(1, dtype=int, device=dev)
        self.evf = wp.zeros((self.cap, 6), dtype=float, device=dev)
        self.evt = wp.zeros((self.cap, 2), dtype=wp.float64, device=dev)
        self.sort_k = wp.zeros(2 * self.cap, dtype=wp.int64, device=dev)     # radix sort needs 2x
        self.sort_v = wp.zeros(2 * self.cap, dtype=wp.int32, device=dev)
        self.evs = wp.zeros((self.cap, 6), dtype=float, device=dev)
        # screen tiles (16x16) for primary-ray culling
        self.TW, self.TH = (W + 15) // 16, (H + 15) // 16
        T = self.TW * self.TH
        cg = corner.reshape(H + 1, W + 1, 2)
        tb = np.zeros((T, 4))
        for ty in range(self.TH):
            for tx in range(self.TW):
                blk = cg[ty * 16:min(H, ty * 16 + 16) + 1, tx * 16:min(W, tx * 16 + 16) + 1].reshape(-1, 2)
                tb[ty * self.TW + tx] = [blk[:, 0].min(), blk[:, 1].min(), blk[:, 0].max(), blk[:, 1].max()]
        self.tile_bounds = tb
        G = max(1, len(self.vis_np))
        self.G = G
        self.tb_wp = wp.array(tb.astype(np.float32), dtype=wp.vec4, device=dev)
        self.tile_cnt = wp.zeros(T, dtype=int, device=dev)
        self.tile_list = wp.zeros(T * G, dtype=int, device=dev)
        self.tile_dirty = wp.ones(T, dtype=int, device=dev)
        self.tile_valid = wp.zeros(T, dtype=int, device=dev)
        self.tile_hit = wp.zeros((T, 6), dtype=float, device=dev)
        # split renderer (v2.5): hit buffer + per-(tile, light) shadow-occluder lists
        self.split = os.environ.get("EVK4_RENDER", "split").lower() != "fused"
        self._bd = int(os.environ.get("EVK4_BLOCK", "128") or 128)          # threads per block (tuning knob)
        nss = self.ns * self.ns
        self.hit_t = wp.zeros(N * nss, dtype=float, device=dev)
        self.hit_g = wp.zeros(N * nss, dtype=int, device=dev)
        self.hit_n = wp.zeros(N * nss, dtype=wp.vec3, device=dev)
        self.hit_h = wp.zeros(N * nss, dtype=int, device=dev)
        self.hit_rho = wp.zeros(N * nss, dtype=float, device=dev)
        self.nlw = (max(1, self.m.nlight) + 31) // 32
        self.occ = wp.zeros((N * nss, self.nlw), dtype=int, device=dev)
        NL = max(1, self.m.nlight)
        self.sh_cnt = wp.zeros(T * NL, dtype=int, device=dev)
        self.sh_list = wp.zeros(T * NL * G, dtype=int, device=dev)
        self.gidx_wp = wp.array(self.vis_np.astype(np.int32) if len(self.vis_np) else np.zeros(1, np.int32), dtype=int, device=dev)
        self.skip_tiles = int(float(c["skip_static_tiles"])) == 1
        self._last = None                                     # poses of the last render
        self.probe_np = np.zeros((0, 3))
        self.probe_every = 10
        self._steps = 0
        self._buf = 0
        self._rec_pix = None                                  # recorded pixel-kernel launch
        self.probe = wp.zeros(PGX * PGY, dtype=wp.vec4, device=dev)
        r = max(1, int(math.ceil(3 * c["psf_sigma_px"])))
        kk = np.exp(-0.5 * np.arange(-r, r + 1) ** 2 / max(1e-9, c["psf_sigma_px"]) ** 2)
        self.blur_r = r; self.blur_k = wp.array((kk / kk.sum()).astype(np.float32), device=dev)
        # host-side readout state
        self.last_t = np.full(N, -1.0); self.last_p = np.full(N, -1, np.int8)
        self.afk_t = np.zeros(N); self.afk_per = np.zeros(N); self.afk_until = np.zeros(N)
        self.initialized = False
        self.t_prev = 0.0; self.sample = 1
        self.pending = np.zeros(0, LABELED_DTYPE)
        self.arb_busy = 0.0
        self.erc = dict(wstart=0.0, count=0.0, keep=1.0, acc=0.0, out=0.0)
        self._events = np.zeros(0, LABELED_DTYPE)
        self.dropped_total = 0
        self.nsub = 0
        self.prev_gpos = self.prev_gmat = None

    # ------------------------------------------------------------------ setup
    def _undistort(self, xd, yd):
        k1, k2, p1, p2, k3 = self.dist
        x, y = xd.copy(), yd.copy()
        if np.any(self.dist != 0):
            for _ in range(20):
                r2 = x * x + y * y
                rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
                dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
                dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
                x, y = (xd - dx) / rad, (yd - dy) / rad
        return np.stack([x, y], -1)

    def _project_px(self, cpos, cmat, w):
        c = (w - cpos) @ cmat                              # R^T (w - p)
        z = -c[:, 2]
        ok = z > 1e-4
        z = np.where(ok, z, 1.0)
        x, y = c[:, 0] / z, -c[:, 1] / z
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([self.cx + self.fx * xd, self.cy + self.fy * yd], 1), ok

    def _build_scene(self):
        m, c, dev = self.m, self.cfg, self.device
        gg = [int(float(x)) for x in str(c["geomgroup"]).split()] + [0] * 6
        vis = []
        for g in range(m.ngeom):
            grp = int(m.geom_group[g])
            if grp < 0 or grp >= 6 or not gg[grp]:
                continue
            if m.geom_bodyid[g] == self.cambody:
                continue
            mat = m.geom_matid[g]
            alpha = m.mat_rgba[mat, 3] if mat >= 0 else m.geom_rgba[g, 3]
            if m.geom_rgba[g, 3] == 0:
                alpha = 0
            t = m.geom_type[g]
            if alpha <= 0 or t in (mujoco.mjtGeom.mjGEOM_HFIELD, mujoco.mjtGeom.mjGEOM_SDF):
                continue
            if t == mujoco.mjtGeom.mjGEOM_MESH and m.geom_dataid[g] < 0:
                continue
            vis.append(g)
        self.vis_np = np.array(vis, np.int32)
        self.vis = wp.array(self.vis_np, dtype=int, device=dev)
        self.gtype = wp.array(m.geom_type.astype(np.int32), dtype=int, device=dev)
        self.gsize = wp.array(m.geom_size.astype(np.float32), dtype=wp.vec3, device=dev)
        self.grb = wp.array(m.geom_rbound.astype(np.float32), device=dev)
        self.static_np = (m.body_weldid[m.geom_bodyid] == 0).astype(np.int32)
        self.static_g = wp.array(self.static_np, dtype=int, device=dev)
        # materials
        rgb, spec, shin, emis, gtex, rep, uni = self._materials()
        self.grgb = wp.array(rgb, dtype=wp.vec3, device=dev)
        self.gspec = wp.array(spec, device=dev); self.gshin = wp.array(shin, device=dev)
        self.gemis = wp.array(emis, device=dev); self.gtex = wp.array(gtex, dtype=int, device=dev)
        self.grep = wp.array(rep, dtype=wp.vec2, device=dev); self.guni = wp.array(uni, dtype=int, device=dev)
        self._app_uploaded = self._appearance_sig()
        self._build_textures_and_meshes()

    def _materials(self):
        m = self.m
        rgb = m.geom_rgba[:, :3].astype(np.float32).copy()
        spec = np.full(m.ngeom, 0.5, np.float32); shin = np.full(m.ngeom, 0.5, np.float32)
        emis = np.zeros(m.ngeom, np.float32); gtex = np.full(m.ngeom, -1, np.int32)
        rep = np.ones((m.ngeom, 2), np.float32); uni = np.zeros(m.ngeom, np.int32)
        for g in range(m.ngeom):
            mat = m.geom_matid[g]
            if mat < 0:
                continue
            if np.all(np.abs(m.geom_rgba[g, :3] - 0.5) < 1e-6):
                rgb[g] = m.mat_rgba[mat, :3]
            spec[g] = m.mat_specular[mat]; shin[g] = m.mat_shininess[mat]; emis[g] = m.mat_emission[mat]
            tid = m.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_RGB]
            if tid < 0:
                tid = m.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_RGBA]
            if tid >= 0 and m.tex_type[tid] == mujoco.mjtTexture.mjTEXTURE_2D:
                gtex[g] = tid
            r0, r1 = m.mat_texrepeat[mat]
            rep[g] = (r0 if r0 > 0 else 1, r1 if r1 > 0 else 1)
            uni[g] = int(m.mat_texuniform[mat])
        return rgb, spec, shin, emis, gtex, rep, uni

    def _refresh_appearance(self):
        """geom_rgba / geom_size / material edits made at run time (same arrays, updated in place)."""
        m = self.m
        rgb, spec, shin, emis, gtex, rep, uni = self._materials()
        self.grgb.assign(rgb); self.gspec.assign(spec); self.gshin.assign(shin); self.gemis.assign(emis)
        self.gtex.assign(gtex); self.grep.assign(rep); self.guni.assign(uni)
        self.gsize.assign(m.geom_size.astype(np.float32)); self.grb.assign(m.geom_rbound.astype(np.float32))
        self._app_uploaded = self._appearance_sig()

    def _build_textures_and_meshes(self):
        m, dev = self.m, self.device
        # textures (float rgb)
        offs, ws, hs, chunks, o = [], [], [], [], 0
        for t in range(m.ntex):
            w, h, nc = int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t])
            data = np.asarray(m.tex_data[m.tex_adr[t]:m.tex_adr[t] + w * h * nc]).reshape(h * w, nc)
            if m.tex_type[t] != mujoco.mjtTexture.mjTEXTURE_2D:
                data = np.zeros((1, 3), np.uint8); w = h = 1
            elif nc < 3:
                data = np.repeat(data[:, :1], 3, 1)
            offs.append(o); ws.append(w); hs.append(h); chunks.append(data[:, :3].astype(np.float32) / 255.0)
            o += len(chunks[-1])
        if not chunks:
            chunks = [np.zeros((1, 3), np.float32)]; offs = ws = hs = [0]
        self.tex_off = wp.array(np.array(offs, np.int32), dtype=int, device=dev)
        self.tex_w = wp.array(np.array(ws, np.int32), dtype=int, device=dev)
        self.tex_h = wp.array(np.array(hs, np.int32), dtype=int, device=dev)
        self.texdata = wp.array(np.concatenate(chunks), dtype=wp.vec3, device=dev)
        # meshes
        self.meshes, ids = [], []
        fn_off, fn, n_off, nr, ft_off, ft, t_off, tc = [], [], [], [], [], [], [], []
        gmesh = np.full(m.ngeom, -1, np.int32)
        nfn = nnr = nft = ntc = 0
        for mi in range(m.nmesh):
            va, vn = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
            fa, fnum = m.mesh_faceadr[mi], m.mesh_facenum[mi]
            pts = wp.array(m.mesh_vert[va:va + vn].astype(np.float32), dtype=wp.vec3, device=dev)
            idx = wp.array(m.mesh_face[fa:fa + fnum].astype(np.int32).reshape(-1), dtype=wp.int32, device=dev)
            mesh = wp.Mesh(points=pts, indices=idx)
            self.meshes.append(mesh); ids.append(mesh.id)
            fn_off.append(nfn); fn.append(m.mesh_facenormal[fa:fa + fnum]); nfn += fnum
            na, nn = m.mesh_normaladr[mi], m.mesh_normalnum[mi]
            n_off.append(nnr); nr.append(m.mesh_normal[na:na + nn]); nnr += nn
            ta = m.mesh_texcoordadr[mi]
            ft_off.append(nft); ft.append(m.mesh_facetexcoord[fa:fa + fnum]); nft += fnum
            if ta >= 0:
                tn = m.mesh_texcoordnum[mi]
                t_off.append(ntc); tc.append(m.mesh_texcoord[ta:ta + tn]); ntc += tn
            else:
                t_off.append(-1)
        for g in range(m.ngeom):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                gmesh[g] = m.geom_dataid[g]
        z3i = np.zeros((1, 3), np.int32); z3 = np.zeros((1, 3), np.float32); z2 = np.zeros((1, 2), np.float32)
        cat = lambda L, z, dt: np.concatenate(L).astype(dt) if L else z
        self.mesh_ids = wp.array(np.array(ids or [0], np.uint64), dtype=wp.uint64, device=dev)
        # shadow rays: full meshes (exact, default) or convex hulls (EVK4_SHADOW_MESH=hull, faster)
        self.shadow_hull = os.environ.get("EVK4_SHADOW_MESH", "full").lower() == "hull"
        sids = list(ids)
        if self.shadow_hull and m.nmesh:
            try:
                from scipy.spatial import ConvexHull
                self.hull_meshes = []
                for mi in range(m.nmesh):
                    va, vn = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
                    V = m.mesh_vert[va:va + vn].astype(np.float64)
                    h = ConvexHull(V)
                    hm = wp.Mesh(points=wp.array(V.astype(np.float32), dtype=wp.vec3, device=dev),
                                 indices=wp.array(h.simplices.astype(np.int32).reshape(-1), dtype=wp.int32, device=dev))
                    self.hull_meshes.append(hm); sids[mi] = hm.id
            except Exception as ex:                     # scipy missing or degenerate mesh
                print("evk4: convex-hull shadows unavailable:", ex)
        self.shadow_ids = wp.array(np.array(sids or [0], np.uint64), dtype=wp.uint64, device=dev)
        self.gmesh = wp.array(gmesh, dtype=int, device=dev)
        self.mfn_off = wp.array(np.array(fn_off or [0], np.int32), dtype=int, device=dev)
        self.mfacenrm = wp.array(cat(fn, z3i, np.int32), dtype=wp.vec3i, device=dev)
        self.mnrm_off = wp.array(np.array(n_off or [0], np.int32), dtype=int, device=dev)
        self.mnrm = wp.array(cat(nr, z3, np.float32), dtype=wp.vec3, device=dev)
        self.mft_off = wp.array(np.array(ft_off or [0], np.int32), dtype=int, device=dev)
        self.mfacetex = wp.array(cat(ft, z3i, np.int32), dtype=wp.vec3i, device=dev)
        self.mtex_off = wp.array(np.array(t_off or [-1], np.int32), dtype=int, device=dev)
        self.mtex = wp.array(cat(tc, z2, np.float32), dtype=wp.vec2, device=dev)
        # lights (static properties; recomputed each step in case they are edited)
        self._upload_lights_static()

    def _upload_lights_static(self):
        m, c, dev = self.m, self.cfg, self.device
        wn = self.rgbw.sum()
        lum = lambda a: (a @ self.rgbw) / wn * c["lux_per_unit"]
        if hasattr(self, "lactive"):                       # run-time change: update the same arrays
            self.lamb.assign(lum(m.light_ambient).astype(np.float32))
            self.ldif.assign(lum(m.light_diffuse).astype(np.float32))
            self.lspe.assign(lum(m.light_specular).astype(np.float32))
            self.lcut.assign(np.cos(np.radians(m.light_cutoff)).astype(np.float32))
            self.lactive.assign(m.light_active.astype(np.int32))
            self._light_sig = self._light_signature()
            return
        self.ltype = wp.array(m.light_type.astype(np.int32), dtype=int, device=dev)
        self.lamb = wp.array(lum(m.light_ambient).astype(np.float32), device=dev)
        self.ldif = wp.array(lum(m.light_diffuse).astype(np.float32), device=dev)
        self.lspe = wp.array(lum(m.light_specular).astype(np.float32), device=dev)
        self.latt = wp.array(m.light_attenuation.astype(np.float32), dtype=wp.vec3, device=dev)
        self.lcut = wp.array(np.cos(np.radians(m.light_cutoff)).astype(np.float32), device=dev)
        self.lexp = wp.array(m.light_exponent.astype(np.float32), device=dev)
        self.lshadow = wp.array(m.light_castshadow.astype(np.int32), dtype=int, device=dev)
        self.lactive = wp.array(m.light_active.astype(np.int32), dtype=int, device=dev)
        self._light_sig = self._light_signature()

    def _appearance_sig(self):
        m = self.m
        return hash((m.geom_rgba.tobytes(), m.geom_size.tobytes(), m.mat_rgba.tobytes(), m.geom_matid.tobytes(),
                     self._light_signature()))

    def _light_signature(self):
        m = self.m
        return np.concatenate([m.light_diffuse.ravel(), m.light_ambient.ravel(), m.light_specular.ravel(),
                               m.light_active.ravel().astype(float), m.light_cutoff.ravel()]).tobytes()

    def _cam(self, pos, mat):
        k = self.dist
        return self._mkcam(pos, mat, k)

    def _mkcam(self, pos, mat, k):
        cam = Cam()
        cam.pos = wp.vec3(*pos.astype(np.float32))
        cam.R = wp.mat33(*mat.reshape(9).astype(np.float32))
        cam.fx, cam.fy, cam.cx, cam.cy = float(self.fx), float(self.fy), float(self.cx), float(self.cy)
        cam.k1, cam.k2, cam.p1, cam.p2, cam.k3 = [float(x) for x in k]
        return cam

    # ------------------------------------------------------------------ rendering
    def _rects(self, gidx, gpos, gmat, cpos, cmat):
        """Screen bounds (normalised coords) of geoms: status 0 hidden, 1 rect, 2 always test."""
        m = self.m
        n = len(gidx)
        stat = np.zeros(n, np.int32); rect = np.full((n, 4), 1e30)
        if n == 0:
            return stat, rect
        typ = m.geom_type[gidx]
        rb = m.geom_rbound[gidx]
        c = (gpos[gidx] - cpos) @ cmat
        z = -c[:, 2]
        front = z + rb >= 1e-3
        straddle = front & (z - rb < 1e-3)
        ab = m.geom_aabb[gidx]
        sg = np.array([[(k & 1) * 2 - 1, ((k >> 1) & 1) * 2 - 1, ((k >> 2) & 1) * 2 - 1] for k in range(8)])
        loc = ab[:, None, :3] + sg[None] * ab[:, None, 3:]
        w = np.einsum("nij,nkj->nki", gmat[gidx].reshape(-1, 3, 3), loc) + gpos[gidx][:, None]
        cc = (w - cpos) @ cmat
        zz = -cc[..., 2]
        allfront = np.all(zz > 1e-4, axis=1)
        zs = np.where(zz > 1e-4, zz, 1.0)
        x, y = cc[..., 0] / zs, -cc[..., 1] / zs
        r = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], 1)
        zc = np.where(z - rb > 1e-6, z, 1.0)
        xc, yc = c[:, 0] / zc, -c[:, 1] / zc
        rr = rb / np.sqrt(np.maximum(zc * zc - rb * rb, 1e-12)) * (1 + xc * xc + yc * yc)
        rs = np.stack([xc - rr, yc - rr, xc + rr, yc + rr], 1)
        r = np.where(allfront[:, None], r, rs) + np.array([-1, -1, 1, 1]) * (3.0 / self.fx)
        stat[front] = 1
        stat[straddle | (typ == 0)] = 2
        rect[stat == 1] = r[stat == 1]
        return stat, rect

    # ------------------------------------------------------------------ low-overhead plumbing (v2.6)
    def _init_stage(self):
        """All per-render host data (poses, screen rects, dirty info, lights) lives in ONE float32
        block per render: filled in a pinned host buffer, sent with ONE async copy into one of two
        device slots (slot k-1 keeps the previous poses for the optical-flow labels)."""
        dev, m = self.device, self.m
        ng, G, nl = m.ngeom, self.G, max(1, m.nlight)
        al = lambda x: (x + 3) // 4 * 4
        lay, o = {}, 0
        for name, n, w in (("gp", ng, 3), ("gm", ng, 9), ("st", G, 1), ("rc", G, 4), ("mr", 2 * G, 4),
                           ("ms", 2 * G, 4), ("lp", nl, 3), ("ld", nl, 3)):
            lay[name] = (o, n, w); o = al(o + n * w)
        self._lay, self._S = lay, o
        dt = {"gp": wp.vec3, "gm": wp.mat33, "st": float, "rc": wp.vec4, "mr": wp.vec4, "ms": wp.vec4,
              "lp": wp.vec3, "ld": wp.vec3}
        self._stg_d = [wp.zeros(o, dtype=float, device=dev) for _ in range(2)]
        self._sv = [{k: wp.array(ptr=b.ptr + 4 * off, dtype=dt[k], shape=(n,), device=dev)
                     for k, (off, n, w) in lay.items()} for b in self._stg_d]
        self._pinned = wp.get_device(dev).is_cuda
        nh = int(self.cfg["max_substeps"]) + 2                  # > renders per physics step
        self._stg_h = [wp.empty(o, dtype=float, device="cpu", pinned=self._pinned) for _ in range(nh)]
        self._stg_hn = [a.numpy() for a in self._stg_h]
        self._hslot = 0
        self._rec = [None, None]
        self._lsig = None

    def _hview(self, h, k):
        off, n, w = self._lay[k]
        v = h[off:off + n * w]
        return v.reshape(n, w) if w > 1 else v

    def _record(self, slot):
        """Record every render launch once per device slot; afterwards only the few per-render
        scalars / structs are updated (a recorded launch costs ~1/4 of a wp.launch from Python)."""
        c, dev, sv, pv = self.cfg, self.device, self._sv[slot], self._sv[1 - slot]
        T = len(self.tile_bounds); nl = max(1, self.m.nlight); sc = self._scene
        cam = self._cam(np.zeros(3), np.eye(3)); bd = self._bd
        r = {}
        rec = lambda k, dim, inputs, **kw: wp.launch(k, dim=dim, inputs=inputs, device=dev, record_cmd=True, **kw)
        r["tile"] = rec(tile_kernel, T, [
            self.G, len(self.vis_np), self.tb_wp, self.gidx_wp, sv["st"], sv["rc"], 1,
            self.tile_valid, 0, sv["mr"], sv["ms"], self.m.nlight, self.ltype, sv["lp"], sv["ld"],
            self.lshadow, self.lactive, int(c["shadows"]), self.tile_hit, self.tile_cnt, self.tile_list,
            self.tile_dirty])
        r["hit"] = rec(hit_kernel, T * 256, [
            self.W, self.H, self.ns, self.TW, self.corner, self.ctr, cam,
            self.G, self.tile_cnt, self.tile_list, self.tile_dirty,
            self.gtype, self.gsize, self.grb, sv["gp"], sv["gm"], self.gmesh, self.mesh_ids,
            self.hit_t, self.hit_g, self.hit_n, self.hit_h, self.lab_g, self.lab_d], block_dim=bd)
        r["flow"] = rec(flow_kernel, self.N, [
            self.W, self.ns, self.TW, self.corner, self.ctr, cam, cam, 1.0, 0,
            self.tile_dirty, sv["gp"], sv["gm"], pv["gp"], pv["gm"], self.lab_g, self.lab_d, self.lab_f])
        r["tilebound"] = rec(tilebound_kernel, T, [
            self.W, self.H, self.ns, self.TW, cam, self.corner, self.ctr, self.tile_dirty,
            self.hit_t, self.hit_g, self.tile_hit])
        if self.m.nlight:
            r["cull"] = rec(shadowcull_kernel, T * nl, [
                self.m.nlight, sc.nvis, int(c["shadows"]), self.tile_dirty, self.tile_hit, self.vis,
                self.gtype, sv["gp"], self.grb, self.ltype, sv["lp"], sv["ld"], self.lshadow, self.lactive,
                self.sh_cnt, self.sh_list])
        r["surf"] = rec(surf_kernel, T * 256, [
            self.W, self.H, self.ns, self.TW, self.corner, self.ctr, cam, self.tile_dirty,
            self.hit_t, self.hit_g, self.hit_n, self.hit_h, self.hit_rho,
            self.gtype, self.gsize, sv["gp"], sv["gm"], self.gmesh, self.mesh_ids,
            self.mfn_off, self.mfacenrm, self.mnrm_off, self.mnrm, self.mft_off, self.mfacetex,
            self.mtex_off, self.mtex, self.grgb, self.gtex, self.grep, self.guni, self._rgbw_wp, sc.wn,
            self.tex_off, self.tex_w, self.tex_h, self.texdata], block_dim=bd)
        self._shadowed = int(c["shadows"]) == 1 and self.m.nlight > 0 and bool(np.any(self.m.light_castshadow))
        if self._shadowed:
            r["shadow"] = rec(shadow_kernel, T * 256, [
                self.W, self.H, self.ns, self.TW, self.nlw, self.corner, self.ctr, cam, sc, self.tile_dirty,
                self.hit_t, self.hit_g, self.hit_n, self.sh_cnt, self.sh_list,
                self.gtype, self.gsize, self.grb, sv["gp"], sv["gm"], self.gmesh, self.shadow_ids,
                self.ltype, sv["lp"], sv["ld"], self.ldif, self.lspe, self.latt, self.lcut, self.lexp,
                self.lshadow, self.lactive, self.occ], block_dim=bd)
        r["light"] = rec(light_kernel, T * 256, [
            self.W, self.H, self.ns, self.TW, int(self._shadowed), self.corner, self.ctr, self.cos4, cam, sc,
            self.tile_dirty, self.hit_t, self.hit_g, self.hit_n, self.hit_rho, self.occ,
            self.gspec, self.gshin, self.gemis, self.ltype, sv["lp"], sv["ld"], self.lamb, self.ldif,
            self.lspe, self.latt, self.lcut, self.lexp, self.lactive, self.raw], block_dim=bd)
        if c["psf_sigma_px"] > 0.05:
            r["blur_h"] = rec(blur_h, self.N, [self.raw, self.tmp, self.blur_k, self.blur_r, self.W])
            r["blur_v"] = rec(blur_v, self.N, [self.tmp, self.rate, self.blur_k, self.blur_r, self.W, self.H])
        self._rec[slot] = r
        return r

    def _render(self, gpos, gmat, cpos, cmat, t, pcpos=None, pcmat=None, dtflow=0.0, flow=False, probe=False):
        """Render one sample.  Poses of the previous render stay in the other device slot and are
        used for the exact optical-flow labels (no re-upload)."""
        c, dev = self.cfg, self.device
        prof = self._prof
        t0 = time.perf_counter() if prof else 0
        if self._light_signature() != self._light_sig:
            self._upload_lights_static()
        app = self._appearance_sig()
        if app != self._app_uploaded:
            self._refresh_appearance()
        sc = self._scene
        fl = 1.0
        if c["flicker_hz"] > 0 and c["flicker_depth"] > 0:
            fl = 1.0 - c["flicker_depth"] * 0.5 * (1.0 + math.cos(2 * math.pi * c["flicker_hz"] * t))
        sc.flick = float(fl)
        if not hasattr(self, "_stg_d"):
            self._init_stage()
        cur = self._buf
        self._buf = 1 - cur                               # next render writes the other slot
        h = self._stg_hn[self._hslot]; hs_ = self._stg_h[self._hslot]
        self._hslot = (self._hslot + 1) % len(self._stg_h)
        cam = self._cam(cpos, cmat)
        pcam = self._cam(pcpos, pcmat) if flow else cam
        hv = lambda k: self._hview(h, k)
        hv("gp")[:] = gpos; hv("gm")[:] = gmat.reshape(-1, 9)
        if self.m.nlight:
            hv("lp")[:] = self._d.light_xpos; hv("ld")[:] = self._d.light_xdir
        # ---- per-tile candidates + dirty flags (computed on the GPU from a few host-side bounds)
        vis = self.vis_np
        stat, rect = self._rects(vis, gpos, gmat.reshape(-1, 9), cpos, cmat)
        hv("st")[:len(vis)] = stat; hv("rc")[:len(vis)] = rect
        last = self._last
        TOL = 1e-10
        same = lambda a, b: np.max(np.abs(a - b)) <= TOL if np.size(a) else True
        all_dirty = (not self.skip_tiles) or last is None or fl != 1.0 or app != last["app"] \
            or not same(cpos, last["cpos"]) or not same(cmat, last["cmat"]) \
            or not same(self._d.light_xpos, last["lpos"]) or not same(self._d.light_xdir, last["ldir"])
        nmov = 0
        if not all_dirty:
            mv = vis[np.any(np.abs(gpos[vis] - last["gpos"][vis]) > TOL, 1) |
                     np.any(np.abs(gmat.reshape(-1, 9)[vis] - last["gmat"][vis]) > TOL, 1)]
            if len(mv):
                if np.any(self.m.geom_type[mv] == 0):
                    all_dirty = True
                else:
                    s1, r1 = self._rects(mv, gpos, gmat.reshape(-1, 9), cpos, cmat)
                    s0, r0 = self._rects(mv, last["gpos"], last["gmat"], cpos, cmat)
                    if np.any(s1 == 2) or np.any(s0 == 2):
                        all_dirty = True
                    else:
                        nmov = 2 * len(mv)
                        hv("mr")[:nmov] = np.concatenate([r1, r0])
                        hv("ms")[:nmov] = np.concatenate([np.c_[gpos[mv], self.m.geom_rbound[mv]],
                                                          np.c_[last["gpos"][mv], self.m.geom_rbound[mv]]])
        self._last = dict(gpos=gpos.copy(), gmat=gmat.reshape(-1, 9).copy(), cpos=cpos.copy(), cmat=cmat.copy(),
                          lpos=self._d.light_xpos.copy(), ldir=self._d.light_xdir.copy(), app=app)
        wp.copy(self._stg_d[cur], hs_)                    # ONE upload per render (async from pinned memory)
        sv, pv = self._sv[cur], self._sv[1 - cur]
        if self.split:
            r = self._rec[cur] or self._record(cur)
            L = r["tile"]; L.set_param_by_name("all_dirty", int(all_dirty)); L.set_param_by_name("nmov", nmov); L.launch()
            if prof:
                self._pt["host"] += time.perf_counter() - t0; wp.synchronize(); t0 = time.perf_counter()
            L = r["hit"]; L.set_param_by_name("cam", cam); L.launch()
            L = r["flow"]; L.set_param_by_name("cam", cam); L.set_param_by_name("pcam", pcam)
            L.set_param_by_name("dtflow", float(max(dtflow, 1e-12))); L.set_param_by_name("do_flow", int(flow)); L.launch()
            L = r["tilebound"]; L.set_param_by_name("cam", cam); L.launch()
            if "cull" in r:
                r["cull"].launch()
            L = r["surf"]; L.set_param_by_name("cam", cam); L.launch()
            if "shadow" in r:
                L = r["shadow"]; L.set_param_by_name("cam", cam); L.set_param_by_name("sc", sc); L.launch()
            L = r["light"]; L.set_param_by_name("cam", cam); L.set_param_by_name("sc", sc); L.launch()
            if "blur_h" in r:
                r["blur_h"].launch(); r["blur_v"].launch()
            else:
                wp.copy(self.rate, self.raw)
        else:                                             # v2.4 fused kernel (reference / A-B)
            wp.launch(tile_kernel, dim=len(self.tile_bounds), device=dev, inputs=[
                self.G, len(vis), self.tb_wp, self.gidx_wp, sv["st"], sv["rc"], int(all_dirty),
                self.tile_valid, nmov, sv["mr"], sv["ms"], self.m.nlight, self.ltype, sv["lp"], sv["ld"],
                self.lshadow, self.lactive, int(c["shadows"]), self.tile_hit, self.tile_cnt, self.tile_list,
                self.tile_dirty])
            if prof:
                self._pt["host"] += time.perf_counter() - t0; wp.synchronize(); t0 = time.perf_counter()
            pp_, pm_ = (pv["gp"], pv["gm"]) if flow else (sv["gp"], sv["gm"])
            wp.launch(render_kernel, dim=self.N, device=dev, inputs=[
                self.W, self.H, self.ns, self.corner, self.ctr, self.cos4, cam, pcam, float(max(dtflow, 1e-12)),
                int(flow), sc, self.TW, self.G, self.tile_cnt, self.tile_list, self.tile_dirty, self.tile_hit,
                self.vis, self.gtype, self.gsize, self.grb, sv["gp"], sv["gm"], pp_, pm_, self.gmesh,
                self.mesh_ids, self.mfn_off, self.mfacenrm, self.mnrm_off, self.mnrm, self.mft_off,
                self.mfacetex, self.mtex_off, self.mtex, self.grgb, self.gspec, self.gshin, self.gemis,
                self.gtex, self.grep, self.guni, self._rgbw_wp, self.tex_off,
                self.tex_w, self.tex_h, self.texdata, self.ltype, sv["lp"], sv["ld"], self.lamb, self.ldif,
                self.lspe, self.latt, self.lcut, self.lexp, self.lshadow, self.lactive, self.shadow_ids,
                self.raw, self.lab_g, self.lab_d, self.lab_f])
            if c["psf_sigma_px"] > 0.05:
                wp.launch(blur_h, dim=self.N, inputs=[self.raw, self.tmp, self.blur_k, self.blur_r, self.W], device=dev)
                wp.launch(blur_v, dim=self.N, inputs=[self.tmp, self.rate, self.blur_k, self.blur_r, self.W, self.H], device=dev)
            else:
                wp.copy(self.rate, self.raw)
        if probe:
            wp.launch(probe_kernel, dim=PGX * PGY, device=dev, inputs=[
                self.W, self.H, max(1, self.W // PGX), max(1, self.H // PGY), self.ctr, cam, self.lab_g,
                self.lab_d, self.static_g, self.probe])
            pr = self.probe.numpy()
            self.probe_np = pr[pr[:, 3] > 0, :3].astype(np.float64)
        if prof:
            wp.synchronize(); self._pt["render"] += time.perf_counter() - t0

    def _motion_substeps(self, gpos, gmat, cpos, cmat, dt):
        m = self.m
        mov = [g for g in self.vis_np if m.geom_type[g] != 0 and not self.static_np[g]]
        maxd = 0.0
        if mov:
            mov = np.array(mov)
            ab = m.geom_aabb[mov]                                        # (n,6)
            sg = np.array([[(c & 1) * 2 - 1, ((c >> 1) & 1) * 2 - 1, ((c >> 2) & 1) * 2 - 1] for c in range(8)])
            loc = ab[:, None, :3] + sg[None] * ab[:, None, 3:]           # (n,8,3)
            w0 = np.einsum("nij,nkj->nki", self.prev_gmat[mov].reshape(-1, 3, 3), loc) + self.prev_gpos[mov][:, None]
            w1 = np.einsum("nij,nkj->nki", gmat[mov].reshape(-1, 3, 3), loc) + gpos[mov][:, None]
            p0, ok0 = self._project_px(self.prev_cpos, self.prev_cmat, w0.reshape(-1, 3))
            p1, ok1 = self._project_px(cpos, cmat, w1.reshape(-1, 3))
            inb = lambda p: (p[:, 0] > -50) & (p[:, 0] < self.W + 50) & (p[:, 1] > -50) & (p[:, 1] < self.H + 50)
            sel = ok0 & ok1 & (inb(p0) | inb(p1))
            if sel.any():
                maxd = max(maxd, np.max(np.linalg.norm(p1[sel] - p0[sel], axis=1)))
        pr = self.probe_np
        if len(pr):
            p0, ok0 = self._project_px(self.prev_cpos, self.prev_cmat, pr)
            p1, ok1 = self._project_px(cpos, cmat, pr)
            sel = ok0 & ok1
            if sel.any():
                maxd = max(maxd, np.max(np.linalg.norm(p1[sel] - p0[sel], axis=1)))
        n = int(math.ceil(maxd / max(0.05, self.cfg["max_px_motion"])))
        if self.cfg["flicker_hz"] > 0 and self.cfg["flicker_depth"] > 0:
            n = max(n, int(math.ceil(dt * self.cfg["flicker_hz"] * 16)))
        return max(1, min(int(self.cfg["max_substeps"]), n))

    # ------------------------------------------------------------------ main update
    def update(self, d: mujoco.MjData):
        """Advance the sensor to d.time (call after every mj_step)."""
        _t_start = time.perf_counter()
        self._d = d
        c, dev = self.cfg, self.device
        prof = self._prof = int(os.environ.get("EVK4_PROFILE", "0") or 0) >= 2
        if not hasattr(self, "_pt"):
            self._pt = dict(host=0.0, render=0.0, pixel=0.0, download=0.0, readout=0.0)
        if not hasattr(self, "_scene"):
            sc = Scene()
            sc.nvis = len(self.vis_np); sc.nlight = self.m.nlight; sc.lux = float(c["lux_per_unit"])
            sc.bg_lv = float(c["background_lux"] / math.pi); sc.k_rate = float(self.k_rate)
            sc.shadows = int(c["shadows"]); sc.spec_scale = float(c["specular_scale"]); sc.wn = float(self.rgbw.sum())
            sc.flick = 1.0
            self._scene = sc
            self._rgbw_wp = wp.vec3(*self.rgbw.astype(np.float32))
        cpos = d.cam_xpos[self.cam_id].copy(); cmat = d.cam_xmat[self.cam_id].reshape(3, 3).copy()
        gpos = d.geom_xpos.copy(); gmat = d.geom_xmat.copy()
        t1 = float(d.time)
        if (not self.initialized) or t1 < self.t_prev - 1e-12 or t1 - self.t_prev > 0.05:
            self._last = None
            self.tile_valid.zero_()
            self._render(gpos, gmat, cpos, cmat, t1, probe=True)
            rate = self.rate.numpy().astype(np.float64)
            x = np.log(np.maximum(0, rate) + c["dark_current"])
            u = np_u01(np_key(int(c["seed"]), np.arange(self.N, dtype=np.uint64), 0, 22))
            ref = x + self.thoff_np - (self.thon_np + self.thoff_np) * (0.05 + 0.9 * u)
            for a in (self.xin, self.L1, self.L2):
                a.assign(x.astype(np.float32))
            self.ref.assign(ref.astype(np.float32)); self.rfr.fill_(-1.0)
            self.initialized = True; self.t_prev = t1; self.sample = 1
            self.pending = np.zeros(0, LABELED_DTYPE); self._events = np.zeros(0, LABELED_DTYPE)
            self.arb_busy = t1; w = c["erc_window_us"] * 1e-6
            self.erc = dict(wstart=math.floor(t1 / w) * w, count=0.0, keep=1.0, acc=0.0, out=0.0)
            self.last_t[:] = -1; self.last_p[:] = -1; self.afk_t[:] = 0; self.afk_per[:] = 0; self.afk_until[:] = 0
            self.prev_gpos, self.prev_gmat, self.prev_cpos, self.prev_cmat = gpos, gmat, cpos, cmat
            self.nsub = 0; self._steps = 0
            return
        if t1 - self.t_prev < 1e-9:
            self._events = np.zeros(0, LABELED_DTYPE)
            return
        t0, DT = self.t_prev, t1 - self.t_prev
        nsub = self._motion_substeps(gpos, gmat, cpos, cmat, DT)
        self.nsub = nsub
        self._steps += 1
        a_cp, a_cm = self.prev_cpos, self.prev_cmat
        ta = t0
        pp = self._pp_struct if hasattr(self, "_pp_struct") else None
        if pp is None:
            pp = PixParams()
            pp.dark = float(c["dark_current"]); pp.fmax = float(c["eff_cutoff"]); pp.r_half = float(self.r_half)
            pp.tau2 = float(1 / (2 * math.pi * c["sf_cutoff_hz"])) if c["sf_cutoff_hz"] > 0 else 0.0
            pp.sf = float(c["sf_cutoff_hz"]); pp.gain = float(c["noise_gain"]); pp.shot = int(c["shot_noise"])
            pp.trefr = c["eff_refr_us"] * 1e-6; pp.lat = float(c["latency_us"]); pp.jit = float(c["latency_jitter_us"])
            pp.seed = wp.uint64(int(c["seed"])); pp.W = self.W; pp.cap = self.cap
            self._pp_struct = pp
        if nsub > 1:
            Rs = _slerp_mats_multi(self.prev_gmat.reshape(-1, 3, 3), gmat.reshape(-1, 3, 3),
                                   [k / nsub for k in range(1, nsub)])
            Rc = _slerp_mats_multi(self.prev_cmat[None], cmat[None], [k / nsub for k in range(1, nsub)])
        self.counter.zero_()                              # events of ALL sub-steps accumulate on the GPU
        for k in range(1, nsub + 1):
            f = k / nsub
            tb = t0 + f * DT
            if k < nsub:
                ip = (1 - f) * self.prev_gpos + f * gpos
                im = Rs[k - 1].reshape(-1, 9)
                icp = (1 - f) * self.prev_cpos + f * cpos
                icm = Rc[k - 1][0]
            else:
                ip, im, icp, icm = gpos, gmat, cpos, cmat
            want_probe = (k == nsub) and (self._steps % self.probe_every == 0)
            self._render(ip, im, icp, icm, tb, a_cp, a_cm, tb - ta, flow=True, probe=want_probe)
            tp = time.perf_counter() if prof else 0
            pp.hpf = float(1.0 - math.exp(-2 * math.pi * c["eff_hpf"] * (tb - ta))) if c["eff_hpf"] > 0 else 0.0
            pp.sample = wp.uint64(self.sample)
            if self._rec_pix is None:
                self._rec_pix = wp.launch(pixel_kernel, dim=self.N, device=dev, record_cmd=True, inputs=[
                    pp, wp.float64(ta), wp.float64(tb), self.rate, self.xin, self.L1, self.L2, self.ref, self.rfr,
                    self.thon, self.thoff, self.leak, self.lab_g, self.lab_d, self.lab_f, self.counter,
                    self.evf, self.evt])
            L = self._rec_pix
            L.set_param_by_name("pp", pp); L.set_param_by_name("ta", wp.float64(ta))
            L.set_param_by_name("tb", wp.float64(tb)); L.launch()
            if prof:
                wp.synchronize(); self._pt["pixel"] += time.perf_counter() - tp
            self.sample += 1
            a_cp, a_cm = icp, icm
            ta = tb
        # ---- one synchronisation + two packed downloads per physics step
        tdl = time.perf_counter()
        n = int(self.counter.numpy()[0])
        if n > self.cap:
            self.dropped_total += n - self.cap; n = self.cap
        ev = np.empty(n, LABELED_DTYPE); te = np.zeros(n)
        if n:
            wp.launch(make_keys, dim=n, inputs=[self.evf, self.evt, self.sort_k, self.sort_v], device=dev)
            wp.utils.radix_sort_pairs(self.sort_k, self.sort_v, n)
            wp.launch(gather_events, dim=n, inputs=[self.sort_v, self.evf, self.evs], device=dev)
            key, ef = self._download(n)
            pix = key & ((1 << 21) - 1)
            ev["t"] = key >> 21
            ev["x"] = pix % self.W; ev["y"] = pix // self.W; ev["p"] = ef[:, 1]
            ev["geom"] = ef[:, 2]; ev["depth"] = ef[:, 3]; ev["flow_x"] = ef[:, 4]; ev["flow_y"] = ef[:, 5]
        tro = time.perf_counter()
        self._events = self._readout(ev, te, t1)
        if prof:
            self._pt["download"] += tro - tdl; self._pt["readout"] += time.perf_counter() - tro
        self.prev_gpos, self.prev_gmat, self.prev_cpos, self.prev_cmat = gpos, gmat, cpos, cmat
        self.t_prev = t1
        if os.environ.get("EVK4_PROFILE"):
            wp.synchronize()
            self._prof_n = getattr(self, "_prof_n", 0) + 1
            self._prof_t = getattr(self, "_prof_t", 0.0) + (time.perf_counter() - _t_start)
            if self._prof_n % 20 == 0:
                msg = (f"[evk4-warp {self.device}] {1e3 * self._prof_t / 20:.2f} ms per step "
                       f"(nsub {nsub}, {len(self._events)} events)")
                if prof:
                    msg += "  |  " + "  ".join(f"{k} {1e3 * v / 20:.2f}" for k, v in self._pt.items())
                    self._pt = {k: 0.0 for k in self._pt}
                print(msg, flush=True)
                self._prof_t = 0.0

    def _download(self, n):
        """Sorted keys + event payload -> host through reusable pinned buffers (async DMA, one sync)."""
        if not getattr(self, "_pinned", False):
            return self.sort_k[:n].numpy(), self.evs[:n].numpy()
        if getattr(self, "_dl_cap", 0) < n:
            cap = min(self.cap, max(1 << 16, 1 << int(math.ceil(math.log2(n * 1.5)))))
            self._dl_k = wp.empty(cap, dtype=wp.int64, device="cpu", pinned=True)
            self._dl_e = wp.empty(cap * 6, dtype=float, device="cpu", pinned=True)
            self._evs_flat = wp.array(ptr=self.evs.ptr, dtype=float, shape=(self.cap * 6,), device=self.device)
            self._dl_cap = cap
        wp.copy(self._dl_k, self.sort_k, count=n)
        wp.copy(self._dl_e, self._evs_flat, count=6 * n)
        wp.synchronize_device(self.device)
        return self._dl_k.numpy()[:n].copy(), self._dl_e.numpy()[:6 * n].reshape(n, 6).copy()

    # ------------------------------------------------------------------ readout (numpy)
    def _readout(self, ev, te, t1):
        c = self.cfg
        roi = [int(float(x)) for x in str(c["roi"]).split()]
        b = int(c["binning"])
        if len(roi) == 4 and roi[2] > roi[0] and roi[3] > roi[1]:
            x, y = ev["x"].astype(int) * b, ev["y"].astype(int) * b
            keep = (x >= roi[0]) & (x < roi[2]) & (y >= roi[1]) & (y < roi[3])
            ev, te = ev[keep], te[keep]
        # events arrive sorted by (t, pixel) from the GPU radix sort
        if c["readout_mevps"] > 0 and len(ev):
            svc = 1.0 / (c["readout_mevps"] * 1e6)
            t = ev["t"] * 1e-6
            k = np.arange(len(t))
            y = np.maximum.accumulate(np.maximum(t - svc * k, self.arb_busy))
            out = y + svc * k
            self.arb_busy = out[-1] + svc
            ev["t"] = np.rint(out * 1e6).astype(np.int64)
        if len(self.pending):
            kk = lambda e: e["t"].astype(np.int64) * (1 << 21) + (e["y"].astype(np.int64) * self.W + e["x"])
            pos = np.searchsorted(kk(ev), kk(self.pending), side="right")
            ev = np.insert(ev, pos, self.pending)
        tnow = int(round(t1 * 1e6))
        split = np.searchsorted(ev["t"], tnow, side="right")
        self.pending = ev[split:].copy()
        ev = ev[:split]
        dropped = 0
        if (c["afk"] or c["stc_us"] > 0 or c["trail_us"] > 0) and len(ev):
            keep = self._esp(ev)
            dropped += int((~keep).sum()); ev = ev[keep]
        if c["erc_mevps"] > 0 and len(ev):
            keep = self._erc(ev)
            dropped += int((~keep).sum()); ev = ev[keep]
        if len(ev) > int(c["max_events"]):
            dropped += len(ev) - int(c["max_events"]); ev = ev[:int(c["max_events"])]
        self.dropped_total += dropped
        self._dropped_step = dropped
        return ev

    def _esp(self, ev):
        c = self.cfg
        n = len(ev)
        pix = ev["y"].astype(np.int64) * self.W + ev["x"]
        t = ev["t"] * 1e-6
        p = ev["p"].astype(np.int8)
        o = np.lexsort((np.arange(n), pix))                 # group by pixel, keep time order
        ps, ts, pls = pix[o], t[o], p[o]
        first = np.ones(n, bool); first[1:] = ps[1:] != ps[:-1]
        prev_t = np.empty(n); prev_p = np.empty(n, np.int8)
        prev_t[1:] = ts[:-1]; prev_p[1:] = pls[:-1]
        prev_t[first] = self.last_t[ps[first]]; prev_p[first] = self.last_p[ps[first]]
        keep = np.ones(n, bool)
        if c["afk"]:
            pmin = 1.0 / c["afk_max_hz"] if c["afk_max_hz"] > 0 else 0.0
            pmax = 1.0 / c["afk_min_hz"] if c["afk_min_hz"] > 0 else 0.0
            bs = (pls == 1) & (prev_p != 1)                     # ON-burst starts
            bi = np.nonzero(bs)[0]
            until = np.full(n, -1e5)
            if len(bi):
                g = ps[bi]
                gfirst = np.ones(len(bi), bool); gfirst[1:] = g[1:] != g[:-1]
                prev_bs = np.empty(len(bi)); prev_bs[1:] = ts[bi[:-1]]; prev_bs[gfirst] = self.afk_t[g[gfirst]]
                per = ts[bi] - prev_bs
                inband = (prev_bs > 0) & (per >= pmin) & (per <= pmax)
                perv = np.where(inband, per, 0.0)
                prevper = np.empty(len(bi)); prevper[1:] = perv[:-1]; prevper[gfirst] = self.afk_per[g[gfirst]]
                periodic = inband & (prevper > 0) & (np.abs(per - prevper) < 0.25 * np.maximum(prevper, 1e-12))
                until[bi] = np.where(periodic, ts[bi] + 2.5 * per, -1e5)
                last = np.ones(len(bi), bool); last[:-1] = g[1:] != g[:-1]
                self.afk_t[g[last]] = ts[bi[last]]; self.afk_per[g[last]] = perv[last]
            # group-wise running max of "until", seeded with the stored state
            grp = np.cumsum(first) - 1
            big = 1e6
            seeded = until.copy()
            seeded[first] = np.maximum(seeded[first], self.afk_until[ps[first]])
            run = np.maximum.accumulate(seeded + grp * big) - grp * big
            keep &= ~(ts < run)
            lastg = np.ones(n, bool); lastg[:-1] = ps[1:] != ps[:-1]
            self.afk_until[ps[lastg]] = np.maximum(self.afk_until[ps[lastg]], run[lastg])
        if c["stc_us"] > 0:
            keep &= (prev_p == pls) & (ts - prev_t <= c["stc_us"] * 1e-6)
        if c["trail_us"] > 0:
            keep &= ~((prev_p == pls) & (ts - prev_t <= c["trail_us"] * 1e-6))
        lastg = np.ones(n, bool); lastg[:-1] = ps[1:] != ps[:-1]
        self.last_t[ps[lastg]] = ts[lastg]; self.last_p[ps[lastg]] = pls[lastg]
        out = np.empty(n, bool); out[o] = keep
        return out

    def _erc(self, ev):
        c, st = self.cfg, self.erc
        win = c["erc_window_us"] * 1e-6
        allowed = c["erc_mevps"] * 1e6 * win
        t = ev["t"] * 1e-6
        keep = np.zeros(len(t), bool)
        i = 0
        while i < len(t):
            while t[i] >= st["wstart"] + win:
                cnt = st["count"]
                st["keep"] = min(1.0, 1.1 * allowed / cnt) if cnt > allowed else 1.0
                st["count"] = 0.0; st["out"] = 0.0
                st["wstart"] = math.floor(t[i] / win) * win if t[i] - st["wstart"] > 10 * win else st["wstart"] + win
            j = np.searchsorted(t, st["wstart"] + win, side="left")
            j = max(j, i + 1)
            nw = j - i
            acc = st["acc"] + st["keep"] * np.arange(1, nw + 1)
            dec = np.floor(acc) > np.floor(np.concatenate([[st["acc"]], acc[:-1]]))
            st["acc"] = float(acc[-1] - math.floor(acc[-1])) if nw else st["acc"]
            room = max(0, int(allowed - st["out"]))
            kept_idx = np.nonzero(dec)[0][:room]
            keep[i + kept_idx] = True
            st["out"] += len(kept_idx); st["count"] += nw
            i = j
        return keep

    # ------------------------------------------------------------------ API
    def events(self):
        e = self._events
        out = np.empty(len(e), EVENT_DTYPE)
        for k in EVENT_DTYPE.names:
            out[k] = e[k]
        return out

    def labeled_events(self):
        return self._events

    def intensity(self):
        return self.L2.numpy().reshape(self.H, self.W).astype(np.float64)

    def geometry(self):
        return self.W, self.H

    def intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy

    def stats(self):
        e = self._events
        return dict(n_events=len(e), n_on=int((e["p"] == 1).sum()), n_off=int((e["p"] == 0).sum()),
                    n_substeps=self.nsub, n_dropped_total=self.dropped_total)
