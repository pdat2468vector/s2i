"""
Paper figures in the Hextech skin (Phase 10, v3). Plan: figure_plan_v2_hextech.md.

    python -m s2i.hexfigs --out OUT [--runs RUNS] [--only fig1 fig2a]

fig1  hero overview (navy, 3D). Read left to right:
      new patient (glass dome, fsaverage5 brain, scalp studs, SEEG shaft) -> raw scalp traces -> the crystal
      (temporal filters shared across patients, meta-learned on the 14 other patients riding the orbit ring)
      -> filtered traces -> brass lens (spatial readout fitted per patient on the first 20 % of trials)
      -> bipolar SEEG contacts lit by prediction quality. The molten arc is the one accent: the gradient runs
      back through the ridge solution into the filters. Schematic only: no result is plotted.
fig2a data-figure skin test (ivory): three models at the primary strict level, one dot per patient.
Every figure writes <name>_numbers.json.
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch, FancyArrowPatch
from matplotlib.path import Path

from . import hexstyle as hs
from .hexstyle import (MM, NAVY, NAVY2, NAVY3, IVORY, BRASS, BRASS_LT, BRASS_DK, PEWTER, PALE_BRASS, HEX, HEX_LT,
                       HEX_DK, MOLTEN, MOLTEN_LT, INK, INK2, INK3, DECO, LBL_FIXED)

# hand-placed positions (mm, figure coordinates); tuned by eye on the render
NUDGE = {
    "fig1": {
        "head": (35.0, 52.0), "head_scale": 0.27, "cam": (28.0, 17.0),
        "crystal": (94.0, 60.0), "lens": (131.0, 58.0), "shaft_x": 158.5, "shaft_y": (29.0, 87.0),
        "lbl_new": (35.0, 16.0), "lbl_scalp": (63.0, 87.0), "lbl_shared": (94.0, 33.0),
        "lbl_others": (94.0, 29.2), "lbl_lens": (131.0, 37.0), "lbl_seeg": (158.5, 22.5),
        "lbl_grad": (112.0, 90.5), "strip": (100.0, 169.0, 7.5), "seed": 7,
    },
}


# ----------------------------------------------------------------------------- small drawing helpers
PT_MM = 25.4 / 72


def tracked(ax, x, y, s, fs=6.6, color=BRASS_LT, ha="center", va="center", spacing=0.16, z=30, family=DECO):
    """Letter-spaced text: glyphs placed one by one (spacing in em). Returns the total width in mm."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    fp = FontProperties(family=family)
    ws = []
    for ch in s:
        if ch == " ":
            ws.append(fs * 0.34 * PT_MM)
        else:
            e = TextPath((0, 0), ch, size=fs, prop=fp).get_extents()
            ws.append((e.x1 + max(e.x0, 0)) * PT_MM)
    gap = spacing * fs * PT_MM
    total = sum(ws) + gap * (len(s) - 1)
    x0 = x - total / 2 if ha == "center" else (x - total if ha == "right" else x)
    yb = y - 0.35 * fs * PT_MM if va == "center" else y
    for ch, w in zip(s, ws):
        if ch != " ":
            ax.text(x0, yb, ch, color=color, fontsize=fs, ha="left", va="baseline", family=family, zorder=z)
        x0 += w + gap
    return total


def label(ax, x, y, s, color=BRASS_LT, fs=6.6, ha="center", va="center", z=30, deco=True):
    if deco:
        return tracked(ax, x, y, s.upper(), fs=fs, color=color, ha=ha, va=va, z=z)
    ax.text(x, y, s, color=color, fontsize=fs, ha=ha, va=va, zorder=z)


def bezier(p0, p1, p2, p3, n=200):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    b = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    d = 3 * (1 - t) ** 2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t ** 2 * (p3 - p2)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return b, np.c_[-d[:, 1], d[:, 0]]


def eeg_like(n, rng, alpha=1.1, k=40):
    t = np.linspace(0, 1, n)
    y = np.zeros(n)
    for f in range(1, k):
        y += np.sin(2 * np.pi * (f * 1.7) * t + rng.uniform(0, 2 * np.pi)) / f ** (alpha * 0.5)
    return y / np.abs(y).max()


def gabor(n, f, sigma=0.17, phase=0.0):
    t = np.linspace(-0.5, 0.5, n)
    return np.exp(-t ** 2 / (2 * sigma ** 2)) * np.cos(2 * np.pi * f * t + phase)


def wave_along(ax, b, nrm, w, amp, color, lw=0.6, glow=True, core=None, z=6, taper=0.12, alpha=1.0):
    n = len(b)
    env = np.ones(n)
    k = max(int(n * taper), 1)
    env[:k] = np.sin(np.linspace(0, np.pi / 2, k)) ** 2
    env[-k:] = env[:k][::-1]
    p = b + nrm * (amp * w * env)[:, None]
    if glow:
        hs.glow_plot(ax, p[:, 0], p[:, 1], color, lw=lw, core=core, layers=5, spread=2.6, alpha=0.08 * alpha, z=z)
    else:
        ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, zorder=z, alpha=alpha)
    return p


# ----------------------------------------------------------------------------- fig 1 hero
def _front_arc(pts):
    """Visible (lower, nearer) arc of a closed projected ring, ordered left to right."""
    i0, i1 = int(np.argmin(pts[:, 0])), int(np.argmax(pts[:, 0]))
    n = len(pts)
    a = pts[np.arange(i0, i0 + ((i1 - i0) % n) + 1) % n]
    b = pts[np.arange(i1, i1 + ((i0 - i1) % n) + 1) % n][::-1]
    arc = a if a[:, 1].mean() < b[:, 1].mean() else b
    return arc if arc[0, 0] <= arc[-1, 0] else arc[::-1]


def _head(ax, N, rng):
    """Glass dome + brain + pedestal + scalp studs + SEEG shaft. Returns stud screen positions used as sources."""
    hx, hy = N["head"]
    s = N["head_scale"]
    R = hs.camera(*N["cam"])
    c = np.array([0.0, -18.0, 4.0])
    radii = np.array([90.0, 112.0, 100.0])
    z_cut = -54.0
    off = np.array([hx, hy]) - s * (R @ c)[:2]
    P = lambda w: s * (np.atleast_2d(w) @ R.T)[:, :2] + off            # noqa: E731
    to_mm = lambda p: s * p + off                                       # noqa: E731

    # glass, back layer (behind the brain)
    ext = (hx - 36, hx + 36, hy - 34, hy + 38)
    back, front, ext = hs.raytrace_ellipsoid(R, c, radii, s, off, ext, res=1100, z_cut=z_cut, a0=0.035, a1=0.5)
    ax.imshow(back, extent=ext, origin="lower", zorder=2, interpolation="bilinear")
    # window highlight: a soft curved streak inside the upper-left rim
    m = front[..., 3] > 0
    ys_, xs_ = np.nonzero(m)
    gx0, gx1 = np.interp([xs_.min(), xs_.max()], [0, m.shape[1] - 1], ext[:2])
    gy1 = np.interp(ys_.max(), [0, m.shape[0] - 1], ext[2:])
    gcx, gw = (gx0 + gx1) / 2, (gx1 - gx0)
    xh, yh = hs.ellipse_xy(gcx, gy1 - gw * 0.42, gw * 0.84, gw * 0.78, t0=np.radians(124), t1=np.radians(156))
    hs.glow_plot(ax, xh, yh, "white", lw=0.7, layers=4, spread=2.0, alpha=0.035, z=4.2)
    ax.plot(xh, yh, color="white", lw=0.45, alpha=0.32, zorder=4.21)

    B = hs.load_brain()
    parts = []
    for h in ("left", "right"):
        v, f, su = B[h]
        vn = hs.vertex_normals(v, f)
        v, f, vn, su = hs.subdivide(v, f, vn, su)
        t = 1 / (1 + np.exp(-su[f].mean(1) / 1.5))
        base = hs.mix("#f7eedc", "#86683f", t)
        parts.append(hs.mesh_faces(v, f, R, to_mm, base, rim=HEX, rim_w=0.6, spec=0.22, vn=vn, ambient=0.24))
    hs.draw_faces(ax, parts, z=3)
    ax.imshow(front, extent=ext, origin="lower", zorder=4, interpolation="bilinear")

    # pedestal: brass collar at the cut + two stepped bands
    t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    ring = lambda k, dz: P(np.c_[c[0] + radii[0] * k * np.cos(t), c[1] + radii[1] * k * np.sin(t),   # noqa: E731
                                 np.full_like(t, z_cut + dz)])
    r0 = np.sqrt(1 - ((z_cut - c[2]) / radii[2]) ** 2)
    steps = [(r0 * 1.00, 0.0, -8.0), (r0 * 1.12, -8.0, -15.0)]
    for i, (k, za, zb) in enumerate(steps[::-1]):
        zz = 2.2 + i * 0.1                                  # between the back glass (2) and the brain (3)
        top, bot = ring(k, za), ring(k, zb)
        ft, fb = _front_arc(top), _front_arc(bot)
        side = PathPatch(Path(np.r_[ft, fb[::-1], ft[:1]]), fc="none", ec="none", zorder=zz)
        ax.add_patch(side)
        x0, x1 = min(ft[:, 0].min(), fb[:, 0].min()), max(ft[:, 0].max(), fb[:, 0].max())
        img = np.zeros((2, 256, 4))                                   # horizontal metal sheen across the band
        u = np.linspace(0, 1, 256)
        img[..., :3] = hs.mix(BRASS_DK, BRASS_LT, 0.15 + 0.85 * np.exp(-((u - 0.36) / 0.22) ** 2))[None]
        img[..., 3] = 1
        im = ax.imshow(img, extent=[x0, x1, fb[:, 1].min(), top[:, 1].max()], zorder=zz, aspect="auto",
                       interpolation="bilinear")
        im.set_clip_path(side)
        for xf in np.linspace(x0 + 1.6, x1 - 1.6, 13 + 2 * i):                     # Art Deco flutes
            yt, yb = np.interp(xf, ft[:, 0], ft[:, 1]), np.interp(xf, fb[:, 0], fb[:, 1])
            ax.plot([xf, xf], [yt - 0.35, yb + 0.35], color=BRASS_DK, lw=0.3, alpha=0.55, zorder=zz + 0.01)
        ax.plot(fb[:, 0], fb[:, 1], color=BRASS_DK, lw=0.5, zorder=zz + 0.02)
        ax.add_patch(Polygon(top, closed=True, fc=hs.mix(NAVY2, BRASS, 0.28), ec=BRASS_LT, lw=0.35,
                             zorder=zz + 0.03))
    collar = ring(r0, 0.0)
    fa = _front_arc(collar)
    hs.brass_ring(ax, collar[:, 0], collar[:, 1], z=2.45, lw=1.3)
    hs.brass_ring(ax, fa[:, 0], fa[:, 1], z=4.9, lw=1.4, closed=False)

    # SEEG shaft (upper-front-right entry, aiming down into the temporal lobe)
    entry_dir = hs._unit([0.72, 0.42, 0.55])
    k = 1 / np.sqrt((entry_dir ** 2 / radii ** 2).sum())
    E = c + entry_dir * k
    T = np.array([38.0, -8.0, -14.0])
    out = E + (E - T) / np.linalg.norm(E - T) * 16
    pe, pt, po = P(E)[0], P(T)[0], P(out)[0]
    hs.glow_plot(ax, [po[0], pt[0]], [po[1], pt[1]], "#c9ced6", lw=0.9, core="white", layers=3, spread=1.4,
                 alpha=0.12, z=5)
    for u in np.linspace(0.0, 0.5, 8):
        q = pt + (pe - pt) * u
        ax.add_patch(plt.Circle(q, 0.5, fc="#eef2f6", ec="#5b6472", lw=0.3, zorder=5.1))
    dv = (po - pe) / np.linalg.norm(po - pe)
    nv = np.array([-dv[1], dv[0]])
    q0, q1 = pe + dv * 0.4, pe + dv * 3.2                              # brass anchor bolt at the entry
    ax.add_patch(Polygon([q0 + nv * 0.9, q1 + nv * 0.75, q1 - nv * 0.75, q0 - nv * 0.9], closed=True, fc=BRASS,
                         ec=BRASS_DK, lw=0.35, zorder=5.2))
    ax.plot(*np.c_[q0 + dv * 1.2 + nv * 0.82, q0 + dv * 1.2 - nv * 0.82], color=BRASS_LT, lw=0.35, zorder=5.21)

    # scalp studs
    dirs = [(0.0, 0.0)]
    for th, cnt, ph in ((24, 6, 0.0), (46, 11, 0.4), (66, 15, 0.1), (84, 18, 0.5)):
        dirs += [(th, 360 * (i + ph) / cnt) for i in range(cnt)]
    studs = []
    for th, az in dirs:
        th, az = np.radians(th), np.radians(az)
        u = np.array([np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th)])
        w = c + u * radii
        if w[2] < z_cut + 10:
            continue
        nw = hs._unit((w - c) / radii ** 2)
        nc = R @ nw
        if nc[2] < 0.15:
            continue
        p = P(w)[0]
        studs.append((p, nc))
    cand = [q for q in studs if q[1][2] > 0.45]
    cand = sorted(cand, key=lambda q: -q[0][0])[:10]
    cand = sorted(cand, key=lambda q: -q[0][1])
    src = [cand[0], cand[len(cand) // 2], cand[-1]]
    for p, nc in studs:
        lit = any(p is q[0] for q in src)
        d = 1.7 if lit else 1.35
        sq = max(abs(nc[2]), 0.3)
        ang = np.degrees(np.arctan2(nc[1], nc[0])) + 90
        ax.add_patch(plt.matplotlib.patches.Ellipse(p, d, d * sq, angle=ang, fc=BRASS_DK, ec="none", zorder=6))
        ax.add_patch(plt.matplotlib.patches.Ellipse(p + np.array([0, 0.1]), d * 0.7, d * 0.7 * sq, angle=ang,
                                                    fc=BRASS_LT if lit else BRASS, ec="none", zorder=6.1))
        ax.add_patch(plt.matplotlib.patches.Ellipse(p + np.array([-0.12, 0.2]), d * 0.22, d * 0.22 * sq,
                                                    angle=ang, fc="white", ec="none", alpha=0.7, zorder=6.2))
        if lit:
            hs.halo(ax, p[0], p[1], 3.4, BRASS_LT, a=0.5, z=5.9)
    return [q[0] for q in src], P, (pe, pt)


def _crystal(ax, cx, cy, rng):
    """Faceted hexagonal bipyramid, glowing; returns its outline (for clipping) and side ports."""
    r, hh = 7.2, 15.5
    ang = np.arange(6) * np.pi / 3
    v = np.r_[[[0, 0, hh]], np.c_[r * np.cos(ang), r * np.sin(ang), np.zeros(6)], [[0, 0, -hh]]]
    f = np.array([[0, 1 + i, 1 + (i + 1) % 6] for i in range(6)] + [[7, 1 + (i + 1) % 6, 1 + i] for i in range(6)])
    Rc = hs.camera(24, 10)
    to_mm = lambda p: p + np.array([cx, cy])                            # noqa: E731
    base = np.array(hs.mix(HEX_DK, HEX, 0.55))
    polys, rgba, depth = hs.mesh_faces(v, f, Rc, to_mm, base, alpha=0.9, cull=True, ambient=0.35, spec=0.9,
                                       shininess=14, rim=HEX_LT, rim_w=0.9, rim_pow=1.4)
    hs.halo(ax, cx, cy, 34, HEX, a=0.32, z=6.8, k=1.6)
    hs.halo(ax, cx, cy, 14, HEX_LT, a=0.45, z=6.9, k=2.2)
    o = np.argsort(depth)
    for i in o:
        ax.add_patch(Polygon(polys[i], closed=True, fc=rgba[i], ec=hs.mix(HEX_LT, "white", 0.4), lw=0.35,
                             zorder=8, joinstyle="round"))
    # brass caps at the apexes
    pv = to_mm((v @ Rc.T)[:, :2])
    for k, sgn in ((0, 1), (7, -1)):
        x0, y0 = pv[k]
        cap = np.array([[x0 - 1.6, y0 + sgn * 0.2], [x0 + 1.6, y0 + sgn * 0.2], [x0 + 1.0, y0 + sgn * 2.3],
                        [x0 - 1.0, y0 + sgn * 2.3]])
        ax.add_patch(Polygon(cap, closed=True, fc=BRASS, ec=BRASS_DK, lw=0.4, zorder=8.5))
        ax.plot([x0 - 0.8, x0 + 0.8], [y0 + sgn * 1.3] * 2, color=BRASS_LT, lw=0.35, zorder=8.6)
    from scipy.spatial import ConvexHull
    hull = pv[ConvexHull(pv).vertices]
    clip = Polygon(hull, closed=True, fc="none", ec="none", zorder=9)
    ax.add_patch(clip)
    # the learned temporal filters inside (three wavelets, white-hot)
    for yy, fq, w in ((cy + 5.0, 2.2, 7.0), (cy, 3.6, 8.5), (cy - 5.0, 5.4, 7.0)):
        x = np.linspace(cx - w / 2, cx + w / 2, 160)
        g = gabor(160, fq, sigma=0.2)
        for lw_, a_ in ((2.2, 0.18), (1.2, 0.35)):
            ln, = ax.plot(x, yy + 1.5 * g, color=HEX_LT, lw=lw_, alpha=a_, zorder=9)
            ln.set_clip_path(clip)
        ln, = ax.plot(x, yy + 1.5 * g, color="white", lw=0.5, zorder=9.1)
        ln.set_clip_path(clip)
    return pv


def _orbit(ax, cx, cy, N):
    w, h, rot, oy = 38.0, 8.5, np.radians(-6), cy - 8.0
    xb, yb = hs.ellipse_xy(cx, oy, w, h, t0=0, t1=np.pi, rot=rot)
    xf, yf = hs.ellipse_xy(cx, oy, w, h, t0=np.pi, t1=2 * np.pi, rot=rot)
    hs.brass_ring(ax, xb, yb, z=6.5, lw=1.3, closed=False)
    hs.brass_ring(ax, xf, yf, z=10, lw=1.5, closed=False)
    ts = np.linspace(0, 2 * np.pi, 14, endpoint=False) + 0.22
    for i, t in enumerate(ts):
        x, y = hs.ellipse_xy(cx, oy, w, h, n=2, t0=t, t1=t, rot=rot)
        x, y = x[0], y[0]
        front = np.sin(t) < 0
        r = 1.45 if front else 1.1
        z = 10.2 if front else 6.6
        early = i in (4, 9)
        ax.add_patch(Polygon(hs.hex_xy(x, y, r), closed=True, fc=NAVY if early else BRASS,
                             ec=BRASS_LT if not early else BRASS, lw=0.45, ls=(0, (1.2, 0.8)) if early else "-",
                             zorder=z))
        if not early:
            ax.add_patch(plt.Circle((x, y), r * 0.33, fc=HEX_LT, ec="none", zorder=z + 0.01, alpha=0.9))


def _lens(ax, cx, cy, rng):
    w, h = 19.0, 29.0
    # outer dial with ticks
    for i in range(36):
        t = 2 * np.pi * i / 36
        r_in, r_out = (1.08, 1.17) if i % 3 else (1.06, 1.23)
        ax.plot([cx + w / 2 * r_in * np.cos(t), cx + w / 2 * r_out * np.cos(t)],
                [cy + h / 2 * r_in * np.sin(t), cy + h / 2 * r_out * np.sin(t)], color=BRASS, lw=0.4,
                zorder=8, alpha=0.85)
    xo, yo = hs.ellipse_xy(cx, cy, w * 1.0, h * 1.0)
    inner = Polygon(np.c_[hs.ellipse_xy(cx, cy, w * 0.86, h * 0.88)], closed=True, fc=NAVY2, ec="none", zorder=7.9)
    ax.add_patch(inner)
    nx, ny = 6, 9
    coef = rng.normal(0, 1, (ny, nx))
    coef = np.tanh(coef * 0.9)
    cw, ch = w * 0.86 / nx, h * 0.88 / ny
    for i in range(ny):
        for j in range(nx):
            v = coef[i, j]
            col = hs.mix(NAVY2, HEX if v > 0 else BRASS, abs(v) ** 0.8)
            x0 = cx - w * 0.43 + j * cw
            y0 = cy - h * 0.44 + i * ch
            sq = Polygon([[x0 + 0.18, y0 + 0.18], [x0 + cw - 0.18, y0 + 0.18], [x0 + cw - 0.18, y0 + ch - 0.18],
                          [x0 + 0.18, y0 + ch - 0.18]], closed=True, fc=col, ec="none", zorder=8)
            ax.add_patch(sq)
            sq.set_clip_path(inner)
    # glass sheen
    sheen = Polygon(np.c_[hs.ellipse_xy(cx - 2.5, cy + 6, w * 0.45, h * 0.35, rot=0.5)], closed=True,
                    fc="white", ec="none", alpha=0.10, zorder=8.2)
    ax.add_patch(sheen)
    hs.brass_ring(ax, xo, yo, z=8.4, lw=3.0)
    xi, yi = hs.ellipse_xy(cx, cy, w * 0.88, h * 0.9)
    ax.plot(xi, yi, color=BRASS_DK, lw=0.5, zorder=8.45)
    return w, h


def _output(ax, x, y0, y1, lens_xy, rng, N):
    n = 10
    ys = np.linspace(y1 - 4, y0 + 3, n)
    r = np.sort(rng.beta(2.2, 4.0, n))[::-1]
    r = r / r.max()
    rank = np.argsort(rng.permutation(n))
    rr = r[rank]
    top = rr >= np.sort(rr)[-3]
    # shaft
    shaft = Polygon([[x - 0.9, y1 + 2], [x + 0.9, y1 + 2], [x + 0.9, y0], [x, y0 - 1.6], [x - 0.9, y0]],
                    closed=True, fc="none", ec="none", zorder=7.5)
    ax.add_patch(shaft)
    im = hs.vgrad(ax, x - 0.9, x + 0.9, y0 - 1.6, y1 + 2, "#5b6472", "#c9ced6", clip=shaft, z=7.5)
    t = np.linspace(0, 1, 2)
    img = np.zeros((2, 64, 4))
    img[..., :3] = hs.mix("#3f4753", "#e6eaef", np.sin(np.linspace(0, np.pi, 64)) ** 0.7)[None, :, :]
    img[..., 3] = 1
    im2 = ax.imshow(img, extent=[x - 0.9, x + 0.9, y0 - 1.6, y1 + 2], zorder=7.51, aspect="auto",
                    interpolation="bilinear")
    im2.set_clip_path(shaft)
    lx, ly = lens_xy
    for i in range(n):
        yc = ys[i]
        a = 0.15 + 0.85 * rr[i] ** 1.6
        c = hs.mix(PEWTER, HEX, rr[i] ** 1.4)
        # beam from the lens
        b, nrm = bezier((lx + 10.8, ly + (yc - ly) * 0.18), (lx + 17, ly + (yc - ly) * 0.35),
                        (x - 9, yc), (x - 1.4, yc), n=80)
        ax.plot(b[:, 0], b[:, 1], color=HEX, lw=0.3 + 0.6 * rr[i] ** 2, alpha=0.12 + 0.55 * rr[i] ** 2, zorder=7)
        # contact ring
        ax.add_patch(Polygon([[x - 1.15, yc - 1.0], [x + 1.15, yc - 1.0], [x + 1.15, yc + 1.0],
                              [x - 1.15, yc + 1.0]], closed=True, fc=c, ec=hs.mix(c, "white", 0.4), lw=0.25,
                             zorder=7.6))
        if top[i]:
            hs.halo(ax, x, yc, 6.5, HEX, a=0.55, z=7.4)
        # predicted trace next to the contact
        xx = np.linspace(x + 2.6, x + 16.5, 120)
        yy = yc + 1.25 * eeg_like(120, rng, alpha=1.6, k=14)
        if top[i]:
            hs.glow_plot(ax, xx, yy, HEX, lw=0.55, core="white", layers=4, spread=2.0, alpha=0.1, z=7.7)
        else:
            ax.plot(xx, yy, color=c, lw=0.4, alpha=a, zorder=7.7)
    return rr.tolist()


def _strip(ax, x0, x1, y, rng):
    n, cal = 60, 12
    sess = [0, 16, 31, 46, 60]
    gap_s, gap_c = 2.2, 1.2
    total = x1 - x0 - gap_s * 3 - gap_c
    tw = total / n
    xs = []
    x = x0
    for i in range(n):
        if i in sess[1:-1]:
            x += gap_s
        if i == cal:
            x += gap_c
        xs.append(x)
        x += tw
    xs = np.array(xs)
    h, sk = 4.2, 0.9
    for i in range(n):
        xa = xs[i]
        if i < cal:
            fc, ec, a = HEX, HEX_LT, 1.0
        else:
            s = np.searchsorted(sess, i, side="right") - 1
            fc, ec, a = BRASS, BRASS_LT, [0.85, 0.85, 0.62, 0.42][s]
        ax.add_patch(Polygon([[xa + 0.12, y], [xa + tw - 0.12, y], [xa + tw - 0.12 + sk, y + h],
                              [xa + 0.12 + sk, y + h]], closed=True, fc=fc, ec="none", alpha=a, zorder=8))
    hs.halo(ax, (xs[0] + xs[cal - 1] + tw) / 2 + sk / 2, y + h / 2, 11, HEX, a=0.35, z=7.5, squash=0.45)
    return xs, tw, h, cal


def fig1_hero(runs, out):
    hs.setup()
    N = NUDGE["fig1"]
    rng = np.random.default_rng(N["seed"])
    Wmm, Hmm = 180.0, 100.0
    fig = plt.figure(figsize=(Wmm * MM, Hmm * MM), facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, Wmm)
    ax.set_ylim(0, Hmm)
    ax.axis("off")
    ax.set_facecolor(NAVY)

    # atmosphere: lifted centre, deco sunburst behind the crystal, chamfered frame
    hs.halo(ax, 92, 56, 120, NAVY3, a=0.75, z=0.1, k=1.2, squash=0.7)
    cx, cy = N["crystal"]
    hs.sunburst(ax, cx, cy, 20, 52, n=48, color=BRASS, a=0.13, lw=0.3, z=0.5)
    hs.deco_frame(ax, 1.6, 1.6, Wmm - 1.6, Hmm - 1.6, c=3.2, gap=1.0, color=BRASS, lw=(0.6, 0.25), a=0.55,
                  fans=7.0)

    srcs, P, (pe, pt) = _head(ax, N, rng)

    # raw scalp traces -> crystal (ports follow the bipyramid silhouette)
    half_w = lambda dy: 7.2 * (1 - abs(dy) / 15.5)                     # noqa: E731
    ports_in = [(cx - half_w(dy) - 0.2, cy + dy) for dy in (6.0, 2.5, -1.0)]
    for k, (p, q) in enumerate(zip(srcs, ports_in)):
        b, nrm = bezier(p, (p[0] + 14, p[1] + 6 - 3 * k), (q[0] - 16, q[1] + 2), q, n=320)
        wave_along(ax, b, nrm, eeg_like(320, rng, alpha=0.9, k=50), 1.3, BRASS_LT, lw=0.5, core=None, z=6.3,
                   alpha=1.0)

    _orbit(ax, cx, cy, N)
    pv = _crystal(ax, cx, cy, rng)

    # filtered traces -> lens
    lx, ly = N["lens"]
    for k, (dy, fq) in enumerate(((6.0, 9), (2.5, 15), (-1.0, 23))):
        p0 = (cx + half_w(dy) + 0.2, cy + dy)
        p1 = (lx - 10.6, ly + (dy - 2.5) * 1.7)
        b, nrm = bezier(p0, (p0[0] + 8, p0[1]), (p1[0] - 8, p1[1]), p1, n=260)
        w = np.sin(np.linspace(0, fq * np.pi, 260))
        wave_along(ax, b, nrm, w, 1.0, HEX, lw=0.55, core=HEX_LT, z=7.2)

    _lens(ax, lx, ly, rng)
    rr = _output(ax, N["shaft_x"], N["shaft_y"][0], N["shaft_y"][1], (lx, ly), rng, N)

    # calibration strip, tied to the lens label
    sx0, sx1, sy = N["strip"]
    xs, tw, h, cal = _strip(ax, sx0, sx1, sy, rng)
    xm = (xs[0] + xs[cal - 1] + tw) / 2
    lb = N["lbl_lens"]
    b, nrm = bezier((xm + 1.5, sy + h + 5.0), (xm + 3, sy + h + 14), (lb[0] - 10, lb[1] - 6), (lb[0] - 4, lb[1] - 2.2),
                    n=120)
    ax.plot(b[:, 0], b[:, 1], color=HEX, lw=0.45, ls=(0, (1.6, 1.4)), zorder=8, alpha=0.9)

    # the accent: gradient through the ridge solution back into the filters
    a0 = (lx - 1.0, ly + 16.4)
    a1 = (cx + 1.6, pv[:, 1].max() + 3.2)
    b, _ = bezier(a0, (a0[0] - 2, a0[1] + 17), (a1[0] + 8, a1[1] + 15), a1, n=200)
    hs.glow_plot(ax, b[:, 0], b[:, 1], MOLTEN, lw=1.1, core=MOLTEN_LT, layers=7, spread=5, alpha=0.08, z=11)
    d = b[-1] - b[-6]
    d /= np.linalg.norm(d)
    nv = np.array([-d[1], d[0]])
    tip = b[-1] + d * 1.2
    ax.add_patch(Polygon([tip, b[-1] - d * 1.6 + nv * 1.2, b[-1] - d * 1.6 - nv * 1.2], closed=True, fc=MOLTEN,
                         ec=MOLTEN_LT, lw=0.3, zorder=11.2))

    # labels (the only words in the figure)
    label(ax, *N["lbl_new"], "new patient")
    label(ax, *N["lbl_scalp"], "scalp EEG", color=BRASS_LT)
    label(ax, *N["lbl_shared"], "shared filters", color=HEX_LT)
    label(ax, *N["lbl_others"], "14 other patients", color=BRASS, fs=5.6)
    label(ax, *N["lbl_lens"], "fitted per patient", color=BRASS_LT)
    label(ax, *N["lbl_seeg"], "bipolar SEEG", color=HEX_LT)
    label(ax, *N["lbl_grad"], "gradient through ridge", color=MOLTEN_LT, fs=6.2)
    label(ax, xm + 0.5, sy + h + 2.8, "first 20 %", color=HEX_LT, fs=5.4)
    label(ax, (xs[cal] + xs[-1] + tw) / 2, sy + h + 2.8, "test", color=BRASS, fs=5.4)

    hs.save(fig, out, "fig1_hero", facecolor=NAVY)
    with open(os.path.join(out, "fig1_hero_numbers.json"), "w") as f:
        json.dump({"schematic": True, "calibration_fraction": 0.2, "n_trials_drawn": 60, "n_cal_drawn": 12,
                   "orbit_tokens": 14, "early_stopping_tokens": 2, "contact_brightness": rr}, f, indent=1)


# ----------------------------------------------------------------------------- data-figure skin test
def _ingot(ax, x, y, w, h, fc, ec, z=5):
    """Chamfered horizontal bar (brass ingot) centred on (x, y), w and h in data units."""
    c = w * 0.12
    ax.add_patch(Polygon([[x - w / 2 + c, y - h / 2], [x + w / 2 - c, y - h / 2], [x + w / 2, y],
                          [x + w / 2 - c, y + h / 2], [x - w / 2 + c, y + h / 2], [x - w / 2, y]], closed=True,
                         fc=fc, ec=ec, lw=0.5, zorder=z))


def fig2a_skin(runs, out):
    """Skin test only: three models at the primary strict level. Values are DEMO numbers unless runs is given."""
    hs.setup()
    rng = np.random.default_rng(3)
    n = 15
    base = np.clip(rng.normal(0.167, 0.055, n), 0.05, None)
    base += 0.167 - base.mean()
    fixed = base + rng.normal(0.025, 0.010, n)
    final = base + np.abs(rng.normal(0.037, 0.013, n))
    fixed += 0.192 - fixed.mean()
    final += 0.204 - final.mean()
    vals = [base, fixed, final]
    cols = [PEWTER, PALE_BRASS, HEX]
    names = ["baseline", LBL_FIXED.replace(" untrained", ""), "ours"]

    fig = plt.figure(figsize=(88 * MM, 66 * MM))
    ax = fig.add_axes([16 / 88, 10 / 66, 58 / 88, 43 / 66])
    xs = np.arange(3)
    for i in range(n):                                                  # one line per patient
        ax.plot(xs, [v[i] for v in vals], color=hs.mix(IVORY, INK3, 0.45), lw=0.35, zorder=1)
    for k, (v, c) in enumerate(zip(vals, cols)):
        off = hs_beeswarm(v, 0.006) * 0.55
        if k == 2:
            ax.scatter(xs[k] + off, v, s=34, color=HEX, alpha=0.16, lw=0, zorder=2)
            ax.scatter(xs[k] + off, v, s=14, color=HEX, edgecolor=HEX_DK, lw=0.4, zorder=3)
        else:
            ax.scatter(xs[k] + off, v, s=11, color=c, edgecolor=hs.mix(c, INK, 0.35), lw=0.35, zorder=3)
        m = v.mean()
        _ingot(ax, xs[k] + 0.36, m, 0.22, 0.012, HEX_DK if k == 2 else BRASS, BRASS_DK, z=4)
        ax.text(xs[k] + 0.50, m, f"{m:.3f}", va="center", ha="left", fontsize=6,
                color=HEX_DK if k == 2 else INK2, fontweight="bold" if k == 2 else "normal")
    # the accent: ours minus baseline
    d = final - base
    yb, yf = base.mean(), final.mean()
    xb = 2.86
    ax.plot([xb - 0.05, xb, xb, xb - 0.05], [yb, yb, yf, yf], color=MOLTEN, lw=0.8, clip_on=False,
            solid_joinstyle="miter")
    ax.text(xb + 0.07, (yb + yf) / 2 + 0.004, f"+{d.mean():.3f}", color=MOLTEN, fontsize=6.6, fontweight="bold",
            va="bottom", ha="left", clip_on=False)
    ax.text(xb + 0.07, (yb + yf) / 2 - 0.002, f"{(d > 0).sum()}/{n}", color=INK2, fontsize=5.8, va="top",
            ha="left", clip_on=False)
    ax.set_xlim(-0.45, 2.75)
    ax.set_ylim(0.0, 0.36)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["0", "0.1", "0.2", "0.3"])
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    for t, c in zip(ax.get_xticklabels(), [INK2, INK2, HEX_DK]):
        t.set_color(c)
    ax.get_xticklabels()[2].set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=4)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_bounds(0, 0.3)
    ax.spines["left"].set_position(("outward", 3))
    ax.set_ylabel("correlation r, test trials")
    # header: double brass rule, badge, tag
    hdr = fig.add_axes([0, 0, 1, 1], zorder=-1)
    hdr.set_xlim(0, 88)
    hdr.set_ylim(0, 66)
    hdr.axis("off")
    hdr.plot([7.5, 86], [61.2, 61.2], color=BRASS, lw=0.7)
    hdr.plot([7.5, 86], [60.4, 60.4], color=BRASS, lw=0.25)
    hs.hex_badge(fig, 1.6, 63.6, "a", r=2.4, fs=8)
    tracked(hdr, 86, 63.2, "PRIMARY  \u00b7  STRICT  \u00b7  N = 15", fs=5.2, color=INK3, ha="right")
    hdr.text(86, 1.5, "demo values (skin test)", fontsize=4.5, color=INK3, ha="right", va="bottom",
             style="italic")
    hs.save(fig, out, "fig2a_skin")


def hs_beeswarm(v, size):
    from .figstyle import beeswarm
    return beeswarm(v, size)


FIGS = {"fig1": fig1_hero, "fig2a": fig2a_skin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    for k, fn in FIGS.items():
        if a.only and k not in a.only:
            continue
        print(k)
        fn(a.runs, a.out)


if __name__ == "__main__":
    main()
