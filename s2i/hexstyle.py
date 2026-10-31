"""
Hextech skin for the paper figures (Phase 10, v3). Plan: figure_plan_v2_hextech.md.

Only generic style elements are used (brass and ivory, glowing cyan energy, Art Deco geometry: chamfered corners,
double rules, hexagons, sunbursts). No game asset, character, weapon, logo or font is reproduced. Everything is
drawn by code. Deco font: Federo (SIL Open Font License, s2i/fonts/OFL_Federo.txt).

Pieces
  * palette + setup()                     rcParams for ivory data figures; NAVY is the hero background
  * glow_plot / halo / sunburst           light effects (stacked strokes, radial RGBA images)
  * brass_ring / deco_frame / hex_badge   metal and Art Deco ornaments
  * camera / mesh_faces / draw_faces      a tiny 3D renderer: orthographic camera, Blinn-Phong + rim light,
                                          painter's sort across several meshes, drawn as one rasterized
                                          PolyCollection (text and axes stay vector)
  * load_brain                            fsaverage5 pial surfaces + sulcal depth (nilearn), cached as .npz
"""
import glob
import os

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib import font_manager                                # noqa: E402
from matplotlib.collections import PolyCollection                  # noqa: E402
from matplotlib.colors import to_rgb                               # noqa: E402
from matplotlib.patches import Polygon                             # noqa: E402

MM = 1 / 25.4
W2, W1 = 180 * MM, 88 * MM
HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- palette
NAVY, NAVY2, NAVY3 = "#0d1828", "#15243a", "#22364f"      # hero background, lifted panels, faint lines
IVORY, IVORY2 = "#f4efe4", "#e8dfcd"                      # data-figure paper, inset fill
BRASS, BRASS_LT, BRASS_DK = "#b08d57", "#e6cd96", "#6e5530"
PEWTER = "#8a8f96"                                        # baselines
PALE_BRASS = "#d9c49a"                                    # fixed untrained bank
HEX, HEX_LT, HEX_DK = "#29b6d8", "#a6ecfa", "#0e6a87"     # our model: the only thing that glows
MOLTEN, MOLTEN_LT = "#e8892b", "#ffcf8a"                  # one accent per figure
EXT = "#4fae7b"                                           # independent cohort
INK, INK2, INK3 = "#1b1f27", "#4b505a", "#8a8d93"
GLASS = "#cfeff8"

DECO = "Federo"
LBL_FIXED = "fixed untrained bank"


def setup():
    for f in glob.glob(os.path.join(HERE, "fonts", "*.ttf")):
        font_manager.fontManager.addfont(f)
    for d in ("/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/liberation2"):
        for f in glob.glob(os.path.join(d, "LiberationSans*.ttf")):
            font_manager.fontManager.addfont(f)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 6,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": BRASS_DK, "xtick.color": BRASS_DK,
        "ytick.color": BRASS_DK, "axes.facecolor": IVORY, "figure.facecolor": IVORY, "savefig.facecolor": IVORY,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
        "axes.labelpad": 2.5, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "legend.frameon": False, "axes.unicode_minus": True, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "figure.dpi": 150, "lines.solid_capstyle": "round",
        "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold", "mathtext.sf": "sans",
    })


def mix(c0, c1, t):
    """Linear blend of two colours, t in [0, 1] (scalar or array) -> RGB array."""
    c0, c1 = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    t = np.clip(np.asarray(t, float), 0, 1)[..., None]
    return c0 * (1 - t) + c1 * t


# ----------------------------------------------------------------------------- light effects
def glow_plot(ax, x, y, color, lw=0.8, core=None, layers=6, spread=4.0, alpha=0.07, z=3, **kw):
    """Line with a soft halo: `layers` wide transparent strokes under a crisp core (optionally lighter)."""
    for i in range(layers, 0, -1):
        ax.plot(x, y, color=color, lw=lw + spread * i / layers, alpha=alpha, zorder=z, solid_capstyle="round",
                **kw)
    ax.plot(x, y, color=color, lw=lw, zorder=z + 0.01, solid_capstyle="round", **kw)
    if core:
        ax.plot(x, y, color=core, lw=lw * 0.38, zorder=z + 0.02, solid_capstyle="round", **kw)


def halo(ax, x, y, r, color, a=0.55, z=1, k=2.6, n=181, squash=1.0):
    """Radial glow as an RGBA image centred on (x, y), radius r (data units). squash < 1 flattens it."""
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    d2 = X ** 2 + (Y / squash) ** 2
    A = a * np.exp(-k * d2 * 2.2) * (d2 <= 1.0 / squash ** 0)
    img = np.zeros((n, n, 4))
    img[..., :3] = to_rgb(color)
    img[..., 3] = np.clip(A, 0, 1)
    ax.imshow(img, extent=[x - r, x + r, y - r, y + r], origin="lower", zorder=z, interpolation="bilinear",
              aspect="auto")


def sunburst(ax, x, y, r0, r1, n=36, color=BRASS, a=0.10, lw=0.35, z=0.5, skip=None):
    """Art Deco rays around (x, y), every other ray shorter."""
    for i in range(n):
        if skip and skip(i):
            continue
        t = 2 * np.pi * i / n
        rr = r1 if i % 2 == 0 else r0 + (r1 - r0) * 0.55
        ax.plot([x + r0 * np.cos(t), x + rr * np.cos(t)], [y + r0 * np.sin(t), y + rr * np.sin(t)], color=color,
                lw=lw, alpha=a, zorder=z, solid_capstyle="butt")


def vgrad(ax, x0, x1, y0, y1, c_bot, c_top, clip=None, z=0, a_bot=1.0, a_top=1.0, n=256):
    """Vertical gradient rectangle (optionally clipped to a patch)."""
    t = np.linspace(0, 1, n)[:, None]
    img = np.zeros((n, 2, 4))
    img[..., :3] = mix(c_bot, c_top, t[:, 0])[:, None, :]
    img[..., 3] = (a_bot + (a_top - a_bot) * t)
    im = ax.imshow(img, extent=[x0, x1, y0, y1], origin="lower", zorder=z, interpolation="bilinear", aspect="auto")
    if clip is not None:
        im.set_clip_path(clip)
    return im


# ----------------------------------------------------------------------------- metal and ornaments
def brass_ring(ax, x, y, z=4, lw=2.4, closed=True):
    """Polyline drawn as bevelled brass: dark rim, body, thin highlight."""
    if closed:
        x, y = np.r_[x, x[:1]], np.r_[y, y[:1]]
    ax.plot(x, y, color=BRASS_DK, lw=lw, zorder=z, solid_joinstyle="round")
    ax.plot(x, y, color=BRASS, lw=lw * 0.62, zorder=z + 0.01, solid_joinstyle="round")
    ax.plot(x, y, color=BRASS_LT, lw=lw * 0.18, zorder=z + 0.02, solid_joinstyle="round", alpha=0.9)


def ellipse_xy(cx, cy, w, h, n=200, t0=0.0, t1=2 * np.pi, rot=0.0):
    t = np.linspace(t0, t1, n)
    x, y = w / 2 * np.cos(t), h / 2 * np.sin(t)
    c, s = np.cos(rot), np.sin(rot)
    return cx + c * x - s * y, cy + s * x + c * y


def chamfer_xy(x0, y0, x1, y1, c):
    return np.array([[x0 + c, y0], [x1 - c, y0], [x1, y0 + c], [x1, y1 - c], [x1 - c, y1], [x0 + c, y1],
                     [x0, y1 - c], [x0, y0 + c], [x0 + c, y0]])


def deco_frame(ax, x0, y0, x1, y1, c=2.5, gap=0.9, color=BRASS, lw=(0.7, 0.3), z=0.8, a=1.0, fans=0.0):
    """Double chamfered rule (outer thicker). fans > 0: Art Deco quarter fans of that radius in the corners."""
    p = chamfer_xy(x0, y0, x1, y1, c)
    ax.plot(p[:, 0], p[:, 1], color=color, lw=lw[0], zorder=z, alpha=a, solid_joinstyle="miter")
    p = chamfer_xy(x0 + gap, y0 + gap, x1 - gap, y1 - gap, max(c - gap * 0.6, 0.3))
    ax.plot(p[:, 0], p[:, 1], color=color, lw=lw[1], zorder=z, alpha=a, solid_joinstyle="miter")
    if fans:
        g = gap + 0.9
        for (cx, cy, t0) in ((x0 + g, y0 + g, 0), (x1 - g, y0 + g, 90), (x1 - g, y1 - g, 180),
                             (x0 + g, y1 - g, 270)):
            for k, rr in enumerate((fans, fans * 0.72, fans * 0.46)):
                t = np.radians(np.linspace(t0 + 8, t0 + 82, 40))
                ax.plot(cx + rr * np.cos(t), cy + rr * np.sin(t), color=color, lw=lw[1] * (1.6 - 0.3 * k),
                        alpha=a, zorder=z)
            for tt in np.radians(np.linspace(t0 + 8, t0 + 82, 6)):
                ax.plot([cx + fans * 0.46 * np.cos(tt), cx + fans * np.cos(tt)],
                        [cy + fans * 0.46 * np.sin(tt), cy + fans * np.sin(tt)], color=color, lw=lw[1], alpha=a,
                        zorder=z)


def hex_xy(x, y, r, rot=np.pi / 6, sy=1.0):
    t = rot + np.arange(7) * np.pi / 3
    return np.c_[x + r * np.cos(t), y + r * sy * np.sin(t)]


def hex_badge(fig, x_mm, y_mm, s, r=2.1, fs=7.5):
    """Panel letter inside a brass hexagon; position in figure mm (top-left anchor of the hexagon)."""
    W, H = fig.get_size_inches() / MM
    ax = fig.add_axes([0, 0, 1, 1], zorder=20)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.patch.set_alpha(0)
    cx, cy = x_mm + r, y_mm - r
    ax.add_patch(Polygon(hex_xy(cx, cy, r), closed=True, fc=NAVY, ec=BRASS, lw=0.8, zorder=1))
    ax.add_patch(Polygon(hex_xy(cx, cy, r * 0.8), closed=True, fc="none", ec=BRASS_DK, lw=0.3, zorder=2))
    ax.text(cx, cy - 0.15, s, ha="center", va="center", fontsize=fs, color=BRASS_LT, family=DECO, zorder=3)
    return ax


# ----------------------------------------------------------------------------- tiny 3D renderer
def camera(az, el):
    """Rows: screen-right, screen-up, toward-viewer, in world (RAS) coordinates. az from +x toward +y, el up."""
    az, el = np.radians(az), np.radians(el)
    f = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    r = np.cross([0.0, 0.0, 1.0], f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.stack([r, u, f])


def face_normals(v, f):
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])     # area-weighted
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def subdivide(v, f, vn, vals=None):
    """One midpoint subdivision (4 triangles per face); normals and per-vertex values are interpolated."""
    e = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
    ue, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.ravel()
    nv = len(v)
    v2 = np.r_[v, (v[ue[:, 0]] + v[ue[:, 1]]) / 2]
    n2 = np.r_[vn, vn[ue[:, 0]] + vn[ue[:, 1]]]
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12
    m = nv + inv.reshape(3, -1).T                                         # midpoints of edges 01, 12, 20
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    m01, m12, m20 = m[:, 0], m[:, 1], m[:, 2]
    f2 = np.r_[np.c_[a, m01, m20], np.c_[m01, b, m12], np.c_[m20, m12, c], np.c_[m01, m12, m20]]
    out = [v2, f2, n2]
    if vals is not None:
        out.append(np.r_[vals, (vals[ue[:, 0]] + vals[ue[:, 1]]) / 2])
    return out


def _unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


LIGHTS = (  # camera-space directions and weights: key (upper left, front), fill (right), top
    (_unit([-0.55, 0.65, 0.55]), 0.85),
    (_unit([0.8, -0.1, 0.45]), 0.25),
    (_unit([0.0, 1.0, 0.1]), 0.20),
)


def mesh_faces(v, f, R, to_mm, base_rgb, alpha=1.0, cull=True, ambient=0.28, spec=0.35, shininess=28,
               rim=None, rim_pow=2.6, rim_w=0.8, glass=False, vn=None):
    """Shade one triangle mesh. Returns (polys [N,3,2] in mm, rgba [N,4], depth [N]).
    base_rgb: [3] or per-face [N,3]. rim: colour of the Fresnel rim light (None = off).
    glass=True: no culling, alpha rises with the Fresnel term (see-through shell).
    vn: per-vertex normals -> smooth shading (face normal = mean of its vertex normals)."""
    pc = v @ R.T
    fnorm = face_normals(v, f) @ R.T
    if vn is not None:
        n = vn[f].mean(1)
        n = (n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)) @ R.T
    else:
        n = fnorm
    keep = np.ones(len(f), bool) if (glass or not cull) else fnorm[:, 2] > 0.0
    f, n = f[keep], n[keep]
    base = np.asarray(base_rgb, float)
    base = base[keep] if base.ndim == 2 else np.broadcast_to(base, (len(f), 3))
    V = np.array([0, 0, 1.0])
    nz = np.abs(n[:, 2]) if glass else np.clip(n[:, 2], 0, 1)
    dif = np.zeros(len(f))
    sp = np.zeros(len(f))
    for L, w in LIGHTS:
        nl = n @ L
        if glass:
            nl = np.abs(nl)
        dif += w * np.clip(nl, 0, 1)
        h = _unit(L + V)
        sp += w * np.clip(n @ h, 0, 1) ** shininess
    col = base * (ambient + dif)[:, None] + spec * sp[:, None]
    fres = (1 - nz) ** rim_pow
    if rim is not None:
        col = col + rim_w * fres[:, None] * np.array(to_rgb(rim))
    col = np.clip(col, 0, 1)
    a = np.full(len(f), float(alpha))
    if glass:
        a = np.clip(alpha * (0.12 + 0.88 * fres ** 1.2) + 0.25 * sp, 0, 1)
    p2 = to_mm(pc[:, :2])
    polys = p2[f]
    depth = pc[f, 2].mean(1)
    return polys, np.c_[col, a], depth


def draw_faces(ax, parts, z=2, lw=0.12, rasterized=True):
    """Painter's sort over several mesh_faces() outputs; one PolyCollection."""
    P = np.concatenate([p[0] for p in parts])
    C = np.concatenate([p[1] for p in parts])
    D = np.concatenate([p[2] for p in parts])
    o = np.argsort(D)
    E = C[o].copy()
    E[E[:, 3] < 0.99, 3] = 0.0                    # strokes only on opaque faces (hide seams there)
    pc = PolyCollection(P[o], facecolors=C[o], edgecolors=E, linewidths=lw, zorder=z,
                        antialiaseds=False)
    pc.set_rasterized(rasterized)
    ax.add_collection(pc)
    return pc


def uv_ellipsoid(c, radii, nu=64, nv=40, z_cut=None):
    """Triangulated ellipsoid (outward faces); rows with z < z_cut are dropped."""
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    t = np.linspace(0, np.pi, nv)
    U, T = np.meshgrid(u, t)
    pts = np.c_[(np.sin(T) * np.cos(U)).ravel(), (np.sin(T) * np.sin(U)).ravel(), np.cos(T).ravel()]
    v = c + pts * radii
    faces = []
    for i in range(nv - 1):
        for j in range(nu):
            a, b = i * nu + j, i * nu + (j + 1) % nu
            cc, d = a + nu, b + nu
            faces += [[a, cc, b], [b, cc, d]]
    faces = np.array(faces)
    if z_cut is not None:
        faces = faces[(v[faces][:, :, 2] >= z_cut).all(1)]
    n = face_normals(v, faces)
    out = (v[faces].mean(1) - c) * np.array(radii) ** -2
    flip = (n * out).sum(1) < 0
    faces[flip] = faces[flip][:, ::-1]
    return v, faces


def raytrace_ellipsoid(R, c, radii, s, off, extent, res=900, z_cut=None, tint=GLASS, rim=HEX_LT,
                       a0=0.05, a1=0.55, spec_a=0.55):
    """Glass ellipsoid rendered analytically (orthographic rays along camera -z).
    s, off: world->mm mapping used for the meshes (mm = s * camera_xy + off). extent = (x0, x1, y0, y1) in mm.
    Returns (back_rgba, front_rgba, extent): draw back below the brain, front above it."""
    x0, x1, y0, y1 = extent
    ny = int(res * (y1 - y0) / (x1 - x0))
    X, Y = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, ny))
    cx, cy = (X - off[0]) / s, (Y - off[1]) / s                           # camera-space x, y
    Rt = R.T
    o = cx[..., None] * Rt[:, 0] + cy[..., None] * Rt[:, 1]                # ray origin (camera z = 0) in world
    d = Rt[:, 2]                                                          # ray direction toward viewer
    inv = 1.0 / np.asarray(radii, float) ** 2
    oc = o - c
    A = (d * d * inv).sum()
    Bq = 2 * (oc * d * inv).sum(-1)
    Cq = (oc * oc * inv).sum(-1) - 1
    disc = Bq ** 2 - 4 * A * Cq
    hit = disc > 0
    sq = np.sqrt(np.where(hit, disc, 0))
    out = []
    V = np.array([0, 0, 1.0])
    for t, sign in (((-Bq - sq) / (2 * A), -1), ((-Bq + sq) / (2 * A), 1)):   # back (far) then front (near)
        p = o + t[..., None] * d
        n = (p - c) * inv
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        nc = n @ R.T
        ok = hit & (True if z_cut is None else p[..., 2] >= z_cut)
        fres = (1 - np.abs(nc[..., 2])) ** 2.2
        sp = np.zeros_like(fres)
        for L, w in LIGHTS:
            h = _unit(L + V)
            sp += w * np.clip(np.abs(nc @ h), 0, 1) ** 60
        col = mix(tint, rim, fres) + spec_a * sp[..., None]
        a = (a0 + (a1 - a0) * fres + spec_a * sp) * (0.55 if sign < 0 else 1.0)
        img = np.zeros(X.shape + (4,))
        img[..., :3] = np.clip(col, 0, 1)
        img[..., 3] = np.where(ok, np.clip(a, 0, 1), 0)
        out.append(img)
    return out[0], out[1], (x0, x1, y0, y1)


def load_brain(cache=None):
    """fsaverage5 pial surfaces (both hemispheres) with per-vertex sulcal depth, outward-oriented faces."""
    cache = cache or os.path.join(os.path.expanduser("~"), ".cache", "s2i", "fs5_pial.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {h: (d[f"v_{h}"], d[f"f_{h}"], d[f"s_{h}"]) for h in ("left", "right")}
    from nilearn import datasets, surface
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    out, save = {}, {}
    for h in ("left", "right"):
        m = surface.load_surf_mesh(fs[f"pial_{h}"])
        v = np.asarray(getattr(m, "coordinates", None) if hasattr(m, "coordinates") else m[0], float)
        f = np.asarray(getattr(m, "faces", None) if hasattr(m, "faces") else m[1], int)
        s = np.asarray(surface.load_surf_data(fs[f"sulc_{h}"]), float)
        n = face_normals(v, f)
        if ((v[f].mean(1) - v.mean(0)) * n).sum(1).mean() < 0:
            f = f[:, ::-1].copy()
        out[h] = (v, f, s)
        save.update({f"v_{h}": v, f"f_{h}": f, f"s_{h}": s})
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **save)
    return out


def save(fig, out, name, facecolor=None):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), facecolor=facecolor or fig.get_facecolor())
    plt.close(fig)
    print("  ->", os.path.join(out, name + ".{pdf,svg,png}"))
