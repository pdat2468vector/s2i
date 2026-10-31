"""
Minimal 3D rendering for the paper schematic (Fig 1): a fsaverage5 cortex drawn with matplotlib only.

Orthographic camera, smooth (vertex-normal) Lambert + weak specular shading, painter's sort, one rasterized
PolyCollection; everything else in the figure stays vector. Brain surface: fsaverage5 pial + sulcal depth
(FreeSurfer, via nilearn), cached as .npz after the first call.
"""
import os

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb


def mix(c0, c1, t):
    """Linear blend of two colours, t in [0, 1] (scalar or array) -> RGB array."""
    c0, c1 = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    t = np.clip(np.asarray(t, float), 0, 1)[..., None]
    return c0 * (1 - t) + c1 * t


def _unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def camera(az, el):
    """Rows: screen-right, screen-up, toward-viewer, in world (RAS) coordinates. az from +x toward +y, el up."""
    az, el = np.radians(az), np.radians(el)
    f = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    r = np.cross([0.0, 0.0, 1.0], f)
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(f, r), f])


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
    f2 = np.r_[np.c_[a, m[:, 0], m[:, 2]], np.c_[m[:, 0], b, m[:, 1]], np.c_[m[:, 2], m[:, 1], c],
               np.c_[m[:, 0], m[:, 1], m[:, 2]]]
    out = [v2, f2, n2]
    if vals is not None:
        out.append(np.r_[vals, (vals[ue[:, 0]] + vals[ue[:, 1]]) / 2])
    return out


LIGHTS = (  # camera-space directions and weights: key (upper left, front), fill (right), top
    (_unit([-0.5, 0.6, 0.62]), 0.78),
    (_unit([0.8, -0.1, 0.5]), 0.22),
    (_unit([0.0, 1.0, 0.2]), 0.12),
)


def mesh_faces(v, f, R, to_mm, base_rgb, vn=None, ambient=0.34, spec=0.10, shininess=24):
    """Shade one opaque mesh (back faces culled). Returns (polys [N,3,2] in mm, rgb [N,3], depth [N])."""
    pc = v @ R.T
    fn = face_normals(v, f) @ R.T
    if vn is not None:
        n = vn[f].mean(1)
        n = (n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)) @ R.T
    else:
        n = fn
    keep = fn[:, 2] > 0.0
    f, n = f[keep], n[keep]
    base = np.asarray(base_rgb, float)
    base = base[keep] if base.ndim == 2 else np.broadcast_to(base, (len(f), 3))
    V = np.array([0, 0, 1.0])
    dif = np.zeros(len(f))
    sp = np.zeros(len(f))
    for L, w in LIGHTS:
        dif += w * np.clip(n @ L, 0, 1)
        sp += w * np.clip(n @ _unit(L + V), 0, 1) ** shininess
    col = np.clip(base * (ambient + dif)[:, None] + spec * sp[:, None], 0, 1)
    return to_mm(pc[:, :2])[f], col, pc[f, 2].mean(1)


def draw_faces(ax, parts, z=2, lw=0.12):
    """Painter's sort over several mesh_faces() outputs; one rasterized PolyCollection."""
    P = np.concatenate([p[0] for p in parts])
    C = np.concatenate([p[1] for p in parts])
    D = np.concatenate([p[2] for p in parts])
    o = np.argsort(D)
    pc = PolyCollection(P[o], facecolors=C[o], edgecolors=C[o], linewidths=lw, zorder=z, antialiaseds=False)
    pc.set_rasterized(True)
    ax.add_collection(pc)
    return pc


def ellipsoid_outline(R, c, radii, s, off, n=360):
    """Screen silhouette of an ellipsoid under the orthographic camera, as an (n, 2) polyline in mm."""
    M = np.diag(radii)                                   # x = c + M u, |u| = 1
    A = R[:2] @ M                                        # screen = A u (+ const)
    U, S, Vt = np.linalg.svd(A)
    t = np.linspace(0, 2 * np.pi, n)
    pts = (U @ np.diag(S) @ np.c_[np.cos(t), np.sin(t)].T).T
    return s * (pts + (R[:2] @ c)) + off


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
        v = np.asarray(m.coordinates if hasattr(m, "coordinates") else m[0], float)
        f = np.asarray(m.faces if hasattr(m, "faces") else m[1], int)
        s = np.asarray(surface.load_surf_data(fs[f"sulc_{h}"]), float)
        if ((v[f].mean(1) - v.mean(0)) * face_normals(v, f)).sum(1).mean() < 0:
            f = f[:, ::-1].copy()
        out[h] = (v, f, s)
        save.update({f"v_{h}": v, f"f_{h}": f, f"s_{h}": s})
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **save)
    return out


def lateral_2d(n=1100, sigma=1.5, cache=None):
    """Flat lateral view of the right hemisphere (anterior to the right), for 2D vector drawing.

    The hemisphere is rasterized once off-screen (front faces, painter's sort, grey = sulcal depth); returns
    pixel-centre coordinates in brain mm (xs, ys), a smoothed inside-mask M and smoothed sulcal field T (0..1,
    > 0.5 = sulcus). Contour M at 0.5 for the outline and T at 0.5 for the gyral pattern.
    """
    cache = cache or os.path.join(os.path.expanduser("~"), ".cache", "s2i", f"fs5_lat2d_{n}_{sigma:g}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["xs"], d["ys"], d["M"], d["T"]
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from scipy.ndimage import gaussian_filter
    v, f, su = load_brain()["right"]
    R = camera(0.0, 0.0)                                  # viewer at +x; screen right = anterior
    vn = vertex_normals(v, f)
    v, f, vn, su = subdivide(v, f, vn, su)
    pc = v @ R.T
    f = f[(face_normals(v, f) @ R.T)[:, 2] > 0]
    P, D = pc[:, :2][f], pc[f, 2].mean(1)
    o = np.argsort(D)
    t = 1 / (1 + np.exp(-su[f].mean(1) / 0.5))
    lo, hi = pc[:, :2].min(0) - 4, pc[:, :2].max(0) + 4
    h = int(round(n * (hi[1] - lo[1]) / (hi[0] - lo[0])))
    fig = Figure(figsize=(n / 100, h / 100), dpi=100)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor((1.0, 0.0, 0.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.axis("off")
    c = np.c_[t, t, t][o]
    ax.add_collection(PolyCollection(P[o], facecolors=c, edgecolors=c, linewidths=0.4, antialiaseds=False))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[::-1, :, :3].astype(float) / 255
    inside = ((img[:, :, 0] - img[:, :, 1]) < 0.5).astype(float)
    M = gaussian_filter(inside, sigma)
    T = gaussian_filter(img[:, :, 1] * inside, sigma * 1.1) / (gaussian_filter(inside, sigma * 1.1) + 1e-6)
    xs = np.linspace(lo[0], hi[0], img.shape[1])
    ys = np.linspace(lo[1], hi[1], img.shape[0])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, xs=xs, ys=ys, M=M, T=T)
    return xs, ys, M, T
