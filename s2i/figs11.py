"""
Paper figures, v11 design system (Journal of Neural Engineering; plan: paper/S2I_figure_plan_v11.md).

Design rules
  * IOP sizes: double column 150 mm, single column 85 mm; text 8-12 pt at print size (panel letter 10 pt bold,
    panel title 9 pt, labels and ticks 8 pt); one sans family
  * axes are placed in millimetres on the page (Canvas.ax), so the layout is fixed and fills the figure
  * short panel titles on the letter line, no subtitles (conditions go to the caption)
  * one semantic palette (ours = ink blue, fixed bank = light blue, baselines = one warm grey, one accent)
  * per-patient points as a deterministic beeswarm; mean + 95 % CI; direct labels, no legends
  * after saving, check_layout() reports overlapping texts and the largest empty area of the page

Each figure reads saved result files only and writes <name>.{pdf,svg,png} plus <name>_numbers.json (every number
printed on the figure). Bootstrap CIs use a seed derived from the comparison name, so they do not depend on the
order in which they are computed.

Usage
  python -m s2i.figs11 --runs RUNS_DIR --out FIG_DIR [--only fig2]
"""
import argparse
import glob
import json
import os
import zlib

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

from s2i import figstyle as st                                     # noqa: E402  (beeswarm, fmt, fonts)
from s2i.paperfigs import (load_dir, load_strict, paired, STRICT)  # noqa: E402

MM = 1 / 25.4
W2, W1 = 150.0, 85.0
FS, FS_T, FS_L = 8, 9, 10                     # labels/ticks, panel titles, panel letters (pt)

OURS, FIXED, BASE, ACC = "#1f4e79", "#8fb3d9", "#8c8c8c", "#c8553d"
FIXED_TXT = "#4f7fb3"                          # text in the fixed-bank colour (readable on white)
INK, INK2, MUTE = "#1d1d1f", "#5a5a5a", "#d9d9d9"
BASE_PT = "#b9b9b9"                           # baseline patient dots (lighter than the baseline mean)


def setup():
    st.setup()                                # registers Liberation Sans on Colab
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#3a3a3a",
        "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a", "text.color": INK, "axes.labelcolor": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True, "savefig.dpi": 600,
        "lines.solid_capstyle": "round"})


def boot_ci(v, name, n=10000):
    """Patient-bootstrap 95 % CI of the mean, seeded by the comparison name (order-independent)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_stats(a, b, name):
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = boot_ci(d, name)
    return {"mean_diff": float(d.mean()), "ci": [lo, hi], "wins": int((d > 0).sum()), "n": int(len(d))}


fmt = st.fmt


def r_of(d, model, frac="0.2"):
    """Explicit-prediction r if the run stored it for this fraction (as in the manuscript), else the
    sufficient-statistics r; None if the model is missing."""
    ex = d.get("explicit", {}).get(model)
    if ex is not None and abs(ex.get("frac", 0.2) - float(frac)) < 1e-9:
        return ex["mean"]["r"]
    m = d.get("fracs", {}).get(frac, {}).get("models", {}).get(model)
    return m["r"] if m else None


def model_rx(D, model, frac="0.2"):
    out = {s: r_of(d, model, frac) for s, d in D.items()}
    return {s: v for s, v in out.items() if v is not None}


def frac_runs(main_run, name="readout_combo5_strict"):
    """{fraction: {subject: result}} with each calibration fraction read from its own run folder."""
    base = os.path.join(main_run, name + STRICT)
    return {"0.2": load_dir(base, "readout_"), "0.1": load_dir(base + "_f10", "readout_"),
            "0.05": load_dir(base + "_f05", "readout_")}


# ----------------------------------------------------------------------------- canvas in millimetres
class Canvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w * MM, h * MM))

    def ax(self, x, y, w, h):
        """Axes with lower-left corner (x, y) and size (w, h) in mm from the lower-left page corner."""
        return self.fig.add_axes([x / self.w, y / self.h, w / self.w, h / self.h])

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FS)
        return self.fig.text(x / self.w, y / self.h, s, **kw)

    def letter(self, x, y, s, title=None):
        self.text(x, y, s, fontsize=FS_L, fontweight="bold", va="baseline")
        if title:
            self.text(x + 4.6, y, title, fontsize=FS_T, va="baseline")

    def key(self, x, y, items, gap=3.0, text_colors=None):
        """Inline key: items = [(label, colour, marker)], laid out left to right from (x, y) in mm."""
        r = self.fig.canvas.get_renderer()
        for lab, col, mk in items:
            self.fig.lines.append(Line2D([(x + 1.0) / self.w], [(y + 1.0) / self.h], marker=mk,
                                         ms=4 if mk in "oDs" else 6, mew=1.3, color=col,
                                         transform=self.fig.transFigure))
            t = self.text(x + 2.6, y, lab, color=(text_colors or {}).get(lab, col), va="baseline")
            x += 2.6 + t.get_window_extent(r).width / self.fig.dpi * 25.4 + gap


def check_layout(fig, name, cell=4.0):
    """Report overlapping text boxes and the largest empty square of the page (mm), from the rendered image."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    gap = 0.3 * fig.dpi / 25.4                                # minimum clearance between texts (0.3 mm)
    over = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > -gap and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
                over.append((boxes[i][0].get_text()[:30], boxes[j][0].get_text()[:30]))
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    dpi = fig.dpi
    px = max(1, int(round(cell * dpi / 25.4)))
    H, W = img.shape[0] // px, img.shape[1] // px
    ink = np.array([[(img[i * px:(i + 1) * px, j * px:(j + 1) * px] < 245).any() for j in range(W)]
                    for i in range(H)])
    best = np.zeros((H + 1, W + 1), int)                      # largest all-empty square (DP)
    for i in range(H):
        for j in range(W):
            if not ink[i, j]:
                best[i + 1, j + 1] = 1 + min(best[i, j + 1], best[i + 1, j], best[i, j])
    side = best.max() * cell
    print(f"  [{name}] layout: {len(over)} text overlaps; largest empty square {side:.0f} mm; "
          f"empty cells {100 * (1 - ink.mean()):.0f} %")
    for a, b in over:
        print(f"     overlap: '{a}' / '{b}'")
    return {"overlaps": over, "largest_empty_square_mm": side, "empty_fraction": float(1 - ink.mean())}


def save(cv, out, name, nums):
    os.makedirs(out, exist_ok=True)
    nums["_layout"] = check_layout(cv.fig, name)
    for ext in ("pdf", "svg", "png"):
        cv.fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(cv.fig)
    json.dump(nums, open(os.path.join(out, f"{name}_numbers.json"), "w"), indent=1)
    print("  ->", os.path.join(out, name + ".{pdf,svg,png}"))


def tl(v, d=2):
    """Tick label: shortest decimal up to d places, leading zero, true minus ("0.2", "0.15", "\u22120.1")."""
    if abs(v) < 1e-12:
        return "0"
    s = f"{v:.{d}f}".rstrip("0").rstrip(".")
    return s.replace("-", "\u2212")


def nice_ticks(lo, hi, maxn=4, steps=(0.01, 0.02, 0.04, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0)):
    """Ticks inside [lo, hi] on the finest step that gives at most maxn ticks."""
    for s in steps:
        v = np.arange(np.ceil(lo / s - 1e-9), np.floor(hi / s + 1e-9) + 1) * s
        if len(v) <= maxn:
            return np.round(v, 6)
    return np.round(v, 6)


def cat_axis(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)


# ----------------------------------------------------------------------------- Fig. 2
def fig2(runs, out):
    main_run, noenv = os.path.join(runs, "shared_bip_car"), os.path.join(runs, "shared_no_env")
    S5 = load_strict(main_run)
    SF = load_dir(os.path.join(noenv, "readout_fixed_strict" + STRICT), "readout_")
    if not S5 or not SF:
        raise FileNotFoundError(2, "missing", os.path.join(main_run, "readout_combo5_strict" + STRICT))
    subs, (base, fixed, final) = paired(model_rx(S5, "base"), model_rx(SF, "combo_nosess"), model_rx(S5, "combo_nosess"))
    gain = paired_stats(final, base, "fig2a final-base primary strict")
    nums = {"a": {"n": len(subs), "mean_base": base.mean(), "mean_fixed": fixed.mean(), "mean_final": final.mean(),
                  "final_minus_base": gain}}

    cv = Canvas(W2, 96)

    # ---------------- a: dumbbell per patient, sorted by gain; gain bars; mean row
    order = np.argsort(final - base)
    yy = np.arange(len(subs)) + 1.4
    ax = cv.ax(10, 10, 44, 75)
    for y, i in zip(yy, order):
        ax.plot([base[i], final[i]], [y, y], color=MUTE, lw=2.2, zorder=1)
        ax.plot(fixed[i], y, marker="|", ms=6, mew=1.3, color=FIXED_TXT, zorder=2)
        ax.plot(base[i], y, "o", ms=3.6, color=BASE, zorder=3)
        ax.plot(final[i], y, "o", ms=3.8, color=OURS, zorder=4)
    mb, mx, mf = base.mean(), fixed.mean(), final.mean()
    ax.plot([mb, mf], [0, 0], color=OURS, lw=2.6, alpha=0.25)
    ax.plot(mx, 0, marker="|", ms=7, mew=1.6, color=FIXED_TXT)
    ax.plot(mb, 0, "o", ms=4.6, color=BASE)
    ax.plot(mf, 0, "o", ms=4.8, color=OURS)
    ax.axhline(0.7, color=MUTE, lw=0.6)
    lab = [s.replace("sub-", "P") for s in subs]
    ax.set_yticks([0] + list(yy), ["mean"] + [lab[i] for i in order])
    ax.get_yticklabels()[0].set_fontweight("bold")
    lo = np.floor(min(base.min(), fixed.min()) * 20) / 20
    hi = np.ceil(max(final.max(), fixed.max()) * 20) / 20
    ax.set_xticks(np.round(np.arange(np.ceil(lo * 10) / 10, hi + 1e-9, 0.1), 1))
    ax.set_xlim(lo - 0.01, hi + 0.005)
    ax.set_xlabel("test r (mean over contacts)")
    ax.set_ylim(-0.6, yy[-1] + 0.6)
    cat_axis(ax)
    cv.key(10, 87.0, [("LAG9 ridge", BASE, "o"), ("fixed bank", FIXED_TXT, "|"), ("final model", OURS, "o")], gap=4,
           text_colors={"fixed bank": FIXED_TXT})

    g = (final - base)[order]
    axg = cv.ax(59, 10, 12, 75)
    axg.barh(yy, g, height=0.62, color=ACC, alpha=0.85)
    lo_ci, hi_ci = gain["ci"]
    axg.errorbar(gain["mean_diff"], 0, xerr=[[gain["mean_diff"] - lo_ci], [hi_ci - gain["mean_diff"]]], fmt="s",
                 ms=4, color=ACC, capsize=0, lw=1.2)
    gmax = max(0.08, np.ceil(g.max() * 50) / 50)
    axg.text(hi_ci + 0.004, 0, fmt(gain["mean_diff"], sign=True), color=ACC, va="center", fontweight="bold")
    axg.set_ylim(ax.get_ylim())
    axg.set_xlim(min(0, g.min() * 1.1), gmax)
    axg.set_xticks([0, gmax], ["0", f"{gmax:.2f}"])
    axg.set_yticks([])
    axg.spines["left"].set_visible(False)
    axg.axvline(0, color=INK2, lw=0.6)
    axg.axhline(0.7, color=MUTE, lw=0.6)
    axg.set_xlabel("gain")
    cv.letter(2, 91.5, "a", "Every patient improves")
    nums["a"]["gain_axis_max"] = gmax

    # ---------------- b: every model on the same patients and test trials (primary level, strict normalisation;
    # deep baselines of Phase 36, CAST of Phase 37, the same sources as strict36 / cast37 report)
    from s2i.parm27 import _ex, _load
    shallow = _ex(_load(os.path.join(runs, "shared_shallow"), "p36_learned"), "combo_nosess")
    lf, nfn, cast = {}, {}, {}
    for f in glob.glob(os.path.join(runs, "p36_leadformer", "fold_sub-*.json")):
        fo = json.load(open(f))
        if "calibrated" in fo:
            lf[fo["subject"]] = float(np.nanmean(fo["calibrated"]["r"]))
    for f in glob.glob(os.path.join(runs, "p36_nfn", "nfn_sub-*.json")):
        fo = json.load(open(f))
        if "nfn_mode" in fo.get("models", {}):
            nfn[fo["subject"]] = float(fo["models"]["nfn_mode"]["r"])
    for f in glob.glob(os.path.join(runs, "p37_cast", "cast_sub-*.json")):
        fo = json.load(open(f))
        if "r" in fo:
            cast[fo["subject"]] = float(fo["r"])
    rows = [("final model", model_rx(S5, "combo_nosess"), OURS), ("fixed bank", model_rx(SF, "combo_nosess"), FIXED),
            ("ShallowConvNet-style", shallow, BASE), ("LAG9 ridge", model_rx(S5, "base"), BASE),
            ("LeadFormer", lf, BASE), ("NeuroFlowNet", nfn, BASE), ("CAST", cast, BASE)]
    rows = [r for r in rows if r[1]]
    rows = rows[:2] + sorted(rows[2:], key=lambda r: -np.mean(list(r[1].values())))   # baselines by mean r
    axb = cv.ax(103, 55, 35, 31)
    nums["b"] = {"level": "primary, strict normalisation"}
    xmax, xmin = 0.0, 0.0
    for k, (name, d, col) in enumerate(rows):
        y = len(rows) - 1 - k
        v = np.array(list(d.values()), float)
        off = st.beeswarm(v, size_data=0.006, max_off=0.30)
        axb.scatter(v, y + off, s=5, color=col if col != BASE else BASE_PT, alpha=0.75 if col == OURS else 0.9,
                    lw=0, zorder=2)
        lo_, hi_ = boot_ci(v, f"fig2b {name}")
        mc = {BASE: INK2, FIXED: FIXED_TXT}.get(col, col)
        axb.plot([lo_, hi_], [y, y], color=mc, lw=1.1, zorder=3)
        axb.plot(v.mean(), y, "D", ms=4.2, color=mc, mec="white", mew=0.6, zorder=4)
        axb.text(1.03, y, fmt(v.mean()), va="center", ha="left", color=mc, transform=axb.get_yaxis_transform(),
                 fontweight="bold" if k == 0 else "normal")
        nums["b"][name] = {"n": len(v), "mean": float(v.mean()), "ci": [lo_, hi_]}
        xmax = max(xmax, np.ceil(v.max() * 20) / 20 + 0.01)
        xmin = min(xmin, np.floor(v.min() * 20) / 20)
    axb.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    for t, r in zip(axb.get_yticklabels(), rows[::-1]):
        t.set_color({BASE: INK2, FIXED: FIXED_TXT}.get(r[2], r[2]))
    axb.set_xlim(xmin - (0.01 if xmin < 0 else 0), xmax)
    if xmin < 0:
        axb.axvline(0, color=MUTE, lw=0.6, zorder=0)
    xv = np.round(np.arange(np.ceil(xmin * 10) / 10, xmax + 1e-9, 0.1), 1)
    axb.set_xticks(xv, [tl(v) for v in xv])
    axb.set_ylim(-0.6, len(rows) - 0.4)
    cat_axis(axb)
    axb.set_xlabel("test r per patient")
    cv.letter(76, 91.5, "b", "Comparison with baselines")

    # ---------------- c: calibration efficiency (primary level, strict)
    FR = frac_runs(main_run)
    fr = [f for f in ("0.05", "0.1", "0.2") if FR[f]]
    subs_c = sorted(set.intersection(*[set(FR[f]) for f in fr]))
    xm = np.array([np.mean([FR[f][s]["fracs"][f]["minutes"] for s in subs_c]) for f in fr])
    Y = {k: np.array([[r_of(FR[f][s], k, f) for f in fr] for s in subs_c], float)
         for k in ("base", "combo_nosess")}
    axc = cv.ax(88, 10, 50, 31)
    nums["c"] = {"minutes": xm.tolist()}
    for k, col, name, dx in (("base", BASE, "ridge", -0.07), ("combo_nosess", OURS, "final", 0.07)):
        mu = Y[k].mean(0)
        ci = np.array([boot_ci(Y[k][:, j], f"fig2c {k} {f}") for j, f in enumerate(fr)])
        for x_, (l_, h_) in zip(xm + dx, ci):                      # thin CI whiskers, models nudged apart
            axc.plot([x_, x_], [l_, h_], color=col, lw=0.8, alpha=0.6, solid_capstyle="butt", zorder=2)
        axc.plot(xm, mu, "-o", color=col, ms=3.8, lw=1.4, zorder=3)
        axc.text(xm[-1] + 0.25, mu[-1], {"ridge": "LAG9\nridge", "final": "final\nmodel"}.get(name, name), color=col, va="center", linespacing=1.0)
        nums["c"][name] = mu.tolist()
        nums["c"][name + "_ci"] = ci.tolist()
    if "0.1" in fr and "0.2" in fr:
        j10, j20 = fr.index("0.1"), fr.index("0.2")
        s_ = paired_stats(Y["combo_nosess"][:, j10], Y["base"][:, j20], "fig2c final10-base20 primary strict")
        nums["c"]["final10_minus_base20"] = s_
        y10, yb20 = Y["combo_nosess"][:, j10].mean(), Y["base"][:, j20].mean()
        axc.plot([xm[j10], xm[j20]], [yb20, yb20], color=ACC, lw=0.8, ls=(0, (2, 1.5)), zorder=2)
        axc.plot([xm[j10], xm[j10]], [yb20, y10], color=ACC, lw=1.4, zorder=2)
        axc.text(xm[j10] + 0.25, np.min(nums["c"]["ridge_ci"]) + 0.002,
                 f"final model, {xm[j10]:.1f} min\n− LAG9 ridge, {xm[j20]:.1f} min\n"
                 f"{fmt(s_['mean_diff'], sign=True)} [{fmt(s_['ci'][0])}, {fmt(s_['ci'][1])}], "
                 f"{s_['wins']}/{s_['n']}", color=ACC, va="bottom", ha="left", linespacing=1.25)
    axc.set_xticks(xm, [f"{v:.1f}" for v in xm])
    axc.set_xlim(xm[0] - 0.3, xm[-1] + 1.2)
    allci = np.concatenate([np.ravel(nums["c"][k]) for k in nums["c"] if k.endswith("_ci")])
    ylo, yhi = np.floor(allci.min() * 50) / 50, np.ceil(allci.max() * 50) / 50
    axc.set_ylim(ylo - 0.004, yhi + 0.004)
    yv = np.round(np.arange(np.ceil(ylo * 20) / 20, yhi + 1e-9, 0.05), 2)
    axc.set_yticks(yv, [fmt(v, 2) for v in yv])
    axc.set_xlabel("calibration (min)")
    axc.set_ylabel("test r")
    cv.letter(76, 45.5, "c", "Calibration efficiency")

    save(cv, out, "fig2", nums)


# ----------------------------------------------------------------------------- Fig. 1 (overview)
DATA_DIR = None                               # set by main(); one patient file gives real shafts, traces, sessions
FIG1_PATIENT = "sub-12"
TEN20 = {"Fp1": (-29, 84, -7), "Fp2": (30, 85, -7), "F7": (-70, 42, -11), "F3": (-50, 53, 42), "Fz": (0, 59, 66),
         "F4": (52, 54, 41), "F8": (73, 44, -12), "T7": (-84, -16, -9), "C3": (-65, -12, 64), "Cz": (0, -9, 100),
         "C4": (67, -11, 64), "T8": (85, -15, -9), "P7": (-72, -73, -2), "P3": (-53, -79, 56), "Pz": (0, -81, 83),
         "P4": (56, -79, 57), "P8": (73, -73, -3), "O1": (-29, -112, 9), "O2": (30, -112, 9)}   # MNE standard_1020, mm
SEEG = ACC                                    # Fig. 1 only: intracranial contacts and recorded SEEG
TINT = "#f4f6f9"


def _arrow(ax, p, q, color=INK2, lw=0.75, ls="-", rad=0.0, ms=6, z=5):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def _rbox(ax, x, y, w, h, fc, ec="none", lw=0.6, r=1.2, z=0):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z))


def _morlet_bank(n=12, half=240, fs=200.0):
    """Encoder initialisation (s2i.shared.TemporalEncoder): 12 Morlet pairs 2-60 Hz, 5 cycles, +-1.2 s."""
    t = np.arange(-half, half + 1) / fs
    f = np.geomspace(2, 60, n)
    g = np.exp(-t[None] ** 2 / (2 * (5 / (2 * np.pi * f[:, None])) ** 2))
    return t, f, g * np.cos(2 * np.pi * f[:, None] * t[None])


def _fig1_inputs():
    """Real inputs if DATA_DIR / runs are available, otherwise illustrative stand-ins (flagged in the numbers)."""
    rng = np.random.default_rng(7)
    src = {"real": []}
    shafts, eeg, seeg, sessions, kernels = None, None, None, None, None
    p = os.path.join(DATA_DIR or "", f"{FIG1_PATIENT}.pt")
    if DATA_DIR and os.path.exists(p):
        import torch
        d = torch.load(p, weights_only=False, map_location="cpu")
        pos, names = np.asarray(d["contact_mni"], float), list(d["contact_names"])
        sh = {}
        for n_, xyz in zip(names, pos):
            sh.setdefault(n_.rstrip("0123456789"), []).append(xyz)
        shafts = [np.array(v) for v in sh.values() if len(v) >= 3]
        b = np.asarray(d["trial_bounds"])
        s0, s1 = b[len(b) // 2]
        eeg = np.asarray(d["eeg_car"][:, s0:s0 + 400], float)[:4]
        seeg = np.asarray(d["ieeg_bip"][:, s0:s0 + 400], float)[:3]
        ses = list(d.get("trial_session", []))
        sessions = [ses.count(s) for s in dict.fromkeys(ses)]
        src["real"] += ["shafts", "traces", "sessions"]
    if shafts is None:                        # illustrative: orthogonal shafts in the left hemisphere
        shafts = []
        for y, z in ((25, 5), (-5, -15), (-25, -18), (-40, 10), (5, 35), (-60, 20)):
            x = np.linspace(-66, -22, 8)
            shafts.append(np.c_[x, np.full(8, y) + rng.normal(0, 1, 8), np.full(8, z) + np.linspace(0, -3, 8)])
    if eeg is None:
        t = np.arange(400) / 200
        eeg = np.array([np.sin(2 * np.pi * (6 + 2 * k) * t + k) * 0.6 + np.cumsum(rng.normal(0, 0.25, 400)) * 0.08
                        + rng.normal(0, 0.25, 400) for k in range(4)])
        lat = np.array([np.convolve(rng.normal(0, 1, 400), np.ones(15) / 15, "same") for _ in range(3)])
        seeg = lat * 3
    if sessions is None:
        sessions = [62, 55, 60, 48]
    t, f, kernels = _morlet_bank()
    return src, shafts, eeg, seeg, sessions, (t, f, kernels)


def _trace(ax, x0, y0, w, h, v, color, lw=0.7, z=4):
    v = np.asarray(v, float)
    v = (v - v.mean()) / (np.abs(v - v.mean()).max() + 1e-9)
    ax.plot(x0 + np.linspace(0, w, len(v)), y0 + v * h / 2, color=color, lw=lw, zorder=z, solid_joinstyle="round")


def _bandpass(x, f0, fs=200.0, q=2.5):
    """Zero-phase FFT band-pass around f0 (Gaussian, width f0 / q) - only for drawing the fan-out."""
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return np.fft.irfft(X * np.exp(-0.5 * ((f - f0) / max(f0 / q, 1.0)) ** 2), len(x))


def _sheet(ax, x0, y0, nx, ny, du, dv, vals, cmap, z, ec="white"):
    """One feature sheet in oblique projection: nx time cells along (du, 0), ny filter cells along dv."""
    from matplotlib.collections import PolyCollection
    polys, cols = [], []
    for i in range(ny):
        for j in range(nx):
            o = np.array([x0 + j * du + i * dv[0], y0 + i * dv[1]])
            polys.append([o, o + [du, 0], o + [du + dv[0], dv[1]], o + [dv[0], dv[1]]])
            cols.append(cmap(vals[i, j]))
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=ec, linewidths=0.25, zorder=z))


def fig1(runs, out):
    from s2i import render3d as r3
    from matplotlib.colors import LinearSegmentedColormap
    W, H = W2, 62.0
    cv = Canvas(W, H)
    ax = cv.ax(0, 0, W, H)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    src, shafts, eeg, seeg, sessions, (kt, kf, kern) = _fig1_inputs()
    nums = {"inputs": src, "patient": FIG1_PATIENT, "sessions": sessions}
    TY = 58.0
    cv.letter(0.5, TY, "a", "Data and protocol")
    cv.letter(37.0, TY, "b", "Model for a new patient")
    cv.letter(120.5, TY, "c", "Meta-training")
    blues = mpl.colormaps["Blues"]

    # ---------------- a: head outline, scalp electrodes, brain with SEEG shafts; session timeline
    br = r3.load_brain()
    R = r3.camera(140, 42)
    s, cen = 0.165, np.array([17.6, 35.0])

    def to_mm(p2):
        return s * p2 + cen
    head = r3.ellipsoid_outline(R, np.array([0.0, -14.0, 18.0]), np.array([88.0, 108.0, 96.0]), s, cen)
    ax.plot(head[:, 0], head[:, 1], color="#c9ced6", lw=0.7, zorder=1.5)
    parts = []
    for h_ in ("left", "right"):
        v, f, su = br[h_]
        t_ = 1 / (1 + np.exp(-su[f].mean(1) / 0.6))
        parts.append(r3.mesh_faces(v, f, R, to_mm, r3.mix("#ece8e2", "#c9c2b8", t_), vn=r3.vertex_normals(v, f),
                                   ambient=0.42, spec=0.06))
    r3.draw_faces(ax, parts, z=2)
    fwd = R[2]
    for k, p3 in TEN20.items():
        p3 = np.array(p3, float) + np.array([0, -4, -10])
        if (p3 - np.array([0, -15, 20])) @ fwd > -5:
            ax.plot(*to_mm(R[:2] @ p3), "o", ms=3.3, color=INK2, mec="white", mew=0.5, zorder=6)
    for sh in shafts:
        xy = to_mm(sh @ R[:2].T)
        ax.plot(xy[:, 0], xy[:, 1], color="white", lw=2.0, zorder=7)
        ax.plot(xy[:, 0], xy[:, 1], color=INK2, lw=0.6, zorder=8)
        ax.plot(xy[:, 0], xy[:, 1], "o", ms=1.8, color=SEEG, mew=0, zorder=9)
    ax.plot(1.6, 15.6, "o", ms=3.3, color=INK2, mec="white", mew=0.5)
    ax.text(3.0, 15.6, "scalp EEG", color=INK2, va="center")
    ax.plot(21.4, 15.6, "o", ms=2.2, color=SEEG, mew=0)
    ax.text(22.8, 15.6, "SEEG", color=SEEG, va="center")
    x0, x1, yb, hb = 0.8, 34.5, 4.2, 3.0
    tot, ncal = sum(sessions), int(round(0.2 * sum(sessions)))
    scale = (x1 - x0 - 1.0 * (len(sessions) - 1)) / tot
    xx, k = x0, 0
    for ns in sessions:
        for j in range(ns):
            col = OURS if k < ncal else ("white" if k == ncal else "#b9b9b9")
            ax.add_patch(mpl.patches.Rectangle((xx + j * scale, yb), scale * 1.02, hb, color=col, lw=0, zorder=3))
            k += 1
        xx += ns * scale + 1.0
    ax.text(x0, yb + hb + 0.9, "calibration", color=OURS, va="bottom")
    ax.text(x1, yb + hb + 0.9, "test", color=INK2, va="bottom", ha="right")
    ax.text(x0, yb - 0.9, "first 20 % of trials", color=OURS, va="top")

    # ---------------- b: the computation  y_k(t) = sum_c sum_f W_kcf (h_f * x_c)(t)
    top, bot = 51.0, 25.0                     # vertical extent of the main row
    lab_y = bot - 1.6
    mid = (top + bot) / 2
    # (1) multichannel scalp EEG, one channel highlighted
    ex0, ex1 = 38.5, 49.0
    nch = len(eeg)
    eeg_y = np.linspace(top - 2.5, bot + 2.5, nch)
    hi_c = 1
    for i, (v, y_) in enumerate(zip(eeg, eeg_y)):
        _trace(ax, ex0, y_, ex1 - ex0, 4.0, v, OURS if i == hi_c else "#9a9a9a", lw=0.8 if i == hi_c else 0.55)
    ax.text(ex0, lab_y, "scalp EEG\n" + r"$x_c(t)$", va="top", linespacing=1.15)
    # (2) fan-out of the highlighted channel through the shared filter bank
    fx0, fx1 = 54.0, 66.0
    freqs = [2.0, 5.0, 12.0, 25.0, 50.0]
    fy = np.linspace(top - 2.0, bot + 2.0, len(freqs))
    for i, (f0, y_) in enumerate(zip(freqs, fy)):
        col = blues(0.45 + 0.5 * i / (len(freqs) - 1))
        ax.plot([ex1 + 0.6, fx0 - 0.8], [eeg_y[hi_c], y_], color=col, lw=0.6, zorder=2)
        _trace(ax, fx0, y_, fx1 - fx0, 3.4, _bandpass(np.asarray(eeg[hi_c], float), f0), col, lw=0.7)
    ax.text(fx1 + 0.5, fy[0], "2 Hz", va="center", color=INK2)
    ax.text(fx1 + 0.5, fy[-1], "60 Hz", va="center", color=INK2)
    ax.text(fx0, lab_y, "filter bank\n" + r"$h_f * x_c$", va="top", linespacing=1.15)
    # (3) feature tensor: one sheet per scalp channel (filters x time)
    rng = np.random.default_rng(5)
    tx0, ty0 = 76.0, 28.0
    du, dv, off = 0.72, (0.30, 1.45), 1.5
    for c in range(4)[::-1]:
        vals = np.clip(np.abs(rng.normal(0.35, 0.22, (12, 9))) + np.linspace(0, 0.25, 9)[None], 0, 1)
        _sheet(ax, tx0 + off * c, ty0 + off * c, 9, 12, du, dv, vals * (0.55 + 0.45 * (c == hi_c)) + 0.05,
               blues, z=3 + (3 - c))
    ax.text(tx0, lab_y, "features\n" + r"$\phi_{cf}(t)$", va="top", linespacing=1.15)
    # (4) patient-specific readout W
    wx0, wx1, wy0, wy1 = 93.0, 98.5, 29.0, 48.0
    div = LinearSegmentedColormap.from_list("d", [OURS, "#f7f7f7", SEEG])
    ax.imshow(rng.normal(0, 1, (12, 4)), extent=(wx0, wx1, wy0, wy1), cmap=div, aspect="auto", zorder=3,
              vmin=-2.4, vmax=2.4, interpolation="nearest")
    ax.text((wx0 + wx1) / 2, lab_y, "readout\n" + r"$W_{kcf}$", va="top", ha="center", linespacing=1.15)
    # (5) SEEG shaft with contacts and recorded / predicted traces
    sx = 102.0
    cy = np.linspace(top - 1.0, bot + 1.0, 7)
    ax.plot([sx, sx], [cy[-1] - 1.5, cy[0] + 1.5], color=INK2, lw=0.8, zorder=3)
    ax.plot(np.full(len(cy), sx), cy, "o", ms=2.6, color="white", mec=SEEG, mew=0.8, zorder=4)
    for j, i in enumerate([1, 3, 5]):
        ax.plot(sx, cy[i], "o", ms=2.6, color=SEEG, mew=0, zorder=5)
        v = np.asarray(seeg[j], float)
        pred = np.convolve(v, np.ones(9) / 9, "same") * 0.8
        _trace(ax, sx + 1.8, cy[i], 13.6, 4.2, v, SEEG, lw=0.6)
        _trace(ax, sx + 1.8, cy[i], 13.6, 4.2, pred, OURS, lw=0.9, z=5)
    ax.text(sx + 1.8, top + 3.8, "recorded", color=SEEG, va="bottom")
    ax.text(sx + 1.8, top + 0.7, "predicted", color=OURS, va="bottom")
    ax.text(sx + 8.6, lab_y, "bipolar SEEG\n" + r"$y_k(t)$", va="top", ha="center", linespacing=1.15)
    for xa, xb in ((fx1 + 6.4, tx0 - 0.6), (tx0 + 13.4, wx0 - 0.6), (wx1 + 0.6, sx - 1.4)):
        _arrow(ax, (xa, mid), (xb, mid))
    # the model in one line, with what is shared and what is fitted per patient
    ax.text(78.0, 10.4, r"$\hat y_k(t)=\sum_{c}\sum_{f}\,W_{kcf}\,(h_f * x_c)(t)$", ha="center", va="center",
            fontsize=FS_T)
    for (xa, xb), lab, col in (((fx0, fx1 + 6.0), r"$h_f$: shared across patients", OURS),
                               ((wx0 - 1.0, 117.4), r"$W$: ridge, per patient", INK2)):
        yb_ = 5.6
        ax.plot([xa, xa, xb, xb], [yb_ + 1.1, yb_, yb_, yb_ + 1.1], color=col, lw=0.7)
        ax.text((xa + xb) / 2, yb_ - 0.9, lab, color=col, ha="center", va="top")

    # ---------------- c: episodic meta-training through a closed-form ridge readout
    cx0, cx1 = 121.5, 146.8
    cm = (cx0 + cx1) / 2
    for i, j in enumerate([0, 3, 6, 9]):
        kk = kern[j][np.abs(kern[j]) > 0.02 * np.abs(kern[j]).max()]
        _trace(ax, cx0 + 0.3 + i * 6.3, 51.0, 5.4, 2.8, kk, blues(0.5 + 0.15 * i), lw=0.7)
    ax.text(cx0, 46.6, r"shared filters $h_f$", color=OURS, va="center")
    _arrow(ax, (cm, 44.4), (cm, 41.8))
    for i in range(3)[::-1]:
        _rbox(ax, cx0 + 1.2 * i, 32.2 - 1.2 * i, cx1 - cx0 - 2.4, 8.4, "white", ec="#c9ced6", lw=0.5, r=0.8, z=3 + i)
    xl, wcard = cx0 + 2.4, cx1 - cx0 - 4.8
    ax.text(xl, 36.4, "training patient", va="center", zorder=8)
    ax.add_patch(mpl.patches.Rectangle((xl, 31.6), wcard * 0.64, 1.8, color=OURS, lw=0, zorder=8))
    ax.add_patch(mpl.patches.Rectangle((xl + wcard * 0.70, 31.6), wcard * 0.30, 1.8, color=SEEG, lw=0, zorder=8))
    ax.text(xl, 31.0, "support", color=OURS, va="top", zorder=8)
    ax.text(xl + wcard, 31.0, "query", color=SEEG, va="top", ha="right", zorder=8)
    _arrow(ax, (cm, 26.6), (cm, 24.2))
    _rbox(ax, cx0, 15.4, cx1 - cx0, 8.6, "white", ec="#c9ced6", lw=0.5, r=0.8, z=3)
    ax.text(cm, 21.6, "closed-form ridge", ha="center", va="center", zorder=4)
    ax.text(cm, 17.8, r"$W=(X^{T}X+\lambda I)^{-1}X^{T}Y$", ha="center", va="center", zorder=4)
    _arrow(ax, (cm, 15.4), (cm, 13.0))
    lt = ax.text(cm - 2.0, 10.4, r"loss $=1-r_{\mathrm{query}}$", ha="center", va="center", color=SEEG)
    cv.fig.canvas.draw()
    bb = lt.get_window_extent(cv.fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    xg = cx1 + 2.2
    _arrow(ax, (xg, 10.4), (xg, 50.6), color=ACC, lw=0.9, ls=(0, (3, 2)))
    ax.plot([bb.x1 + 0.8, xg], [10.4, 10.4], color=ACC, lw=0.9, ls=(0, (3, 2)))
    ax.text(xg, 5.4, "gradient updates " + r"$h_f$", color=ACC, ha="right", va="center")

    save(cv, out, "fig1", nums)


# ----------------------------------------------------------------------------- Fig. 3 (what transfers)
def _init_pairs(cfg, fs=200.0):
    """Initial quadrature pairs of the temporal encoder (same formula as s2i.shared.TemporalEncoder)."""
    half, n = cfg.get("half", 240), cfg.get("n_pairs", 12)
    t = np.arange(-half, half + 1) / fs
    f = np.geomspace(cfg.get("fmin", 2.0), cfg.get("fmax", 60.0), n)
    g = np.exp(-t[None] ** 2 / (2 * (cfg.get("cycles", 5.0) / (2 * np.pi * f[:, None])) ** 2))
    w = np.zeros((2 * n, t.size))
    w[0::2] = g * np.cos(2 * np.pi * f[:, None] * t[None])
    w[1::2] = g * np.sin(2 * np.pi * f[:, None] * t[None])
    return w


def _pair_response(W, n, nfft=4096, fs=200.0):
    """Quadrature-pair amplitude response sqrt(|H_cos|^2 + |H_sin|^2), peak-normalised per pair."""
    f = np.fft.rfftfreq(nfft, 1 / fs)
    H = np.sqrt(np.abs(np.fft.rfft(W[0:2 * n:2], nfft)) ** 2 + np.abs(np.fft.rfft(W[1:2 * n:2], nfft)) ** 2)
    return f, H / H.max(1, keepdims=True)


def _bank(main_run):
    """(f, H0 initial, Hs learned per fold) from the encoder checkpoints enc_sub-*.pt (one per held-out patient)."""
    import glob
    encs = sorted(glob.glob(os.path.join(main_run, "enc_sub-*.pt")))
    if not encs:
        raise FileNotFoundError(2, "missing", os.path.join(main_run, "enc_sub-*.pt"))
    import torch
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in encs]
    cfg = cks[0]["cfg"]
    n = cfg.get("n_pairs", 12)
    f, H0 = _pair_response(_init_pairs(cfg), n)
    Hs = np.array([_pair_response(c["state"]["conv.weight"].numpy()[:, 0], n)[1] for c in cks])
    return f, H0, Hs, [os.path.basename(p)[4:-3] for p in encs]


def _bw(f, H, pk0):
    """Width (octaves) of the contiguous band around each pair's initial peak where the gain is >= -3 dB."""
    out = []
    for k in range(H.shape[0]):
        i = int(np.argmin(np.abs(f - pk0[k])))
        ok = H[k] >= 10 ** (-3 / 20)
        lo, hi = i, i
        while lo > 1 and ok[lo - 1]:
            lo -= 1
        while hi < len(f) - 1 and ok[hi + 1]:
            hi += 1
        out.append(float(np.log2(f[hi] / f[max(lo, 1)])) if ok[i] else float("nan"))
    return out


def _offpeak(f, H, pk0, lo=0.75, hi=1.25):
    """Mean gain (dB re the pair's maximum) 0.75-1.25 octave away from the initial peak, both sides."""
    out = []
    for k in range(H.shape[0]):
        d = np.abs(np.log2(np.maximum(f, 1e-9) / pk0[k]))
        m = (d >= lo) & (d <= hi) & (f >= 0.5) & (f <= 95)
        out.append(float(np.mean(20 * np.log10(H[k, m] + 1e-12))) if m.any() else float("nan"))
    return out


def fig3(runs, out):
    main_run, noenv = os.path.join(runs, "shared_bip_car"), os.path.join(runs, "shared_no_env")
    S5 = load_strict(main_run)
    L = lambda name: load_dir(os.path.join(noenv, name + STRICT), "readout_")   # noqa: E731
    SF, F3, L3 = L("readout_fixed_strict"), L("readout_fixed_arch3_strict"), L("readout_learned_arch3_strict")
    if not (S5 and SF and F3 and L3):
        raise FileNotFoundError(2, "missing", os.path.join(noenv, "readout_*_arch3_strict" + STRICT))
    steps = [("LAG9 ridge", model_rx(S5, "base"), BASE, INK2),
             ("fixed bank", model_rx(SF, "combo_nosess"), FIXED, FIXED_TXT),
             ("3 fixed banks", model_rx(F3, "combo_nosess"), FIXED, FIXED_TXT),
             ("3 learned banks", model_rx(L3, "combo_nosess"), ACC, ACC),
             ("final model", model_rx(S5, "combo_nosess"), OURS, OURS)]
    import glob
    dense = {}                       # Phase 24: ridge on every lag within +-1.2 s, mixed with LAG9 (Table S12)
    for fp in glob.glob(os.path.join(runs, "dense24", "dense24_sub-*.json")):
        d = json.load(open(fp))
        m_ = d["models"]["dense1200+lag9"]
        dense[d["subject"]] = m_.get("r_explicit", m_["r"])      # same scoring as the registered readouts
    if dense:
        steps.insert(1, ("dense-lag ridge", dense, BASE, INK2))
    subs, V = paired(*[s[1] for s in steps])
    m = np.array([v.mean() for v in V])
    tot = paired_stats(V[-1], V[0], "fig3a final-base primary strict")
    nums = {"a": {"n": len(subs), "means": {s[0]: float(v) for s, v in zip(steps, m)}, "steps": {},
                  "final_minus_base": tot}}

    cv = Canvas(W2, 70)
    TOP, BOT = 58.5, 15.0

    # ---------------- a: horizontal waterfall; bar = mean step, dots = previous mean + each patient's step
    ax = cv.ax(19, BOT, 31, TOP - BOT)
    ys = np.arange(len(steps))[::-1].astype(float)               # first step on top
    ax.plot([m[0]] * 2, [ys[0] - 0.22, ys[0] + 0.22], color=BASE, lw=2.4, solid_capstyle="butt", zorder=3)
    ax.text(1.03, ys[0], fmt(m[0]), transform=ax.get_yaxis_transform(), va="center", color=INK2)
    for k in range(1, len(steps)):
        lab, _, col, tcol = steps[k]
        d = V[k] - V[k - 1]
        y, x0, x1 = ys[k], m[k - 1], m[k]
        ax.add_patch(plt.Rectangle((min(x0, x1), y - 0.2), abs(x1 - x0), 0.4, fc=col, ec="none", zorder=3))
        ax.plot([x0, x0], [ys[k - 1] - 0.2, y + 0.2], color=INK2, lw=0.5, ls=(0, (1.5, 1.5)), zorder=1)
        off = st.beeswarm(x0 + d, size_data=0.006, max_off=0.36)
        ax.scatter(x0 + d, y + off, s=4, color=col if col != FIXED else FIXED_TXT, alpha=0.45, lw=0, zorder=2)
        s_ = paired_stats(V[k], V[k - 1], f"fig3a {lab} step primary strict")
        nums["a"]["steps"][lab] = s_
        ax.plot([x0 + s_["ci"][0], x0 + s_["ci"][1]], [y - 0.32] * 2, color=tcol, lw=0.9, solid_capstyle="butt",
                zorder=4)
        hot = col == ACC
        ax.text(1.03, y, f"{fmt(s_['mean_diff'], sign=True)}", transform=ax.get_yaxis_transform(), va="bottom",
                color=tcol, fontweight="bold" if hot else "normal")
        ax.text(1.03, y - 0.06, f"{s_['wins']}/{s_['n']}", transform=ax.get_yaxis_transform(), va="top",
                color=INK2)
    TICK = {"LAG9 ridge": "LAG9 ridge", "dense-lag ridge": "dense-lag\nridge", "fixed bank": "fixed bank",
            "3 fixed banks": "3 fixed\nbanks", "3 learned banks": "3 learned\nbanks", "final model": "final model"}
    ax.set_yticks(ys, [TICK[s[0]] for s in steps], linespacing=1.0)
    for t, s in zip(ax.get_yticklabels(), steps):
        t.set_color(s[3])
    cat_axis(ax)
    pts = np.concatenate([m[:1]] + [V[k] - V[k - 1] + m[k - 1] for k in range(1, len(steps))])
    pad = 0.04 * (pts.max() - pts.min())
    ax.set_xlim(pts.min() - pad, pts.max() + pad)
    xv = nice_ticks(*ax.get_xlim(), maxn=4)
    ax.set_xticks(xv, [fmt(v, 2) for v in xv])
    ax.set_ylim(-0.6, len(steps) - 0.45)
    ax.set_xlabel("test r")
    cv.text(2, 3.0, f"final model − LAG9 ridge {fmt(tot['mean_diff'], sign=True)} [{fmt(tot['ci'][0])}, "
            f"{fmt(tot['ci'][1])}], {tot['wins']}/{tot['n']}", color=OURS, fontweight="bold")
    cv.letter(1, 64.5, "a", "Gain decomposition")

    # ---------------- b: learned response of each pair on an axis relative to its initial peak (octaves)
    f, H0, Hs, folds = _bank(main_run)
    H1 = Hs.mean(0)
    n = H0.shape[0]
    sel = (f >= 1.5) & (f <= 80)
    pk0 = f[np.argmax(H0 * sel, 1)]
    og = np.linspace(-1.5, 1.5, 301)
    FL = -30.0                                                   # dB floor of the display

    def rel(H):
        Z = np.full((n, og.size), np.nan)
        for k in range(n):
            fq = pk0[k] * 2 ** og
            ok = (fq >= f[1]) & (fq <= 95)
            Z[k, ok] = np.clip(20 * np.log10(np.interp(fq[ok], f, H[k]) + 1e-12), FL, 0)
        return Z
    G0, G1 = rel(H0), rel(H1)
    D = G1 - G0
    dm = float(max(6, min(24, np.ceil(np.nanpercentile(np.abs(D), 99) / 6) * 6)))
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("s2i_seq", ["#ffffff", "#c6d8ec", "#5d8fc4", OURS, "#0d2440"])
    div = LinearSegmentedColormap.from_list("s2i_div", [ACC, "#efc2b6", "#ffffff", "#b4cbe6", OURS])
    for c_ in (seq, div):
        c_.set_bad("#ececec")
    edges = np.array([[og[np.nonzero(r >= -3)[0][0]], og[np.nonzero(r >= -3)[0][-1]]] for r in G0])
    ext = [og[0], og[-1], -0.5, n - 0.5]
    hx, hw, hg = 73.0, 22.0, 2.5
    for j, (Z, cmap, vmin, vmax, lab) in enumerate(((G1, seq, FL, 0, "learned"),
                                                    (D, div, -dm, dm, "learned \u2212 initial"))):
        x_ = hx + j * (hw + hg)
        axh = cv.ax(x_, BOT, hw, TOP - BOT)
        axh.imshow(Z, aspect="auto", origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
        for k in range(n):                                       # initial half-power band of each pair
            for e in edges[k]:
                axh.plot([e, e], [k - 0.5, k + 0.5], color=INK2 if j else "white", lw=0.6, ls=(0, (1.6, 1.2)))
        axh.set_xticks([-1, 0, 1], ["\u22121", "0", "1"])
        axh.set_xlim(ext[0], ext[1])
        if j == 0:
            axh.set_yticks(range(n), [f"{v:.0f}" if v >= 9.5 else f"{v:.1f}" for v in pk0])
            axh.set_ylabel("pair (initial peak, Hz)")
        else:
            axh.set_yticks([])
        for s in ("left", "bottom"):
            axh.spines[s].set_visible(False)
        axh.tick_params(axis="y", length=0)
        cv.text(x_ + hw / 2, TOP + 1.2, lab, ha="center", va="bottom", color=ACC if j else INK)
    cv.text(hx + hw + hg / 2, 7.6, "octaves from initial peak", ha="center", va="baseline")
    for (x0_, w_, cmap, l0, l1) in ((hx + 6, hw - 14, seq, fmt(FL, 0), "0 dB"),
                                    (hx + hw + hg + 6, hw - 15, div, fmt(-dm, 0), fmt(dm, 0, sign=True) + " dB")):
        axk = cv.ax(x0_, 2.4, w_, 1.6)
        axk.imshow(np.linspace(0, 1, 128)[None], aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])
        axk.set_axis_off()
        cv.text(x0_ - 0.8, 2.6, l0, ha="right", va="bottom")
        cv.text(x0_ + w_ + 0.8, 2.6, l1, ha="left", va="bottom")
    cv.letter(62, 64.5, "b", "Shared filter bank")
    nums["b"] = {"n_pairs": int(n), "n_folds": int(len(Hs)), "folds": folds, "diff_scale_db": dm,
                 "initial_half_power_oct": edges.tolist(), "max_abs_diff_db": float(np.nanmax(np.abs(D))),
                 "bw3db_oct_initial": _bw(f, H0, pk0), "bw3db_oct_learned_mean": _bw(f, H1, pk0),
                 "bw3db_oct_learned_median_fold": np.median([_bw(f, H, pk0) for H in Hs], 0).tolist(),
                 "offpeak_db_initial": _offpeak(f, H0, pk0),
                 "offpeak_db_learned_median_fold": np.median([_offpeak(f, H, pk0) for H in Hs], 0).tolist(),
                 "offpeak_db_definition": "mean gain (dB re peak) between 0.75 and 1.25 octave on both sides of "
                                          "the initial peak, frequencies 0.5-95 Hz"}

    # ---------------- c: peak shift per pair (octaves), dots = folds, rows aligned with b
    pk = np.array([f[np.argmax(H * sel, 1)] for H in Hs])
    sh = np.log2(pk / pk0[None])
    med = np.median(sh, 0)
    mx = float(min(1.0, max(0.25, np.ceil(np.percentile(np.abs(sh), 95) * 4) / 4)))
    axc = cv.ax(126.5, BOT, 14.0, TOP - BOT)
    axc.axvline(0, color=INK2, lw=0.6, zorder=1)
    for k in range(n):
        o = st.beeswarm(sh[:, k], size_data=0.02, max_off=0.3)
        ok_ = np.abs(sh[:, k]) <= mx
        axc.scatter(sh[ok_, k], k + o[ok_], s=3, color=OURS, alpha=0.35, lw=0, zorder=2)
        axc.plot([0, med[k]], [k, k], color=OURS, lw=1.0, zorder=3)
        axc.plot(med[k], k, "D", ms=3.2, color=OURS, mec="white", mew=0.5, zorder=4)
        axc.text(1.14, k, f"{pk0[k] * 2 ** med[k]:.0f}" if pk0[k] * 2 ** med[k] >= 9.5 else
                 f"{pk0[k] * 2 ** med[k]:.1f}", transform=axc.get_yaxis_transform(), va="center", color=OURS)
    axc.set_xlim(-mx, mx)
    out_ = np.abs(sh) > mx
    if out_.any():
        kk = np.nonzero(out_)
        axc.scatter(np.clip(sh[kk], -mx, mx) * 0.96, kk[1], s=6, facecolors="none", edgecolors=OURS, lw=0.5, zorder=2,
                    clip_on=False)
    axc.set_xticks([-mx, 0, mx], [fmt(-mx, 2 if mx % 0.5 else 1), "0", fmt(mx, 2 if mx % 0.5 else 1, sign=True)])
    axc.set_ylim(-0.5, n - 0.5)
    axc.set_yticks([])
    axc.spines["left"].set_visible(False)
    axc.set_xlabel("octaves")
    cv.text(141.5, TOP + 1.2, "Hz", ha="left", va="bottom", color=OURS)
    cv.letter(124.5, 64.5, "c", "Peak shift")
    nums["c"] = {"peak_initial_hz": pk0.tolist(), "peak_shift_oct_median": med.tolist(),
                 "peak_learned_hz_median": (pk0 * 2 ** med).tolist()}

    save(cv, out, "fig3", nums)


# ----------------------------------------------------------------------------- Fig. 4 (protocol sensitivity)
# one factor at a time: (label, variant A, metric A, variant B, metric B), as in s2i.protocol.ONE_FACTOR
FIG4_FACTORS = [
    ("random split, 20 %", "rand_trials20", "r_concat", "clean", "r_concat"),
    ("random split, 90 %", "rand_trials90", "r_concat", "chrono90", "r_concat"),
    ("90 % training", "chrono90", "r_concat", "clean", "r_concat"),
    ("1-s segments", "rand_seg80", "r_concat", "rand_trials80", "r_concat"),
    ("monopolar target", "mono_car", "r_concat", "clean", "r_concat"),
    ("original scalp ref.", "bip_orig", "r_concat", "clean", "r_concat"),
    ("target + reference", "mono_orig", "r_concat", "clean", "r_concat"),
    ("flattened metric", "clean", "r_nfn", "clean", "r_concat"),
]
FIG4_KEY = "target + reference"
# composite emulations named by their ingredients: (variant, metric, select patients, ingredients)
FIG4_INGR = ["random 90 %", "flattened r", "monopolar", "original ref.", "1-s segments", "selected pts",
             "noise prediction"]
FIG4_EMUL = [("ours", "clean", "r_concat", False, ()),
             ("r90", "rand_trials90", "r_concat", False, ("random 90 %",)),
             ("r90f", "rand_trials90", "r_nfn", False, ("random 90 %", "flattened r")),
             ("mo", "mono_orig", "r_concat", False, ("monopolar", "original ref.")),
             ("mos", "mono_orig_seg80", "r_concat", True, ("monopolar", "original ref.", "1-s segments",
                                                            "selected pts")),
             ("noise", "rand_trials90", "r_nfn_null", False, ("random 90 %", "flattened r", "noise prediction"))]


def fig4(runs, out):
    per = load_dir(os.path.join(runs, "protocol"), "protocol_")
    if not per:
        raise FileNotFoundError(2, "missing", os.path.join(runs, "protocol", "protocol_sub-*.json"))
    subs = sorted(per)
    R = lambda v, met, m="enc": np.array([per[s][v][m][met] for s in subs], float)   # noqa: E731
    nums = {"n": len(subs), "a": {}, "b": {}, "c": {}}
    cv = Canvas(W2, 66)
    TOP, BOT = 54.5, 10.0

    # ---------------- a: forest plot, one factor at a time (encoder with CI; ridge as open circle)
    ax = cv.ax(25, BOT, 20, TOP - BOT)
    ys = np.arange(len(FIG4_FACTORS))[::-1].astype(float)
    eff = {}
    for y, (lab, va, ma, vb, mb) in zip(ys, FIG4_FACTORS):
        s_ = paired_stats(R(va, ma), R(vb, mb), f"fig4a {lab} enc")
        b_ = float((R(va, ma, "base") - R(vb, mb, "base")).mean())
        eff[lab] = s_["mean_diff"]
        key = lab == FIG4_KEY
        col = ACC if key else OURS
        sig = s_["ci"][0] > 0 or s_["ci"][1] < 0
        ax.plot(s_["ci"], [y, y], color=col, lw=1.6 if key else 1.0, solid_capstyle="butt", zorder=2)
        ax.plot(b_, y, "o", ms=3.4, color=BASE_PT, zorder=3)
        ax.plot(s_["mean_diff"], y, "o", ms=4.6 if key else 3.8, mfc=col if sig else "white", mec=col, mew=1.0,
                zorder=4)
        cv.text(45.6, BOT + (y + 0.6) / (len(FIG4_FACTORS) + 0.2) * (TOP - BOT),
                fmt(s_["mean_diff"], sign=True).replace("−0.000", "0.000"), va="center",
                color=col if key else INK, fontweight="bold" if key else "normal")
        cv.text(63.2, BOT + (y + 0.6) / (len(FIG4_FACTORS) + 0.2) * (TOP - BOT), f"{s_['wins']}/{s_['n']}",
                va="center", ha="right", color=INK2)
        nums["a"][lab] = {**s_, "base_mean_diff": b_}
    add = eff["monopolar target"] + eff["original scalp ref."]
    yk = ys[[f[0] for f in FIG4_FACTORS].index(FIG4_KEY)]
    ax.plot(add, yk, "D", ms=4.0, mfc="white", mec=INK, mew=0.9, zorder=5)
    ax.text(add, yk + 0.32, "sum", ha="center", va="bottom", color=INK)
    nums["a"]["sum_of_single_effects"] = add
    ax.axvline(0, color=INK2, lw=0.6, zorder=0)
    ax.set_yticks(ys, [f[0] for f in FIG4_FACTORS])
    for t, f_ in zip(ax.get_yticklabels(), FIG4_FACTORS):
        if f_[0] == FIG4_KEY:
            t.set_color(ACC)
            t.set_fontweight("bold")
    cat_axis(ax)
    hi = max(v["ci"][1] for k, v in nums["a"].items() if isinstance(v, dict))
    lo = min(v["ci"][0] for k, v in nums["a"].items() if isinstance(v, dict))
    ax.set_xlim(min(-0.03, lo - 0.01), np.ceil(hi * 10) / 10 + 0.01)
    ax.set_xticks(np.round(np.arange(0, np.ceil(hi * 10) / 10 + 1e-9, 0.1), 1),
                  ["0"] + [fmt(v, 1) for v in np.arange(0.1, np.ceil(hi * 10) / 10 + 1e-9, 0.1)])
    ax.set_ylim(-0.6, len(ys) - 0.4)
    ax.set_xlabel("change in test r")
    cv.key(2, 57.0, [("encoder", OURS, "o"), ("LAG9 ridge", BASE_PT, "o")], gap=3, text_colors={"LAG9 ridge": INK2})
    cv.letter(1, 61.5, "a", "One factor at a time")

    # ---------------- b: composite emulations, UpSet style (bars = r, matrix = ingredients)
    rows = []
    for k, v, met, sel, ing in FIG4_EMUL:
        x = R(v, met)
        if sel:
            x = x[np.array([per[s][v]["enc"]["cv_r"] > 0.2 for s in subs])]
        rows.append((k, float(x.mean()), len(x), ing))
    r0 = rows[0][1]
    bx, cw = 91.0, 5.8                                   # first column (mm), column width
    nc = len(rows)
    axb = cv.ax(bx, 35.0, nc * cw, TOP - 35.0)
    for j, (k, v, nn, ing) in enumerate(rows):
        col = OURS if k == "ours" else (MUTE if k == "noise" else (ACC if "monopolar" in ing else BASE))
        axb.bar(j, v, width=0.62, color=col, zorder=2)
        axb.text(j, v + 0.012, fmt(v, 2), ha="center", va="bottom", rotation=90,
                 color={MUTE: INK2, BASE: INK2}.get(col, col))
        nums["b"][k] = {"r": v, "n": nn, "ingredients": list(ing), "x_ours": v / r0}
    axb.set_xlim(-0.5, nc - 0.5)
    ymax = np.ceil(max(r[1] for r in rows) * 10) / 10 + 0.2
    axb.set_ylim(0, ymax)
    yv = np.round(np.arange(0, max(r[1] for r in rows) + 0.05, 0.2), 1)
    axb.set_yticks(yv, [fmt(v, 1) if v else "0" for v in yv])
    axb.set_xticks([])
    axb.spines["bottom"].set_visible(False)
    axb.axhline(0, color=INK2, lw=0.6)
    cv.text(bx - 1.0, TOP + 1.2, "test r", ha="right", va="baseline")
    yx = 32.0                                           # "x ours" row
    cv.text(bx - 1.5, yx, "× ours", ha="right", va="center", color=INK2)
    for j, (k, v, nn, ing) in enumerate(rows):
        s = "1" if k == "ours" else ("–" if k == "noise" else f"{v / r0:.1f}")
        cv.text(bx + (j + 0.5) * cw, yx, s, ha="center", va="center",
                color=ACC if "monopolar" in ing else (OURS if k == "ours" else INK))
    axm = cv.ax(bx, BOT, nc * cw, 19.5)
    ni = len(FIG4_INGR)
    for j, (k, v, nn, ing) in enumerate(rows):
        on = [i for i, g in enumerate(FIG4_INGR) if g in ing]
        axm.scatter([j] * ni, range(ni), s=9, color=MUTE, lw=0, zorder=1)
        if on:
            c_ = ACC if "monopolar" in ing else INK2
            axm.plot([j, j], [min(on), max(on)], color=c_, lw=1.0, zorder=2)
            axm.scatter([j] * len(on), on, s=11, color=c_, lw=0, zorder=3)
    axm.set_xlim(-0.5, nc - 0.5)
    axm.set_ylim(ni - 0.5, -0.5)
    axm.set_yticks(range(ni), FIG4_INGR)
    axm.set_xticks([])
    for s in ("left", "bottom"):
        axm.spines[s].set_visible(False)
    axm.tick_params(axis="y", length=0)
    cv.text(bx - 9.2, TOP - 1.0, "emulations,\nnot\nreproductions", ha="right", va="top",
            color=INK2, linespacing=1.15)
    cv.letter(67, 61.5, "b", "Composite emulations")

    # ---------------- c: per patient, our protocol -> monopolar target + original scalp reference
    a_, b_ = R("clean", "r_concat"), R("mono_orig", "r_concat")
    jump = b_ - a_
    order = np.argsort(jump)
    top = set(order[-4:].tolist())
    axc = cv.ax(132.0, BOT, 14.5, TOP - BOT)
    yy = np.arange(len(subs)).astype(float)
    for y, i in zip(yy, order):
        hot = i in top
        axc.plot([a_[i], b_[i]], [y, y], color=ACC if hot else MUTE, lw=1.4 if hot else 1.2, alpha=0.5 if hot else 1,
                 zorder=1)
        axc.plot(a_[i], y, "o", ms=3.2, color=OURS, zorder=3)
        axc.plot(b_[i], y, "o", ms=3.4, color=ACC if hot else BASE, zorder=3)
    axc.set_yticks(yy, [subs[i].replace("sub-", "P") for i in order])
    for t, i in zip(axc.get_yticklabels(), order):
        if i in top:
            t.set_color(ACC)
            t.set_fontweight("bold")
    cat_axis(axc)
    xm = np.ceil(max(a_.max(), b_.max()) * 5) / 5
    axc.set_xlim(0, xm + 0.02)
    axc.set_xticks(np.round(np.arange(0, xm + 1e-9, 0.4), 1), ["0"] + [fmt(v, 1)
                                                                     for v in np.arange(0.4, xm + 1e-9, 0.4)])
    axc.set_ylim(-0.6, len(subs) - 0.4)
    axc.set_xlabel("test r")
    cv.key(120.5, 57.0, [("ours", OURS, "o"), ("mono + orig.", ACC, "o")], gap=2)
    cv.letter(119.5, 61.5, "c", "Per patient")
    nums["c"] = {"top4": sorted(subs[i] for i in top), "ours": dict(zip(subs, a_.tolist())),
                 "mono_orig": dict(zip(subs, b_.tolist())), "top4_mono_orig_range": [float(b_[list(top)].min()),
                                                                                    float(b_[list(top)].max())]}
    save(cv, out, "fig4", nums)


# ----------------------------------------------------------------------------- Fig. 5 (external validation)
DEV, EXT = INK2, OURS                          # cohort colours in Fig. 5: development (ds004752), external
THIRD = "#3a8f7b"                              # third cohort (Parmigiani et al. 2022)


def _interaction(p_):
    return p_["mono|orig"] - p_["mono|car"] - p_["bip|orig"] + p_["bip|car"]


def _p17_ex(D, m):
    return {s: r["explicit"][m]["mean"]["r"] for s, r in D.items() if m in r.get("explicit", {})}


def fig5(runs, out):
    R = os.path.join(runs, "p17")
    rp = os.path.join(R, "p17_report.json")
    if not os.path.exists(rp):
        raise FileNotFoundError(2, "missing", rp)
    rep = json.load(open(rp))
    desc = {d["comparison"]: d for d in rep.get("descriptive", []) if d.get("level") == "primary" and
            d.get("metric", "r") == "r"}
    cm17 = rep.get("common_mode", {})
    it17 = {s: _interaction(v["pred"]) for s, v in cm17.items()}
    ext = {"h1": rep["H1"], "h2": rep["H2"], "fixed1": desc.get("Final − Fixed-1"),
           "half": next((v for k, v in desc.items() if k.startswith("Final, 10 %")), None)}
    if it17:
        v = np.array(list(it17.values()))
        ext["inter"] = paired_stats(v, np.zeros_like(v), "fig5b interaction ScaleOLDNEW")

    main_run, noenv = os.path.join(runs, "shared_bip_car"), os.path.join(runs, "shared_no_env")
    S5 = load_strict(main_run)
    L = lambda name: load_dir(os.path.join(noenv, name + STRICT), "readout_")   # noqa: E731
    SF, F3, L3 = L("readout_fixed_strict"), L("readout_fixed_arch3_strict"), L("readout_learned_arch3_strict")
    dev = {}
    for k, a_, b_ in (("h1", model_rx(S5, "combo_nosess"), model_rx(S5, "base")),
                      ("h2", model_rx(L3, "combo_nosess"), model_rx(F3, "combo_nosess")),
                      ("fixed1", model_rx(S5, "combo_nosess"), model_rx(SF, "combo_nosess")),
                      ("half", model_rx(frac_runs(main_run)["0.1"], "combo_nosess", "0.1"), model_rx(S5, "base", "0.2"))):
        if a_ and b_:
            _, (x, y) = paired(a_, b_)
            dev[k] = paired_stats(x, y, f"fig5b {k} ds004752 primary strict")
    V = load_dir(os.path.join(runs, "review13"), "volts_")
    cmds = {s: (abs(v["r_scalp_cm_vs_ieeg_cm"]), _interaction(v["pred"])) for s, v in V.items()}
    if cmds:
        v = np.array([c[1] for c in cmds.values()])
        dev["inter"] = paired_stats(v, np.zeros_like(v), "fig5b interaction ds004752")
    third = {}                                          # third cohort: H1/H2 of its own report (Table 6), optional
    r27 = os.path.join(runs, "p27", "p27_report.json")
    if os.path.exists(r27):
        rep27 = json.load(open(r27))
        third = {"h1": rep27.get("H1 (system, r)"), "h2": rep27.get("H2 (learning, r)")}
    strip = lambda d: {k: {q: w for q, w in v.items() if q != "per_patient"} for k, v in d.items() if v}  # noqa: E731
    nums = {"ds004752": dev, "ScaleOLDNEW": strip(ext), "third_cohort": strip(third)}
    cv = Canvas(W2, 68)
    TOP, BOT = 56.5, 10.0

    # ---------------- a: H1 per patient (as Fig. 2a), sorted by gain; gain bars
    F = load_dir(os.path.join(R, "main", "final_primary"), "readout_")
    subs, (base, final) = paired(_p17_ex(F, "base"), _p17_ex(F, "combo_nosess"))
    order = np.argsort(final - base)
    yy = np.arange(len(subs)) + 1.4
    ax = cv.ax(9, BOT, 30, TOP - BOT)
    for y, i in zip(yy, order):
        ax.plot([base[i], final[i]], [y, y], color=MUTE, lw=2.2, zorder=1)
        ax.plot(base[i], y, "o", ms=3.6, color=BASE, zorder=3)
        ax.plot(final[i], y, "o", ms=3.8, color=OURS, zorder=4)
    ax.plot([base.mean(), final.mean()], [0, 0], color=OURS, lw=2.6, alpha=0.25)
    ax.plot(base.mean(), 0, "o", ms=4.6, color=BASE)
    ax.plot(final.mean(), 0, "o", ms=4.8, color=OURS)
    ax.axhline(0.7, color=MUTE, lw=0.6)
    ax.set_yticks([0] + list(yy), ["mean"] + [subs[i].replace("sub-", "") for i in order])
    ax.get_yticklabels()[0].set_fontweight("bold")
    lo, hi = np.floor(base.min() * 10) / 10, np.ceil(final.max() * 10) / 10
    xv = nice_ticks(lo, hi, maxn=4)
    ax.set_xticks(xv, [fmt(v, 1) if v else "0" for v in xv])
    ax.set_xlim(lo - 0.01, hi + 0.01)
    ax.set_ylim(-0.6, yy[-1] + 0.6)
    ax.set_xlabel("test r")
    cat_axis(ax)
    cv.key(1.5, 59.0, [("LAG9 ridge", BASE, "o"), ("final model", OURS, "o")], gap=3)
    h1 = rep["H1"]
    g = (final - base)[order]
    axg = cv.ax(42, BOT, 11, TOP - BOT)
    axg.barh(yy, g, height=0.62, color=ACC, alpha=0.85)
    axg.errorbar(h1["mean_diff"], 0, xerr=[[h1["mean_diff"] - h1["ci"][0]], [h1["ci"][1] - h1["mean_diff"]]],
                 fmt="s", ms=4, color=ACC, capsize=0, lw=1.2)
    gmax = max(0.04, np.ceil(g.max() * 50) / 50)
    axg.set_ylim(ax.get_ylim())
    axg.set_xlim(min(0, g.min() * 1.1), gmax)
    axg.set_xticks([0, gmax], ["0", fmt(gmax, 2)])
    axg.set_yticks([])
    axg.spines["left"].set_visible(False)
    axg.axvline(0, color=INK2, lw=0.6)
    axg.axhline(0.7, color=MUTE, lw=0.6)
    axg.set_xlabel("gain")
    cv.letter(1, 63.5, "a", "H1 on ScaleOLDNEW")
    nums["a"] = {"n": len(subs), "mean_base": float(base.mean()), "mean_final": float(final.mean()),
                 "gain_min": float(g.min()), "gain_max": float(g.max()), "H1_report": {q: w for q, w in h1.items()
                                                                                      if q != "per_patient"}}

    # ---------------- b: each development finding in both cohorts (95 % CI); interaction on its own scale
    rows = [("h1", "H1  final model\n− LAG9 ridge", True), ("h2", "H2  learned\n− fixed", True),
            ("fixed1", "final model\n− fixed bank", True), ("half", "half calibration", False)]
    bx, bw = 85.0, 25.0

    def draw(axx, items, y0):
        for j, (k, lab, ok) in enumerate(items):
            y = y0 - j
            for dy, src, col in ((0.24, dev.get(k), DEV), (0.0, ext.get(k), EXT), (-0.24, third.get(k), THIRD)):
                if src:
                    axx.plot(src["ci"], [y + dy] * 2, color=col, lw=1.1, solid_capstyle="butt", zorder=2)
                    axx.plot(src["mean_diff"], y + dy, "o", ms=3.8, color=col, mec="white", mew=0.5, zorder=3)
        axx.axvline(0, color=INK2, lw=0.6, zorder=0)
        axx.set_yticks([y0 - j for j in range(len(items))], [it[1] for it in items])
        for t, it in zip(axx.get_yticklabels(), items):
            t.set_color(INK if it[2] else ACC)
        cat_axis(axx)
    hb = 33.0                                            # height of the 4-row block (mm)
    axb = cv.ax(bx, TOP - hb, bw, hb)
    draw(axb, rows, 3)
    allx = [v for k, _, _ in rows for d in (dev, ext, third) if d.get(k) for v in d[k]["ci"]]
    x0, x1 = np.floor(min(allx + [0]) * 50) / 50, np.ceil(max(allx) * 50) / 50
    axb.set_xlim(x0 - 0.004, x1 + 0.004)
    xv = nice_ticks(x0, x1, maxn=3)
    axb.set_xticks(xv, [fmt(v, 2) if v else "0" for v in xv])
    axb.set_ylim(-0.5, 3.5)
    axb.axhline(0.5, color=MUTE, lw=0.6)
    axi = cv.ax(bx, BOT, bw, 7.5)
    draw(axi, [("inter", "target × reference", False)], 0)
    allx = [v for d in (dev, ext) if d.get("inter") for v in d["inter"]["ci"]]
    if allx:
        xi1 = np.ceil(max(allx) * 10) / 10
        axi.set_xlim(min(-0.05, min(allx) - 0.02), xi1 + 0.01)
        xv = np.round(np.arange(0, xi1 + 1e-9, 0.1), 1)
        axi.set_xticks(xv, [fmt(v, 1) if v else "0" for v in xv])
    axi.set_ylim(-0.5, 0.5)
    cv.text(bx + bw / 2, 3.0, "difference in test r", ha="center", va="baseline")
    cv.key(60.0 if not third else 58.5, 59.0, [("ds004752", DEV, "o"), ("ScaleOLDNEW", EXT, "o")] +
           ([("cohort 3", THIRD, "o")] if third else []), gap=3 if not third else 2)
    cv.letter(58.5, 63.5, "b", "Replication")

    # ---------------- c: scalp-iEEG common mode vs target x reference interaction, per patient
    axc = cv.ax(124, BOT, 25, TOP - BOT - 7.0)
    for s, (x, it) in cmds.items():
        axc.plot(x, it, "o", ms=3.6, color=DEV, mec="white", mew=0.4, alpha=0.9, zorder=2)
    for s, v in cm17.items():
        axc.plot(abs(v["r_scalp_cm_vs_ieeg_cm"]), it17[s], "o", ms=3.8, color=EXT, mec="white", mew=0.4, zorder=3)
    THR = 0.6                                            # "same signal" threshold for the labels
    hd = sorted((it, s.replace("sub-", "")) for s, (x, it) in cmds.items() if x > THR)
    he = [(abs(v["r_scalp_cm_vs_ieeg_cm"]), it17[s], s.replace("sub-", "")) for s, v in cm17.items()
          if abs(v["r_scalp_cm_vs_ieeg_cm"]) > THR]
    for x, it, _ in [(cmds["sub-" + s][0], it, s) for it, s in hd] + he:    # ring the patients above threshold
        axc.plot(x, it, "o", ms=7.0, mfc="none", mec=ACC, mew=0.8, zorder=4)
    kx = 114.5                                           # key above the axes: ring = |r| above threshold
    cv.fig.lines.append(Line2D([(kx + 1.0) / cv.w], [(TOP - 0.2) / cv.h], marker="o", ms=6, mfc="none", mec=ACC,
                               mew=0.8, transform=cv.fig.transFigure))
    cv.text(kx + 2.8, TOP - 1.2, f"|r| > {THR}", color=ACC, va="baseline")
    t_ = cv.text(kx + 2.8, TOP - 4.8, ", ".join(sorted(s for _, s in hd)) or "none", color=DEV, va="baseline")
    r_ = cv.fig.canvas.get_renderer()
    cv.text(kx + 2.8 + t_.get_window_extent(r_).width / cv.fig.dpi * 25.4 + 2.0, TOP - 4.8,
            ", ".join(sorted(s for _, _, s in he)) or "none", color=EXT, fontweight="bold", va="baseline")
    axc.axhline(0, color=INK2, lw=0.6, zorder=0)
    axc.set_xlim(-0.03, 1.03)
    axc.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    yt = [v for v in axc.get_yticks() if axc.get_ylim()[0] <= v <= axc.get_ylim()[1]]
    axc.set_yticks(yt, [fmt(v, 1) if abs(v) > 1e-9 else "0" for v in yt])
    axc.set_xlabel("common-mode |r|")
    axc.set_ylabel("target × reference")
    nums["c_counts"] = {"threshold": THR, "ds004752": [len(hd), len(cmds)], "ScaleOLDNEW": [len(he), len(cm17)]}
    cv.letter(113, 63.5, "c", "Common mode")
    nums["c"] = {"ds004752": {s: list(v) for s, v in cmds.items()},
                 "ScaleOLDNEW": {s: [abs(v["r_scalp_cm_vs_ieeg_cm"]), it17[s]] for s, v in cm17.items()}}
    save(cv, out, "fig5", nums)


# ----------------------------------------------------------------------------- supplementary figures
def _spread(ys, gap):
    """Push label positions apart (order kept) so that neighbours are at least `gap` apart."""
    o = np.argsort(ys)
    y = np.asarray(ys, float)[o].copy()
    for _ in range(100):
        moved = False
        for i in range(1, len(y)):
            if y[i] - y[i - 1] < gap - 1e-12:
                d = (gap - (y[i] - y[i - 1])) / 2
                y[i - 1] -= d
                y[i] += d
                moved = True
        if not moved:
            break
    out = np.empty_like(y)
    out[o] = y
    return out


def _mixed_slope(rows):
    """Slope of r on session gap with a random intercept per patient (statsmodels); OLS if unavailable."""
    G = np.array([r[1] for r in rows], float)
    V = np.array([r[2] for r in rows], float)
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        df = pd.DataFrame(rows, columns=["subject", "gap", "r"])
        fit = smf.mixedlm("r ~ gap", df, groups=df["subject"]).fit()
        return float(fit.params["gap"]), float(fit.params["Intercept"]), fit.conf_int().loc["gap"].tolist(), "mixed"
    except Exception:
        s, i = np.polyfit(G, V, 1)
        return float(s), float(i), [float("nan"), float("nan")], "ols"


def _contacts(D):
    out = {}
    for s, d in D.items():
        m = d["fracs"]["0.2"]["models"]["combo_nosess"]
        cv, te = np.array(m["cv_contacts"], float), np.array(m["contacts"], float)
        ok = np.isfinite(cv) & np.isfinite(te)
        out[s] = (cv[ok], te[ok])
    return out


def figS1(runs, out):
    """Contact reliability (primary), coverage (secondary), longevity across sessions (secondary)."""
    from scipy.stats import spearmanr
    main_run = os.path.join(runs, "shared_bip_car")
    P1 = _contacts(load_strict(main_run))
    P2 = _contacts(load_dir(os.path.join(main_run, "readout_combo5_strict"), "readout_"))
    if not P1 or not P2:
        raise FileNotFoundError(2, "missing", os.path.join(main_run, "readout_combo5_strict"))
    cv_ = Canvas(W2, 64)
    TOP, BOT = 53.0, 10.5
    nums = {}

    # ---------------- a: calibration cross-validation r vs test r, every contact (primary level)
    subs = sorted(P1)
    rho = np.array([spearmanr(*P1[s]).correlation for s in subs])
    cv = np.concatenate([P1[s][0] for s in subs])
    te = np.concatenate([P1[s][1] for s in subs])
    side = TOP - BOT
    ax = cv_.ax(11, BOT, side, side)
    lim = (np.floor(min(np.percentile(cv, 0.3), np.percentile(te, 0.3)) * 10) / 10,
           np.ceil(max(np.percentile(cv, 99.8), np.percentile(te, 99.8)) * 10) / 10)
    ax.plot(lim, lim, color=MUTE, lw=0.6, zorder=0)
    ax.scatter(cv, te, s=3, color=OURS, alpha=0.3, lw=0, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    tk = nice_ticks(*lim, maxn=5)
    ax.set_xticks(tk, [fmt(v, 1) if v else "0" for v in tk])
    ax.set_yticks(tk, [fmt(v, 1) if v else "0" for v in tk])
    ax.set_xlabel("calibration r")
    ax.set_ylabel("test r")
    ax.text(0.04, 0.97, f"Spearman per patient\nmedian {np.median(rho):.2f} ({rho.min():.2f}–{rho.max():.2f})",
            transform=ax.transAxes, va="top", color=OURS, linespacing=1.2)
    cv_.letter(1, 58.5, "a", "Reliability")
    nums["a"] = {"level": "primary, strict", "n_contacts": int(len(cv)), "rho_median": float(np.median(rho)),
                 "rho_mean": float(rho.mean()), "rho_range": [float(rho.min()), float(rho.max())],
                 "rho_pooled": float(spearmanr(cv, te).correlation)}

    # ---------------- b: coverage, secondary level: mean test r of the contacts kept, most reliable first
    cv = np.concatenate([P2[s][0] for s in sorted(P2)])
    te = np.concatenate([P2[s][1] for s in sorted(P2)])
    order = np.argsort(-cv)
    kept = np.arange(1, len(cv) + 1) / len(cv) * 100
    cum = np.cumsum(te[order]) / np.arange(1, len(cv) + 1)
    k0 = 10
    axb = cv_.ax(72, BOT, 34, side)
    axb.plot(kept[k0:], cum[k0:], color=OURS, lw=1.4, zorder=3)
    axb.axhline(te.mean(), color=INK2, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    nums["b"] = {"level": "secondary, strict, final model", "n_contacts": int(len(cv)),
                 "all_contacts_mean": float(te.mean()), "thresholds": {}}
    rowsT = []
    for thr in (0.3, 0.4, 0.5):
        sel = cv > thr
        if sel.sum() < 10:
            continue
        x, y = sel.mean() * 100, te[sel].mean()
        axb.plot(x, y, "o", ms=4, color=ACC, mec="white", mew=0.5, zorder=4)
        rowsT.append((thr, x, y))
        nums["b"]["thresholds"][str(thr)] = {"kept_pct": float(x), "mean_test_r": float(y)}
    axb.set_xlim(0, 100)
    axb.set_xticks([0, 50, 100])
    ylo, yhi = te.mean() - 0.04, cum[k0:].max() + 0.02
    axb.set_ylim(ylo, yhi)
    yv = nice_ticks(ylo, yhi, maxn=4)
    axb.set_yticks(yv, [fmt(v, 1) for v in yv])
    axb.set_xlabel("contacts kept (%)")
    axb.set_ylabel("mean test r of kept contacts")
    axb.text(0.04, (te.mean() - ylo) / (yhi - ylo) - 0.02, f"all contacts {fmt(te.mean(), 2)}",
             transform=axb.transAxes, ha="left", va="top", color=INK2)
    # threshold table in the empty lower-left corner (few contacts kept, low r)
    dl = 3.3 / side
    for xc, head, cell in ((0.50, "cal. r", lambda q: f"> {q[0]:.1f}"), (0.75, "kept", lambda q: f"{q[1]:.0f} %"),
                           (0.98, "test r", lambda q: fmt(q[2], 2))):          # right-aligned columns
        axb.text(xc, 0.97, head, transform=axb.transAxes, color=INK2, ha="right", va="top")
        for j, q in enumerate(rowsT):
            axb.text(xc, 0.97 - (j + 1) * dl, cell(q), transform=axb.transAxes, color=ACC, ha="right", va="top")
    cv_.letter(61, 58.5, "b", "Coverage")

    # ---------------- c: longevity across sessions (secondary level, session-wise normalisation)
    lon = load_dir(os.path.join(main_run, "readout_longevity"), "readout_")
    axc = cv_.ax(122, BOT, 26, side)
    rows, rows_adv = [], []
    for d in lon.values():
        fr, order_ = d["fracs"]["0.2"], d["session_order"]
        last = order_.index(fr["calib_sessions"][-1])
        mods = fr["models"]
        pts = sorted((order_.index(s) - last, v, mods.get("base", {}).get("sessions", {}).get(s))
                     for s, v in mods["combo"].get("sessions", {}).items()
                     if d["test_trials_per_session"].get(s, 0) >= 5)
        if len(pts) >= 2:
            axc.plot([p[0] for p in pts], [p[1] for p in pts], color=MUTE, lw=0.7, zorder=1)
        rows += [(d["subject"], p[0], p[1]) for p in pts]
        rows_adv += [(d["subject"], p[0], p[1] - p[2]) for p in pts if p[2] is not None]
    nums["c"] = {"n_patient_sessions": len(rows)}
    if rows:
        slope, icpt, ci, kind = _mixed_slope(rows)
        G = np.array([r[1] for r in rows])
        gx = np.array([G.min(), G.max()])
        axc.plot(gx, icpt + slope * gx, color=OURS, lw=1.6, zorder=3)
        axc.text(0.04, 0.04, f"{fmt(slope, sign=True)}\nper session", transform=axc.transAxes, ha="left",
                 va="bottom", color=OURS, fontweight="bold", linespacing=1.15)
        nums["c"].update({"slope": slope, "intercept": icpt, "slope_ci": ci, "model": kind})
        if rows_adv:
            sa, _, cia, _ = _mixed_slope(rows_adv)
            nums["c"].update({"advantage_slope": sa, "advantage_slope_ci": cia})
        axc.set_xticks(range(int(G.min()), int(G.max()) + 1, 2))
        axc.set_xlim(G.min() - 0.3, G.max() + 0.3)
    yv = nice_ticks(*axc.get_ylim(), maxn=4)
    axc.set_yticks(yv, [fmt(v, 1) for v in yv])
    axc.set_xlabel("session gap")
    axc.set_ylabel("test r")
    cv_.letter(112, 58.5, "c", "Longevity")
    save(cv_, out, "figS1", nums)


BANDS = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β"}


def figS2(runs, out):
    """Band power of a new patient's contacts without their SEEG (secondary level, leave one patient out)."""
    p = os.path.join(runs, "zeroshot_power", "zeroshot_subjects.json")
    if not os.path.exists(p):
        raise FileNotFoundError(2, "missing", p)
    zs = json.load(open(p))
    bands = list(zs[0]["bands"])
    rows = [("zs_global", "global scalp average", BASE_PT, (0, (1.2, 1.4)), 0),
            ("zs_nearest", "nearest scalp channel", BASE, (0, (1.2, 1.4)), 0),
            ("zs_geo_shuf", "positions shuffled", INK2, (0, (3, 1.6)), 0),
            ("cal_ridge", "ridge, 4 min SEEG", FIXED_TXT, "-", 1),
            ("zs_geo", "zero-shot, no SEEG", ACC, "-", 2),
            ("zs_cal", "zero-shot + ridge", OURS, "-", 1)]
    get = lambda m, b: np.array([r["bands"][b][m]["r_induced"] for r in zs], float)   # noqa: E731
    cv = Canvas(W2, 62)
    TOP, BOT = 52.0, 9.0
    nums = {"n": len(zs), "a": {}}

    ax = cv.ax(12, BOT, 62, TOP - BOT)
    xb = np.arange(len(bands), dtype=float)
    ends = []
    dodge = np.linspace(-0.08, 0.08, len(rows))
    for (m, lab, col, ls, emph), dx in zip(rows, dodge):
        mu = np.array([get(m, b).mean() for b in bands])
        ci = np.array([boot_ci(get(m, b), f"figS2 {m} {b}") for b in bands])
        ax.plot(xb + dx, mu, color=col, lw=(0.9, 1.1, 1.7)[emph], ls=ls, zorder=2 + emph)
        for x, (l_, h_) in zip(xb + dx, ci):
            ax.plot([x, x], [l_, h_], color=col, lw=0.6, alpha=0.8, zorder=2 + emph, solid_capstyle="butt")
        ax.plot(xb + dx, mu, "o", ms=(3.0, 3.4, 4.0)[emph], color=col, mec="white", mew=0.4, zorder=3 + emph)
        ends.append(mu[-1])
        nums["a"][m] = {b: {"mean": float(u), "ci": list(c)} for b, u, c in zip(bands, mu, ci)}
    yl, yh = ax.get_ylim()
    ly = _spread(ends, 3.3 / (TOP - BOT) * (yh - yl))
    for (m, lab, col, ls, emph), y0, y1 in zip(rows, ends, ly):
        ax.plot([xb[-1] + 0.14, xb[-1] + 0.24], [y0, y1], color=col, lw=0.5)
        ax.text(xb[-1] + 0.27, y1, lab, color=col, va="center", fontweight="bold" if emph == 2 else "normal")
    ax.set_xticks(xb, [BANDS.get(b, b) for b in bands])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.3, len(bands) - 1 + 1.45)
    ax.set_ylim(min(yl, min(ly)) , max(yh, max(ly)))
    yv = nice_ticks(*ax.get_ylim(), maxn=4)
    ax.set_yticks(yv, [tl(v) for v in yv])
    ax.set_ylabel("r of log band power")
    cv.letter(1, 57.0, "a", "Band power without SEEG")

    zg = np.mean([get("zs_geo", b) for b in bands], 0)
    cr = np.mean([get("cal_ridge", b) for b in bands], 0)
    side = TOP - BOT
    axb = cv.ax(150 - 3 - side, BOT, side, side)
    lim = (np.floor(min(zg.min(), cr.min()) * 20) / 20, np.ceil(max(zg.max(), cr.max()) * 20) / 20)
    axb.plot(lim, lim, color=MUTE, lw=0.6, zorder=0)
    axb.plot(cr, zg, "o", ms=4, color=ACC, mec="white", mew=0.5, zorder=3)
    axb.set_xlim(lim)
    axb.set_ylim(lim)
    tk = nice_ticks(*lim, maxn=4)
    axb.set_xticks(tk, [tl(v) for v in tk])
    axb.set_yticks(tk, [tl(v) for v in tk])
    axb.set_xlabel("ridge, 4 min SEEG")
    axb.set_ylabel("zero-shot, no SEEG")
    above = int((zg > cr).sum())
    axb.text(0.04, 0.97, f"{above}/{len(zg)} above the diagonal", transform=axb.transAxes, va="top", color=ACC)
    cv.letter(150 - 3 - side - 13, 57.0, "b", "Per patient")
    nums["b"] = {"zero_shot_mean": float(zg.mean()), "ridge_mean": float(cr.mean()), "zero_shot_above": above,
                 "per_patient_mean_over_bands": True}
    save(cv, out, "figS2", nums)


def figS3(runs, out):
    """ScaleOLDNEW scalp-channel audit: raw amplitude vs similarity to amplifier inputs without a 10-05 name."""
    import csv
    p = os.path.join(runs, "chanqc20", "chanqc20_channels_v20_1.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(2, "missing", p)
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["group"] != "no 10-05 name"]

    def num(r, k):
        try:
            return float(r[k])
        except (TypeError, ValueError):
            return np.nan
    groups = [("kept, listed in the authors' files", lambda g: g == "kept, in .set", BASE_PT, 3.0, 0.9),
              ("kept, not listed", lambda g: g == "kept, not in .set", OURS, 4.2, 1.0),
              ("not kept (bad channel)", lambda g: g.startswith("not kept"), ACC, 4.2, 1.0)]
    cv = Canvas(W1, 76)
    ax = cv.ax(13, 10, 69, 50)
    nums = {}
    for lab, fn, col, ms, a in groups:
        R_ = [r for r in rows if fn(r["group"])]
        x = np.array([num(r, "raw_sd_uV") for r in R_])
        y = np.clip(np.array([num(r, "r_unnamed_max") for r in R_]), 1e-3, 0.99985)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.plot(x[ok], y[ok], "o", ms=ms, color=col, alpha=a, mec="white", mew=0.3, ls="none",
                zorder=2 if col == BASE_PT else 3)
        nums[lab] = {"n": int(ok.sum()), "raw_sd_gt500_and_r_ge099": int(((x > 500) & (y >= 0.99)).sum()),
                     "max_raw_sd": float(np.nanmax(x)) if len(x) else None}
    ax.set_xscale("log")
    ax.set_yscale("logit")
    ally = np.concatenate([l.get_ydata() for l in ax.get_lines()])
    y0 = 10 ** np.floor(np.log10(max(1e-3, np.nanmin(ally))) - 1e-9)       # lowest decade with data
    ax.set_ylim(y0 * 0.6, 0.9999)
    yt = [v for v in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999) if v >= y0]
    ax.set_yticks(yt, [tl(v, 3) for v in yt])
    ax.minorticks_off()
    ax.set_xticks([10, 100, 1000], ["10", "100", "1000"])
    ax.axvline(500, color=INK2, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    ax.axhline(0.99, color=INK2, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    ax.set_xlabel("raw standard deviation (µV)")
    ax.set_ylabel("max |r| with unnamed inputs")
    keys = [(f"{lab} ({nums[lab]['n']})", col, "o") for lab, fn, col, ms, a in groups]   # key above the axes
    cv.key(1, 65.0, keys[:1], text_colors={keys[0][0]: INK2})
    cv.key(1, 61.6, keys[1:], gap=4)
    cv.text(1, 71.5, "ScaleOLDNEW scalp channels, patients 01–07", fontsize=FS_T, va="baseline")
    save(cv, out, "figS3", nums)


def fig6(runs, out):
    """(a) Recorded SEEG and predictions of the final model and the LAG9 ridge for the best, median and worst contact
    of the median patient (a 4-s window of the middle test trial); (b) coherence between prediction and recording,
    mean over patients with patient-bootstrap 95 % CI."""
    import glob
    d = os.path.join(runs, "shared_bip_car", "readout_combo5_strict" + STRICT + "_pred")
    files = sorted(glob.glob(os.path.join(d, "pred_sub-*.npz")))
    if not files:
        raise FileNotFoundError(2, "missing", os.path.join(d, "pred_sub-*.npz"))
    Z = {os.path.basename(p)[5:-4]: np.load(p) for p in files}
    rmean = {s: float(np.nanmean(z["r_final"])) for s, z in Z.items()}
    med = float(np.median(list(rmean.values())))
    sid = min(rmean, key=lambda s: (abs(rmean[s] - med), s))           # patient closest to the median
    z = Z[sid]
    tl_ = z["trial_len"].astype(int)
    m = len(tl_) // 2                                                 # middle test trial, first 4 s
    s0 = int(tl_[:m].sum())
    n = min(int(tl_[m]), 800)
    tt = np.arange(n) / 200.0
    nums = {"patient": sid, "patient_mean_r_final": rmean[sid], "median_over_patients": med, "trial": int(m),
            "window_s": n / 200.0, "contacts": []}
    cv = Canvas(W2, 78)
    labs = ("best contact", "median contact", "worst contact")
    top, hh, gap = 62.0, 15.5, 3.0
    for j, lab in enumerate(labs):
        k = int(z["pick"][j])
        ax = cv.ax(25, top - j * (hh + gap) - hh, 72, hh)
        y = z["sig_target"][j, s0:s0 + n].astype(float)
        ax.plot(tt, y, color=INK, lw=0.7, zorder=3)
        ax.plot(tt, z["sig_base"][j, s0:s0 + n].astype(float), color=BASE, lw=0.7, zorder=2)
        ax.plot(tt, z["sig_final"][j, s0:s0 + n].astype(float), color=OURS, lw=0.9, zorder=4)
        lim = np.nanpercentile(np.abs(y), 99.5) * 1.15
        ax.set_ylim(-lim, lim)
        ax.set_xlim(0, tt[-1])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        if j < 2:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_xlabel("time in test trial (s)")
        rf, rb = float(z["r_final"][k]), float(z["r_base"][k])
        cv.text(1.0, top - j * (hh + gap) - hh / 2 + 1.2, lab, color=INK, va="center")
        cv.text(1.0, top - j * (hh + gap) - hh / 2 - 2.6, f"r {fmt(rf)} / {fmt(rb)}", color=INK2, va="center")
        nums["contacts"].append({"label": lab, "index": k, "r_final": rf, "r_base": rb})
    cv.letter(1, 73.0, "a", "Recorded and predicted SEEG")
    cv.key(25, 66.8, [("recording", INK, "_"), ("final model", OURS, "_"), ("LAG9 ridge", BASE, "_")], gap=4,
           text_colors={"recording": INK})
    # ---- b: coherence spectrum over patients
    fr = Z[sid]["freqs"]
    sel = (fr >= 1) & (fr <= 70)
    axb = cv.ax(112, 10, 35, 52)
    nums["coherence"] = {}
    for key, col, lab in (("base", BASE, "LAG9 ridge"), ("final", OURS, "final model")):
        C = np.array([zz[f"coh_{key}"] for zz in Z.values()])[:, sel]
        mu = np.nanmean(C, 0)
        ci = np.array([boot_ci(C[:, i], f"fig6 coherence {key} {fr[sel][i]:.2f}") for i in range(C.shape[1])])
        axb.fill_between(fr[sel], ci[:, 0], ci[:, 1], color=col, alpha=0.18, lw=0)
        axb.plot(fr[sel], mu, color=col, lw=1.2)
        nums["coherence"][lab] = {"freqs": fr[sel].tolist(), "mean": mu.tolist()}
    axb.set_xscale("log")
    axb.set_xticks([1, 4, 10, 30, 70], ["1", "4", "10", "30", "70"])
    axb.minorticks_off()
    axb.set_xlim(1, 70)
    ytop = float(axb.get_ylim()[1]) * 1.02
    yt = nice_ticks(0, ytop)
    axb.set_yticks(yt, [tl(v) for v in yt])
    axb.set_ylim(0, ytop)
    axb.set_xlabel("frequency (Hz)")
    axb.set_ylabel("coherence")
    cv.letter(103, 73.0, "b", "Coherence")
    nums["n_patients"] = len(Z)
    save(cv, out, "fig6", nums)


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6,
        "figS1": figS1, "figS2": figS2, "figS3": figS3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--data", default=None, help="ds004752 data folder (s2i_v1) for the real inputs of Fig. 1")
    a = ap.parse_args()
    global DATA_DIR
    DATA_DIR = a.data
    setup()
    for k, fn in FIGS.items():
        if a.only and k not in a.only:
            continue
        print(k)
        try:
            fn(a.runs, a.out)
        except FileNotFoundError as e:
            print(f"  skipped: missing {e.filename}")
        except Exception:                                  # keep going; the traceback says what to fix
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {k}")


if __name__ == "__main__":
    main()
