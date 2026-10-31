"""
Shared figure style for the paper (Phase 10, v2). Every figure module imports from here.

Design rules (decided 12 Oct 2026, figure_plan.md section 5):
  * final print size: double column 180 mm, single column 88 mm; text 6-8 pt; one sans family
  * no gridlines: 1-2 reference lines with a meaning (zero, baseline mean) instead
  * range frame: spines run from the first to the last tick and sit 3 pt away from the data
  * panel headline = a short conclusion in ink, left-aligned with the axis; details go to the caption
  * colour: warm ink, muted ink-blue for our model, one earth-orange accent per figure, warm greys
  * patients: deterministic beeswarm or paired lines, never random jitter
  * labels are placed next to the data; positions that need hand tuning are read from NUDGE[fig][key]
"""
import glob
import os

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import matplotlib.transforms as mtransforms                        # noqa: E402
from matplotlib import font_manager                                # noqa: E402

MM = 1 / 25.4
PT = 1 / 72
W2, W1 = 180 * MM, 88 * MM

# ----------------------------------------------------------------------------- palette
INK, INK2, INK3 = "#1d1d1b", "#55534e", "#8d8a83"   # text / secondary text, ticks / tertiary notes
PATIENT = "#cfcac1"                                  # individual patients (background)
BASE = "#7d7a73"                                     # baselines
FIXED = "#9dbad6"                                    # fixed untrained filter bank
OURS = "#1f4e79"                                     # our model
OURS_SOFT = "#dce6f0"                                # band / fill for our model
ACCENT = "#c4512b"                                   # the one thing to look at (one per figure)
ACCENT_SOFT = "#f6e1d8"
EXT = "#2f7d6d"                                      # external cohort (figure 7)
WASH = "#f3efe8"                                     # warm background band instead of a grid
RULE = "#bdb8ae"                                     # reference lines

LBL_FIXED = "fixed untrained bank"


STYLE = "own"                                        # "own" (rules above) or "sp" (SciencePlots base)


def setup(style="own"):
    global STYLE
    STYLE = style
    for d in ("/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/liberation2"):
        for f in glob.glob(os.path.join(d, "LiberationSans*.ttf")):
            font_manager.fontManager.addfont(f)
    if style == "sp":
        # SciencePlots: boxed axes, inward mirrored ticks, minor ticks, 7 pt sans (nature). No LaTeX needed.
        import scienceplots                                        # noqa: F401  (registers the styles)
        plt.style.use(["science", "no-latex", "nature"])
        mpl.rcParams.update({"savefig.bbox": "standard", "savefig.dpi": 600, "pdf.fonttype": 42,
                             "svg.fonttype": "none", "axes.unicode_minus": True, "axes.titlelocation": "left",
                             "figure.dpi": 150, "text.color": INK})
        return
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 6,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "axes.labelpad": 2.5, "xtick.direction": "out",
        "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.titlepad": 5, "axes.titlelocation": "left", "legend.frameon": False,
        "axes.unicode_minus": True, "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600,
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "butt", "figure.dpi": 150,
        "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold", "mathtext.sf": "sans",
    })


# ----------------------------------------------------------------------------- axes helpers
def range_frame(ax, x=True, y=True, offset=3):
    """Spines from the first to the last tick, pushed `offset` pt away from the data (no-op in SciencePlots mode)."""
    if STYLE == "sp":
        return
    ax.figure.canvas.draw_idle()
    if y:
        t = [v for v in ax.get_yticks() if ax.get_ylim()[0] - 1e-9 <= v <= ax.get_ylim()[1] + 1e-9]
        if t:
            ax.spines["left"].set_bounds(t[0], t[-1])
        ax.spines["left"].set_position(("outward", offset))
    if x:
        t = [v for v in ax.get_xticks() if ax.get_xlim()[0] - 1e-9 <= v <= ax.get_xlim()[1] + 1e-9]
        if t:
            ax.spines["bottom"].set_bounds(t[0], t[-1])
        ax.spines["bottom"].set_position(("outward", offset))


def bare_y(ax):
    """Category axis on the left: no spine, no ticks."""
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)


def headline(ax, s, tag=None):
    """Short conclusion above the axis, left-aligned with the axis (ink, not grey). `tag` = one line of context
    (level, normalisation...) in small grey type between the headline and the axis."""
    if tag:
        tr = mtransforms.offset_copy(ax.transAxes, fig=ax.figure, x=0, y=2.5, units="points")
        ax.text(0.0, 1.0, tag, transform=tr, fontsize=5.6, color=INK3, va="bottom", ha="left")
    ax.set_title(s, loc="left", color=INK, fontsize=7, pad=11.5 if tag else 5)


def letters(fig, items):
    """Panel letters aligned with the outer edge of each axis' tick labels, top-aligned with its headline.
    items: list of (ax, 'a', dx_mm). dx_mm nudges horizontally (+ right)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    for ax, s, dx in items:
        bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        t = ax._left_title if ax._left_title.get_text() else ax.title
        title = t.get_window_extent(r).transformed(fig.transFigure.inverted())
        fig.text(bb.x0 + dx * MM / W, title.y1 + 0.4 * MM / H, s, fontsize=8, fontweight="bold", va="top",
                 ha="left", color=INK)


def ref_line(ax, v, axis="y", **kw):
    st = dict(color=RULE, lw=0.5, ls=(0, (2, 2)), zorder=0)
    st.update(kw)
    (ax.axhline if axis == "y" else ax.axvline)(v, **st)


def beeswarm(v, size_data, max_off=0.42):
    """Deterministic 1-D beeswarm: offsets (in category units) so that points `size_data` apart do not overlap."""
    v = np.asarray(v, float)
    order = np.argsort(v)
    off = np.zeros(len(v))
    placed = []
    step = 0.11
    for i in order:
        for k in range(0, 40):
            o = (k + 1) // 2 * step * (1 if k % 2 else -1) if k else 0.0
            if abs(o) > max_off:
                o = np.sign(o) * max_off
                break
            if all(abs(v[i] - v[j]) >= size_data or abs(o - off[j]) >= step * 0.95 for j in placed):
                break
        off[i] = o
        placed.append(i)
    return off


def boot_ci(v, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(v, d=3, sign=False):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "\u2212")


def bracket(ax, x, y0, y1, text, color=INK, w=0.06, pad=0.05, fs=6.5, weight="bold", side="right", sub=None):
    """Vertical bracket between y0 and y1 at x (data), with a label beside it."""
    s = 1 if side == "right" else -1
    ax.plot([x, x + s * w, x + s * w, x], [y0, y0, y1, y1], color=color, lw=0.6, solid_joinstyle="miter",
            clip_on=False)
    ym = (y0 + y1) / 2
    ax.text(x + s * (w + pad), ym, text, color=color, fontsize=fs, fontweight=weight, va="center",
            ha="left" if s > 0 else "right", clip_on=False)
    if sub:
        ax.text(x + s * (w + pad), ym, "\n\n" + sub, color=INK2, fontsize=5.8, va="center",
                ha="left" if s > 0 else "right", clip_on=False, linespacing=1.0)


def _sp_box(fig):
    """SciencePlots mode: every visible axis gets its full box back; category axes (major tick length 0) lose the
    automatic minor ticks that would fall between categories."""
    from matplotlib.ticker import NullLocator
    for ax in fig.axes:
        if not ax.axison:
            continue
        for sp in ax.spines.values():
            sp.set_visible(True)
        for axis in (ax.xaxis, ax.yaxis):
            t = axis.get_major_ticks()
            if not t or t[0].tick1line.get_markersize() == 0 or not len(axis.get_majorticklocs()):
                axis.set_minor_locator(NullLocator())


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    if STYLE == "sp":
        _sp_box(fig)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("  ->", os.path.join(out, name + ".{pdf,svg,png}"))
