"""
Paper figures (Phase 10), drawn from the saved result files only. One shared style, final print size.

  fig1_overview   method schematic (no data)
  fig2_main       a per-patient r: lagged ridge -> fixed filter bank -> final model (strict)
                  b all models, per-patient dots (strict; deep baselines under the same split)
                  c calibration curve (fixed test set) and minutes-equivalence
  fig3_transfer   a fixed vs learned filter banks, diversity-matched (transductive)
                  b frequency responses of the filter bank, initialisation vs learned
                  c peak frequency, learned vs initial
  fig4_reliab     a calibration K-fold r vs test r (contacts)  b coverage curve  c longevity
  fig5_power      calibration-free band power, per band
  fig6_protocol   a one factor at a time  b composite recipes  c per-patient target-reference effect

Every figure is saved as PDF + SVG (vector, editable text) + PNG (600 dpi) at its final size
(double column 180 mm). Text 6-8 pt; one meaning per colour in every figure:
grey = baselines / individual patients, light blue = fixed (untrained) filter bank, blue = our model,
orange = the one effect the reader should look at.

Usage
-----
python -m s2i.figures --runs RUNS_DIR --out FIG_DIR [--only fig2 fig6]
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib import font_manager                                # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle   # noqa: E402

# ----------------------------------------------------------------------------- style
MM = 1 / 25.4
W2, W1 = 180 * MM, 88 * MM                     # double / single column
INK, INK2, MUTED, GRID, RULE = "#0b0b0b", "#52514e", "#9a9893", "#e9e8e4", "#cfcdc6"
BLUE, LIGHT_BLUE, ORANGE, GREY, DARK_GREY = "#2a78d6", "#86b6ef", "#eb6834", "#b9b7b0", "#6f6d68"
BLUE_FILL, ORANGE_FILL, GREY_FILL = "#e3eefb", "#fde8df", "#f2f1ee"
RNG = np.random.default_rng(0)


def setup_fonts():
    for d in ("/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/liberation2"):
        for f in glob.glob(os.path.join(d, "LiberationSans*.ttf")):
            font_manager.fontManager.addfont(f)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2,
        "ytick.major.pad": 2, "axes.labelpad": 3, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlepad": 4, "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "lines.solid_capstyle": "round",
    })


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=9, fontweight="bold", va="top", ha="left")


def subtitle(ax, s):
    ax.set_title(s, loc="left", color=INK2, fontsize=7)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def boot_ci(v, n=10000):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    m = np.array([v[RNG.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("  ->", os.path.join(out, name + ".{pdf,svg,png}"))


# ----------------------------------------------------------------------------- loading
STRICT_SUFFIX = "_qc_none"                            # primary level; set by --strict_suffix


def load_dir(path, prefix):
    out = {}
    for f in sorted(glob.glob(os.path.join(path, f"{prefix}sub-*.json"))):
        d = json.load(open(f))
        out[d.get("subject", os.path.basename(f)[len(prefix):-5])] = d
    return out


def model_r(D, model, frac="0.2", key="r"):
    """{subject: value} from readout files (sufficient-statistics r, the numbers of the tables)."""
    return {s: d["fracs"][frac]["models"][model][key] for s, d in D.items()
            if frac in d.get("fracs", {}) and model in d["fracs"][frac]["models"]}


def paired(*dicts):
    subs = sorted(set.intersection(*[set(d) for d in dicts]))
    return subs, [np.array([d[s] for s in subs], float) for d in dicts]


def find_leadformer(runs):
    for run in sorted(glob.glob(os.path.join(runs, "*"))):
        cfgp = os.path.join(run, "config.json")
        if not os.path.exists(cfgp):
            continue
        cfg = json.load(open(cfgp))
        if cfg.get("target") != "bip" or cfg.get("eeg_ref") != "car" or "calib_frac" not in cfg:
            continue
        out = {}
        for f in glob.glob(os.path.join(run, "fold_sub-*.json")):
            fo = json.load(open(f))
            if "calibrated" in fo:
                out[fo["subject"]] = float(np.nanmean(fo["calibrated"]["r"]))
        if out:
            return out
    return {}


# ----------------------------------------------------------------------------- figure 1 (schematic)
def _box(ax, x, y, w, h, fc, ec, text=None, size=6.5, weight="normal", color=INK, r=1.2, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight=weight,
                color=color, linespacing=1.25)


def _arrow(ax, p, q, color=INK2, lw=0.8, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=6, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def _trace(ax, x0, x1, y, amp, seed, color, lw=0.6, f=(3, 7, 13)):
    t = np.linspace(0, 1, 220)
    g = np.random.default_rng(seed)
    s = sum(g.uniform(0.4, 1) * np.sin(2 * np.pi * (fi + g.uniform(-1, 1)) * t + g.uniform(0, 6)) for fi in f)
    s = s / np.abs(s).max()
    ax.plot(x0 + (x1 - x0) * t, y + amp * s, color=color, lw=lw)


def _wavelet(ax, xc, yc, w, h, cyc, color, lw=0.7):
    t = np.linspace(-1, 1, 160)
    s = np.exp(-(t * 2.2) ** 2) * np.cos(np.pi * cyc * t)
    ax.plot(xc + w / 2 * t, yc + h / 2 * s, color=color, lw=lw)


def fig1_overview(runs, out):
    fig = plt.figure(figsize=(W2, 84 * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 84)
    ax.axis("off")

    # --- a: protocol
    letter(fig, 0.008, 0.985, "a")
    ax.text(5, 79.2, "Deployment-faithful evaluation (one patient)", fontsize=7, color=INK2, va="center")
    y0, h, x1 = 69.0, 4.6, 124.0
    ax.add_patch(Rectangle((5, y0), 21.0, h, fc=BLUE, ec="none"))
    ax.add_patch(Rectangle((27.4, y0), x1 - 27.4, h, fc=GREY_FILL, ec=RULE, lw=0.5))
    for sx in (31.0, 55.0, 79.0, 103.0):            # session boundaries (schematic): ticks below the bar
        ax.plot([sx, sx], [y0 - 1.6, y0], color=INK2, lw=0.5)
    ax.text(15.5, y0 + h / 2, "calibration", ha="center", va="center", fontsize=6.3, color="white", fontweight="bold")
    ax.text((27.4 + x1) / 2, y0 + h / 2, "test: all later trials, in the same and in later sessions", ha="center",
            va="center", fontsize=6.3, color=INK2)
    ax.text(5, y0 - 3.3, "first 20 % of trials (≈ 4 min)", ha="left", va="center", fontsize=6, color=BLUE)
    ax.text(26.7, y0 + h + 2.4, "1-trial gap", ha="center", va="center", fontsize=5.8, color=MUTED)
    ax.plot([26.7, 26.7], [y0 + h + 1.2, y0 + h - 0.2], color=MUTED, lw=0.5)
    ax.text(103.6, y0 - 3.3, "ticks: session boundaries", ha="left", va="center", fontsize=5.8, color=MUTED)
    _arrow(ax, (5, y0 - 6.6), (x1, y0 - 6.6), color=MUTED, lw=0.6)
    ax.text(x1 + 1.0, y0 - 6.6, "time", fontsize=5.8, color=MUTED, va="center")
    rules = ["bipolar SEEG targets", "common-average scalp EEG", "normalisation and hyper-parameters",
             "    from calibration trials only", "all contacts and patients reported",
             "statistics over patients (n = 15)"]
    for i, s in enumerate(rules):
        ax.text(132, 77.6 - i * 2.9, ("•  " if not s.startswith(" ") else "") + s, fontsize=5.9, color=INK2,
                va="center")

    # --- b: model
    letter(fig, 0.008, 0.66, "b")
    ax.text(5, 52.0, "Model for a new patient", fontsize=7, color=INK2, va="center")
    # scalp EEG
    for i, yy in enumerate((44.5, 39.5, 34.5)):
        _trace(ax, 5, 21, yy, 1.6, seed=i, color=INK2)
    ax.text(13, 30.0, "scalp EEG\nC channels (6–19)", ha="center", va="top", fontsize=5.9, color=INK2)
    # shared filter bank
    _box(ax, 27, 30.5, 34, 17.5, BLUE_FILL, BLUE)
    ax.text(44, 45.6, "shared temporal filter bank", ha="center", va="center", fontsize=6.4, fontweight="bold",
            color=BLUE)
    for j, cyc in enumerate((2.5, 4.5, 7, 10)):
        _wavelet(ax, 33 + j * 7.4, 39.6, 6.2, 4.6, cyc, BLUE)
    ax.text(44, 34.0, "same filters for every channel\nand every patient; learned on\nother patients",
            ha="center", va="center", fontsize=5.6, color=INK2, linespacing=1.2)
    _arrow(ax, (22, 39.5), (26.4, 39.5))
    # features
    for k in range(4):
        ax.add_patch(Rectangle((66 + k * 1.3, 33.5 + k * 1.3), 8, 10, fc="white", ec=BLUE, lw=0.6))
    ax.text(74.0, 31.2, "features\nC × 40", ha="center", va="top", fontsize=5.9, color=INK2)
    _arrow(ax, (61.6, 39.5), (65.4, 39.5))
    # per-patient readout
    _box(ax, 83, 30.5, 30, 17.5, GREY_FILL, DARK_GREY)
    ax.text(98, 45.6, "patient-specific readout", ha="center", va="center", fontsize=6.4, fontweight="bold",
            color=INK)
    ax.text(98, 38.6, "ridge regression\nscalp features → contacts\nfitted on calibration", ha="center",
            va="center", fontsize=5.6, color=INK2, linespacing=1.2)
    _arrow(ax, (78.2, 39.5), (82.4, 39.5))
    # output
    for i, yy in enumerate((44.5, 39.5, 34.5)):
        _trace(ax, 120, 136, yy, 1.6, seed=10 + i, color=BLUE, f=(2, 5, 9))
    ax.text(128, 30.0, "bipolar SEEG\nK contacts", ha="center", va="top", fontsize=5.9, color=INK2)
    _arrow(ax, (113.6, 39.5), (118.6, 39.5))
    # ensemble note + reliability
    ax.text(141, 46.0, "Final model", fontsize=6.3, fontweight="bold", color=INK, va="center")
    for i, s in enumerate(["average of 5 learned filter banks", "mixed with lagged ridge",
                           "(weight by calibration K-fold r)", "", "Calibration K-fold r per contact",
                           "→ which contacts to trust"]):
        ax.text(141, 42.4 - i * 3.0, s, fontsize=5.8, color=INK2 if i != 4 else INK,
                fontweight="bold" if i == 4 else "normal", va="center")

    # --- c: meta-training
    letter(fig, 0.008, 0.285, "c")
    ax.text(5, 20.6, "Episodic meta-training of the filter bank (leave-one-patient-out)", fontsize=7,
            color=INK2, va="center")
    yb, hb, hs = 3.5, 12.0, 4.8                       # bottom, total height, small box height
    yt = yb + hb - hs                                 # top row
    _box(ax, 5, yb, 25, hb, GREY_FILL, RULE, "training patient\n(12 others; 2 more\nfor early stopping)", size=5.8,
         color=INK2)
    _box(ax, 38, yt, 23, hs, BLUE_FILL, BLUE, "support: ≈ 4 min", size=5.9, color=BLUE)
    _box(ax, 38, yb, 23, hs, "white", BLUE, "query: next 8 trials", size=5.9, color=BLUE)
    _arrow(ax, (30.6, yt + hs / 2), (37.4, yt + hs / 2))
    _arrow(ax, (30.6, yb + hs / 2), (37.4, yb + hs / 2))
    _box(ax, 69, yt, 30, hs, "white", DARK_GREY, "closed-form ridge W(λ)", size=5.9)
    _box(ax, 69, yb, 30, hs, "white", DARK_GREY, "predict query  →  loss 1 − r", size=5.9)
    _arrow(ax, (61.6, yt + hs / 2), (68.4, yt + hs / 2))
    _arrow(ax, (61.6, yb + hs / 2), (68.4, yb + hs / 2))
    _arrow(ax, (84, yt - 0.1), (84, yb + hs + 0.2), lw=0.7)
    _box(ax, 108, yb + 2.0, 30, 8.0, BLUE_FILL, BLUE,
         "update filter bank\nand λ by back-propagation\nthrough the ridge solution", size=5.8, color=BLUE)
    _arrow(ax, (99.6, yb + hs / 2), (107.4, yb + 4.6), color=ORANGE, lw=0.9)
    ax.text(103.5, yb + 0.4, "gradient", fontsize=5.6, color=ORANGE, ha="center", va="center")
    ax.text(141, yb + 8.0, "The spatial mapping is refitted", fontsize=5.8, color=INK, va="center", fontweight="bold")
    ax.text(141, yb + 5.0, "for every episode, so only what", fontsize=5.8, color=INK2, va="center")
    ax.text(141, yb + 2.0, "transfers across patients is learned:", fontsize=5.8, color=INK2, va="center")
    ax.text(141, yb - 1.0, "the temporal filtering.", fontsize=5.8, color=INK2, va="center")
    save(fig, out, "fig1_overview")


# ----------------------------------------------------------------------------- figure 2 (main result)
def fig2_main(runs, out):
    main_run, noenv = os.path.join(runs, "shared_bip_car"), os.path.join(runs, "shared_no_env")
    S5 = load_dir(os.path.join(main_run, "readout_combo5_strict" + STRICT_SUFFIX), "readout_")
    SF = load_dir(os.path.join(noenv, "readout_fixed_strict" + STRICT_SUFFIX), "readout_")
    subs, (base, fixed, final) = paired(model_r(S5, "base"), model_r(SF, "combo_nosess"), model_r(S5, "combo_nosess"))
    mins = np.mean([d["fracs"]["0.2"].get("minutes", np.nan) for d in S5.values()])
    # panel b: the deep baselines were run with session-wise normalisation on artefact-screened trials, so every row
    # of that panel uses the same setting (transductive final model, fixed bank and base on the same trials)
    T5 = load_dir(os.path.join(main_run, "readout_combo5"), "readout_")
    TF = load_dir(os.path.join(noenv, "readout_fixed"), "readout_")
    t_base, t_fixed, t_final = model_r(T5, "base"), model_r(TF, "combo"), model_r(T5, "combo")
    nfn = {s: d["models"]["nfn_mode"]["r"] for s, d in load_dir(os.path.join(runs, "nfn_baseline"), "nfn_").items()}
    lead = find_leadformer(runs)

    fig = plt.figure(figsize=(W2, 66 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.15], wspace=0.42, left=0.065, right=0.985,
                          top=0.80, bottom=0.17)

    # a: per-patient slope
    ax = fig.add_subplot(gs[0])
    xs = [0, 1, 2]
    for b, f, c in zip(base, fixed, final):
        ax.plot(xs, [b, f, c], color=GREY, lw=0.6, zorder=1)
        ax.scatter(xs, [b, f, c], s=5, color=GREY, zorder=2, linewidths=0)
    m = [base.mean(), fixed.mean(), final.mean()]
    ax.plot(xs, m, color=BLUE, lw=1.6, zorder=3)
    ax.scatter(xs, m, s=20, color=[GREY, LIGHT_BLUE, BLUE], zorder=4, edgecolors="white", linewidths=0.8)
    ax.scatter(xs, m, s=20, facecolors="none", edgecolors=[DARK_GREY, BLUE, BLUE], zorder=5, linewidths=0.6)
    for x, v, col in zip(xs, m, [DARK_GREY, BLUE, BLUE]):
        below = x == 0                                  # first label under the point (lines rise to the right)
        ax.text(x, v - 0.013 if below else v + 0.014, f"{v:.3f}", ha="center", va="top" if below else "bottom",
                fontsize=6.5, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    ax.set_xticks(xs, ["lagged\nridge", "fixed\nfilter bank", "final\nmodel"])
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylabel("test r (mean over contacts)")
    ygrid(ax)
    qc = "no automatic QC" if STRICT_SUFFIX == "_qc_none" else "artefact-screened"
    subtitle(ax, f"Strict, {qc}" + (f", {mins:.1f} min calib." if np.isfinite(mins) else "") + "\n"
             f"final > ridge in {int((final > base).sum())}/{len(subs)} patients")

    # b: all models
    ax = fig.add_subplot(gs[1])
    rows = [("NeuroFlowNet", nfn, GREY), ("LeadFormer", lead, GREY), ("lagged ridge", t_base, GREY),
            ("fixed filter bank", t_fixed, LIGHT_BLUE), ("final model", t_final, BLUE)]
    rows = [r for r in rows if r[1]]
    for i, (lab, d, col) in enumerate(rows):
        v = np.array(list(d.values()), float)
        jit = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.36
        ax.scatter(v, i + jit, s=6, color=col if col != GREY else GREY, alpha=0.9, linewidths=0, zorder=2)
        lo, hi = boot_ci(v)
        mc = DARK_GREY if col == GREY else BLUE
        ax.plot([lo, hi], [i, i], color=mc, lw=1.6, zorder=3)
        ax.scatter([v.mean()], [i], s=26, color=mc if col != LIGHT_BLUE else LIGHT_BLUE, edgecolors=mc if col == LIGHT_BLUE else "white",
                   linewidths=0.8, zorder=4, marker="D")
        ax.text(1.02, i, f"{v.mean():.3f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.3,
                color=mc, fontweight="bold" if lab == "final model" else "normal")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(left=min(0, ax.get_xlim()[0]))
    ax.set_xlabel("test r per patient (mean, 95 % CI)")
    xgrid(ax)
    subtitle(ax, "Session-wise norm., artefact-screened\nsame trials; dots = patients, diamond = mean")

    # c: calibration curve
    ax = fig.add_subplot(gs[2])
    C = load_dir(os.path.join(main_run, "curve"), "curve_")
    if C:
        fr = sorted({f for d in C.values() for f in d["curve"]}, key=float)
        full = [d for d in C.values() if all(f in d["curve"] for f in fr)]
        xm = np.array([np.mean([d["curve"][f]["minutes"] for d in full]) for f in fr])
        for key, col, lab in (("base", DARK_GREY, "lagged ridge"), ("enc_ridge_cal", BLUE, "encoder")):
            Y = np.array([[d["curve"][f][key] for f in fr] for d in full])
            mu = Y.mean(0)
            ci = np.array([boot_ci(Y[:, j], 4000) for j in range(len(fr))])
            ax.fill_between(xm, ci[:, 0], ci[:, 1], color=col, alpha=0.09, lw=0)
            ax.plot(xm, mu, color=col, lw=1.4)
            ax.scatter(xm, mu, s=9, color=col, zorder=3, edgecolors="white", linewidths=0.5)
            ax.text(xm[-1] + 0.4, mu[-1], lab, color=col, fontsize=6.3, va="center")
        # equivalence: minutes the baseline needs to match the encoder at ~2 min
        yb = np.array([np.mean([d["curve"][f]["base"] for d in full]) for f in fr])
        ye = np.array([np.mean([d["curve"][f]["enc_ridge_cal"] for d in full]) for f in fr])
        j = int(np.argmin(np.abs(xm - 2.2)))
        need = float(np.interp(ye[j], yb, xm)) if ye[j] <= yb.max() else np.nan
        if np.isfinite(need):
            ax.plot([xm[j], need], [ye[j], ye[j]], color=ORANGE, lw=0.9, ls=(0, (2, 1.5)))
            ax.scatter([xm[j], need], [ye[j], ye[j]], s=12, color=ORANGE, zorder=4, edgecolors="white", linewidths=0.5)
            ax.text(need + 0.5, ye[j] - 0.004, f"ridge needs {need:.1f} min to match\n{xm[j]:.1f} min of the encoder "
                    f"(≈ {need / xm[j]:.1f}×)", ha="left", va="top", fontsize=6.0, color=ORANGE, linespacing=1.2)
        ax.set_xlim(0, xm[-1] * 1.36)
        ax.set_xlabel("calibration (minutes)")
        ax.set_ylabel("test r")
        ygrid(ax)
        subtitle(ax, f"Fixed test set, single encoder\n{len(full)} patients, mean and 95 % CI")
    letter(fig, 0.004, 0.985, "a")
    letter(fig, 0.335, 0.985, "b")
    letter(fig, 0.665, 0.985, "c")
    save(fig, out, "fig2_main")


# ----------------------------------------------------------------------------- figure 3 (what transfers)
def _bank_response(W, n_pairs, fs=200, nfft=4096):
    """Magnitude response of each quadrature pair (cos + i sin), peak-normalised."""
    f = np.fft.rfftfreq(nfft, 1 / fs)
    H = np.abs(np.fft.rfft(W[0:2 * n_pairs:2], nfft)) ** 2 + np.abs(np.fft.rfft(W[1:2 * n_pairs:2], nfft)) ** 2
    H = np.sqrt(H)
    return f, H / H.max(1, keepdims=True)


def fig3_transfer(runs, out):
    import torch
    from s2i.shared import TemporalEncoder
    main_run, noenv = os.path.join(runs, "shared_bip_car"), os.path.join(runs, "shared_no_env")
    steps = [("lagged ridge", load_dir(os.path.join(noenv, "readout_fixed"), "readout_"), "base", GREY),
             ("1 fixed bank", load_dir(os.path.join(noenv, "readout_fixed"), "readout_"), "combo", LIGHT_BLUE),
             ("3 fixed banks", load_dir(os.path.join(noenv, "readout_fixed_arch3"), "readout_"), "combo", LIGHT_BLUE),
             ("same 3, learned", load_dir(os.path.join(noenv, "readout_learned_arch3"), "readout_"), "combo", BLUE),
             ("5 learned", load_dir(os.path.join(main_run, "readout_combo5"), "readout_"), "combo", BLUE)]
    subs, vals = paired(*[model_r(D, m) for _, D, m, _ in steps])

    fig = plt.figure(figsize=(W2, 64 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.15, 0.7], wspace=0.7, left=0.175, right=0.985,
                          top=0.80, bottom=0.17)
    # a: paired increments, one row per step (per-patient dots, mean and patient-bootstrap CI)
    ax = fig.add_subplot(gs[0])
    inc = [("filter bank", "fixed bank − lagged ridge", 1, 0, LIGHT_BLUE),
           ("diversity", "3 fixed banks − 1 fixed bank", 2, 1, LIGHT_BLUE),
           ("learning", "same 3 banks: learned − fixed", 3, 2, ORANGE),
           ("more encoders", "5 learned − 3 learned", 4, 3, BLUE)]
    yy = np.arange(len(inc))[::-1]
    lo_all = []
    for yi, (lab, sub, i, j, col) in zip(yy, inc):
        d = vals[i] - vals[j]
        lo_all.append(d.min())
        mc = BLUE if col == LIGHT_BLUE else col
        jit = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.34
        ax.scatter(d, yi + jit, s=5, color=col if col != ORANGE else "#f3a27f", linewidths=0, zorder=2)
        lo, hi = boot_ci(d)
        ax.plot([lo, hi], [yi, yi], color=mc, lw=1.6, zorder=3)
        ax.scatter([d.mean()], [yi], s=26, marker="D", color=mc, edgecolors="white", linewidths=0.8, zorder=4)
        tr = ax.get_yaxis_transform()
        ax.text(1.03, yi, f"{d.mean():+.3f}", transform=tr, va="center", ha="left", fontsize=6.2, color=mc,
                fontweight="bold" if col == ORANGE else "normal")
        ax.text(1.03, yi - 0.32, f"{int((d > 0).sum())}/{len(d)}", transform=tr, va="center", ha="left",
                fontsize=5.8, color=INK2)
    ax.set_yticks(yy, [f"{lab}\n{sub}" for lab, sub, *_ in inc], fontsize=6.1)
    for t, (lab, *_r) in zip(ax.get_yticklabels(), inc):
        t.set_linespacing(1.15)
        if lab == "learning":
            t.set_color(ORANGE)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_ylim(-0.6, len(inc) - 0.4)
    ax.set_xlabel("Δ test r per patient (mean, 95 % CI)")
    xgrid(ax)
    lv = " → ".join(f"{v.mean():.3f}" for v in vals)
    subtitle(ax, f"What the gain is made of (transductive, 4 min)\nmean r: {lv}")

    # b, c: filter bank
    encs = sorted(glob.glob(os.path.join(main_run, "enc_sub-*.pt")))
    if encs:
        ck = torch.load(encs[0], map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        n_pairs = cfg["n_pairs"]
        W0 = TemporalEncoder(**cfg).conv.weight.detach().numpy()[:, 0]
        f, H0 = _bank_response(W0, n_pairs)
        Hs, pk = [], []
        for p in encs:
            W = torch.load(p, map_location="cpu", weights_only=False)["state"]["conv.weight"].numpy()[:, 0]
            _, H = _bank_response(W, n_pairs)
            Hs.append(H)
            pk.append(f[H.argmax(1)])
        H1, pk = np.mean(Hs, 0), np.array(pk)
        ax = fig.add_subplot(gs[1])
        sel = (f >= 1) & (f <= 80)
        for k in range(n_pairs):
            ax.plot(f[sel], H0[k, sel], color=DARK_GREY, lw=0.55, ls=(0, (1.6, 1.2)))
            ax.plot(f[sel], H1[k, sel], color=BLUE, lw=0.85)
        ax.axvspan(46, 54, color=ORANGE_FILL, lw=0, zorder=0)
        ax.text(50, 1.04, "50 Hz", ha="center", va="bottom", fontsize=5.8, color=ORANGE)
        ax.set_xscale("log")
        ax.set_xticks([2, 4, 8, 16, 32, 64], ["2", "4", "8", "16", "32", "64"])
        ax.minorticks_off()
        ax.set_xlim(1.5, 80)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("normalised gain")
        ax.text(0.0, 1.035, "- - initial (Morlet)", transform=ax.transAxes, fontsize=6.2, color=DARK_GREY,
                va="bottom")
        ax.text(0.52, 1.035, f"— learned ({len(encs)} folds)", transform=ax.transAxes, fontsize=6.2,
                color=BLUE, va="bottom")
        subtitle(ax, f"Shared filter bank: {n_pairs} quadrature pairs\n ")

        ax = fig.add_subplot(gs[2])
        f0 = f[H0.argmax(1)]
        ax.plot([1.5, 80], [1.5, 80], color=RULE, lw=0.6, zorder=0)
        med = np.median(pk, 0)
        lo, hi = np.percentile(pk, 10, 0), np.percentile(pk, 90, 0)
        ax.vlines(f0, lo, hi, color=BLUE, lw=1.0, alpha=0.6)
        ax.scatter(f0, med, s=16, color=BLUE, zorder=3, edgecolors="white", linewidths=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for a_ in (ax.xaxis, ax.yaxis):
            a_.set_major_locator(mpl.ticker.FixedLocator([2, 4, 8, 16, 32, 64]))
            a_.set_major_formatter(mpl.ticker.FixedFormatter(["2", "4", "8", "16", "32", "64"]))
            a_.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xlim(1.5, 80)
        ax.set_ylim(1.5, 80)
        ax.set_aspect("equal")
        ax.set_xlabel("initial peak (Hz)")
        ax.set_ylabel("learned peak (Hz)")
        subtitle(ax, "Peak frequency\n(median, 10–90 %)")
    fig.canvas.draw()                                 # apply the equal aspect before reading positions
    axes = fig.get_axes()
    for ax_, s in zip(axes, "abc"):
        x0 = 0.004 if s == "a" else ax_.get_position().x0 - 0.065
        letter(fig, x0, 0.985, s)
    save(fig, out, "fig3_transfer")


# ----------------------------------------------------------------------------- figure 4 (reliability, longevity)
def fig4_reliab(runs, out):
    main_run = os.path.join(runs, "shared_bip_car")
    S5 = load_dir(os.path.join(main_run, "readout_combo5_strict" + STRICT_SUFFIX), "readout_")
    cv, te = [], []
    for d in S5.values():
        m = d["fracs"]["0.2"]["models"]["combo_nosess"]
        cv += m["cv_contacts"]
        te += m["contacts"]
    cv, te = np.array(cv), np.array(te)
    ok = np.isfinite(cv) & np.isfinite(te)
    cv, te = cv[ok], te[ok]
    from scipy.stats import spearmanr
    rho = spearmanr(cv, te).correlation

    fig = plt.figure(figsize=(W2, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.42, left=0.065, right=0.985, top=0.80, bottom=0.17)
    ax = fig.add_subplot(gs[0])
    lim = (min(-0.1, np.percentile(np.r_[cv, te], 0.5)), max(0.7, np.percentile(np.r_[cv, te], 99.8)))
    ax.plot(lim, lim, color=RULE, lw=0.6, zorder=0)
    ax.scatter(cv, te, s=3, color=BLUE, alpha=0.35, linewidths=0, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("calibration K-fold r (before use)")
    ax.set_ylabel("test r (later trials)")
    ax.text(0.04, 0.96, f"Spearman ρ = {rho:.2f}\n{len(cv)} contacts", transform=ax.transAxes, va="top", fontsize=6.3,
            color=INK2)
    subtitle(ax, "Calibration predicts test accuracy\nper contact (final model, strict)")

    ax = fig.add_subplot(gs[1])
    order = np.argsort(-cv)
    kept = np.arange(1, len(cv) + 1) / len(cv)
    cum = np.cumsum(te[order]) / np.arange(1, len(cv) + 1)
    k0 = 10                                           # start once 10 contacts are kept (stable mean)
    ax.plot(kept[k0:] * 100, cum[k0:], color=BLUE, lw=1.4)
    ax.axhline(te.mean(), color=DARK_GREY, lw=0.6, ls=(0, (2, 1.5)))
    ax.text(3, te.mean() + 0.008, f"all contacts: {te.mean():.3f}", ha="left", va="bottom", fontsize=6.2,
            color=DARK_GREY)
    for thr in (0.3, 0.4, 0.5):
        sel = cv > thr
        if sel.sum() < 10:                            # too few contacts for a stable mean
            continue
        x, y = sel.mean() * 100, te[sel].mean()
        ax.scatter([x], [y], s=14, color=ORANGE, zorder=3, edgecolors="white", linewidths=0.5)
        ax.text(x + 2.5, y + 0.004, f"cal r > {thr}: {x:.0f} % kept, r = {y:.2f}", fontsize=5.9, color=ORANGE,
                va="bottom")
    ax.set_xlim(0, 100)
    ax.set_xlabel("contacts kept (%), most reliable first")
    ax.set_ylabel("mean test r of kept contacts")
    ygrid(ax)
    subtitle(ax, "Reliability-aware contact selection")

    ax = fig.add_subplot(gs[2])
    lon = load_dir(os.path.join(main_run, "readout_longevity"), "readout_")
    rows = []
    for d in lon.values():
        fr, order_ = d["fracs"]["0.2"], d["session_order"]
        last = order_.index(fr["calib_sessions"][-1])
        pts = []
        for s, v in fr["models"]["combo"].get("sessions", {}).items():
            if d["test_trials_per_session"].get(s, 0) >= 5:
                pts.append((order_.index(s) - last, v))
        pts.sort()
        if len(pts) >= 2:
            g, v = zip(*pts)
            ax.plot(g, v, color=GREY, lw=0.6, zorder=1)
            ax.scatter(g, v, s=5, color=GREY, linewidths=0, zorder=2)
        rows += [(d["subject"], gg, vv) for gg, vv in pts]
    if rows:
        G = np.array([r[1] for r in rows], float)
        V = np.array([r[2] for r in rows], float)
        slope, icpt = None, None
        try:
            import pandas as pd
            import statsmodels.formula.api as smf
            df = pd.DataFrame(rows, columns=["subject", "gap", "r"])
            fit = smf.mixedlm("r ~ gap", df, groups=df["subject"]).fit()
            slope, icpt = fit.params["gap"], fit.params["Intercept"]
        except Exception:
            slope, icpt = np.polyfit(G, V, 1)
        gx = np.array([G.min(), G.max()])
        ax.plot(gx, icpt + slope * gx, color=BLUE, lw=1.6, zorder=3)
        ax.text(gx[1], icpt + slope * gx[1] - 0.03, f"{slope:+.3f} r per session".replace("-", "−"), ha="right",
                va="top", fontsize=6.3, color=BLUE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
        ax.set_xticks(range(int(G.min()), int(G.max()) + 1))
    ax.set_xlabel("sessions after the calibration session")
    ax.set_ylabel("test r in that session")
    ygrid(ax)
    subtitle(ax, "Calibration longevity (mixed model;\nlines = patients)")
    letter(fig, 0.004, 0.985, "a")
    letter(fig, 0.335, 0.985, "b")
    letter(fig, 0.655, 0.985, "c")
    save(fig, out, "fig4_reliability")


# ----------------------------------------------------------------------------- figure 5 (band power)
def fig5_power(runs, out):
    zs = json.load(open(os.path.join(runs, "zeroshot_power", "zeroshot_subjects.json")))
    bands = list(zs[0]["bands"])
    rows = [("global scalp average", "zs_global", GREY, "^"),
            ("nearest scalp channel", "zs_nearest", GREY, "o"),
            ("positions shuffled", "zs_geo_shuf", GREY, "s"),
            ("zero-shot (no SEEG)", "zs_geo", BLUE, "D"),
            ("ridge, 4 min calibration", "cal_ridge", DARK_GREY, "o"),
            ("zero-shot + calibration", "zs_cal", BLUE, "o")]
    fig = plt.figure(figsize=(W2, 52 * MM))
    gs = fig.add_gridspec(1, len(bands), wspace=0.12, left=0.155, right=0.985, top=0.78, bottom=0.21)
    xmax = 0
    axes = []
    for bi, b in enumerate(bands):
        ax = fig.add_subplot(gs[bi])
        axes.append(ax)
        for i, (lab, m, col, mk) in enumerate(rows[::-1]):
            v = np.array([r["bands"][b][m]["r_induced"] for r in zs], float)
            lo, hi = boot_ci(v)
            xmax = max(xmax, hi)
            ax.plot([lo, hi], [i, i], color=col, lw=1.3)
            filled = m in ("zs_geo", "cal_ridge")
            ax.scatter([v.mean()], [i], s=24, marker=mk, zorder=3, color=col if filled else "white",
                       edgecolors=col, linewidths=1.0)
        if bi == 0:
            ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
        else:
            ax.set_yticks(range(len(rows)), [])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        xgrid(ax)
        ax.set_title(b, loc="center", fontsize=7, color=INK, fontweight="bold")
    for ax in axes:
        ax.set_xlim(0, np.ceil(xmax * 20) / 20 + 0.01)
        ax.axvline(0, color=INK2, lw=0.6)
    fig.text(0.57, 0.04, "r of log band power beyond the task-locked part (r_induced); mean and 95 % CI over "
             f"{len(zs)} patients", ha="center", fontsize=7)
    fig.text(0.01, 0.97, "Band power of every contact in a new patient, from scalp EEG and contact coordinates "
             "only (no intracranial recording from that patient)", fontsize=7, color=INK2, va="top")
    save(fig, out, "fig5_band_power")


# ----------------------------------------------------------------------------- figure 6 (protocol)
ONE_FACTOR_LABELS = [
    ("split: random vs chronological (20 % calibration)", "Split, 20 % calibration", "random vs chronological"),
    ("split: random vs chronological (90 % training)", "Split, 90 % training", "random vs chronological"),
    ("units: random 1-s segments vs random trials (80 %)", "Units", "1-s segments vs trials"),
    ("amount: 90 % vs 20 % training, both chronological", "Training amount", "90 % vs 20 %, chronological"),
    ("target: monopolar vs bipolar (CAR scalp, clean split)", "Target", "monopolar vs bipolar"),
    ("scalp reference: original vs CAR (bipolar target, clean split)", "Scalp reference", "original vs CAR"),
    ("target + reference: monopolar/original vs bipolar/CAR", "Target + reference", "monopolar/original vs bipolar/CAR"),
    ("metric: NeuroFlowNet-style vs ours (same predictions, clean)", "Metric", "flattened (NeuroFlowNet) vs ours"),
]


def fig6_protocol(runs, out):
    pdir = os.path.join(runs, "protocol")
    summ = json.load(open(os.path.join(pdir, "protocol.json")))
    per = load_dir(pdir, "protocol_")
    fig = plt.figure(figsize=(W2, 78 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.95, 0.7], wspace=0.62, left=0.205, right=0.975,
                          top=0.84, bottom=0.14)
    # a: one factor at a time
    ax = fig.add_subplot(gs[0])
    rows = [(a, b, summ[f"one_factor/{k}/enc"]) for k, a, b in ONE_FACTOR_LABELS if f"one_factor/{k}/enc" in summ]
    y = np.arange(len(rows))[::-1]
    n = len(per) or 15
    for yi, (a_, b_, r) in zip(y, rows):
        key = a_ == "Target + reference"
        col = ORANGE if key else BLUE
        lo, hi = r["ci"]
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.2)
        ax.scatter([r["diff"]], [yi], s=22, zorder=3, color=col if sig else "white", edgecolors=col, linewidths=1.0)
        tr = ax.get_yaxis_transform()
        ax.text(1.03, yi, f"{r['diff']:+.3f}" if abs(r["diff"]) >= 5e-4 else "0.000", va="center", ha="left",
                fontsize=6.2, color=INK2, transform=tr)
        ax.text(1.20, yi, f"{r['wins']}/{n}", va="center", ha="left", fontsize=6.2, color=INK2, transform=tr)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y, [f"{a_}\n{b_}" for a_, b_, _ in rows], fontsize=6.1, color=INK)
    for t, (a_, _, _) in zip(ax.get_yticklabels(), rows):
        t.set_linespacing(1.15)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    xgrid(ax)
    tr = ax.get_yaxis_transform()
    ax.text(1.03, len(rows) - 0.45, "Δr", ha="left", fontsize=6.2, color=MUTED, transform=tr)
    ax.text(1.20, len(rows) - 0.45, "A > B", ha="left", fontsize=6.2, color=MUTED, transform=tr)
    ax.set_xlabel("change in mean r (95 % CI)")
    ki = [i for i, (a_, _, _) in enumerate(rows) if a_ == "Target + reference"]
    if ki:
        ax.text(0.05, y[ki[0]] + 0.45, "each alone +0.02–0.03, together +0.17", fontsize=6.0, color=ORANGE,
                va="center")
    subtitle(ax, "One evaluation factor changed at a time\n(single encoder; filled = CI excludes 0)")

    # b: composite recipes
    ax = fig.add_subplot(gs[1])
    rec = [("ours", "recipe/ours (clean, r_concat, all patients)", BLUE),
           ("random 90 % trials", "rand_trials90/r_concat", GREY),
           ("NeuroFlowNet-like", "recipe/NeuroFlowNet-like (random 90 % trials, target de-normalisation, flattened r)", GREY),
           ("monopolar + orig. ref.", "mono_orig/r_concat", GREY),
           ("CAST-like", "recipe/CAST-like (monopolar, orig ref, random segments, viable patients)", ORANGE)]
    rec = [r for r in rec if r[1] in summ]
    base_r = summ["recipe/ours (clean, r_concat, all patients)"]["enc"]
    yy = np.arange(len(rec))[::-1]
    for yi, (lab, k, col) in zip(yy, rec):
        v = summ[k]["enc"]
        tc = INK2 if col == GREY else col
        ax.plot([0, v], [yi, yi], color=col, lw=2.2)
        ax.scatter([v], [yi], s=18, color=col if col != GREY else DARK_GREY, zorder=3, edgecolors="white",
                   linewidths=0.6)
        ax.text(0.0, yi + 0.3, lab, va="bottom", ha="left", fontsize=6.1, color=INK if col == GREY else col)
        ax.text(v + 0.015, yi, f"{v:.2f}  ({v / base_r:.1f}×)", va="center", fontsize=6.1, color=tc)
    nk = "recipe/NeuroFlowNet metric on RANDOM predictions"
    if nk in summ:
        ax.axvline(summ[nk]["enc"], color=MUTED, lw=0.6, ls=(0, (2, 1.5)))
        ax.text(summ[nk]["enc"] + 0.008, -0.75, "flattened metric,\nrandom predictions", fontsize=5.6, color=MUTED,
                va="top")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 0.68)
    ax.set_ylim(-1.6, len(rec) - 0.1)
    ax.set_xlabel("mean test r (same model)")
    xgrid(ax)
    subtitle(ax, "Literature-inspired\nprotocol emulations")

    # c: per-patient contamination
    ax = fig.add_subplot(gs[2])
    if per:
        subs = sorted(per)
        a_ = np.array([per[s]["clean"]["enc"]["r_concat"] for s in subs])
        b_ = np.array([per[s]["mono_orig"]["enc"]["r_concat"] for s in subs])
        jump = b_ - a_
        top = set(np.argsort(-jump)[:4])
        for i, s in enumerate(subs):
            col = ORANGE if i in top else GREY
            ax.plot([0, 1], [a_[i], b_[i]], color=col, lw=0.9 if i in top else 0.6, zorder=2 if i in top else 1)
            ax.scatter([0, 1], [a_[i], b_[i]], s=6, color=col, linewidths=0, zorder=3)
        # patient labels, pushed apart so they never overlap
        lab = sorted((b_[i], subs[i].replace("sub-", "P")) for i in top)
        span = max(b_.max(), a_.max()) - min(b_.min(), a_.min())
        ys = []
        for y_, _ in lab:
            ys.append(max(y_, ys[-1] + 0.045 * span) if ys else y_)
        for (y_, s), yl in zip(lab, ys):
            ax.text(1.07, yl, s, fontsize=5.8, color=ORANGE, va="center")
        ax.set_xticks([0, 1], ["bipolar\nCAR", "monopolar\noriginal ref."])
        ax.set_xlim(-0.25, 1.35)
        ax.set_ylabel("test r per patient")
        ygrid(ax)
        subtitle(ax, "Target–reference effect\nper patient")
    letter(fig, 0.004, 0.985, "a")
    letter(fig, 0.555, 0.985, "b")
    letter(fig, 0.80, 0.985, "c")
    save(fig, out, "fig6_protocol")


FIGS = {"fig1": fig1_overview, "fig2": fig2_main, "fig3": fig3_transfer, "fig4": fig4_reliab,
        "fig5": fig5_power, "fig6": fig6_protocol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--strict_suffix", default="_qc_none",
                    help="strict runs of fig. 2a and 4: '_qc_none' = primary level (no automatic QC), '' = artefact-screened")
    a = ap.parse_args()
    global STRICT_SUFFIX
    STRICT_SUFFIX = a.strict_suffix
    setup_fonts()
    for k, fn in FIGS.items():
        if a.only and k not in a.only:
            continue
        print(k)
        try:
            fn(a.runs, a.out)
        except FileNotFoundError as e:
            print(f"  skipped: missing {e.filename}")


if __name__ == "__main__":
    main()
