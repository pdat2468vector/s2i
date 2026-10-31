"""
Paper figures, version 2 (Phase 10 rework, 12 Oct 2026). Style: s2i/figstyle.py; plan: figure_plan.md.

Each figure reads the saved result files only and writes <name>.{pdf,svg,png} plus <name>_numbers.json, which lists
every number printed on the figure, so that it can be checked against the manuscript.

  fig2_main   a per-patient r: lagged ridge -> fixed untrained bank -> final model (primary level, strict)
              b all models incl. deep baselines (secondary level, session-wise normalisation, same trials)
              c calibration efficiency (primary level, strict; 5 / 10 / 20 % calibration on the same test trials)

Usage
  python -m s2i.paperfigs --runs RUNS_DIR --out FIG_DIR [--only fig2]
"""
import argparse
import glob
import json
import os

import numpy as np

from s2i import figstyle as st
from s2i.figstyle import (ACCENT, BASE, EXT, FIXED, INK, INK2, INK3, MM, OURS, OURS_SOFT, PATIENT, RULE, W2,
                          WASH, plt)

STRICT = "_qc_none"                      # primary level (no additional automatic QC)

# hand-tuned label offsets (data units unless noted); edit after looking at the real-data render
NUDGE = {
    "fig1": {"brain": (34.0, 34.0), "brain_w": 58.0},
    "fig2": {"a_base_lab": (-0.16, 0.0), "a_fixed_lab": (0.0, 0.012), "a_final_lab": (0.13, 0.0),
             "c_note": (0.22, 0.026), "c_dx": 0.07},
}


# ----------------------------------------------------------------------------- loading
def load_dir(path, prefix):
    out = {}
    for f in sorted(glob.glob(os.path.join(path, f"{prefix}sub-*.json"))):
        d = json.load(open(f))
        out[d.get("subject", os.path.basename(f)[len(prefix):-5])] = d
    return out


def load_strict(main_run, name="readout_combo5_strict"):
    """Primary-level strict readouts with every calibration fraction in one `fracs` dict: the 20 % run holds frac
    "0.2"; the 10 % and 5 % runs are separate folders (suffix _f10 / _f05, frac keys "0.1" / "0.05")."""
    D = load_dir(os.path.join(main_run, name + STRICT), "readout_")
    for suf in ("_f10", "_f05"):
        for s, d in load_dir(os.path.join(main_run, name + STRICT + suf), "readout_").items():
            if s in D:
                for f, v in d.get("fracs", {}).items():
                    D[s]["fracs"].setdefault(f, v)
    return D


def model_r(D, model, frac="0.2", key="r"):
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


def paired_stats(a, b):
    d = np.asarray(a) - np.asarray(b)
    lo, hi = st.boot_ci(d)
    return {"mean_diff": float(d.mean()), "ci": [lo, hi], "wins": int((d > 0).sum()), "n": int(len(d))}


def dump_numbers(out, name, nums):
    os.makedirs(out, exist_ok=True)
    json.dump(nums, open(os.path.join(out, f"{name}_numbers.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- figure 1 (schematic, no data)
# Every coordinate below is in millimetres on the printed page (the axes fill the figure, 1 data unit = 1 mm).
def _rbox(ax, x, y, w, h, fc, ec="none", lw=0.6, r=1.0, ls="-", z=1):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def _arr(ax, p, q, color=INK2, lw=0.55, ls="-", rad=0.0, head=4.5, z=3):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def _sig(n, seed, f=(3, 7, 13, 21)):
    t = np.linspace(0, 1, n)
    g = np.random.default_rng(seed)
    s = sum(g.uniform(0.35, 1) / (1 + 0.08 * fi) * np.sin(2 * np.pi * (fi + g.uniform(-1, 1)) * t + g.uniform(0, 6))
            for fi in f)
    return t, s / np.abs(s).max()


def _wave(ax, xc, yc, w, h, cyc, color, lw=0.7, ls="-", skew=0.0, shift=0.0, z=4):
    t = np.linspace(-1, 1, 200)
    env = np.exp(-((t - shift) * 2.3) ** 2) * (1 + skew * t)
    s = env * np.cos(np.pi * cyc * t + skew * 2.0)
    ax.plot(xc + w / 2 * t, yc + h / 2 * s / np.abs(s).max(), color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="round")


def _vdots(ax, x, y, gap=1.0, r=0.22, color=INK3):
    for k in (-1, 0, 1):
        ax.add_patch(plt.Circle((x, y + k * gap), r, fc=color, ec="none", zorder=3))


def _ptitle(ax, x, y, letter, title, tag):
    ax.text(1.6, y, letter, fontsize=8, fontweight="bold", color=INK, va="baseline")
    ax.text(x, y, title, fontsize=7, color=INK, va="baseline")
    ax.text(x, y - 3.3, tag, fontsize=5.6, color=INK3, va="baseline")


def _ucurve(ax, x0, y0, x1, y1, ybot, r=2.2, color=ACCENT, lw=0.8):
    """Arrow from (x0, y0) down to ybot, across, and up into (x1, y1), with rounded corners."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    sx = np.sign(x1 - x0)
    v = [(x0, y0), (x0, ybot + r), (x0, ybot), (x0 + sx * r, ybot), (x1 - sx * r, ybot), (x1, ybot), (x1, ybot + r),
         (x1, y1)]
    c = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO]
    ax.add_patch(FancyArrowPatch(path=Path(v, c), arrowstyle="-|>", mutation_scale=5, color=color, lw=lw, zorder=5))


def fig1_overview(runs, out):
    H = 104.0
    fig = plt.figure(figsize=(W2, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, H)
    ax.axis("off")
    TEST = "#d6d1c7"

    # ================= a: deployment-faithful evaluation (one patient's recording, as a row of trials)
    _ptitle(ax, 7, 100.2, "a", "Evaluated the way it would be deployed",
            "one patient's recording as a row of trials · hollow = artefact annotated in the dataset, excluded")
    y0, h, slot, bw, sgap = 84.0, 5.0, 1.62, 1.12, 3.2
    sess = [17, 15, 16, 12]                                   # trials per session (schematic)
    ncal, artef = 12, {21, 34, 35, 49}                        # 20 % of 60 trials; dataset-annotated artefacts
    xs, x, k = [], 7.0, 0
    for si, n in enumerate(sess):
        x_s0 = x
        for _ in range(n):
            xs.append(x)
            x += slot
        ax.text((x_s0 + x - slot + bw) / 2, y0 - 2.6, f"session {si + 1}", fontsize=5.2, color=INK3, ha="center",
                va="top")
        x += sgap
    for i, xi in enumerate(xs):
        if i < ncal:
            ax.add_patch(plt.Rectangle((xi, y0), bw, h, fc=OURS, ec="none", zorder=2))
        elif i == ncal:                                       # one-trial guard gap
            ax.plot([xi + bw / 2] * 2, [y0 - 0.3, y0 - 2.6], color=INK3, lw=0.35)
            ax.text(xi + bw / 2 + 0.6, y0 - 2.6, "1-trial gap", fontsize=5.0, color=INK3, ha="left", va="top")
        elif i in artef:
            ax.add_patch(plt.Rectangle((xi + 0.1, y0 + 0.1), bw - 0.2, h - 0.2, fc="white", ec=RULE, lw=0.4, zorder=2))
        else:
            ax.add_patch(plt.Rectangle((xi, y0), bw, h, fc=TEST, ec="none", zorder=2))
    xc1, xt0, xt1 = xs[ncal - 1] + bw, xs[ncal + 1], xs[-1] + bw
    yb = y0 + h + 1.3
    for (a_, b_, col) in ((xs[0], xc1, OURS), (xt0, xt1, INK2)):
        ax.plot([a_, a_, b_, b_], [yb - 0.7, yb, yb, yb - 0.7], color=col, lw=0.45, solid_joinstyle="miter")
    ax.text(xs[0], yb + 1.1, "calibration: first 20 % of trials (≈ 6 min)", fontsize=5.8, color=OURS, ha="left",
            va="bottom", fontweight="bold")
    ax.text(xt1, yb + 1.1, "test: every later trial, same and later sessions", fontsize=5.8, color=INK2, ha="right",
            va="bottom")
    ax.text(xt1 + 1.4, y0 + h / 2, "time", fontsize=5.2, color=INK3, va="center")
    _arr(ax, (xt1 + 6.6, y0 + h / 2), (xt1 + 9.6, y0 + h / 2), color=INK3, lw=0.45, head=3.5)
    # rules (numbered, hand-wrapped)
    rules = [("bipolar SEEG targets, common-average scalp EEG",), ("normalisation and hyper-parameters from",
             "calibration trials only"), ("no automatic artefact rejection beyond", "the dataset's annotations"),
             ("every contact and every patient reported",), ("statistics over patients (n = 15)",),
             ("two main hypotheses pre-specified and tested", "on an independent cohort (n = 12; Fig. 7)")]
    yy = 99.6
    for i, lines in enumerate(rules):
        col = EXT if i == len(rules) - 1 else INK2
        ax.text(130.0, yy, f"{i + 1}", fontsize=5.4, color=INK3 if i < len(rules) - 1 else EXT, va="baseline")
        for j, s in enumerate(lines):
            ax.text(133.0, yy - j * 2.45, s, fontsize=5.6, color=col, va="baseline")
        yy -= 2.45 * len(lines) + 0.55
    ax.plot([127.6, 127.6], [yy + 1.6, 101.6], color=RULE, lw=0.4)

    # ================= b: model for a new patient
    _ptitle(ax, 7, 73.6, "b", "Temporal filters are shared across patients; the spatial readout is not",
            "the model applied to a new patient")
    zb, zt = 34.4, 67.4
    _rbox(ax, 44.0, zb, 40.0, zt - zb, OURS_SOFT, r=1.6, z=0)
    _rbox(ax, 86.0, zb, 92.0, zt - zb, WASH, r=1.6, z=0)
    ax.text(45.6, zt - 2.3, "shared across patients", fontsize=5.6, color=OURS, fontweight="bold", va="center")
    ax.text(87.6, zt - 2.3, "fitted per patient, on calibration trials only", fontsize=5.6, color=INK2,
            fontweight="bold", va="center")
    # head, top view: scalp electrodes and one SEEG shaft
    cx, cy, R = 18.0, 52.0, 10.8
    th = np.linspace(0, 2 * np.pi, 300)
    ax.fill(cx + R * np.cos(th), cy + R * np.sin(th), color="white", zorder=1)
    ax.plot(cx + R * np.cos(th), cy + R * np.sin(th), color=INK2, lw=0.6, zorder=2)
    ax.plot([cx - 1.6, cx, cx + 1.6], [cy + R - 0.15, cy + R + 1.9, cy + R - 0.15], color=INK2, lw=0.6, zorder=2)
    for s in (-1, 1):
        e = np.linspace(-0.55, 0.55, 40)
        ax.plot(cx + s * (R + 0.9 * np.cos(e * 2.6)), cy + 2.6 * e * 1.4, color=INK2, lw=0.6, zorder=2)
    pos = {"Fp1": (-.30, .86), "Fp2": (.30, .86), "F7": (-.72, .47), "F3": (-.36, .42), "Fz": (0, .40),
           "F4": (.36, .42), "F8": (.72, .47), "T7": (-.88, 0), "C3": (-.44, 0), "Cz": (0, 0), "C4": (.44, 0),
           "T8": (.88, 0), "P7": (-.72, -.47), "P3": (-.36, -.42), "Pz": (0, -.40), "P4": (.36, -.42),
           "P8": (.72, -.47), "O1": (-.30, -.86), "O2": (.30, -.86)}
    E = {k: (cx + .82 * R * u, cy + .82 * R * v) for k, (u, v) in pos.items()}
    for k, (ex, ey) in E.items():
        ax.add_patch(plt.Circle((ex, ey), 0.62, fc=INK if k in ("F4", "C4", "P4") else "white", ec=INK, lw=0.4,
                                zorder=4))
    p0, p1 = np.array([2.6, 49.0]), np.array([16.8, 50.4])          # shaft from outside the head inwards
    ax.plot(*zip(p0, p1), color=INK3, lw=0.9, zorder=3, solid_capstyle="butt")
    for j, tt in enumerate(np.linspace(0.42, 0.97, 6)):
        q = p0 + tt * (p1 - p0)
        ax.add_patch(plt.Rectangle((q[0] - 0.5, q[1] - 0.5), 1.0, 1.0, angle=0, fc=OURS if j >= 4 else INK2,
                                   ec="none", zorder=4))
    ax.text(cx, zb + 4.0, "scalp EEG (6–19 channels)", fontsize=5.4, color=INK2, ha="center", va="center")
    ax.text(cx, zb + 1.6, "SEEG shaft, bipolar pairs", fontsize=5.4, color=OURS, ha="center", va="center")
    # scalp channels -> traces
    ty = {"F4": 59.0, "C4": 53.6, "P4": 44.6}
    for k, yv in ty.items():
        ax.plot([E[k][0] + 0.6, 31.0], [E[k][1], yv], color=INK3, lw=0.3, zorder=3)
        t, s = _sig(160, {"F4": 1, "C4": 2, "P4": 3}[k])
        ax.plot(31.4 + 10.6 * t, yv + 1.5 * s, color=INK, lw=0.5, zorder=3)
    _vdots(ax, 36.7, 49.4)
    # filter bank: a stack of identical cards = the same bank on every channel
    bx, by, bw_, bh = 48.4, 40.8, 22.0, 19.4
    for o in (2.6, 1.3):
        _rbox(ax, bx + o, by + o, bw_, bh, "white", OURS, lw=0.4, r=0.9, z=1)
    _rbox(ax, bx, by, bw_, bh, "white", OURS, lw=0.65, r=0.9, z=2)
    for j, (cyc, sk, sh) in enumerate(((1.4, 0.25, 0.08), (2.8, -0.2, -0.05), (5.0, 0.3, 0.04), (8.5, -0.15, 0.0))):
        yc = by + bh - 3.0 - j * 4.75
        _wave(ax, bx + bw_ / 2, yc, 17.0, 3.4, cyc, FIXED, lw=0.6, ls=(0, (1.6, 1.1)))
        _wave(ax, bx + bw_ / 2, yc, 17.0, 3.4, cyc * (1 + 0.06 * (j - 1.5)), OURS, lw=0.75, skew=sk, shift=sh)
    for k, yv in ty.items():
        _arr(ax, (42.6, yv), (bx - 0.5, by + 3 + (yv - 44.6) / 14.4 * (bh - 6)), color=INK2, lw=0.45, head=3.6)
    ax.text(bx, by - 2.0, "28 filters + 12 envelopes per channel", fontsize=5.1, color=INK2, va="center")
    ax.plot([bx, bx + 3.2], [by - 4.5] * 2, color=FIXED, lw=0.7, ls=(0, (1.6, 1.1)))
    ax.text(bx + 4.0, by - 4.5, "Morlet init", fontsize=5.1, color=INK3, va="center")
    ax.plot([bx + 15.4, bx + 18.6], [by - 4.5] * 2, color=OURS, lw=0.8)
    ax.text(bx + 19.4, by - 4.5, "meta-learned", fontsize=5.1, color=INK3, va="center")
    # features
    for o in range(4):
        ax.add_patch(plt.Rectangle((75.0 + o * 0.9, 46.4 + o * 0.9), 4.6, 8.4, fc="white", ec=OURS, lw=0.45,
                                   zorder=2 + o))
    ax.text(79.0, 44.5, "C × 40", fontsize=5.4, color=OURS, ha="center", va="center")
    _arr(ax, (bx + bw_ + 3.2, 51.5), (74.6, 51.5), color=INK2, lw=0.45, head=3.6)
    # ridge readout (matrix)
    g = np.random.default_rng(7)
    Wm = np.tanh(g.normal(0, 0.9, (9, 6)) + np.outer(np.sin(np.arange(9) / 1.7), np.cos(np.arange(6) / 1.3)))
    mx, my, cs = 90.4, 42.6, 1.95
    for i in range(9):
        for j in range(6):
            v = Wm[i, j]
            ax.add_patch(plt.Rectangle((mx + j * cs, my + i * cs), cs, cs, fc=OURS if v > 0 else BASE,
                                       alpha=0.12 + 0.75 * abs(v), ec="white", lw=0.35, zorder=2))
    ax.text(mx, my - 2.0, "ridge readout W", fontsize=5.4, color=INK, va="center")
    ax.text(mx, my - 3.4, "penalty by\nblocked 5-fold r", fontsize=5.1, color=INK3, va="top", linespacing=1.15)
    _arr(ax, (82.4, 51.5), (mx - 0.6, 51.5), color=INK2, lw=0.45, head=3.6)
    # outputs: recorded vs predicted bipolar SEEG
    ox0, ox1 = 107.6, 131.0
    for j, yv in enumerate((58.4, 52.9, 44.6)):
        t, s = _sig(220, 20 + j, f=(2, 5, 9, 16))
        p = np.convolve(s, np.ones(9) / 9, mode="same") * 0.78 + 0.1 * _sig(220, 40 + j, f=(11, 23))[1]
        ax.plot(ox0 + (ox1 - ox0) * t, yv + 1.7 * s, color=TEST, lw=1.25, zorder=2, solid_capstyle="round")
        ax.plot(ox0 + (ox1 - ox0) * t, yv + 1.7 * p, color=OURS, lw=0.6, zorder=3)
    _vdots(ax, (ox0 + ox1) / 2, 49.0)
    ax.plot([ox0, ox0 + 3.4], [62.4] * 2, color=TEST, lw=1.25)
    ax.text(ox0 + 4.1, 62.4, "recorded", fontsize=5.1, color=INK3, va="center")
    ax.plot([ox0 + 13.2, ox0 + 16.6], [62.4] * 2, color=OURS, lw=0.7)
    ax.text(ox0 + 17.3, 62.4, "predicted", fontsize=5.1, color=INK3, va="center")
    _arr(ax, (mx + 6 * cs + 1.0, 51.5), (ox0 - 1.2, 51.5), color=INK2, lw=0.45, head=3.6)
    ax.text(ox0 - 1.0, my - 2.0, "bipolar SEEG, every contact", fontsize=5.4, color=INK, va="center")
    ax.text(ox0 - 1.0, my - 4.5, "final: 5 encoders averaged, mixed with ridge", fontsize=5.1, color=INK3,
            va="center")
    # contact ranking from calibration cross-validation
    rx = 147.0
    vals = np.array([.43, .39, .36, .31, .28, .26, .22, .19, .16, .12, .09, .05])
    for i, v in enumerate(vals):
        yv = 58.8 - i * 1.42
        ax.add_patch(plt.Rectangle((rx, yv - 0.48), v / vals[0] * 25.0, 0.96, fc=OURS if i < 3 else TEST, ec="none",
                                   zorder=2))
    ax.plot([rx - 0.9, rx - 1.6, rx - 1.6, rx - 0.9], [58.8 + 0.6, 58.8 + 0.6, 58.8 - 2 * 1.42 - 0.6,
            58.8 - 2 * 1.42 - 0.6], color=OURS, lw=0.45, solid_joinstyle="miter")
    ax.text(rx - 2.4, 58.8 - 1.42, "top\nquarter", fontsize=5.0, color=OURS, ha="right", va="center",
            linespacing=1.05)
    ax.text(rx, 62.4, "contacts ranked by calibration r", fontsize=5.1, color=INK3, va="center")
    ax.text(rx, my - 2.0, "which contacts to trust", fontsize=5.4, color=INK, va="center")
    ax.text(rx, my - 4.5, "before any test data", fontsize=5.1, color=INK3, va="center")
    _arr(ax, (ox1 + 1.2, 51.5), (rx - 7.2, 51.5), color=INK3, lw=0.45, ls=(0, (2, 1.5)), head=3.6)

    # ================= c: episodic meta-training
    _ptitle(ax, 7, 30.8, "c", "The filters are meta-learned through the ridge readout",
            "episodic training on the other patients (leave-one-patient-out)")
    # patient pool
    gx, gy = [7.0, 10.0, 13.0, 16.0, 19.0], [21.6, 18.1, 14.6]
    role = {(1, 2): "test", (2, 0): "stop", (0, 3): "stop"}
    for r_, yv in enumerate(gy):
        for c_, xv in enumerate(gx):
            k = role.get((r_, c_), "train")
            if k == "test":
                ax.add_patch(plt.Circle((xv, yv), 1.05, fc="white", ec=INK, lw=0.7, zorder=3))
            elif k == "stop":
                ax.add_patch(plt.Circle((xv, yv), 1.0, fc="white", ec=BASE, lw=0.45, ls=(0, (1.2, 0.8)), zorder=3))
            else:
                ax.add_patch(plt.Circle((xv, yv), 1.0, fc=BASE, ec="none", zorder=3))
    for j, (k, s) in enumerate((("test", "held-out test patient"), ("stop", "2 for early stopping"),
                                ("train", "12 for training"))):
        yv = 10.0 - j * 2.7
        if k == "test":
            ax.add_patch(plt.Circle((7.0, yv), 0.8, fc="white", ec=INK, lw=0.6))
        elif k == "stop":
            ax.add_patch(plt.Circle((7.0, yv), 0.75, fc="white", ec=BASE, lw=0.45, ls=(0, (1.2, 0.8))))
        else:
            ax.add_patch(plt.Circle((7.0, yv), 0.75, fc=BASE, ec="none"))
        ax.text(8.8, yv, s, fontsize=5.2, color=INK2, va="center")
    _arr(ax, (21.4, 18.1), (28.6, 18.1), color=INK2, lw=0.45, head=3.6)
    ax.text(25.0, 19.3, "sample 2", fontsize=5.0, color=INK3, ha="center", va="bottom")
    # episode: support / gap / query
    ex0, es, ebw, ey, eh = 30.0, 1.5, 1.05, 15.6, 5.0
    for i in range(20):
        xi = ex0 + i * es
        if i < 11:
            ax.add_patch(plt.Rectangle((xi, ey), ebw, eh, fc=OURS_SOFT, ec=OURS, lw=0.35))
        elif i > 11:
            ax.add_patch(plt.Rectangle((xi, ey), ebw, eh, fc="white", ec=OURS, lw=0.35))
    ax.text(ex0, ey + eh + 1.0, "support\n≈ 4 min", fontsize=5.2, color=OURS, va="bottom", linespacing=1.1)
    ax.text(ex0 + 20 * es - (es - ebw), ey + eh + 1.0, "query\n8 trials", fontsize=5.2, color=OURS, va="bottom",
            ha="right", linespacing=1.1)
    ax.text(ex0 + 11 * es + ebw / 2, ey - 1.4, "gap", fontsize=4.8, color=INK3, ha="center", va="top")
    # encoder chip -> ridge -> loss
    cx0, cw, cy0, ch = 66.0, 13.0, 13.6, 9.0
    _rbox(ax, cx0, cy0, cw, ch, "white", OURS, lw=0.65, r=0.9, z=2)
    for j, cyc in enumerate((1.6, 3.4, 7.0)):
        _wave(ax, cx0 + cw / 2, cy0 + ch - 1.8 - j * 2.7, 10.0, 1.9, cyc, OURS, lw=0.6, skew=0.15 * (j - 1))
    ax.text(cx0 + cw / 2, cy0 + ch + 1.2, "filter bank", fontsize=5.2, color=OURS, ha="center", va="bottom")
    _arr(ax, (ex0 + 20 * es + 0.6, 18.1), (cx0 - 0.5, 18.1), color=INK2, lw=0.45, head=3.6)
    rx0, rw = 85.0, 30.0
    _rbox(ax, rx0, cy0, rw, ch, "white", BASE, lw=0.55, r=0.9, z=2)
    ax.text(rx0 + rw / 2, cy0 + ch - 2.5, "closed-form ridge on support", fontsize=5.6, color=INK, ha="center",
            va="center")
    ax.text(rx0 + rw / 2, cy0 + 2.8, r"$W = (X^{\mathrm{T}}X + \lambda I)^{-1}X^{\mathrm{T}}Y$", fontsize=5.6, color=INK2,
            ha="center", va="center")
    _arr(ax, (cx0 + cw + 0.6, 18.1), (rx0 - 0.5, 18.1), color=INK2, lw=0.45, head=3.6)
    lx0, lw_ = 121.0, 22.0
    _rbox(ax, lx0, cy0, lw_, ch, "white", BASE, lw=0.55, r=0.9, z=2)
    ax.text(lx0 + lw_ / 2, cy0 + ch - 2.5, "predict query", fontsize=5.6, color=INK, ha="center", va="center")
    ax.text(lx0 + lw_ / 2, cy0 + 2.8, "loss = 1 − mean r", fontsize=5.6, color=INK2, ha="center", va="center")
    _arr(ax, (rx0 + rw + 0.6, 18.1), (lx0 - 0.5, 18.1), color=INK2, lw=0.45, head=3.6)
    # the accent: gradient back through the closed-form solution into the filters (and lambda)
    _ucurve(ax, lx0 + lw_ / 2, cy0 - 0.3, cx0 + cw / 2, cy0 - 0.4, 6.4)
    _arr(ax, (rx0 + rw / 2 + 6.5, 6.4), (rx0 + rw / 2 + 6.5, cy0 - 0.4), color=ACCENT, lw=0.6, head=4.2, z=5)
    ax.text(rx0 + rw / 2 + 7.6, 9.6, "λ", fontsize=5.4, color=ACCENT, va="center")
    ax.text((cx0 + lx0 + lw_) / 2, 4.6, "gradient back-propagated through the ridge solution", fontsize=5.6,
            color=ACCENT, ha="center", va="top")
    # take-home
    for j, s in enumerate(("The readout is refitted in every", "episode, so only what transfers",
                           "across patients is learned:", "the temporal filters.")):
        ax.text(148.0, 22.2 - j * 2.6, s, fontsize=5.6, color=INK if j == 3 else INK2, va="baseline",
                fontweight="bold" if j == 3 else "normal")
    ax.plot([146.2, 146.2], [22.2 - 3 * 2.6 - 0.8, 22.2 + 2.0], color=RULE, lw=0.4)

    st.save(fig, out, "fig1_overview")
    dump_numbers(out, "fig1_overview", {"calibration_label_min": "≈ 6 (manuscript: about 6 min; primary 5.7)",
                                         "support_s": 240, "query_trials": 8, "train_patients": 12,
                                         "early_stop_patients": 2, "features_per_channel": 40,
                                         "external_n": 12, "n_patients": 15})


# ----------------------------------------------------------------------------- figure 1, v3 (white scene)
# One left-to-right scene, few words; detail goes to the caption. 1 data unit = 1 mm on the page.
def _trace(n, seed, k=30, alpha=1.0):
    g = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    y = sum(np.sin(2 * np.pi * f * 1.6 * t + g.uniform(0, 6.3)) / f ** (0.5 * alpha) for f in range(1, k))
    return y / np.abs(y).max()


FIG1_CH = ("#1a9a8b", "#7051a6", "#3f88c5", "#d39a27")   # four scalp channels (teal, violet, azure, ochre)
BRAIN_GY, BRAIN_SU, BRAIN_LN = "#f7ebe5", "#ead2c8", "#d6b2a5"
PATIENT_DK = "#bdb6aa"                               # recorded SEEG trace (a touch darker than PATIENT)


def _wt_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("wt", ["#d39a27", "#fbf9f5", OURS])


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("q", ["#eef4fa", "#6aa6d6", OURS])


def _brain2d(ax, cx, cy, w, z=2.0, outline_only=False, fc=None, ec=INK2, lw=0.6, ls="-"):
    """Flat lateral right hemisphere (fsaverage5, anterior right) centred at (cx, cy), w mm wide.
    Returns (outline polyline in mm, bbox (x0, y0, x1, y1))."""
    import contourpy
    from matplotlib.patches import Polygon
    from s2i import render3d as r3
    xs, ys, M, T = r3.lateral_2d()
    raw = max(contourpy.contour_generator(xs, ys, M).lines(0.5), key=len)
    x0, x1, y0, y1 = raw[:, 0].min(), raw[:, 0].max(), raw[:, 1].min(), raw[:, 1].max()
    s = w / (x1 - x0)
    X, Y = cx + s * (xs - (x0 + x1) / 2), cy + s * (ys - (y0 + y1) / 2)
    out = np.c_[cx + s * (raw[:, 0] - (x0 + x1) / 2), cy + s * (raw[:, 1] - (y0 + y1) / 2)]
    bb = (out[:, 0].min(), out[:, 1].min(), out[:, 0].max(), out[:, 1].max())
    if outline_only:
        ax.add_patch(Polygon(out, closed=True, fc=fc or "none", ec=ec, lw=lw, ls=ls, zorder=z))
        return out, bb
    poly = Polygon(out, closed=True, fc=BRAIN_GY, ec="none", zorder=z)
    ax.add_patch(poly)
    cs = ax.contourf(X, Y, np.where(M > 0.2, T, 0.0), levels=[0.45, 2.0], colors=[BRAIN_SU], zorder=z + 0.1)
    cs.set_clip_path(poly)
    cl = ax.contour(X, Y, np.where(M > 0.2, T, 0.0), levels=[0.45], colors=[BRAIN_LN], linewidths=0.3,
                    zorder=z + 0.2)
    cl.set_clip_path(poly)
    ax.plot(out[:, 0], out[:, 1], color=ec, lw=lw, zorder=z + 0.3)
    return out, bb


def _head2d(ax, cx, cy, w, q, pad=(2.5, 7.0)):
    """Flat head: brain, scalp arc with electrodes (4 coloured = shown channels) and one SEEG shaft whose
    contacts carry the same colours as the SEEG panel. Sizes scale with w."""
    out, (x0, y0, x1, y1) = _brain2d(ax, cx, cy, w, lw=0.7)
    bw_, bh_ = x1 - x0, y1 - y0
    k = w / 40.0
    hc = np.array([cx, cy + 0.10 * bh_])
    hr = np.array([bw_ / 2 + pad[0], bh_ / 2 + pad[1]])
    t = np.radians(np.linspace(-22, 202, 300))
    ax.plot(hc[0] + hr[0] * np.cos(t), hc[1] + hr[1] * np.sin(t), color=INK3, lw=0.6, zorder=3)
    on = lambda a: hc + hr * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])      # noqa: E731
    for a in (178, 160, 142, 124, 106):
        ax.add_patch(plt.Circle(on(a), 0.55 * k, fc="white", ec=INK3, lw=0.45, zorder=4))
    for a, c in zip((84, 62, 40, 18), FIG1_CH):
        ax.add_patch(plt.Circle(on(a), 0.85 * k, fc=c, ec="white", lw=0.45, zorder=4.2))
    # SEEG shaft: in through the vertex, tip in the temporal lobe
    pe = on(114) + np.array([0.0, 1.8 * k])
    pt = np.array([x0 + 0.60 * bw_, y0 + 0.24 * bh_])
    ax.plot([pe[0], pt[0]], [pe[1], pt[1]], color=INK, lw=0.85, zorder=5, solid_capstyle="butt")
    u = (pe - pt) / np.linalg.norm(pe - pt)
    cm = _seq_cmap()
    for j, qk in enumerate(q):                            # j = 0 shallowest ... last = tip
        p = pt + u * 1.45 * k * (len(q) - 1 - j)
        ax.add_patch(plt.Circle(p, 0.55 * k, fc=cm(qk), ec=INK, lw=0.35, zorder=5.1))


def fig1_scene(runs, out):
    """Two-tier scene: the patient's head spans the full height on the left; the model runs along the top
    (scalp EEG -> shared filters -> readout) and drops into the predicted SEEG along the bottom."""
    H = 74.0
    yh = 70.0                                             # baseline of the top-row labels
    fig = plt.figure(figsize=(W2, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, H)
    ax.axis("off")
    N = NUDGE["fig1"]
    fs, fs2 = 6.5, 5.6
    q = np.array([0.82, 0.35, 0.58, 0.95, 0.70, 0.22, 0.45, 0.30])     # schematic per-contact accuracy
    mid = 57.25                                           # top-row flow line

    # ---- new patient: big flat head on the left, full height
    _head2d(ax, *N["brain"], w=N["brain_w"], q=q)
    ax.text(N["brain"][0], 3.6, "new patient", ha="center", fontsize=fs, color=INK)

    # ---- scalp EEG: four channels in the electrode colours, gathered by a bracket into the filters
    tx0, tx1 = 69.5, 94.0
    ys = [mid + 8.25, mid + 2.75, mid - 2.75, mid - 8.25]
    for j, (y, c) in enumerate(zip(ys, FIG1_CH)):
        x = np.linspace(tx0, tx1, 200)
        ax.plot(x, y + 1.6 * _trace(200, 10 + j), color=c, lw=0.65, zorder=3)
    xb_ = tx1 + 1.6
    ax.plot([xb_ - 0.8, xb_, xb_, xb_ - 0.8], [ys[0] + 1.9, ys[0] + 1.9, ys[-1] - 1.9, ys[-1] - 1.9],
            color=INK3, lw=0.5, zorder=3, solid_joinstyle="miter")
    ax.text((tx0 + tx1) / 2, yh, "scalp EEG", ha="center", fontsize=fs, color=INK)

    # ---- the shared, meta-learned filter bank (the idea of the paper) + the 14 patients behind it
    bx, bw, bh = 100.0, 34.0, 19.0
    by = mid - bh / 2
    _arr(ax, (xb_, mid), (bx - 0.6, mid), color=INK2, lw=0.55, head=4)
    for d in (1.8, 0.9):
        _rbox(ax, bx + d, by + d, bw, bh, "white", ec=OURS, lw=0.4, r=1.2, z=2)
    _rbox(ax, bx, by, bw, bh, OURS_SOFT, ec=OURS, lw=0.75, r=1.2, z=3)
    for j, (cyc, sk) in enumerate(((2.2, 0.0), (3.6, 0.25), (5.6, -0.2))):
        yc = mid + 5.6 - 5.6 * j
        _wave(ax, bx + bw / 2, yc, bw - 8, 4.4, cyc, FIXED, lw=0.6, ls=(0, (1.6, 1.1)), z=4)
        _wave(ax, bx + bw / 2, yc, bw - 8, 4.4, cyc * 1.12, OURS, lw=1.0, skew=sk, shift=0.06 * (j - 1), z=5)
    ax.text(bx + bw / 2, yh, "shared temporal filters", ha="center", fontsize=fs, color=OURS, fontweight="bold")
    for i in range(14):                                   # 12 training (filled), 2 early stopping (dashed)
        es = i in (5, 10)
        _brain2d(ax, bx + 1.25 + i * (bw - 2.5) / 13, by - 2.7, 2.25, z=3, outline_only=True,
                 fc="white" if es else "#a9c1da", ec=INK3 if es else OURS, lw=0.3, ls=(0, (1.0, 0.7)) if es else "-")
    ax.text(bx + bw / 2 + 1.5, by - 6.6, "learned on 14 other patients", ha="center", fontsize=fs2, color=INK2)

    # ---- per-patient readout (fitted on the first 20 % of trials)
    rx, rw, rh = 146.0, 16.0, 22.0
    ry = mid - rh / 2
    g = np.random.default_rng(4)
    Wm = np.tanh(g.normal(0, 1, (9, 6)) * 0.9)
    ax.imshow(Wm, extent=[rx, rx + rw, ry, ry + rh], cmap=_wt_cmap(), vmin=-1, vmax=1, zorder=3,
              interpolation="nearest", aspect="auto")
    ax.add_patch(plt.Rectangle((rx, ry), rw, rh, fc="none", ec=INK2, lw=0.5, zorder=4))
    ax.text(rx + rw / 2 + 2.0, yh, "per-patient readout", ha="center", fontsize=fs, color=INK)
    ax.text(rx + rw + 1.6, mid, "first\n20 % of\ntrials", ha="left", va="center", fontsize=fs2, color=OURS,
            linespacing=1.1)

    # ---- forward pass (grey) and, just under it, the one accent: the gradient back through the ridge
    x_a, x_b = bx + bw + 3.0, rx - 1.0
    yg = mid - 3.8
    _arr(ax, (x_a, mid), (x_b, mid), color=INK2, lw=0.55, head=4)
    _arr(ax, (x_b, yg), (x_a, yg), color=ACCENT, lw=1.05, head=5.5, z=6)
    ax.text((x_a + x_b) / 2, yg - 1.5, "gradient\nthrough\nridge", ha="center", va="top", fontsize=fs2,
            color=ACCENT, linespacing=1.05)

    # ---- bipolar SEEG along the bottom, contacts coloured by how well they are predicted (as in the brain)
    shx, y0, y1 = 73.0, 3.5, 37.5
    cy_ = np.linspace(y1 - 2.0, y0 + 2.0, len(q))
    cm = _seq_cmap()
    ax.plot([shx, shx], [y0, y1], color=INK, lw=0.85, zorder=3, solid_capstyle="butt")
    for j, (yc, qk) in enumerate(zip(cy_, q)):
        ax.add_patch(plt.Rectangle((shx - 1.25, yc - 1.05), 2.5, 2.1, fc=cm(qk), ec=INK, lw=0.4, zorder=4))
        xx = np.linspace(shx + 3.2, 178.5, 320)
        rec = _trace(320, 40 + j, k=40, alpha=1.3)
        pred = qk * rec + (1 - qk) * _trace(320, 80 + j, k=40, alpha=1.3)
        pred = pred / np.abs(pred).max()
        ax.plot(xx, yc + 1.5 * rec, color=PATIENT_DK, lw=1.0, zorder=3, solid_capstyle="round")
        ax.plot(xx, yc + 1.5 * pred, color=OURS, lw=0.6, alpha=0.35 + 0.65 * qk, zorder=4)
    _arr(ax, (rx + rw / 2, ry - 0.8), (rx + rw / 2, y1 + 1.2), color=INK2, lw=0.55, head=4)
    ax.text(shx - 3.2, (y0 + y1) / 2, "bipolar SEEG", ha="center", va="center", rotation=90, fontsize=fs,
            color=INK)
    ly = 41.0
    ax.plot([shx - 3.0, shx + 0.0], [ly] * 2, color=OURS, lw=0.6)
    ax.text(shx + 0.6, ly, "predicted", fontsize=fs2, color=INK2, va="center")
    ax.plot([shx + 12.2, shx + 15.2], [ly] * 2, color=PATIENT_DK, lw=1.0)
    ax.text(shx + 15.8, ly, "recorded", fontsize=fs2, color=INK2, va="center")

    st.save(fig, out, "fig1_overview")
    dump_numbers(out, "fig1_overview", {"schematic": True, "calibration_fraction": 0.2, "other_patients": 14,
                                         "early_stop_patients": 2})


def st_mix(c0, c1, t):
    from s2i.render3d import mix
    return tuple(mix(c0, c1, t))


def _div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("w", [INK3, "#f4f2ee", OURS])


# ----------------------------------------------------------------------------- figure 2
def fig2_main(runs, out):
    N = NUDGE["fig2"]
    main_run, noenv = os.path.join(runs, "shared_bip_car"), os.path.join(runs, "shared_no_env")
    S5 = load_strict(main_run)
    SF = load_dir(os.path.join(noenv, "readout_fixed_strict" + STRICT), "readout_")
    if not S5 or not SF:
        raise FileNotFoundError(2, "missing", os.path.join(main_run, "readout_combo5_strict" + STRICT))
    subs, (base, fixed, final) = paired(model_r(S5, "base"), model_r(SF, "combo_nosess"), model_r(S5, "combo_nosess"))
    mins = float(np.mean([d["fracs"]["0.2"].get("minutes", np.nan) for d in S5.values()]))
    nums = {"a": {"n": len(subs), "mean_base": base.mean(), "mean_fixed": fixed.mean(), "mean_final": final.mean(),
                  "final_minus_base": paired_stats(final, base), "minutes_20": mins}}

    fig = plt.figure(figsize=(W2, 80 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.78, 1.0], height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.78,
                          left=0.075, right=0.975, top=0.875, bottom=0.115)

    # ---------------- a: paired per-patient lines (tall, left)
    ax = fig.add_subplot(gs[:, 0])
    xs = np.array([0.0, 1.0, 2.0])
    for b, f, c in zip(base, fixed, final):
        ax.plot(xs, [b, f, c], color=PATIENT, lw=0.45, zorder=1)
        ax.scatter(xs, [b, f, c], s=2.2, color=PATIENT, zorder=2, linewidths=0)
    m = np.array([base.mean(), fixed.mean(), final.mean()])
    ax.plot(xs, m, color=OURS, lw=1.3, zorder=3)
    for x, v, fc in zip(xs, m, (BASE, FIXED, OURS)):
        ax.scatter([x], [v], s=17, color=fc, edgecolors="white", linewidths=0.6, zorder=4)
    # mean labels next to the points
    dx, dy = N["a_base_lab"]
    ax.text(xs[0] + dx, m[0] + dy, st.fmt(m[0]), ha="right", va="center", fontsize=6.5, color=BASE, fontweight="bold")
    dx, dy = N["a_fixed_lab"]
    ax.text(xs[1] + dx, m[1] + dy, st.fmt(m[1]), ha="center", va="bottom", fontsize=6.5, color=OURS)
    # the effect: bracket from the base level to the final level, right of the final point
    xb = xs[2] + 0.2
    ax.plot([xs[0], xb], [m[0], m[0]], color=RULE, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.text(xs[2] + 0.11, m[2], st.fmt(m[2]), ha="left", va="bottom", fontsize=6.5, color=OURS, fontweight="bold")
    s_ = nums["a"]["final_minus_base"]
    st.bracket(ax, xb + 0.02, m[0], m[2] - 0.004, st.fmt(s_["mean_diff"], sign=True), color=ACCENT, w=0.07, pad=0.05)
    ax.text(xb + 0.14, (m[0] + m[2]) / 2 - 0.0125,
            f"[{st.fmt(s_['ci'][0])}, {st.fmt(s_['ci'][1])}]\n{s_['wins']}/{s_['n']} patients",
            fontsize=5.8, color=INK2, va="top", ha="left", linespacing=1.25, clip_on=False)
    ax.set_xticks(xs, ["lagged\nridge", "fixed\nuntrained bank", "final\nmodel"])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.42, 2.75)
    lo_, hi_ = min(base.min(), fixed.min(), final.min()), max(base.max(), fixed.max(), final.max())
    ax.set_ylim(lo_ - 0.012, hi_ + 0.008)
    ax.set_yticks(np.arange(np.ceil((lo_ - 0.012) * 20) / 20, hi_ + 0.008 + 1e-9, 0.05))
    ax.set_ylabel("test r, mean over contacts")
    st.range_frame(ax, x=False)
    st.headline(ax, "The final model is above ridge in every patient",
                tag=f"primary level · strict normalisation · {mins:.1f} min calibration")
    axa = ax

    # ---------------- b: all models (top right), session-wise normalisation
    T5 = load_dir(os.path.join(main_run, "readout_combo5"), "readout_")
    TF = load_dir(os.path.join(noenv, "readout_fixed"), "readout_")
    nfn = {s: d["models"]["nfn_mode"]["r"] for s, d in load_dir(os.path.join(runs, "nfn_baseline"), "nfn_").items()}
    rows = [("final model", model_r(T5, "combo"), OURS), (st.LBL_FIXED, model_r(TF, "combo"), FIXED),
            ("lagged ridge", model_r(T5, "base"), BASE), ("LeadFormer", find_leadformer(runs), BASE),
            ("NeuroFlowNet", nfn, BASE)]
    rows = [r for r in rows if r[1]]
    ax = fig.add_subplot(gs[0, 1])
    yy = np.arange(len(rows))[::-1].astype(float)
    ax.axhspan(yy[1] - 0.5, yy[0] + 0.5, color=WASH, lw=0, zorder=0)        # our two rows
    nums["b"] = {}
    for y, (lab, d, col) in zip(yy, rows):
        v = np.array(list(d.values()), float)
        off = st.beeswarm(v, size_data=0.0065, max_off=0.30)
        dot = col if col != BASE else "#b3aea5"
        ax.scatter(v, y + off, s=2.6, color=dot, linewidths=0, zorder=2, alpha=0.95)
        lo, hi = st.boot_ci(v)
        mc = OURS if col in (OURS, FIXED) else INK2
        ax.plot([lo, hi], [y, y], color=mc, lw=1.0, zorder=3)
        ax.scatter([v.mean()], [y], s=13, marker="D", color=col if col != BASE else INK2, edgecolors="white",
                   linewidths=0.5, zorder=4)
        ax.text(max(hi, v.max()) + 0.012, y, st.fmt(v.mean()), va="center", ha="left", fontsize=6.2, color=mc,
                fontweight="bold" if lab == "final model" else "normal")
        nums["b"][lab] = {"n": len(v), "mean": float(v.mean()), "ci": [lo, hi]}
    ax.set_yticks(yy, [r[0] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(OURS if r[2] in (OURS, FIXED) else INK)
    st.bare_y(ax)
    ax.set_ylim(yy[-1] - 0.6, yy[0] + 0.6)
    ax.set_xlim(0, 0.42)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("test r per patient")
    if "lagged ridge" in nums["b"]:
        st.ref_line(ax, nums["b"]["lagged ridge"]["mean"], axis="x")
    st.range_frame(ax, y=False)
    st.headline(ax, "Above every baseline on the same trials",
                tag="secondary level · session-wise normalisation · diamond = mean, bar = 95 % CI")
    axb = ax

    # ---------------- c: calibration efficiency (bottom right), primary level, strict
    ax = fig.add_subplot(gs[1, 1])
    fr = [f for f in ("0.05", "0.1", "0.2") if all(f in d["fracs"] for d in S5.values())]
    xm = np.array([np.mean([d["fracs"][f]["minutes"] for d in S5.values()]) for f in fr])
    Y = {k: np.array([[d["fracs"][f]["models"][k]["r"] for f in fr] for d in S5.values()]) for k in ("base", "combo_nosess")}
    nums["c"] = {"minutes": xm.tolist()}
    cdx = N["c_dx"]
    for k, col, lab, sh in (("base", BASE, "lagged ridge", -cdx), ("combo_nosess", OURS, "final model", cdx)):
        mu = Y[k].mean(0)
        ci = np.array([st.boot_ci(Y[k][:, j], 4000) for j in range(len(fr))])
        # CI as thin whiskers, nudged apart horizontally so the two models never sit on top of each other
        for x, (l, h) in zip(xm + sh, ci):
            ax.plot([x, x], [l, h], color=col, lw=0.55, alpha=0.75, solid_capstyle="butt", zorder=2)
        ax.plot(xm + sh, mu, color=col, lw=1.1, zorder=3)
        ax.scatter(xm + sh, mu, s=10, color=col, edgecolors="white", linewidths=0.5, zorder=4)
        ax.text(xm[-1] + 0.28, mu[-1], lab, color=col, fontsize=6.2, va="center", ha="left")
        nums["c"][lab] = mu.tolist()
        nums["c"][lab + "_ci"] = ci.tolist()
    note_top = -np.inf
    if "0.1" in fr and "0.2" in fr:
        j10, j20 = fr.index("0.1"), fr.index("0.2")
        yb20 = Y["base"][:, j20].mean()
        ax.plot([xm[0] - 0.25, xm[j20] - cdx], [yb20, yb20], color=RULE, lw=0.5, ls=(0, (2, 2)), zorder=0)
        s_ = paired_stats(Y["combo_nosess"][:, j10], Y["base"][:, j20])
        nums["c"]["final10_minus_base20"] = s_
        y10 = Y["combo_nosess"][:, j10].mean()
        x10 = xm[j10] + cdx
        ax.scatter([x10], [y10], s=48, facecolors="none", edgecolors=INK, linewidths=0.6, zorder=5)
        dx, dy = N["c_note"]
        note_top = y10 + dy + 0.027
        ax.annotate(f"final {xm[j10]:.1f} min − ridge {xm[j20]:.1f} min\n"
                    f"{st.fmt(s_['mean_diff'], sign=True)} [{st.fmt(s_['ci'][0])}, {st.fmt(s_['ci'][1])}], "
                    f"{s_['wins']}/{s_['n']}",
                    xy=(x10, y10), xytext=(x10 + dx, y10 + dy),
                    fontsize=5.8, color=INK, ha="left", va="bottom", linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.45, shrinkA=0.5, shrinkB=3.2,
                                    connectionstyle="angle,angleA=0,angleB=90,rad=0"))
    ax.set_xticks(xm, [f"{v:.1f}" for v in xm])
    ax.set_xlim(xm[0] - 0.4, xm[-1] + 1.9)
    allci = np.concatenate([np.ravel(nums["c"][k]) for k in nums["c"] if k.endswith("_ci")])
    lo_, hi_ = allci.min(), allci.max()
    ax.set_ylim(lo_ - 0.006, max(hi_, note_top) + 0.004)
    ax.set_yticks(np.round(np.arange(np.ceil(lo_ * 25) / 25, hi_ + 1e-9, 0.04), 3))
    ax.set_xlabel("calibration (min)")
    ax.set_ylabel("test r")
    st.range_frame(ax)
    st.headline(ax, "Half the calibration already beats ridge",
                tag="primary level · strict · same test trials · ds004752 only (not replicated, Fig. 7)")
    axc = ax

    st.letters(fig, [(axa, "a", -1.5), (axb, "b", -1.5), (axc, "c", -1.5)])
    st.save(fig, out, "fig2_main")
    dump_numbers(out, "fig2_main", nums)


FIGS = {"fig1": fig1_scene, "fig2": fig2_main}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--style", default="own", choices=["own", "sp"], help="sp = SciencePlots base style")
    a = ap.parse_args()
    st.setup(a.style)
    from s2i import paperfigs_b
    FIGS.update(paperfigs_b.FIGS)
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
