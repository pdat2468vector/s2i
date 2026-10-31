"""
Paper figures 3-7 and S1 (Phase 10 rework, 12 Oct 2026). Style: s2i/figstyle.py; helpers and loaders: s2i/paperfigs.py.
Registered in paperfigs.main(), so `python -m s2i.paperfigs --runs RUNS --out OUT --only fig3` renders them.

  fig3_transfer   a gain decomposition, primary level, strict (ridge -> fixed bank -> 3 fixed -> 3 learned -> final)
                  b frequency response of the shared filter bank, initial vs learned; c peak frequency per pair
  fig4_reliab     a calibration K-fold r vs test r per contact (primary, strict), per-patient Spearman in the inset
                  b coverage (primary, strict); b' top quarter - patient mean; c longevity (secondary, session-wise)
  fig5_power      band power without intracranial calibration (secondary): a per band, b zero-shot vs 4-min ridge
  fig6_protocol   a one factor at a time; b literature-inspired emulations; c per-patient target + reference effect
  fig7_external   a H1 per patient (ScaleOLDNEW); b each finding in both cohorts; c common mode vs interaction
  figS1_chanqc    ScaleOLDNEW channel audit (post hoc): raw SD vs |r| with inputs without a 10-05 name

Every figure writes <name>_numbers.json with each number printed on it.
"""
import colorsys
import glob
import json
import os

import numpy as np

from s2i import figstyle as st
from s2i.figstyle import (ACCENT, ACCENT_SOFT, BASE, EXT, FIXED, INK, INK2, INK3, MM, OURS, OURS_SOFT, PATIENT, RULE,
                          W2, plt)
from s2i.paperfigs import STRICT, dump_numbers, load_dir, load_strict, model_r, paired, paired_stats

# hand-tuned label offsets; edit after looking at the real-data render
NUDGE = {
    "fig3": {"total_x": 4.95},
    "fig4": {},
    "fig5": {"lab_gap": 0.0135},
    "fig6": {},
    "fig7": {},
}
P17 = "p17"


def _runs(runs):
    return os.path.join(runs, "shared_bip_car"), os.path.join(runs, "shared_no_env")


def _need(D, path):
    if not D:
        raise FileNotFoundError(2, "missing", path)
    return D


def _spread(ys, gap):
    """Push label positions apart (keeps order, minimal moves) so neighbours are at least `gap` apart."""
    o = np.argsort(ys)
    y = np.asarray(ys, float)[o].copy()
    for _ in range(50):
        moved = False
        for i in range(1, len(y)):
            if y[i] - y[i - 1] < gap:
                d = (gap - (y[i] - y[i - 1])) / 2
                y[i - 1] -= d
                y[i] += d
                moved = True
        if not moved:
            break
    out = np.empty_like(y)
    out[o] = y
    return out


def _patient_colours(n):
    return [colorsys.hls_to_rgb((0.04 + i / n) % 1, 0.66, 0.38) for i in range(n)]


def _short(s):
    return s.replace("sub-", "")


# ============================================================================= figure 3
def _bank_responses(main_run, nfft=4096, fs=200):
    """(f, H0 initial, Hs learned per fold) of the quadrature pairs, peak-normalised. Learned banks are read from the
    encoder checkpoints (torch); without torch a cached bank_responses.npz in the same folder is used."""
    encs = sorted(glob.glob(os.path.join(main_run, "enc_sub-*.pt")))
    try:
        import torch
        from s2i.shared import TemporalEncoder
    except ImportError:
        torch = None
    if encs and torch is not None:
        def resp(W, n):
            f = np.fft.rfftfreq(nfft, 1 / fs)
            H = np.sqrt(np.abs(np.fft.rfft(W[0:2 * n:2], nfft)) ** 2 + np.abs(np.fft.rfft(W[1:2 * n:2], nfft)) ** 2)
            return f, H / H.max(1, keepdims=True)
        cfg = torch.load(encs[0], map_location="cpu", weights_only=False)["cfg"]
        n = cfg["n_pairs"]
        f, H0 = resp(TemporalEncoder(**cfg).conv.weight.detach().numpy()[:, 0], n)
        Hs = [resp(torch.load(p, map_location="cpu", weights_only=False)["state"]["conv.weight"].numpy()[:, 0], n)[1]
              for p in encs]
        return f, H0, np.array(Hs)
    cache = os.path.join(main_run, "bank_responses.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["f"], z["H0"], z["Hs"]
    raise FileNotFoundError(2, "missing", os.path.join(main_run, "enc_sub-*.pt"))


def fig3_transfer(runs, out):
    N = NUDGE["fig3"]
    main_run, noenv = _runs(runs)
    S5 = _need(load_strict(main_run), main_run)
    L = lambda name: load_dir(os.path.join(noenv, name + STRICT), "readout_")   # noqa: E731
    SF, F3, L3 = L("readout_fixed_strict"), L("readout_fixed_arch3_strict"), L("readout_learned_arch3_strict")
    _need(F3, os.path.join(noenv, "readout_fixed_arch3_strict" + STRICT))
    steps = [("lagged\nridge", model_r(S5, "base"), BASE), ("fixed\nbank", model_r(SF, "combo_nosess"), FIXED),
             ("3 fixed\nbanks", model_r(F3, "combo_nosess"), FIXED), ("3 learned\nbanks", model_r(L3, "combo_nosess"), ACCENT),
             ("final\nmodel", model_r(S5, "combo_nosess"), OURS)]
    subs, V = paired(*[s[1] for s in steps])
    m = np.array([v.mean() for v in V])
    mins = float(np.mean([S5[s]["fracs"]["0.2"].get("minutes", np.nan) for s in subs]))
    nums = {"a": {"n": len(subs), "minutes": mins, "means": dict(zip([s[0].replace("\n", " ") for s in steps], m.tolist())),
                  "steps": {}, "final_minus_base": paired_stats(V[-1], V[0])}}

    fig = plt.figure(figsize=(W2, 72 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.08], height_ratios=[1.55, 1.0], wspace=0.24, hspace=0.62,
                          left=0.075, right=0.985, top=0.875, bottom=0.12)

    # ---------------- a: waterfall; bar = mean step, dots = previous mean + each patient's step
    ax = fig.add_subplot(gs[:, 0])
    w_ = 0.58
    ax.plot([-w_ / 2, w_ / 2], [m[0], m[0]], color=BASE, lw=2.4, solid_capstyle="butt", zorder=3)
    ax.text(-w_ / 2 - 0.08, m[0], st.fmt(m[0]), ha="right", va="center", fontsize=6.3, color=BASE, fontweight="bold")
    for k in range(1, len(steps)):
        lab, _, col = steps[k]
        d = V[k] - V[k - 1]
        y0, y1 = m[k - 1], m[k]
        ax.add_patch(plt.Rectangle((k - w_ / 2, min(y0, y1)), w_, abs(y1 - y0), fc=col, ec="none", zorder=2,
                                   alpha=0.9 if col != FIXED else 1.0))
        ax.plot([k - 1 + w_ / 2, k - w_ / 2], [y0, y0], color=INK3, lw=0.45, ls=(0, (1.5, 1.5)), zorder=1)
        off = st.beeswarm(y0 + d, size_data=0.0012, max_off=0.24)
        dc = INK if col == ACCENT else (OURS if col in (FIXED, OURS) else INK2)
        ax.scatter(k + off, y0 + d, s=3.2, color="white", edgecolors=dc, linewidths=0.45, zorder=4)
        s_ = paired_stats(V[k], V[k - 1])
        nums["a"]["steps"][lab.replace("\n", " ")] = s_
        top = max(y1, (y0 + d).max())
        hot = col == ACCENT
        ax.text(k, top + 0.0022, st.fmt(s_["mean_diff"], sign=True), ha="center", va="bottom", fontsize=6.6 if hot else 6.2,
                color=ACCENT if hot else (OURS if col != BASE else INK2), fontweight="bold" if hot else "normal")
        ax.text(k, top + 0.0022 + 0.0042, f"{s_['wins']}/{s_['n']}", ha="center", va="bottom", fontsize=5.6, color=INK3)
    ax.text(len(steps) - 1 + w_ / 2 + 0.08, m[-1], st.fmt(m[-1]), ha="left", va="center", fontsize=6.3, color=OURS,
            fontweight="bold")
    xt = N["total_x"]
    s_ = nums["a"]["final_minus_base"]
    ax.plot([0 + w_ / 2, xt], [m[0], m[0]], color=RULE, lw=0.5, ls=(0, (2, 2)), zorder=0)
    st.bracket(ax, xt, m[0], m[-1], st.fmt(s_["mean_diff"], sign=True), color=INK, w=0.08, pad=0.06, fs=6.4)
    ax.set_xticks(range(len(steps)), [s[0] for s in steps])
    for t, s in zip(ax.get_xticklabels(), steps):
        t.set_color(ACCENT if s[2] == ACCENT else INK)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.spines["bottom"].set_visible(False)
    lo = min(m[0], min((V[k] - V[k - 1] + m[k - 1]).min() for k in range(1, len(steps))))
    hi = max(m.max(), max((V[k] - V[k - 1] + m[k - 1]).max() for k in range(1, len(steps)))) + 0.011
    ax.set_xlim(-0.75, xt + 0.55)
    ax.set_ylim(lo - 0.004, hi)
    ax.set_yticks(np.round(np.arange(np.ceil((lo - 0.004) * 100) / 100, hi + 1e-9, 0.01), 2))
    ax.set_ylabel("test r, mean over patients")
    st.range_frame(ax, x=False)
    st.headline(ax, "Learning the filters adds to a fixed bank",
                tag=f"primary level · strict · {mins:.1f} min calibration · dots = patients")
    axa = ax

    # ---------------- b: frequency response; c: peak frequency per pair (shared log-frequency axis)
    f, H0, Hs = _bank_responses(main_run)
    H1 = Hs.mean(0)
    n = H0.shape[0]
    sel = (f >= 1.5) & (f <= 80)
    axb = fig.add_subplot(gs[0, 1])
    for k in range(n):
        axb.fill_between(f[sel], H0[k, sel], H1[k, sel], color=ACCENT, alpha=0.10, lw=0, zorder=1)
        axb.plot(f[sel], H0[k, sel], color=INK3, lw=0.5, ls=(0, (2.2, 1.6)), zorder=2)
        axb.plot(f[sel], H1[k, sel], color=OURS, lw=0.8, zorder=3)
    axb.set_xscale("log")
    axb.set_xlim(1.5, 80)
    axb.set_ylim(0, 1.04)
    axb.set_yticks([0, 0.5, 1.0])
    axb.set_ylabel("gain")
    axb.tick_params(axis="x", which="both", labelbottom=False)
    st.range_frame(axb, x=False)
    axb.spines["bottom"].set_visible(False)
    st.headline(axb, "Shared filter bank, before and after learning",
                tag=f"dashed = initial · solid = learned, mean of {len(Hs)} folds · {n} quadrature pairs")

    axc = fig.add_subplot(gs[1, 1], sharex=axb)
    pk0 = f[np.argmax(H0 * sel, 1)]
    pk = np.array([f[np.argmax(H * sel, 1)] for H in Hs])
    sh = np.log2(pk / pk0[None])                          # shift of each pair's peak, in octaves
    med, plo, phi = np.median(sh, 0), np.percentile(sh, 10, 0), np.percentile(sh, 90, 0)
    st.ref_line(axc, 0)
    for k in range(n):
        axc.plot([pk0[k], pk0[k]], [0, med[k]], color=ACCENT, lw=1.1, alpha=0.75, zorder=1, solid_capstyle="butt")
        axc.plot([pk0[k] * 1.035] * 2, [plo[k], phi[k]], color=OURS, lw=0.5, alpha=0.7, zorder=2)
    axc.scatter(pk0, med, s=9, color=OURS, edgecolors="white", linewidths=0.4, zorder=4)
    axc.set_xscale("log")
    axc.set_xticks([2, 4, 8, 16, 32, 64], ["2", "4", "8", "16", "32", "64"])
    axc.minorticks_off()
    axc.set_xlim(1.5, 80)
    m_ = max(0.25, np.abs(np.r_[plo, phi]).max() * 1.05)
    yt = 0.25 if m_ < 0.6 else 0.5
    m_ = np.ceil(m_ / yt - 1e-9) * yt                     # whiskers end inside the range frame
    axc.set_ylim(-m_ * 1.04, m_ * 1.04)
    axc.set_yticks(np.arange(-m_, m_ + 1e-9, yt))
    axc.set_ylabel("peak shift (oct.)")
    axc.set_xlabel("frequency (Hz)")
    st.range_frame(axc)
    st.headline(axc, "Peak shift per pair", tag="median and 10–90 % of folds, at the initial peak frequency")
    nums["b"] = {"n_pairs": int(n), "n_folds": int(len(Hs))}
    nums["c"] = {"peak_initial_hz": pk0.tolist(), "peak_shift_oct_median": med.tolist()}

    st.letters(fig, [(axa, "a", -1.5), (axb, "b", -1.5), (axc, "c", -1.5)])
    st.save(fig, out, "fig3_transfer")
    dump_numbers(out, "fig3_transfer", nums)


# ============================================================================= figure 4
def _mixed_slope(rows):
    """Slope of r on session gap with a random intercept per patient (statsmodels); OLS if unavailable."""
    G = np.array([r[1] for r in rows], float)
    V = np.array([r[2] for r in rows], float)
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        df = pd.DataFrame(rows, columns=["subject", "gap", "r"])
        fit = smf.mixedlm("r ~ gap", df, groups=df["subject"]).fit()
        ci = fit.conf_int().loc["gap"].tolist()
        return float(fit.params["gap"]), float(fit.params["Intercept"]), ci, "mixed"
    except Exception:
        s, i = np.polyfit(G, V, 1)
        return float(s), float(i), [np.nan, np.nan], "ols"


def fig4_reliab(runs, out):
    from scipy.stats import spearmanr
    main_run, _ = _runs(runs)
    S5 = _need(load_strict(main_run), main_run)
    subs = sorted(S5)
    cols = _patient_colours(len(subs))
    per = {}
    for s in subs:
        mm = S5[s]["fracs"]["0.2"]["models"]["combo_nosess"]
        cv, te = np.array(mm["cv_contacts"], float), np.array(mm["contacts"], float)
        ok = np.isfinite(cv) & np.isfinite(te)
        per[s] = (cv[ok], te[ok])
    rho = np.array([spearmanr(*per[s]).correlation for s in subs])
    cv = np.concatenate([per[s][0] for s in subs])
    te = np.concatenate([per[s][1] for s in subs])
    nums = {"a": {"n_contacts": int(len(cv)), "rho_per_patient": dict(zip(subs, rho.tolist())),
                  "rho_median": float(np.median(rho)), "rho_mean": float(rho.mean()),
                  "rho_range": [float(rho.min()), float(rho.max())],
                  "rho_pooled_reference_only": float(spearmanr(cv, te).correlation)}}

    fig = plt.figure(figsize=(W2, 92 * MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.42, 0.5, 0.92], height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.72,
                          left=0.07, right=0.985, top=0.905, bottom=0.095)

    # ---------------- a: square scatter, light per-patient colours
    ax = fig.add_subplot(gs[:, 0])
    lim = (np.floor(min(np.percentile(cv, 0.3), np.percentile(te, 0.3)) * 10) / 10,
           np.ceil(max(np.percentile(cv, 99.8), np.percentile(te, 99.8)) * 10) / 10)
    ax.plot(lim, lim, color=RULE, lw=0.5, zorder=0)
    for s, c in zip(subs, cols):
        ax.scatter(*per[s], s=3.2, color=c, alpha=0.75, linewidths=0, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    tk = np.round(np.arange(lim[0], lim[1] + 1e-9, 0.2), 1)
    ax.set_xticks(tk)
    ax.set_yticks(tk)
    ax.set_xlabel("calibration K-fold r (before use)")
    ax.set_ylabel("test r (later trials)")
    st.range_frame(ax)
    ax.text(0.04, 0.965, f"ρ = {np.median(rho):.2f}", transform=ax.transAxes, fontsize=9, color=OURS,
            fontweight="bold", va="top", ha="left")
    ax.text(0.04, 0.885, "median per patient", transform=ax.transAxes, fontsize=5.8, color=INK2, va="top", ha="left")
    # inset: one dot per patient on a 0-1 rho scale (lower right is empty in a correlated scatter)
    ins = ax.inset_axes([0.55, 0.08, 0.42, 0.13])
    off = st.beeswarm(rho, size_data=0.03, max_off=0.42)
    ins.scatter(rho, off, s=6, c=cols, linewidths=0.3, edgecolors="white", zorder=2)
    ins.plot([np.median(rho)] * 2, [-0.55, 0.55], color=OURS, lw=1.0, zorder=3)
    ins.set_xlim(0, 1)
    ins.set_ylim(-0.6, 0.6)
    ins.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ins.tick_params(labelsize=5.4, length=1.5, pad=1)
    ins.set_yticks([])
    for sp in ("left", "top", "right"):
        ins.spines[sp].set_visible(False)
    ins.patch.set_alpha(0)
    ins.set_title("ρ per patient", fontsize=5.6, color=INK2, pad=1.5, loc="left")
    st.headline(ax, "Calibration predicts which contacts work",
                tag=f"final model · primary level · strict · {len(cv)} contacts, 15 colours = patients")
    axa = ax

    # ---------------- b: coverage curve (pooled contacts, most reliable first)
    ax = fig.add_subplot(gs[0, 1:])
    order = np.argsort(-cv)
    kept = np.arange(1, len(cv) + 1) / len(cv) * 100
    cum = np.cumsum(te[order]) / np.arange(1, len(cv) + 1)
    k0 = 10
    ax.plot(kept[k0:], cum[k0:], color=OURS, lw=1.3, zorder=3)
    ax.fill_between(kept[k0:], te.mean(), cum[k0:], color=OURS_SOFT, lw=0, zorder=1)
    st.ref_line(ax, te.mean())
    ax.text(52, te.mean() - 0.005, f"all contacts {st.fmt(te.mean())}", ha="center", va="top", fontsize=5.8,
            color=INK2)
    nums["b"] = {"all_contacts_mean": float(te.mean()), "thresholds": {}}
    for thr in (0.3, 0.4, 0.5):
        sel = cv > thr
        if sel.sum() < 10:
            continue
        x, y = sel.mean() * 100, te[sel].mean()
        ax.scatter([x], [y], s=13, color=ACCENT, zorder=4, edgecolors="white", linewidths=0.5)
        ax.text(x + 1.6, y + 0.004, f"> {thr:.1f}", fontsize=5.8, color=ACCENT, va="bottom", ha="left")
        nums["b"]["thresholds"][str(thr)] = {"kept_pct": float(x), "mean_test_r": float(y)}
    ax.set_xlim(0, 100)
    ax.set_ylim(te.mean() - 0.03, cum[k0:].max() + 0.02)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("contacts kept (%), most reliable first")
    ax.set_ylabel("mean test r")
    st.range_frame(ax)
    st.headline(ax, "Keeping reliable contacts raises accuracy", tag="orange = calibration r threshold")
    axb = ax

    # ---------------- b': top quarter (by calibration r) minus the patient's mean, per patient
    ax = fig.add_subplot(gs[1, 1])
    dq = np.array([per[s][1][per[s][0] >= np.percentile(per[s][0], 75)].mean() - per[s][1].mean() for s in subs])
    off = st.beeswarm(dq, size_data=0.008, max_off=0.34)
    ax.scatter(off, dq, s=6, c=cols, linewidths=0.3, edgecolors="white", zorder=2)
    lo, hi = st.boot_ci(dq)
    ax.plot([0.5, 0.5], [lo, hi], color=OURS, lw=1.0, zorder=3)
    ax.scatter([0.5], [dq.mean()], s=14, marker="D", color=OURS, edgecolors="white", linewidths=0.5, zorder=4)
    ax.text(0.62, dq.mean(), st.fmt(dq.mean(), sign=True), fontsize=6.2, color=OURS, fontweight="bold", va="center")
    ax.text(0.62, dq.mean() - 0.022, f"{int((dq > 0).sum())}/{len(dq)}", fontsize=5.6, color=INK3, va="center")
    st.ref_line(ax, 0)
    ax.set_xlim(-0.5, 1.2)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(min(0, dq.min()) - 0.01, dq.max() + 0.015)
    ax.set_ylabel("Δ test r")
    st.range_frame(ax, x=False)
    st.headline(ax, "Top quarter", tag="vs patient mean")
    nums["b_prime"] = {"top_quarter_minus_mean": paired_stats(dq, np.zeros_like(dq))}
    axb2 = ax

    # ---------------- c: longevity across sessions (secondary level, session-wise normalisation)
    ax = fig.add_subplot(gs[1, 2])
    lon = load_dir(os.path.join(main_run, "readout_longevity"), "readout_")
    rows, rows_adv = [], []
    for d in lon.values():
        fr, order_ = d["fracs"]["0.2"], d["session_order"]
        last = order_.index(fr["calib_sessions"][-1])
        mods = fr["models"]
        pts = []
        for s, v in mods["combo"].get("sessions", {}).items():
            if d["test_trials_per_session"].get(s, 0) >= 5:
                pts.append((order_.index(s) - last, v, mods.get("base", {}).get("sessions", {}).get(s)))
        pts.sort()
        if len(pts) >= 2:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PATIENT, lw=0.5, zorder=1)
        rows += [(d["subject"], p[0], p[1]) for p in pts]
        rows_adv += [(d["subject"], p[0], p[1] - p[2]) for p in pts if p[2] is not None]
    nums["c"] = {"n_patient_sessions": len(rows)}
    if rows:
        slope, icpt, ci, kind = _mixed_slope(rows)
        G = np.array([r[1] for r in rows])
        gx = np.array([G.min(), G.max()])
        ax.plot(gx, icpt + slope * gx, color=OURS, lw=1.5, zorder=3)
        ax.text(gx[1] + 0.12, icpt + slope * gx[1], f"{st.fmt(slope, sign=True)}\nper session", ha="left", va="center",
                fontsize=6.0, color=OURS, fontweight="bold", linespacing=1.15)
        nums["c"].update({"slope": slope, "slope_ci": ci, "model": kind})
        if rows_adv:
            sa, _, cia, _ = _mixed_slope(rows_adv)
            nums["c"].update({"advantage_slope": sa, "advantage_slope_ci": cia})
        ax.set_xticks(range(int(G.min()), int(G.max()) + 1))
        ax.set_xlim(G.min() - 0.3, G.max() + 1.6)
    ax.set_xlabel("sessions after calibration")
    ax.set_ylabel("test r")
    st.range_frame(ax)
    st.headline(ax, "Accuracy drifts across sessions", tag="secondary level · lines = patients")
    axc = ax

    st.letters(fig, [(axa, "a", -1.5), (axb, "b", -1.5), (axb2, "c", -1.5), (axc, "d", -1.5)])
    st.save(fig, out, "fig4_reliability")
    dump_numbers(out, "fig4_reliability", nums)


# ============================================================================= figure 5
BAND_SYM = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β"}


def fig5_power(runs, out):
    N = NUDGE["fig5"]
    p = os.path.join(runs, "zeroshot_power", "zeroshot_subjects.json")
    if not os.path.exists(p):
        raise FileNotFoundError(2, "missing", p)
    zs = json.load(open(p))
    bands = list(zs[0]["bands"])
    # (key, label, colour, line style, marker, emphasis)
    rows = [("zs_global", "global scalp average", "#b9b4ab", (0, (1.2, 1.4)), "^", 0),
            ("zs_nearest", "nearest scalp channel", "#a8a39a", (0, (1.2, 1.4)), "s", 0),
            ("zs_geo_shuf", "positions shuffled", "#8f8a82", (0, (3, 1.6)), "o", 0),
            ("cal_ridge", "ridge, 4 min SEEG", BASE, "-", "o", 1),
            ("zs_geo", "zero-shot, no SEEG", ACCENT, "-", "D", 2),
            ("zs_cal", "zero-shot + ridge", OURS, "-", "o", 1)]
    get = lambda m, b: np.array([r["bands"][b][m]["r_induced"] for r in zs], float)   # noqa: E731
    fig = plt.figure(figsize=(W2, 66 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.3, left=0.065, right=0.985, top=0.85, bottom=0.15)

    ax = fig.add_subplot(gs[0])
    xb = np.arange(len(bands), dtype=float)
    nums = {"a": {}, "n": len(zs)}
    # patients behind the zero-shot line
    for bi, b in enumerate(bands):
        v = get("zs_geo", b)
        off = st.beeswarm(v, size_data=0.006, max_off=0.2)
        ax.scatter(bi + off * 0.6, v, s=2.4, color=ACCENT_SOFT, edgecolors="#e2b7a6", linewidths=0.3, zorder=1)
    ends = []
    dodge = np.linspace(-0.075, 0.075, len(rows))
    for (m, lab, col, ls, mk, emph), dx in zip(rows, dodge):
        mu = np.array([get(m, b).mean() for b in bands])
        ci = np.array([st.boot_ci(get(m, b), 4000) for b in bands])
        ax.plot(xb + dx, mu, color=col, lw=(0.7, 0.9, 1.5)[emph], ls=ls, zorder=2 + emph)
        for x, (l_, h_) in zip(xb + dx, ci):
            ax.plot([x, x], [l_, h_], color=col, lw=0.5, alpha=0.8, zorder=2 + emph, solid_capstyle="butt")
        ax.scatter(xb + dx, mu, s=(9, 11, 16)[emph], marker=mk, color=col if emph else "white", edgecolors=col,
                   linewidths=0.7, zorder=3 + emph)
        ends.append(mu[-1])
        nums["a"][m] = {b: {"mean": float(u), "ci": c.tolist()} for b, u, c in zip(bands, mu, ci)}
    ly = _spread(ends, N["lab_gap"])
    for (m, lab, col, ls, mk, emph), y0, y1 in zip(rows, ends, ly):
        ax.plot([xb[-1] + 0.12, xb[-1] + 0.2], [y0, y1], color=col, lw=0.4, zorder=1)
        ax.text(xb[-1] + 0.23, y1, lab, color=col if emph else INK2, fontsize=6.0, va="center", ha="left",
                fontweight="bold" if emph == 2 else "normal")
    ax.set_xticks(xb, [BAND_SYM.get(b, b) for b in bands], fontsize=8)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.3, len(bands) - 1 + 1.25)
    ax.set_ylabel("r of log band power (induced)")
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))
    st.range_frame(ax, x=False)
    st.headline(ax, "Band power of a new patient's contacts, without their SEEG",
                tag="secondary level · leave one patient out · mean and 95 % CI · dots = patients (zero-shot)")
    axa = ax

    # ---------------- b: zero-shot vs 4-min ridge per patient (mean over bands)
    ax = fig.add_subplot(gs[1])
    zg = np.mean([get("zs_geo", b) for b in bands], 0)
    cr = np.mean([get("cal_ridge", b) for b in bands], 0)
    lim = (np.floor(min(zg.min(), cr.min()) * 20) / 20, np.ceil(max(zg.max(), cr.max()) * 20) / 20)
    ax.plot(lim, lim, color=RULE, lw=0.5, zorder=0)
    ax.scatter(cr, zg, s=16, color=ACCENT, edgecolors="white", linewidths=0.6, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    tk = np.round(np.arange(lim[0], lim[1] + 1e-9, 0.05), 2)
    ax.set_xticks(tk)
    ax.set_yticks(tk)
    ax.set_xlabel("ridge, 4 min of the patient's SEEG")
    ax.set_ylabel("zero-shot, no SEEG")
    st.range_frame(ax)
    above = int((zg > cr).sum())
    ax.text(0.05, 0.95, f"{above}/{len(zg)} above", transform=ax.transAxes, fontsize=6.0, color=ACCENT, va="top")
    st.headline(ax, "Comparable to 4 min of calibration", tag="per patient · mean over the four bands")
    nums["b"] = {"zero_shot_mean": float(zg.mean()), "ridge_mean": float(cr.mean()), "zero_shot_above": above}
    axb = ax

    st.letters(fig, [(axa, "a", -1.5), (axb, "b", -1.5)])
    st.save(fig, out, "fig5_band_power")
    dump_numbers(out, "fig5_band_power", nums)


# ============================================================================= figure 6
ONE_FACTOR = [   # protocol.py label -> short label (what was changed relative to our protocol)
    ("split: random vs chronological (20 % calibration)", "random split, 20 %"),
    ("split: random vs chronological (90 % training)", "random split, 90 %"),
    ("amount: 90 % vs 20 % training, both chronological", "90 % training"),
    ("units: random 1-s segments vs random trials (80 %)", "1-s segments"),
    ("target: monopolar vs bipolar (CAR scalp, clean split)", "monopolar target"),
    ("scalp reference: original vs CAR (bipolar target, clean split)", "original scalp reference"),
    ("target + reference: monopolar/original vs bipolar/CAR", "target + reference"),
    ("metric: NeuroFlowNet-style vs ours (same predictions, clean)", "flattened metric"),
]
KEY_ROW = "target + reference"


def fig6_protocol(runs, out):
    pdir = os.path.join(runs, "protocol")
    pj = os.path.join(pdir, "protocol.json")
    if not os.path.exists(pj):
        raise FileNotFoundError(2, "missing", pj)
    summ = json.load(open(pj))
    per = load_dir(pdir, "protocol_")
    n = len(per) or 15
    fig = plt.figure(figsize=(W2, 82 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[1.0, 1.08], wspace=0.5, hspace=0.78,
                          left=0.15, right=0.975, top=0.89, bottom=0.1)

    # ---------------- a: one factor at a time
    ax = fig.add_subplot(gs[:, 0])
    rows = [(lab, summ[f"one_factor/{k}/enc"]) for k, lab in ONE_FACTOR if f"one_factor/{k}/enc" in summ]
    y = np.arange(len(rows))[::-1].astype(float)
    nums = {"a": {}}
    for yi, (lab, r) in zip(y, rows):
        key = lab == KEY_ROW
        col = ACCENT if key else OURS
        lo, hi = r["ci"]
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.6 if key else 1.0, solid_capstyle="butt", zorder=2)
        ax.scatter([r["diff"]], [yi], s=30 if key else 18, zorder=3, color=col if sig else "white", edgecolors=col,
                   linewidths=0.9)
        ax.text(hi + 0.008, yi, st.fmt(r["diff"], sign=True) if abs(r["diff"]) >= 5e-4 else "0.000", va="center",
                ha="left", fontsize=6.6 if key else 6.0, color=ACCENT if key else INK2,
                fontweight="bold" if key else "normal")
        nums["a"][lab] = r
    sing = [r["diff"] for lab, r in rows if lab in ("monopolar target", "original scalp reference")]
    ki = [i for i, (lab, _) in enumerate(rows) if lab == KEY_ROW]
    if ki and len(sing) == 2:
        yk = y[ki[0]]
        add = sum(sing)
        ax.plot([add, add], [yk - 0.3, yk + 0.3], color=INK, lw=0.8, zorder=4)
        ax.text(add, yk + 0.36, "additive", ha="center", va="bottom", fontsize=5.6, color=INK)
        nums["a"]["additive_expectation"] = add
    ax.axvline(0, color=INK2, lw=0.5, zorder=0)
    ax.set_yticks(y, [lab for lab, _ in rows])
    for t, (lab, _) in zip(ax.get_yticklabels(), rows):
        t.set_color(ACCENT if lab == KEY_ROW else INK)
        t.set_fontweight("bold" if lab == KEY_ROW else "normal")
    st.bare_y(ax)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("change in mean test r")
    xmax = max(r["ci"][1] for _, r in rows)
    ax.set_xlim(min(-0.03, min(r["ci"][0] for _, r in rows) - 0.01), xmax + 0.06)
    st.range_frame(ax, y=False)
    st.headline(ax, "Changing target and reference together inflates r",
                tag=f"one change at a time · single encoder · {n} patients · filled = CI excludes 0")
    axa = ax

    # ---------------- b: composite emulations
    ax = fig.add_subplot(gs[0, 1])
    ours_k = "recipe/ours (clean, r_concat, all patients)"
    rec = [("ours", ours_k, OURS),
           ("random 90 %", "rand_trials90/r_concat", BASE),
           ("chronological 90 %", "chrono90/r_concat", BASE),
           ("NeuroFlowNet-like", "recipe/NeuroFlowNet-like (random 90 % trials, target de-normalisation, flattened r)", BASE),
           ("monopolar + original ref.", "mono_orig/r_concat", ACCENT),
           ("CAST-like", "recipe/CAST-like (monopolar, orig ref, random segments, viable patients)", ACCENT)]
    rec = [r for r in rec if r[1] in summ]
    base_r = summ[ours_k]["enc"]
    yy = np.arange(len(rec))[::-1].astype(float)
    nums["b"] = {}
    for yi, (lab, k, col) in zip(yy, rec):
        v = summ[k]["enc"]
        ax.plot([0, v], [yi, yi], color=col if col != BASE else "#b3aea5", lw=2.6, solid_capstyle="butt", zorder=2)
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=6.1, color=col if col != BASE else INK2,
                fontweight="bold" if col != BASE else "normal")
        ax.text(v + 0.075, yi, f"{v / base_r:.1f}×" if lab != "ours" else "", va="center", fontsize=5.6, color=INK3)
        nums["b"][lab] = {"r": v, "ratio": v / base_r}
    nk = "recipe/NeuroFlowNet metric on RANDOM predictions"
    if nk in summ:
        xn = summ[nk]["enc"]
        ax.plot([xn, xn], [yy[-1] - 0.5, yy[0] + 0.5], color=INK2, lw=0.5, ls=(0, (2, 1.5)), zorder=3)
        ax.text(xn + 0.008, yy[-1] - 0.5, "random predictions, flattened metric", fontsize=5.4, color=INK2,
                va="top", ha="left")
        nums["b"]["random_predictions_flattened"] = xn
    ax.set_yticks(yy, [r[0] for r in rec])
    for t, r in zip(ax.get_yticklabels(), rec):
        t.set_color(r[2] if r[2] != BASE else INK)
    st.bare_y(ax)
    ax.set_ylim(yy[-1] - 1.0, yy[0] + 0.6)
    ax.set_xlim(0, 0.62)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("mean test r, same model")
    st.range_frame(ax, y=False)
    st.headline(ax, "Literature-inspired protocols", tag="emulations, not reproductions")
    axb = ax

    # ---------------- c: per patient, our protocol -> monopolar target + original reference
    ax = fig.add_subplot(gs[1, 1])
    axc = ax
    if per:
        subs = sorted(per)
        a_ = np.array([per[s]["clean"]["enc"]["r_concat"] for s in subs])
        b_ = np.array([per[s]["mono_orig"]["enc"]["r_concat"] for s in subs])
        jump = b_ - a_
        top = set(np.argsort(-jump)[:4].tolist())
        for i in range(len(subs)):
            hot = i in top
            ax.plot([0, 1], [a_[i], b_[i]], color=ACCENT if hot else PATIENT, lw=0.9 if hot else 0.5, zorder=2 if hot else 1)
            ax.scatter([0, 1], [a_[i], b_[i]], s=5 if hot else 3, color=ACCENT if hot else PATIENT, linewidths=0,
                       zorder=3)
        lo_, hi_ = np.floor(min(a_.min(), b_.min()) / 0.2) * 0.2, np.ceil(max(a_.max(), b_.max()) / 0.2) * 0.2
        lab = sorted((b_[i], _short(subs[i])) for i in top)
        ys = _spread([v for v, _ in lab], 0.08 * (hi_ - lo_ + 0.04))
        for (v, s), yl in zip(lab, ys):
            ax.text(1.08, yl, s, fontsize=5.8, color=ACCENT, va="center")
        ax.set_xticks([0, 1], ["ours", "monopolar +\noriginal ref."])
        ax.tick_params(axis="x", length=0, pad=3)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlim(-0.3, 1.4)
        ax.set_ylim(lo_ - 0.02, hi_ + 0.02)
        ax.set_yticks(np.round(np.arange(lo_, hi_ + 1e-9, 0.2), 1))
        ax.set_ylabel("test r")
        st.range_frame(ax, x=False)
        nums["c"] = {"top4": sorted(_short(subs[i]) for i in top), "jump_top4": [float(jump[i]) for i in sorted(top)]}
        st.headline(ax, "Carried by four patients", tag="lines = patients")
    st.letters(fig, [(axa, "a", -1.5), (axb, "b", -1.5), (axc, "c", -1.5)])
    st.save(fig, out, "fig6_protocol")
    dump_numbers(out, "fig6_protocol", nums)


# ============================================================================= figure 7
def _interaction(p_):
    return p_["mono|orig"] - p_["mono|car"] - p_["bip|orig"] + p_["bip|car"]


def _p17_load(R, name):
    out = {}
    for f in glob.glob(os.path.join(R, name, "readout_sub-P*.json")):
        d = json.load(open(f))
        out[d["subject"]] = d
    return out


def _p17_ex(D, m):
    return {s: r["explicit"][m]["mean"]["r"] for s, r in D.items() if m in r.get("explicit", {})}


def _ds_findings(runs):
    """The same five comparisons on ds004752 (primary level, strict; interaction: volts, lagged ridge)."""
    main_run, noenv = _runs(runs)
    S5 = load_strict(main_run)
    L = lambda name: load_dir(os.path.join(noenv, name + STRICT), "readout_")   # noqa: E731
    SF, F3, L3 = L("readout_fixed_strict"), L("readout_fixed_arch3_strict"), L("readout_learned_arch3_strict")
    out = {}

    def add(key, a, b):
        if a and b:
            _, (x, y) = paired(a, b)
            out[key] = paired_stats(x, y)
    add("h1", model_r(S5, "combo_nosess"), model_r(S5, "base"))
    add("h2", model_r(L3, "combo_nosess"), model_r(F3, "combo_nosess"))
    add("fixed1", model_r(S5, "combo_nosess"), model_r(SF, "combo_nosess"))
    add("half", model_r(S5, "combo_nosess", "0.1"), model_r(S5, "base", "0.2"))
    V = load_dir(os.path.join(runs, "review13"), "volts_")
    cm = {}
    if V:
        it = np.array([_interaction(v["pred"]) for v in V.values()])
        out["inter"] = paired_stats(it, np.zeros_like(it))
        out["inter"]["source"] = "review13 volts, lagged ridge"
        cm = {s: (abs(v["r_scalp_cm_vs_ieeg_cm"]), _interaction(v["pred"])) for s, v in V.items()}
    return out, cm


def fig7_external(runs, out):
    R = os.path.join(runs, P17)
    rp = os.path.join(R, "p17_report.json")
    if not os.path.exists(rp):
        raise FileNotFoundError(2, "missing", rp)
    rep = json.load(open(rp))
    setch_p = os.path.join(R, "p17_setch_report.json")
    setch = json.load(open(setch_p)) if os.path.exists(setch_p) else {}
    desc = {d["comparison"]: d for d in rep.get("descriptive", []) if d.get("level") == "primary"}
    ext = {"h1": rep["H1"], "h2": rep["H2"], "fixed1": desc.get("Final − Fixed-1"),
           "half": next((v for k, v in desc.items() if k.startswith("Final, 10 %")), None)}
    cm17 = rep.get("common_mode", {})
    if cm17:
        it = np.array([_interaction(v["pred"]) for v in cm17.values()])
        ext["inter"] = paired_stats(it, np.zeros_like(it))
    add = {"h1": setch.get("primary_setch|H1: Final − Base"), "h2": setch.get("primary_setch|H2: Learned-3 − Fixed-3")}
    ds, cm_ds = _ds_findings(runs)
    nums = {"ds004752": ds, "ScaleOLDNEW": {k: v for k, v in ext.items() if v}, "addendum_A": add}

    fig = plt.figure(figsize=(W2, 70 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.52, 1.58, 0.82], wspace=0.3, left=0.07, right=0.985, top=0.85,
                          bottom=0.15)

    # ---------------- a: H1 per patient
    ax = fig.add_subplot(gs[0])
    F = _p17_load(os.path.join(R, "main"), "final_primary")
    b17, f17 = _p17_ex(F, "base"), _p17_ex(F, "combo_nosess")
    subs, (bb, ff) = paired(b17, f17)
    for b, f in zip(bb, ff):
        ax.plot([0, 1], [b, f], color="#a9cbc2", lw=0.5, zorder=1)
        ax.scatter([0, 1], [b, f], s=3, color="#a9cbc2", linewidths=0, zorder=2)
    if len(subs):
        ax.plot([0, 1], [bb.mean(), ff.mean()], color=EXT, lw=1.4, zorder=3)
        ax.scatter([0, 1], [bb.mean(), ff.mean()], s=16, color=[BASE, EXT], edgecolors="white", linewidths=0.6, zorder=4)
        ax.text(-0.1, bb.mean(), st.fmt(bb.mean()), ha="right", va="center", fontsize=6.0, color=BASE, fontweight="bold")
        ax.text(1.1, ff.mean(), st.fmt(ff.mean()), ha="left", va="center", fontsize=6.0, color=EXT, fontweight="bold")
    h = rep["H1"]
    ax.set_xticks([0, 1], ["ridge", "final"])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("test r")
    st.range_frame(ax, x=False)
    st.headline(ax, "Every patient gains",
                tag=f"ScaleOLDNEW · H1 {st.fmt(h['mean_diff'], sign=True)}, {h['wins']}/{h['n']}")
    nums["a"] = {"n": len(subs), "mean_base": float(bb.mean()) if len(subs) else None,
                 "mean_final": float(ff.mean()) if len(subs) else None}
    axa = ax

    # ---------------- b: each development finding in both cohorts (labels inside the axis, above each row)
    ax = fig.add_subplot(gs[1])
    rows = [("h1", "final − ridge (H1)", True), ("h2", "learned − fixed, 3 banks (H2)", True),
            ("fixed1", "final − fixed bank", True), ("half", "half calibration − ridge", False),
            ("inter", "target × reference", False)]
    ypos, y = [], 0.0
    for i, (k, lab, rep_) in enumerate(rows):
        if i == 3:
            y -= 0.75                                     # gap between the two groups
        ypos.append(y)
        y -= 1.3
    ds_c, ex_c = INK3, EXT
    allx = [v["ci"][i] for d in (ds, ext) for v in d.values() if v for i in (0, 1)]
    x0 = min(-0.03, min(allx) - 0.01)
    # value columns on the right (x in axes units, y in data); headers double as the legend
    cols = [("ds004752", ds, ds_c, True, INK2), ("ScaleOLDNEW", ext, ex_c, True, EXT), ("subset", add, ex_c, False, EXT)]
    if not any(add.values()):
        cols = cols[:2]
    cx = NUDGE["fig7"].get("cols", [0.465, 0.625, 0.885])
    yax = ax.get_yaxis_transform()
    halo = dict(fc="white", ec="none", pad=0.6)
    for (k, lab, rep_), yi in zip(rows, ypos):
        ax.text(x0, yi + 0.24, lab, fontsize=6.2, color=INK if rep_ else INK2, va="bottom", ha="left", zorder=5,
                bbox=halo)
        for dy, src, col, filled in ((0.0, ds.get(k), ds_c, True), (-0.22, ext.get(k), ex_c, True),
                                     (-0.44, add.get(k), ex_c, False)):
            if not src:
                continue
            lo, hi = src["ci"]
            ax.plot([lo, hi], [yi + dy, yi + dy], color=col, lw=1.1 if filled else 0.7, solid_capstyle="butt", zorder=2)
            ax.scatter([src["mean_diff"]], [yi + dy], s=15 if filled else 10, marker="o" if filled else "D",
                       color=col if filled else "white", edgecolors=col, linewidths=0.8, zorder=3)
        for (cn, D, col, filled, tc), x_ in zip(cols, cx):
            v = (D.get(k) or {}).get("mean_diff")
            if v is not None:
                ax.text(x_ + 0.03, yi + 0.24, st.fmt(v, sign=True), transform=yax, fontsize=6.0, color=tc, va="bottom",
                        ha="left", fontweight="bold" if (rep_ and cn == "ScaleOLDNEW") else "normal")
    ax.axvline(0, color=INK2, lw=0.5, zorder=0)
    yh = ypos[0] + 0.95
    ax.text(x0, yh, "replicated", fontsize=6.4, color=EXT, fontweight="bold", va="bottom", ha="left", zorder=5,
            bbox=halo)
    ax.text(x0, ypos[3] + 0.95, "did not replicate", fontsize=6.4, color=INK2, fontweight="bold", va="bottom",
            ha="left", zorder=5, bbox=halo)
    for (cn, D, col, filled, tc), x_ in zip(cols, cx):
        ax.scatter([x_ + 0.01], [yh + 0.17], s=15 if filled else 10, marker="o" if filled else "D", transform=yax,
                   color=col if filled else "white", edgecolors=col, linewidths=0.8, zorder=4, clip_on=False)
        ax.text(x_ + 0.03, yh, cn, transform=yax, fontsize=5.8, color=tc, va="bottom", ha="left",
                fontweight="bold" if cn == "ScaleOLDNEW" else "normal")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(ypos[-1] - 0.6, ypos[0] + 1.35)
    ax.set_xlim(x0, max(allx) + 0.03)
    ax.set_xlabel("difference in mean test r (95 % CI)")
    st.range_frame(ax, y=False)
    st.headline(ax, "Both pre-specified effects replicate at the same size",
                tag="primary level · strict · ScaleOLDNEW never seen during development")
    axb = ax

    # ---------------- c: common mode shared across modalities vs the target x reference interaction
    ax = fig.add_subplot(gs[2])
    hot_ds, hot17 = [], []
    for s, (x, it) in cm_ds.items():
        ax.scatter([x], [it], s=11, color=INK3, edgecolors="white", linewidths=0.4, zorder=2)
        if x > 0.6:
            hot_ds.append((x, it, _short(s)))
    for s, v in cm17.items():
        x, it = abs(v["r_scalp_cm_vs_ieeg_cm"]), _interaction(v["pred"])
        ax.scatter([x], [it], s=12, color=EXT, edgecolors="white", linewidths=0.4, zorder=3)
        if x > 0.6:
            hot17.append((x, it, _short(s)))
    for x, it, s in hot_ds + hot17:
        ax.scatter([x], [it], s=34, facecolors="none", edgecolors=ACCENT, linewidths=0.7, zorder=4)
    if hot_ds:
        ys = _spread([h[1] for h in hot_ds], 0.03)
        for (x, it, s), yl in zip(hot_ds, ys):
            ax.text(x - 0.07, yl, s, fontsize=5.6, color=ACCENT, ha="right", va="center")
    for x, it, s in hot17:
        ax.text(x - 0.07, it, s, fontsize=5.6, color=ACCENT, ha="right", va="center", fontweight="bold")
    st.ref_line(ax, 0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("|r|, scalp vs SEEG common mode")
    ax.set_ylabel("target × reference interaction")
    ax.text(0.97, 0.24, "ds004752", transform=ax.transAxes, fontsize=5.8, color=INK2, ha="right")
    ax.text(0.97, 0.16, "ScaleOLDNEW", transform=ax.transAxes, fontsize=5.8, color=EXT, ha="right", fontweight="bold")
    st.range_frame(ax)
    st.headline(ax, "Where the interaction comes from", tag="per patient · volts · lagged ridge")
    nums["c"] = {"ds004752": {s: list(v) for s, v in cm_ds.items()},
                 "ScaleOLDNEW": {s: [abs(v["r_scalp_cm_vs_ieeg_cm"]), _interaction(v["pred"])] for s, v in cm17.items()}}
    axc = ax

    st.letters(fig, [(axa, "a", -1.0), (axb, "b", -4.0), (axc, "c", -1.0)])
    st.save(fig, out, "fig7_external")
    dump_numbers(out, "fig7_external", nums)


# ============================================================================= figure S1
def figS1_chanqc(runs, out):
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
    groups = [("kept, listed in authors' file", lambda g: g == "kept, in .set", "#c3beb5", 5, 0.4),
              ("kept, not listed", lambda g: g == "kept, not in .set", OURS, 11, 0.9),
              ("not kept", lambda g: g.startswith("not kept"), ACCENT, 9, 0.9)]
    fig = plt.figure(figsize=(120 * MM, 80 * MM))
    ax = fig.add_axes([0.12, 0.15, 0.85, 0.72])
    nums = {}
    for lab, fn, col, s, a in groups:
        R_ = [r for r in rows if fn(r["group"])]
        x = np.array([num(r, "raw_sd_uV") for r in R_])
        y = np.clip(np.array([num(r, "r_unnamed_max") for r in R_]), 1e-3, 0.9995)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=s, color=col, alpha=a, edgecolors="white" if s > 6 else "none", linewidths=0.4,
                   zorder=3 if s > 6 else 2)
        both = int(((x > 500) & (y >= 0.99)).sum())
        nums[lab] = {"n": int(ok.sum()), "raw_sd_gt500_and_r_ge099": both,
                     "max_raw_sd": float(np.nanmax(x)) if len(x) else None}
    ax.set_xscale("log")
    ax.set_yscale("logit")
    ax.set_ylim(6e-4, 0.9993)
    ax.set_yticks([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999], ["0.001", "0.01", "0.1", "0.5", "0.9", "0.99", "0.999"])
    ax.minorticks_off()
    ax.set_xticks([10, 30, 100, 300, 1000, 3000], ["10", "30", "100", "300", "1000", "3000"])
    for v, axis in ((500, "x"), (0.99, "y")):
        st.ref_line(ax, v, axis=axis, color=INK2, lw=0.6)
    ax.text(500 / 1.06, 0.9975, "500 µV", fontsize=5.6, color=INK2, va="center", ha="right")
    ax.text(ax.get_xlim()[0] * 1.1 if ax.get_xlim()[0] > 0 else 6, 0.99 * 1.0, "|r| 0.99", fontsize=5.6, color=INK2,
            va="bottom", ha="left")
    ax.set_xlabel("raw standard deviation (µV)")
    ax.set_ylabel("max |r| with inputs without a 10-05 name")
    st.range_frame(ax)
    st.headline(ax, "Raw amplitude and similarity to unnamed inputs",
                tag="ScaleOLDNEW patients 01–07 · descriptive · both thresholds post hoc")
    # legend as coloured text in the empty lower right (high amplitude, low similarity)
    for i, (lab, fn, col, s, a) in enumerate(groups):
        ax.text(0.985, 0.30 - i * 0.075, f"{lab}  ({nums[lab]['n']})", transform=ax.transAxes, ha="right", va="center",
                color=col if col != "#c3beb5" else INK2, fontsize=6.0, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.8))
    st.save(fig, out, "figS1_chanqc")
    dump_numbers(out, "figS1_chanqc", nums)


FIGS = {"fig3": fig3_transfer, "fig4": fig4_reliab, "fig5": fig5_power, "fig6": fig6_protocol, "fig7": fig7_external,
        "figS1": figS1_chanqc}
