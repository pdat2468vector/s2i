"""
Statistics report for the paper (patient-level inference), computed from the saved result files only.

Every test is paired over PATIENTS (one value per patient and model); contacts are never treated as
independent observations. For each comparison: mean difference, 95 % CI from a patient-level bootstrap
(10 000 resamples), wins, Wilcoxon signed-rank p and Holm-adjusted p within its family.

Families
  main       final model (combo) vs base / single encoder on r, r_no50, r_induced, R^2   (readout_final)
  calib      combo vs base at 1.1 / 2.1 / 4.1 min of calibration                       (readout_final)
  zeroshot   zs_geo vs nearest / task timing / shuffled positions, hybrid vs calibrated ridge, per band
                                                                                  (zeroshot_power)
  depth      combo - base per depth class: per-patient mean over that patient's contacts, then over patients
Longevity: linear mixed model r ~ session gap + (1 | patient) for combo and for combo - base
           (statsmodels), plus per-patient slopes (readout_longevity).

Usage
-----
python -m s2i.stats_report --runs RUNS_DIR [--out RUNS_DIR/stats_report.md]
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

RNG = np.random.default_rng(0)


def boot_ci(d, n=10000):
    d = np.asarray(d, float)
    m = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, running = np.empty_like(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def family(rows):
    """rows: list of (label, a_values, b_values) -> markdown lines with Holm-adjusted p."""
    ps, out = [], []
    for label, a, b in rows:
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        d = a - b
        try:
            p = float(wilcoxon(a, b).pvalue)
        except ValueError:
            p = float("nan")
        lo, hi = boot_ci(d)
        ps.append(p)
        out.append([label, len(d), a.mean(), b.mean(), d.mean(), lo, hi, (d > 0).sum(), p])
    adj = holm(np.nan_to_num(ps, nan=1.0))
    lines = ["| comparison | patients | A | B | A − B | 95 % CI | A > B | p | p (Holm) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r, pa in zip(out, adj):
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:+.4f} | [{r[5]:+.4f}, {r[6]:+.4f}] | "
                     f"{r[7]}/{r[1]} | {r[8]:.4f} | {pa:.4f} |")
    return lines


def load_dir(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--main_run", default="shared_bip_car")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = os.path.join(a.runs, a.main_run)
    L = ["# Statistics report (patient-level)", ""]

    fin = load_dir(os.path.join(run, "readout_final", "readout_sub-*.json"))
    if fin:
        L += [f"## Main comparison, calibration 20 % ({len(fin)} patients, explicit predictions)"]
        rows = []
        for k in ("r", "r_no50", "r_induced", "r2"):
            c = [r["explicit"]["combo"]["mean"][k] for r in fin]
            for ref in ("base", "enc"):
                rows.append((f"combo − {ref}: {k}", c, [r["explicit"][ref]["mean"][k] for r in fin]))
        L += family(rows) + [""]
        L += ["## Calibration size (combo vs base, r from sufficient statistics)"]
        rows = []
        for f in ("0.05", "0.1", "0.2"):
            rr = [r for r in fin if f in r["fracs"]]
            if not rr:
                continue
            mins = np.mean([r["fracs"][f]["minutes"] for r in rr])
            rows.append((f"{mins:.1f} min: combo − base", [r["fracs"][f]["models"]["combo"]["r"] for r in rr],
                         [r["fracs"][f]["models"]["base"]["r"] for r in rr]))
        L += family(rows) + [""]

    zs_path = os.path.join(a.runs, "zeroshot_power", "zeroshot_subjects.json")
    depth_labels = {}
    if os.path.exists(zs_path):
        zs = json.load(open(zs_path))
        depth_labels = {r["subject"]: r["depth"] for r in zs}
        L += [f"## Calibration-free band power ({len(zs)} patients, r_induced)"]
        rows = []
        for b in zs[0]["bands"]:
            g = lambda m: [r["bands"][b][m]["r_induced"] for r in zs]          # noqa: E731
            rows += [(f"{b}: zs_geo − zs_nearest", g("zs_geo"), g("zs_nearest")),
                     (f"{b}: zs_geo − zs_geo_shuf", g("zs_geo"), g("zs_geo_shuf")),
                     (f"{b}: zs_geo − 0", g("zs_geo"), [0.0] * len(zs))]
            if "zs_cal" in zs[0]["bands"][b]:
                rows.append((f"{b}: zs_cal − cal_ridge", g("zs_cal"), g("cal_ridge")))
        L += family(rows) + [""]

    if fin and depth_labels:
        L += ["## By depth class: combo − base (per-patient mean over its contacts, then over patients)",
              "| depth | patients | contacts | mean diff | 95 % CI | patients improved | p |", "|---|---|---|---|---|---|---|"]
        per = {}
        for r in fin:
            sid = r["subject"]
            if sid not in depth_labels:
                continue
            d = np.array(r["explicit"]["combo"]["r"]) - np.array(r["explicit"]["base"]["r"])
            for cls in set(depth_labels[sid]):
                m = np.array([x == cls for x in depth_labels[sid]])
                if m.sum() and len(m) == len(d):
                    per.setdefault(cls, []).append((np.nanmean(d[m]), int(m.sum())))
        for cls, v in sorted(per.items()):
            diffs = np.array([x[0] for x in v])
            lo, hi = boot_ci(diffs) if len(diffs) > 1 else (np.nan, np.nan)
            try:
                p = float(wilcoxon(diffs).pvalue) if len(diffs) >= 5 else float("nan")
            except ValueError:
                p = float("nan")
            L.append(f"| {cls} | {len(v)} | {sum(x[1] for x in v)} | {diffs.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     f"{(diffs > 0).sum()}/{len(v)} | {p:.4f} |")
        L.append("")

    lon = load_dir(os.path.join(run, "readout_longevity", "readout_sub-*.json"))
    if lon:
        rows = []
        for r in lon:
            fr, order = r["fracs"]["0.2"], r["session_order"]
            last = order.index(fr["calib_sessions"][-1])
            for s, v in fr["models"]["combo"].get("sessions", {}).items():
                if r["test_trials_per_session"].get(s, 0) >= 5:
                    rows.append((r["subject"], order.index(s) - last, v,
                                 v - fr["models"]["base"]["sessions"][s]))
        L += [f"## Calibration longevity ({len({x[0] for x in rows})} patients, {len(rows)} patient-sessions)"]
        try:
            import pandas as pd
            import statsmodels.formula.api as smf
            df = pd.DataFrame(rows, columns=["subject", "gap", "r", "adv"])
            for y, lab in (("r", "combo r"), ("adv", "combo − base")):
                fit = smf.mixedlm(f"{y} ~ gap", df, groups=df["subject"]).fit()
                lo, hi = fit.conf_int().loc["gap"]
                L.append(f"- Mixed model {lab} ~ gap + (1|patient): slope {fit.params['gap']:+.4f} per session, "
                         f"95 % CI [{lo:+.4f}, {hi:+.4f}], p = {fit.pvalues['gap']:.4f}")
        except Exception as e:                         # statsmodels missing or singular fit
            L.append(f"- mixed model not available ({type(e).__name__}: {e})")
        L.append("")

    md = "\n".join(L)
    print(md)
    out = a.out or os.path.join(a.runs, "stats_report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print("->", out)


if __name__ == "__main__":
    main()
