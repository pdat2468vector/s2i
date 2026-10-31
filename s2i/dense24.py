"""
Phase 24 - dense-lag ridge baselines on ds004752 (development analysis; settings fixed on 13 Oct 2026 before running).

Question. LAG9 (lags 0, +-25, +-50, +-100, +-200 ms) cannot form narrow-band filters. Does the gain of the filter bank
over LAG9 come from (i) denser lag sampling, (ii) a receptive field as long as the encoder's (+-1.2 s), or (iii) the
frequency structure of the bank, which acts as a low-dimensional prior when only minutes of calibration exist? A ridge
on every lag within +-1.2 s spans every linear filter of that support, including the 28 kernels of the fixed no-env
bank, so it is the representation-matched linear control.

Models (per patient; primary level = data/s2i_v1_noqc, strict normalisation, 20 % chronological calibration, one-trial
gap, the same 5 contiguous calibration folds, ridge penalties by pooled out-of-fold r, r scored on the test trials,
exactly as s2i.readout; all quantities from second-order sufficient statistics):
  base            LAG9 ridge, penalty grid ALPHAS (reproduces the registered base)
  dense200        ridge on every lag in -200..+200 ms (81 lags), grid DENSE_ALPHAS
  dense1200       ridge on every lag in -1.2..+1.2 s (481 lags, the encoder's support), grid DENSE_ALPHAS
  fixed1          fixed untrained no-env bank (28 features per channel), grid ALPHAS_WIDE (reproduces registered 'enc')
  <model>+lag9    each of dense200 / dense1200 / fixed1 mixed with LAG9 ridge, weight w in {0, .25, .5, .75, 1} by
                  pooled out-of-fold r (as the readout's mixtures; fixed1+lag9 reproduces the registered fixed bank)
Penalty grid of the dense ridges: 10^-3 ... 10^7 in half decades (wider than the other models because of the many
collinear lags); a choice at either end of the grid is flagged in the output.

Usage
  python -m s2i.dense24 run    --data DATA/s2i_v1_noqc --noenv RUNS/shared_no_env --out RUNS/dense24
  python -m s2i.dense24 report --out RUNS/dense24 --runs RUNS
  python -m s2i.dense24 smoke  --out /tmp/dense24_smoke
"""
import argparse
import glob
import json
import os
import sys
import time
import zlib

import numpy as np
import torch

from s2i import readout as ro
from s2i.baselines import DEFAULT_LAGS, lagged
from s2i.ceiling import ALPHAS, chrono_split
from s2i.shared import TemporalEncoder, np_feat
from s2i.within import ALPHAS_WIDE, _blocks

FRAC = 0.2
DENSE_ALPHAS = tuple(10.0 ** e for e in np.arange(-3, 7.01, 0.5))
L200, L1200 = tuple(range(-40, 41)), tuple(range(-240, 241))


def log(msg):
    print(time.strftime("%H:%M:%S"), "|", msg, flush=True)


def rows_of(lags_all, lags_sub, C):
    """Feature rows of a lag subset inside a lagged() matrix ordered lag-major (row = j * C + c)."""
    pos = {l: j for j, l in enumerate(lags_all)}
    return np.concatenate([np.arange(pos[l] * C, (pos[l] + 1) * C) for l in lags_sub])


def ridge_one(tr, alpha):
    """Ridge at one penalty, identical to s2i.readout.ridge_models (penalty alpha * mean eigenvalue)."""
    from scipy.linalg import solve
    mu, sd, Gs, Bs, ym = ro.standardise(tr)
    sc = max(np.trace(Gs) / len(Gs), 1e-12)
    Ws = solve(Gs + alpha * sc * np.eye(len(Gs)), Bs, assume_a="pos")
    return ro.to_raw(Ws, mu, sd, ym)


class Comp:
    """One ridge component on rows `rows` of a statistics family: penalty by pooled out-of-fold r (two passes, so
    that no per-penalty fold model of a large block is kept in memory)."""

    def __init__(self, fold_st, tot, rows, alphas):
        self.rows, self.alphas = rows, alphas
        full = len(rows) == len(tot.sx) and np.array_equal(rows, np.arange(len(rows)))
        take = (lambda s: s) if full else (lambda s: s.take(rows))          # noqa: E731
        tot_r = take(tot)
        oof = None
        for fs in fold_st:
            fs_r = take(fs)
            tr = tot_r - fs_r
            s = np.stack([ro.pred_sums(A, c, fs_r) for A, c in ro.ridge_models(tr, alphas)])
            oof = s if oof is None else oof + s
            del tr, fs_r
        self.score = [np.nanmean(ro.r_of(o)) for o in oof]
        self.best = int(np.nanargmax(self.score))
        self.oof = oof[self.best]
        self.alpha = alphas[self.best]
        self.edge = self.best in (0, len(alphas) - 1)
        self.fold_models = [ridge_one(tot_r - take(fs), self.alpha) for fs in fold_st]
        self.A, self.c = ridge_one(tot_r, self.alpha)


def xscore(A, c, oof, tot, te):
    """s2i.readout.score plus the r of the EXPLICIT predictions of the main analysis: the per-contact amplitude gain
    (least squares on the pooled out-of-fold predictions, clipped at 0) leaves r unchanged where it is positive and
    makes the prediction constant where it is 0, and a constant prediction scores r = 0 (s2i.metrics)."""
    s = ro.score(A, c, oof, tot, te)
    n, sp, spp, spy, sy, _ = oof
    ym = tot.sy / tot.n
    g = np.clip((spy - ym * sp - ym * sy + n * ym ** 2) / np.maximum(spp - 2 * ym * sp + n * ym ** 2, 1e-12), 0, None)
    rc = np.asarray(s["contacts"], float)
    s["contacts_explicit"] = np.where(g > 0, rc, 0.0).tolist()
    s["r_explicit"] = float(np.nanmean(s["contacts_explicit"]))
    s["n_zero_gain"] = int((g <= 0).sum())
    return s


def mixture(comp, lag, fold_st, tot, te, sel):
    """comp and LAG9 (rows inside `sel`) mixed with w in ENS_W by pooled out-of-fold r (as s2i.readout.mixture)."""
    K = len(tot.sy)
    loc = {r: i for i, r in enumerate(sel)}
    li, lj = [loc[r] for r in comp.rows], [loc[r] for r in lag.rows]

    def combine(w, m1, m2):
        A, c = np.zeros((len(sel), K)), np.zeros(K)
        A[li] += w * m1[0]
        A[lj] += (1 - w) * m2[0]
        return A, w * m1[1] + (1 - w) * m2[1]
    full = len(sel) == len(tot.sx) and np.array_equal(sel, np.arange(len(sel)))
    take = (lambda s: s) if full else (lambda s: s.take(sel))              # noqa: E731
    oofs = [0.0] * len(ro.ENS_W)
    for f, fs in enumerate(fold_st):
        fs_s = take(fs)
        for wi, w in enumerate(ro.ENS_W):
            oofs[wi] = oofs[wi] + ro.pred_sums(*combine(w, comp.fold_models[f], lag.fold_models[f]), fs_s)
        del fs_s
    wi = int(np.nanargmax([np.nanmean(ro.r_of(o)) for o in oofs]))
    A, c = combine(ro.ENS_W[wi], (comp.A, comp.c), (lag.A, lag.c))
    return xscore(A, c, oofs[wi], take(tot), take(te)), ro.ENS_W[wi]


def stats(E, Y, feat, D, fold_ids, test):
    K = Y[0].shape[0]
    fold_st = []
    for ids in fold_ids:
        s = ro.St(D, K)
        for i in ids:
            s.add(feat(E[i]), Y[i])
        fold_st.append(s)
    te = ro.St(D, K)
    for i in test:
        te.add(feat(E[i]), Y[i])
    return fold_st, ro.total(fold_st), te


def analyze(E, Y, enc0, dev, skip_1200=False):
    n, C = len(E), E[0].shape[0]
    _, test = chrono_split(n, FRAC, gap=1)
    n_c = max(3, int(round(n * FRAC)))
    fold_ids = _blocks(n_c, 5)
    out, info = {}, {"n_trials": n, "n_test": len(test), "n_calib": n_c, "n_scalp": C,
                     "minutes": sum(E[i].shape[1] for i in range(n_c)) / ro.FS / 60}

    def rec(name, sc, comp=None, w=None):
        out[name] = {"r": sc["r"], "r_explicit": sc["r_explicit"], "n_zero_gain": sc["n_zero_gain"],
                     "r2": sc["r2"], "contacts": sc["contacts"], "contacts_explicit": sc["contacts_explicit"]}
        if comp is not None:
            out[name].update(alpha=comp.alpha, alpha_at_grid_edge=comp.edge)
        if w is not None:
            out[name]["w_component"] = w

    # family A: the dense lags (LAG9 and the +-200 ms lags are row subsets)
    lags_all = L200 if skip_1200 else L1200
    DA = C * len(lags_all)
    t0 = time.time()
    fs_a, tot_a, te_a = stats(E, Y, lambda e: lagged(e, lags_all), DA, fold_ids, test)
    r9, r200 = rows_of(lags_all, DEFAULT_LAGS, C), rows_of(lags_all, L200, C)
    base = Comp(fs_a, tot_a, r9, ALPHAS)
    rec("base", xscore(base.A, base.c, base.oof, tot_a.take(r9), te_a.take(r9)), base)
    for name, rows in (("dense200", r200),) + ((("dense1200", np.arange(DA)),) if not skip_1200 else ()):
        m = Comp(fs_a, tot_a, rows, DENSE_ALPHAS)
        rec(name, xscore(m.A, m.c, m.oof, tot_a.take(rows), te_a.take(rows)), m)
        sc, w = mixture(m, base, fs_a, tot_a, te_a, rows)          # LAG9 rows lie inside `rows`
        rec(name + "+lag9", sc, w=w)
        del m
    info["seconds_dense"] = time.time() - t0
    del fs_a, tot_a, te_a

    # family B: [fixed no-env bank | LAG9]
    f0 = np_feat(enc0, dev, dev.type == "cuda")
    F = enc0.n_feat
    DB = C * F + C * len(DEFAULT_LAGS)
    fs_b, tot_b, te_b = stats(E, Y, lambda e: np.concatenate([f0(e), lagged(e, DEFAULT_LAGS)]).astype(np.float32),
                              DB, fold_ids, test)
    rb, r9b = np.arange(C * F), np.arange(C * F, DB)
    fx = Comp(fs_b, tot_b, rb, ALPHAS_WIDE)
    rec("fixed1", xscore(fx.A, fx.c, fx.oof, tot_b.take(rb), te_b.take(rb)), fx)
    lag_b = Comp(fs_b, tot_b, r9b, ALPHAS)
    sc, w = mixture(fx, lag_b, fs_b, tot_b, te_b, np.arange(DB))
    rec("fixed1+lag9", sc, w=w)
    return {"models": out, **info}


def run(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ro._DEV = dev                                       # Gram matrices on the GPU when available
    os.makedirs(a.out, exist_ok=True)
    for dp in sorted(glob.glob(os.path.join(a.data, "sub-*.pt"))):
        sid = os.path.basename(dp)[:-3]
        if sid.endswith("_normstats") or (a.subjects and sid not in a.subjects):
            continue
        fp = os.path.join(a.out, f"dense24_{sid}.json")
        if os.path.exists(fp) and not a.overwrite:
            log(f"[{sid}] already done")
            continue
        t0 = time.time()
        E, Y, ses, _ = ro.load_subject(dp)
        st = torch.load(os.path.join(a.data, f"{sid}_normstats.pt"), weights_only=False)
        assert len(st["trial_bounds"]) == len(E) and list(st["trial_session"]) == ses, "stats / data mismatch"
        E, Y = ro.strict_renorm(E, Y, ses, st, max(3, int(round(len(E) * FRAC))))
        cfg = torch.load(os.path.join(a.noenv, f"enc_{sid}.pt"), map_location="cpu", weights_only=False)["cfg"]
        enc0 = TemporalEncoder(**cfg).to(dev).eval()                         # fixed untrained no-env bank
        res = analyze(E, Y, enc0, dev, a.skip_1200)
        res["subject"] = sid
        json.dump(res, open(fp, "w"))
        log(f"[{sid}] {res['n_scalp']} scalp, {res['minutes']:.1f} min | " +
            " ".join(f"{k}={v['r_explicit']:.3f}" for k, v in res["models"].items()) + f" | {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- diagnosis of one patient
def diagnose(a):
    """
    Why can this script and s2i.readout give a different base / fixed-bank r for one patient? Recomputes both, in
    the readout's feature view [fixed bank | LAG9] and in this script's dense-lag view, with the Gram matrices
    accumulated in float64 and in float32 (on the GPU when available, as in the registered runs), and prints the
    pooled out-of-fold r and the test r of every candidate penalty with the one selected.
    """
    sid = a.subjects[0]
    dp = os.path.join(a.data, f"{sid}.pt")
    E, Y, ses, _ = ro.load_subject(dp)
    st = torch.load(os.path.join(a.data, f"{sid}_normstats.pt"), weights_only=False)
    E, Y = ro.strict_renorm(E, Y, ses, st, max(3, int(round(len(E) * FRAC))))
    cfg = torch.load(os.path.join(a.noenv, f"enc_{sid}.pt"), map_location="cpu", weights_only=False)["cfg"]
    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    n, C = len(E), E[0].shape[0]
    _, test = chrono_split(n, FRAC, gap=1)
    n_c = max(3, int(round(n * FRAC)))
    fold_ids = _blocks(n_c, 5)
    print(f"{sid}: {n} trials, {C} scalp channels, {n_c} calibration trials in {len(fold_ids)} folds, "
          f"{len(test)} test trials")
    out = {}
    for prec in ("float64", "float32"):
        ro._DEV = gpu if prec == "float32" else None
        dev = gpu or torch.device("cpu")
        f0 = np_feat(TemporalEncoder(**cfg).to(dev).eval(), dev, dev.type == "cuda")
        F = TemporalEncoder(**cfg).n_feat
        views = {"readout view [fixed | LAG9]": (lambda e: np.concatenate([f0(e), lagged(e, DEFAULT_LAGS)]),
                                                C * F + 9 * C, {"base": (np.arange(C * F, C * F + 9 * C), ALPHAS),
                                                                "fixed1": (np.arange(C * F), ALPHAS_WIDE)}),
                 "dense view [lags within +-200 ms]": (lambda e: lagged(e, L200), 81 * C,
                                                       {"base": (rows_of(L200, DEFAULT_LAGS, C), ALPHAS)})}
        for vname, (feat, D, models) in views.items():
            cast = (lambda e: feat(e).astype(np.float64)) if prec == "float64" else \
                   (lambda e: feat(e).astype(np.float32))
            fs, tot, te = stats(E, Y, cast, D, fold_ids, test)
            for mname, (rows, alphas) in models.items():
                comp = Comp(fs, tot, rows, alphas)
                tr = [float(np.nanmean(ro.r_of(ro.pred_sums(A, c, te.take(rows)))))
                      for A, c in ro.ridge_models(tot.take(rows), alphas)]
                key = f"{mname} | {vname} | {prec}{' (GPU)' if prec == 'float32' and gpu else ''}"
                out[key] = {"alpha": comp.alpha, "oof_r": comp.score, "test_r": tr}
                print(f"\n{key}: selected penalty {comp.alpha:g}, test r {tr[comp.best]:.4f}")
                for al, o, r_ in zip(alphas, comp.score, tr):
                    print(f"    penalty {al:8.0e}  out-of-fold r {o:.6f}  test r {r_:.4f}"
                          + ("   <- selected" if al == comp.alpha else ""))
    ro._DEV = None
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"dense24_diagnose_{sid}.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- report
def boot_ci(v, name, n=10000):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def reg_r(path, model):
    """Registered per-patient r (explicit-prediction r when stored for 20 %, as in the manuscript)."""
    out = {}
    for f in glob.glob(os.path.join(path, "readout_sub-*.json")):
        d = json.load(open(f))
        ex = d.get("explicit", {}).get(model)
        if ex is not None and abs(ex.get("frac", 0.2) - FRAC) < 1e-9:
            out[d["subject"]] = ex["mean"]["r"]
        elif model in d.get("fracs", {}).get("0.2", {}).get("models", {}):
            out[d["subject"]] = d["fracs"]["0.2"]["models"][model]["r"]
    return out


def report(a):
    from scipy.stats import wilcoxon
    R = {}
    for f in sorted(glob.glob(os.path.join(a.out, "dense24_sub-*.json"))):
        d = json.load(open(f))
        R[d["subject"]] = d
    subs = sorted(R)
    key = "r_explicit" if a.metric == "explicit" else "r"
    if key == "r_explicit" and not all("r_explicit" in m for s in subs for m in R[s]["models"].values()):
        sys.exit("these results have no r_explicit: rerun 'run --overwrite' with this version, or use --metric stats")
    M = {k: np.array([R[s]["models"][k][key] for s in subs]) for k in R[subs[0]]["models"]}
    if a.cohort == "scaleoldnew":                       # --runs = RUNS17 (protocol v1.6 run root), Addendum C
        fin, fix = os.path.join(a.runs, "main", "final_primary"), os.path.join(a.runs, "no_env", "fixed1_primary")
    else:                                               # --runs = RUNS (ds004752 run root)
        fin = os.path.join(a.runs, "shared_bip_car", "readout_combo5_strict_qc_none")
        fix = os.path.join(a.runs, "shared_no_env", "readout_fixed_strict_qc_none")
    reg = {"final (registered)": reg_r(fin, "combo_nosess"), "base (registered)": reg_r(fin, "base"),
           "fixed1+lag9 (registered)": reg_r(fix, "combo_nosess"), "fixed1 (registered)": reg_r(fix, "enc")}
    for k, v in reg.items():
        if all(s in v for s in subs):
            M[k] = np.array([v[s] for s in subs])
    title = ("Addendum C (post hoc): dense-lag ridge on ScaleOLDNEW" if a.cohort == "scaleoldnew"
             else "Phase 24 - dense-lag ridge baselines (ds004752")
    L = [f"# {title}, primary level, strict, 20 % calibration)", "",
         f"{len(subs)} patients. Mean test r over patients (r per patient averaged over contacts). Scoring: "
         + ("explicit-prediction convention of the main analysis (contacts whose amplitude gain is clipped to 0 "
            "score r = 0)" if key == "r_explicit" else "sufficient-statistics r before amplitude calibration")
         + ".", "",
         "| model | mean r | penalty at the LOW / HIGH end of its grid (patients) | chosen penalties | "
         "mean LAG9 weight (1 - w) |", "|---|---|---|---|---|"]
    GRID = {"base": ALPHAS, "fixed1": ALPHAS_WIDE, "dense200": DENSE_ALPHAS, "dense1200": DENSE_ALPHAS}
    for k, v in M.items():
        lohi, al = "", ""
        if k in GRID:
            a_ = [R[s]["models"][k]["alpha"] for s in subs]
            g = GRID[k]
            lohi = f"{sum(np.isclose(x, g[0]) for x in a_)} / {sum(np.isclose(x, g[-1]) for x in a_)}"
            al = " ".join(f"{x:.0e}" for x in a_)
        w = ([R[s]["models"][k].get("w_component") for s in subs] if k in R[subs[0]]["models"] else [])
        wl = f"{1 - np.mean(w):.2f}" if w and w[0] is not None else ""
        L.append(f"| {k} | {v.mean():.4f} | {lohi} | {al} | {wl} |")
    L += ["", "## Sanity checks (this script vs the registered readouts; should agree to about 0.001)", ""]
    for mine, theirs in (("base", "base (registered)"), ("fixed1", "fixed1 (registered)"),
                         ("fixed1+lag9", "fixed1+lag9 (registered)")):
        if theirs in M:
            dd = np.abs(M[mine] - M[theirs])
            off = [f"{s} {M[mine][i]:.4f} vs {M[theirs][i]:.4f}" for i, s in enumerate(subs) if dd[i] > 0.001]
            L.append(f"- {mine}: mean {M[mine].mean():.4f} vs {M[theirs].mean():.4f}; largest per-patient difference "
                     f"{dd.max():.4f}; patients differing by > 0.001: {', '.join(off) or 'none'}")
    comps = [("dense200", "base", "denser lag sampling within +-200 ms"),
             ("dense1200", "base", "receptive field of +-1.2 s, unstructured"),
             ("dense1200", "dense200", "longer support, unstructured"),
             ("dense200+lag9", "base", "dense +-200 ms (+ LAG9) vs LAG9"),
             ("dense1200+lag9", "base", "dense +-1.2 s (+ LAG9) vs LAG9 (Fig. 3a step)"),
             ("fixed1", "dense1200", "frequency structure vs matched-support linear FIR (alone)"),
             ("fixed1+lag9", "dense1200+lag9", "frequency structure vs matched-support linear FIR (+ LAG9)"),
             ("fixed1+lag9", "dense200+lag9", "fixed bank vs dense +-200 ms (+ LAG9)"),
             ("final (registered)", "dense1200+lag9", "final model vs the best unstructured linear control"),
             ("final (registered)", "dense200+lag9", "final model vs dense +-200 ms (+ LAG9)")]
    cols = [k for k in ("base", "dense200", "dense200+lag9", "dense1200", "dense1200+lag9", "fixed1", "fixed1+lag9",
                        "final (registered)") if k in M]
    L += ["", "## Per patient (same scoring)", "", "| patient | " + " | ".join(cols) + " |",
          "|" + "---|" * (len(cols) + 1)]
    for i, s in enumerate(subs):
        L.append(f"| {s} | " + " | ".join(f"{M[k][i]:.3f}" for k in cols) + " |")
    L += ["", "## Paired comparisons (patient bootstrap 95 % CI seeded by the comparison name; two-sided Wilcoxon)", "",
          "| A | B | question | mean A − B [95 % CI] | A > B | p |", "|---|---|---|---|---|---|"]
    out = {"n": len(subs), "means": {k: float(v.mean()) for k, v in M.items()}, "comparisons": {}}
    for x, y, q in comps:
        if x not in M or y not in M:
            continue
        d = M[x] - M[y]
        lo, hi = boot_ci(d, f"dense24 {x} - {y}")
        try:
            p = float(wilcoxon(M[x], M[y]).pvalue)
        except ValueError:
            p = float("nan")
        out["comparisons"][f"{x} - {y}"] = {"mean_diff": float(d.mean()), "ci": [lo, hi], "wins": int((d > 0).sum()),
                                            "n": len(d), "p": p, "per_patient": dict(zip(subs, d.round(5).tolist()))}
        L.append(f"| {x} | {y} | {q} | {d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}] | {(d > 0).sum()}/{len(d)} | {p:.4f} |")
    md = "\n".join(L)
    print(md)
    open(os.path.join(a.out, "dense24_report.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(os.path.join(a.out, "dense24_report.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- smoke test
def smoke(a):
    """Synthetic patient: targets = 10-Hz band of scalp channels + noise. The narrow-band relation is invisible to
    LAG9 but available to the dense lags and the bank; checks that everything runs and orders as expected."""
    from scipy.signal import butter, sosfiltfilt
    rng = np.random.default_rng(0)
    sos = butter(4, (8, 12), btype="band", fs=200, output="sos")
    E, Y = [], []
    for _ in range(40):
        e = rng.normal(size=(4, 800)).astype(np.float32)
        y = np.stack([sosfiltfilt(sos, e[i]) * 3 for i in (0, 1, 2)]) + rng.normal(size=(3, 800))
        E.append(e)
        Y.append(y.astype(np.float32))
    dev = torch.device("cpu")
    enc0 = TemporalEncoder(n_pairs=12, half=240, n_free=4, envelope=False).eval()
    res = analyze(E, Y, enc0, dev)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "dense24_smoke.json"), "w"))
    m = {k: v["r"] for k, v in res["models"].items()}
    print(" ".join(f"{k}={v:.3f}" for k, v in m.items()))
    assert m["dense200"] > m["base"] + 0.05 and m["fixed1"] > m["base"] + 0.05, m
    assert m["dense200+lag9"] >= m["base"] - 1e-6, m
    print("smoke test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report", "smoke", "diagnose"))
    ap.add_argument("--data")
    ap.add_argument("--noenv", help="run folder of the no-env encoders (cfg of the fixed bank)")
    ap.add_argument("--runs", help="RUNS root (registered readouts for the sanity checks)")
    ap.add_argument("--out")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_1200", action="store_true", help="only the +-200 ms dense ridge (fast)")
    ap.add_argument("--metric", choices=("explicit", "stats"), default="explicit",
                    help="report: r of the explicit predictions (as the registered values) or the statistics r")
    ap.add_argument("--cohort", choices=("ds004752", "scaleoldnew"), default="ds004752",
                    help="report: where the registered readouts are (scaleoldnew: --runs = RUNS17)")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "report": report, "smoke": smoke, "diagnose": diagnose}[a.cmd](a)


if __name__ == "__main__":
    main()
