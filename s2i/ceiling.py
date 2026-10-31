"""
Phase 0 - information ceiling and confound controls (run BEFORE designing the next model).

Question
--------
How much of the scalp -> bipolar-iEEG relation is genuine single-trial translation, and how
much is (a) chance, (b) stimulus-locked structure shared by both modalities (ERP), or
(c) ocular artefact? And what is the best a linear model can do with the data we have?

ds004752 is a Sternberg task with fixed-structure trials stored from trial onset, so both EEG
and iEEG carry a trial-averaged (stimulus-locked) component. A model can reach r > 0 by
predicting that component without translating anything; "solving" the task means predicting
the single-trial part as well.

Models (fitted per subject on the SAME chronological calibration trials as train_loso.py)
-------------------------------------------------------------------------------------------
  erp_only          trial-averaged iEEG of the calibration trials (uses NO EEG)
  ridge_lag9        lagged ridge, lags +-200 ms  (= the ridge baseline of train_loso.py)
  ridge_lag_long    lagged ridge, lags +-500 ms
  ridge_fbank       filter-bank waveforms (delta..gamma) + Hilbert envelopes, lagged
                    (cheap probe for information a non-linear model could exploit)
  ridge_induced     ERP template removed from BOTH EEG and iEEG (templates from calib only),
                    scored against the induced (single-trial) target
  erp_plus_induced  erp_only + ridge_induced, scored against the raw target
  frontal_only      ridge_lag9 on Fp/AF/F7/F8 channels only   (ocular-artefact probe)
  no_frontal        ridge_lag9 on all other channels
Nulls (applied to the fitted ridge_lag9)
  null_trial_shuffle  EEG of another test trial (half the test set away): keeps the
                      trial-locked structure, destroys single-trial correspondence
  null_circular       test EEG rolled by half its length: chance level
Calibration curve
  fixed test set (last `curve_test_frac` of trials); calibration on the first f of trials.

Usage
-----
python -m s2i.ceiling --data DATA_DIR --out RUNS_DIR/ceiling
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from s2i.baselines import DEFAULT_LAGS, lagged
from s2i.metrics import BANDS, per_contact_metrics

FS = 200
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
LONG_LAGS = tuple(range(-100, 101, 10))           # +-500 ms, 50 ms step
FBANK_LAGS = (-20, -10, 0, 10, 20)                # +-100 ms
FRONTAL = {"FP1", "FP2", "FPZ", "AF3", "AF4", "AF7", "AF8", "AFZ", "F7", "F8"}
_SOS = [butter(4, [lo, min(hi, FS / 2 - 1)], btype="band", fs=FS, output="sos")
        for lo, hi in BANDS.values()]


# ----------------------------------------------------------------------------- features
def feat_lag(lags, chans=None):
    def f(e):
        return lagged(e if chans is None else e[chans], lags)
    return f


def feat_fbank(e):
    parts = []
    for sos in _SOS:
        b = sosfiltfilt(sos, e, axis=1)
        parts += [b, np.abs(hilbert(b, axis=1))]
    return lagged(np.concatenate(parts, 0).astype(np.float32), FBANK_LAGS)


# ----------------------------------------------------------------------------- ridge
class StreamRidge:
    """
    Ridge regression with a per-target alpha chosen on a chronological inner validation
    split (same rule as baselines.LaggedRidge). Fitted from streamed sufficient statistics,
    so wide feature sets (filter bank x lags) never need the full design matrix in memory.
    """

    def __init__(self, feat, alphas=ALPHAS):
        self.feat, self.alphas = feat, alphas

    def _stats(self, E, Y):
        st = None
        for e, y in zip(E, Y):
            X = self.feat(e).astype(np.float64)
            y = y.astype(np.float64)
            if st is None:
                F, K = X.shape[0], y.shape[0]
                st = [0, np.zeros(F), np.zeros((F, F)), np.zeros(K), np.zeros((F, K))]
            st[0] += X.shape[1]
            st[1] += X.sum(1)
            st[2] += X @ X.T
            st[3] += y.sum(1)
            st[4] += X @ y.T
        return st

    @staticmethod
    def _solve(st):
        n, sx, sxx, sy, sxy = st
        mu, ym = sx / n, sy / n
        G = sxx - n * np.outer(mu, mu)
        sd = np.sqrt(np.clip(np.diag(G) / n, 0, None)) + 1e-8
        G = G / np.outer(sd, sd)
        B = (sxy - n * np.outer(mu, ym)) / sd[:, None]
        lam, Q = np.linalg.eigh((G + G.T) / 2)
        lam = np.clip(lam, 0, None)
        return dict(mu=mu, sd=sd, ym=ym, lam=lam, Q=Q, QB=Q.T @ B, scale=max(lam.mean(), 1e-12))

    def _design(self, s, e):
        return (self.feat(e).T.astype(np.float64) - s["mu"]) / s["sd"]

    def fit(self, E, Y, val_frac=0.2):
        n, K = len(E), Y[0].shape[0]
        n_tr = max(1, int(round(n * (1 - val_frac))))
        default = self.alphas.index(10.0) if 10.0 in self.alphas else 0
        if n - n_tr >= 1:
            st_tr = self._stats(E[:n_tr], Y[:n_tr])
            s = self._solve(st_tr)
            A = len(self.alphas)
            acc = np.zeros((6, A, K))                       # n, sp, st, spp, stt, spt
            for e, y in zip(E[n_tr:], Y[n_tr:]):
                Z = self._design(s, e) @ s["Q"]             # (L, F)
                t = y.T.astype(np.float64)
                for ai, a in enumerate(self.alphas):
                    p = Z @ (s["QB"] / (s["lam"] + a * s["scale"])[:, None]) + s["ym"]
                    acc[:, ai] += [np.full(K, len(t)), p.sum(0), t.sum(0), (p * p).sum(0),
                                   (t * t).sum(0), (p * t).sum(0)]
            nn, sp, stt_, spp, stt, spt = acc
            cov = nn * spt - sp * stt_
            den = np.sqrt(np.clip(nn * spp - sp ** 2, 0, None) * np.clip(nn * stt - stt_ ** 2, 0, None))
            r = np.where(den > 0, cov / np.maximum(den, 1e-12), -np.inf)
            best = np.argmax(r, 0)
            st = [x + y for x, y in zip(st_tr, self._stats(E[n_tr:], Y[n_tr:]))]
        else:
            st, best = self._stats(E, Y), np.full(K, default)
        s = self._solve(st)
        W = np.zeros((len(s["mu"]), K))
        for ai, a in enumerate(self.alphas):
            cols = np.where(best == ai)[0]
            if len(cols):
                W[:, cols] = s["Q"] @ (s["QB"][:, cols] / (s["lam"] + a * s["scale"])[:, None])
        self.s, self.W, self.best_alpha = s, W, best
        return self

    def predict(self, E):
        return [(self._design(self.s, e) @ self.W + self.s["ym"]).T.astype(np.float32) for e in E]


# ----------------------------------------------------------------------------- ERP template
def template(trials):
    L = max(t.shape[1] for t in trials)
    acc = np.zeros((trials[0].shape[0], L))
    cnt = np.zeros(L)
    for t in trials:
        acc[:, :t.shape[1]] += t
        cnt[:t.shape[1]] += 1
    return (acc / np.maximum(cnt, 1)).astype(np.float32)


def tmpl_at(T, L):
    if L <= T.shape[1]:
        return T[:, :L]
    return np.concatenate([T, np.zeros((T.shape[0], L - T.shape[1]), np.float32)], 1)


def chrono_split(n_trials, first_frac, gap=1):
    """Identical to s2i.data.dataset.chrono_split (duplicated so this module imports without torch)."""
    n_a = max(1, int(round(n_trials * first_frac)))
    return list(range(0, n_a)), list(range(min(n_trials, n_a + gap), n_trials))


def _mean(m):
    return {k: float(np.nanmean(v)) for k, v in m.items()}


# ----------------------------------------------------------------------------- per subject
def analyze(E, Y, eeg_names, a, log=print):
    """E, Y: lists of (C, L) / (K, L) float32 arrays, one per trial, chronological."""
    n = len(E)
    out = {"n_trials": n, "models": {}, "curve": {}}
    M = out["models"]

    def score(name, P, T, bands=True):
        m = per_contact_metrics(P, T, FS, bands=bands)
        M[name] = {"mean": _mean(m), "r": [float(x) for x in m["r"]]}
        extra = "".join(f" {b}={M[name]['mean'][f'r_{b}']:.3f}" for b in ("delta", "theta", "alpha", "beta")
                        if f"r_{b}" in M[name]["mean"])
        log(f"    {name:20s} r={M[name]['mean']['r']:+.3f}  R2={M[name]['mean']['r2']:+.3f}{extra}")

    cal, test = chrono_split(n, a.calib_frac, gap=1)
    Ec, Yc = [E[i] for i in cal], [Y[i] for i in cal]
    Et, Yt = [E[i] for i in test], [Y[i] for i in test]
    out.update(n_calib=len(cal), n_test=len(test),
               calib_minutes=sum(e.shape[1] for e in Ec) / FS / 60)

    tY, tE = template(Yc), template(Ec)
    erp = [tmpl_at(tY, y.shape[1]) for y in Yt]
    score("erp_only", erp, Yt)

    base = StreamRidge(feat_lag(DEFAULT_LAGS)).fit(Ec, Yc)
    score("ridge_lag9", base.predict(Et), Yt)
    score("ridge_lag_long", StreamRidge(feat_lag(LONG_LAGS)).fit(Ec, Yc).predict(Et), Yt)
    if not a.no_fbank:
        score("ridge_fbank", StreamRidge(feat_fbank).fit(Ec, Yc).predict(Et), Yt)

    res = lambda X, T: [x - tmpl_at(T, x.shape[1]) for x in X]          # noqa: E731
    Pi = StreamRidge(feat_lag(DEFAULT_LAGS)).fit(res(Ec, tE), res(Yc, tY)).predict(res(Et, tE))
    score("ridge_induced", Pi, res(Yt, tY))
    score("erp_plus_induced", [p + q for p, q in zip(erp, Pi)], Yt)

    nT = len(Et)
    if nT > 1:
        sh = nT // 2
        Ps, Ts = [], []
        for i in range(nT):
            j = (i + sh) % nT
            L = min(Et[j].shape[1], Yt[i].shape[1])
            Ps.append(base.predict([Et[j][:, :L]])[0])
            Ts.append(Yt[i][:, :L])
        score("null_trial_shuffle", Ps, Ts, bands=False)
        lens = [e.shape[1] for e in Et]
        rolled = np.roll(np.concatenate(Et, 1), sum(lens) // 2, axis=1)
        score("null_circular", base.predict(np.split(rolled, np.cumsum(lens)[:-1], axis=1)), Yt, bands=False)

    up = [c.upper() for c in eeg_names]
    fr = [i for i, c in enumerate(up) if c in FRONTAL]
    nf = [i for i, c in enumerate(up) if c not in FRONTAL]
    out["frontal_channels"] = [eeg_names[i] for i in fr]
    if fr and nf:
        score("frontal_only", StreamRidge(feat_lag(DEFAULT_LAGS, fr)).fit(Ec, Yc).predict(Et), Yt, bands=False)
        score("no_frontal", StreamRidge(feat_lag(DEFAULT_LAGS, nf)).fit(Ec, Yc).predict(Et), Yt, bands=False)

    # ---- calibration-size curve on a fixed test set
    t0 = int(round(n * (1 - a.curve_test_frac)))
    tc = list(range(min(n, t0 + 1), n))
    if len(tc) >= 2:
        Et2, Yt2 = [E[i] for i in tc], [Y[i] for i in tc]
        for f in a.curve_fracs:
            n_c = int(round(n * f))
            if n_c < 3 or n_c > t0:
                continue
            Ec2, Yc2 = E[:n_c], Y[:n_c]
            tY2, tE2 = template(Yc2), template(Ec2)
            erp2 = [tmpl_at(tY2, y.shape[1]) for y in Yt2]
            Pr = StreamRidge(feat_lag(DEFAULT_LAGS)).fit(Ec2, Yc2).predict(Et2)
            Pi2 = StreamRidge(feat_lag(DEFAULT_LAGS)).fit(res(Ec2, tE2), res(Yc2, tY2)).predict(res(Et2, tE2))
            r = lambda P, T: float(np.nanmean(per_contact_metrics(P, T, FS, bands=False)["r"]))  # noqa: E731
            out["curve"][f"{f:g}"] = {
                "n_calib": n_c, "minutes": sum(e.shape[1] for e in Ec2) / FS / 60,
                "erp_only": r(erp2, Yt2), "ridge_lag9": r(Pr, Yt2),
                "ridge_induced": r(Pi2, res(Yt2, tY2)),
                "erp_plus_induced": r([p + q for p, q in zip(erp2, Pi2)], Yt2)}
        log("    curve: " + " | ".join(f"f={k} ({v['minutes']:.1f}min) ridge={v['ridge_lag9']:.3f} "
                                        f"erp={v['erp_only']:.3f} ind={v['ridge_induced']:.3f}"
                                        for k, v in out["curve"].items()))
    return out


# ----------------------------------------------------------------------------- summary
COMPARISONS = [  # (a, b, question)
    ("ridge_lag9", "null_circular", "ridge above chance?"),
    ("ridge_lag9", "erp_only", "EEG adds beyond the trial-averaged response?"),
    ("ridge_lag9", "null_trial_shuffle", "single-trial correspondence matters?"),
    ("ridge_induced", None, "single-trial (induced) part predictable? (vs 0)"),
    ("erp_plus_induced", "ridge_lag9", "explicit ERP + induced split helps?"),
    ("ridge_lag_long", "ridge_lag9", "longer temporal context helps?"),
    ("ridge_fbank", "ridge_lag9", "band envelopes (non-linear features) help?"),
    ("no_frontal", "frontal_only", "non-frontal channels carry more than frontal (ocular) ones?"),
]


def summarize(results, out_dir, log=print):
    from scipy.stats import wilcoxon
    models = sorted({m for r in results for m in r["models"]})
    order = ["null_circular", "null_trial_shuffle", "erp_only", "frontal_only", "no_frontal", "ridge_lag9",
             "ridge_lag_long", "ridge_fbank", "ridge_induced", "erp_plus_induced"]
    models = [m for m in order if m in models] + [m for m in models if m not in order]
    tab = {r["subject"]: {m: r["models"][m]["mean"]["r"] for m in models if m in r["models"]} for r in results}

    lines = ["| subject | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    for sid, row in tab.items():
        lines.append(f"| {sid} | " + " | ".join(f"{row[m]:.3f}" if m in row else "-" for m in models) + " |")
    means = {m: float(np.nanmean([row[m] for row in tab.values() if m in row])) for m in models}
    sds = {m: float(np.nanstd([row[m] for row in tab.values() if m in row])) for m in models}
    lines.append("| **MEAN** | " + " | ".join(f"**{means[m]:.3f}**" for m in models) + " |")
    lines.append("| SD | " + " | ".join(f"{sds[m]:.3f}" for m in models) + " |")

    tests = []
    for ma, mb, q in COMPARISONS:
        if ma not in models or (mb is not None and mb not in models):
            continue
        sids = [s for s, row in tab.items() if ma in row and (mb is None or mb in row)]
        x = np.array([tab[s][ma] for s in sids])
        y = np.zeros_like(x) if mb is None else np.array([tab[s][mb] for s in sids])
        try:
            p = float(wilcoxon(x, y).pvalue)
        except ValueError:
            p = float("nan")
        tests.append({"a": ma, "b": mb or "0", "question": q, "n": len(sids),
                      "mean_diff": float(np.mean(x - y)), "wins": int((x > y).sum()), "p": p})
    lines += ["", "| question | A | B | mean(A-B) | A wins | Wilcoxon p |", "|---|---|---|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['question']} | {t['a']} | {t['b']} | {t['mean_diff']:+.3f} | "
                     f"{t['wins']}/{t['n']} | {t['p']:.4f} |")

    curve = {}
    for r in results:
        for f, v in r["curve"].items():
            curve.setdefault(f, []).append(v)
    ckeys = ["minutes", "erp_only", "ridge_lag9", "ridge_induced", "erp_plus_induced"]
    lines += ["", "| calib frac | n subj | " + " | ".join(ckeys) + " |", "|---" * (len(ckeys) + 2) + "|"]
    curve_mean = {}
    for f in sorted(curve, key=float):
        curve_mean[f] = {k: float(np.nanmean([v[k] for v in curve[f]])) for k in ckeys}
        curve_mean[f]["n_subjects"] = len(curve[f])
        lines.append(f"| {f} | {len(curve[f])} | " + " | ".join(f"{curve_mean[f][k]:.3f}" for k in ckeys) + " |")

    md = "\n".join(lines)
    log("\n" + md)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"per_subject": tab, "mean": means, "sd": sds, "tests": tests, "curve": curve_mean}, fh, indent=1)
    return means, tests, curve_mean


# ----------------------------------------------------------------------------- main
def load_trials(path, target, eeg_ref):
    from s2i.data.dataset import Subject
    s = Subject(path, target, eeg_ref)
    E, Y = [], []
    for i in range(s.n_trials):
        e, y = s.trial(i)
        E.append(e.float().numpy())
        Y.append(y.float().numpy())
    return s, E, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", choices=["bip", "mono"], default="bip")
    ap.add_argument("--eeg_ref", choices=["car", "orig"], default="car")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--min_trials", type=int, default=20)
    ap.add_argument("--calib_frac", type=float, default=0.2)
    ap.add_argument("--curve_fracs", type=float, nargs="*", default=[0.05, 0.1, 0.2, 0.3, 0.45, 0.59])
    ap.add_argument("--curve_test_frac", type=float, default=0.4)
    ap.add_argument("--no_fbank", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "ceiling.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"config: {vars(a)}")
    paths = sorted(glob.glob(os.path.join(a.data, "sub-[0-9][0-9].pt")))
    if a.subjects:
        paths = [p for p in paths if os.path.basename(p)[:-3] in a.subjects]
    results = []
    for p in paths:
        sid = os.path.basename(p)[:-3]
        fp = os.path.join(a.out, f"ceiling_{sid}.json")
        if os.path.exists(fp) and not a.overwrite:
            results.append(json.load(open(fp)))
            log(f"[{sid}] already done, loaded")
            continue
        t0 = time.time()
        s, E, Y = load_trials(p, a.target, a.eeg_ref)
        if len(E) < a.min_trials or s.n_targets == 0:
            log(f"[{sid}] skipped ({len(E)} trials, {s.n_targets} targets)")
            continue
        log(f"[{sid}] {len(E)} trials, {len(s.eeg_names)} scalp, {s.n_targets} targets")
        r = analyze(E, Y, list(s.eeg_names), a, log)
        r.update(subject=sid, eeg_names=list(s.eeg_names), names=list(s.tgt_names),
                 depth=list(s.tgt_depth), region=list(s.tgt_region))
        with open(fp + ".tmp", "w") as fh:
            json.dump(r, fh)
        os.replace(fp + ".tmp", fp)
        results.append(r)
        log(f"[{sid}] done in {time.time() - t0:.0f}s")
    if results:
        summarize(results, a.out, log)


if __name__ == "__main__":
    main()
