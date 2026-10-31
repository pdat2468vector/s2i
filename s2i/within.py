"""
Phase 1b (linear) - within-subject variance reduction (CPU only).

Phase 0/1a established that (i) the single-trial EEG->bipolar-iEEG signal is real but
(ii) it does NOT transfer across subjects (zero-shot r ~ 0 even with correct MNI geometry and
shaft orientation), and (iii) within a subject the bottleneck is estimation noise
(1 -> 12 min of calibration: r 0.138 -> 0.210). So the way to beat ridge is to use the few
minutes of calibration more efficiently. Candidates, all fitted on the calibration trials only:

  ridge              lagged ridge, per-contact alpha on ONE chronological 80/20 split (baseline,
                     identical to train_loso.py / ceiling.py)
  ridge_kfold        same model, per-contact alpha by K-fold over contiguous trial blocks
  ridge_kfold_shared one alpha shared by all contacts (K-fold)
  rrr                reduced-rank ridge: contacts share a low-dimensional set of spatio-temporal
                     EEG filters (neighbouring bipolar channels on a shaft are highly redundant);
                     rank chosen by K-fold
  stack_env          ridge_kfold, then a second ridge on filter-bank waveforms + band envelopes
                     fitted to the OUT-OF-FOLD residual (alpha grid reaches "off"), so envelope
                     information is added without diluting the delta/theta fit
  rrr_stack_env      rrr as stage 1, same stage 2

Scored on the same fixed test set as Phase 0/1a (trials after the 20 % split) for several
calibration sizes.

Usage
-----
python -m s2i.within --data DATA_DIR --out RUNS_DIR/within_bip_car
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from s2i.baselines import DEFAULT_LAGS
from s2i.ceiling import ALPHAS, StreamRidge, chrono_split, feat_fbank, feat_lag
from s2i.metrics import per_contact_metrics

FS = 200
LAG9 = feat_lag(DEFAULT_LAGS)
ALPHAS_WIDE = ALPHAS + (1e4, 1e5)
RANKS = (1, 2, 3, 4, 6, 8, 12, 16, 24)


def _blocks(n, k):
    k = max(2, min(k, n))
    e = np.linspace(0, n, k + 1).round().astype(int)
    return [list(range(e[i], e[i + 1])) for i in range(k)]


class _RAcc:
    """Streaming per-target Pearson r for several candidate predictors at once."""

    def __init__(self, n_cand, K):
        self.a = np.zeros((6, n_cand, K))

    def add(self, P, t):                         # P (n_cand, L, K), t (L, K)
        self.a[0] += t.shape[0]
        self.a[1] += P.sum(1)
        self.a[2] += t.sum(0)[None]
        self.a[3] += (P * P).sum(1)
        self.a[4] += (t * t).sum(0)[None]
        self.a[5] += (P * t[None]).sum(1)

    def r(self):
        n, sp, st, spp, stt, spt = self.a
        den = np.sqrt(np.clip(n * spp - sp ** 2, 0, None) * np.clip(n * stt - st ** 2, 0, None))
        return np.where(den > 0, (n * spt - sp * st) / np.maximum(den, 1e-12), -np.inf)


def _nanmean_finite(r, axis):
    return np.nanmean(np.where(np.isfinite(r), r, np.nan), axis)


class KFoldRidge:
    """
    Lagged ridge with hyper-parameters chosen by K-fold CV over contiguous trial blocks.
    Fold models are solved from sufficient statistics (total - fold), so CV costs ~2 passes.
    Optional reduced-rank step (ranks): W <- W V_q V_q^T, V = top eigenvectors of the fitted-value
    covariance W^T G W; q chosen by the same K-fold.
    Keeps out-of-fold predictions (self.oof) for stacking.
    """

    def __init__(self, feat, alphas=ALPHAS, k=5, shared=False, ranks=None, keep_oof=True):
        self.feat, self.alphas, self.k, self.shared = feat, alphas, k, shared
        self.ranks, self.keep_oof = ranks, keep_oof
        self._sr = StreamRidge(feat, alphas)

    def _W(self, s, best):
        W = np.zeros((len(s["mu"]), len(best)))
        for ai, a in enumerate(self.alphas):
            cols = np.where(best == ai)[0]
            if len(cols):
                W[:, cols] = s["Q"] @ (s["QB"][:, cols] / (s["lam"] + a * s["scale"])[:, None])
        return W

    @staticmethod
    def _eigvecs(s, W):
        G = (s["Q"] * s["lam"]) @ s["Q"].T
        M = W.T @ G @ W
        _, V = np.linalg.eigh((M + M.T) / 2)
        return V[:, ::-1]

    def fit(self, E, Y):
        n, K = len(E), Y[0].shape[0]
        folds = _blocks(n, self.k)
        fold_of = np.empty(n, int)
        for fi, idx in enumerate(folds):
            fold_of[idx] = fi
        st = [self._sr._stats([E[i] for i in idx], [Y[i] for i in idx]) for idx in folds]
        total = [sum(x) for x in zip(*st)]
        models = [StreamRidge._solve([a - b for a, b in zip(total, s)]) for s in st]

        acc, oof_all = _RAcc(len(self.alphas), K), []
        for i in range(n):
            s = models[fold_of[i]]
            Z = self._sr._design(s, E[i]) @ s["Q"]
            P = np.stack([Z @ (s["QB"] / (s["lam"] + a * s["scale"])[:, None]) + s["ym"] for a in self.alphas])
            acc.add(P, Y[i].T.astype(np.float64))
            if self.keep_oof and not self.ranks:
                oof_all.append(P.astype(np.float32))
        r = acc.r()
        if self.shared:
            best = np.full(K, int(np.nanargmax(_nanmean_finite(r, 1))))
        else:
            best = np.argmax(r, 0)
        self.best, self.cv_r = best, r[best, np.arange(K)]
        full = StreamRidge._solve(total)
        self.s, self.W = full, self._W(full, best)
        if self.keep_oof and not self.ranks:
            self.oof = [P[best, :, np.arange(K)] for P in oof_all]          # (K, L) each

        self.rank = None
        if self.ranks:
            ranks = sorted({q for q in self.ranks if q < K} | {K})
            W_f = [self._W(s, best) for s in models]
            V_f = [self._eigvecs(s, W) for s, W in zip(models, W_f)]
            acc2, oof = _RAcc(len(ranks), K), []
            for i in range(n):
                f = fold_of[i]
                P0 = self._sr._design(models[f], E[i]) @ W_f[f]
                V = V_f[f]
                P = np.stack([(P0 @ V[:, :q]) @ V[:, :q].T + models[f]["ym"] for q in ranks])
                acc2.add(P, Y[i].T.astype(np.float64))
                if self.keep_oof:
                    oof.append(P.astype(np.float32))
            qi = int(np.nanargmax(_nanmean_finite(acc2.r(), 1)))
            self.rank = ranks[qi]
            V = self._eigvecs(full, self.W)[:, :self.rank]
            self.W = self.W @ V @ V.T
            if self.keep_oof:
                self.oof = [P[qi].T for P in oof]
        return self

    def predict(self, E):
        return [(self._sr._design(self.s, e) @ self.W + self.s["ym"]).T.astype(np.float32) for e in E]


def stacked(stage1, Ec, Yc, Et, feat2=feat_fbank):
    """Stage 2 fitted on the out-of-fold residual of stage 1 (alpha grid reaches 'switched off')."""
    m2 = KFoldRidge(feat2, ALPHAS_WIDE, keep_oof=False).fit(Ec, [y - o for y, o in zip(Yc, stage1.oof)])
    off = float(np.mean(m2.best >= len(ALPHAS_WIDE) - 2))
    return [p + q for p, q in zip(stage1.predict(Et), m2.predict(Et))], off


# ----------------------------------------------------------------------------- per subject
def analyze(E, Y, a, log=print):
    n = len(E)
    _, test = chrono_split(n, 0.2, gap=1)
    Et, Yt = [E[i] for i in test], [Y[i] for i in test]
    res = {"n_trials": n, "n_test": len(test), "fracs": {}}
    for f in a.fracs:
        n_c = max(3, int(round(n * f)))
        if n_c >= test[0]:
            continue
        Ec, Yc = E[:n_c], Y[:n_c]
        bands = abs(f - 0.2) < 1e-9
        out, extra = {}, {}

        def score(name, P):
            m = per_contact_metrics(P, Yt, FS, bands=bands)
            out[name] = {"mean": {k: float(np.nanmean(v)) for k, v in m.items()}, "r": [float(x) for x in m["r"]]}

        score("ridge", StreamRidge(LAG9).fit(Ec, Yc).predict(Et))
        kf = KFoldRidge(LAG9).fit(Ec, Yc)
        score("ridge_kfold", kf.predict(Et))
        score("ridge_kfold_shared", KFoldRidge(LAG9, shared=True, keep_oof=False).fit(Ec, Yc).predict(Et))
        rrr = KFoldRidge(LAG9, ranks=RANKS).fit(Ec, Yc)
        score("rrr", rrr.predict(Et))
        extra["rrr_rank"] = rrr.rank
        if not a.no_env:
            P, extra["stack_off"] = stacked(kf, Ec, Yc, Et)
            score("stack_env", P)
            P, extra["rrr_stack_off"] = stacked(rrr, Ec, Yc, Et)
            score("rrr_stack_env", P)
        mins = sum(e.shape[1] for e in Ec) / FS / 60
        res["fracs"][f"{f:g}"] = {"n_calib": n_c, "minutes": mins, "extra": extra,
                                  "mean": {k: v["mean"]["r"] for k, v in out.items()},
                                  "bands": {k: v["mean"] for k, v in out.items()} if bands else {},
                                  "r": {k: v["r"] for k, v in out.items()}}
        log(f"    f={f:g} ({mins:.1f} min) " + " ".join(f"{k}={v['mean']['r']:+.3f}" for k, v in out.items()) +
            f" | rank={extra['rrr_rank']}" + (f" env_off={extra.get('stack_off', float('nan')):.2f}"))
    return res


def summarize(results, out_dir, log=print):
    from scipy.stats import wilcoxon
    lines, summ = [], {}
    fracs = sorted({f for r in results for f in r["fracs"]}, key=float)
    for f in fracs:
        rows = [r for r in results if f in r["fracs"]]
        models = list(rows[0]["fracs"][f]["mean"])
        M = {m: np.array([r["fracs"][f]["mean"][m] for r in rows]) for m in models}
        mins = np.mean([r["fracs"][f]["minutes"] for r in rows])
        ranks = [r["fracs"][f]["extra"]["rrr_rank"] for r in rows]
        lines += [f"\n### calib fraction {f}  ({len(rows)} subjects, mean {mins:.1f} min, "
                  f"median RRR rank {int(np.median(ranks))})",
                  "| model | mean r | SD | vs ridge | wins | Wilcoxon p |", "|---|---|---|---|---|---|"]
        summ[f] = {}
        for m in models:
            d = M[m] - M["ridge"]
            try:
                p = float(wilcoxon(M[m], M["ridge"]).pvalue) if m != "ridge" else float("nan")
            except ValueError:
                p = float("nan")
            summ[f][m] = {"mean": float(M[m].mean()), "diff": float(d.mean()), "wins": int((d > 0).sum()), "p": p}
            lines.append(f"| {m} | {M[m].mean():.3f} | {M[m].std():.3f} | {d.mean():+.4f} | "
                         f"{(d > 0).sum()}/{len(d)} | {p:.4f} |")
    if "0.2" in fracs:
        rows = [r for r in results if "0.2" in r["fracs"] and r["fracs"]["0.2"]["bands"]]
        models = list(rows[0]["fracs"]["0.2"]["bands"])
        keys = ["r", "r2", "r_delta", "r_theta", "r_alpha", "r_beta", "env_alpha", "env_beta"]
        lines += ["\n### calib fraction 0.2: by band (mean over subjects)",
                  "| model | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for m in models:
            lines.append(f"| {m} | " + " | ".join(
                f"{np.nanmean([r['fracs']['0.2']['bands'][m].get(k, np.nan) for r in rows]):.3f}" for k in keys) + " |")
        models = list(results[0]["fracs"]["0.2"]["mean"])
        lines += ["\n### per subject (calib fraction 0.2)", "| subject | rank | " + " | ".join(models) + " |",
                  "|---" * (len(models) + 2) + "|"]
        for r in results:
            if "0.2" in r["fracs"]:
                fr = r["fracs"]["0.2"]
                lines.append(f"| {r['subject']} | {fr['extra']['rrr_rank']} | " +
                             " | ".join(f"{fr['mean'][m]:.3f}" for m in models) + " |")
    md = "\n".join(lines)
    log(md)
    if out_dir:
        with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(summ, fh, indent=1)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_trials", type=int, default=20)
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.05, 0.1, 0.2])
    ap.add_argument("--no_env", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "within.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    from s2i.ceiling import load_trials
    log(f"config: {vars(a)}")
    results = []
    for p in sorted(glob.glob(os.path.join(a.data, "sub-[0-9][0-9].pt"))):
        sid = os.path.basename(p)[:-3]
        fp = os.path.join(a.out, f"within_{sid}.json")
        if os.path.exists(fp) and not a.overwrite:
            results.append(json.load(open(fp)))
            log(f"[{sid}] already done, loaded")
            continue
        t0 = time.time()
        s, E, Y = load_trials(p, "bip", "car")
        if len(E) < a.min_trials or s.n_targets == 0:
            log(f"[{sid}] skipped")
            continue
        log(f"[{sid}] {len(E)} trials, {len(s.eeg_names)} scalp, {s.n_targets} targets")
        r = analyze(E, Y, a, log)
        r["subject"] = sid
        with open(fp + ".tmp", "w") as fh:
            json.dump(r, fh)
        os.replace(fp + ".tmp", fp)
        results.append(r)
        log(f"[{sid}] done in {time.time() - t0:.0f}s")
    if results:
        summarize(results, a.out, log)


if __name__ == "__main__":
    main()
