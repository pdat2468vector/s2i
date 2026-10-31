"""
Phase 1c (CPU) - can ANY non-linear EEG feature beat a well-regularised linear model with a
few minutes of calibration?

Phase 1b: the only linear lever that helped was better regularisation (ridge with one alpha
shared by all contacts, chosen by K-fold: +0.008 at 1-2 min, 15/15 subjects). Envelope
features raised alpha-band r by 17 % but the residual stage was selected by Pearson r, which
does not penalise amplitude, so it added noise (R^2 -> -0.16) and hurt small subjects.

Here the second stage is fitted on the OUT-OF-FOLD residual of the linear model and is
  * selected by MSE (not r), per contact, and
  * GATED: switched on for a contact only if its cross-validated MSE beats predicting the
    residual mean.
so it can only add what cross-validation says is real.

Models
------
  ridge         baseline of train_loso.py / ceiling.py (per-contact alpha, one 80/20 split)
  base          ridge_kfold_shared from Phase 1b (new strong baseline)
  env_gated     base + gated stage 2 on alpha/beta band waveforms + theta/alpha/beta envelopes
  wav_gated     base + gated stage 2 on per-channel Morlet envelopes, 12 log-spaced bands 2-64 Hz
                (exhaustive per-channel spectral power, incl. gamma)
  rocket_gated  base + gated stage 2 on random non-linear convolutional features
                (random spatial filter -> random Morlet wavelet -> envelope or ReLU).
                A cheap stand-in for "a non-linear encoder", with no training.
  all_gated     base + gated stage 2 on [wavelet envelopes, random features]

Usage
-----
python -m s2i.nonlin --data DATA_DIR --out RUNS_DIR/nonlin_bip_car
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import butter, hilbert, sosfiltfilt

from s2i.baselines import lagged
from s2i.ceiling import StreamRidge, chrono_split
from s2i.metrics import per_contact_metrics
from s2i.within import ALPHAS_WIDE, LAG9, KFoldRidge, _blocks

FS = 200
ENV_LAGS = (-20, -10, 0, 10, 20)
_B = {k: butter(4, f, btype="band", fs=FS, output="sos")
      for k, f in {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}.items()}


# ----------------------------------------------------------------------------- features
def feat_env(e):
    a = sosfiltfilt(_B["alpha"], e, axis=1)
    b = sosfiltfilt(_B["beta"], e, axis=1)
    t = sosfiltfilt(_B["theta"], e, axis=1)
    parts = [a, b, np.abs(hilbert(t, axis=1)), np.abs(hilbert(a, axis=1)), np.abs(hilbert(b, axis=1))]
    return lagged(np.concatenate(parts, 0).astype(np.float32), ENV_LAGS)


def morlet(freq, cycles):
    """Complex Morlet kernels (len(freq), 2H+1), unit L1 norm, centred at index H."""
    sig = np.asarray(cycles) / (2 * np.pi * np.asarray(freq))        # Gaussian width, s
    H = int(np.ceil(3 * sig.max() * FS))
    t = np.arange(-H, H + 1) / FS
    w = np.exp(-t[None] ** 2 / (2 * sig[:, None] ** 2)) * np.exp(2j * np.pi * np.asarray(freq)[:, None] * t[None])
    return (w / np.abs(w).sum(1, keepdims=True)).astype(np.complex64), H


def wconv(S, kern, H):
    """Zero-phase convolution of each row of S (M, L) with its own kernel (M, 2H+1) -> complex (M, L)."""
    L = S.shape[1]
    n = next_fast_len(L + 2 * H)
    return np.fft.ifft(np.fft.fft(S, n) * np.fft.fft(kern, n), n)[:, H:H + L]


class MorletEnvBank:
    """Per-channel Morlet envelope at n_freq log-spaced frequencies (all channels x all bands), lagged."""

    def __init__(self, freqs=tuple(np.geomspace(2, 64, 12)), cycles=5, lags=(-20, 0, 20)):
        self.kern1, self.H = morlet(freqs, np.full(len(freqs), cycles))
        self.nf, self.lags = len(freqs), lags

    def __call__(self, e):
        C = e.shape[0]
        S = np.repeat(e, self.nf, 0)
        env = np.abs(wconv(S, np.tile(self.kern1, (C, 1)), self.H))
        return lagged(env.astype(np.float32), self.lags)


class RandomConvFeatures:
    """
    M random (spatial filter -> complex Morlet wavelet -> non-linearity) features, fixed seed.
    A cheap stand-in for the first layer of a convolutional encoder, with no training:
      spatial filter : 1/3 single channel, 1/3 channel-pair difference, 1/3 dense random
      wavelet        : centre frequency log-uniform in [fmin, fmax] Hz, 3-7 cycles
      non-linearity  : half envelope |w * s|, half ReLU(Re(w * s) - b * rms), b ~ U(-1, 1)
    mode (same random filters/wavelets whatever the mode, so modes form a matched ablation):
      "mix" (above) | "env" all envelopes | "relu" all ReLU | "linear" Re(w * s), no non-linearity
    Synthetic check: dense spatial filters + short random 9-tap kernels (plain ROCKET) missed an
    alpha-envelope signal that feat_env recovered - broadband kernels dilute narrow-band power.
    """

    def __init__(self, n_chan, M=512, fmin=2.0, fmax=60.0, seed=0, mode="mix"):
        assert mode in ("mix", "env", "relu", "linear")
        self.mode = mode
        g = np.random.default_rng(seed)
        self.Ws = np.zeros((M, n_chan), np.float32)
        kind = np.arange(M) % 3
        for i in range(M):
            if kind[i] == 0 or n_chan < 2:
                self.Ws[i, g.integers(n_chan)] = 1
            elif kind[i] == 1:
                c = g.choice(n_chan, 2, replace=False)
                self.Ws[i, c] = (1, -1)
            else:
                self.Ws[i] = g.normal(size=n_chan)
        self.Ws /= np.linalg.norm(self.Ws, axis=1, keepdims=True)
        self.freq = np.exp(g.uniform(np.log(fmin), np.log(fmax), M))
        self.kern, self.H = morlet(self.freq, g.uniform(3, 7, M))
        self.env = g.random(M) < 0.5
        self.bias = g.uniform(-1, 1, M).astype(np.float32)

    def __call__(self, e):
        C = wconv(self.Ws @ e, self.kern, self.H)
        R = C.real
        if self.mode == "linear":
            return R.astype(np.float32)
        if self.mode == "env":
            return np.abs(C).astype(np.float32)
        out = np.maximum(R - self.bias[:, None] * np.sqrt((R * R).mean(1, keepdims=True)), 0)
        if self.mode == "mix":
            out[self.env] = np.abs(C[self.env])
        return out.astype(np.float32)


def concat_feats(*fs):
    return lambda e: np.concatenate([f(e) for f in fs], 0)


# ----------------------------------------------------------------------------- gated stage 2
class GatedResidualRidge:
    """
    Ridge on a residual target with per-contact alpha chosen by K-fold MSE, and a per-contact
    gate: ON only if the best CV MSE beats predicting the training-fold mean by `margin`.
    """

    def __init__(self, feat, alphas=ALPHAS_WIDE, k=5, margin=0.002):
        self.feat, self.alphas, self.k, self.margin = feat, alphas, k, margin
        self._sr = StreamRidge(feat, alphas)

    def fit(self, E, R):
        n, K = len(E), R[0].shape[0]
        folds = _blocks(n, self.k)
        fold_of = np.empty(n, int)
        for fi, idx in enumerate(folds):
            fold_of[idx] = fi
        st = [self._sr._stats([E[i] for i in idx], [R[i] for i in idx]) for idx in folds]
        total = [sum(x) for x in zip(*st)]
        models = [StreamRidge._solve([a - b for a, b in zip(total, s)]) for s in st]
        sse = np.zeros((len(self.alphas), K))
        sse_off = np.zeros(K)
        for i in range(n):
            s = models[fold_of[i]]
            t = R[i].T.astype(np.float64)
            Z = self._sr._design(s, E[i]) @ s["Q"]
            for ai, a in enumerate(self.alphas):
                p = Z @ (s["QB"] / (s["lam"] + a * s["scale"])[:, None]) + s["ym"]
                sse[ai] += ((p - t) ** 2).sum(0)
            sse_off += ((s["ym"] - t) ** 2).sum(0)
        best = np.argmin(sse, 0)
        self.gate = sse[best, np.arange(K)] < sse_off * (1 - self.margin)
        full = StreamRidge._solve(total)
        W = np.zeros((len(full["mu"]), K))
        for ai, a in enumerate(self.alphas):
            cols = np.where((best == ai) & self.gate)[0]
            if len(cols):
                W[:, cols] = full["Q"] @ (full["QB"][:, cols] / (full["lam"] + a * full["scale"])[:, None])
        self.s, self.W = full, W
        return self

    def predict(self, E):
        if not self.gate.any():
            return [np.zeros((len(self.gate), e.shape[1]), np.float32) for e in E]
        return [(self._sr._design(self.s, e) @ self.W + self.s["ym"] * self.gate).T.astype(np.float32) for e in E]


def gated(base, Ec, Yc, Et, feat):
    m = GatedResidualRidge(feat).fit(Ec, [y - o for y, o in zip(Yc, base.oof)])
    return [p + q for p, q in zip(base.predict(Et), m.predict(Et))], float(m.gate.mean())


# ----------------------------------------------------------------------------- per subject
def analyze(E, Y, a, log=print):
    n = len(E)
    _, test = chrono_split(n, 0.2, gap=1)
    Et, Yt = [E[i] for i in test], [Y[i] for i in test]
    rfeat = RandomConvFeatures(E[0].shape[0], M=a.n_random, seed=a.seed)
    wfeat = MorletEnvBank()
    res = {"n_trials": n, "fracs": {}}
    for f in a.fracs:
        n_c = max(3, int(round(n * f)))
        if n_c >= test[0]:
            continue
        Ec, Yc = E[:n_c], Y[:n_c]
        bands = abs(f - 0.2) < 1e-9
        out, on = {}, {}

        def score(name, P):
            m = per_contact_metrics(P, Yt, FS, bands=bands)
            out[name] = {"mean": {k: float(np.nanmean(v)) for k, v in m.items()}, "r": [float(x) for x in m["r"]]}

        score("ridge", StreamRidge(LAG9).fit(Ec, Yc).predict(Et))
        base = KFoldRidge(LAG9, shared=True).fit(Ec, Yc)
        score("base", base.predict(Et))
        P, on["env_gated"] = gated(base, Ec, Yc, Et, feat_env)
        score("env_gated", P)
        P, on["wav_gated"] = gated(base, Ec, Yc, Et, wfeat)
        score("wav_gated", P)
        P, on["rocket_gated"] = gated(base, Ec, Yc, Et, rfeat)
        score("rocket_gated", P)
        if not a.no_all:
            P, on["all_gated"] = gated(base, Ec, Yc, Et, concat_feats(wfeat, rfeat))
            score("all_gated", P)
        mins = sum(e.shape[1] for e in Ec) / FS / 60
        res["fracs"][f"{f:g}"] = {"n_calib": n_c, "minutes": mins, "gate_on": on,
                                  "mean": {k: v["mean"]["r"] for k, v in out.items()},
                                  "bands": {k: v["mean"] for k, v in out.items()} if bands else {},
                                  "r": {k: v["r"] for k, v in out.items()}}
        log(f"    f={f:g} ({mins:.1f} min) " + " ".join(f"{k}={v['mean']['r']:+.3f}" for k, v in out.items()) +
            " | gate on: " + " ".join(f"{k}={v:.2f}" for k, v in on.items()))
    return res


def summarize(results, out_dir, log=print):
    from scipy.stats import wilcoxon

    def wp(x, y):
        try:
            return float(wilcoxon(x, y).pvalue)
        except ValueError:
            return float("nan")

    lines, summ = [], {}
    fracs = sorted({f for r in results for f in r["fracs"]}, key=float)
    for f in fracs:
        rows = [r for r in results if f in r["fracs"]]
        models = list(rows[0]["fracs"][f]["mean"])
        M = {m: np.array([r["fracs"][f]["mean"][m] for r in rows]) for m in models}
        mins = np.mean([r["fracs"][f]["minutes"] for r in rows])
        lines += [f"\n### calib fraction {f}  ({len(rows)} subjects, mean {mins:.1f} min)",
                  "| model | mean r | vs ridge | p | vs base | wins vs base | p | gate on |",
                  "|---|---|---|---|---|---|---|---|"]
        summ[f] = {}
        for m in models:
            d0, d1 = M[m] - M["ridge"], M[m] - M["base"]
            gon = np.mean([r["fracs"][f]["gate_on"].get(m, np.nan) for r in rows])
            p0 = wp(M[m], M["ridge"]) if m != "ridge" else float("nan")
            p1 = wp(M[m], M["base"]) if m not in ("ridge", "base") else float("nan")
            summ[f][m] = {"mean": float(M[m].mean()), "diff_ridge": float(d0.mean()), "p_ridge": p0,
                          "diff_base": float(d1.mean()), "wins_base": int((d1 > 0).sum()), "p_base": p1,
                          "gate_on": float(gon)}
            lines.append(f"| {m} | {M[m].mean():.3f} | {d0.mean():+.4f} | {p0:.4f} | {d1.mean():+.4f} | "
                         f"{(d1 > 0).sum()}/{len(d1)} | {p1:.4f} | {gon:.2f} |")
    if "0.2" in fracs:
        rows = [r for r in results if "0.2" in r["fracs"]]
        models = list(rows[0]["fracs"]["0.2"]["bands"])
        keys = ["r", "r2", "r_delta", "r_theta", "r_alpha", "r_beta", "env_alpha", "env_beta", "env_gamma"]
        keys += [k for k in ("r_no50", "r_induced") if k in rows[0]["fracs"]["0.2"]["bands"][models[0]]]
        lines += ["\n### calib fraction 0.2: by band", "| model | " + " | ".join(keys) + " |",
                  "|---" * (len(keys) + 1) + "|"]
        for m in models:
            lines.append(f"| {m} | " + " | ".join(
                f"{np.nanmean([r['fracs']['0.2']['bands'][m].get(k, np.nan) for r in rows]):.3f}" for k in keys) + " |")
        models = list(rows[0]["fracs"]["0.2"]["mean"])
        lines += ["\n### per subject (calib fraction 0.2)", "| subject | " + " | ".join(models) + " |",
                  "|---" * (len(models) + 1) + "|"]
        for r in rows:
            lines.append(f"| {r['subject']} | " + " | ".join(f"{r['fracs']['0.2']['mean'][m]:.3f}" for m in models) + " |")
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
    ap.add_argument("--n_random", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_all", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "nonlin.log"), "a", encoding="utf-8")

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
        fp = os.path.join(a.out, f"nonlin_{sid}.json")
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
