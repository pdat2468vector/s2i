"""
Phase 4 - readout improvements on the SAVED shared encoders (CPU, no retraining).

Phase 3 analysis of the shared encoder (15 subjects, 20 % calibration):
  (1) training traded high frequencies for low ones: the untrained Morlet bank (enc0) beats the
      trained encoder in alpha/beta, the trained one wins in delta/theta; two pairs collapsed on 50 Hz;
  (2) half of the features (pairs >= 13 Hz and their envelopes) sit in bands with little predictable
      signal, but one shared ridge penalty treats them like the useful ones;
  (3) at 1-2 min of calibration the readout (C x 40 weights per contact) is the bottleneck.
Every candidate below keeps the encoder fixed and changes only the readout:

  enc         reference = enc_ridge of s2i.shared (ridge, one K-fold alpha)
  base        reference = ridge_kfold_shared on LAG9
  enc_low     only low-frequency encoder features (pairs < 13 Hz, their envelopes, free filters)
  enc_group   high-frequency features down-weighted by a K-fold-chosen factor (two penalties)
  enc_enc0hi  + the UNTRAINED encoder's high-frequency features (alpha/beta), own K-fold weight
  enc_lag9    + LAG9 (raw lags +-200 ms), own K-fold weight
  enc_bands   target split into 5 complementary bands (<4, 4-8, 8-13, 13-30, >30 Hz); one ridge per
              band with its own K-fold penalty (chosen by MSE); predictions summed
  enc_sess    encoder features standardised per recording session (EEG only, label-free,
              transductive: uses the EEG of every trial of the session)
  enc_warp    filters time-scaled by s in {0.9, 1, 1.1} (frequencies x 1/s), s chosen per subject by
              K-fold (individual frequency adaptation)
  enc_bilin   structured readout W_k = a_k b_k^T (channels x features, rank 1-3, spectral profiles b
              orthonormal, ridge penalty on a), rank and penalty by K-fold: (C + F) R weights per contact
  ens_base    enc and base averaged with a K-fold-chosen weight
  ens_seed    enc of this run and of the seed-1 run averaged with a K-fold-chosen weight (if present)
--strict: undo preprocessing's whole-session normalisation and re-normalise EEG and target with the
calibration trials' statistics only; session-standardised models are skipped (combo_nosess = combo
without session standardisation is reported in every mode).
Phase 4b (--only_combo, combinations of the three Phase-4 winners):
  ens_all       ridge readouts of ALL available encoders (main + seed runs) averaged
  ens_all_sess  same, every encoder's features standardised per session
  combo         ens_all_sess mixed with base (LAG9 ridge), weight chosen by K-fold

Exactness: every readout is linear in its features, so K-fold out-of-fold r, the amplitude gain and
test r / R^2 are computed EXACTLY from second-order sufficient statistics (sums of x, xx', y, y^2,
xy') accumulated in one pass over the trials. `enc` therefore reproduces enc_ridge of s2i.shared.
Test trials, calibration fractions and fold blocks are those of phases 0-3.

Usage
-----
python -m s2i.readout --data DATA_DIR --run RUNS_DIR/shared_bip_car [--seed_run RUNS_DIR/shared_seed1]
python -m s2i.readout --data DATA_DIR --run RUNS_DIR/shared_bip_car --only_combo
       --seed_runs RUNS_DIR/shared_seed1 RUNS_DIR/shared_seed2     (Phase 4b: combinations of the winners)
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from s2i.ceiling import ALPHAS, chrono_split, feat_lag, template, tmpl_at
from s2i.baselines import DEFAULT_LAGS
from s2i.metrics import per_contact_metrics
from s2i.shared import TemporalEncoder, np_feat, r_single_trial, r_without_line
from s2i.within import ALPHAS_WIDE, _blocks

FS = 200
HIGH_HZ = 13.0
Y_CUTS = (4.0, 8.0, 13.0, 30.0)                       # -> 5 complementary target bands
GROUP_SCALES = (1.0, 0.5, 0.25, 0.1, 0.0)             # weight of the high-frequency group
ADD_SCALES = (0.0, 0.25, 0.5, 1.0, 2.0)               # weight of an added feature block
ENS_W = (0.0, 0.25, 0.5, 0.75, 1.0)                   # weight of enc in an ensemble
WARPS = (0.9, 1.1)
BILIN = tuple((r, al) for r in (1, 2, 3) for al in (0.001, 0.01, 0.1, 1.0, 10.0))
LAG9 = feat_lag(DEFAULT_LAGS)
_LP = {c: butter(4, c, btype="low", fs=FS, output="sos") for c in Y_CUTS}
_DEV = None                                           # set in main(); Gram matrices on the GPU if cuda


# ----------------------------------------------------------------------------- statistics
class St:
    """Second-order sufficient statistics of (features x, target y); optional band targets."""

    def __init__(self, D, K, nb=0, Db=0):
        self.n = 0.0
        self.sx, self.sxx = np.zeros(D), np.zeros((D, D))
        self.sy, self.syy, self.sxy = np.zeros(K), np.zeros(K), np.zeros((D, K))
        self.syb, self.syyb, self.sxyb = np.zeros((nb, K)), np.zeros((nb, K)), np.zeros((nb, Db, K))

    def add(self, X, y, yb=None):
        self.n += X.shape[1]
        self.sx += X.sum(1)
        if _DEV is not None and _DEV.type == "cuda":             # large Gram matrices on the GPU
            Xt, yt = torch.from_numpy(X).to(_DEV), torch.from_numpy(np.ascontiguousarray(y)).to(_DEV)
            self.sxx += (Xt @ Xt.T).double().cpu().numpy()
            self.sxy += (Xt @ yt.T).double().cpu().numpy()
        else:
            self.sxx += X @ X.T
            self.sxy += X @ y.T
        self.sy += y.sum(1)
        self.syy += (y * y).sum(1)
        if yb is not None:
            Xb = X[:self.sxyb.shape[1]]
            self.syb += yb.sum(2)
            self.syyb += (yb * yb).sum(2)
            for b in range(len(yb)):
                self.sxyb[b] += Xb @ yb[b].T

    def _new(self):
        return St.__new__(St)

    def __add__(self, o):
        s = self._new()
        for k in ("n", "sx", "sxx", "sy", "syy", "sxy", "syb", "syyb", "sxyb"):
            setattr(s, k, getattr(self, k) + getattr(o, k))
        return s

    def __sub__(self, o):
        s = self._new()
        for k in ("n", "sx", "sxx", "sy", "syy", "sxy", "syb", "syyb", "sxyb"):
            setattr(s, k, getattr(self, k) - getattr(o, k))
        return s

    def take(self, idx, band=None, transform=None):
        """Sub-block of features `idx` (band b replaces the target), optionally x -> (x - m) / d."""
        s = self._new()
        s.n, sx, sxx = self.n, self.sx[idx], self.sxx[np.ix_(idx, idx)]
        if band is None:
            s.sy, s.syy, sxy = self.sy, self.syy, self.sxy[idx]
        else:
            s.sy, s.syy, sxy = self.syb[band], self.syyb[band], self.sxyb[band][idx]
        if transform is not None:
            m, d = transform
            sxx = (sxx - np.outer(m, sx) - np.outer(sx, m) + s.n * np.outer(m, m)) / np.outer(d, d)
            sxy = (sxy - np.outer(m, s.sy)) / d[:, None]
            sx = (sx - s.n * m) / d
        s.sx, s.sxx, s.sxy = sx, sxx, sxy
        K = len(s.sy)
        s.syb, s.syyb, s.sxyb = np.zeros((0, K)), np.zeros((0, K)), np.zeros((0, len(idx), K))
        return s


def total(sts):
    out = sts[0]
    for s in sts[1:]:
        out = out + s
    return out


# ----------------------------------------------------------------------------- linear readouts
def standardise(tr):
    n = tr.n
    mu, ym = tr.sx / n, tr.sy / n
    G = tr.sxx - n * np.outer(mu, mu)
    sd = np.sqrt(np.clip(np.diag(G) / n, 0, None)) + 1e-8
    Gs = G / np.outer(sd, sd)
    Bs = (tr.sxy - n * np.outer(mu, ym)) / sd[:, None]
    return mu, sd, (Gs + Gs.T) / 2, Bs, ym


def to_raw(Ws, mu, sd, ym):
    A = Ws / sd[:, None]
    return A, ym - mu @ A


def ridge_models(tr, alphas, scale_vec=None):
    """Ridge for each alpha (penalty alpha * mean eigenvalue). scale_vec multiplies standardised features."""
    mu, sd, Gs, Bs, ym = standardise(tr)
    sc = max(np.trace(Gs) / len(Gs), 1e-12)      # penalty scale of the UNSCALED design (= n)
    if scale_vec is not None:
        Gs, Bs = Gs * np.outer(scale_vec, scale_vec), Bs * scale_vec[:, None]
    lam, Q = np.linalg.eigh(Gs)
    lam = np.clip(lam, 0, None)
    QB = Q.T @ Bs
    out = []
    for a in alphas:
        Ws = Q @ (QB / (lam + a * sc)[:, None])
        if scale_vec is not None:
            Ws = Ws * scale_vec[:, None]
        out.append(to_raw(Ws, mu, sd, ym))
    return out


def bilinear_models(tr, C, F, cands, iters=5):
    """
    W_k = a_k b_k^T over (channel, feature) with rank R: b_k (F x R) has orthonormal columns (spectral
    profiles), a_k (C x R) carries the ridge penalty alpha * n, exactly the ridge scale. Alternating
    least squares, initialised by the SVD of a ridge fit.
    """
    mu, sd, Gs, Bs, ym = standardise(tr)
    K, sc = Bs.shape[1], max(np.trace(Gs) / len(Gs), 1e-12)
    G4, B3 = Gs.reshape(C, F, C, F), Bs.reshape(C, F, K)
    lam, Q = np.linalg.eigh(Gs)
    W0 = Q @ ((Q.T @ Bs) / (np.clip(lam, 0, None) + sc)[:, None])
    _, _, Vt = np.linalg.svd(W0.T.reshape(K, C, F), full_matrices=False)
    out = []
    for R, alpha in cands:
        R_ = min(R, Vt.shape[1])
        b = Vt[:, :R_, :].transpose(0, 2, 1)                     # (K, F, R) orthonormal columns

        def solve_a(b):
            # Ga[k] = (I_C x b_k)' G (I_C x b_k): contract g with b (one matmul), then f (batched matmul)
            M = (Gs.reshape(C * F * C, F) @ b.transpose(1, 0, 2).reshape(F, K * R_)).reshape(C, F, C, K, R_)
            Ga = np.matmul(M.transpose(3, 0, 2, 4, 1), b[:, None, None]).transpose(0, 1, 4, 2, 3)
            ca = np.einsum("cfk,kfr->kcr", B3, b).reshape(K, C * R_, 1)
            return np.linalg.solve(Ga.reshape(K, C * R_, C * R_) + alpha * sc * np.eye(C * R_),
                                   ca)[..., 0].reshape(K, C, R_)
        a = solve_a(b)
        for _ in range(iters):
            # Gb[k] = (a_k x I_F)' G (a_k x I_F): contract c with a (one matmul), then d (batched matmul)
            M = (a.transpose(0, 2, 1).reshape(K * R_, C) @ Gs.reshape(C, F * C * F)).reshape(K, R_, F, C, F)
            Gb = np.matmul(M.transpose(0, 1, 2, 4, 3), a[:, None, None]).transpose(0, 2, 1, 3, 4)
            Gb = Gb.reshape(K, F * R_, F * R_)
            cb = np.einsum("cfk,kcr->kfr", B3, a).reshape(K, F * R_, 1)
            b = np.linalg.solve(Gb + 1e-4 * sc * np.eye(F * R_), cb)[..., 0].reshape(K, F, R_)
            b, _ = np.linalg.qr(b)                               # keep only the profile subspace
            a = solve_a(b)
        Ws = np.einsum("kcr,kfr->cfk", a, b).reshape(C * F, K)
        out.append(to_raw(Ws, mu, sd, ym))
    return out


def pred_sums(A, c, te):
    """Sums of prediction p = x'A + c over the samples summarised by `te`: (n, sp, spp, spy, sy, syy)."""
    Ax = A.T @ te.sx
    sp = Ax + te.n * c
    spp = ((te.sxx @ A) * A).sum(0) + 2 * c * Ax + te.n * c ** 2
    spy = (A * te.sxy).sum(0) + c * te.sy
    return np.stack([np.full_like(sp, te.n), sp, spp, spy, te.sy, te.syy])


def r_of(s):
    n, sp, spp, spy, sy, syy = s
    den = np.sqrt(np.clip(n * spp - sp ** 2, 0, None) * np.clip(n * syy - sy ** 2, 0, None))
    return np.where(den > 0, (n * spy - sp * sy) / np.maximum(den, 1e-12), np.nan)


def sse_of(s):
    return s[2] - 2 * s[3] + s[5]


class Fit:
    """K-fold selection over candidates from statistics. fit(train_stats) -> [(A, c)] per candidate."""

    def __init__(self, fit, folds, tot, crit="r"):
        self.fit, self.tot = fit, tot
        self.oof = None
        for tr, te in folds:
            s = np.stack([pred_sums(A, c, te) for A, c in fit(tr)])
            self.oof = s if self.oof is None else self.oof + s
        if crit == "r":
            score = [np.nanmean(r_of(o)) for o in self.oof]
            self.best = int(np.nanargmax(score))
        else:
            self.best = int(np.argmin([np.nansum(sse_of(o)) for o in self.oof]))
        self.A, self.c = fit(tot)[self.best]
        self.fold_models = None

    def fold_model(self, folds):
        if self.fold_models is None:
            self.fold_models = [self.fit(tr)[self.best] for tr, _ in folds]
        return self.fold_models


def calibrated(A, c, oof, ym):
    """Per-contact least-squares gain on the pooled out-of-fold predictions (r unchanged, R^2 honest)."""
    n, sp, spp, spy, sy, _ = oof
    num = spy - ym * sp - ym * sy + n * ym ** 2
    den = spp - 2 * ym * sp + n * ym ** 2
    g = np.clip(num / np.maximum(den, 1e-12), 0, None)
    return A * g, ym + g * (c - ym)


def score(A, c, oof, tot, te, te_sessions=None):
    ym = tot.sy / tot.n
    s = pred_sums(A, c, te)
    Ac, cc = calibrated(A, c, oof, ym)
    sc = pred_sums(Ac, cc, te)
    r2 = 1 - sse_of(sc) / np.maximum(sc[5] - sc[4] ** 2 / sc[0], 1e-12)
    out = {"r": float(np.nanmean(r_of(s))), "r2": float(np.nanmean(r2)), "contacts": [float(x) for x in r_of(s)],
           "cv_contacts": [float(x) for x in r_of(oof)]}          # K-fold r on calibration (reliability map)
    if te_sessions:                                            # calibration longevity: r per test session
        out["sessions"] = {k: float(np.nanmean(r_of(pred_sums(A, c, v)))) for k, v in te_sessions.items()}
    return out


def embed(A, c, idx, D):
    out = np.zeros((D, A.shape[1]))
    out[idx] = A
    return out, c


# ----------------------------------------------------------------------------- features
def load_encoder(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    enc = TemporalEncoder(**ck["cfg"]).to(dev)
    enc.load_state_dict(ck["state"])
    return enc.eval()


def warped(enc, s, dev):
    """Same encoder with every kernel time-scaled by s: h_s(t) = h(s t), i.e. frequencies x 1/s."""
    out = TemporalEncoder(**enc.cfg).to(dev)
    w = enc.conv.weight.detach().cpu().numpy()[:, 0]
    H = w.shape[1] // 2
    t = np.arange(-H, H + 1)
    ws = np.stack([np.interp(t / s, t, k, left=0.0, right=0.0) for k in w])
    ws *= np.linalg.norm(w, axis=1, keepdims=True) / (np.linalg.norm(ws, axis=1, keepdims=True) + 1e-12)
    with torch.no_grad():
        out.conv.weight.copy_(torch.tensor(ws, dtype=torch.float32)[:, None])
    return out.eval()


def high_mask(enc, use_init):
    """Per-feature mask (length n_feat): True for quadrature pairs >= HIGH_HZ and their envelopes."""
    cfg = enc.cfg
    P = cfg["n_pairs"]
    if cfg.get("arch", "morlet") != "morlet":                 # no quadrature pairs (ShallowConvNet-style front end)
        return np.zeros(enc.n_feat, bool), np.zeros(0)
    if use_init:
        f = np.geomspace(cfg["fmin"], cfg["fmax"], P)
    else:
        w = enc.conv.weight.detach().cpu().numpy()[:, 0]
        spec = np.abs(np.fft.rfft(w, 4096, axis=1)) ** 2
        fr = np.fft.rfftfreq(4096, 1 / FS)
        f = fr[np.argmax(spec[0:2 * P:2] + spec[1:2 * P:2], 1)]
    m = np.zeros(enc.n_feat, bool)
    for p in np.where(f >= HIGH_HZ)[0]:
        m[[2 * p, 2 * p + 1]] = True
        if cfg["envelope"]:
            m[2 * P + cfg["n_free"] + p] = True
    return m, f


def band_targets(y):
    lp = [sosfiltfilt(_LP[c], y, axis=1) for c in Y_CUTS]
    parts = [lp[0]] + [lp[i] - lp[i - 1] for i in range(1, len(lp))] + [y - lp[-1]]
    return np.stack(parts).astype(np.float32)


# ----------------------------------------------------------------------------- per subject
def load_subject(path):
    """-> trials of CAR EEG, trials of bipolar targets, trial sessions, stimulus events (or None)."""
    d = torch.load(path, weights_only=False)
    eeg, tgt = d["eeg_car"].half().float().numpy(), d["ieeg_bip"].half().float().numpy()
    b = d["trial_bounds"].numpy()
    ses = list(d.get("trial_session", ["ses-01"] * len(b)))
    ev = d.get("stim_events")
    return ([eeg[:, s0:s1] for s0, s1 in b], [tgt[:, s0:s1] for s0, s1 in b], ses,
            None if ev is None else np.asarray(ev, int).reshape(-1, 2))


EPOCH = (-100, 500)                                   # -0.5 to +2.5 s around a stimulus marker (samples)


def r_event(P, T, test, events):
    """
    Per-contact r without the stimulus-locked component: each side minus ITS OWN average over the test stimulus
    epochs (-0.5 to +2.5 s around each stimulus marker). events: rows (trial index, onset sample within the trial).
    An epoch is used only if it lies inside one test trial / window; a sample covered by two epochs takes the
    later one. Returns NaN per contact if there is no such epoch.
    """
    pos = {int(w): j for j, w in enumerate(test)}
    ep = sorted((pos[int(w)], int(o)) for w, o in events
                if int(w) in pos and o + EPOCH[0] >= 0 and o + EPOCH[1] <= P[pos[int(w)]].shape[1])
    if not ep:
        return np.full(P[0].shape[0], np.nan)
    tP = np.mean([P[j][:, o + EPOCH[0]:o + EPOCH[1]] for j, o in ep], 0)
    tT = np.mean([T[j][:, o + EPOCH[0]:o + EPOCH[1]] for j, o in ep], 0)
    Pr, Tr = [p.astype(np.float64) for p in P], [t.astype(np.float64) for t in T]
    for j, o in ep:                                   # ascending: a later epoch overwrites the overlap
        Pr[j][:, o + EPOCH[0]:o + EPOCH[1]] = P[j][:, o + EPOCH[0]:o + EPOCH[1]] - tP
        Tr[j][:, o + EPOCH[0]:o + EPOCH[1]] = T[j][:, o + EPOCH[0]:o + EPOCH[1]] - tT
    return per_contact_metrics(Pr, Tr, FS, bands=False)["r"]


def r_induced_by_session(P, T, test, ses):
    """
    r_induced with the trial average taken per SESSION: each side minus its own average over the test trials of the
    same session (e.g. one stimulation site per run in the third cohort). Same rule as r_single_trial otherwise.
    """
    Pr, Tr = [None] * len(P), [None] * len(T)
    for s in dict.fromkeys(ses[i] for i in test):
        js = [j for j, i in enumerate(test) if ses[i] == s]
        tP, tT = template([P[j] for j in js]), template([T[j] for j in js])
        for j in js:
            Pr[j] = P[j] - tmpl_at(tP, P[j].shape[1])
            Tr[j] = T[j] - tmpl_at(tT, T[j].shape[1])
    return per_contact_metrics(Pr, Tr, FS, bands=False)["r"]


def save_predictions(a, res, preds, Yt, test):
    """pred_<sub>.npz: signals of the best, median and worst contact (final-model test r) for the final model, the
    LAG9 ridge and the recording (float16, all test trials concatenated, with the trial lengths), every contact's r,
    and the magnitude-squared coherence between prediction and recording averaged over contacts."""
    from scipy.signal import coherence
    rr = np.asarray(res["explicit"]["combo_nosess"]["r"], float)
    ok = np.where(np.isfinite(rr))[0]
    order = ok[np.argsort(rr[ok])]
    pick = [int(order[-1]), int(order[len(order) // 2]), int(order[0])]
    T = np.concatenate(Yt, 1).astype(np.float64)
    out = {"pick": np.array(pick), "trial_len": np.array([y.shape[1] for y in Yt]), "test": np.array(test),
           "r_final": rr, "r_base": np.asarray(res["explicit"]["base"]["r"], float),
           "sig_target": T[pick].astype(np.float16)}
    for nm, key in (("combo_nosess", "final"), ("base", "base")):
        P = np.concatenate(preds[nm], 1).astype(np.float64)
        out[f"sig_{key}"] = P[pick].astype(np.float16)
        C = []
        for k in range(T.shape[0]):
            if np.ptp(P[k]) == 0:
                continue                                  # constant prediction: coherence undefined
            fr, c = coherence(P[k], T[k], fs=FS, nperseg=256)
            C.append(c)
        out["freqs"] = fr
        out[f"coh_{key}"] = np.nanmean(C, 0) if C else np.full(129, np.nan)
    # log band power per test trial in the last 2 s of the maintenance period (4-6 s after trial onset)
    from scipy.signal import butter, sosfiltfilt
    w0, w1 = 4 * FS, 6 * FS
    for band, lo, hi in (("theta", 4.0, 8.0), ("alpha", 8.0, 12.0)):
        sos = butter(4, (lo, hi), btype="band", fs=FS, output="sos")
        for key, S in (("target", Yt), ("final", preds["combo_nosess"]), ("base", preds["base"])):
            bp = np.full((len(S), S[0].shape[0]), np.nan, np.float32)
            for j, x in enumerate(S):
                if x.shape[1] >= w1:
                    y = sosfiltfilt(sos, x.astype(np.float64), axis=1)[:, w0:w1]
                    bp[j] = np.log(np.mean(y ** 2, 1) + 1e-12)
            out[f"bp_{band}_{key}"] = bp
    np.savez_compressed(os.path.join(a._out, f"pred_{a._sid}.npz"), **out)


def _robust(x):
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True) * 1.4826
    return med, np.where(mad < 1e-12, 1.0, mad)


def strict_renorm(E, Y, ses, stats, n_cal):
    """
    Undo the whole-session robust normalisation of preprocessing (x * MAD_s + median_s) and re-normalise every
    trial with the median / MAD of the CALIBRATION trials only (first n_cal), for EEG and target alike:
    no statistic of a test trial is used anywhere (strict, deployment-faithful normalisation).
    """
    def redo(X, key):
        raw = []
        for x, s in zip(X, ses):
            st = stats["norm_stats"][key][s]
            raw.append(x * np.asarray(st["mad"], np.float32)[:, None] + np.asarray(st["med"], np.float32)[:, None])
        med, mad = _robust(np.concatenate(raw[:n_cal], 1))
        return [((r - med) / mad).astype(np.float32) for r in raw]
    return redo(E, "eeg_car"), redo(Y, "ieeg_bip")


def analyze(E, Y, ses, enc, seed_encs, a, dev, log, events=None):
    amp = dev.type == "cuda"
    zoo = not a.only_combo
    n, C, K = len(E), E[0].shape[0], Y[0].shape[0]
    _, test = chrono_split(n, getattr(a, "test_frac", 0.2), gap=1)
    enc0 = TemporalEncoder(**enc.cfg).to(dev).eval()
    F = enc.n_feat
    hi_enc, f_enc = high_mask(enc, use_init=False)
    hi_e0, _ = high_mask(enc0, use_init=True)
    f1, f0 = np_feat(enc, dev, amp), np_feat(enc0, dev, amp)
    fss = [np_feat(se, dev, amp) for se in seed_encs]
    fs1 = fss[0] if fss else None
    fw = {s: np_feat(warped(enc, s, dev), dev, amp) for s in WARPS} if zoo else {}
    e0_rows = np.where(np.tile(hi_e0, C))[0]
    seeds = [f"seed{j}" for j in range(len(fss))]

    # main view = [enc | enc0 high-frequency rows (zoo only) | LAG9 | seed encoders]
    blocks = [("enc", C * F)] + ([("enc0hi", len(e0_rows))] if zoo else []) + [("lag9", C * len(DEFAULT_LAGS))]
    blocks += [(b, C * se.n_feat) for b, se in zip(seeds, seed_encs)]     # encoders may differ (e.g. no_env)
    idx, o = {}, 0
    for name, size in blocks:
        idx[name] = np.arange(o, o + size)
        o += size
    D = o
    hi_rows = np.where(np.tile(hi_enc, C))[0]
    lo_rows = np.where(~np.tile(hi_enc, C))[0]

    def main_x(e):
        x1 = f1(e)
        parts = [x1] + ([f0(e)[e0_rows]] if zoo else []) + [LAG9(e)] + [f(e) for f in fss]
        return np.concatenate(parts).astype(np.float32), x1

    # session statistics of every encoder's features (EEG of every trial, no target)
    enc_fns = {"enc": f1, **dict(zip(seeds, fss))}
    sess_sum = {}
    for i in range(n):
        for b, fn in enc_fns.items():
            x = fn(E[i]).astype(np.float64)
            s = sess_sum.setdefault((b, ses[i]), [0, np.zeros(len(idx[b])), np.zeros(len(idx[b]))])
            s[0] += x.shape[1]
            s[1] += x.sum(1)
            s[2] += (x * x).sum(1)
    sess_tf = {k: (v[1] / v[0], np.sqrt(np.clip(v[2] / v[0] - (v[1] / v[0]) ** 2, 0, None)) + 1e-6)
               for k, v in sess_sum.items()}

    def tf_for(names, session):
        """Per-session standardisation of encoder blocks; identity for other blocks (e.g. LAG9)."""
        m = [sess_tf[(b, session)][0] if (b, session) in sess_tf else np.zeros(len(idx[b])) for b in names]
        d = [sess_tf[(b, session)][1] if (b, session) in sess_tf else np.ones(len(idx[b])) for b in names]
        return np.concatenate(m), np.concatenate(d)

    # test statistics (per session, for the session-standardised readout)
    te_main, te_w = {}, {s: St(C * F, K) for s in fw}
    for i in test:
        X, _ = main_x(E[i])
        te_main.setdefault(ses[i], St(D, K)).add(X, Y[i])
        for s in fw:
            te_w[s].add(fw[s](E[i]).astype(np.float32), Y[i])
    te = total(list(te_main.values()))
    te_enc = te.take(idx["enc"])
    te_sess = total([v.take(idx["enc"], transform=tf_for(["enc"], k)) for k, v in te_main.items()])

    order = list(dict.fromkeys(ses))                     # sessions in recording order
    res = {"n_trials": n, "n_test": len(test), "n_sessions": len({k[1] for k in sess_sum}),
           "session_order": order, "test_trials_per_session": {k: sum(ses[i] == k for i in test) for k in order},
           "n_encoders": 1 + len(fss),
           "pair_peak_hz": [float(x) for x in f_enc], "fracs": {}}
    for frac in a.fracs:
        n_c = max(3, int(round(n * frac)))
        if n_c >= test[0]:
            continue
        t0 = time.time()
        fold_ids = _blocks(n_c, 5)
        cal = {}                                 # (fold, session) -> St (main view, with band targets)
        cal_w = {s: [St(C * F, K) for _ in fold_ids] for s in fw}
        nbt = len(Y_CUTS) + 1 if zoo else 0
        for fi, ids in enumerate(fold_ids):
            for i in ids:
                X, _ = main_x(E[i])
                cal.setdefault((fi, ses[i]), St(D, K, nb=nbt, Db=C * F if zoo else 0)).add(
                    X, Y[i], band_targets(Y[i]) if zoo else None)
                for s in fw:
                    cal_w[s][fi].add(fw[s](E[i]).astype(np.float32), Y[i])
        fold_st = [total([v for (f, _), v in cal.items() if f == fi]) for fi in range(len(fold_ids))]
        tot = total(fold_st)

        def folds_of(take):
            return [(take(tot - fs), take(fs)) for fs in fold_st]

        out, extra = {}, {}

        def run(name, take, fit, test_st, crit="r", note=None):
            fl = folds_of(take)
            m = Fit(fit, fl, take(tot), crit)
            out[name] = score(m.A, m.c, m.oof[m.best], take(tot), test_st,
                              {k: take(v) for k, v in te_main.items()})
            if note:
                extra[name] = note(m)
            return m, fl

        t_enc = lambda s: s.take(idx["enc"])                                    # noqa: E731
        m_enc, fl_enc = run("enc", t_enc, lambda tr: ridge_models(tr, ALPHAS_WIDE), te_enc,
                            note=lambda m: ALPHAS_WIDE[m.best])
        t_lag = lambda s: s.take(idx["lag9"])                                   # noqa: E731
        m_base, fl_base = run("base", t_lag, lambda tr: ridge_models(tr, ALPHAS), te.take(idx["lag9"]))
        store = {}                       # model -> (feature rows, calibrated A, c, session-standardised blocks)
        for nm, m_, rows in (("enc", m_enc, idx["enc"]), ("base", m_base, idx["lag9"])):
            ts = tot.take(rows)
            store[nm] = (rows, *calibrated(m_.A, m_.c, m_.oof[m_.best], ts.sy / ts.n), None)

        if zoo:
            lo = idx["enc"][lo_rows]
            run("enc_low", lambda s: s.take(lo), lambda tr: ridge_models(tr, ALPHAS_WIDE), te.take(lo))

            gmask = np.zeros(C * F, bool)
            gmask[hi_rows] = True
            cands = [(g, al) for g in GROUP_SCALES for al in ALPHAS_WIDE]

            def fit_group(tr):
                outm = []
                for g in GROUP_SCALES:
                    outm += ridge_models(tr, ALPHAS_WIDE, np.where(gmask, g, 1.0))
                return outm
            run("enc_group", t_enc, fit_group, te_enc, note=lambda m: cands[m.best][0])

            def added(name, block):
                sel = np.concatenate([idx["enc"], idx[block]])
                mask = np.r_[np.zeros(len(idx["enc"]), bool), np.ones(len(idx[block]), bool)]
                cands_ = [(g, al) for g in ADD_SCALES for al in ALPHAS_WIDE]

                def fit(tr):
                    outm = []
                    for g in ADD_SCALES:
                        outm += ridge_models(tr, ALPHAS_WIDE, np.where(mask, g, 1.0))
                    return outm
                run(name, lambda s: s.take(sel), fit, te.take(sel), note=lambda m: cands_[m.best][0])
            added("enc_enc0hi", "enc0hi")
            added("enc_lag9", "lag9")

            # one ridge per target band (penalty by MSE on that band), predictions summed
            nb = len(Y_CUTS) + 1
            bm = [Fit(lambda tr: ridge_models(tr, ALPHAS_WIDE), folds_of(lambda s, b=b: s.take(idx["enc"], band=b)),
                      tot.take(idx["enc"], band=b), crit="mse") for b in range(nb)]
            A = sum(m.A for m in bm)
            c = sum(m.c for m in bm)
            fms = [m.fold_model([(tr, None) for tr, _ in folds_of(lambda s, b=b: s.take(idx["enc"], band=b))])
                   for b, m in enumerate(bm)]
            oof = sum(pred_sums(sum(fms[b][fi][0] for b in range(nb)), sum(fms[b][fi][1] for b in range(nb)), te_f)
                      for fi, (_, te_f) in enumerate(fl_enc))
            out["enc_bands"] = score(A, c, oof, t_enc(tot), te_enc)
            extra["enc_bands"] = [ALPHAS_WIDE[m.best] for m in bm]

            # per-session standardisation of the encoder features
            sess_tot = total([v.take(idx["enc"], transform=tf_for(["enc"], s)) for (f, s), v in cal.items()])
            sess_folds = [total([v.take(idx["enc"], transform=tf_for(["enc"], s)) for (f, s), v in cal.items()
                                 if f == fi])
                          for fi in range(len(fold_ids))]
            m = Fit(lambda tr: ridge_models(tr, ALPHAS_WIDE), [(sess_tot - fs, fs) for fs in sess_folds], sess_tot)
            out["enc_sess"] = score(m.A, m.c, m.oof[m.best], sess_tot, te_sess)

            # individual frequency adaptation: pick s in {0.9, 1, 1.1} by K-fold r
            best = ("1.0", m_enc, np.nanmean(r_of(m_enc.oof[m_enc.best])), te_enc, t_enc(tot))
            for s in WARPS:
                tw = total(cal_w[s])
                mw = Fit(lambda tr: ridge_models(tr, ALPHAS_WIDE), [(tw - fs, fs) for fs in cal_w[s]], tw)
                sw = np.nanmean(r_of(mw.oof[mw.best]))
                if sw > best[2]:
                    best = (f"{s:g}", mw, sw, te_w[s], tw)
            out["enc_warp"] = score(best[1].A, best[1].c, best[1].oof[best[1].best], best[4], best[3])
            extra["enc_warp"] = best[0]

            # structured (bilinear) readout
            if not a.no_bilin:
                run("enc_bilin", t_enc, lambda tr: bilinear_models(tr, C, F, BILIN), te_enc,
                    note=lambda m: list(BILIN[m.best]))

        # ensembles: enc + base, enc + seed-1 (weights chosen by pooled out-of-fold r)
        def ensemble(name, m1, fl1, i1, m2, fl2, i2):
            sel = np.concatenate([i1, i2])
            n1 = len(i1)
            fm1 = m1.fold_model(fl1)
            fm2 = m2.fold_model(fl2)
            t_sel = lambda s: s.take(sel)                                       # noqa: E731
            fl = folds_of(t_sel)
            oofs = []
            for w in ENS_W:
                oofs.append(sum(pred_sums(np.r_[w * fm1[fi][0], (1 - w) * fm2[fi][0]], w * fm1[fi][1] +
                                          (1 - w) * fm2[fi][1], te_f) for fi, (_, te_f) in enumerate(fl)))
            wi = int(np.nanargmax([np.nanmean(r_of(o)) for o in oofs]))
            w = ENS_W[wi]
            A_ = np.r_[w * m1.A, (1 - w) * m2.A]
            out[name] = score(A_, w * m1.c + (1 - w) * m2.c, oofs[wi], t_sel(tot), te.take(sel))
            extra[name] = w
            assert len(A_) == len(sel) and n1 == len(m1.A)
        ensemble("ens_base", m_enc, fl_enc, idx["enc"], m_base, fl_base, idx["lag9"])
        if fss:
            t_seed = lambda s: s.take(idx["seed0"])                             # noqa: E731
            m_seed, fl_seed = run("seed", t_seed, lambda tr: ridge_models(tr, ALPHAS_WIDE), te.take(idx["seed0"]))
            ensemble("ens_seed", m_enc, fl_enc, idx["enc"], m_seed, fl_seed, idx["seed0"])

        # combinations of the three Phase-4 winners: several encoders averaged (ens_seed), session
        # standardisation (enc_sess) and mixing with LAG9 ridge (ens_base); exact from statistics
        def mixture(name, encs, sess, fixed_w=None):
            names = encs + ["lag9"]
            sel = np.concatenate([idx[b] for b in names])

            def T(st, session):
                return st.take(sel, transform=tf_for(names, session) if sess else None)
            folds_T = [total([T(v, ss) for (f, ss), v in cal.items() if f == fi]) for fi in range(len(fold_ids))]
            tot_T = total(folds_T)
            te_T = total([T(v, ss) for ss, v in te_main.items()])
            comps, o_ = [], 0
            for b in names:
                loc = np.arange(o_, o_ + len(idx[b]))
                o_ += len(idx[b])
                al = ALPHAS if b == "lag9" else ALPHAS_WIDE
                fl = [((tot_T - fs).take(loc), fs.take(loc)) for fs in folds_T]
                m = Fit(lambda tr, al=al: ridge_models(tr, al), fl, tot_T.take(loc))
                comps.append((b, loc, m, m.fold_model(fl)))

            def combine(w, models):
                A_, c_ = np.zeros((len(sel), K)), np.zeros(K)
                for (b, loc, _, _), (Ab, cb) in zip(comps, models):
                    wt = (1 - w) if b == "lag9" else w / len(encs)
                    A_[loc] += wt * Ab
                    c_ += wt * cb
                return A_, c_
            ws = ENS_W if fixed_w is None else (fixed_w,)
            oofs = [sum(pred_sums(*combine(w, [cm[3][fi] for cm in comps]), fs) for fi, fs in enumerate(folds_T))
                    for w in ws]
            wi = int(np.nanargmax([np.nanmean(r_of(o)) for o in oofs]))
            A_, c_ = combine(ws[wi], [(cm[2].A, cm[2].c) for cm in comps])
            out[name] = score(A_, c_, oofs[wi], tot_T, te_T, {ss: T(v, ss) for ss, v in te_main.items()})
            store[name] = (sel, *calibrated(A_, c_, oofs[wi], tot_T.sy / tot_T.n), names if sess else None)
            extra[name] = ws[wi]

        all_encs = ["enc"] + seeds
        if not zoo and not a.strict:
            mixture("enc_sess", ["enc"], sess=True, fixed_w=1.0)
        if seeds:
            mixture("ens_all", all_encs, sess=False, fixed_w=1.0)
            if not a.strict:
                mixture("ens_all_sess", all_encs, sess=True, fixed_w=1.0)
        if not a.strict:                         # session standardisation uses the EEG of every trial (transductive)
            mixture("combo", all_encs, sess=True)
        mixture("combo_nosess", all_encs, sess=False)

        if a.explicit and abs(frac - a.explicit_frac) < 1e-9:
            # explicit test predictions of the final models: band / 50 Hz / single-trial metrics
            res["explicit"] = {}
            Yt = [Y[i] for i in test]
            preds = {}
            for nm in [m for m in ("base", "enc", "combo", "combo_nosess") if m in store]:
                rows, A_, c_, names = store[nm]
                P = []
                for i in test:
                    X = main_x(E[i])[0][rows].astype(np.float64)
                    if names:
                        mm, dd = tf_for(names, ses[i])
                        X = (X - mm[:, None]) / dd[:, None]
                    P.append((X.T @ A_ + c_).T.astype(np.float32))
                met = per_contact_metrics(P, Yt, FS, bands=True)
                met["r_no50"], met["r_induced"] = r_without_line(P, Yt), r_single_trial(P, Yt)
                if getattr(a, "induced_by_session", False):
                    met["r_induced_ses"] = r_induced_by_session(P, Yt, test, ses)
                if events is not None:
                    met["r_event"] = r_event(P, Yt, test, events)
                preds[nm] = P
                Pc = np.concatenate(P, 1)
                res["explicit"][nm] = {"frac": frac, "mean": {k: float(np.nanmean(v)) for k, v in met.items()},
                                       "r": [float(x) for x in met["r"]],
                                       "r_induced": [float(x) for x in met["r_induced"]],
                                       "n_const": int((Pc.max(1) == Pc.min(1)).sum())}   # constant -> r = 0
            if getattr(a, "save_pred", False) and "combo_nosess" in preds and "base" in preds:
                save_predictions(a, res, preds, Yt, test)
            log("    explicit: " + " ".join(f"{k} r={v['mean']['r']:.4f} (stats {out[k]['r']:.4f}) "
                                          f"r_ind={v['mean']['r_induced']:.4f}"
                                          for k, v in res["explicit"].items()))
        mins = sum(E[i].shape[1] for i in range(n_c)) / FS / 60
        res["fracs"][f"{frac:g}"] = {"n_calib": n_c, "minutes": mins, "models": out, "choice": extra,
                                     "calib_sessions": sorted({ses[i] for i in range(n_c)}, key=order.index)}
        log(f"    f={frac:g} ({mins:.1f} min, {time.time() - t0:.0f}s) " +
            " ".join(f"{k}={v['r']:.3f}" for k, v in out.items()))
    return res


# ----------------------------------------------------------------------------- summary
def summarize(results, out_dir, log):
    from scipy.stats import wilcoxon
    lines, summ = [], {}
    fracs = sorted({f for r in results for f in r["fracs"]}, key=float)
    for f in fracs:
        rows = [r["fracs"][f] for r in results if f in r["fracs"]]
        models = [m for m in rows[0]["models"] if all(m in r["models"] for r in rows)]
        M = {m: np.array([r["models"][m]["r"] for r in rows]) for m in models}
        R2 = {m: np.array([r["models"][m]["r2"] for r in rows]) for m in models}
        mins = np.mean([r["minutes"] for r in rows])
        lines += [f"\n### calib fraction {f}  ({len(rows)} subjects, mean {mins:.1f} min)",
                  "| model | mean r | vs enc | wins vs enc | p vs enc | vs base | R² (calibrated) | K-fold choices |",
                  "|---|---|---|---|---|---|---|---|"]
        summ[f] = {}
        for m in models:
            d = M[m] - M["enc"]
            try:
                p = float(wilcoxon(M[m], M["enc"]).pvalue) if m != "enc" and np.any(d != 0) else float("nan")
            except ValueError:
                p = float("nan")
            ch = [r["choice"].get(m) for r in rows if m in r["choice"]]
            ch = [tuple(x) if isinstance(x, list) else x for x in ch]
            if ch and m != "enc_bands":
                vals, cnt = np.unique([str(x) for x in ch], return_counts=True)
                chs = ", ".join(f"{v}×{c}" for v, c in zip(vals, cnt))
            else:
                chs = ""
            summ[f][m] = {"mean": float(M[m].mean()), "diff_enc": float(d.mean()), "wins_enc": int((d > 0).sum()),
                          "p_enc": p, "diff_base": float((M[m] - M["base"]).mean()), "r2": float(R2[m].mean())}
            lines.append(f"| {m} | {M[m].mean():.4f} | {d.mean():+.4f} | {(d > 0).sum()}/{len(d)} | {p:.4f} | "
                         f"{(M[m] - M['base']).mean():+.4f} | {R2[m].mean():.3f} | {chs} |")
    md = "\n".join(lines)
    log(md)
    with open(os.path.join(out_dir, "readout.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(out_dir, "readout.json"), "w") as fh:
        json.dump(summ, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="output dir of s2i.shared (enc_<sub>.pt)")
    ap.add_argument("--seed_run", default=None, help="second run for the seed ensemble (enc_<sub>.pt)")
    ap.add_argument("--seed_runs", nargs="*", default=[], help="further runs (encoders) for the ensemble")
    ap.add_argument("--explicit", action="store_true",
                    help="also predict the test trials explicitly with base / enc / combo (all metrics)")
    ap.add_argument("--explicit_frac", type=float, default=0.2)
    ap.add_argument("--strict", action="store_true",
                    help="re-normalise EEG and target with calibration-trial statistics only (needs <sub>_normstats.pt "
                         "from preprocess --stats_only); no session standardisation; requires a single --fracs value")
    ap.add_argument("--stats_dir", default=None, help="folder with <sub>_normstats.pt (default: --data)")
    ap.add_argument("--untrained", action="store_true",
                    help="replace every encoder by its Morlet initialisation (fixed filter bank, no meta-learning)")
    ap.add_argument("--only_combo", action="store_true",
                    help="skip the Phase-4 variants already rejected; compute enc, base, ensembles and combos")
    ap.add_argument("--out_name", default=None, help="output sub-folder (default: readout / readout_combo)")
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.05, 0.1, 0.2])
    ap.add_argument("--test_frac", type=float, default=0.2,
                    help="the test trials follow the first round(test_frac * n) trials and a one-trial gap (fixed for "
                         "every calibration fraction; fractions above it are skipped)")
    ap.add_argument("--save_pred", action="store_true",
                    help="explicit: save pred_<sub>.npz (three contacts' test signals and coherence spectra)")
    ap.add_argument("--induced_by_session", action="store_true",
                    help="explicit: also r_induced_ses (trial average per session; third cohort, one site per run)")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--no_bilin", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    if a.strict:
        assert len(a.fracs) == 1, "--strict needs exactly one calibration fraction (the statistics depend on it)"
    out = os.path.join(a.run, a.out_name or ("readout_strict" if a.strict else
                                             "readout_combo" if a.only_combo else "readout"))
    os.makedirs(out, exist_ok=True)
    log_f = open(os.path.join(out, "readout.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    dev = torch.device(a.device)
    torch.set_grad_enabled(False)
    global _DEV
    _DEV = dev
    seed_runs = ([a.seed_run] if a.seed_run else []) + list(a.seed_runs)
    log(f"config: {vars(a)}")
    results = []
    for ep in sorted(glob.glob(os.path.join(a.run, "enc_sub-*.pt"))):
        sid = os.path.basename(ep)[4:-3]
        if a.subjects and sid not in a.subjects:
            continue
        fp = os.path.join(out, f"readout_{sid}.json")
        sps = [sp for sp in (os.path.join(r_, f"enc_{sid}.pt") for r_ in seed_runs) if os.path.exists(sp)]
        if not os.path.exists(os.path.join(a.data, f"{sid}.pt")):
            log(f"[{sid}] no data file in {a.data} (excluded at this level), skip")
            continue
        if os.path.exists(fp) and not a.overwrite:
            done = json.load(open(fp))
            # resume: keep a finished subject only if it used the same number of encoders as now
            if done.get("n_encoders", 1 + len(sps)) == 1 + len(sps):
                results.append(done)
                log(f"[{sid}] already done, loaded")
                continue
            log(f"[{sid}] done with {done['n_encoders']} encoders, now {1 + len(sps)}: recomputing")
        t0 = time.time()
        enc = load_encoder(ep, dev)
        seed_encs = [load_encoder(sp, dev) for sp in sps]
        if a.untrained:
            enc = TemporalEncoder(**enc.cfg).to(dev).eval()
            seed_encs = [TemporalEncoder(**se.cfg).to(dev).eval() for se in seed_encs]
        E, Y, ses, events = load_subject(os.path.join(a.data, f"{sid}.pt"))
        if a.strict:
            st = torch.load(os.path.join(a.stats_dir or a.data, f"{sid}_normstats.pt"), weights_only=False)
            assert len(st["trial_bounds"]) == len(E) and list(st["trial_session"]) == ses, "stats / data mismatch"
            E, Y = strict_renorm(E, Y, ses, st, max(3, int(round(len(E) * a.fracs[0]))))
        log(f"[{sid}] {len(E)} trials, {E[0].shape[0]} scalp, {Y[0].shape[0]} targets, "
            f"{len(set(ses))} sessions, extra encoders: {len(seed_encs)}")
        a._sid, a._out = sid, out
        r = analyze(E, Y, ses, enc, seed_encs, a, dev, log, events)
        r["subject"] = sid
        with open(fp + ".tmp", "w") as fh:
            json.dump(r, fh)
        os.replace(fp + ".tmp", fp)
        results.append(r)
        log(f"[{sid}] done in {(time.time() - t0) / 60:.1f} min")
    if results:
        summarize(results, out, log)


if __name__ == "__main__":
    main()
