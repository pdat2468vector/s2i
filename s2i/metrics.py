"""
Evaluation metrics, computed per contact on the concatenation of all test trials.

Differences vs. old compute_metrics():
* R^2 is NOT clipped at 0 (clipping max(R^2, 0) hid failures and inflated means).
* Band-limited waveform r and band envelope r are reported, so we can see in
  which frequency range information actually exists.
* Filtering is done per trial (no filter ringing across trial boundaries).
"""
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 70)}


def _r_rows(P, T):
    P = P - P.mean(1, keepdims=True)
    T = T - T.mean(1, keepdims=True)
    den = np.sqrt((P * P).sum(1) * (T * T).sum(1)) + 1e-12
    return (P * T).sum(1) / den


def _r2_rows(P, T):
    ss_res = ((T - P) ** 2).sum(1)
    ss_tot = ((T - T.mean(1, keepdims=True)) ** 2).sum(1) + 1e-12
    return 1.0 - ss_res / ss_tot


def per_contact_metrics(preds, tgts, fs=200, bands=True):
    """preds/tgts: lists of (K, L_trial) arrays. Returns dict of (K,) arrays."""
    P = np.concatenate(preds, 1).astype(np.float64)
    T = np.concatenate(tgts, 1).astype(np.float64)
    out = {"r": _r_rows(P, T), "r2": _r2_rows(P, T),
           "rmse": np.sqrt(((P - T) ** 2).mean(1))}
    if bands:
        for name, (lo, hi) in BANDS.items():
            sos = butter(4, [lo, min(hi, fs / 2 - 1)], btype="band", fs=fs, output="sos")
            pb, tb, pe, te = [], [], [], []
            for p, t in zip(preds, tgts):
                if p.shape[1] < 3 * 2 * len(sos) * 3:
                    continue
                fp, ft = sosfiltfilt(sos, p, axis=1), sosfiltfilt(sos, t, axis=1)
                pb.append(fp)
                tb.append(ft)
                pe.append(np.abs(hilbert(fp, axis=1)))
                te.append(np.abs(hilbert(ft, axis=1)))
            if pb:
                out[f"r_{name}"] = _r_rows(np.concatenate(pb, 1), np.concatenate(tb, 1))
                out[f"env_{name}"] = _r_rows(np.concatenate(pe, 1), np.concatenate(te, 1))
    return out


def summarize(m):
    return {k: float(np.nanmean(v)) for k, v in m.items()}
