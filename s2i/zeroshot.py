"""
Phase 6 (CPU) - calibration-free (zero-shot) prediction of intracranial BAND POWER.

Why: waveforms do not transfer across patients (Phase 1a): a bipolar channel flips sign with the
shaft orientation, which differs between patients. Band power is sign-invariant, and region-level
theta power transferred across patients in Subramanian et al. 2025 (hd-EEG, R^2 0.10-0.15).
Question: with the sparse 10-20 montages of ds004752, can the power time course of a contact in a NEW
patient be predicted with NO intracranial data from that patient?

Signals: 1-s non-overlapping windows inside each trial; log band power (delta 1-4, theta 4-8,
alpha 8-13, beta 13-30 Hz) of every bipolar contact (target) and every scalp channel (input,
z-scored per session using EEG only).

Models (leave-one-subject-out; test windows = trials after the 20 % split, as in all phases)
  zs_task     task timing only: mean z-power at the same window position in the trial, from the other
              patients (no EEG) -> how much of the power course is simply stimulus-locked
  zs_global   mean scalp power of the same band (no training)
  zs_nearest  power of the scalp channel nearest to the contact (no training)
  zs_geo      learned geometry readout: scalp power of every band averaged with Gaussian distance
              kernels (30..140 mm) around the contact + the global mean; one linear model per target
              band fitted on the OTHER patients (penalty by leave-one-patient-out inside them),
              applied unchanged to the new patient
  zs_geo_shuf CONTROL (anatomy mismatch): zs_geo with the test patient's contact positions permuted
              among its contacts -> if it equals zs_geo, the geometry contributes nothing
  cal_ridge   reference WITH calibration: ridge per contact on all scalp channels x bands, fitted on the
              first 20 % of the patient's own trials (penalty by 5-fold r)
  zs_cal      hybrid: per contact, ridge on [zs_geo prediction, all scalp channels x bands] fitted on the same
              calibration windows; the zs_geo term and the intercept are NOT penalised, so a large penalty
              falls back to (rescaled) zs_geo and calibration only adds a shrunk correction
Metrics per contact on the test windows: r, and r_induced (each side minus its own mean at the same
window position across the test trials, i.e. beyond the task-locked part).

Note: scalp positions are the MNE standard 10-20 template (head frame ~ fsaverage/MNI305), contacts are
in MNI; the frames agree only approximately, which the distance kernels partly absorb.

Usage
-----
python -m s2i.zeroshot --data DATA_DIR --out RUNS_DIR/zeroshot_power
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

FS, WIN, EDGE = 200, 200, 200          # 1-s windows; 1 s dropped at each trial edge
PBANDS = (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30))
_PS = [butter(4, (lo, hi), btype="band", fs=FS, output="sos") for _, lo, hi in PBANDS]
CENTERS, WIDTH = np.array([30.0, 50.0, 70.0, 90.0, 110.0, 140.0]), 15.0
ALPHAS_ZS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
MODELS = ("zs_task", "zs_global", "zs_nearest", "zs_geo", "zs_geo_shuf", "cal_ridge", "zs_cal")


# ----------------------------------------------------------------------------- data
def logpow(x):
    """
    (channels, L) one trial -> (n_windows, channels, n_bands) log mean power per 1-s window.
    The first and last EDGE samples are dropped: per-trial band-pass filtering rings at the trial edges
    in BOTH modalities, which creates spurious, position-locked power correlation (synthetic check:
    delta r ~0.5 with no coupling at all).
    """
    nw = (x.shape[1] - 2 * EDGE) // WIN
    if nw <= 0:
        return np.zeros((0, x.shape[0], len(_PS)))
    out = []
    for sos in _PS:
        b = sosfiltfilt(sos, x, axis=1)[:, EDGE:EDGE + nw * WIN]
        out.append(np.log((b.reshape(x.shape[0], nw, WIN) ** 2).mean(2) + 1e-6))
    return np.stack(out, -1).transpose(1, 0, 2)


def load(path):
    d = torch.load(path, weights_only=False)
    eeg, tgt = d["eeg_car"].half().float().numpy(), d["ieeg_bip"].half().float().numpy()
    bounds = d["trial_bounds"].numpy()
    ses = list(d.get("trial_session", ["ses-01"] * len(bounds)))
    X, Y, tr, pos, ss = [], [], [], [], []
    for i, (s0, s1) in enumerate(bounds):
        x, y = logpow(eeg[:, s0:s1]), logpow(tgt[:, s0:s1])
        X.append(x)
        Y.append(y)
        tr += [i] * len(x)
        pos += list(range(len(x)))
        ss += [ses[i]] * len(x)
    X, Y, ss = np.concatenate(X), np.concatenate(Y), np.array(ss)
    for s in np.unique(ss):                                   # scalp: z-score per session (EEG only)
        m = ss == s
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-6)
    return {"sid": d["subject"], "X": X.astype(np.float32), "Y": Y.astype(np.float32), "trial": np.array(tr),
            "pos": np.array(pos), "n_trials": len(bounds), "eeg_pos": np.asarray(d["eeg_pos"], float),
            "con_pos": np.asarray(d["bip_mni"], float), "depth": list(d["bip_depth"]),
            "region": list(d["bip_region"])}


def zscore(Y):
    return (Y - Y.mean(0)) / (Y.std(0) + 1e-6)


def geo_features(S):
    """(n_windows, K, (J + 1) * n_bands): scalp power averaged with distance kernels around each contact."""
    d = np.linalg.norm(S["con_pos"][:, None] - S["eeg_pos"][None], axis=2)          # (K, C)
    phi = np.exp(-(d[:, None, :] - CENTERS[None, :, None]) ** 2 / (2 * WIDTH ** 2))   # (K, J, C)
    phi = np.concatenate([phi, np.ones((len(d), 1, d.shape[1]))], 1)                 # + global mean
    tot = phi.sum(2, keepdims=True)
    phi = np.where(tot > 1e-3, phi / np.maximum(tot, 1e-12), 0.0)                    # empty ring -> 0
    F = np.einsum("kjc,wcb->wkjb", phi, S["X"])
    return F.reshape(F.shape[0], F.shape[1], -1).astype(np.float32), d


# ----------------------------------------------------------------------------- metrics
def r_cols(P, T):
    P, T = P - P.mean(0), T - T.mean(0)
    den = np.sqrt((P * P).sum(0) * (T * T).sum(0))
    return np.where(den > 0, (P * T).sum(0) / np.maximum(den, 1e-12), np.nan)


def remove_position_mean(Z, pos):
    out = Z.copy()
    for p in np.unique(pos):
        m = pos == p
        out[m] -= Z[m].mean(0)
    return out


def ridge_fit(A, B, alpha, free=None):
    """Ridge with penalty alpha * mean diagonal; `free` = indices of unpenalised columns."""
    D = len(A)
    pen = np.ones(D)
    if free is not None:
        pen[free] = 0.0
    return np.linalg.solve(A + alpha * np.trace(A) / D * np.diag(pen) + 1e-9 * np.eye(D), B)


# ----------------------------------------------------------------------------- LOSO
def run_fold(t, subs, feats, log):
    S = subs[t]
    n = S["n_trials"]
    n_c = max(3, int(round(n * 0.2)))
    cal, test = S["trial"] < n_c, S["trial"] >= n_c + 1
    others = [s for s in range(len(subs)) if s != t]
    nB = len(PBANDS)
    Yt = S["Y"][test]                                                 # (w, K, nB)
    res = {"subject": S["sid"], "n_test_windows": int(test.sum()), "bands": {}}

    # zero-shot geometry readout: normal equations per training patient and band
    stats = {}
    for s in others:
        F = feats[s][0]
        Fi = np.concatenate([F, np.ones(F.shape[:2] + (1,), np.float32)], 2)         # + intercept
        Yz = zscore(subs[s]["Y"])
        A = np.einsum("wkd,wke->de", Fi, Fi)
        B = np.einsum("wkd,wkb->db", Fi, Yz)
        stats[s] = (A, B, Fi, Yz)
    # penalty by leave-one-patient-out among the training patients (mean r over contacts)
    A_all, B_all = sum(stats[s][0] for s in others), sum(stats[s][1] for s in others)
    score = np.zeros((len(ALPHAS_ZS), nB))
    for s in others:
        A, B, Fi, Yz = stats[s]
        for ai, al in enumerate(ALPHAS_ZS):
            th = ridge_fit(A_all - A, B_all - B, al)
            P = np.einsum("wkd,db->wkb", Fi, th)
            for b in range(nB):
                score[ai, b] += np.nanmean(r_cols(P[:, :, b], Yz[:, :, b]))
    best = score.argmax(0)
    Ft = feats[t][0][test]
    Fti = np.concatenate([Ft, np.ones(Ft.shape[:2] + (1,), np.float32)], 2)

    # task timing template from the other patients (z-power by window position)
    tmpl = {}
    for s in others:
        Yz, pos = zscore(subs[s]["Y"]).mean(1), subs[s]["pos"]                     # (w, nB)
        for p in np.unique(pos):
            tmpl.setdefault(p, []).append(Yz[pos == p].mean(0))
    tmpl = {p: np.mean(v, 0) for p, v in tmpl.items()}
    pos_t = S["pos"][test]

    # calibrated reference: ridge on all scalp channels x bands, the patient's first 20 % of trials
    Xc = S["X"][cal].reshape(cal.sum(), -1)
    Xt = S["X"][test].reshape(test.sum(), -1)
    Yc = S["Y"][cal]
    blocks = np.array_split(np.arange(n_c), 5)
    tr_c = S["trial"][cal]
    d_near = feats[t][1].argmin(1)
    perm = np.random.default_rng(0).permutation(len(S["con_pos"]))
    F_sh = geo_features(dict(S, con_pos=S["con_pos"][perm]))[0][test]
    F_sh = np.concatenate([F_sh, np.ones(F_sh.shape[:2] + (1,), np.float32)], 2)
    Fci = np.concatenate([feats[t][0][cal], np.ones((cal.sum(), feats[t][0].shape[1], 1), np.float32)], 2)

    for b, (bname, _, _) in enumerate(PBANDS):
        th = ridge_fit(A_all, B_all, ALPHAS_ZS[best[b]])[:, b]
        preds = {
            "zs_task": np.repeat(np.stack([tmpl.get(p, np.zeros(nB))[b] for p in pos_t])[:, None], Yt.shape[1], 1),
            "zs_global": np.repeat(S["X"][test][:, :, b].mean(1, keepdims=True), Yt.shape[1], 1),
            "zs_nearest": S["X"][test][:, d_near, b],
            "zs_geo": Fti @ th,
            "zs_geo_shuf": F_sh @ th,
        }
        # calibrated ridge (shared penalty by 5-fold r over trial blocks)
        sc = []
        for al in ALPHAS_ZS:
            rr = []
            for blk in blocks:
                ho = np.isin(tr_c, blk)
                Xa = np.c_[Xc[~ho], np.ones((~ho).sum())]
                W = ridge_fit(Xa.T @ Xa, Xa.T @ Yc[~ho][:, :, b], al)
                rr.append(r_cols(np.c_[Xc[ho], np.ones(ho.sum())] @ W, Yc[ho][:, :, b]))
            sc.append(np.nanmean(rr))
        Xa = np.c_[Xc, np.ones(len(Xc))]
        W = ridge_fit(Xa.T @ Xa, Xa.T @ Yc[:, :, b], ALPHAS_ZS[int(np.nanargmax(sc))])
        preds["cal_ridge"] = np.c_[Xt, np.ones(len(Xt))] @ W

        # hybrid: zero-shot prediction as an extra (standardised) feature of a per-contact calibrated ridge
        zc, zt = Fci @ th, preds["zs_geo"]                                  # (w_cal, K), (w_test, K)
        mu_, sd_ = zc.mean(0), zc.std(0) + 1e-6
        zc, zt = (zc - mu_) / sd_, (zt - mu_) / sd_
        sc = np.zeros(len(ALPHAS_ZS))
        for ai, al in enumerate(ALPHAS_ZS):
            rr = []
            for blk in blocks:
                ho = np.isin(tr_c, blk)
                P = np.zeros((ho.sum(), zc.shape[1]))
                for k in range(zc.shape[1]):
                    Xa = np.c_[zc[~ho, k], Xc[~ho], np.ones((~ho).sum())]
                    w = ridge_fit(Xa.T @ Xa, Xa.T @ Yc[~ho][:, k, b], al, free=[0, Xa.shape[1] - 1])
                    P[:, k] = np.c_[zc[ho, k], Xc[ho], np.ones(ho.sum())] @ w
                rr.append(r_cols(P, Yc[ho][:, :, b]))
            sc[ai] = np.nanmean(rr)
        al = ALPHAS_ZS[int(np.nanargmax(sc))]
        P = np.zeros_like(zt)
        for k in range(zc.shape[1]):
            Xa = np.c_[zc[:, k], Xc, np.ones(len(Xc))]
            w = ridge_fit(Xa.T @ Xa, Xa.T @ Yc[:, k, b], al, free=[0, Xa.shape[1] - 1])
            P[:, k] = np.c_[zt[:, k], Xt, np.ones(len(Xt))] @ w
        preds["zs_cal"] = P

        T = Yt[:, :, b]
        Ti = remove_position_mean(T, pos_t)
        out = {}
        for m, P in preds.items():
            r = r_cols(P, T)
            ri = r_cols(remove_position_mean(P, pos_t), Ti)
            out[m] = {"r": float(np.nanmean(r)), "r_induced": float(np.nanmean(ri)),
                      "contacts": [float(x) for x in r], "contacts_induced": [float(x) for x in ri]}
        out["zs_geo"]["alpha"] = ALPHAS_ZS[best[b]]
        out["zs_geo"]["weights"] = [float(x) for x in th]
        res["bands"][bname] = out
        log(f"    {bname:5s} " + " ".join(f"{m}={v['r']:+.3f}/{v['r_induced']:+.3f}" for m, v in out.items()))
    res["depth"], res["region"] = S["depth"], S["region"]
    return res


def summarize(results, out_dir, log):
    from scipy.stats import wilcoxon

    def wp(x, y=None):
        try:
            return float(wilcoxon(x, y).pvalue if y is not None else wilcoxon(x).pvalue)
        except ValueError:
            return float("nan")
    lines, summ = [], {}
    for metric in ("r", "r_induced"):
        lines += [f"\n### metric: {metric}  (mean over contacts, then over {len(results)} patients; "
                  "p = Wilcoxon over patients)",
                  "| band | " + " | ".join(MODELS) + " | zs_geo vs 0: p | zs_geo − zs_nearest: wins, p | "
                  "zs_geo − zs_task: wins, p | zs_geo − zs_geo_shuf: wins, p | zs_cal − cal_ridge: wins, p |",
                  "|---" * (len(MODELS) + 6) + "|"]
        for bname, _, _ in PBANDS:
            M = {m: np.array([r["bands"][bname][m][metric] for r in results]) for m in MODELS}
            g, nn, tk = M["zs_geo"], M["zs_nearest"], M["zs_task"]
            fin = np.isfinite(tk)
            summ.setdefault(metric, {})[bname] = {m: float(np.nanmean(v)) for m, v in M.items()}
            lines.append(f"| {bname} | " + " | ".join(f"{np.nanmean(M[m]):+.3f}" for m in MODELS) +
                         f" | {wp(g):.4f} | {(g > nn).sum()}/{len(g)}, {wp(g, nn):.4f} | " +
                         (f"{(g[fin] > tk[fin]).sum()}/{fin.sum()}, {wp(g[fin], tk[fin]):.4f} |" if fin.sum() >= 5
                          else "- |") +
                         f" {(g > M['zs_geo_shuf']).sum()}/{len(g)}, {wp(g, M['zs_geo_shuf']):.4f} |"
                         f" {(M['zs_cal'] > M['cal_ridge']).sum()}/{len(g)}, {wp(M['zs_cal'], M['cal_ridge']):.4f} |")
    # by depth (pooled contacts) for zs_geo and cal_ridge
    lines += ["\n### zs_geo and cal_ridge by depth (pooled contacts, r / r_induced)",
              "| band | depth | n | zs_geo r | zs_geo r_ind | cal_ridge r | cal_ridge r_ind |", "|---|---|---|---|---|---|---|"]
    for bname, _, _ in PBANDS:
        pools = {}
        for r in results:
            B = r["bands"][bname]
            for k, dep in enumerate(r["depth"]):
                pools.setdefault(dep, []).append([B["zs_geo"]["contacts"][k], B["zs_geo"]["contacts_induced"][k],
                                                  B["cal_ridge"]["contacts"][k], B["cal_ridge"]["contacts_induced"][k]])
        for dep, v in sorted(pools.items()):
            v = np.array(v, float)
            lines.append(f"| {bname} | {dep} | {len(v)} | " + " | ".join(f"{x:+.3f}" for x in np.nanmean(v, 0)) + " |")
    md = "\n".join(lines)
    log(md)
    with open(os.path.join(out_dir, "zeroshot.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(out_dir, "zeroshot.json"), "w") as fh:
        json.dump(summ, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_trials", type=int, default=20)
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "zeroshot.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"config: {vars(a)}")
    subs = []
    for p in sorted(glob.glob(os.path.join(a.data, "sub-[0-9][0-9].pt"))):
        S = load(p)
        if S["n_trials"] < a.min_trials or S["Y"].shape[1] == 0:
            continue
        subs.append(S)
        log(f"{S['sid']}: {len(S['X'])} windows, {S['X'].shape[1]} scalp, {S['Y'].shape[1]} contacts")
    feats = [geo_features(S) for S in subs]
    results = []
    for t in range(len(subs)):
        log(f"[{subs[t]['sid']}] test (r / r_induced)")
        results.append(run_fold(t, subs, feats, log))
    with open(os.path.join(a.out, "zeroshot_subjects.json"), "w") as fh:
        json.dump(results, fh)
    summarize(results, a.out, log)


if __name__ == "__main__":
    main()
