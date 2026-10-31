"""
Phase 29 (post hoc, review C3) - EEG source imaging as a baseline on ds004752 (primary level, strict normalisation,
20 % chronological calibration, the test trials of the main analysis).

Virtual SEEG without any intracranial data of the patient:
  1. template head model: fsaverage 3-layer BEM, volume source space (free orientation) inside the inner skull, the
     scalp channels at their standard 10-05 positions, average reference;
  2. eLORETA inverse operator (ad hoc diagonal noise covariance, lambda^2 = 1/9) -> source currents J(t) = K x(t);
  3. potential of these current dipoles at each SEEG contact in an infinite homogeneous conductor,
     V_c = sum_s p_s . (r_c - r_s) / (4 pi sigma (|r_c - r_s|^2 + a^2)^(3/2)), a = 5 mm (smoothing for sources close
     to a contact); contact MNI coordinates are used as fsaverage coordinates (template anatomy);
  4. bipolar virtual SEEG = V_a - V_b for each target pair: a fixed linear map M_bip of the scalp EEG.
Models scored on the test trials (Pearson r per target, mean over targets):
  zero-shot      M_bip x(t) as is (no calibration at all; the sign is fixed by the physics)
  src-ridge      ridge on the virtual bipolar signals of all targets (lag 0), fitted on the calibration trials
  src+LAG9       ridge on [virtual signals, LAG9 features]
  LAG9 (here)    ridge on LAG9 features with the same code (sanity reference for the registered LAG9 ridge)
Penalties by pooled out-of-fold r over 5 contiguous calibration blocks, as in the main analysis.

python -m s2i.srcimg29 run    --data DATA_NOQC --out RUNS/srcimg29
python -m s2i.srcimg29 report --out RUNS/srcimg29 --runs RUNS
"""
import argparse
import glob
import json
import os
import time
import zlib

import numpy as np

FS = 200
LAGS = (-40, -20, -10, -5, 0, 5, 10, 20, 40)
ALPHAS_LAG = (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000)
ALPHAS_SRC = tuple(10.0 ** k for k in np.arange(-3, 7.5, 0.5))
SIGMA, SMOOTH = 0.33, 0.005


def chrono(n, frac=0.2, gap=1):
    n_c = max(3, int(round(n * frac)))
    return list(range(n_c)), list(range(n_c + gap, n))


def blocks(n, k=5):
    k = max(2, min(k, n))
    e = np.linspace(0, n, k + 1).round().astype(int)
    return [list(range(e[i], e[i + 1])) for i in range(k)]


def lagged(e, lags=LAGS):
    C, L = e.shape
    out = np.zeros((len(lags), C, L), np.float32)
    for j, d in enumerate(lags):
        if d >= 0:
            out[j, :, d:] = e[:, :L - d] if d else e
        else:
            out[j, :, :L + d] = e[:, -d:]
    return out.reshape(len(lags) * C, L)


def r_rows(P, T):
    P = P - P.mean(1, keepdims=True)
    T = T - T.mean(1, keepdims=True)
    return (P * T).sum(1) / (np.sqrt((P * P).sum(1) * (T * T).sum(1)) + 1e-12)


# ----------------------------------------------------------------------------- data
def load(path):
    """-> scalp EEG (microvolts, CAR) and targets per trial, sessions, names, contact MNI (m), bipolar pairs."""
    import torch
    d = torch.load(path, weights_only=False)
    b = d["trial_bounds"].numpy()
    ses = list(d.get("trial_session", ["ses-01"] * len(b)))
    out = {}
    for key in ("eeg_car", "ieeg_bip"):
        x = d[key].float().numpy()
        st = d["norm_stats"][key]
        tr = []
        for (s0, s1), s in zip(b, ses):
            med, mad = np.asarray(st[s]["med"], np.float32)[:, None], np.asarray(st[s]["mad"], np.float32)[:, None]
            tr.append(x[:, s0:s1] * mad + med)
        out[key] = tr
    return (out["eeg_car"], out["ieeg_bip"], ses, list(d["eeg_names"]),
            d["contact_mni"].numpy().astype(np.float64) / 1000.0, [tuple(p) for p in d["bip_pairs"]])


def strict(trials, n_cal):
    x = np.concatenate(trials[:n_cal], 1)
    med = np.median(x, 1, keepdims=True)
    mad = np.median(np.abs(x - med), 1, keepdims=True) * 1.4826
    mad = np.where(mad < 1e-12, 1.0, mad)
    return [((t - med) / mad).astype(np.float32) for t in trials]


# ----------------------------------------------------------------------------- source model
_FS = {}


def fsaverage():
    if not _FS:
        import mne
        d = mne.datasets.fetch_fsaverage(verbose=False)
        sub = os.path.dirname(d)
        bem = os.path.join(d, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
        src = mne.setup_volume_source_space("fsaverage", pos=7.0, subjects_dir=sub,
                                            bem=bem,
                                            verbose=False)
        trans = mne.read_trans(os.path.join(d, "bem", "fsaverage-trans.fif"))
        _FS.update(sub=sub, bem=bem, src=src, trans=trans)
    return _FS


def virtual_map(eeg_names, contact_mri_m, pairs):
    """Linear map M_bip (targets x scalp channels) from average-referenced scalp EEG to bipolar virtual SEEG."""
    import mne
    F = fsaverage()
    info = mne.create_info(eeg_names, FS, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False, on_missing="raise")
    ev = mne.EvokedArray(np.eye(len(eeg_names)) * 1e-6, info, tmin=0.0, verbose=False)
    ev.set_eeg_reference("average", projection=True, verbose=False)
    fwd = mne.make_forward_solution(ev.info, F["trans"], F["src"], F["bem"], eeg=True, meg=False, verbose=False)
    cov = mne.make_ad_hoc_cov(ev.info, verbose=False)
    inv = mne.minimum_norm.make_inverse_operator(ev.info, fwd, cov, loose=1.0, depth=None, verbose=False)
    stc = mne.minimum_norm.apply_inverse(ev, inv, lambda2=1.0 / 9.0, method="eLORETA", pick_ori="vector",
                                         verbose=False)
    K = stc.data / 1e-6                                     # (n_src, 3, n_eeg): source moment per volt of input
    rr = fwd["source_rr"]                                   # head coordinates (m)
    head = mne.transforms.apply_trans(mne.transforms.invert_transform(F["trans"]), contact_mri_m)
    V = np.zeros((len(head), len(eeg_names)))
    for c, rc in enumerate(head):
        dv = rc[None, :] - rr                               # (n_src, 3)
        g = dv / (4 * np.pi * SIGMA * ((dv ** 2).sum(1, keepdims=True) + SMOOTH ** 2) ** 1.5)
        V[c] = np.einsum("sk,ske->e", g, K)
    M = np.stack([V[a] - V[b] for a, b in pairs])
    return M, {"n_sources": int(len(rr)), "inside_brain_contacts": int(len(head))}


# ----------------------------------------------------------------------------- ridge (as in the main analysis)
def ridge_cv(Xtr, Ytr, alphas, folds):
    """Pooled out-of-fold r over contiguous calibration blocks -> alpha; refit on all calibration trials."""
    def fit(X, Y, a):
        mu, sd = X.mean(1, keepdims=True), X.std(1, keepdims=True) + 1e-8
        Z = (X - mu) / sd
        ym = Y.mean(1, keepdims=True)
        G = Z @ Z.T
        lam, Q = np.linalg.eigh((G + G.T) / 2)
        B = Q.T @ (Z @ (Y - ym).T)
        n = Z.shape[1]
        return mu, sd, ym, Q, np.clip(lam, 0, None), B, n

    def pred(m, X, a):
        mu, sd, ym, Q, lam, B, n = m
        W = Q @ (B / (lam + a * n)[:, None])
        return W.T @ ((X - mu) / sd) + ym
    oof = {a: [] for a in alphas}
    tgt = []
    for f in folds:
        tr = [i for g in folds if g is not f for i in g]
        Xa, Ya = np.concatenate([Xtr[i] for i in tr], 1), np.concatenate([Ytr[i] for i in tr], 1)
        m = fit(Xa, Ya, None)
        Xf = np.concatenate([Xtr[i] for i in f], 1)
        for a in alphas:
            oof[a].append(pred(m, Xf, a))
        tgt.append(np.concatenate([Ytr[i] for i in f], 1))
    T = np.concatenate(tgt, 1)
    score = {a: float(np.nanmean(r_rows(np.concatenate(oof[a], 1), T))) for a in alphas}
    best = max(score, key=score.get)
    m = fit(np.concatenate(list(Xtr.values()), 1), np.concatenate(list(Ytr.values()), 1), None)
    return (lambda X: pred(m, X, best)), best, score


def analyze(E_uv, Y_raw, M):
    n = len(E_uv)
    cal, test = chrono(n)
    E = strict(E_uv, len(cal))                       # strict scalp normalisation (for LAG9 features)
    Y = strict(Y_raw, len(cal))
    folds = blocks(len(cal))
    folds = [[cal[i] for i in f] for f in folds]
    V = [(M @ e).astype(np.float32) for e in E_uv]   # virtual bipolar SEEG from the microvolt CAR scalp EEG
    Ttest = np.concatenate([Y[i] for i in test], 1)
    out = {"n_trials": n, "n_cal": len(cal), "n_test": len(test)}
    # zero-shot
    P = np.concatenate([V[i] for i in test], 1)
    rz = r_rows(P, Ttest)
    out["zero_shot"] = {"r": float(np.nanmean(rz)), "frac_positive": float(np.mean(rz > 0)),
                        "abs_r": float(np.nanmean(np.abs(rz))), "contacts": rz.tolist()}
    feats = {"lag9": [lagged(e) for e in E], "src": V,
             "src+lag9": [np.concatenate([v / (np.std(v) + 1e-8), lagged(e)]) for v, e in zip(V, E)]}
    alph = {"lag9": ALPHAS_LAG, "src": ALPHAS_SRC, "src+lag9": ALPHAS_SRC}
    Ytr = {i: Y[i] for i in cal}
    for k, F in feats.items():
        Xtr = {i: F[i] for i in cal}
        f, a, _ = ridge_cv(Xtr, Ytr, alph[k], folds)
        P = np.concatenate([f(F[i]) for i in test], 1)
        rr = r_rows(P, Ttest)
        out[k] = {"r": float(np.nanmean(rr)), "alpha": a, "contacts": rr.tolist()}
    return out


# ----------------------------------------------------------------------------- commands
def run(a):
    os.makedirs(a.out, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(a.data, "sub-*.pt"))):
        if p.endswith("_normstats.pt"):
            continue
        sid = os.path.basename(p)[:-3]
        fo = os.path.join(a.out, f"src_{sid}.json")
        if os.path.exists(fo) and not a.overwrite:
            print(f"[{sid}] done")
            continue
        t0 = time.time()
        E, Y, ses, names, mni, pairs = load(p)
        M, info = virtual_map(names, mni, pairs)
        r = analyze(E, Y, M)
        r.update(subject=sid, n_eeg=len(names), n_targets=len(pairs), **info)
        json.dump(r, open(fo + ".tmp", "w"))
        os.replace(fo + ".tmp", fo)
        print(f"[{sid}] {len(names)} scalp, {len(pairs)} targets | zero-shot r {r['zero_shot']['r']:+.4f} "
              f"(|r| {r['zero_shot']['abs_r']:.4f}, {100 * r['zero_shot']['frac_positive']:.0f} % > 0) | src-ridge "
              f"{r['src']['r']:.4f} | src+LAG9 {r['src+lag9']['r']:.4f} | LAG9 {r['lag9']['r']:.4f} | "
              f"{time.time() - t0:.0f}s", flush=True)


def boot(d, name, n=10000):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(a):
    from scipy.stats import wilcoxon
    R = {json.load(open(f))["subject"]: json.load(open(f)) for f in glob.glob(os.path.join(a.out, "src_sub-*.json"))}
    reg = {}
    for f in glob.glob(os.path.join(a.runs, "shared_bip_car", "readout_combo5_strict_qc_none", "readout_sub-*.json")):
        d = json.load(open(f))
        reg[d["subject"]] = {m: d["explicit"][m]["mean"]["r"] for m in ("base", "combo_nosess")}
    subs = sorted(set(R) & set(reg))
    col = {"zero-shot source imaging": [R[s]["zero_shot"]["r"] for s in subs],
           "source ridge (virtual SEEG, calibrated)": [R[s]["src"]["r"] for s in subs],
           "source + LAG9 ridge": [R[s]["src+lag9"]["r"] for s in subs],
           "LAG9 ridge (this script)": [R[s]["lag9"]["r"] for s in subs],
           "LAG9 ridge (main analysis)": [reg[s]["base"] for s in subs],
           "final model (main analysis)": [reg[s]["combo_nosess"] for s in subs]}
    L = ["# Phase 29 (post hoc): EEG source imaging baseline on ds004752", "",
         f"{len(subs)} patients, primary level, strict normalisation, 20 % calibration, test trials of the main analysis.", "",
         "| model | mean r |", "|---|---|"] + [f"| {k} | {np.mean(v):.4f} |" for k, v in col.items()]
    L += ["", f"Zero-shot: mean |r| {np.mean([R[s]['zero_shot']['abs_r'] for s in subs]):.4f}; contacts with r > 0: "
              f"{100 * np.mean([R[s]['zero_shot']['frac_positive'] for s in subs]):.0f} % (mean over patients).", "",
          "| A | B | mean A − B [95 % CI] | A > B | p (Wilcoxon) |", "|---|---|---|---|---|"]
    for A, B in (("zero-shot source imaging", "LAG9 ridge (main analysis)"),
                 ("source ridge (virtual SEEG, calibrated)", "LAG9 ridge (main analysis)"),
                 ("source + LAG9 ridge", "LAG9 ridge (this script)"),
                 ("final model (main analysis)", "source + LAG9 ridge"),
                 ("LAG9 ridge (this script)", "LAG9 ridge (main analysis)")):
        d = np.array(col[A]) - np.array(col[B])
        lo, hi = boot(d, f"srcimg29 {A} {B}")
        L.append(f"| {A} | {B} | {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | {(d > 0).sum()}/{len(d)} | "
                 f"{wilcoxon(d).pvalue:.2e} |")
    L += ["", "| patient | scalp | targets | zero-shot | |r| | source ridge | source + LAG9 | LAG9 (here) | LAG9 (main) | final |",
          "|" + "---|" * 10]
    for s in subs:
        r = R[s]
        L.append(f"| {s} | {r['n_eeg']} | {r['n_targets']} | {r['zero_shot']['r']:+.3f} | {r['zero_shot']['abs_r']:.3f} | "
                 f"{r['src']['r']:.3f} | {r['src+lag9']['r']:.3f} | {r['lag9']['r']:.3f} | {reg[s]['base']:.3f} | "
                 f"{reg[s]['combo_nosess']:.3f} |")
    md = "\n".join(L)
    print(md)
    open(os.path.join(a.out, "srcimg29.md"), "w", encoding="utf-8").write(md + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("--data", help="run: ds004752 data folder of the primary level (s2i_v1_noqc)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", help="report: run root with the main-analysis readouts")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    {"run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
