"""
Phase 13 - third external review: common mode in VOLTS, and quality-control (QC) statistics.

  volts   The Phase-11 common-mode analysis (review11 refcheck) worked on data normalised per channel and per
          session (median / MAD), where a channel mean is not the physical common mode. Here every signal is
          first returned to filtered volts (x * MAD_session + median_session, statistics from
          <sub>_normstats.pt), the common modes and the "monopolar - common mode" targets are formed in volts,
          and only then is every signal normalised with CALIBRATION-trial statistics (strict protocol).
            iEEG common mode  = mean of all retained monopolar contacts (volts, recording reference)
            scalp common mode = mean of all retained scalp channels (volts, original reference)
          Same lagged ridge, same clean split as review11. iEEG samples clipped at |z| = 10 during preprocessing
          stay clipped (their share is reported by `qc`).
  qc      per patient: trials kept / rejected by the scalp-artefact rule, share of test samples at the iEEG
          clipping bound (bipolar and monopolar), and the raw-volt common-mode diagnostics saved by preprocessing
          (correlation of the raw iEEG mean with each raw scalp channel, first-eigenvalue share of the monopolar
          correlation matrix), all computed on the whole session before any re-referencing.

Usage
-----
python -m s2i.review13 volts --data DATA_DIR --out RUNS_DIR/review13
python -m s2i.review13 qc    --data DATA_DIR --out RUNS_DIR/review13
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

FS = 200
RNG = np.random.default_rng(0)


def boot_ci(d, n=10000):
    d = np.asarray(d, float)
    m = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def wil(a, b):
    from scipy.stats import wilcoxon
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


PATTERN = "sub-[0-9][0-9].pt"                         # ds004752; --pattern for other datasets (e.g. sub-P*.pt)


def subjects(data):
    return sorted(p for p in glob.glob(os.path.join(data, PATTERN)) if not p.endswith("_normstats.pt"))


def load_volts(path):
    """-> trial bounds, trial sessions, {key: volts (C, T)} for eeg_orig / ieeg_mono / ieeg_bip, data dict."""
    d = torch.load(path, weights_only=False)
    sp = path[:-3] + "_normstats.pt"
    st = torch.load(sp, weights_only=False) if os.path.exists(sp) else d
    ns, ses = st["norm_stats"], list(st["trial_session"])
    bounds = d["trial_bounds"].numpy()
    assert len(ses) == len(bounds), "normstats / data mismatch"
    out = {}
    for key in ("eeg_orig", "ieeg_mono", "ieeg_bip"):
        x = d[key].half().float().numpy().astype(np.float64)
        v = np.empty_like(x)
        for (s0, s1), s in zip(bounds, ses):
            med = np.asarray(ns[key][s]["med"], float)[:, None]
            mad = np.asarray(ns[key][s]["mad"], float)[:, None]
            v[:, s0:s1] = x[:, s0:s1] * mad + med
        out[key] = v
    return bounds, ses, out, d


def calib_norm(x, bounds, n_cal):
    cal = np.concatenate([x[:, s0:s1] for s0, s1 in bounds[:n_cal]], 1)
    med = np.median(cal, axis=1, keepdims=True)
    mad = np.median(np.abs(cal - med), axis=1, keepdims=True) * 1.4826
    return ((x - med) / np.where(mad < 1e-30, 1.0, mad)).astype(np.float32)


def volts(a, log):
    from s2i.protocol import ALPHAS, LAG9, run_variant, split
    from s2i.metrics import per_contact_metrics
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for p in subjects(a.data):
        sid = os.path.basename(p)[:-3]
        fp = os.path.join(a.out, f"volts_{sid}.json")
        if os.path.exists(fp):
            res[sid] = json.load(open(fp))
            continue
        bounds, ses, V, _ = load_volts(p)
        if len(bounds) < 20:
            continue
        lens = [int(s1 - s0) for s0, s1 in bounds]
        train, test, folds = split("clean", lens, np.random.default_rng(0))
        n_cal = len(train)
        e, mono, bip = V["eeg_orig"], V["ieeg_mono"], V["ieeg_bip"]
        cm_i, cm_e = mono.mean(0, keepdims=True), e.mean(0, keepdims=True)
        te = np.concatenate([np.arange(bounds[i][0], bounds[i][1]) for i, _, _ in test])
        r_cm = float(np.corrcoef(cm_i[0, te], cm_e[0, te])[0, 1])
        share = float(np.nanmean([np.corrcoef(mono[k, te], cm_i[0, te])[0, 1] ** 2 for k in range(mono.shape[0])]))
        inputs = {"orig": calib_norm(e, bounds, n_cal), "car": calib_norm(e - cm_e, bounds, n_cal)}
        targets = {"mono": calib_norm(mono, bounds, n_cal), "mono_minus_cm": calib_norm(mono - cm_i, bounds, n_cal),
                   "bip": calib_norm(bip, bounds, n_cal), "cm": calib_norm(cm_i, bounds, n_cal)}
        out = {"subject": sid, "r_scalp_cm_vs_ieeg_cm": r_cm, "ieeg_variance_share_of_cm": share, "pred": {}}
        for ref, X in inputs.items():
            E = [X[:, s0:s1] for s0, s1 in bounds]
            for tg, Yall in targets.items():
                Y = [Yall[:, s0:s1] for s0, s1 in bounds]
                P, T, _ = run_variant(LAG9, ALPHAS, E, Y, train, test, folds)
                out["pred"][f"{tg}|{ref}"] = float(np.nanmean(per_contact_metrics(P, T, FS, bands=False)["r"]))
        json.dump(out, open(fp, "w"))
        res[sid] = out
        log(f"[{sid}] volts: scalp-cm vs iEEG-cm r={r_cm:+.3f}, cm share={share:.2f} | " +
            " ".join(f"{k}={v:+.3f}" for k, v in out["pred"].items()))
    subs = sorted(res)
    keys = list(res[subs[0]]["pred"])
    M = {k: np.array([res[s]["pred"][k] for s in subs]) for k in keys}
    old = {}
    for s in subs:                                    # Phase-11 values (normalised domain), if present
        f = os.path.join(a.old, f"refcheck_{s}.json") if a.old else ""
        if f and os.path.exists(f):
            old[s] = json.load(open(f))
    lab = {"mono": "monopolar", "mono_minus_cm": "monopolar − iEEG common mode", "bip": "bipolar",
           "cm": "iEEG common mode itself"}
    L = ["# Common mode in volts (strict normalisation afterwards; lagged ridge, clean split)", "",
         "| target \\ scalp reference | original (volts) | CAR (volts) | original, Phase 11 (normalised domain) |",
         "|---|---|---|---|"]
    for tg in ("mono", "mono_minus_cm", "bip", "cm"):
        o = (np.mean([old[s]["pred"][tg + "|orig"] for s in old]) if len(old) == len(subs) else np.nan)
        L.append(f"| {lab[tg]} | {M[tg + '|orig'].mean():.3f} | {M[tg + '|car'].mean():.3f} | {o:.3f} |")
    d = M["mono|orig"] - M["mono_minus_cm|orig"]
    lo, hi = boot_ci(d)
    L += ["", f"- removing the iEEG common mode in volts (original scalp): {-d.mean():+.3f} [{-hi:+.3f}, {-lo:+.3f}], "
          f"lower in {int((d > 0).sum())}/{len(subs)} patients, p = {wil(M['mono|orig'], M['mono_minus_cm|orig']):.5f}"]
    x = M["mono|orig"] - M["mono|car"] - M["bip|orig"] + M["bip|car"]
    lo, hi = boot_ci(x)
    L += [f"- interaction (mono − bip) × (orig − CAR), volts: {x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], "
          f"positive in {int((x > 0).sum())}/{len(subs)}, p = {wil(x, np.zeros_like(x)):.5f}"]
    if len(old) == len(subs):
        dv = np.array([res[s]["pred"]["mono_minus_cm|orig"] - old[s]["pred"]["mono_minus_cm|orig"] for s in subs])
        lo, hi = boot_ci(dv)
        L += [f"- monopolar − common mode, volts minus normalised-domain: {dv.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"]
    L += ["", "| patient | scalp-cm vs iEEG-cm r (volts) | cm variance share (volts) | mono, orig | mono − cm, orig | "
          "bip, CAR | cm, orig | cm, CAR |", "|---|---|---|---|---|---|---|---|"]
    for s in subs:
        q, p_ = res[s], res[s]["pred"]
        L.append(f"| {s} | {q['r_scalp_cm_vs_ieeg_cm']:+.2f} | {q['ieeg_variance_share_of_cm']:.2f} | "
                 f"{p_['mono|orig']:.3f} | {p_['mono_minus_cm|orig']:.3f} | {p_['bip|car']:.3f} | "
                 f"{p_['cm|orig']:.3f} | {p_['cm|car']:.3f} |")
    out = "\n".join(L)
    log(out)
    open(os.path.join(a.out, "volts.md"), "w", encoding="utf-8").write(out + "\n")


def qc(a, log):
    from s2i.protocol import split
    rows = []
    for p in subjects(a.data):
        d = torch.load(p, weights_only=False)
        b = d["trial_bounds"].numpy()
        lens = [int(s1 - s0) for s0, s1 in b]
        _, test, _ = split("clean", lens, np.random.default_rng(0))
        te = np.concatenate([np.arange(b[i][0], b[i][1]) for i, _, _ in test])
        clip = float(d.get("norm_clip_z", 10.0))
        frac = {}
        for k in ("ieeg_bip", "ieeg_mono"):
            x = d[k].half().float().numpy()[:, te]
            frac[k] = float((np.abs(x) >= clip * 0.999).mean())
        dg = d.get("diagnostics", {}) or {}
        rows.append((os.path.basename(p)[:-3], len(b), int(d.get("n_rejected_trials", -1)), frac["ieeg_bip"],
                     frac["ieeg_mono"], dg.get("mean_abs_corr_iEEGmean_vs_EEG", np.nan),
                     dg.get("max_abs_corr_iEEGmean_vs_EEG", np.nan), dg.get("mono_pc1_ratio", np.nan)))
    kept = sum(r[1] for r in rows)
    rej = sum(max(r[2], 0) for r in rows)
    L = ["# Quality control (thresholds computed from whole-session median / MAD)", "",
         f"- trials kept {kept}, rejected by the scalp-artefact rule (|z| > 15, CAR EEG) {rej} "
         f"({100 * rej / max(kept + rej, 1):.2f} %); trials marked as artefacts in the dataset are excluded before this",
         f"- test samples at the iEEG clipping bound (|z| = 10): bipolar median "
         f"{100 * np.median([r[3] for r in rows]):.3f} % (max {100 * max(r[3] for r in rows):.3f} %), monopolar median "
         f"{100 * np.median([r[4] for r in rows]):.3f} % (max {100 * max(r[4] for r in rows):.3f} %)", "",
         "| patient | trials kept | rejected | test bip clipped % | test mono clipped % | raw iEEG-mean vs scalp, mean |r| | "
         "max |r| | mono PC1 share |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {s} | {k} | {r} | {100 * fb:.3f} | {100 * fm:.3f} | {m1:.2f} | {m2:.2f} | {pc:.2f} |"
          for s, k, r, fb, fm, m1, m2, pc in rows]
    out = "\n".join(L)
    log(out)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "qc.md"), "w", encoding="utf-8").write(out + "\n")


def main():
    global PATTERN
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("volts", "qc"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old", default=None, help="folder with Phase-11 refcheck_<sub>.json (for the comparison column)")
    ap.add_argument("--pattern", default=PATTERN, help="subject file pattern inside --data")
    a = ap.parse_args()
    PATTERN = a.pattern
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    {"volts": volts, "qc": qc}[a.cmd](a, print)


if __name__ == "__main__":
    main()
