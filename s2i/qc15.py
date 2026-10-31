"""
Phase 15 - sensitivity of every main result to the quality-control (QC) rule.

Input: ds004752 preprocessed WITHOUT automatic QC (preprocess_ds004752 --reject_z inf --clip_z inf): only trials
flagged as artefacts in the dataset are excluded, targets are not clipped. From it, three data folders are made:

  none     no automatic QC (the input itself; only the trial table is written)
  session  the original rule: drop a trial if |z| of CAR scalp EEG > 15, clip targets at |z| = 10, both z computed
           with WHOLE-SESSION median / MAD (reproduces the main analysis)
  calib    the same thresholds, but z computed with the median / MAD of the CALIBRATION candidates only (first 20 %
           of the candidate trials, chronological): no statistic of later trials decides which trials are kept or
           how targets are clipped

Every folder holds <sub>.pt and <sub>_normstats.pt in the format of s2i_v1 (session-normalised float16), so
readout.py --strict runs on it unchanged. A per-patient table reports candidate / kept trials, calibration and test
trials and the real calibration minutes after QC (calibration = first 20 % of the KEPT trials, as in all analyses).

Usage
-----
python -m s2i.qc15 --src DATA_ROOT/s2i_v1_noqc --out_root DATA_ROOT
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

FS = 200
REJECT_Z, CLIP_Z, CAL_FRAC = 15.0, 10.0, 0.2
KEYS = ("eeg_orig", "eeg_car", "ieeg_mono", "ieeg_bip")


def n_cal(n):
    return max(3, int(round(n * CAL_FRAC)))


def robust(x):
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True) * 1.4826
    return med, np.where(mad < 1e-30, 1.0, mad)


def session_stats(d, key, ses):
    ns = d["norm_stats"][key][ses]
    return np.asarray(ns["med"], float)[:, None], np.asarray(ns["mad"], float)[:, None]


def to_volts(d, key, x, ses):
    med, mad = session_stats(d, key, ses)
    return x * mad + med


def to_session(d, key, v, ses):
    med, mad = session_stats(d, key, ses)
    return (v - med) / mad


def variant(d, mode):
    b = d["trial_bounds"].numpy()
    ses = list(d["trial_session"])
    X = {k: d[k].float().numpy() for k in KEYS}
    trials = [{k: X[k][:, s0:s1].astype(np.float64) for k in KEYS} for s0, s1 in b]
    if mode == "session":
        keep = [np.abs(t["eeg_car"]).max() <= REJECT_Z for t in trials]
        for t in trials:
            for k in ("ieeg_mono", "ieeg_bip"):
                t[k] = np.clip(t[k], -CLIP_Z, CLIP_Z)
    elif mode == "calib":
        nc = n_cal(len(trials))
        stats = {}
        for k in ("eeg_car", "ieeg_mono", "ieeg_bip"):
            v = np.concatenate([to_volts(d, k, trials[i][k], ses[i]) for i in range(nc)], 1)
            stats[k] = robust(v)
        keep = []
        for i, t in enumerate(trials):
            med, mad = stats["eeg_car"]
            z = (to_volts(d, "eeg_car", t["eeg_car"], ses[i]) - med) / mad
            keep.append(np.abs(z).max() <= REJECT_Z)
            for k in ("ieeg_mono", "ieeg_bip"):
                med, mad = stats[k]
                v = np.clip(to_volts(d, k, t[k], ses[i]), med - CLIP_Z * mad, med + CLIP_Z * mad)
                t[k] = to_session(d, k, v, ses[i])
    else:
        keep = [True] * len(trials)
    idx = [i for i, k in enumerate(keep) if k]
    out = dict(d)
    bounds, c = [], 0
    for i in idx:
        L = b[i][1] - b[i][0]
        bounds.append((c, c + L))
        c += L
    for k in KEYS:
        out[k] = torch.from_numpy(np.concatenate([trials[i][k] for i in idx], 1).astype(np.float16))
    out["trial_bounds"] = torch.tensor(bounds, dtype=torch.long)
    out["trial_session"] = [ses[i] for i in idx]
    if "trial_id" in d:
        out["trial_id"] = [d["trial_id"][i] for i in idx]
    out["qc_mode"], out["n_candidate_trials"] = mode, len(trials)
    out["norm_clip_z"] = CLIP_Z if mode != "none" else float("inf")
    return out, len(trials)


def table_row(sid, d, n_cand):
    b = d["trial_bounds"].numpy()
    n = len(b)
    nc = n_cal(n)
    lens = (b[:, 1] - b[:, 0]) / FS / 60
    return (sid, n_cand, n, 100 * (1 - n / max(n_cand, 1)), nc, max(n - nc - 1, 0), lens[:nc].sum(), lens[nc + 1:].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="data folder preprocessed with --reject_z inf --clip_z inf")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--modes", nargs="*", default=["none", "session", "calib"])
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    rows = {m: [] for m in a.modes}
    for p in sorted(glob.glob(os.path.join(a.src, "sub-[0-9][0-9].pt"))):
        sid = os.path.basename(p)[:-3]
        d = torch.load(p, weights_only=False)
        for m in a.modes:
            if m == "none":
                rows[m].append(table_row(sid, d, len(d["trial_bounds"])))
                continue
            od = os.path.join(a.out_root, f"s2i_qc_{m}")
            os.makedirs(od, exist_ok=True)
            fp = os.path.join(od, f"{sid}.pt")
            if os.path.exists(fp):
                v = torch.load(fp, weights_only=False)
                n_cand = v["n_candidate_trials"]
            else:
                v, n_cand = variant(d, m)
                torch.save(v, fp + ".tmp")
                os.replace(fp + ".tmp", fp)
                torch.save({k: v[k] for k in ("subject", "norm_stats", "norm_clip_z", "trial_bounds", "trial_session")},
                           os.path.join(od, f"{sid}_normstats.pt"))
            rows[m].append(table_row(sid, v, n_cand))
        print(f"[{sid}] " + " | ".join(f"{m}: {rows[m][-1][2]} kept" for m in a.modes), flush=True)
    L = ["# Trials and calibration time after each QC rule", ""]
    for m in a.modes:
        r = rows[m]
        L += [f"## {m}", f"- kept {sum(x[2] for x in r)} of {sum(x[1] for x in r)} candidate trials; "
              f"mean calibration {np.mean([x[6] for x in r]):.1f} min, mean test {np.mean([x[7] for x in r]):.1f} min", "",
              "| patient | candidates | kept | rejected % | calib trials | test trials | calib min | test min |",
              "|---|---|---|---|---|---|---|---|"]
        L += [f"| {s} | {c} | {k} | {rj:.1f} | {nc} | {nt} | {cm:.2f} | {tm:.1f} |" for s, c, k, rj, nc, nt, cm, tm in r]
        L.append("")
    out = "\n".join(L)
    print(out)
    open(os.path.join(a.out_root, "qc15_trials.md"), "w", encoding="utf-8").write(out + "\n")


if __name__ == "__main__":
    main()
