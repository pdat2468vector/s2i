"""
Phase 30 (post hoc, review M7) - does the predicted SEEG reproduce the working-memory load effect of the recorded SEEG?

ds004752, primary level, strict normalisation, 20 % calibration, the test trials of the main analysis. For every
bipolar target and test trial: log band power (theta 4-8 Hz, alpha 8-12 Hz) in the last 2 s of the maintenance period
(4-6 s after trial onset; Dimakopoulos et al. 2022) of the recording, the final model and the LAG9 ridge (saved by
s2i.readout --save_pred). Set size (4, 6 or 8 letters) per trial from the BIDS events files.

Per target: load effect = Spearman rho between set size and log power across test trials (recording, final, LAG9).
  agreement   per patient, Pearson r across targets between the recorded and the predicted load effects
  detection   among targets with a recorded load effect (p < 0.05, Benjamini-Hochberg within patient), the share whose
              prediction shows an effect of the same sign with p < 0.05
  hippocampus mean load effect over hippocampal targets (recording vs predictions)

python -m s2i.load30 --pred RUNS/shared_bip_car/readout_combo5_strict_qc_none_pred --data DATA_NOQC --out RUNS/load30
"""
import argparse
import glob
import json
import os
import urllib.request
import zlib

import numpy as np

S3 = "https://s3.amazonaws.com/openneuro.org/ds004752"


def events(sid, ses, cache):
    os.makedirs(cache, exist_ok=True)
    fn = f"{sid}_{ses}_task-verbalWM_run-01_events.tsv"
    p = os.path.join(cache, fn)
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"{S3}/{sid}/{ses}/ieeg/{fn}", p)
    lines = open(p, encoding="utf-8").read().splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]
    return {int(float(r["nTrial"])): int(float(r["SetSize"])) for r in rows}


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def boot(v, name, n=10000):
    v = np.asarray(v, float)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    import torch
    from scipy.stats import spearmanr, wilcoxon
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for f in sorted(glob.glob(os.path.join(a.pred, "pred_sub-*.npz"))):
        sid = os.path.basename(f)[5:-4]
        z = np.load(f)
        if "bp_theta_target" not in z:
            raise SystemExit(f"{f} has no band power: rerun s2i.readout --save_pred with this version")
        d = torch.load(os.path.join(a.data, f"{sid}.pt"), weights_only=False)
        ses, tid, cat = list(d["trial_session"]), list(d["trial_id"]), list(d["bip_cat"])
        ev = {s: events(sid, s, os.path.join(a.out, "events")) for s in sorted(set(ses))}
        test = z["test"].astype(int)
        load = np.array([ev[ses[i]].get(int(tid[i]), np.nan) for i in test], float)
        R = {"n_test": int(len(test)), "set_sizes": {str(k): int((load == k).sum()) for k in (4, 6, 8)},
             "categories": cat}
        for band in ("theta", "alpha"):
            for key in ("target", "final", "base"):
                X = z[f"bp_{band}_{key}"]
                rho, p = [], []
                for k in range(X.shape[1]):
                    ok = np.isfinite(X[:, k]) & np.isfinite(load)
                    if ok.sum() < 10 or np.ptp(X[ok, k]) == 0:
                        rho.append(np.nan); p.append(np.nan); continue
                    r_, p_ = spearmanr(load[ok], X[ok, k])
                    rho.append(float(r_)); p.append(float(p_))
                R[f"{band}_{key}_rho"], R[f"{band}_{key}_p"] = rho, p
        res[sid] = R
        print(f"[{sid}] {R['n_test']} test trials {R['set_sizes']}", flush=True)
    json.dump(res, open(os.path.join(a.out, "load30.json"), "w"))

    L = ["# Phase 30 (post hoc): working-memory load effect in recorded and predicted SEEG (ds004752)", "",
         f"{len(res)} patients; load effect = Spearman rho(set size, log band power, last 2 s of maintenance) per "
         "target across test trials.", ""]
    for band in ("theta", "alpha"):
        agree, det, n_sig = {"final": [], "base": []}, {"final": [], "base": []}, []
        hip = {"target": [], "final": [], "base": []}
        allrec = []
        for sid, R in res.items():
            rt = np.array(R[f"{band}_target_rho"], float)
            pt = np.array(R[f"{band}_target_p"], float)
            ok = np.isfinite(rt)
            allrec += list(rt[ok])
            q = np.full(len(pt), np.nan)
            q[ok] = bh(pt[ok])
            sig = ok & (q < 0.05)
            n_sig.append(int(sig.sum()))
            hipm = np.array([c == "Hippocampus" for c in R["categories"]]) & ok
            if hipm.any():
                hip["target"].append(float(np.nanmean(rt[hipm])))
            for key in ("final", "base"):
                rp = np.array(R[f"{band}_{key}_rho"], float)
                pp = np.array(R[f"{band}_{key}_p"], float)
                m = ok & np.isfinite(rp)
                agree[key].append(float(np.corrcoef(rt[m], rp[m])[0, 1]) if m.sum() > 3 else np.nan)
                if sig.any():
                    s2 = sig & np.isfinite(rp)
                    det[key].append(float(np.mean((np.sign(rp[s2]) == np.sign(rt[s2])) & (pp[s2] < 0.05))))
                if hipm.any():
                    hip[key].append(float(np.nanmean(rp[hipm & np.isfinite(rp)])))
        L += [f"## {band}", "",
              f"- recorded load effects: mean rho {np.mean(allrec):+.3f}; targets with a recorded effect (BH q < 0.05 within "
              f"patient): {sum(n_sig)} of {len(allrec)} in {sum(1 for n in n_sig if n)} patients",
              f"- positive recorded rho: {100 * np.mean(np.array(allrec) > 0):.0f} % of targets"]
        for key, lab in (("final", "final model"), ("base", "LAG9 ridge")):
            v = np.array(agree[key], float)
            v = v[np.isfinite(v)]
            lo, hi = boot(v, f"load30 {band} agree {key}")
            L.append(f"- agreement of predicted with recorded load effects ({lab}): mean r across targets {v.mean():+.3f} "
                     f"[{lo:+.3f}, {hi:+.3f}] over {len(v)} patients, positive in {(v > 0).sum()}/{len(v)}")
        fa, ba = np.array(agree["final"]), np.array(agree["base"])
        m = np.isfinite(fa) & np.isfinite(ba)
        d = fa[m] - ba[m]
        lo, hi = boot(d, f"load30 {band} agree final-base")
        L.append(f"- agreement, final model − LAG9 ridge: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], {(d > 0).sum()}/{len(d)}, "
                 f"p = {wilcoxon(d).pvalue:.3f}")
        for key, lab in (("final", "final model"), ("base", "LAG9 ridge")):
            if det[key]:
                L.append(f"- detection ({lab}): among recorded effects, share reproduced with the same sign at p < 0.05: "
                         f"mean {100 * np.mean(det[key]):.0f} % over {len(det[key])} patients")
        if hip["target"]:
            L.append(f"- hippocampal targets ({len(hip['target'])} patients): mean rho recorded {np.mean(hip['target']):+.3f}, "
                     f"final {np.mean(hip['final']):+.3f}, LAG9 ridge {np.mean(hip['base']):+.3f}")
        L.append("")
    L += ["## Per patient (theta)", "", "| patient | test trials (4/6/8) | targets | recorded effects (BH) | agreement final | "
          "agreement LAG9 |", "|---|---|---|---|---|---|"]
    for sid, R in res.items():
        rt = np.array(R["theta_target_rho"], float)
        pt = np.array(R["theta_target_p"], float)
        ok = np.isfinite(rt)
        q = np.full(len(pt), np.nan)
        q[ok] = bh(pt[ok])
        ag = []
        for key in ("final", "base"):
            rp = np.array(R[f"theta_{key}_rho"], float)
            m = ok & np.isfinite(rp)
            ag.append(np.corrcoef(rt[m], rp[m])[0, 1] if m.sum() > 3 else np.nan)
        ss = R["set_sizes"]
        L.append(f"| {sid} | {R['n_test']} ({ss['4']}/{ss['6']}/{ss['8']}) | {ok.sum()} | {int((q < 0.05).sum())} | "
                 f"{ag[0]:+.3f} | {ag[1]:+.3f} |")
    md = "\n".join(L)
    print(md)
    open(os.path.join(a.out, "load30.md"), "w", encoding="utf-8").write(md + "\n")


if __name__ == "__main__":
    main()
