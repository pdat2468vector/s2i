"""
Phase 36 (review 6, M3; plan paper/review6_experiments_plan.md, 24 Oct 2026; post hoc) - deep baselines at the primary
level with strict normalisation (ds004752), so that every model is compared under the conditions of the main analysis.

  build   NOQC/sub-XX.pt -> OUT/sub-XX.pt with eeg_car and ieeg_bip re-normalised by the median / MAD of the patient's
          own calibration trials (first max(3, round(0.2 n)), exactly s2i.readout --strict). Every other field is
          unchanged, so s2i.train_loso (LeadFormer) and s2i.nfn_baseline (NeuroFlowNet) read it as before. A patient's
          normalisation never uses its test trials; when it serves as a training patient of another fold, its
          calibration-based statistics are as legitimate as any other.
  report  one table: every model on the same 15 patients and test trials -> RUNS/strict36_report.md / .json

python -m s2i.strict36 build  --data NOQC --out DATA36
python -m s2i.strict36 report --runs RUNS
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def build(a):
    import torch
    from s2i.readout import strict_renorm
    os.makedirs(a.out, exist_ok=True)
    for fp in sorted(glob.glob(os.path.join(a.data, "sub-[0-9][0-9].pt"))):
        sid = os.path.basename(fp)[:-3]
        d = torch.load(fp, weights_only=False)
        st = torch.load(os.path.join(a.data, f"{sid}_normstats.pt"), weights_only=False)
        b = d["trial_bounds"].numpy()
        ses = list(d.get("trial_session", ["ses-01"] * len(b)))
        eeg, tgt = d["eeg_car"].half().float().numpy(), d["ieeg_bip"].half().float().numpy()
        E, Y = [eeg[:, s0:s1] for s0, s1 in b], [tgt[:, s0:s1] for s0, s1 in b]
        n_cal = max(3, int(round(len(E) * a.frac)))
        E, Y = strict_renorm(E, Y, ses, st, n_cal)
        e = dict(d)
        e["eeg_car"] = torch.from_numpy(np.clip(np.concatenate(E, 1), -6e4, 6e4).astype(np.float16))
        e["ieeg_bip"] = torch.from_numpy(np.clip(np.concatenate(Y, 1), -6e4, 6e4).astype(np.float16))
        e["normalisation"] = f"strict: median / MAD of the first {n_cal} trials (calibration)"
        torch.save(e, os.path.join(a.out, f"{sid}.pt.tmp"))
        os.replace(os.path.join(a.out, f"{sid}.pt.tmp"), os.path.join(a.out, f"{sid}.pt"))
        print(f"{sid}: {len(E)} trials, calibration {n_cal}", flush=True)


def report(a):
    from s2i.parm27 import _ex, _load, compare
    rng = np.random.default_rng(0)
    R = a.runs
    models = {}
    rd = lambda rel: _load(os.path.join(R, os.path.dirname(rel)), os.path.basename(rel))  # noqa: E731
    one, fin = rd("shared_bip_car/readout_combo1_strict_qc_none"), rd("shared_bip_car/readout_combo5_strict_qc_none")
    models["LAG9 ridge"] = _ex(one, "base")
    models["final model"] = _ex(fin, "combo_nosess")
    models["one learned encoder, mixed with the LAG9 ridge"] = _ex(one, "combo_nosess")
    models["one learned encoder, alone"] = _ex(one, "enc")
    for lab, rel in (("ShallowConvNet-style front end, untrained", "shared_shallow/p36_fixed"),
                     ("ShallowConvNet-style front end, learned", "shared_shallow/p36_learned")):
        D = rd(rel)
        models[lab + ", mixed with the LAG9 ridge"] = _ex(D, "combo_nosess")
        models[lab + ", alone"] = _ex(D, "enc")
    lf = {}
    for f in glob.glob(os.path.join(R, "p36_leadformer", "fold_sub-*.json")):
        fo = json.load(open(f))
        if "calibrated" in fo:
            lf[fo["subject"]] = float(np.nanmean(fo["calibrated"]["r"]))
    models["LeadFormer (pre-trained, fine-tuned)"] = lf
    for mode, lab in (("nfn_mode", "zero-latent output"), ("nfn_sample", "single sample"), ("nfn_mean", "mean of 8 samples")):
        x = {}
        for f in glob.glob(os.path.join(R, "p36_nfn", "nfn_sub-*.json")):
            fo = json.load(open(f))
            if mode in fo.get("models", {}):
                x[fo["subject"]] = float(fo["models"][mode]["r"])
        models[f"NeuroFlowNet, {lab}"] = x
    base, final = models["LAG9 ridge"], models["final model"]
    L = ["# Phase 36: all models at the primary level, strict normalisation (ds004752; post hoc, unadjusted p)", "",
         "| model | n | mean r | − LAG9 ridge [95 % CI] | > LAG9 ridge | p | final model − model [95 % CI] | final > model |",
         "|---|---|---|---|---|---|---|---|"]
    out = {}
    for lab, x in models.items():
        if not x:
            L.append(f"| {lab} | missing | | | | | | |")
            continue
        h = compare(x, base, rng) if lab != "LAG9 ridge" else None
        g = compare(final, x, rng) if lab != "final model" else None
        out[lab] = {"mean": float(np.mean(list(x.values()))), "n": len(x), "minus_base": h, "final_minus": g}
        c = lambda q: f"{q['mean_diff']:+.4f} [{q['ci'][0]:+.4f}, {q['ci'][1]:+.4f}]"  # noqa: E731
        L.append(f"| {lab} | {len(x)} | {out[lab]['mean']:.4f} | " + (f"{c(h)} | {h['wins']}/{h['n']} | {h['p']:.2e} | "
                 if h else "– | – | – | ") + (f"{c(g)} | {g['wins']}/{g['n']} |" if g else "– | – |"))
    L += ["", "Matched comparisons (same readout, same training procedure):", "",
          "| A | B | n | A − B [95 % CI] | A > B | p |", "|---|---|---|---|---|---|"]
    for A, B in (("one learned encoder, mixed with the LAG9 ridge", "ShallowConvNet-style front end, learned, mixed with the LAG9 ridge"),
                 ("one learned encoder, alone", "ShallowConvNet-style front end, learned, alone"),
                 ("ShallowConvNet-style front end, learned, mixed with the LAG9 ridge", "ShallowConvNet-style front end, untrained, mixed with the LAG9 ridge"),
                 ("one learned encoder, mixed with the LAG9 ridge", "LeadFormer (pre-trained, fine-tuned)")):
        if models.get(A) and models.get(B):
            h = compare(models[A], models[B], rng)
            out[f"{A} | {B}"] = h
            L.append(f"| {A} | {B} | {h['n']} | {h['mean_diff']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] | "
                     f"{h['wins']}/{h['n']} | {h['p']:.2e} |")
    md = "\n".join(L)
    print(md)
    open(os.path.join(R, "strict36_report.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(os.path.join(R, "strict36_report.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "report"))
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--runs")
    ap.add_argument("--frac", type=float, default=0.2)
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    build(a) if a.cmd == "build" else report(a)


if __name__ == "__main__":
    main()
