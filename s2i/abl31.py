"""
Phase 31 (review M1) - which part of the training explains the learning effect? Main architecture, one encoder,
ds004752 primary level, strict normalisation, 20 % calibration, explicit predictions. Every variant is evaluated with
the same ridge readout (enc) and the same mixture with the LAG9 ridge (combo_nosess).

  untrained    Morlet initialisation, no training
  episodic     the published method: closed-form ridge readout inside the loss, learnable lambda
  fixed_lam    the same episodes with lambda fixed
  multitask    conventional multi-patient pre-training: per-patient linear heads learned by gradient, no ridge

python -m s2i.abl31 --runs RUNS_DIR --out RUNS_DIR/abl31
"""
import argparse
import glob
import json
import os
import zlib

import numpy as np

VARIANTS = {"untrained": "shared_bip_car/readout_untrained1_strict_qc_none",
            "episodic": "shared_bip_car/readout_combo1_strict_qc_none",
            "fixed_lam": "shared_abl_fixedlam/readout_combo1_strict_qc_none",
            "multitask": "shared_abl_multitask/readout_combo1_strict_qc_none"}


def load(runs, rel, model):
    out = {}
    for f in glob.glob(os.path.join(runs, rel, "readout_sub-*.json")):
        d = json.load(open(f))
        ex = d.get("explicit", {}).get(model)
        if ex is not None:
            out[d["subject"]] = ex["mean"]["r"]
    return out


def boot(v, name, n=10000):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    from scipy.stats import wilcoxon
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = ["# Phase 31: training-objective ablation (main architecture, one encoder, ds004752 primary strict)", ""]
    res = {}
    for model, lab in (("combo_nosess", "mixed with the LAG9 ridge"), ("enc", "encoder alone")):
        R = {v: load(a.runs, rel, model) for v, rel in VARIANTS.items()}
        subs = sorted(set.intersection(*[set(x) for x in R.values() if x])) if all(R.values()) else []
        L += [f"## {lab} ({model})", "", "| variant | patients | mean r |", "|---|---|---|"]
        for v, x in R.items():
            L.append(f"| {v} | {len(x)} | {np.mean([x[s] for s in subs]) if subs else float('nan'):.4f} |")
        if not subs:
            L.append("\nmissing results: " + ", ".join(v for v, x in R.items() if not x))
            continue
        L += ["", "| A | B | mean A − B [95 % CI] | A > B | p (Wilcoxon) |", "|---|---|---|---|---|"]
        for A, B in (("episodic", "untrained"), ("fixed_lam", "untrained"), ("multitask", "untrained"),
                     ("episodic", "multitask"), ("episodic", "fixed_lam")):
            d = np.array([R[A][s] - R[B][s] for s in subs])
            lo, hi = boot(d, f"abl31 {model} {A} {B}")
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else float("nan")
            res[f"{model}|{A}-{B}"] = {"mean": float(d.mean()), "ci": [lo, hi], "wins": int((d > 0).sum()),
                                       "n": len(d), "p": p}
            L.append(f"| {A} | {B} | {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | {(d > 0).sum()}/{len(d)} | {p:.4f} |")
        L.append("")
    md = "\n".join(L)
    print(md)
    open(os.path.join(a.out, "abl31.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(res, open(os.path.join(a.out, "abl31.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
