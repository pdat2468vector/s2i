"""
Phase 28A - statistical sensitivity of the system and learning effects in the three cohorts (review: Fisher z, BCa,
linear mixed model). Reads saved readout results only.

For each cohort and comparison (A − B, the same contacts in both arms):
  reported   mean over patients of the patient-mean r, percentile patient bootstrap (as in the manuscript)
  BCa        bias-corrected and accelerated bootstrap CI of the same mean (scipy.stats.bootstrap, 10,000 resamples)
  Fisher z   per-contact atanh(r), averaged per patient; mean difference, percentile CI, Wilcoxon p
  LMM        per-contact difference (in r and in z) ~ 1 + (1 | patient), statsmodels MixedLM (REML); intercept,
             95 % CI and p. Contacts, not patients, are the observations, and patients are a random intercept.

python -m s2i.sens28 --runs RUNS_DIR --out RUNS_DIR/sens28
"""
import argparse
import glob
import json
import os
import zlib

import numpy as np

COHORTS = {
    "ds004752": ("0.2", {"system": (("shared_bip_car/readout_combo5_strict_qc_none", "combo_nosess"),
                                    ("shared_bip_car/readout_combo5_strict_qc_none", "base")),
                         "learning": (("shared_no_env/readout_learned_arch3_strict_qc_none", "combo_nosess"),
                                      ("shared_no_env/readout_fixed_arch3_strict_qc_none", "combo_nosess"))}),
    "ScaleOLDNEW": ("0.2", {"system": (("p17/main/final_primary", "combo_nosess"), ("p17/main/final_primary", "base")),
                            "learning": (("p17/no_env/learned3_primary", "combo_nosess"),
                                         ("p17/no_env/fixed3_primary", "combo_nosess"))}),
    "Parmigiani 2022": ("0.3", {"system": (("p27/short/learned_primary", "combo_nosess"),
                                           ("p27/short/learned_primary", "base")),
                                "learning": (("p27/short/learned_primary", "combo_nosess"),
                                             ("p27/short/fixed_primary", "combo_nosess"))}),
}


def contacts(runs, folder, model, frac):
    """{subject: per-contact r (explicit predictions; statistics r if no explicit result was stored)}, and the source."""
    out, src = {}, set()
    for f in glob.glob(os.path.join(runs, folder, "readout_sub-*.json")):
        d = json.load(open(f))
        ex = d.get("explicit", {}).get(model)
        if ex is not None and abs(ex.get("frac", 0.2) - float(frac)) < 1e-9 and "r" in ex:
            out[d["subject"]] = np.asarray(ex["r"], float)
            src.add("explicit")
        else:
            m = d.get("fracs", {}).get(frac, {}).get("models", {}).get(model)
            if m and "contacts" in m:
                out[d["subject"]] = np.asarray(m["contacts"], float)
                src.add("statistics")
    return out, sorted(src)


def pct_ci(v, name, n=10000):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyse(A, B, name):
    from scipy.stats import bootstrap, wilcoxon
    import pandas as pd
    import statsmodels.formula.api as smf
    subs = sorted(set(A) & set(B))
    rows, dr, dz = [], [], []
    for s in subs:
        a, b = A[s], B[s]
        assert len(a) == len(b), f"{name} {s}: contact counts differ"
        ok = np.isfinite(a) & np.isfinite(b)
        za, zb = np.arctanh(np.clip(a[ok], -0.999, 0.999)), np.arctanh(np.clip(b[ok], -0.999, 0.999))
        dr.append(a[ok].mean() - b[ok].mean())
        dz.append(za.mean() - zb.mean())
        rows += [{"patient": s, "dr": x, "dz": y} for x, y in zip(a[ok] - b[ok], za - zb)]
    dr, dz = np.array(dr), np.array(dz)
    res = {"n_patients": len(subs), "n_contacts": len(rows)}
    res["reported"] = {"mean": float(dr.mean()), "ci": pct_ci(dr, name + " r"), "wins": int((dr > 0).sum())}
    bc = bootstrap((dr,), np.mean, n_resamples=10000, method="BCa", random_state=zlib.crc32(name.encode()))
    res["bca"] = [float(bc.confidence_interval.low), float(bc.confidence_interval.high)]
    res["fisher_z"] = {"mean": float(dz.mean()), "ci": pct_ci(dz, name + " z"), "wins": int((dz > 0).sum()),
                       "p_wilcoxon": float(wilcoxon(dz).pvalue)}
    df = pd.DataFrame(rows)
    for col in ("dr", "dz"):
        fit = smf.mixedlm(f"{col} ~ 1", df, groups=df["patient"]).fit(reml=True)
        lo, hi = fit.conf_int().loc["Intercept"].tolist()
        res[f"lmm_{col}"] = {"estimate": float(fit.params["Intercept"]), "ci": [float(lo), float(hi)],
                             "p": float(fit.pvalues["Intercept"]), "patient_sd": float(np.sqrt(max(fit.cov_re.iloc[0, 0], 0))),
                             "residual_sd": float(np.sqrt(fit.scale))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {}
    L = ["# Phase 28A: statistical sensitivity of the system and learning effects", "",
         "Reported = mean of patient-mean differences with the percentile patient bootstrap of the manuscript. "
         "BCa = bias-corrected and accelerated bootstrap of the same mean. Fisher z = per-contact atanh(r) averaged per "
         "patient. LMM = per-contact difference ~ 1 + (1 | patient), REML; contacts are observations.", "",
         "| cohort | effect | patients / contacts | source | reported [95 % CI] | BCa 95 % CI | Fisher z [95 % CI], Wilcoxon p "
         "| LMM on r [95 % CI], p | LMM on z [95 % CI], p |", "|" + "---|" * 9]
    for coh, (frac, comps) in COHORTS.items():
        for eff, ((fa, ma), (fb, mb)) in comps.items():
            A, sa = contacts(a.runs, fa, ma, frac)
            B, sb = contacts(a.runs, fb, mb, frac)
            if not A or not B:
                L.append(f"| {coh} | {eff} | – | missing results ({fa} / {fb}) | | | | | |")
                continue
            r = analyse(A, B, f"sens28 {coh} {eff}")
            r["source"] = sorted(set(sa) | set(sb))
            out[f"{coh}|{eff}"] = r
            rep, z, lr, lz = r["reported"], r["fisher_z"], r["lmm_dr"], r["lmm_dz"]
            L.append(f"| {coh} | {eff} | {r['n_patients']} / {r['n_contacts']} | {', '.join(r['source'])} | "
                     f"{rep['mean']:+.4f} [{rep['ci'][0]:+.4f}, {rep['ci'][1]:+.4f}] ({rep['wins']}/{r['n_patients']}) | "
                     f"[{r['bca'][0]:+.4f}, {r['bca'][1]:+.4f}] | {z['mean']:+.4f} [{z['ci'][0]:+.4f}, {z['ci'][1]:+.4f}], "
                     f"{z['p_wilcoxon']:.2e} | {lr['estimate']:+.4f} [{lr['ci'][0]:+.4f}, {lr['ci'][1]:+.4f}], {lr['p']:.2e} | "
                     f"{lz['estimate']:+.4f} [{lz['ci'][0]:+.4f}, {lz['ci'][1]:+.4f}], {lz['p']:.2e} |")
    md = "\n".join(L)
    print(md)
    open(os.path.join(a.out, "sens28.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(os.path.join(a.out, "sens28.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
