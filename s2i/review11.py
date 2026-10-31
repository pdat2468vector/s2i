"""
Phase 11 - answers to the external review, computed from saved results (no retraining) except `refcheck`.

  nested       leave-one-patient-out (nested) selection of the model configuration: for each patient, the
               configuration with the best mean test r on the OTHER 14 patients is chosen and scored on the
               held-out patient. Candidate pool = every (readout folder, model) with results for all patients,
               separately for the strict and the transductive setting. Also: per-patient selection by
               calibration K-fold r only (no test data at all), and the optimism of picking the best on all 15.
  reliab       calibration K-fold r vs test r per contact: Spearman per patient (median, range) and pooled.
  refmeta      reference / ground / amplifier fields of the BIDS sidecars (*_eeg.json, *_ieeg.json, channels.tsv).
  refcheck     what makes "monopolar + original scalp reference" so predictable (clean split, lagged ridge):
                 - common-mode check: correlation of the mean monopolar iEEG with the mean original-reference scalp
                 - share of each monopolar contact's variance explained by the iEEG common mode
                 - prediction of: monopolar, monopolar minus the iEEG common mode, bipolar, and the iEEG common mode
                   itself, each from original-reference and from CAR scalp EEG.
               If removing the iEEG common mode brings monopolar/original back to the bipolar level and the common
               mode is far more predictable from original-reference scalp than from CAR scalp, the inflation is a
               component common to all contacts that is also present in non-re-referenced scalp EEG.

Usage
-----
python -m s2i.review11 nested   --runs RUNS_DIR
python -m s2i.review11 reliab   --runs RUNS_DIR
python -m s2i.review11 refmeta  --bids RAW_BIDS_DIR --out RUNS_DIR/review11
python -m s2i.review11 refcheck --data DATA_DIR --out RUNS_DIR/review11
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

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


# ----------------------------------------------------------------------------- nested selection
def collect(runs, strict, frac, match=None):
    """-> subjects, candidate names, R (subjects x candidates) test r, CV (same, calibration K-fold r or nan).
    match: only readout folders whose name contains this string (e.g. 'qc_none'); without it, folders of the QC
    variants ('qc_') are left out so that one data version is never mixed with another."""
    cand = {}
    for d in sorted(glob.glob(os.path.join(runs, "shared_*", "readout*"))):
        name = os.path.basename(d)
        if not os.path.isdir(d) or (("strict" in name) != strict):
            continue
        if (match and match not in name) or (not match and "qc_" in name):
            continue
        for f in glob.glob(os.path.join(d, "readout_sub-*.json")):
            r = json.load(open(f))
            fr = r.get("fracs", {}).get(frac)
            if not fr:
                continue
            for m, v in fr["models"].items():
                if not isinstance(v, dict) or not isinstance(v.get("r"), (int, float)):
                    continue
                key = f"{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}:{m}"
                cv = np.nanmean(v["cv_contacts"]) if v.get("cv_contacts") else np.nan
                cand.setdefault(key, {})[r["subject"]] = (v["r"], cv)
    if not cand:
        return [], [], None, None
    n_max = max(len(v) for v in cand.values())
    cand = {k: v for k, v in cand.items() if len(v) == n_max}
    subs = sorted(next(iter(cand.values())))
    names = sorted(cand)
    R = np.array([[cand[k][s][0] for k in names] for s in subs], float)
    CV = np.array([[cand[k][s][1] for k in names] for s in subs], float)
    return subs, names, R, CV


def nested(a, log):
    L = ["# Nested (leave-one-patient-out) selection of the model configuration", ""]
    for strict in (True, False):
        for frac in ("0.2", "0.1"):
            subs, names, R, CV = collect(a.runs, strict, frac, a.match)
            if R is None or len(subs) < 5:
                continue
            setting = "strict" if strict else "transductive"
            base_idx = [i for i, k in enumerate(names) if k.endswith(":base")]
            base = R[:, base_idx[0]]
            n = len(subs)
            pick, val = [], np.zeros(n)
            for i in range(n):
                keep = np.arange(n) != i
                j = int(np.nanargmax(np.nanmean(R[keep], 0)))
                pick.append(names[j])
                val[i] = R[i, j]
            oracle = int(np.nanargmax(np.nanmean(R, 0)))
            d = val - base
            lo, hi = boot_ci(d)
            L += [f"## {setting}, calibration {frac} ({n} patients, {len(names)} candidate configurations)",
                  f"- best on all {n} patients (optimistic): `{names[oracle]}` r = {R[:, oracle].mean():.4f}",
                  f"- nested (chosen without the held-out patient): r = {val.mean():.4f}; "
                  f"optimism of the non-nested pick = {R[:, oracle].mean() - val.mean():+.4f}",
                  f"- nested − base: {d.mean():+.4f} [95 % CI {lo:+.4f}, {hi:+.4f}], {int((d > 0).sum())}/{n} patients, "
                  f"Wilcoxon p = {wil(val, base):.5f}",
                  "- configurations chosen: " + ", ".join(f"`{k}` ×{pick.count(k)}" for k in sorted(set(pick)))]
            ok = np.isfinite(CV).all(0)
            if ok.sum() >= 2:
                cvn = [k for k, o in zip(names, ok) if o]
                Rc, Cc = R[:, ok], CV[:, ok]
                sel = np.nanargmax(Cc, 1)
                vc = Rc[np.arange(n), sel]
                d = vc - base
                lo, hi = boot_ci(d)
                L += [f"- per-patient choice by calibration K-fold r only ({ok.sum()} candidates with K-fold r): "
                      f"r = {vc.mean():.4f}; − base {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], {int((d > 0).sum())}/{n}, "
                      f"p = {wil(vc, base):.5f}",
                      "  chosen: " + ", ".join(f"`{k}` ×{list(sel).count(i)}" for i, k in enumerate(cvn) if i in sel)]
            top = np.argsort(-np.nanmean(R, 0))[:8]
            L += ["", "| candidate (top 8 by mean r) | mean r |", "|---|---|"]
            L += [f"| {names[j]} | {np.nanmean(R[:, j]):.4f} |" for j in top]
            L.append("")
    out = "\n".join(L)
    log(out)
    os.makedirs(os.path.join(a.runs, "review11"), exist_ok=True)
    open(os.path.join(a.runs, "review11", f"nested{a.tag}.md"), "w", encoding="utf-8").write(out + "\n")


# ----------------------------------------------------------------------------- reliability per patient
def reliab(a, log):
    from scipy.stats import spearmanr
    d = os.path.join(a.runs, a.main_run, a.reliab_dir)
    rows, cv_all, te_all = [], [], []
    for f in sorted(glob.glob(os.path.join(d, "readout_sub-*.json"))):
        r = json.load(open(f))
        m = r["fracs"]["0.2"]["models"][a.reliab_model]
        cv, te = np.array(m["cv_contacts"], float), np.array(m["contacts"], float)
        ok = np.isfinite(cv) & np.isfinite(te)
        cv, te = cv[ok], te[ok]
        rho = spearmanr(cv, te).correlation if len(cv) > 3 else np.nan
        k = max(1, int(round(0.25 * len(cv))))
        top = np.argsort(-cv)[:k]
        rows.append((r["subject"], len(cv), rho, te[top].mean(), te.mean()))
        cv_all.append(cv)
        te_all.append(te)
    rho = np.array([x[2] for x in rows])
    pooled = spearmanr(np.concatenate(cv_all), np.concatenate(te_all)).correlation
    gain = np.array([x[3] - x[4] for x in rows])
    lo, hi = boot_ci(gain)
    L = [f"# Reliability: calibration K-fold r vs test r ({a.reliab_dir}:{a.reliab_model})", "",
         f"- Spearman per patient: median {np.nanmedian(rho):.2f}, mean {np.nanmean(rho):.2f}, "
         f"range {np.nanmin(rho):.2f}–{np.nanmax(rho):.2f} ({len(rows)} patients)",
         f"- pooled over {sum(x[1] for x in rows)} contacts (contacts within a patient are not independent): {pooled:.2f}",
         f"- top 25 % contacts by calibration r, per patient: test r higher than the patient's mean by "
         f"{gain.mean():+.3f} [95 % CI {lo:+.3f}, {hi:+.3f}], {int((gain > 0).sum())}/{len(rows)} patients", "",
         "| patient | contacts | Spearman | test r, top 25 % | test r, all |", "|---|---|---|---|---|"]
    L += [f"| {s} | {n} | {rh:.2f} | {t:.3f} | {m:.3f} |" for s, n, rh, t, m in rows]
    out = "\n".join(L)
    log(out)
    os.makedirs(os.path.join(a.runs, "review11"), exist_ok=True)
    open(os.path.join(a.runs, "review11", f"reliability{a.tag}.md"), "w", encoding="utf-8").write(out + "\n")


# ----------------------------------------------------------------------------- reference metadata
FIELDS = ("EEGReference", "iEEGReference", "EEGGround", "iEEGGround", "Manufacturer", "ManufacturersModelName",
          "RecordingType", "SamplingFrequency", "PowerLineFrequency", "EEGPlacementScheme", "iEEGPlacementScheme",
          "ElectricalStimulation", "SoftwareFilters", "HardwareFilters")


def refmeta(a, log):
    rows = []
    for f in sorted(glob.glob(os.path.join(a.bids, "**", "*.json"), recursive=True)):
        name = os.path.basename(f)
        if not (name.endswith("_eeg.json") or name.endswith("_ieeg.json")) or "derivatives" in f:
            continue
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        rows.append((name, {k: j[k] for k in FIELDS if k in j}))
    L = ["# Reference metadata (BIDS sidecars)", ""]
    if os.path.exists(os.path.join(a.bids, "dataset_description.json")):
        dd = json.load(open(os.path.join(a.bids, "dataset_description.json"), encoding="utf-8"))
        L.append(f"Dataset: {dd.get('Name', '')}  |  references: {dd.get('ReferencesAndLinks', '')}")
    # unique values per field and modality (compact)
    for mod in ("_eeg.json", "_ieeg.json"):
        L += ["", f"## {mod}"]
        vals = {}
        for name, d in rows:
            if name.endswith(mod) and not (mod == "_eeg.json" and name.endswith("_ieeg.json")):
                for k, v in d.items():
                    vals.setdefault(k, {}).setdefault(json.dumps(v, ensure_ascii=False), []).append(name.split("_")[0])
        for k, v in vals.items():
            for val, subs in v.items():
                L.append(f"- **{k}** = {val}  ({len(subs)} files: {', '.join(sorted(set(subs)))})")
    # channels.tsv columns and any reference column
    refcol = {}
    for f in sorted(glob.glob(os.path.join(a.bids, "**", "*_channels.tsv"), recursive=True)):
        if "derivatives" in f:
            continue
        lines = open(f, encoding="utf-8").read().splitlines()
        if not lines:
            continue
        head = lines[0].split("\t")
        key = "ieeg" if "_ieeg" in f else "eeg"
        refcol.setdefault(key, set()).add(tuple(head))
        if "reference" in head:
            j = head.index("reference")
            for ln in lines[1:]:
                p = ln.split("\t")
                if len(p) > j:
                    refcol.setdefault(f"{key}:reference values", set()).add(p[j])
    L += ["", "## channels.tsv"]
    for k, v in refcol.items():
        L.append(f"- {k}: {sorted(v)[:20]}")
    out = "\n".join(L)
    log(out)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "refmeta.md"), "w", encoding="utf-8").write(out + "\n")


# ----------------------------------------------------------------------------- reference mechanism
def refcheck(a, log):
    from s2i.protocol import ALPHAS, LAG9, load, run_variant, split
    from s2i.metrics import per_contact_metrics
    FS = 200
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for p in sorted(glob.glob(os.path.join(a.data, "sub-[0-9][0-9].pt"))):
        sid = os.path.basename(p)[:-3]
        fp = os.path.join(a.out, f"refcheck_{sid}.json")
        if os.path.exists(fp):
            res[sid] = json.load(open(fp))
            continue
        bounds, eegs, tgts = load(p)
        if len(bounds) < 20:
            continue
        lens = [int(s1 - s0) for s0, s1 in bounds]
        train, test, folds = split("clean", lens, np.random.default_rng(0))
        mono, bip = tgts["mono"], tgts["bip"]
        cm = mono.mean(0, keepdims=True)                         # iEEG common mode (all monopolar contacts)
        targets = {"mono": mono, "mono_minus_cm": mono - cm, "bip": bip, "cm": cm}
        te_idx = np.concatenate([np.arange(bounds[i][0], bounds[i][1]) for i, _, _ in test])
        ce = eegs["orig"][:, te_idx].mean(0)
        r_cm = float(np.corrcoef(ce, cm[0, te_idx])[0, 1])
        cmt = cm[0, te_idx]
        share = float(np.nanmean([np.corrcoef(mono[k, te_idx], cmt)[0, 1] ** 2 for k in range(mono.shape[0])]))
        out = {"r_scalp_cm_vs_ieeg_cm": r_cm, "ieeg_variance_share_of_cm": share, "pred": {}}
        for ref in ("orig", "car"):
            E = [eegs[ref][:, s0:s1] for s0, s1 in bounds]
            for tg, Yall in targets.items():
                Y = [Yall[:, s0:s1] for s0, s1 in bounds]
                P, T, _ = run_variant(LAG9, ALPHAS, E, Y, train, test, folds)
                out["pred"][f"{tg}|{ref}"] = float(np.nanmean(per_contact_metrics(P, T, FS, bands=False)["r"]))
        out["subject"] = sid
        json.dump(out, open(fp, "w"))
        res[sid] = out
        log(f"[{sid}] scalp-cm vs iEEG-cm r={r_cm:+.3f}, iEEG cm share={share:.2f} | " +
            " ".join(f"{k}={v:+.3f}" for k, v in out["pred"].items()))
    subs = sorted(res)
    keys = list(res[subs[0]]["pred"])
    M = {k: np.array([res[s]["pred"][k] for s in subs]) for k in keys}
    L = ["# What makes monopolar + original-reference scalp predictable (lagged ridge, clean split)", "",
         "| target \\ scalp reference | original | CAR |", "|---|---|---|"]
    lab = {"mono": "monopolar", "mono_minus_cm": "monopolar − iEEG common mode", "bip": "bipolar",
           "cm": "iEEG common mode itself"}
    for tg in ("mono", "mono_minus_cm", "bip", "cm"):
        L.append(f"| {lab[tg]} | {M[tg + '|orig'].mean():.3f} | {M[tg + '|car'].mean():.3f} |")
    d = M["mono|orig"] - M["mono_minus_cm|orig"]
    lo, hi = boot_ci(d)
    L += ["", f"- removing the iEEG common mode (original scalp): {-d.mean():+.3f} [{-hi:+.3f}, {-lo:+.3f}], "
          f"lower in {int((d > 0).sum())}/{len(subs)} patients, p = {wil(M['mono|orig'], M['mono_minus_cm|orig']):.5f}"]
    x = M["mono|orig"] - M["mono|car"] - M["bip|orig"] + M["bip|car"]
    lo, hi = boot_ci(x)
    L += [f"- interaction (mono − bip) × (orig − CAR): {x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], "
          f"positive in {int((x > 0).sum())}/{len(subs)} patients, p = {wil(x, np.zeros_like(x)):.5f}", "",
          "| patient | scalp-cm vs iEEG-cm r | iEEG common-mode variance share | mono, orig | mono − cm, orig | "
          "bip, CAR | cm, orig | cm, CAR |",
          "|---|---|---|---|---|---|---|---|"]
    for s in subs:
        q = res[s]
        p_ = q["pred"]
        L.append(f"| {s} | {q['r_scalp_cm_vs_ieeg_cm']:+.2f} | {q['ieeg_variance_share_of_cm']:.2f} | "
                 f"{p_['mono|orig']:.3f} | {p_['mono_minus_cm|orig']:.3f} | {p_['bip|car']:.3f} | "
                 f"{p_['cm|orig']:.3f} | {p_['cm|car']:.3f} |")
    out = "\n".join(L)
    log(out)
    open(os.path.join(a.out, "refcheck.md"), "w", encoding="utf-8").write(out + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("nested", "reliab", "refmeta", "refcheck"))
    ap.add_argument("--runs", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--bids", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--main_run", default="shared_bip_car")
    ap.add_argument("--reliab_dir", default="readout_combo5_strict")
    ap.add_argument("--reliab_model", default="combo_nosess")
    ap.add_argument("--match", default=None, help="nested: only readout folders containing this string")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    {"nested": nested, "reliab": reliab, "refmeta": refmeta, "refcheck": refcheck}[a.cmd](a, print)


if __name__ == "__main__":
    main()
