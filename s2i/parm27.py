"""
Phase 27 - third cohort (Parmigiani et al. 2022; OSF wsgzp, CC BY-NC-SA 4.0), exactly as in
paper/cohort3_protocol.md (v1.0, 21 Oct 2026, written after the Phase 26 inventory and before any model was applied).

Commands
  build   extracted ccepcoreg tree (Phase 26) -> one s2i data file per included patient (format read by
          s2i.readout --strict) + p27_build.md / p27_build.json (execution log)
  runs    copy the final short encoder trained on ds004752 (s2i.shared --final --half 40 -> <enc_root>/short/
          enc_final.pt, the encoder used for ScaleOLDNEW) to <runs>/short/enc_<sub>.pt
  report  H1-H3 (two-sided Wilcoxon, Holm over H1-H3, median direction, patient bootstrap CI) + descriptive analyses
          -> p27_report.md / p27_report.json

Rules (protocol sections in brackets)
  runs [3]        a run is used only if its EEG-SEEG pairing is consistent: len(eeg_badtrials flags) = number of SEEG
                  epochs and the number of zero flags = number of EEG epochs (the k-th zero flag pairs SEEG epoch with
                  EEG epoch k); runs in run-number order (the only order available)
  channels [4]    EEG: status "good" in every used run; CAR over these channels. SEEG: bipolar channels as shared,
                  status "good" in every used run, sharing no contact with the stimulated pair of any used run
  signals [4]     each epoch (-0.3 ... +0.7 s at 1000 Hz) resampled to 200 Hz (scipy resample_poly 1/5), microvolts
  inclusion [3]   >= 100 paired epochs, >= 3 EEG channels, >= 3 valid targets (non-zero variance in calibration and test)
  split [5]       epochs in run order; calibration = first round(0.3 N); one-epoch gap; the rest is the test set
Storage as in s2i.scales17: robust-normalised with the patient's median / MAD of all kept samples, float16, with the
statistics (one copy per run = session); s2i.readout --strict re-normalises with the calibration epochs only.

python -m s2i.parm27 build  --src /content/parm --out DATA27
python -m s2i.parm27 runs   --enc_root ENC17 --data DATA27 --runs RUNS27
python -m s2i.parm27 report --data DATA27 --runs RUNS27
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np

FS_IN, DOWN, FS = 1000, 5, 200
CAL_FRAC, MIN_EPOCHS, MIN_EEG, MIN_TGT = 0.3, 100, 3, 3


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)


def n_cal_of(n, frac=CAL_FRAC):
    return max(3, int(round(n * frac)))


def robust(x):
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True) * 1.4826
    return med, np.where(mad < 1e-30, 1.0, mad)


def read_tsv(f):
    lines = open(f, encoding="utf-8").read().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]


def contacts_of(ch):
    m = re.match(r"(\S+?)(\d+)-(\d+)$", ch)
    return (m.group(1), {int(m.group(2)), int(m.group(3))}) if m else (None, set())


def stim_pair(desc):
    m = re.search(r"Stimulation of\s+(\S+?)(\d+)-(\d+)\s", desc)
    return (m.group(1), {int(m.group(2)), int(m.group(3))}) if m else (None, set())


# ----------------------------------------------------------------------------- one patient
def patient_runs(root, bad, sid):
    """-> [(run, base_ieeg, base_eeg, flags, description)] with consistent pairing, and the list of skipped runs."""
    d = os.path.join(root, "derivatives", "epochs", sid)
    use, skip = [], []
    for fi in sorted(glob.glob(os.path.join(d, "ieeg", f"{sid}_task-ccepcoreg_run-*_epochs.npy")),
                     key=lambda p: int(re.search(r"run-(\d+)", p).group(1))):
        run = re.search(r"(run-\d+)", fi).group(1)
        fe = fi.replace(os.sep + "ieeg" + os.sep, os.sep + "eeg" + os.sep)
        flags = bad.get(sid, {}).get(run)
        if not os.path.exists(fe) or flags is None:
            skip.append((run, "missing EEG file or flags"))
            continue
        ni, ne = np.load(fi, mmap_mode="r").shape[0], np.load(fe, mmap_mode="r").shape[0]
        if len(flags) != ni or len(flags) - int(sum(flags)) != ne:
            skip.append((run, f"pairing inconsistent (SEEG {ni}, EEG {ne}, flags {len(flags)}/{len(flags) - int(sum(flags))})"))
            continue
        desc = json.load(open(fi[:-len("_epochs.npy")] + "_epochs.json")).get("Description", "")
        use.append((run, fi[:-len("_epochs.npy")], fe[:-len("_epochs.npy")], np.asarray(flags), desc))
    return use, skip


def build_patient(root, bad, sid):
    from scipy.signal import resample_poly
    info = {"subject": sid, "excluded": None}
    use, skip = patient_runs(root, bad, sid)
    info["runs_used"], info["runs_skipped"] = [u[0] for u in use], skip
    if not use:
        info["excluded"] = "no run with consistent pairing"
        return info, None
    chi = [read_tsv(u[1] + "_channels.tsv") for u in use]
    che = [read_tsv(u[2] + "_channels.tsv") for u in use]
    names_i, names_e = [c["name"] for c in chi[0]], [c["name"] for c in che[0]]
    assert all([c["name"] for c in x] == names_i for x in chi) and all([c["name"] for c in x] == names_e for x in che), \
        f"{sid}: channel lists differ between runs"
    good_e = [n for j, n in enumerate(names_e) if all(x[j].get("status") == "good" for x in che)]
    stims = [stim_pair(u[4]) for u in use]
    near = set()
    for n in names_i:
        sh, cs = contacts_of(n)
        if any(sh == s and cs & c for s, c in stims):
            near.add(n)
    good_i = [n for j, n in enumerate(names_i) if all(x[j].get("status") == "good" for x in chi) and n not in near]
    info.update(n_eeg_listed=len(names_e), n_eeg=len(good_e), n_seeg_listed=len(names_i),
                n_seeg_near_stim=len(near), n_seeg=len(good_i),
                stim_pairs=[f"{s}{'-'.join(map(str, sorted(c)))}" for s, c in stims])
    ie, ii = [names_e.index(n) for n in good_e], [names_i.index(n) for n in good_i]
    E, Y, ses, desc = [], [], [], []
    for run, bi, be, flags, d in use:
        xi = np.load(bi + "_epochs.npy", mmap_mode="r")
        xe = np.load(be + "_epochs.npy", mmap_mode="r")
        keep = np.where(flags == 0)[0]
        for k, j in enumerate(keep):                          # k-th EEG epoch <-> j-th SEEG epoch
            e = resample_poly(np.asarray(xe[k][ie], np.float64) * 1e6, 1, DOWN, axis=-1)
            y = resample_poly(np.asarray(xi[j][ii], np.float64) * 1e6, 1, DOWN, axis=-1)
            E.append(e - e.mean(0, keepdims=True))            # CAR over the kept EEG channels
            Y.append(y)
            ses.append(run)
        desc.append({"run": run, "description": d, "n_pairs": int(len(keep))})
    N = len(E)
    nc = n_cal_of(N)
    info.update(n_epochs=N, n_cal=nc, n_test=max(N - nc - 1, 0), calib_minutes=nc * E[0].shape[1] / FS / 60,
                test_minutes=max(N - nc - 1, 0) * E[0].shape[1] / FS / 60, samples_per_epoch=int(E[0].shape[1]),
                runs=desc, calib_runs=sorted(set(ses[:nc]), key=ses.index), test_runs=sorted(set(ses[nc + 1:]), key=ses.index))
    valid = [k for k in range(len(good_i))
             if np.var(np.concatenate([y[k] for y in Y[:nc]])) > 0 and np.var(np.concatenate([y[k] for y in Y[nc + 1:]])) > 0]
    info["n_valid_targets"] = len(valid)
    if N < MIN_EPOCHS:
        info["excluded"] = f"{N} paired epochs (< {MIN_EPOCHS})"
    elif len(good_e) < MIN_EEG:
        info["excluded"] = f"{len(good_e)} EEG channels (< {MIN_EEG})"
    elif len(valid) < MIN_TGT:
        info["excluded"] = f"{len(valid)} valid targets (< {MIN_TGT})"
    return info, (E, [y[valid] for y in Y], ses, good_e, [good_i[k] for k in valid])


def write_patient(a, root, sid, data):
    import torch
    E, Y, ses, eeg_names, bip_names = data
    mni = {}
    f = glob.glob(os.path.join(root, "derivatives", "epochs", sid, "ieeg", "*space-MNI152*_electrodes.tsv"))
    if f:
        for r_ in read_tsv(f[0]):
            try:
                mni[r_["name"]] = [float(r_["x"]), float(r_["y"]), float(r_["z"])]
            except (KeyError, ValueError):
                pass
    pos = {}
    f = glob.glob(os.path.join(root, "derivatives", "epochs", sid, "eeg", "*_electrodes.tsv"))
    if f:
        for r_ in read_tsv(f[0]):
            try:
                pos[r_["name"]] = [float(r_["x"]), float(r_["y"]), float(r_["z"])]
            except (KeyError, ValueError):
                pass
    runs = list(dict.fromkeys(ses))
    d = {"subject": sid, "dataset": "Parmigiani2022", "fs": FS, "level": "primary", "eeg_names": eeg_names,
         "bip_names": bip_names, "eeg_pos": [pos.get(n, [np.nan] * 3) for n in eeg_names],
         "bip_mni": torch.tensor([mni.get(n, [np.nan] * 3) for n in bip_names], dtype=torch.float32),
         "epoch_t0_sample": 60, "norm_clip_z": float("inf"), "norm_stats": {}}
    bounds, c = [], 0
    for e in E:
        bounds.append((c, c + e.shape[1]))
        c += e.shape[1]
    d["trial_bounds"] = torch.tensor(bounds, dtype=torch.long)
    d["trial_session"] = list(ses)
    n_inf = 0
    for k, X in (("eeg_car", np.concatenate(E, 1)), ("ieeg_bip", np.concatenate(Y, 1))):
        med, mad = robust(X)
        z = (X - med) / mad
        n_inf += int((np.abs(z) > 6e4).sum())
        d[k] = torch.from_numpy(np.clip(z, -6e4, 6e4).astype(np.float16))
        st = {"med": med[:, 0].tolist(), "mad": mad[:, 0].tolist()}
        d["norm_stats"][k] = {r_: st for r_ in runs}             # one copy per run (= session in s2i.readout)
    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, f"{sid}.pt")
    torch.save(d, fp + ".tmp")
    os.replace(fp + ".tmp", fp)
    torch.save({k: d[k] for k in ("subject", "norm_stats", "norm_clip_z", "trial_bounds", "trial_session")},
               os.path.join(a.out, f"{sid}_normstats.pt"))
    return n_inf


def build(a):
    root = os.path.join(a.src, "ccepcoreg")
    bad = json.load(open(os.path.join(a.src, "eeg_badtrials.json")))
    os.makedirs(a.out, exist_ok=True)
    jp = os.path.join(a.out, "p27_build.json")
    res = json.load(open(jp)) if os.path.exists(jp) else {"patients": {}}
    res["rules"] = {"fs_in": FS_IN, "fs": FS, "cal_frac": CAL_FRAC, "min_epochs": MIN_EPOCHS, "min_eeg": MIN_EEG,
                    "min_targets": MIN_TGT}
    subs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "derivatives", "epochs", "sub-*")))
    for sid in subs:
        if a.patients and sid not in a.patients:
            continue
        if sid in res["patients"] and not a.overwrite:
            log(f"[{sid}] already built")
            continue
        t0 = time.time()
        info, data = build_patient(root, bad, sid)
        if info["excluded"] is None:
            info["float16_overflow_samples"] = write_patient(a, root, sid, data)
        res["patients"][sid] = info
        json.dump(res, open(jp + ".tmp", "w"), indent=1, default=str)
        os.replace(jp + ".tmp", jp)
        log(f"[{sid}] runs {len(info['runs_used'])} (skipped {len(info['runs_skipped'])}), {info.get('n_epochs', 0)} "
            f"epochs (calib {info.get('n_cal', 0)}, test {info.get('n_test', 0)}), EEG {info.get('n_eeg', 0)}, SEEG "
            f"{info.get('n_valid_targets', 0)}, excluded: {info['excluded']} | {time.time() - t0:.0f}s")
    write_build_md(a, res)


def write_build_md(a, res):
    P = res["patients"]
    L = ["# Phase 27 execution log: data (cohort-3 protocol v1.0, Sections 3-5)", "", f"- rules: {res['rules']}", "",
         "| patient | runs used | runs skipped | epochs | calib | test | calib min | test min | EEG ch (listed) | "
         "SEEG targets (listed, near stim) | calib runs | excluded |", "|" + "---|" * 12]
    for sid, i in sorted(P.items()):
        L.append(f"| {sid} | {len(i['runs_used'])} | {len(i['runs_skipped'])} | {i.get('n_epochs', 0)} | "
                 f"{i.get('n_cal', '')} | {i.get('n_test', '')} | {i.get('calib_minutes', 0):.2f} | "
                 f"{i.get('test_minutes', 0):.1f} | {i.get('n_eeg', '')} ({i.get('n_eeg_listed', '')}) | "
                 f"{i.get('n_valid_targets', '')} ({i.get('n_seeg_listed', '')}, {i.get('n_seeg_near_stim', '')}) | "
                 f"{', '.join(i.get('calib_runs', []))} | {i['excluded'] or ''} |")
    inc = [s for s, i in P.items() if i["excluded"] is None]
    L += ["", f"included: {len(inc)} patients: {', '.join(sorted(inc))}", "", "Skipped runs:", ""]
    L += [f"- {sid}: " + "; ".join(f"{r} ({w})" for r, w in i["runs_skipped"]) for sid, i in sorted(P.items())
          if i["runs_skipped"]]
    out = "\n".join(L)
    open(os.path.join(a.out, "p27_build.md"), "w", encoding="utf-8").write(out + "\n")
    print(out)


# ----------------------------------------------------------------------------- runs
def runs(a):
    subs = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(a.data, "sub-*.pt"))
                  if not p.endswith("_normstats.pt"))
    src = os.path.join(a.enc_root, "short", "enc_final.pt")
    assert os.path.exists(src), f"missing final short encoder {src}"
    od = os.path.join(a.runs, "short")
    os.makedirs(od, exist_ok=True)
    for s in subs:
        dst = os.path.join(od, f"enc_{s}.pt")
        if not os.path.exists(dst):
            shutil.copy(src, dst)
    log(f"short encoder copied for {len(subs)} patients -> {od}")


# ----------------------------------------------------------------------------- report
def _load(run, name):
    out = {}
    for f in glob.glob(os.path.join(run, name, "readout_sub-*.json")):
        d = json.load(open(f))
        out[d["subject"]] = d
    return out


def _ex(D, m, k="r"):
    return {s: r["explicit"][m]["mean"][k] for s, r in D.items() if m in r.get("explicit", {})
            and k in r["explicit"][m]["mean"]}


def compare(a_, b_, rng):
    from scipy.stats import wilcoxon
    s = sorted(set(a_) & set(b_))
    x, y = np.array([a_[q] for q in s]), np.array([b_[q] for q in s])
    d = x - y
    if len(d) == 0:
        return None
    bt = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    try:
        p = float(wilcoxon(x, y, alternative="two-sided").pvalue)
    except ValueError:
        p = float("nan")
    return {"n": len(d), "mean_a": float(x.mean()), "mean_b": float(y.mean()), "median_diff": float(np.median(d)),
            "mean_diff": float(d.mean()), "ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
            "wins": int((d > 0).sum()), "p": p, "per_patient": dict(zip(s, d.round(5).tolist()))}


def holm(ps):
    order = np.argsort(ps)
    adj, prev = [0.0] * len(ps), 0.0
    for rank, j in enumerate(order):
        prev = max(prev, min(1.0, (len(ps) - rank) * ps[j]))
        adj[j] = prev
    return adj


def report(a):
    rng = np.random.default_rng(0)
    R = os.path.join(a.runs, "short")
    B = json.load(open(os.path.join(a.data, "p27_build.json")))["patients"]
    learned, fixed = _load(R, "learned_primary"), _load(R, "fixed_primary")
    included = sorted(s for s, i in B.items() if i["excluded"] is None)
    L = ["# Phase 27: third cohort (Parmigiani et al. 2022), cohort-3 protocol v1.0", "",
         f"- included patients: {len(included)}: {', '.join(included)}"]
    for lab, D in (("learned", learned), ("fixed", fixed)):
        if sorted(D) != included:
            L.append(f"- **WARNING**: {lab} results exist for {len(D)} of {len(included)} patients: run incomplete")
    H = [("H1 (system, r)", "Final-short", "Base", compare(_ex(learned, "combo_nosess"), _ex(learned, "base"), rng)),
         ("H2 (learning, r)", "Learned-short", "Fixed-short",
          compare(_ex(learned, "combo_nosess"), _ex(fixed, "combo_nosess"), rng)),
         ("H3 (system, r_induced_ses)", "Final-short", "Base",
          compare(_ex(learned, "combo_nosess", "r_induced_ses"), _ex(learned, "base", "r_induced_ses"), rng))]
    adj = holm([h[3]["p"] for h in H])
    out = {}
    L += ["", "## Pre-specified hypotheses (two-sided Wilcoxon, Holm over H1-H3)", "",
          "| hypothesis | A | B | n | mean A | mean B | median A − B | mean A − B [95 % CI] | A > B | p | p (Holm) | supported |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (name, la, lb, h), pa in zip(H, adj):
        sup = pa < 0.05 and h["median_diff"] > 0
        h["p_holm"], h["supported"] = pa, bool(sup)
        out[name] = h
        L.append(f"| {name} | {la} | {lb} | {h['n']} | {h['mean_a']:.4f} | {h['mean_b']:.4f} | {h['median_diff']:+.4f} | "
                 f"{h['mean_diff']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] | {h['wins']}/{h['n']} | {h['p']:.2e} | "
                 f"{pa:.2e} | {'**yes**' if sup else 'no'} |")
    L += ["", "Per-patient differences (A − B):", ""]
    for name, _, _, h in H:
        L.append(f"- {name}: " + ", ".join(f"{s} {v:+.4f}" for s, v in h["per_patient"].items()))
    L += ["", "## Descriptive analyses (unadjusted p)", "",
          "| comparison | metric | n | mean A | mean B | mean A − B [95 % CI] | median A − B | A > B | p |",
          "|---|---|---|---|---|---|---|---|---|"]
    desc = []

    def row(lab, A, B, metric):
        h = compare(A, B, rng)
        if h is None:
            return
        desc.append({"comparison": lab, "metric": metric, **h})
        L.append(f"| {lab} | {metric} | {h['n']} | {h['mean_a']:.4f} | {h['mean_b']:.4f} | {h['mean_diff']:+.4f} "
                 f"[{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] | {h['median_diff']:+.4f} | {h['wins']}/{h['n']} | {h['p']:.2e} |")
    for metric in ("r_induced", "r_no50", "r2"):
        row("Final-short − Base", _ex(learned, "combo_nosess", metric), _ex(learned, "base", metric), metric)
    row("Learned-short − Fixed-short", _ex(learned, "combo_nosess", "r_induced_ses"),
        _ex(fixed, "combo_nosess", "r_induced_ses"), "r_induced_ses")
    row("Fixed-short − Base", _ex(fixed, "combo_nosess"), _ex(fixed, "base"), "r")
    row("Learned encoder alone − Base", _ex(learned, "enc"), _ex(learned, "base"), "r")
    l20 = _load(R, "learned_primary_f20")
    if l20:
        row("Final-short, 20 % calibration − Base, 20 %", _ex(l20, "combo_nosess"), _ex(l20, "base"), "r")
        row("Final-short, 20 % − Base, 30 %", _ex(l20, "combo_nosess"), _ex(learned, "base"), "r")
    out["descriptive"] = desc
    L += ["", "Mean over patients:", "", "| model | r | r_induced_ses | r_induced | r_no50 | R² | constant predictions |",
          "|---|---|---|---|---|---|---|"]
    for lab, D, m in (("Base", learned, "base"), ("Fixed-short (+ LAG9)", fixed, "combo_nosess"),
                      ("Learned encoder alone", learned, "enc"), ("Final-short (learned + LAG9)", learned, "combo_nosess")):
        v = [_ex(D, m, k) for k in ("r", "r_induced_ses", "r_induced", "r_no50", "r2")]
        nc = sum(r["explicit"][m].get("n_const", 0) for r in D.values() if m in r.get("explicit", {}))
        L.append(f"| {lab} | " + " | ".join(f"{np.mean(list(x.values())):.4f}" if x else "" for x in v) + f" | {nc} |")
    L += ["", "Per patient (r):", "", "| patient | epochs | calib min | EEG | targets | Base | Fixed-short | Final-short |",
          "|---|---|---|---|---|---|---|---|"]
    fb, ff, fl = _ex(learned, "base"), _ex(fixed, "combo_nosess"), _ex(learned, "combo_nosess")
    for s in included:
        i = B[s]
        L.append(f"| {s} | {i['n_epochs']} | {i['calib_minutes']:.2f} | {i['n_eeg']} | {i['n_valid_targets']} | "
                 f"{fb.get(s, float('nan')):.3f} | {ff.get(s, float('nan')):.3f} | {fl.get(s, float('nan')):.3f} |")
    md = "\n".join(L)
    print(md)
    open(os.path.join(a.runs, "p27_report.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(os.path.join(a.runs, "p27_report.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "runs", "report"))
    ap.add_argument("--src", help="build: Phase-26 folder with ccepcoreg/ and eeg_badtrials.json")
    ap.add_argument("--out", help="build: output data folder")
    ap.add_argument("--data", help="runs / report: data folder")
    ap.add_argument("--enc_root", help="runs: folder with short/enc_final.pt (Phase 17 encoders)")
    ap.add_argument("--runs", help="runs / report: run root")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    {"build": build, "runs": runs, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
