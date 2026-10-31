"""
Phase 34 (review 6, Section 3.2 items 2-3; plan paper/review6_experiments_plan.md, 24 Oct 2026; post hoc) -
leave-one-cohort-out training of the short (+-200 ms) encoder and a curve over the number of training patients.

Every encoder is trained exactly as the external-validation encoders (s2i.shared, episodic ridge-in-the-loop, seed 42,
2 early-stopping patients drawn from the pool), only on patients of OTHER cohorts. Patient ids are prefixed with their
cohort (ds: ds004752, so: ScaleOLDNEW, c3: third cohort) so that names never collide. The support of a training
episode is 240 s, as in the main analysis, except in the third cohort, whose episodes use 30 % of the patient's
epochs (its calibration fraction). Third-cohort patients enter training with their 32-channel montages (Phase 33).

Commands
  train   --pools ds=DIR so=DIR ... [--n_opt N --subset_seed S] -> OUT/enc_final.pt
  copy    copy OUT/enc_final.pt to RUN/enc_<sid>.pt for every patient file of DATA (input of s2i.readout)
  report  paired comparisons per held-out cohort -> RUNS34/loco34_report.md / .json

python -m s2i.loco34 train --pools ds=DATA_DIR so=DATA17/primary --out RUNS34/enc_c3
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
import types

import numpy as np

PATTERNS = {"ds": "sub-[0-9][0-9].pt", "so": "sub-P[0-9][0-9].pt", "c3": "sub-[0-9][0-9].pt"}
SHARED_DEFAULTS = dict(n_pairs=12, half=40, fmin=2.0, fmax=60.0, cycles=5.0, n_free=4, no_env=False, arch="morlet",
                       steps=3000, eval_every=150, patience=6, subj_per_step=2, sup_sec=240.0, n_query=8, lr=3e-4,
                       lr_lam=1e-2, lam0=1.0, objective="episodic", lr_head=1e-3, n_val_subj=2, val_trials=40,
                       amp=True, seed=42)


def load_pool(tag, d, min_trials=20):
    import torch
    out = []
    for p in sorted(glob.glob(os.path.join(d, PATTERNS[tag]))):
        x = torch.load(p, weights_only=False)
        eeg, tgt = x["eeg_car"].float().numpy(), x["ieeg_bip"].float().numpy()
        b = x["trial_bounds"].numpy()
        E, Y = [eeg[:, s0:s1] for s0, s1 in b], [tgt[:, s0:s1] for s0, s1 in b]
        if len(E) < min_trials or tgt.shape[0] == 0:
            continue
        out.append((f"{tag}:{x['subject']}", E, Y))
    return out


def train(a):
    import torch
    from s2i import shared
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "enc_final.pt")
    log_f = open(os.path.join(a.out, "loco34.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()
    if os.path.exists(path):
        log("already trained: " + path)
        return
    subjects = []
    for spec in a.pools:
        tag, d = spec.split("=", 1)
        s = load_pool(tag, d)
        log(f"pool {tag}: {len(s)} patients from {d}")
        subjects += s
    sa = types.SimpleNamespace(**{**SHARED_DEFAULTS, "seed": a.seed, "out": a.out, "steps": a.steps,
                                  "eval_every": a.eval_every})
    sa.sup_by = {sid: 0.3 * sum(e.shape[1] for e in E) / shared.FS for sid, E, _ in subjects if sid.startswith("c3:")}
    torch.manual_seed(a.seed)
    order = np.random.default_rng(a.seed).permutation(len(subjects))
    val = [subjects[i] for i in order[:sa.n_val_subj]]
    train_ = [subjects[i] for i in order[sa.n_val_subj:]]
    if a.n_opt:
        pick = np.random.default_rng(1000 + a.subset_seed).permutation(len(train_))[:a.n_opt]
        train_ = [train_[i] for i in sorted(pick)]
    log(f"optimisation ({len(train_)}): {[s[0] for s in train_]}")
    log(f"early stopping ({len(val)}): {[s[0] for s in val]}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ckpt = os.path.join(a.out, "ckpt.pt")
    enc, lam, best, hist = shared.train_encoder(sa, train_, val, dev, log, ckpt)
    torch.save({"state": enc.state_dict(), "cfg": enc.cfg, "lam": lam, "best_val": best, "hist": hist,
                "train_subjects": [s[0] for s in train_], "val_subjects": [s[0] for s in val], "seed": a.seed,
                "pools": a.pools, "n_opt": a.n_opt, "subset_seed": a.subset_seed}, path + ".tmp")
    os.replace(path + ".tmp", path)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    log(f"encoder: val r {hist[0][1]:.4f} (init) -> {best:.4f}, lam={lam:.3g}, {len(train_)} training patients | "
        f"{(time.time() - t0) / 60:.1f} min -> {path}")


def copy(a):
    os.makedirs(a.run, exist_ok=True)
    src = os.path.join(a.enc, "enc_final.pt")
    n = 0
    for p in sorted(glob.glob(os.path.join(a.data, "sub-*.pt"))):
        if p.endswith("_normstats.pt"):
            continue
        dst = os.path.join(a.run, "enc_" + os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        n += 1
    print(f"{src} -> {n} patients in {a.run}")


def report(a):
    from s2i.parm27 import _ex, _load, compare
    rng = np.random.default_rng(0)
    R, L, out = a.runs, ["# Phase 34: leave-one-cohort-out and training-set size (post hoc, unadjusted p)", ""], {}

    def get(rel):
        return _load(os.path.dirname(os.path.join(R, rel)), os.path.basename(rel))

    def block(title, rows, n_ref=None):
        nonlocal L
        L += [f"## {title}", "", "| comparison | n | mean A | mean B | A − B [95 % CI] | A > B | p |", "|---|---|---|---|---|---|---|"]
        res = {}
        for lab, A, B in rows:
            if not A or not B:
                L.append(f"| {lab} | missing | | | | | |")
                continue
            h = compare(A, B, rng)
            res[lab] = h
            L.append(f"| {lab} | {h['n']} | {h['mean_a']:.4f} | {h['mean_b']:.4f} | {h['mean_diff']:+.4f} "
                     f"[{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] | {h['wins']}/{h['n']} | {h['p']:.2e} |")
        L.append("")
        return res

    for cohort, fixed, within, loco, within_lab in a.sets:
        F, W, X = get(fixed), get(within), get(loco)
        fx, w, x = _ex(F, "combo_nosess"), _ex(W, "combo_nosess"), _ex(X, "combo_nosess")
        base = _ex(X or W, "base")
        out[cohort] = block(f"Held-out cohort: {cohort} (encoder mixed with the LAG9 ridge)", [
            (f"{within_lab} − fixed", w, fx), ("other cohorts − fixed", x, fx), (f"other cohorts − {within_lab}", x, w),
            ("other cohorts − LAG9 ridge", x, base), (f"{within_lab} − LAG9 ridge", w, base)])
        we, xe = _ex(W, "enc"), _ex(X, "enc")
        out[cohort + " | encoder alone"] = block(f"Held-out cohort: {cohort} (encoder alone)", [
            (f"other cohorts − {within_lab}", xe, we)])
    if a.curve:
        F = _ex(get(a.curve_fixed), "combo_nosess")
        rows, cur = [], {}
        for spec in a.curve:
            n, rel = spec.split("=", 1)
            x = _ex(get(rel), "combo_nosess")
            rows.append((f"{n} training patients − fixed", x, F))
        cur = block("Training-set size (third cohort, 32 channels; learned − fixed, mixed with the LAG9 ridge)", rows)
        out["curve"] = cur
    md = "\n".join(L)
    print(md)
    open(os.path.join(R, "loco34_report.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(os.path.join(R, "loco34_report.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "copy", "report"))
    ap.add_argument("--pools", nargs="*", default=[])
    ap.add_argument("--out")
    ap.add_argument("--n_opt", type=int, default=0)
    ap.add_argument("--subset_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=150)
    ap.add_argument("--enc")
    ap.add_argument("--run")
    ap.add_argument("--data")
    ap.add_argument("--runs")
    ap.add_argument("--sets", nargs=5, action="append", default=[],
                    metavar=("COHORT", "FIXED", "WITHIN", "LOCO", "WITHIN_LABEL"),
                    help="readout folders relative to --runs")
    ap.add_argument("--curve", nargs="*", default=[], help="N=readout folder (relative to --runs)")
    ap.add_argument("--curve_fixed")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    {"train": train, "copy": copy, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
