"""
Phase 5B (CPU) - protocol analysis: how much do the evaluation choices used in the literature inflate r?

Same data, same two models - base (LAG9 ridge) and enc (shared encoder trained WITHOUT the test patient +
ridge readout), penalties always chosen by 5-fold r inside the training data - only the protocol changes:

  clean            ours: first 20 % of trials (chronological) -> calibration, trials after a 1-trial gap -> test;
                   bipolar iEEG, CAR EEG
  rand_trials20    20 % of the trials drawn at random for calibration (interleaved in time with the test trials)
  rand_trials90    90 % random trials for training, 10 % for test (NeuroFlowNet-style amount and split)
  rand_seg80       every trial cut into 1-s segments, random 80/20 split of segments: test samples are temporally
                   adjacent to training samples (as with CAST's 50 %-overlapping 2-s windows split at random)
  mono_orig        clean split, but MONOPOLAR iEEG target and original-reference EEG (shared recording reference)
  mono_orig_seg80  monopolar + original reference + random 1-s segments (CAST-like data protocol)
Metrics (on the test samples)
  r_concat    ours: per-contact r over the concatenated test samples, mean over contacts
  r_window    per-contact r inside each 1-s test window, averaged
  r_nfn       NeuroFlowNet-style: per 1-s window the prediction is z-scored per channel, de-normalised with the
              TARGET window's own mean/std, and r is computed on the channel-flattened window; mean over windows
  r_nfn_null  the same metric for a random prediction (Gaussian noise): what the metric gives for free
Selection
  CAST-style "viable" patients: calibration K-fold r > 0.2; the mean is then reported on those patients only

Usage
-----
python -m s2i.protocol --data DATA_DIR --run RUNS_DIR/shared_bip_car --out RUNS_DIR/protocol
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

from s2i.baselines import DEFAULT_LAGS
from s2i.ceiling import ALPHAS, feat_lag
from s2i.metrics import per_contact_metrics
from s2i.readout import Fit, St, load_encoder, r_of, ridge_models, total
from s2i.shared import np_feat
from s2i.within import ALPHAS_WIDE, _blocks

FS, SEG = 200, 200
LAG9 = feat_lag(DEFAULT_LAGS)
VARIANTS = ("clean", "rand_trials20", "rand_trials90", "rand_seg80", "mono_orig", "mono_orig_seg80",
            "chrono90", "rand_trials80", "mono_car", "bip_orig")
# one factor changed at a time (paired over patients): (label, variant A, metric A, variant B, metric B)
ONE_FACTOR = [
    ("split: random vs chronological (20 % calibration)", "rand_trials20", "r_concat", "clean", "r_concat"),
    ("amount: 90 % vs 20 % training, both chronological", "chrono90", "r_concat", "clean", "r_concat"),
    ("split: random vs chronological (90 % training)", "rand_trials90", "r_concat", "chrono90", "r_concat"),
    ("units: random 1-s segments vs random trials (80 %)", "rand_seg80", "r_concat", "rand_trials80", "r_concat"),
    ("target: monopolar vs bipolar (CAR scalp, clean split)", "mono_car", "r_concat", "clean", "r_concat"),
    ("scalp reference: original vs CAR (bipolar target, clean split)", "bip_orig", "r_concat", "clean", "r_concat"),
    ("target + reference: monopolar/original vs bipolar/CAR", "mono_orig", "r_concat", "clean", "r_concat"),
    ("metric: NeuroFlowNet-style vs ours (same predictions, clean)", "clean", "r_nfn", "clean", "r_concat"),
]
METRICS = ("r_concat", "r_window", "r_nfn", "r_nfn_null")


def load(path):
    d = torch.load(path, weights_only=False)
    g = lambda k: d[k].half().float().numpy()                                  # noqa: E731
    return d["trial_bounds"].numpy(), {"car": g("eeg_car"), "orig": g("eeg_orig")}, \
        {"bip": g("ieeg_bip"), "mono": g("ieeg_mono")}


def split(variant, lens, rng):
    """-> (train units, test units, folds); a unit = (trial, start, stop)."""
    n = len(lens)
    trial_unit = lambda i: (i, 0, lens[i])                                     # noqa: E731
    if variant == "chrono90":
        n_tr = max(3, int(round(n * 0.9)))
        folds = [[trial_unit(i) for i in blk] for blk in _blocks(n_tr, 5)]
        return [trial_unit(i) for i in range(n_tr)], [trial_unit(i) for i in range(n_tr + 1, n)], folds
    if variant in ("clean", "mono_orig", "mono_car", "bip_orig"):
        n_c = max(3, int(round(n * 0.2)))
        folds = [[trial_unit(i) for i in blk] for blk in _blocks(n_c, 5)]
        return [trial_unit(i) for i in range(n_c)], [trial_unit(i) for i in range(n_c + 1, n)], folds
    if variant in ("rand_trials20", "rand_trials80", "rand_trials90"):
        n_tr = max(3, int(round(n * {"rand_trials20": 0.2, "rand_trials80": 0.8, "rand_trials90": 0.9}[variant])))
        perm = rng.permutation(n)
        tr, te = sorted(perm[:n_tr]), sorted(perm[n_tr:])
        folds = [[trial_unit(i) for i in sorted(f)] for f in np.array_split(rng.permutation(tr), 5)]
        return [trial_unit(i) for i in tr], [trial_unit(i) for i in te], folds
    segs = [(i, k * SEG, (k + 1) * SEG) for i in range(n) for k in range(lens[i] // SEG)]
    perm = rng.permutation(len(segs))
    n_tr = int(round(0.8 * len(segs)))
    tr = [segs[j] for j in sorted(perm[:n_tr])]
    te = [segs[j] for j in sorted(perm[n_tr:])]
    folds = [[tr[j] for j in sorted(f)] for f in np.array_split(rng.permutation(len(tr)), 5)]
    return tr, te, folds


def by_trial(units):
    out = {}
    for i, s0, s1 in units:
        out.setdefault(i, []).append((s0, s1))
    return out


def run_variant(feat, alphas, E, Y, train, test, folds):
    D, K = feat(E[train[0][0]]).shape[0], Y[0].shape[0]
    fold_of = {u: fi for fi, f in enumerate(folds) for u in f}
    st = [St(D, K) for _ in folds]
    for i, segs in by_trial(train).items():
        X = feat(E[i]).astype(np.float32)
        for s0, s1 in segs:
            st[fold_of[(i, s0, s1)]].add(X[:, s0:s1], Y[i][:, s0:s1])
    tot = total(st)
    m = Fit(lambda tr: ridge_models(tr, alphas), [(tot - f, f) for f in st], tot)
    cv = float(np.nanmean(r_of(m.oof[m.best])))
    P, T = [], []
    for i, segs in by_trial(test).items():
        pred = (feat(E[i]).astype(np.float64).T @ m.A + m.c).T
        for s0, s1 in segs:
            P.append(pred[:, s0:s1].astype(np.float32))
            T.append(Y[i][:, s0:s1])
    return P, T, cv


def windows(P, T):
    for p, t in zip(P, T):
        for k in range(p.shape[1] // SEG):
            yield p[:, k * SEG:(k + 1) * SEG], t[:, k * SEG:(k + 1) * SEG]


def metrics(P, T, rng):
    out = {"r_concat": float(np.nanmean(per_contact_metrics(P, T, FS, bands=False)["r"]))}
    rw, nfn, null = [], [], []
    for p, t in windows(P, T):
        pc, tc = p - p.mean(1, keepdims=True), t - t.mean(1, keepdims=True)
        den = np.sqrt((pc * pc).sum(1) * (tc * tc).sum(1))
        rw.append(np.where(den > 0, (pc * tc).sum(1) / np.maximum(den, 1e-12), np.nan))
        for q, store in ((p, nfn), (rng.normal(size=p.shape), null)):
            sd = q.std(1, keepdims=True)
            ok = (sd[:, 0] > 0) & (t.std(1) > 0)
            if ok.sum() < 2:
                continue
            z = (q[ok] - q[ok].mean(1, keepdims=True)) / sd[ok]
            d = z * t[ok].std(1, keepdims=True) + t[ok].mean(1, keepdims=True)
            store.append(np.corrcoef(d.ravel(), t[ok].ravel())[0, 1])
    out["r_window"] = float(np.nanmean(np.concatenate(rw))) if rw else float("nan")
    out["r_nfn"] = float(np.nanmean(nfn)) if nfn else float("nan")
    out["r_nfn_null"] = float(np.nanmean(null)) if null else float("nan")
    return out


REFS = {"mono_orig": ("orig", "mono"), "mono_orig_seg80": ("orig", "mono"), "mono_car": ("car", "mono"),
        "bip_orig": ("orig", "bip")}


def analyze(path, enc, dev, log, seed, res=None):
    bounds, eegs, tgts = load(path)
    lens = [int(s1 - s0) for s0, s1 in bounds]
    amp = dev.type == "cuda"
    models = {"base": (LAG9, ALPHAS), "enc": (np_feat(enc, dev, amp), ALPHAS_WIDE)}
    res = dict(res or {})
    for v in VARIANTS:
        if v in res:                                   # resume: only the variants still missing
            continue
        rng = np.random.default_rng(seed)
        train, test, folds = split(v, lens, rng)
        ref, tg = REFS.get(v, ("car", "bip"))
        E = [eegs[ref][:, s0:s1] for s0, s1 in bounds]
        Y = [tgts[tg][:, s0:s1] for s0, s1 in bounds]
        res[v] = {}
        for name, (feat, alphas) in models.items():
            t0 = time.time()
            P, T, cv = run_variant(feat, alphas, E, Y, train, test, folds)
            res[v][name] = {**metrics(P, T, np.random.default_rng(seed + 1)), "cv_r": cv}
            log(f"    {v:16s} {name:4s} " + " ".join(f"{k}={res[v][name][k]:+.3f}" for k in METRICS + ("cv_r",)) +
                f" | {time.time() - t0:.0f}s")
    return res


def summarize(results, out_dir, log):
    subs = sorted(results)
    lines = ["| protocol | metric | base | enc | enc / clean enc |", "|---|---|---|---|---|"]
    clean_enc = np.mean([results[s]["clean"]["enc"]["r_concat"] for s in subs])
    summ = {}
    for v in VARIANTS:
        for mt in METRICS:
            row = {m: float(np.nanmean([results[s][v][m][mt] for s in subs])) for m in ("base", "enc")}
            summ[f"{v}/{mt}"] = row
            lines.append(f"| {v} | {mt} | {row['base']:.3f} | {row['enc']:.3f} | {row['enc'] / clean_enc:.2f}x |")
    # literature-style recipes
    lines += ["", "| recipe | n patients | base | enc | enc / clean enc |", "|---|---|---|---|---|"]

    def recipe(name, v, mt, select=False):
        ss = [s for s in subs if not select or results[s][v]["enc"]["cv_r"] > 0.2]
        vals = {m: float(np.nanmean([results[s][v][m][mt] for s in ss])) if ss else float("nan") for m in ("base", "enc")}
        summ[f"recipe/{name}"] = {**vals, "n": len(ss)}
        lines.append(f"| {name} | {len(ss)} | {vals['base']:.3f} | {vals['enc']:.3f} | {vals['enc'] / clean_enc:.2f}x |")
    recipe("ours (clean, r_concat, all patients)", "clean", "r_concat")
    recipe("CAST-like (monopolar, orig ref, random segments, viable patients)", "mono_orig_seg80", "r_concat", True)
    recipe("NeuroFlowNet-like (random 90 % trials, target de-normalisation, flattened r)", "rand_trials90", "r_nfn")
    recipe("NeuroFlowNet metric on RANDOM predictions", "rand_trials90", "r_nfn_null")
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(0)
    lines += ["", "### one factor changed at a time (paired over patients; 95 % CI = patient bootstrap)",
              "| comparison | model | A | B | A − B | 95 % CI | A > B | Wilcoxon p |", "|---|---|---|---|---|---|---|---|"]
    for label, va, ma, vb, mb in ONE_FACTOR:
        for m in ("base", "enc"):
            x = np.array([results[s][va][m][ma] for s in subs])
            y = np.array([results[s][vb][m][mb] for s in subs])
            d = x - y
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
            try:
                p = float(wilcoxon(x, y).pvalue)
            except ValueError:
                p = float("nan")
            summ[f"one_factor/{label}/{m}"] = {"diff": float(d.mean()), "ci": [float(np.percentile(boot, 2.5)),
                                               float(np.percentile(boot, 97.5))], "p": p, "wins": int((d > 0).sum())}
            lines.append(f"| {label} | {m} | {x.mean():.3f} | {y.mean():.3f} | {d.mean():+.3f} | "
                         f"[{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}] | "
                         f"{(d > 0).sum()}/{len(d)} | {p:.4f} |")
    sel = [s for s in subs if results[s]["clean"]["enc"]["cv_r"] > 0.2]
    lines.append(f"\nPatient selection on the clean protocol (calibration K-fold r > 0.2): {len(sel)}/{len(subs)} "
                 f"patients, enc r = {np.mean([results[s]['clean']['enc']['r_concat'] for s in sel]):.3f} "
                 f"vs {np.mean([results[s]['clean']['enc']['r_concat'] for s in subs]):.3f} on all patients.")
    lines.append("Note: chrono90 is tested on the last 10 % of the trials (vs the last 80 % for clean), so the "
                 "'amount' comparison also changes the test period.")
    md = "\n".join(lines)
    log("\n" + md)
    with open(os.path.join(out_dir, "protocol.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(out_dir, "protocol.json"), "w") as fh:
        json.dump(summ, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="run with enc_<sub>.pt (encoders trained without that patient)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "protocol.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    dev = torch.device(a.device)
    torch.set_grad_enabled(False)
    log(f"config: {vars(a)}")
    results = {}
    for ep in sorted(glob.glob(os.path.join(a.run, "enc_sub-*.pt"))):
        sid = os.path.basename(ep)[4:-3]
        fp = os.path.join(a.out, f"protocol_{sid}.json")
        prev = json.load(open(fp)) if os.path.exists(fp) else {}
        if all(v in prev for v in VARIANTS):
            results[sid] = prev
            log(f"[{sid}] already done, loaded")
            continue
        t0 = time.time()
        log(f"[{sid}]" + (f" adding {[v for v in VARIANTS if v not in prev]}" if prev else ""))
        r = analyze(os.path.join(a.data, f"{sid}.pt"), load_encoder(ep, dev), dev, log, a.seed, prev)
        with open(fp + ".tmp", "w") as fh:
            json.dump(r, fh)
        os.replace(fp + ".tmp", fp)
        results[sid] = r
        log(f"[{sid}] done in {(time.time() - t0) / 60:.1f} min")
    if results:
        summarize(results, a.out, log)


if __name__ == "__main__":
    main()
