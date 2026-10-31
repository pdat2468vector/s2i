"""
Phase 1d (CPU) - WHY do random non-linear features beat the linear baseline?

Phase 1c (15 subjects, 20 % calibration = 4.1 min): every gated stage 2 beat
base = ridge_kfold_shared (15/15, p = 1e-4), random Morlet features most (+0.0062 vs +0.0032 for
per-channel spectral power), and the gain grew with calibration (1 -> 2 -> 4 min:
+0.0025 -> +0.0037 -> +0.0062). But the gain sat in delta/theta, not alpha, and a 2 Hz Morlet
spans ~ +-1.7 s while the ridge sees +-200 ms. So the gain may be LINEAR long temporal context,
not non-linearity. The two answers lead to different phase-2 models (and the first one means
the baseline a deep model must beat is stronger than base).

Matched ablation: every rocket_* model uses the SAME random spatial filters and wavelets
(seed 0), only the output non-linearity differs.

  ridge, base        as in Phase 1c
  base_long          ridge_kfold_shared with lags -1 s .. +1 s (100 ms step): direct long context
  lin_gated          base + gated stage 2 on Re(w * s)            (linear, long context)
  rocket_gated       base + gated stage 2, half envelope / half ReLU (= Phase 1c)
  env_only_gated     base + gated stage 2, all envelopes |w * s|
  relu_only_gated    base + gated stage 2, all ReLU(Re(w * s) - b)
  rocket2k_gated     rocket_gated with 4x more random features (capacity scaling)

Reading
  rocket ~ lin            -> gain = long linear context; non-linearity is not the lever
  rocket > lin clearly    -> genuine non-linear information; env_only vs relu_only says which kind
  rocket2k > rocket       -> more (and better: learned) features keep helping -> deep encoder

Usage
-----
python -m s2i.probe --data DATA_DIR --out RUNS_DIR/probe_bip_car
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from s2i.ceiling import StreamRidge, chrono_split, feat_lag
from s2i.metrics import per_contact_metrics
from s2i.nonlin import RandomConvFeatures, gated, summarize
from s2i.within import LAG9, KFoldRidge

FS = 200
LAGS_1S = tuple(range(-200, 201, 20))           # +-1 s, 100 ms step


def analyze(E, Y, a, log=print):
    n = len(E)
    _, test = chrono_split(n, 0.2, gap=1)
    Et, Yt = [E[i] for i in test], [Y[i] for i in test]
    C = E[0].shape[0]
    feats = {m: RandomConvFeatures(C, M=a.n_random, seed=a.seed, mode=mode)
             for m, mode in [("lin_gated", "linear"), ("rocket_gated", "mix"),
                             ("env_only_gated", "env"), ("relu_only_gated", "relu")]}
    if a.n_big:
        feats["rocket2k_gated"] = RandomConvFeatures(C, M=a.n_big, seed=a.seed, mode="mix")
    res = {"n_trials": n, "fracs": {}}
    for f in a.fracs:
        n_c = max(3, int(round(n * f)))
        if n_c >= test[0]:
            continue
        Ec, Yc = E[:n_c], Y[:n_c]
        bands = abs(f - 0.2) < 1e-9
        out, on = {}, {}

        def score(name, P):
            m = per_contact_metrics(P, Yt, FS, bands=bands)
            out[name] = {"mean": {k: float(np.nanmean(v)) for k, v in m.items()}, "r": [float(x) for x in m["r"]]}

        score("ridge", StreamRidge(LAG9).fit(Ec, Yc).predict(Et))
        base = KFoldRidge(LAG9, shared=True).fit(Ec, Yc)
        score("base", base.predict(Et))
        score("base_long", KFoldRidge(feat_lag(LAGS_1S), shared=True, keep_oof=False).fit(Ec, Yc).predict(Et))
        for name, feat in feats.items():
            P, on[name] = gated(base, Ec, Yc, Et, feat)
            score(name, P)
        mins = sum(e.shape[1] for e in Ec) / FS / 60
        res["fracs"][f"{f:g}"] = {"n_calib": n_c, "minutes": mins, "gate_on": on,
                                  "mean": {k: v["mean"]["r"] for k, v in out.items()},
                                  "bands": {k: v["mean"] for k, v in out.items()} if bands else {},
                                  "r": {k: v["r"] for k, v in out.items()}}
        log(f"    f={f:g} ({mins:.1f} min) " + " ".join(f"{k}={v['mean']['r']:+.3f}" for k, v in out.items()) +
            " | gate on: " + " ".join(f"{k}={v:.2f}" for k, v in on.items()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_trials", type=int, default=20)
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.1, 0.2])
    ap.add_argument("--n_random", type=int, default=512)
    ap.add_argument("--n_big", type=int, default=2048, help="0 = skip the capacity-scaling model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "probe.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    from s2i.ceiling import load_trials
    log(f"config: {vars(a)}")
    results = []
    for p in sorted(glob.glob(os.path.join(a.data, "sub-[0-9][0-9].pt"))):
        sid = os.path.basename(p)[:-3]
        fp = os.path.join(a.out, f"probe_{sid}.json")
        if os.path.exists(fp) and not a.overwrite:
            results.append(json.load(open(fp)))
            log(f"[{sid}] already done, loaded")
            continue
        t0 = time.time()
        s, E, Y = load_trials(p, "bip", "car")
        if len(E) < a.min_trials or s.n_targets == 0:
            log(f"[{sid}] skipped")
            continue
        log(f"[{sid}] {len(E)} trials, {len(s.eeg_names)} scalp, {s.n_targets} targets")
        r = analyze(E, Y, a, log)
        r["subject"] = sid
        with open(fp + ".tmp", "w") as fh:
            json.dump(r, fh)
        os.replace(fp + ".tmp", fp)
        results.append(r)
        log(f"[{sid}] done in {time.time() - t0:.0f}s")
    if results:
        summarize(results, a.out, log)


if __name__ == "__main__":
    main()
