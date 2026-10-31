"""
Phase 3a - calibration-size curve of the shared-encoder model (no retraining).

Uses the encoders saved by s2i.shared (enc_<sub>.pt, each trained WITHOUT that subject).
Fixed test set = last `test_frac` of trials after a 1-trial gap (as the Phase-0 curve);
calibration = the first f of trials. Models: ridge (train_loso baseline), base
(ridge_kfold_shared) and enc_ridge_cal (shared encoder + amplitude-calibrated ridge readout).
Question: how many minutes of calibration does the linear baseline need to match the encoder?

Usage
-----
python -m s2i.curve --data DATA_DIR --run RUNS_DIR/shared_bip_car
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

from s2i.ceiling import StreamRidge, load_trials
from s2i.metrics import per_contact_metrics
from s2i.shared import TemporalEncoder, amplitude_calibrate, np_feat
from s2i.within import ALPHAS_WIDE, LAG9, KFoldRidge

FS = 200
MODELS = ("ridge", "base", "enc_ridge_cal")


def analyze(E, Y, enc, a, dev, log):
    amp = dev.type == "cuda"
    n = len(E)
    t0 = int(round(n * (1 - a.test_frac)))
    test = list(range(min(n, t0 + 1), n))
    Et, Yt = [E[i] for i in test], [Y[i] for i in test]
    feat = np_feat(enc, dev, amp)
    res = {"n_trials": n, "n_test": len(test), "curve": {}}
    for f in a.fracs:
        n_c = int(round(n * f))
        if n_c < 3 or n_c > t0:
            continue
        Ec, Yc = E[:n_c], Y[:n_c]
        preds = {"ridge": StreamRidge(LAG9).fit(Ec, Yc).predict(Et),
                 "base": KFoldRidge(LAG9, shared=True, keep_oof=False).fit(Ec, Yc).predict(Et)}
        kf = KFoldRidge(feat, ALPHAS_WIDE, shared=True).fit(Ec, Yc)
        preds["enc_ridge_cal"] = amplitude_calibrate(kf, Yc, kf.predict(Et))
        row = {"n_calib": n_c, "minutes": sum(e.shape[1] for e in Ec) / FS / 60}
        for m, P in preds.items():
            mt = per_contact_metrics(P, Yt, FS, bands=False)
            row[m] = float(np.nanmean(mt["r"]))
            row[m + "_r2"] = float(np.nanmean(mt["r2"]))
        res["curve"][f"{f:g}"] = row
        log(f"    f={f:g} ({row['minutes']:.1f} min) " + " ".join(f"{m}={row[m]:.3f}" for m in MODELS))
    return res


def equivalent_minutes(xm, yb, target):
    """Minutes at which the (mean) baseline curve reaches `target`, by linear interpolation."""
    for i in range(1, len(xm)):
        if yb[i - 1] < target <= yb[i]:
            return xm[i - 1] + (target - yb[i - 1]) * (xm[i] - xm[i - 1]) / (yb[i] - yb[i - 1])
    return float("nan") if target > max(yb) else xm[0]


def summarize(results, out_dir, log):
    from scipy.stats import wilcoxon
    fracs = sorted({f for r in results for f in r["curve"]}, key=float)
    lines = ["| calib frac | n subj | minutes | " + " | ".join(MODELS) + " | enc − base | wins | Wilcoxon p | "
             "R² base | R² enc |", "|---" * (len(MODELS) + 8) + "|"]
    summ = {}
    for f in fracs:
        rows = [r["curve"][f] for r in results if f in r["curve"]]
        M = {m: np.array([r[m] for r in rows]) for m in MODELS}
        d = M["enc_ridge_cal"] - M["base"]
        try:
            p = float(wilcoxon(M["enc_ridge_cal"], M["base"]).pvalue)
        except ValueError:
            p = float("nan")
        summ[f] = {"n": len(rows), "minutes": float(np.mean([r["minutes"] for r in rows])),
                   **{m: float(M[m].mean()) for m in MODELS}, "diff": float(d.mean()), "wins": int((d > 0).sum()),
                   "p": p, "r2_base": float(np.mean([r["base_r2"] for r in rows])),
                   "r2_enc": float(np.mean([r["enc_ridge_cal_r2"] for r in rows]))}
        s = summ[f]
        lines.append(f"| {f} | {s['n']} | {s['minutes']:.1f} | " + " | ".join(f"{s[m]:.3f}" for m in MODELS) +
                     f" | {s['diff']:+.4f} | {s['wins']}/{s['n']} | {p:.4f} | {s['r2_base']:.3f} | {s['r2_enc']:.3f} |")

    # minutes-equivalence on subjects that have every fraction (consistent mean curves)
    full = [r for r in results if all(f in r["curve"] for f in fracs)]
    eq = []
    if len(full) >= 3:
        xm = [float(np.mean([r["curve"][f]["minutes"] for r in full])) for f in fracs]
        yb = [float(np.mean([r["curve"][f]["base"] for r in full])) for f in fracs]
        ye = [float(np.mean([r["curve"][f]["enc_ridge_cal"] for r in full])) for f in fracs]
        lines += ["", f"### minutes the baseline (`base`) needs to match the encoder ({len(full)} subjects with every "
                      "calibration size)",
                  "| encoder minutes | encoder r | base needs (min) | ratio |", "|---|---|---|---|"]
        for x, y in zip(xm, ye):
            need = equivalent_minutes(xm, yb, y)
            eq.append({"minutes": x, "enc_r": y, "base_minutes": need})
            lines.append(f"| {x:.1f} | {y:.3f} | " + (f"{need:.1f} | {need / x:.1f}x |" if np.isfinite(need)
                                                      else f"> {xm[-1]:.1f} (never reached) | - |"))
    md = "\n".join(lines)
    log("\n" + md)
    with open(os.path.join(out_dir, "curve.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(out_dir, "curve.json"), "w") as fh:
        json.dump({"by_frac": summ, "equivalence": eq}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="output dir of s2i.shared (contains enc_<sub>.pt)")
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.05, 0.1, 0.2, 0.3, 0.45, 0.59])
    ap.add_argument("--test_frac", type=float, default=0.4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    out = os.path.join(a.run, "curve")
    os.makedirs(out, exist_ok=True)
    log_f = open(os.path.join(out, "curve.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    dev = torch.device(a.device)
    log(f"config: {vars(a)}")
    results = []
    for ep in sorted(glob.glob(os.path.join(a.run, "enc_sub-*.pt"))):
        sid = os.path.basename(ep)[4:-3]
        fp = os.path.join(out, f"curve_{sid}.json")
        if os.path.exists(fp) and not a.overwrite:
            results.append(json.load(open(fp)))
            log(f"[{sid}] already done, loaded")
            continue
        t0 = time.time()
        ck = torch.load(ep, map_location=dev, weights_only=False)
        enc = TemporalEncoder(**ck["cfg"]).to(dev)
        enc.load_state_dict(ck["state"])
        _, E, Y = load_trials(os.path.join(a.data, f"{sid}.pt"), "bip", "car")
        log(f"[{sid}] {len(E)} trials")
        r = analyze(E, Y, enc, a, dev, log)
        r["subject"] = sid
        with open(fp + ".tmp", "w") as fh:
            json.dump(r, fh)
        os.replace(fp + ".tmp", fp)
        results.append(r)
        log(f"[{sid}] done in {time.time() - t0:.0f}s")
    if results:
        summarize(results, out, log)


if __name__ == "__main__":
    main()
