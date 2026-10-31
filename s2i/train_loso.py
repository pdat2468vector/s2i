"""
Leave-one-subject-out protocol for LeadFormer.

Per fold (test subject S):
  1. Joint pre-training on all other subjects (mixed, subject-balanced batches;
     old code trained subjects *sequentially* and threw the decoder away).
     Model selection = zero-shot r on held-out VALIDATION subjects
     (old code picked the epoch with the lowest *training* loss).
  2. Zero-shot evaluation on S (no data from S used at all).
  3. Calibration on the first `calib_frac` of S's trials (chronological), with an
     internal chronological validation split for early stopping; adapters +
     decoder are tuned, encoder frozen.
  4. Lagged-ridge baseline trained on exactly the same calibration trials.
  5. Everything evaluated on the remaining trials (after a 1-trial gap).

Primary metric: mean over ALL contacts of S, then mean over subjects. No
outcome-based subject/channel subsets.

Example
-------
python -m s2i.train_loso --data c:\\iEEG_EEG\\data\\s2i_v1 --out runs\\bip_car --target bip --eeg_ref car
"""
import argparse
import copy
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from s2i.baselines import LaggedRidge
from s2i.data.dataset import Subject, WindowDataset, chrono_split, collate
from s2i.inference import predict_trials, streaming_r
from s2i.losses import total_loss
from s2i.metrics import per_contact_metrics, summarize
from s2i.model.leadformer import LeadFormer, SubjectAdapter


# ----------------------------------------------------------------------------- utils
def fix_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def augment(b, p_drop=0.4, min_ch=4, noise=0.05):
    """Scalp-channel dropout (montage robustness), per-channel gain, additive noise."""
    eeg, mask = b["eeg"], b["eeg_mask"]
    B, C, _ = eeg.shape
    rate = torch.rand(B, 1, device=eeg.device) * p_drop
    drop = (torch.rand(B, C, device=eeg.device) < rate) & mask
    keep_n = (mask & ~drop).sum(1)
    drop[keep_n < min_ch] = False
    new_mask = mask & ~drop
    gain = 0.8 + 0.4 * torch.rand(B, C, 1, device=eeg.device)
    eeg = (eeg * gain + noise * torch.randn_like(eeg)) * new_mask[..., None]
    b = dict(b)
    b["eeg"], b["eeg_mask"] = eeg, new_mask
    return b


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_logger(path):
    log = logging.getLogger(path)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def build_model(a):
    return LeadFormer(d=a.d, heads=a.heads, enc_layers=a.enc_layers, dec_layers=a.dec_layers,
                      patch=a.patch, dropout=a.dropout)


# ----------------------------------------------------------------------------- stage 1
def pretrain(a, train_subj, val_subj, dev, log, ckpt_path=None):
    model = build_model(a).to(dev)
    log.info(f"  params: {model.count_parameters() / 1e6:.2f} M")
    tr_ds = WindowDataset(train_subj, {s.sid: list(range(s.n_trials)) for s in train_subj},
                          a.win, "train", k_max=a.k_max, crops_per_trial=1, seed=a.seed)
    sampler = WeightedRandomSampler(tr_ds.subject_balanced_weights(), a.steps * a.bs, replacement=True)
    tr_ld = DataLoader(tr_ds, a.bs, sampler=sampler, collate_fn=collate, num_workers=0, drop_last=True)
    va_trials = {s.sid: list(range(min(s.n_trials, a.val_trials))) for s in val_subj}
    va_ds = WindowDataset(val_subj, va_trials, a.win, "eval", stride=a.win)
    va_ld = DataLoader(va_ds, a.bs, shuffle=False, collate_fn=collate, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    total = a.epochs * a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05,
                                                anneal_strategy="cos")
    scaler = make_scaler(a.amp and dev.type == "cuda")
    best, best_state, bad, start = -1e9, None, 0, 0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, best_state, bad, start = ck["best"], ck["best_state"], ck["bad"], ck["epoch"] + 1
        log.info(f"  resumed from epoch {start} (best val r={best:.4f})")
    for ep in range(start, a.epochs):
        if bad >= a.patience:
            break
        model.train()
        t0, agg = time.time(), {"nll": 0.0, "r": 0.0, "stft": 0.0}
        for b in tr_ld:
            b = augment(to_dev(b, dev))
            with torch.autocast(device_type=dev.type, enabled=a.amp and dev.type == "cuda"):
                pred, lv = model(b["eeg"], b["eeg_pos"], b["eeg_mask"], b["tgt_pos"], b["tgt_mask"])
            loss, parts = total_loss(pred, lv, b["tgt"], b["tgt_mask"], a.w_r, a.w_s)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k in agg:
                agg[k] += parts[k] / a.steps
        val_r = streaming_r(model, va_ld, len(val_subj), [s.n_targets for s in val_subj], dev, a.amp)
        log.info(f"  ep {ep + 1:3d} | train nll={agg['nll']:.3f} r={agg['r']:.3f} stft={agg['stft']:.3f}"
                 f" | val zero-shot r={val_r:.4f} | {time.time() - t0:.0f}s")
        if val_r > best:
            best, bad = val_r, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ckpt_path:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "best": best, "best_state": best_state,
                        "bad": bad, "epoch": ep}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
        if bad >= a.patience:
            log.info("  early stop")
            break
    model.load_state_dict(best_state)
    return model, best


# ----------------------------------------------------------------------------- stage 3
def calibrate(a, base, subj, cal_trials, dev, log):
    model = copy.deepcopy(base).to(dev)
    adapter = SubjectAdapter(subj.n_targets, a.d).to(dev)
    enc, dec = model.param_groups()
    for p in enc:
        p.requires_grad_(False)
    fit_tr, fit_va = chrono_split(len(cal_trials), 0.8, gap=0)
    fit_tr = [cal_trials[i] for i in fit_tr]
    fit_va = [cal_trials[i] for i in fit_va] or fit_tr
    tr_ds = WindowDataset([subj], {subj.sid: fit_tr}, a.win, "train", k_max=a.k_max,
                          crops_per_trial=4, seed=a.seed)
    va_ds = WindowDataset([subj], {subj.sid: fit_va}, a.win, "eval", stride=a.win // 2)
    va_ld = DataLoader(va_ds, a.bs, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW([{"params": dec, "lr": a.calib_lr_dec, "weight_decay": a.wd},
                             {"params": adapter.parameters(), "lr": a.calib_lr_adapter, "weight_decay": 0.0}])
    scaler = make_scaler(a.amp and dev.type == "cuda")
    best = streaming_r(model, va_ld, 1, [subj.n_targets], dev, a.amp, adapter)
    log.info(f"  calib start (zero-shot on calib-val) r={best:.4f}")
    best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(adapter.state_dict()))
    bad, ep = 0, -1
    for ep in range(a.calib_epochs):
        model.train()
        ld = DataLoader(tr_ds, a.bs, shuffle=True, collate_fn=collate, drop_last=len(tr_ds) > a.bs)
        for b in ld:
            b = augment(to_dev(b, dev))
            with torch.autocast(device_type=dev.type, enabled=a.amp and dev.type == "cuda"):
                pred, lv = model(b["eeg"], b["eeg_pos"], b["eeg_mask"], b["tgt_pos"], b["tgt_mask"],
                                 b["cidx"], adapter)
            loss, _ = total_loss(pred, lv, b["tgt"], b["tgt_mask"], a.w_r, a.w_s)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(dec) + list(adapter.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
        r = streaming_r(model, va_ld, 1, [subj.n_targets], dev, a.amp, adapter)
        if r > best:
            best, bad = r, 0
            best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(adapter.state_dict()))
        else:
            bad += 1
            if bad >= a.calib_patience:
                break
    model.load_state_dict(best_state[0])
    adapter.load_state_dict(best_state[1])
    log.info(f"  calib done: best calib-val r={best:.4f} after {ep + 1} epochs")
    return model, adapter


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", choices=["bip", "mono"], default="bip")
    ap.add_argument("--eeg_ref", choices=["car", "orig"], default="car")
    ap.add_argument("--subjects", nargs="*", default=None, help="subset of subjects to load")
    ap.add_argument("--folds", nargs="*", default=None, help="test subjects to run (default: all)")
    ap.add_argument("--min_trials", type=int, default=20)
    # model
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=3)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    # training
    ap.add_argument("--win", type=int, default=800)
    ap.add_argument("--k_max", type=int, default=48)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=300, help="optimizer steps per epoch")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--n_val_subj", type=int, default=2)
    ap.add_argument("--val_trials", type=int, default=60)
    ap.add_argument("--w_r", type=float, default=1.0)
    ap.add_argument("--w_s", type=float, default=0.1)
    # calibration
    ap.add_argument("--calib_frac", type=float, default=0.2)
    ap.add_argument("--calib_epochs", type=int, default=40)
    ap.add_argument("--calib_patience", type=int, default=6)
    ap.add_argument("--calib_lr_dec", type=float, default=1e-4)
    ap.add_argument("--calib_lr_adapter", type=float, default=3e-3)
    ap.add_argument("--no_calib", action="store_true")
    ap.add_argument("--no_ridge", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.amp = not a.no_amp
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    log = make_logger(os.path.join(a.out, "run.log"))
    dev = torch.device(a.device)
    log.info(f"config: {vars(a)}")
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    files = sorted(f for f in os.listdir(a.data) if f.endswith(".pt") and "_normstats" not in f)
    subjects = []
    for f in files:
        sid = f[:-3]
        if a.subjects and sid not in a.subjects:
            continue
        s = Subject(os.path.join(a.data, f), a.target, a.eeg_ref)
        if s.n_trials < a.min_trials or s.n_targets == 0:
            log.info(f"skip {sid}: {s.n_trials} trials, {s.n_targets} targets")
            continue
        subjects.append(s)
        log.info(f"{sid}: {s.n_trials} trials, {len(s.eeg_names)} scalp, {s.n_targets} targets, "
                 f"diag={s.diagnostics}")
    folds = [s for s in subjects if not a.folds or s.sid in a.folds]

    all_res = []
    for fi, test in enumerate(folds):
        fold_path = os.path.join(a.out, f"fold_{test.sid}.json")
        if os.path.exists(fold_path):
            log.info(f"[{test.sid}] already done, loading")
            with open(fold_path) as f:
                all_res.append(json.load(f))
            continue
        fix_seed(a.seed + fi)
        log.info("=" * 78)
        log.info(f"[{fi + 1}/{len(folds)}] TEST {test.sid}")
        pool = [s for s in subjects if s.sid != test.sid]
        rng = np.random.default_rng(a.seed + fi)
        order = rng.permutation(len(pool))
        val_subj = [pool[i] for i in order[:a.n_val_subj]]
        train_subj = [pool[i] for i in order[a.n_val_subj:]]
        log.info(f"  train={[s.sid for s in train_subj]} val={[s.sid for s in val_subj]}")

        t0 = time.time()
        model_path = os.path.join(a.out, f"model_{test.sid}.pt")
        ckpt_path = os.path.join(a.out, f"ckpt_{test.sid}.pt")
        if os.path.exists(model_path):
            model = build_model(a).to(dev)
            saved = torch.load(model_path, map_location=dev, weights_only=False)
            model.load_state_dict(saved["state"])
            best_val = saved["best_val"]
            log.info(f"  loaded pretrained model (val r={best_val:.4f})")
        else:
            model, best_val = pretrain(a, train_subj, val_subj, dev, log, ckpt_path)
            torch.save({"state": model.state_dict(), "best_val": best_val}, model_path)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

        cal_tr, test_tr = chrono_split(test.n_trials, a.calib_frac, gap=1)
        log.info(f"  split: calib {len(cal_tr)} trials | test {len(test_tr)} trials")
        res = {"subject": test.sid, "val_subjects": [s.sid for s in val_subj], "best_val_r": best_val,
               "names": test.tgt_names, "region": test.tgt_region, "depth": test.tgt_depth,
               "n_calib_trials": len(cal_tr), "n_test_trials": len(test_tr),
               "calib_minutes": sum(test.trial_len(i) for i in cal_tr) / 200 / 60,
               "diagnostics": test.diagnostics}

        # zero-shot
        P, T, lv_test = predict_trials(model, test, test_tr, a.win, a.win // 2, dev, amp=a.amp)
        m = per_contact_metrics(P, T)
        res["zero_shot"] = {k: v.tolist() for k, v in m.items()}
        # observability predicted from calibration-trial INPUTS only (no targets used)
        _, _, lv_cal = predict_trials(model, test, cal_tr, a.win, a.win // 2, dev, amp=a.amp)
        res["pred_logvar_calib"] = lv_cal.tolist()
        log.info(f"  ZERO-SHOT  : {fmt(summarize(m))}")

        if not a.no_calib:
            cm, ad = calibrate(a, model, test, cal_tr, dev, log)
            P, T, _ = predict_trials(cm, test, test_tr, a.win, a.win // 2, dev, ad, amp=a.amp)
            m = per_contact_metrics(P, T)
            res["calibrated"] = {k: v.tolist() for k, v in m.items()}
            log.info(f"  CALIBRATED : {fmt(summarize(m))}")
            torch.save({"state": cm.state_dict(), "adapter": ad.state_dict(),
                        "n_targets": test.n_targets, "cal_trials": cal_tr, "test_trials": test_tr},
                       os.path.join(a.out, f"calib_{test.sid}.pt"))
            del cm, ad

        if not a.no_ridge:
            ce = [test.trial(i)[0].float().numpy() for i in cal_tr]
            cy = [test.trial(i)[1].float().numpy() for i in cal_tr]
            rg = LaggedRidge().fit(ce, cy)
            te = [test.trial(i)[0].float().numpy() for i in test_tr]
            ty = [test.trial(i)[1].float().numpy() for i in test_tr]
            m = per_contact_metrics(rg.predict(te), ty)
            res["ridge"] = {k: v.tolist() for k, v in m.items()}
            log.info(f"  RIDGE      : {fmt(summarize(m))}")

        res["minutes"] = (time.time() - t0) / 60
        with open(fold_path, "w") as f:
            json.dump(res, f)
        all_res.append(res)
        del model
        torch.cuda.empty_cache()

    report(all_res, a.out, log)


def fmt(s):
    keys = ["r", "r2", "r_delta", "r_theta", "r_alpha", "r_beta", "r_gamma", "env_gamma"]
    return " ".join(f"{k}={s[k]:.3f}" for k in keys if k in s)


def report(all_res, out, log):
    from scipy.stats import spearmanr, wilcoxon
    methods = [m for m in ("zero_shot", "calibrated", "ridge") if all(m in r for r in all_res)]
    lines = ["| Subject | " + " | ".join(f"{m} r" for m in methods) + " | " +
             " | ".join(f"{m} R²" for m in methods) + " |",
             "|---" * (1 + 2 * len(methods)) + "|"]
    per_subj = {m: [] for m in methods}
    for r in all_res:
        row = [r["subject"]]
        for m in methods:
            per_subj[m].append(np.nanmean(r[m]["r"]))
            row.append(f"{np.nanmean(r[m]['r']):.3f}")
        for m in methods:
            row.append(f"{np.nanmean(r[m]['r2']):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| **MEAN** | " + " | ".join(f"**{np.mean(per_subj[m]):.3f}**" for m in methods) +
                 " | " + " | ".join(f"{np.mean([np.nanmean(r[m]['r2']) for r in all_res]):.3f}"
                                    for m in methods) + " |")
    summary = {"n_subjects": len(all_res),
               "mean_r": {m: float(np.mean(per_subj[m])) for m in methods},
               "std_r": {m: float(np.std(per_subj[m])) for m in methods}}
    if "calibrated" in methods and "ridge" in methods and len(all_res) >= 5:
        st = wilcoxon(per_subj["calibrated"], per_subj["ridge"])
        summary["wilcoxon_calibrated_vs_ridge_p"] = float(st.pvalue)
    # depth breakdown (pooled contacts)
    depth = {}
    for m in methods:
        pools = {}
        for r in all_res:
            for dpt, val in zip(r["depth"], r[m]["r"]):
                pools.setdefault(dpt, []).append(val)
        depth[m] = {k: {"n": len(v), "mean_r": float(np.nanmean(v))} for k, v in pools.items()}
    summary["depth"] = depth
    # does predicted observability (from inputs only) rank contacts correctly?
    key = "calibrated" if "calibrated" in methods else "zero_shot"
    rhos = [spearmanr(-np.asarray(r["pred_logvar_calib"]), r[key]["r"]).correlation for r in all_res]
    summary["observability_spearman_mean"] = float(np.nanmean(rhos))
    top = []
    for r in all_res:
        lv = np.asarray(r["pred_logvar_calib"])
        k = max(1, len(lv) // 4)
        top.append(np.nanmean(np.asarray(r[key]["r"])[np.argsort(lv)[:k]]))
    summary["top25pct_predicted_observable_mean_r"] = float(np.mean(top))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")
    log.info("\n" + "\n".join(lines))
    log.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
