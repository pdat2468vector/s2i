"""
Phase 37 (review 6, C2; post hoc) - the CAST architecture of our earlier preprint (Pham and Tran 2026, arXiv
2605.18897) re-run under the protocol of the present study, on ds004752.

Kept from CAST (its code, model/cast.py and the ds004752 LOSO script): the architecture (multi-scale patches 8/16/32,
d = 128, 4 heads, 4 encoder and 2 decoder layers, dropout 0.2), the loss (MSE + (1 - r) + 0.1 x log-spectral), 2-s
windows (400 samples) with a 1-s stride for training, per-window z-scoring of the windows used for training, the
augmentation, leave-one-patient-out sequential encoder pre-training (2 rounds x 50 epochs per training patient,
AdamW lr 3e-4, weight decay 0.01, batch 16), and the decoder calibration on all contacts (200 epochs, encoder fine-tuned
at a tenth of the learning rate, best epoch by training loss).
Replaced by the protocol of the present study: bipolar targets and CAR scalp EEG (primary level, all 15 patients, all
contacts), strict normalisation (calibration statistics only, data of s2i.strict36), chronological calibration on the
first max(3, round(0.2 n)) trials with a one-trial gap, and the explicit-prediction r of every contact over the test
trials (s2i.metrics.per_contact_metrics). Test trials are tiled with 2-s windows (the last one aligned with the trial
end), each window is z-scored on its scalp EEG only, and overlapping predictions are averaged.
Also reported on the same predictions: CAST's own metric (each 2-s test window and its target z-scored separately, r on
the concatenated windows).

python -m s2i.cast37 run    --data DATA36 --out RUNS/p37_cast
python -m s2i.cast37 report --runs RUNS
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np

W, STRIDE, MAX_EEG = 400, 200, 21
ENC_KEYS = ("patch_embeds.", "scale_embed", "pos_encs.", "encoder.")


def build_model_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SinCos(nn.Module):
        def __init__(self, d, max_len=256):
            super().__init__()
            pe = torch.zeros(max_len, d)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    def pearson_loss(p, t):
        vx, vy = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
        return (1 - (vx * vy).sum(-1) / torch.sqrt((vx ** 2).sum(-1) * (vy ** 2).sum(-1) + 1e-8)).mean()

    def spectral_loss(p, t):
        return F.mse_loss(torch.log(torch.fft.rfft(p, dim=-1).abs() + 1e-8),
                          torch.log(torch.fft.rfft(t, dim=-1).abs() + 1e-8))

    class CAST(nn.Module):
        """Copy of model/cast.py (CAST v3) of the earlier preprint."""

        def __init__(self, c_eeg=21, c_ieeg=64, seq_len=400, d_model=128, n_heads=4, n_enc_layers=4,
                     n_dec_layers=2, dropout=0.2):
            super().__init__()
            self.c_ieeg, self.seq_len, self.d_model = c_ieeg, seq_len, d_model
            self.patch_sizes = [8, 16, 32]
            self.n_patches_per_scale = [seq_len // ps for ps in self.patch_sizes]
            self.total_tokens = sum(self.n_patches_per_scale)
            self.patch_embeds = nn.ModuleList([nn.Sequential(
                nn.Linear(c_eeg * ps, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model), nn.LayerNorm(d_model)) for ps in self.patch_sizes])
            self.scale_embed = nn.Parameter(torch.randn(len(self.patch_sizes), 1, d_model) * 0.02)
            self.pos_encs = nn.ModuleList([SinCos(d_model, n + 1) for n in self.n_patches_per_scale])
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True), num_layers=n_enc_layers)
            n_q = self.n_patches_per_scale[1]
            self.temporal_queries = nn.Parameter(torch.randn(1, n_q, d_model) * 0.02)
            self.channel_embed = nn.Embedding(c_ieeg, d_model)
            self.query_pos = SinCos(d_model, n_q + 1)
            self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True), num_layers=n_dec_layers)
            self.unpatch = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(d_model, 16))

        def forward(self, eeg):
            B = eeg.shape[0]
            toks = []
            for i, ps in enumerate(self.patch_sizes):
                p = eeg.unfold(2, ps, ps).permute(0, 2, 1, 3).reshape(B, self.n_patches_per_scale[i], -1)
                toks.append(self.pos_encs[i](self.patch_embeds[i](p)) + self.scale_embed[i])
            memory = self.encoder(torch.cat(toks, 1))
            n_q = self.temporal_queries.shape[1]
            mem = memory.unsqueeze(1).expand(B, self.c_ieeg, -1, -1).reshape(B * self.c_ieeg, self.total_tokens, -1)
            q = self.query_pos(self.temporal_queries + self.channel_embed(
                torch.arange(self.c_ieeg, device=eeg.device)).unsqueeze(1))
            q = q.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.c_ieeg, n_q, self.d_model)
            out = self.unpatch(self.decoder(q, mem)).reshape(B, self.c_ieeg, -1)
            return out[:, :, :self.seq_len]

        def compute_loss(self, eeg, ieeg):
            pred = self.forward(eeg)
            return F.mse_loss(pred, ieeg) + pearson_loss(pred, ieeg) + 0.1 * spectral_loss(pred, ieeg)

    return CAST


def zs(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)


def pad_eeg(e):
    return np.concatenate([e, np.zeros((MAX_EEG - e.shape[0], e.shape[1]), e.dtype)]) if e.shape[0] < MAX_EEG else e


def train_windows(E, Y):
    xs, ys = [], []
    for e, y in zip(E, Y):
        for s in range(0, e.shape[1] - W + 1, STRIDE):
            xs.append(zs(pad_eeg(e[:, s:s + W])))
            ys.append(zs(y[:, s:s + W]))
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def starts(L):
    s = list(range(0, L - W + 1, W))
    if s and s[-1] + W < L:
        s.append(L - W)
    return s


def windows_index(E):
    """(trial, start) of every training-style window (2 s, 1-s stride) of a patient."""
    return [(k, s) for k, e in enumerate(E) for s in range(0, e.shape[1] - W + 1, STRIDE)]


def load(path, eeg_key="eeg_car", tgt_key="ieeg_bip"):
    import torch
    d = torch.load(path, weights_only=False)
    eeg, tgt = d[eeg_key].float().numpy(), d[tgt_key].float().numpy()
    b = d["trial_bounds"].numpy()
    return d["subject"], [eeg[:, s0:s1] for s0, s1 in b], [tgt[:, s0:s1] for s0, s1 in b]


def run(a):
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, TensorDataset
    from s2i.metrics import per_contact_metrics
    CAST = build_model_class()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    seed = 42

    def fix(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)

    def augment(e, i):
        e, i = e + torch.randn_like(e) * 0.05, i + torch.randn_like(i) * 0.05
        s = 0.8 + 0.4 * torch.rand(e.shape[0], 1, 1, device=e.device)
        sh = torch.randint(-5, 6, (1,)).item()
        if sh:
            e, i = torch.roll(e, sh, -1), torch.roll(i, sh, -1)
        return e * s, i * s

    def model(n_ieeg):
        return CAST(c_eeg=MAX_EEG, c_ieeg=n_ieeg, seq_len=W, d_model=128, n_heads=4, n_enc_layers=4,
                    n_dec_layers=2, dropout=0.2).to(dev)

    def enc_of(m):
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if k.startswith(ENC_KEYS)}

    def loader(x, y, s):
        bs = min(16, len(x))
        return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=bs, shuffle=True,
                          drop_last=len(x) > bs, generator=torch.Generator().manual_seed(s))

    subs = {}
    for p in sorted(glob.glob(os.path.join(a.data, "sub-[0-9][0-9].pt"))):
        sid, E, Y = load(p, "eeg_car" if a.eeg == "car" else "eeg_orig", "ieeg_bip" if a.target == "bip" else "ieeg_mono")
        subs[sid] = (E, Y)
        print(f"{sid}: {len(E)} trials, {E[0].shape[0]} scalp, {Y[0].shape[0]} targets", flush=True)
    order = sorted(subs)
    for tsid in (a.only or order):                     # --only: folds run in the given order
        ti = order.index(tsid)                         # seeds depend on the patient, not on the run order
        fp = os.path.join(a.out, f"cast_{tsid}.json")
        if os.path.exists(fp):
            print(f"[{tsid}] done", flush=True)
            continue
        t0 = time.time()
        E, Y = subs[tsid]
        n = len(E)
        n_cal = max(3, int(round(n * 0.2)))
        cal, test = list(range(n_cal)), list(range(n_cal + 1, n))
        # ---- encoder: sequential pre-training on the other patients (resumable after every patient)
        ep = os.path.join(a.out, f"enc_{tsid}.pt")
        prog = os.path.join(a.out, f"progress_{tsid}.pt")
        train_sids = [s for s in order if s != tsid]
        if os.path.exists(ep):
            enc_state = torch.load(ep, weights_only=False)
        else:
            fix(seed + ti)
            base = model(Y[0].shape[0])
            done = 0
            if os.path.exists(prog):
                pr = torch.load(prog, weights_only=False)
                base.load_state_dict(pr["enc"], strict=False)
                done = pr["done"]
            steps = [(r_, s) for r_ in range(a.rounds) for s in train_sids]
            for k, (round_i, sid) in enumerate(steps):
                if k < done:
                    continue
                Es, Ys = subs[sid]
                x, y = train_windows(Es, Ys)
                tmp = model(Ys[0].shape[0])
                tmp.load_state_dict(enc_of(base), strict=False)
                opt = AdamW(tmp.parameters(), lr=3e-4, weight_decay=0.01)
                dl = loader(x, y, seed + round_i)
                for _ in range(a.epochs_pre):
                    tmp.train()
                    for e, i in dl:
                        e, i = augment(e.to(dev), i.to(dev))
                        loss = tmp.compute_loss(e, i)
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(tmp.parameters(), 1.0)
                        opt.step()
                base.load_state_dict(enc_of(tmp), strict=False)
                del tmp, opt
                torch.cuda.empty_cache()
                torch.save({"enc": enc_of(base), "done": k + 1}, prog + ".tmp")
                os.replace(prog + ".tmp", prog)
                print(f"  [{tsid}] pre-training {k + 1}/{len(steps)} ({sid}, round {round_i + 1}) | "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            enc_state = enc_of(base)
            torch.save(enc_state, ep)
            if os.path.exists(prog):
                os.remove(prog)
            del base
        # ---- decoder calibration on all contacts (CAST "ALL channels" setting)
        fix(seed + ti * 100)
        K = Y[0].shape[0]
        m = model(K)
        m.load_state_dict(enc_state, strict=False)
        enc_p = [p for nme, p in m.named_parameters() if nme.startswith(ENC_KEYS)]
        dec_p = [p for nme, p in m.named_parameters() if not nme.startswith(ENC_KEYS)]
        opt = AdamW([{"params": enc_p, "lr": 3e-5}, {"params": dec_p, "lr": 3e-4}], weight_decay=0.01)
        sched = CosineAnnealingLR(opt, T_max=a.epochs_dec)
        if a.split == "random_windows":
            # CAST's own split: every 2-s window (1-s stride) of the patient, permuted with seed 42, first 20 % for
            # calibration (int(0.2 N), at least 3), the rest for testing
            idx = windows_index(E)
            xa, ya = train_windows(E, Y)
            perm = np.random.RandomState(42).permutation(len(idx))
            n_cw = max(3, int(len(idx) * 0.2))
            cw, tw = perm[:n_cw], perm[n_cw:]
            x, y = xa[cw], ya[cw]
            calset = {idx[j] for j in cw}
            shared = float(np.mean([any((idx[j][0], idx[j][1] + d) in calset for d in (-STRIDE, STRIDE)) for j in tw]))
        else:
            x, y = train_windows([E[i] for i in cal], [Y[i] for i in cal])
        dl = loader(x, y, seed)
        best, best_st = float("inf"), None
        for _ in range(a.epochs_dec):
            m.train()
            tot = 0.0
            for e, i in dl:
                e, i = augment(e.to(dev), i.to(dev))
                loss = m.compute_loss(e, i)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                tot += loss.item()
            sched.step()
            if tot / len(dl) < best:
                best, best_st = tot / len(dl), {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(best_st)
        m.eval()
        if a.split == "random_windows":                     # CAST's metric: r of the concatenated z-scored windows
            with torch.no_grad():
                xt = torch.from_numpy(xa[tw])
                pw = np.concatenate([m(xt[j:j + 8].to(dev)).cpu().numpy() for j in range(0, len(tw), 8)])
            r_cast = per_contact_metrics([np.concatenate(list(pw), 1)], [np.concatenate(list(ya[tw]), 1)], 200,
                                         bands=False)["r"]
            res = {"subject": tsid, "split": "random_windows", "target": a.target, "eeg": a.eeg,
                   "n_windows": len(idx), "n_cal_windows": int(n_cw), "test_windows_sharing_1s_with_calibration": shared,
                   "n_targets": K, "r_cast_metric": float(np.nanmean(r_cast)), "minutes": (time.time() - t0) / 60}
            json.dump(res, open(fp + ".tmp", "w"))
            os.replace(fp + ".tmp", fp)
            print(f"[{tsid}] CAST protocol: r = {res['r_cast_metric']:.4f} | test windows sharing 1 s with "
                  f"calibration {shared:.2f} | {res['minutes']:.1f} min", flush=True)
            del m, opt
            torch.cuda.empty_cache()
            continue
        # ---- prediction of the test trials (2-s tiles, scalp EEG z-scored per window, overlaps averaged)
        P, Tt, wp, wt = [], [], [], []
        with torch.no_grad():
            for i in test:
                e, t = E[i], Y[i]
                L = e.shape[1]
                ss = starts(L)
                if not ss:
                    continue
                xw = torch.from_numpy(np.stack([zs(pad_eeg(e[:, s:s + W])) for s in ss]).astype(np.float32)).to(dev)
                pw = np.concatenate([m(xw[j:j + 8]).cpu().numpy() for j in range(0, len(ss), 8)])
                acc, cnt = np.zeros((K, L)), np.zeros(L)
                for s, p in zip(ss, pw):
                    acc[:, s:s + W] += p
                    cnt[s:s + W] += 1
                cov = cnt > 0
                P.append((acc[:, cov] / cnt[cov]).astype(np.float32))
                Tt.append(t[:, cov])
                for s, p in zip(ss, pw):
                    if s % W == 0:                                               # non-overlapping tiles only
                        wp.append(p)
                        wt.append(zs(t[:, s:s + W]))
        r_ours = per_contact_metrics(P, Tt, 200, bands=False)["r"]
        Wp, Wt = np.concatenate(wp, 1), np.concatenate(wt, 1)
        r_cast = per_contact_metrics([Wp], [Wt], 200, bands=False)["r"]
        res = {"subject": tsid, "n_trials": n, "n_cal": n_cal, "n_test": len(test), "n_targets": K,
               "r": float(np.nanmean(r_ours)), "r_cast_metric": float(np.nanmean(r_cast)),
               "r_contacts": [float(v) for v in r_ours], "minutes": (time.time() - t0) / 60}
        json.dump(res, open(fp + ".tmp", "w"))
        os.replace(fp + ".tmp", fp)
        print(f"[{tsid}] r = {res['r']:.4f} (CAST metric {res['r_cast_metric']:.4f}) | {res['minutes']:.1f} min",
              flush=True)
        del m, opt
        torch.cuda.empty_cache()


def report(a):
    from s2i.parm27 import _ex, _load, compare
    rng = np.random.default_rng(0)
    R = a.runs
    cast = {json.load(open(f))["subject"]: json.load(open(f)) for f in glob.glob(os.path.join(R, "p37_cast", "cast_sub-*.json"))}
    x = {s: v["r"] for s, v in cast.items()}
    xm = {s: v["r_cast_metric"] for s, v in cast.items()}
    fin = _load(os.path.join(R, "shared_bip_car"), "readout_combo5_strict_qc_none")
    base, final = _ex(fin, "base"), _ex(fin, "combo_nosess")
    L = ["# Phase 37: CAST re-run under the present protocol (ds004752, primary, strict; post hoc, unadjusted p)", "",
         f"- patients: {len(x)}; CAST mean r {np.mean(list(x.values())):.4f}; with its own metric on the same "
         f"predictions {np.mean(list(xm.values())):.4f}", "",
         "| A | B | n | mean A | mean B | A − B [95 % CI] | A > B | p |", "|---|---|---|---|---|---|---|---|"]
    out = {"mean_r": float(np.mean(list(x.values()))), "mean_r_cast_metric": float(np.mean(list(xm.values())))}
    for lab, A, B in (("CAST − LAG9 ridge", x, base), ("final model − CAST", final, x),
                      ("CAST metric − present metric (same predictions)", xm, x)):
        h = compare(A, B, rng)
        out[lab] = h
        L.append(f"| {lab.split(' − ')[0]} | {lab.split(' − ')[1]} | {h['n']} | {h['mean_a']:.4f} | {h['mean_b']:.4f} | "
                 f"{h['mean_diff']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] | {h['wins']}/{h['n']} | {h['p']:.2e} |")
    L += ["", "Per patient (r): " + ", ".join(f"{s} {v:.3f}" for s, v in sorted(x.items()))]
    ctrl = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "p37_cast_origprotocol", "cast_sub-*.json")))]
    reported = {"sub-01": 0.268, "sub-13": 0.784}      # CAST preprint, ds004752 LOSO, all channels
    L += ["", "Positive control (same code, CAST's own protocol: monopolar, original reference, random split of "
              "overlapping 2-s windows, CAST metric):"]
    for c in ctrl:
        L.append(f"- {c['subject']}: r = {c['r_cast_metric']:.3f} (CAST preprint {reported.get(c['subject'], float('nan')):.3f}); "
                 f"test windows sharing 1 s with calibration {c['test_windows_sharing_1s_with_calibration']:.2f}")
        out[f"control {c['subject']}"] = c
    if not ctrl:
        L.append("- not run yet")
    md = "\n".join(L)
    print(md)
    open(os.path.join(R, "cast37_report.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(os.path.join(R, "cast37_report.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--runs")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--epochs_pre", type=int, default=50)
    ap.add_argument("--epochs_dec", type=int, default=200)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--target", choices=("bip", "mono"), default="bip")
    ap.add_argument("--eeg", choices=("car", "orig"), default="car")
    ap.add_argument("--split", choices=("chrono", "random_windows"), default="chrono",
                    help="random_windows (with --target mono --eeg orig): positive control with CAST's own protocol")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
