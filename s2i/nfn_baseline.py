"""
External baseline: NeuroFlowNet (conditional normalizing flow, arXiv 2603.03354) re-implemented from the
architecture of its public code (github.com/hdy6438/NeuroFlowNet) and run under OUR clean protocol.

Architecture (as in the public code): MultiScaleConditionalFlow - n_scales scales of n_steps flow steps;
each step = invertible 1x1 convolution (QR init) + conditional affine coupling whose network is
conv(k3) -> ReLU -> conv(k1) -> ReLU -> multi-head self-attention over time -> conv(k3) -> (shift, log-scale),
shift = 10 tanh(.), log-scale clamped to [-5, 5]; the scalp EEG window is concatenated to the coupling
input at every step; standard-normal prior; trained by negative log-likelihood (AdamW, lr 1e-3,
weight decay 1e-2, batch 64, grad-norm clip 1); 1-s windows, training stride 20 samples, scalp windows
z-scored per window and channel.

Changes required by the clean protocol (stated in the paper):
  * one model per patient trained on the SAME calibration trials as every other model (first 20 % of the
    trials, chronological); early stopping on the last 20 % of those calibration trials (patience 10,
    criterion = our r), then refit on all calibration trials for the selected number of epochs;
  * the iEEG target is normalised with the per-channel mean / std of the CALIBRATION data, not with the
    statistics of each test window (which is target information and unavailable at deployment);
  * test trials are predicted with consecutive 1-s windows (overlaps averaged); r per contact on the
    concatenated test trials, as for all models.
  * the number of output channels must be divisible by 2^n_scales; missing channels are padded with
    independent noise channels that are ignored.
Predictions reported: nfn_mode (latent = 0), nfn_sample (one sample, as in the original evaluation) and
nfn_mean (average of 8 samples ~ conditional mean).

Usage
-----
python -m s2i.nfn_baseline --data DATA_DIR --out RUNS_DIR/nfn_baseline
"""
import argparse
import copy
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from s2i.ceiling import chrono_split
from s2i.metrics import per_contact_metrics
from s2i.shared import r_single_trial

FS, WIN = 200, 200


# ----------------------------------------------------------------------------- flow
class Coupling(nn.Module):
    def __init__(self, c, hid, ctx, heads):
        super().__init__()
        self.h = c // 2
        self.c1 = nn.Conv1d(self.h + ctx, hid, 3, padding=1)
        self.c2 = nn.Conv1d(hid, hid, 1)
        self.ln = nn.LayerNorm(hid)
        self.att = nn.MultiheadAttention(hid, heads, batch_first=True)
        self.c3 = nn.Conv1d(hid, 2 * (c - self.h), 3, padding=1)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    def st(self, xa, ctx):
        h = Fn.relu(self.c2(Fn.relu(self.c1(torch.cat([xa, ctx], 1)))))
        t = h.transpose(1, 2)
        q = self.ln(t)
        t = t + self.att(q, q, q, need_weights=False)[0]
        shift, logs = self.c3(t.transpose(1, 2)).chunk(2, 1)
        return 10 * torch.tanh(shift / 10), logs.clamp(-5, 5)

    def forward(self, x, ctx):
        xa, xb = x[:, :self.h], x[:, self.h:]
        shift, logs = self.st(xa, ctx)
        return torch.cat([xa, xb * logs.exp() + shift], 1), logs.sum((1, 2))

    def inverse(self, y, ctx):
        ya, yb = y[:, :self.h], y[:, self.h:]
        shift, logs = self.st(ya, ctx)
        return torch.cat([ya, (yb - shift) * (-logs).exp()], 1)


class Step(nn.Module):
    def __init__(self, c, hid, ctx, heads):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(c, c))
        self.W = nn.Parameter(q)
        self.cp = Coupling(c, hid, ctx, heads)

    def forward(self, x, ctx):
        x = torch.einsum("ij,bjl->bil", self.W, x)
        ld = x.shape[2] * torch.linalg.slogdet(self.W)[1]
        x, ld2 = self.cp(x, ctx)
        return x, ld + ld2

    def inverse(self, y, ctx):
        return torch.einsum("ij,bjl->bil", torch.linalg.inv(self.W), self.cp.inverse(y, ctx))


class Flow(nn.Module):
    def __init__(self, c_out, c_in, hid=128, n_steps=6, n_scales=2, heads=8):
        super().__init__()
        assert c_out % (2 ** n_scales) == 0
        self.scales, c = nn.ModuleList(), c_out
        for _ in range(n_scales):
            self.scales.append(nn.ModuleList([Step(c, hid, c_in, heads) for _ in range(n_steps)]))
            c //= 2
        self.c_out = c_out

    def log_prob(self, x, ctx):
        ld, zs = 0.0, []
        for steps in self.scales:
            for st in steps:
                x, l_ = st(x, ctx)
                ld = ld + l_
            half = x.shape[1] // 2
            zs.append(x[:, :half])
            x = x[:, half:]
        zs.append(x)
        prior = sum(-0.5 * (z ** 2 + math.log(2 * math.pi)).sum((1, 2)) for z in zs)
        return prior + ld

    def sample(self, ctx, temp=1.0):
        B, L = ctx.shape[0], ctx.shape[2]
        sizes, c = [], self.c_out
        for _ in self.scales:
            sizes.append(c // 2)
            c //= 2
        x = torch.randn(B, c, L, device=ctx.device) * temp
        for steps, s in zip(reversed(self.scales), reversed(sizes)):
            x = torch.cat([torch.randn(B, s, L, device=ctx.device) * temp, x], 1)
            for st in reversed(steps):
                x = st.inverse(x, ctx)
        return x


# ----------------------------------------------------------------------------- data
def load(path):
    d = torch.load(path, weights_only=False)
    eeg, tgt = d["eeg_car"].half().float().numpy(), d["ieeg_bip"].half().float().numpy()
    b = d["trial_bounds"].numpy()
    return [eeg[:, s0:s1] for s0, s1 in b], [tgt[:, s0:s1] for s0, s1 in b]


def zwin(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)


def windows(E, Y, idx, stride, mu, sd, k_pad, rng):
    X, T = [], []
    for i in idx:
        L = E[i].shape[1]
        for s in range(0, L - WIN + 1, stride):
            X.append(zwin(E[i][:, s:s + WIN]))
            y = (Y[i][:, s:s + WIN] - mu) / sd
            if k_pad:
                y = np.concatenate([y, rng.normal(size=(k_pad, WIN)).astype(np.float32)])
            T.append(y)
    return torch.tensor(np.stack(X), dtype=torch.float32), torch.tensor(np.stack(T), dtype=torch.float32)


@torch.no_grad()
def predict(model, E, idx, mu, sd, K, dev, mode, n_mean=8, bs=64):
    """Consecutive 1-s windows per trial (last one aligned to the end, overlaps averaged)."""
    model.eval()
    out = []
    for i in idx:
        L = E[i].shape[1]
        starts = list(range(0, L - WIN + 1, WIN))
        if starts[-1] != L - WIN:
            starts.append(L - WIN)
        ctx = torch.tensor(np.stack([zwin(E[i][:, s:s + WIN]) for s in starts]), dtype=torch.float32, device=dev)
        P = []
        for b0 in range(0, len(ctx), bs):
            c = ctx[b0:b0 + bs]
            if mode == "mode":
                p = model.sample(c, temp=0.0)
            elif mode == "sample":
                p = model.sample(c)
            else:
                p = sum(model.sample(c) for _ in range(n_mean)) / n_mean
            P.append(p[:, :K].cpu().numpy())
        P = np.concatenate(P)
        acc, cnt = np.zeros((K, L)), np.zeros(L)
        for p, s in zip(P, starts):
            acc[:, s:s + WIN] += p
            cnt[s:s + WIN] += 1
        out.append((acc / cnt * sd + mu).astype(np.float32))
    return out


def r_mean(P, T):
    return float(np.nanmean(per_contact_metrics(P, T, FS, bands=False)["r"]))


# ----------------------------------------------------------------------------- per patient
def train(E, Y, fit_idx, val_idx, a, dev, log, n_epochs=None):
    K = Y[0].shape[0]
    k_pad = (-K) % (2 ** a.n_scales)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    cal = np.concatenate([Y[i] for i in fit_idx], 1)
    mu, sd = cal.mean(1, keepdims=True), cal.std(1, keepdims=True) + 1e-6
    Xw, Tw = windows(E, Y, fit_idx, a.stride, mu, sd, k_pad, rng)
    model = Flow(K + k_pad, E[0].shape[0], a.hidden, a.n_steps, a.n_scales, a.heads).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    best, best_state, bad, best_ep = -1e9, None, 0, 0
    epochs = n_epochs if n_epochs is not None else a.max_epochs
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xw))[:a.steps_per_epoch * a.bs]
        tot = 0.0
        for b0 in range(0, len(perm), a.bs):
            j = perm[b0:b0 + a.bs]
            loss = -model.log_prob(Tw[j].to(dev), Xw[j].to(dev)).mean() / ((K + k_pad) * WIN)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        if n_epochs is not None:
            continue
        vr = r_mean(predict(model, E, val_idx, mu, sd, K, dev, "mode"), [Y[i] for i in val_idx])
        if vr > best:
            best, bad, best_ep, best_state = vr, 0, ep + 1, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            log(f"      ep {ep + 1:3d} nll/dim={tot / max(1, len(perm) // a.bs):+.3f} val r={vr:.4f}" + (" *" if bad == 0 else ""))
        if bad >= a.patience:
            break
    if n_epochs is None:
        model.load_state_dict(best_state)
    return model, mu, sd, best, best_ep


def analyze(E, Y, a, dev, log):
    n = len(E)
    cal, test = chrono_split(n, a.calib_frac, gap=1)
    n_fit = max(2, int(round(len(cal) * 0.8)))
    fit_idx, val_idx = cal[:n_fit], cal[n_fit:] or cal[-1:]
    _, _, _, best_val, best_ep = train(E, Y, fit_idx, val_idx, a, dev, log)
    log(f"    early stopping: best val r={best_val:.4f} at epoch {best_ep}; refit on all calibration trials")
    model, mu, sd, _, _ = train(E, Y, cal, None, a, dev, log, n_epochs=max(1, best_ep))
    Yt = [Y[i] for i in test]
    res = {"n_trials": n, "best_val_r": best_val, "epochs": best_ep, "models": {}}
    for mode in ("mode", "sample", "mean"):
        P = predict(model, E, test, mu, sd, Y[0].shape[0], dev, mode, a.n_mean)
        m = per_contact_metrics(P, Yt, FS, bands=False)
        ri = r_single_trial(P, Yt)
        res["models"][f"nfn_{mode}"] = {"r": float(np.nanmean(m["r"])), "r2": float(np.nanmean(m["r2"])),
                                        "r_induced": float(np.nanmean(ri)), "contacts": [float(x) for x in m["r"]]}
    log("    test: " + " ".join(f"{k} r={v['r']:.4f} r_ind={v['r_induced']:.4f}" for k, v in res["models"].items()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--calib_frac", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_steps", type=int, default=6)
    ap.add_argument("--n_scales", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max_epochs", type=int, default=500)
    ap.add_argument("--steps_per_epoch", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--n_mean", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "nfn.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    dev = torch.device(a.device)
    log(f"config: {vars(a)}")
    results = []
    for p in sorted(glob.glob(os.path.join(a.data, "sub-[0-9][0-9].pt"))):
        sid = os.path.basename(p)[:-3]
        if a.subjects and sid not in a.subjects:
            continue
        fp = os.path.join(a.out, f"nfn_{sid}.json")
        if os.path.exists(fp):
            results.append(json.load(open(fp)))
            log(f"[{sid}] already done, loaded")
            continue
        t0 = time.time()
        E, Y = load(p)
        if len(E) < 20:
            continue
        log(f"[{sid}] {len(E)} trials, {E[0].shape[0]} scalp, {Y[0].shape[0]} contacts")
        r = analyze(E, Y, a, dev, log)
        r["subject"] = sid
        with open(fp + ".tmp", "w") as fh:
            json.dump(r, fh)
        os.replace(fp + ".tmp", fp)
        results.append(r)
        log(f"[{sid}] done in {(time.time() - t0) / 60:.1f} min")
    if results:
        lines = ["| model | mean r | r_induced | R² |", "|---|---|---|---|"]
        for m in results[0]["models"]:
            v = [r["models"][m] for r in results]
            lines.append(f"| {m} | {np.mean([x['r'] for x in v]):.4f} | {np.mean([x['r_induced'] for x in v]):.4f} | "
                         f"{np.mean([x['r2'] for x in v]):.4f} |")
        md = "\n".join(lines)
        log("\n" + md)
        with open(os.path.join(a.out, "nfn.md"), "w", encoding="utf-8") as fh:
            fh.write(md + "\n")


if __name__ == "__main__":
    main()
