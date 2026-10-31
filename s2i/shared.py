"""
Phase 2 - shared temporal encoder + per-subject ridge decoder (LOSO, GPU).

Evidence from phases 0-1d (15 subjects, bipolar iEEG, CAR EEG, 20 % chronological calibration):
  * the SPATIAL scalp -> contact mapping does not transfer across subjects (1a: zero-shot r ~ 0);
  * within a subject the lever is frequency-selective, long-support LINEAR temporal filtering
    (1d: random Morlet filter bank, linear, gated on top of the ridge: +0.0075, 14/15 subjects),
    while non-linearity and more random features add ~nothing at 4 min of calibration.
Hypothesis: the TEMPORAL part (which filters of the scalp signal predict intracranial activity)
is shared across subjects even though the spatial part is not. Hence:
  encoder : per-channel filter bank, shared by all channels and all subjects, trained on the
            OTHER subjects. Quadrature pairs initialised as complex Morlet 2-60 Hz (+-1.2 s
            support) + the envelope of each pair + a few free filters.
  decoder : per-subject linear readout (channels x filters -> contacts), fitted in closed form by
            ridge on the calibration trials only, exactly like the baselines.
Training is episodic so it matches deployment: each step takes a training subject, fits a ridge
readout on ~4 min of consecutive trials (support), scores Pearson r on the following trials
(query), and back-propagates through the ridge solve into the filters. Model selection: the
same procedure on 2 held-out validation subjects (first 20 % of trials -> later trials).

Evaluation on the test subject (never seen in training; same split and calibration sizes as
phases 0-1d, so ridge / base / lin_gated must reproduce the phase-1d numbers):
  ridge, base, lin_gated  baselines (lin_gated = best model of phase 1d: the bar to beat)
  enc0_gated              base + gated stage 2 on the encoder AT INITIALISATION (untrained Morlet
                          bank): control - does the gain come from learning or from the init?
  lin_ridge, enc0_ridge   single-stage controls: ridge (K-fold, shared alpha) on the random linear
                          Morlet bank / on the UNTRAINED encoder
  enc_ridge               ridge (K-fold, shared alpha) on learned encoder features alone
  enc_ridge_cal           enc_ridge with a per-contact amplitude gain fitted on out-of-fold
                          calibration predictions (same r, honest R^2)
Every model is also scored with
  r_no50     46-54 Hz removed from prediction and target (the trained filters develop a sharp 50 Hz
             peak: mains residue after the notch filter);
  r_induced  each side minus its own trial average over the test trials, i.e. how well
             the SINGLE-TRIAL (non stimulus-locked) part is predicted.
  enc_gated               base + gated stage 2 on learned encoder features      <- main model

Usage
-----
python -m s2i.shared --data DATA_DIR --out RUNS_DIR/shared_bip_car
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt

from s2i.ceiling import StreamRidge, chrono_split, load_trials, template, tmpl_at
from s2i.metrics import per_contact_metrics
from s2i.nonlin import RandomConvFeatures, gated, summarize
from s2i.within import ALPHAS_WIDE, LAG9, KFoldRidge

FS = 200
EXTRA_METRICS = ("r_no50", "r_induced")


# ----------------------------------------------------------------------------- encoder
class TemporalEncoder(nn.Module):
    """Per-channel filter bank shared by all channels and subjects: (B, C, L) -> (B, C, F, L)."""

    def __init__(self, n_pairs=12, half=240, fmin=2.0, fmax=60.0, cycles=5.0, n_free=4, envelope=True, arch="morlet"):
        super().__init__()
        self.cfg = dict(n_pairs=n_pairs, half=half, fmin=fmin, fmax=fmax, cycles=cycles, n_free=n_free,
                        envelope=envelope)
        if arch != "morlet":
            self.cfg["arch"] = arch
        self.arch = arch
        if arch == "shallow":
            # ShallowConvNet-style temporal front end (Schirrmeister et al. 2017), applied to every channel separately:
            # 40 filters of 25 samples, random (Kaiming) initialisation; features = the filter outputs and the log of
            # their squared output smoothed over 75 samples (the square-pool-log stage, without down-sampling)
            self.n_pairs, self.envelope, self.n_shallow = 0, envelope, 40
            self.conv = nn.Conv1d(1, self.n_shallow, 2 * half + 1, padding=half, bias=False)
            with torch.no_grad():
                g = torch.Generator().manual_seed(0)
                self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g)
                                       * np.sqrt(2.0 / (2 * half + 1)))
            return
        self.n_pairs, self.envelope = n_pairs, envelope
        t = np.arange(-half, half + 1) / FS
        f = np.geomspace(fmin, fmax, n_pairs)
        gauss = np.exp(-t[None] ** 2 / (2 * (cycles / (2 * np.pi * f[:, None])) ** 2))
        w = np.zeros((2 * n_pairs + n_free, t.size))
        w[0:2 * n_pairs:2] = gauss * np.cos(2 * np.pi * f[:, None] * t[None])
        w[1:2 * n_pairs:2] = gauss * np.sin(2 * np.pi * f[:, None] * t[None])
        if n_free:
            w[2 * n_pairs, half] = 1.0                                   # identity (broadband, lag 0)
            g = np.random.default_rng(0)
            w[2 * n_pairs + 1:] = g.normal(size=(n_free - 1, t.size)) * np.exp(-t ** 2 / (2 * 0.05 ** 2))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        self.conv = nn.Conv1d(1, w.shape[0], 2 * half + 1, padding=half, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor(w, dtype=torch.float32)[:, None])

    @property
    def n_feat(self):
        if self.arch == "shallow":
            return self.n_shallow * (2 if self.envelope else 1)
        return self.conv.out_channels + (self.n_pairs if self.envelope else 0)

    def forward(self, x):
        B, C, L = x.shape
        h = self.conv(x.reshape(B * C, 1, L))
        if self.arch == "shallow":
            if self.envelope:
                p = nn.functional.avg_pool1d(h.float() ** 2, 75, stride=1, padding=37, count_include_pad=False)
                h = torch.cat([h.float(), torch.log(p[..., :L] + 1e-6)], 1)
            return h.reshape(B, C, -1, L)
        if self.envelope:
            p = h[:, :2 * self.n_pairs].float()
            h = torch.cat([h.float(), torch.sqrt(p[:, 0::2] ** 2 + p[:, 1::2] ** 2 + 1e-6)], 1)
        return h.reshape(B, C, -1, L)


# ----------------------------------------------------------------------------- torch helpers
def pad_trials(E, Y, idx, dev):
    L = max(E[i].shape[1] for i in idx)
    x = np.zeros((len(idx), E[idx[0]].shape[0], L), np.float32)
    y = np.zeros((len(idx), Y[idx[0]].shape[0], L), np.float32)
    m = np.zeros((len(idx), L), bool)
    for j, i in enumerate(idx):
        n = E[i].shape[1]
        x[j, :, :n], y[j, :, :n], m[j, :n] = E[i], Y[i], True
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev), torch.from_numpy(m).to(dev)


def rows(enc, E, Y, idx, dev, amp, chunk=16):
    """Encoder features and targets of the given trials as (N, C*F) / (N, K) rows (padding removed)."""
    X, T = [], []
    for s in range(0, len(idx), chunk):
        x, y, m = pad_trials(E, Y, idx[s:s + chunk], dev)
        with torch.autocast(device_type=dev.type, enabled=amp):
            h = enc(x)
        B, C, F, L = h.shape
        X.append(h.float().permute(0, 3, 1, 2).reshape(B, L, C * F)[m])
        T.append(y.permute(0, 2, 1)[m])
    return torch.cat(X), torch.cat(T)


def ridge_torch(Xs, Ys, Xq, lam):
    """Ridge on standardised features; lam is relative to the mean eigenvalue (= 1 after scaling)."""
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-4
    A, Q = (Xs - mu) / sd, (Xq - mu) / sd
    ym = Ys.mean(0)
    N, D = A.shape
    G = A.T @ A / N + lam * torch.eye(D, device=A.device)
    return Q @ torch.linalg.solve(G, A.T @ (Ys - ym) / N) + ym


def col_r(P, T):
    P, T = P - P.mean(0), T - T.mean(0)
    return (P * T).sum(0) / ((P * P).sum(0).sqrt() * (T * T).sum(0).sqrt() + 1e-8)


def episode(n_trials, mean_len, rng, sup_sec, n_query):
    """Chronological episode: ~sup_sec of consecutive support trials, gap 1, then n_query trials."""
    n_q = min(n_query, max(1, n_trials // 4))
    n_s = int(np.clip(round(sup_sec * FS / mean_len), 3, n_trials - n_q - 1))
    start = int(rng.integers(0, n_trials - n_s - n_q - 1 + 1))
    return list(range(start, start + n_s)), list(range(start + n_s + 1, start + n_s + 1 + n_q))


@torch.no_grad()
def val_score(enc, lam, subjects, dev, amp, max_test):
    enc.eval()
    rs = []
    for _, E, Y in subjects:
        cal, test = chrono_split(len(E), 0.2, gap=1)
        Xs, Ys = rows(enc, E, Y, cal, dev, amp)
        Xq, Yq = rows(enc, E, Y, test[:max_test], dev, amp)
        rs.append(col_r(ridge_torch(Xs, Ys, Xq, lam), Yq).mean().item())
    return float(np.mean(rs))


# ----------------------------------------------------------------------------- training
def train_encoder(a, train, val, dev, log, ckpt_path):
    enc = TemporalEncoder(**enc_cfg(a)).to(dev)
    log_lam = nn.Parameter(torch.tensor(float(np.log(a.lam0)), device=dev))
    objective = getattr(a, "objective", "episodic")
    groups = [{"params": enc.parameters(), "lr": a.lr, "weight_decay": 0.0}]
    if objective == "episodic":
        groups.append({"params": [log_lam], "lr": a.lr_lam, "weight_decay": 0.0})
    heads = None
    if objective == "multitask":                      # one linear readout per training patient, learned by gradient
        heads = nn.ModuleDict({sid: nn.Linear(E[0].shape[0] * enc.n_feat, Y[0].shape[0]) for sid, E, Y in train}).to(dev)
        for h in heads.values():                      # small random init: a constant prediction has no r gradient
            nn.init.normal_(h.weight, std=1e-3)
            nn.init.zeros_(h.bias)
        groups.append({"params": heads.parameters(), "lr": a.lr_head, "weight_decay": 1e-2})
    opt = torch.optim.AdamW(groups)
    amp = a.amp and dev.type == "cuda"
    rng = np.random.default_rng(a.seed)
    mean_len = {sid: float(np.mean([e.shape[1] for e in E])) for sid, E, _ in train}
    step, best, best_state, bad, hist = 0, -1e9, None, 0, []
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        enc.load_state_dict(ck["enc"])
        if heads is not None and "heads" in ck:
            heads.load_state_dict(ck["heads"])
        opt.load_state_dict(ck["opt"])
        with torch.no_grad():
            log_lam.copy_(ck["log_lam"])
        step, best, best_state, bad, hist = ck["step"], ck["best"], ck["best_state"], ck["bad"], ck["hist"]
        rng = np.random.default_rng(a.seed + step)
        log(f"  resumed at step {step} (best val r={best:.4f})")
    else:
        r0 = val_score(enc, log_lam.exp().item(), val, dev, amp, a.val_trials)
        hist.append((0, r0))
        log(f"  step     0 | val r (Morlet init) = {r0:.4f}")
        best, best_state = r0, (copy.deepcopy(enc.state_dict()), log_lam.exp().item())
    t0, run_r = time.time(), []
    while step < a.steps and bad < a.patience:
        enc.train()
        loss = 0.0
        for si in rng.choice(len(train), a.subj_per_step, replace=len(train) < a.subj_per_step):
            sid, E, Y = train[si]
            if heads is not None:
                n_b = min(2 * a.n_query, len(E))
                s0 = int(rng.integers(0, len(E) - n_b + 1))
                Xb, Yb = rows(enc, E, Y, list(range(s0, s0 + n_b)), dev, amp)
                Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-4)
                r = col_r(heads[sid](Xb), Yb).mean()
            else:
                sup, qry = episode(len(E), mean_len[sid], rng, getattr(a, "sup_by", {}).get(sid, a.sup_sec), a.n_query)
                Xs, Ys = rows(enc, E, Y, sup, dev, amp)
                Xq, Yq = rows(enc, E, Y, qry, dev, amp)
                r = col_r(ridge_torch(Xs, Ys, Xq, log_lam.exp()), Yq).mean()
            loss = loss + (1 - r) / a.subj_per_step
            run_r.append(r.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        step += 1
        if step % a.eval_every == 0 or step == a.steps:
            vr = val_score(enc, log_lam.exp().item(), val, dev, amp, a.val_trials)
            hist.append((step, vr))
            if vr > best:
                best, bad, best_state = vr, 0, (copy.deepcopy(enc.state_dict()), log_lam.exp().item())
            else:
                bad += 1
            log(f"  step {step:5d} | train query r={np.mean(run_r):.4f} | val r={vr:.4f} | "
                f"lam={log_lam.exp().item():.3g} | {time.time() - t0:.0f}s" + (" *" if bad == 0 else ""))
            run_r, t0 = [], time.time()
            if ckpt_path:
                torch.save({"enc": enc.state_dict(), "opt": opt.state_dict(), "log_lam": log_lam.detach(),
                            "step": step, "best": best, "best_state": best_state, "bad": bad, "hist": hist,
                            **({"heads": heads.state_dict()} if heads is not None else {})},
                           ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)
    enc.load_state_dict(best_state[0])
    return enc, best_state[1], best, hist


def enc_cfg(a):
    return dict(n_pairs=a.n_pairs, half=a.half, fmin=a.fmin, fmax=a.fmax, cycles=a.cycles,
                n_free=a.n_free, envelope=not a.no_env, arch=getattr(a, "arch", "morlet"))


# ----------------------------------------------------------------------------- evaluation
def np_feat(enc, dev, amp):
    """Encoder as a numpy feature function (C, L) -> (C*F, L) for the StreamRidge machinery."""
    enc = enc.eval()

    @torch.no_grad()
    def f(e):
        x = torch.from_numpy(np.ascontiguousarray(e, np.float32))[None].to(dev)
        with torch.autocast(device_type=dev.type, enabled=amp):
            h = enc(x)[0]
        return h.float().reshape(-1, h.shape[-1]).cpu().numpy()
    return f


_LINE = butter(4, (46, 54), btype="bandstop", fs=FS, output="sos")


def r_without_line(P, T):
    """Per-contact r after removing 46-54 Hz (mains residue) from BOTH prediction and target."""
    fp = [sosfiltfilt(_LINE, p, axis=1) for p in P]
    ft = [sosfiltfilt(_LINE, t, axis=1) for t in T]
    return per_contact_metrics(fp, ft, FS, bands=False)["r"]


def r_single_trial(P, T):
    """
    Per-contact r of the single-trial parts: each side minus ITS OWN trial average over the test
    trials (prediction minus the mean prediction, target minus the mean target). Removing the same
    template from both sides would add a shared component and inflate r.
    """
    tP, tT = template(P), template(T)
    return per_contact_metrics([p - tmpl_at(tP, p.shape[1]) for p in P],
                               [t - tmpl_at(tT, t.shape[1]) for t in T], FS, bands=False)["r"]


def amplitude_calibrate(kf, Yc, P):
    """
    Per-contact least-squares gain on the OUT-OF-FOLD calibration predictions. Alpha is chosen by
    r, which ignores amplitude; this fixes the amplitude (R^2) without changing r (gain >= 0).
    """
    ym = kf.s["ym"][:, None]
    num = sum(((o - ym) * (y - ym)).sum(1) for o, y in zip(kf.oof, Yc))
    den = sum(((o - ym) ** 2).sum(1) for o in kf.oof)
    g = np.clip(num / np.maximum(den, 1e-12), 0, None)[:, None]
    return [(ym + g * (p - ym)).astype(np.float32) for p in P]


def evaluate(E, Y, enc0, enc, a, dev, log):
    amp = a.amp and dev.type == "cuda"
    n = len(E)
    _, test = chrono_split(n, 0.2, gap=1)
    Et, Yt = [E[i] for i in test], [Y[i] for i in test]
    f0, f1 = np_feat(enc0, dev, amp), np_feat(enc, dev, amp)
    lin = RandomConvFeatures(E[0].shape[0], M=512, seed=0, mode="linear")
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
            m["r_no50"] = r_without_line(P, Yt)
            m["r_induced"] = r_single_trial(P, Yt)
            out[name] = {"mean": {k: float(np.nanmean(v)) for k, v in m.items()}, "r": [float(x) for x in m["r"]],
                         **{x: [float(v) for v in m[x]] for x in EXTRA_METRICS}}

        score("ridge", StreamRidge(LAG9).fit(Ec, Yc).predict(Et))
        base = KFoldRidge(LAG9, shared=True).fit(Ec, Yc)
        score("base", base.predict(Et))
        if not a.fast_eval:
            for name, feat in [("lin_gated", lin), ("enc0_gated", f0), ("enc_gated", f1)]:
                P, on[name] = gated(base, Ec, Yc, Et, feat)
                score(name, P)
        for name, feat in [("lin_ridge", lin)] * (not a.fast_eval) + [("enc0_ridge", f0)]:
            score(name, KFoldRidge(feat, ALPHAS_WIDE, shared=True, keep_oof=False).fit(Ec, Yc).predict(Et))
        kf = KFoldRidge(f1, ALPHAS_WIDE, shared=True).fit(Ec, Yc)
        P = kf.predict(Et)
        score("enc_ridge", P)
        score("enc_ridge_cal", amplitude_calibrate(kf, Yc, P))
        mins = sum(e.shape[1] for e in Ec) / FS / 60
        res["fracs"][f"{f:g}"] = {"n_calib": n_c, "minutes": mins, "gate_on": on,
                                  "mean": {k: v["mean"]["r"] for k, v in out.items()},
                                  **{x: {k: v[x] for k, v in out.items()} for x in EXTRA_METRICS},
                                  "bands": {k: v["mean"] for k, v in out.items()} if bands else {},
                                  "r": {k: v["r"] for k, v in out.items()}}
        log(f"    f={f:g} ({mins:.1f} min) " + " ".join(f"{k}={v['mean']['r']:+.3f}" for k, v in out.items()) +
            " | gate on: " + " ".join(f"{k}={v:.2f}" for k, v in on.items()))
    return res


KEY_TESTS = [("enc_ridge", "enc0_ridge", "learning helps (vs same encoder untrained)?"),
             ("enc_ridge", "lin_ridge", "learned filters beat random filters?"),
             ("enc0_ridge", "lin_ridge", "Morlet init beats random filters?"),
             ("enc_ridge", "base", "beats the strong linear baseline?"),
             ("lin_ridge", "base", "single-stage filter bank beats LAG9?")]


def key_tests(results, out_dir, log):
    """Paired subject-level tests for the decisive comparisons, on r and on every EXTRA_METRICS."""
    from scipy.stats import wilcoxon
    lines, js = ["\n### key comparisons (paired over subjects)",
                 "| calib | question | A | B | metric | mean A | mean B | A - B | A wins | Wilcoxon p |",
                 "|---|---|---|---|---|---|---|---|---|---|"], []
    fracs = sorted({f for r in results for f in r["fracs"]}, key=float)
    for f in fracs:
        rows = [r for r in results if f in r["fracs"] and "r_no50" in r["fracs"][f]]
        metrics = ("r",) + tuple(x for x in EXTRA_METRICS if rows and x in rows[0]["fracs"][f])
        for ma, mb, q in KEY_TESTS:
            if not rows or ma not in rows[0]["fracs"][f]["mean"] or mb not in rows[0]["fracs"][f]["mean"]:
                continue
            for metric in metrics:
                if metric == "r":
                    get = lambda r, m: r["fracs"][f]["mean"][m]                         # noqa: E731
                else:
                    get = lambda r, m: float(np.nanmean(r["fracs"][f][metric][m]))     # noqa: E731
                x = np.array([get(r, ma) for r in rows])
                y = np.array([get(r, mb) for r in rows])
                try:
                    p = float(wilcoxon(x, y).pvalue)
                except ValueError:
                    p = float("nan")
                js.append({"frac": f, "a": ma, "b": mb, "metric": metric, "mean_a": float(x.mean()),
                           "mean_b": float(y.mean()), "diff": float((x - y).mean()), "wins": int((x > y).sum()),
                           "n": len(x), "p": p})
                lines.append(f"| {f} | {q} | {ma} | {mb} | {metric} | {x.mean():.3f} | {y.mean():.3f} | "
                             f"{(x - y).mean():+.4f} | {(x > y).sum()}/{len(x)} | {p:.4f} |")
    md = "\n".join(lines)
    log(md)
    with open(os.path.join(out_dir, "summary.md"), "a", encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(out_dir, "key_tests.json"), "w") as fh:
        json.dump(js, fh, indent=1)


def train_final(a, subjects, dev, log):
    """
    External-validation encoder: the leave-one-patient-out procedure with no test patient. The validation patients
    are drawn exactly as in a LOSO fold (generator seeded with --seed, first n_val_subj of a permutation); they are
    used only for early stopping / checkpoint selection, and the encoder is not refitted on them.
    """
    torch.manual_seed(a.seed)
    order = np.random.default_rng(a.seed).permutation(len(subjects))
    val = [subjects[i] for i in order[:a.n_val_subj]]
    train = [subjects[i] for i in order[a.n_val_subj:]]
    log("=" * 78)
    log(f"FINAL encoder | optimisation ({len(train)}): {[s[0] for s in train]} | "
        f"early stopping ({len(val)}): {[s[0] for s in val]}")
    t0 = time.time()
    ckpt = os.path.join(a.out, "ckpt_final.pt")
    enc, lam, best, hist = train_encoder(a, train, val, dev, log, ckpt)
    path = os.path.join(a.out, "enc_final.pt")
    torch.save({"state": enc.state_dict(), "cfg": enc.cfg, "lam": lam, "best_val": best, "hist": hist,
                "train_subjects": [s[0] for s in train], "val_subjects": [s[0] for s in val], "seed": a.seed,
                "config": vars(a)}, path + ".tmp")
    os.replace(path + ".tmp", path)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    best_step = max(hist, key=lambda h: h[1])[0]
    log(f"final encoder: val r {hist[0][1]:.4f} (init) -> {best:.4f} (best, step {best_step}), last step "
        f"{hist[-1][0]}, lam={lam:.3g}, {enc.n_feat} features/channel | {(time.time() - t0) / 60:.1f} min -> {path}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", nargs="*", default=None, help="subset of subjects to load")
    ap.add_argument("--folds", nargs="*", default=None, help="test subjects to run (default: all)")
    ap.add_argument("--min_trials", type=int, default=20)
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.1, 0.2])
    # encoder
    ap.add_argument("--n_pairs", type=int, default=12)
    ap.add_argument("--half", type=int, default=240, help="filter half-length in samples (240 = 1.2 s)")
    ap.add_argument("--fmin", type=float, default=2.0)
    ap.add_argument("--fmax", type=float, default=60.0)
    ap.add_argument("--cycles", type=float, default=5.0)
    ap.add_argument("--n_free", type=int, default=4)
    ap.add_argument("--no_env", action="store_true")
    ap.add_argument("--arch", choices=("morlet", "shallow"), default="morlet",
                    help="shallow: ShallowConvNet-style per-channel front end, random init (use with --half 12)")
    # episodic training
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=150)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--subj_per_step", type=int, default=2)
    ap.add_argument("--sup_sec", type=float, default=240.0, help="support size (seconds), ~4 min as at test")
    ap.add_argument("--n_query", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_lam", type=float, default=1e-2)
    ap.add_argument("--lam0", type=float, default=1.0)
    ap.add_argument("--objective", choices=("episodic", "fixed_lam", "multitask"), default="episodic",
                    help="episodic: ridge readout in the loss, learnable lambda (default, the published method); "
                         "fixed_lam: same with lambda fixed at --lam0; multitask: per-patient linear heads trained by "
                         "gradient, no ridge in the loop (ablation)")
    ap.add_argument("--lr_head", type=float, default=1e-3, help="multitask: learning rate of the per-patient heads")
    ap.add_argument("--n_val_subj", type=int, default=2)
    ap.add_argument("--val_trials", type=int, default=40, help="test trials per validation subject")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--fast_eval", action="store_true",
                    help="only ridge, base, enc0_ridge, enc_ridge, enc_ridge_cal (for seed / ablation runs)")
    ap.add_argument("--reeval", action="store_true",
                    help="redo the evaluation of finished folds with their saved encoders (no retraining)")
    ap.add_argument("--final", action="store_true",
                    help="train ONE encoder on all subjects (external validation): n_val_subj subjects drawn with "
                         "--seed for early stopping, the rest for optimisation -> OUT/enc_final.pt; no evaluation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.amp = not a.no_amp
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "shared.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    dev = torch.device(a.device)
    log(f"config: {vars(a)}")
    if a.final and os.path.exists(os.path.join(a.out, "enc_final.pt")):
        log("final encoder already trained: " + os.path.join(a.out, "enc_final.pt"))
        return
    subjects = []
    for p in sorted(glob.glob(os.path.join(a.data, "sub-[0-9][0-9].pt"))):
        sid = os.path.basename(p)[:-3]
        if a.subjects and sid not in a.subjects:
            continue
        s, E, Y = load_trials(p, "bip", "car")
        if len(E) < a.min_trials or s.n_targets == 0:
            log(f"skip {sid}: {len(E)} trials, {s.n_targets} targets")
            continue
        subjects.append((sid, E, Y))
        log(f"{sid}: {len(E)} trials, {E[0].shape[0]} scalp, {s.n_targets} targets")
    if a.final:
        train_final(a, subjects, dev, log)
        return
    folds = [s for s in subjects if not a.folds or s[0] in a.folds]

    results = []
    for fi, (sid, E, Y) in enumerate(folds):
        fp = os.path.join(a.out, f"fold_{sid}.json")
        if os.path.exists(fp):
            done = json.load(open(fp))
            # --reeval redoes only folds evaluated by an older version (missing a metric), so it can resume
            if not a.reeval or all(x in fr for fr in done["fracs"].values() for x in EXTRA_METRICS):
                results.append(done)
                log(f"[{sid}] already done, loaded")
                continue
        torch.manual_seed(a.seed + fi)
        log("=" * 78)
        log(f"[{fi + 1}/{len(folds)}] TEST {sid}")
        pool = [s for s in subjects if s[0] != sid]
        order = np.random.default_rng(a.seed + fi).permutation(len(pool))
        val = [pool[i] for i in order[:a.n_val_subj]]
        train = [pool[i] for i in order[a.n_val_subj:]]
        log(f"  train={[s[0] for s in train]} val={[s[0] for s in val]}")
        t0 = time.time()
        enc_path = os.path.join(a.out, f"enc_{sid}.pt")
        if os.path.exists(enc_path):
            ck = torch.load(enc_path, map_location=dev, weights_only=False)
            enc = TemporalEncoder(**ck["cfg"]).to(dev)
            enc.load_state_dict(ck["state"])
            lam, best, hist = ck["lam"], ck["best_val"], ck["hist"]
            log(f"  loaded trained encoder (val r={best:.4f})")
        else:
            ckpt = os.path.join(a.out, f"ckpt_{sid}.pt")
            enc, lam, best, hist = train_encoder(a, train, val, dev, log, ckpt)
            torch.save({"state": enc.state_dict(), "cfg": enc.cfg, "lam": lam, "best_val": best, "hist": hist},
                       enc_path)
            if os.path.exists(ckpt):
                os.remove(ckpt)
        enc0 = TemporalEncoder(**enc.cfg).to(dev)
        log(f"  encoder: val r {hist[0][1]:.4f} (init) -> {best:.4f} (trained), lam={lam:.3g}, "
            f"{enc.n_feat} features/channel | {time.time() - t0:.0f}s")
        r = evaluate(E, Y, enc0, enc, a, dev, log)
        r.update(subject=sid, val_subjects=[s[0] for s in val], val_r_init=hist[0][1], val_r=best, lam=lam,
                 minutes=(time.time() - t0) / 60)
        with open(fp + ".tmp", "w") as fh:
            json.dump(r, fh)
        os.replace(fp + ".tmp", fp)
        results.append(r)
        log(f"[{sid}] done in {(time.time() - t0) / 60:.1f} min")
        del enc, enc0
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if results:
        summarize(results, a.out, log)
        key_tests(results, a.out, log)
        vi = [r["val_r_init"] for r in results]
        vt = [r["val_r"] for r in results]
        log(f"validation subjects, ridge readout on 20 %: Morlet init r={np.mean(vi):.4f} -> trained {np.mean(vt):.4f}")


if __name__ == "__main__":
    main()
