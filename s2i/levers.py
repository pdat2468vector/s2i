"""
Phase 1a - cheap linear tests of the levers suggested by Phase 0 (CPU only).

Phase 0 established (15 subjects, bipolar + CAR, 20 % chronological calibration):
  * genuine single-trial signal (ridge 0.176 vs trial-shuffle null 0.012, 15/15 subjects),
    not ERP, not ocular;
  * strongly data-limited (1 min -> 12 min of calibration: r 0.138 -> 0.210);
  * session shift exists (within-session calibration +0.026, 15/15 subjects positive).

This script asks, with linear models only, whether each candidate lever is real before
we build it into a deep model:

  1. Session alignment (unsupervised).  Euclidean alignment: every session's EEG is
     whitened by its own spatial covariance C_s^{-1/2}. Uses EEG only (no iEEG labels).
     NOTE: C_s is estimated from all EEG of that session (transductive, label-free).
  2. Cross-subject transfer with the CORRECT bipolar geometry.
     A bipolar channel V(n+1) - V(n) is approximately grad(V) . (p_{n+1} - p_n): its sign
     and size depend on the shaft direction, not only on the midpoint. A decoder that only
     sees the midpoint (LeadFormer's query) cannot transfer across subjects whose shafts
     point different ways. We test this with a linear "gradient-field" prior:
       - every training subject's scalp EEG is mapped onto a common 19-channel virtual
         10-20 montage, and a lagged ridge is fitted per contact (weights w_j);
       - for a test contact at p_k with direction u_k, a local linear field G(p_k) is fitted
         to the neighbours' weights, w_j ~= G u_j (Gaussian kernel, width sigma), and
         the prior weight is w0_k = G(p_k) u_k;
       - "mid" prior: plain kernel average of neighbours' weights (direction ignored).
     zs_*    : zero-shot prediction with the prior only (no labels of the test subject).
     prior_* : per-contact gain on the prior + ridge on the residual (shrinks to the prior).

All models are scored on the same fixed test set (trials after the 20 % split, as in
train_loso.py / ceiling.py) for several calibration sizes.

Usage
-----
python -m s2i.levers --data DATA_DIR --out RUNS_DIR/levers_bip_car
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from s2i.baselines import DEFAULT_LAGS
from s2i.ceiling import StreamRidge, chrono_split, feat_lag
from s2i.metrics import per_contact_metrics

FS = 200
VIRTUAL = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
           "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
LAG9 = feat_lag(DEFAULT_LAGS)


# ----------------------------------------------------------------------------- data
def load_subject(path):
    import torch
    d = torch.load(path, weights_only=False)
    b = d["trial_bounds"].numpy()
    eeg = d["eeg_car"].float().numpy()
    bip = d["ieeg_bip"].float().numpy()
    mni = d["contact_mni"].numpy().astype(np.float64)
    pairs = [tuple(p) for p in d["bip_pairs"]]
    u = np.stack([mni[a] - mni[c] for a, c in pairs]) if pairs else np.zeros((0, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
    return dict(sid=d["subject"], E=[eeg[:, s0:s1] for s0, s1 in b], Y=[bip[:, s0:s1] for s0, s1 in b],
                ses=np.array(d["trial_session"]), names=list(d["eeg_names"]),
                pos=d["eeg_pos"].numpy().astype(np.float64),
                bip_pos=d["bip_mni"].numpy().astype(np.float64), bip_dir=u)


def euclidean_align(E, ses):
    """Whiten each session by its own EEG spatial covariance (pseudo-inverse: CAR is rank C-1)."""
    R = {}
    for s in np.unique(ses):
        idx = np.where(ses == s)[0]
        C = sum(E[i].astype(np.float64) @ E[i].T.astype(np.float64) for i in idx)
        C /= sum(E[i].shape[1] for i in idx)
        lam, Q = np.linalg.eigh(C)
        inv = np.where(lam > lam.max() * 1e-6, 1.0 / np.sqrt(np.clip(lam, 1e-30, None)), 0.0)
        R[s] = ((Q * inv) @ Q.T).astype(np.float32)
    return [R[s] @ e for s, e in zip(ses, E)]


def virtual_map(pos, vpos, sigma=30.0):
    """(V, C) interpolation matrix onto the virtual montage (exact copy if the electrode exists)."""
    D = np.linalg.norm(vpos[:, None] - pos[None], axis=-1)
    W = np.exp(-D ** 2 / (2 * sigma ** 2))
    exact = D.min(1) < 1.0
    W[exact] = (D[exact] == D[exact].min(1, keepdims=True)).astype(float)
    return (W / W.sum(1, keepdims=True)).astype(np.float32)


# ----------------------------------------------------------------------------- priors
def field_priors(t, subjects, W_all, sigma):
    """Prior weights (K_t, F) for test subject t from all other subjects' per-contact weights."""
    P = np.concatenate([subjects[s]["bip_pos"] for s in W_all if s != t])
    U = np.concatenate([subjects[s]["bip_dir"] for s in W_all if s != t])
    Wj = np.concatenate([W_all[s].T for s in W_all if s != t])                # (J, F)
    pk, uk = subjects[t]["bip_pos"], subjects[t]["bip_dir"]
    om = np.exp(-((pk[:, None] - P[None]) ** 2).sum(-1) / (2 * sigma ** 2))  # (K, J)
    support = om.sum(1)
    mid = (om @ Wj) / (support[:, None] + 1e-12)
    J, F = Wj.shape
    A = (om @ (Wj[:, :, None] * U[:, None, :]).reshape(J, F * 3)).reshape(-1, F, 3)
    B = (om @ (U[:, :, None] * U[:, None, :]).reshape(J, 9)).reshape(-1, 3, 3)
    eps = 0.1 * np.trace(B, axis1=1, axis2=2) / 3 + 1e-9
    G = np.einsum("kfd,kde->kfe", A, np.linalg.inv(B + eps[:, None, None] * np.eye(3)))
    grad = np.einsum("kfd,kd->kf", G, uk)
    return {"mid": mid, "grad": grad}, support


def feat_standardizer(E):
    X = np.concatenate([LAG9(e) for e in E], 1).astype(np.float64)
    return X.mean(1), X.std(1) + 1e-8


def prior_predict(E, mu, sd, W0):
    return [(((LAG9(e).T - mu) / sd) @ W0.T).T.astype(np.float32) for e in E]


def prior_plus_ridge(Ec, Yc, Et, P0c, P0t):
    p = np.concatenate(P0c, 1).astype(np.float64)
    y = np.concatenate(Yc, 1).astype(np.float64)
    pc, yc = p - p.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    beta = ((pc * yc).sum(1) / ((pc * pc).sum(1) + 1e-12)).astype(np.float32)[:, None]
    m = StreamRidge(LAG9).fit(Ec, [yy - beta * q for yy, q in zip(Yc, P0c)])
    return [beta * q + r for q, r in zip(P0t, m.predict(Et))]


def r_contacts(P, T):
    return per_contact_metrics(P, T, FS, bands=False)["r"]


# ----------------------------------------------------------------------------- main
def run(subjects, a, log=print):
    vnames = [v for v in VIRTUAL if any(v in S["names"] for S in subjects)]
    vpos = np.stack([np.mean([S["pos"][S["names"].index(v)] for S in subjects if v in S["names"]], 0)
                     for v in vnames])
    log(f"virtual montage: {len(vnames)} channels")
    for S in subjects:
        S["Ev"] = [virtual_map(S["pos"], vpos) @ e for e in S["E"]]
        S["Eea"] = euclidean_align(S["E"], S["ses"])

    W_all = {}
    for i, S in enumerate(subjects):          # full-data weights, used ONLY as priors for other subjects
        W_all[i] = StreamRidge(LAG9).fit(S["Ev"], S["Y"]).W
    log("per-subject virtual-montage weights fitted")

    results = []
    for t, S in enumerate(subjects):
        t0 = time.time()
        n = len(S["E"])
        _, test = chrono_split(n, 0.2, gap=1)
        Yt = [S["Y"][i] for i in test]
        pri, support = field_priors(t, subjects, W_all, a.sigma)
        res = {"subject": S["sid"], "n_targets": len(S["bip_pos"]), "support_mean": float(support.mean()),
               "fracs": {}}
        for f in a.fracs:
            n_c = max(3, int(round(n * f)))
            if n_c >= test[0]:
                continue
            cal = list(range(n_c))
            pick = lambda X, idx: [X[i] for i in idx]                       # noqa: E731
            out = {}
            out["ridge"] = r_contacts(StreamRidge(LAG9).fit(pick(S["E"], cal), pick(S["Y"], cal))
                                      .predict(pick(S["E"], test)), Yt)
            out["ridge_EA"] = r_contacts(StreamRidge(LAG9).fit(pick(S["Eea"], cal), pick(S["Y"], cal))
                                         .predict(pick(S["Eea"], test)), Yt)
            out["ridge_virtual"] = r_contacts(StreamRidge(LAG9).fit(pick(S["Ev"], cal), pick(S["Y"], cal))
                                              .predict(pick(S["Ev"], test)), Yt)
            mu, sd = feat_standardizer(pick(S["Ev"], cal))
            for name, W0 in pri.items():
                P0c = prior_predict(pick(S["Ev"], cal), mu, sd, W0)
                P0t = prior_predict(pick(S["Ev"], test), mu, sd, W0)
                out[f"zs_{name}"] = r_contacts(P0t, Yt)
                out[f"prior_{name}"] = r_contacts(prior_plus_ridge(pick(S["E"], cal), pick(S["Y"], cal),
                                                                   pick(S["E"], test), P0c, P0t), Yt)
            res["fracs"][f"{f:g}"] = {"n_calib": n_c,
                                      "minutes": sum(S["E"][i].shape[1] for i in cal) / FS / 60,
                                      "mean": {k: float(np.nanmean(v)) for k, v in out.items()},
                                      "r": {k: [float(x) for x in v] for k, v in out.items()}}
            log(f"  [{S['sid']}] f={f:g} ({res['fracs'][f'{f:g}']['minutes']:.1f} min) " +
                " ".join(f"{k}={v:+.3f}" for k, v in res["fracs"][f"{f:g}"]["mean"].items()))
        results.append(res)
        if a.out:
            with open(os.path.join(a.out, f"levers_{S['sid']}.json"), "w") as fh:
                json.dump(res, fh)
        log(f"  [{S['sid']}] support={res['support_mean']:.1f}  ({time.time() - t0:.0f}s)")
    return results


def summarize(results, out_dir, log=print):
    from scipy.stats import wilcoxon
    lines = []
    summ = {}
    fracs = sorted({f for r in results for f in r["fracs"]}, key=float)
    for f in fracs:
        rows = [r for r in results if f in r["fracs"]]
        models = list(rows[0]["fracs"][f]["mean"])
        M = {m: np.array([r["fracs"][f]["mean"][m] for r in rows]) for m in models}
        mins = np.mean([r["fracs"][f]["minutes"] for r in rows])
        lines += [f"\n### calib fraction {f}  ({len(rows)} subjects, mean {mins:.1f} min)",
                  "| model | mean r | SD | vs ridge: mean diff | wins | Wilcoxon p | vs 0: p |",
                  "|---|---|---|---|---|---|---|"]
        summ[f] = {}
        for m in models:
            d = M[m] - M["ridge"]
            try:
                p = float(wilcoxon(M[m], M["ridge"]).pvalue) if m != "ridge" else float("nan")
            except ValueError:
                p = float("nan")
            try:
                p0 = float(wilcoxon(M[m]).pvalue)
            except ValueError:
                p0 = float("nan")
            summ[f][m] = {"mean": float(M[m].mean()), "sd": float(M[m].std()), "diff": float(d.mean()),
                          "wins": int((d > 0).sum()), "p_vs_ridge": p, "p_vs_0": p0}
            lines.append(f"| {m} | {M[m].mean():.3f} | {M[m].std():.3f} | {d.mean():+.3f} | "
                         f"{(d > 0).sum()}/{len(d)} | {p:.4f} | {p0:.4f} |")
    f_main = "0.2" if "0.2" in fracs else fracs[-1]
    models = list(results[0]["fracs"][f_main]["mean"])
    lines += [f"\n### per subject (calib fraction {f_main})", "| subject | support | " + " | ".join(models) + " |",
              "|---" * (len(models) + 2) + "|"]
    for r in results:
        if f_main in r["fracs"]:
            lines.append(f"| {r['subject']} | {r['support_mean']:.1f} | " +
                         " | ".join(f"{r['fracs'][f_main]['mean'][m]:.3f}" for m in models) + " |")
    md = "\n".join(lines)
    log(md)
    if out_dir:
        with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(summ, fh, indent=1)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_trials", type=int, default=20)
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.05, 0.1, 0.2])
    ap.add_argument("--sigma", type=float, default=15.0, help="kernel width (mm) for cross-subject priors")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "levers.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"config: {vars(a)}")
    subjects = []
    for p in sorted(glob.glob(os.path.join(a.data, "sub-[0-9][0-9].pt"))):
        S = load_subject(p)
        if len(S["E"]) < a.min_trials or len(S["bip_pos"]) == 0:
            log(f"skip {S['sid']}")
            continue
        subjects.append(S)
        log(f"loaded {S['sid']}: {len(S['E'])} trials, {len(S['names'])} scalp, {len(S['bip_pos'])} bipolar, "
            f"{len(np.unique(S['ses']))} sessions")
    results = run(subjects, a, log)
    summarize(results, a.out, log)


if __name__ == "__main__":
    main()
