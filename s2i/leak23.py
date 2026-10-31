"""
Phase 23 - Addendum B of the external-validation protocol (post hoc; written 13 Oct 2026 before this code was run).

B1  input-target audit, all 12 ScaleOLDNEW patients, descriptive. For every kept scalp channel against every
    intracranial signal, per block, the largest |r| over small lags, its lag and the affine fit at that lag, at three
    signal stages:
      raw    unfiltered ADES samples (4096 Hz), the six 10-s segments per block of Phase 20, vs every recorded contact,
             lags -8..+8 raw samples (about +-2 ms)
      proc   Phase-14 output (notch, 0.5-70 Hz, 200 Hz, original reference, whole blocks minus 3.4-s edges) vs every
             recorded contact, every bipolar target and the iEEG common mode (mean of the recorded contacts),
             lags -10..+10 samples (+-50 ms)
      input  model input: CAR scalp EEG vs the bipolar targets, the primary-level windows, lags +-50 ms
    Copy-like criterion (Addendum B, fixed before running), evaluated at the proc stage: in EVERY block of the patient
      (1) largest lagged |r| >= 0.90, (2) its lag within +-1 sample of the median lag over blocks,
      (3) |r| with the signal exceeds |r| with the iEEG common mode by >= 0.10.
B2  H1 / H2 with only the seven input-like channels of Phase 20 removed (patients 05, 06, 07); CAR recomputed,
    readouts refitted, everything else unchanged; the other nine patients keep their registered results (identical
    data, deterministic readout).

Never modifies Phase-14 / Phase-17 files; writes only to its own output folder (and DATA17/primary_drop7 for B2).

Usage
  python -m s2i.leak23 audit       --sc_root SC_ROOT --data DATA17 --out RUNS/leak23
  python -m s2i.leak23 drop        --data DATA17                                   (-> DATA17/primary_drop7)
  python -m s2i.leak23 report_drop --runs RUNS17 --out RUNS/leak23
  python -m s2i.leak23 smoke       --out /tmp/leak23_smoke                          (synthetic data, planted copy)
"""
import argparse
import csv
import glob
import gzip
import json
import os
import re
import sys
import time

import numpy as np

FS = 200
EDGE = 680                                             # 3.4 s at 200 Hz, as the protocol's block-edge margin
SEVEN = {"sub-P05": ["C3", "FC1", "FC5", "CP5"], "sub-P06": ["CP6"], "sub-P07": ["AF1", "FC1"]}
CRIT = {"r": 0.90, "lag_tol": 1, "cm_margin": 0.10}
MAXLAG = {"raw": 8, "proc": 10, "input": 10}
RAW_SEG = (6, 10.0)                                    # segments per block, seconds (Phase 20)


def log(msg):
    print(time.strftime("%H:%M:%S"), "|", msg, flush=True)


# ----------------------------------------------------------------------------- lagged correlation
def lagcorr(Xs, Ys, maxlag):
    """
    Xs, Ys: lists of segments (channels x samples) of the same lengths. For each lag L in -maxlag..maxlag, the Pearson
    r between x[t] and y[t + L] over the central samples of every segment (pooled, each row z-scored over the pooled
    samples). Returns R (lags, nx, ny) and, for the affine fit x = a*y + b, the stds and means of x and of y per lag.
    """
    m = maxlag
    keep = [(x, y) for x, y in zip(Xs, Ys) if x.shape[1] > 2 * m + 20]
    xp = np.concatenate([x[:, m:x.shape[1] - m] for x, _ in keep], 1).astype(np.float64)
    mx, sx = xp.mean(1), xp.std(1)
    xz = ((xp - mx[:, None]) / np.where(sx > 0, sx, 1)[:, None]).astype(np.float32)
    N = xz.shape[1]
    lags = np.arange(-m, m + 1)
    R = np.zeros((len(lags), xz.shape[0], keep[0][1].shape[0]), np.float32)
    my, sy = np.zeros((len(lags), R.shape[2])), np.zeros((len(lags), R.shape[2]))
    for i, L in enumerate(lags):
        yp = np.concatenate([y[:, m + L:y.shape[1] - m + L] for _, y in keep], 1).astype(np.float64)
        my[i], sy[i] = yp.mean(1), yp.std(1)
        yz = ((yp - my[i][:, None]) / np.where(sy[i] > 0, sy[i], 1)[:, None]).astype(np.float32)
        R[i] = xz @ yz.T / N
    return lags, R, (mx, sx), (my, sy)


def best_lag(lags, R, xs, ys):
    """Per pair: largest |r| over lags, its lag, and the affine fit x = slope * y + intercept at that lag."""
    k = np.abs(R).argmax(0)                                            # (nx, ny)
    r = np.take_along_axis(R, k[None], 0)[0]
    mx, sx = xs
    my, sy = ys
    myk = my[k, np.arange(k.shape[1])[None, :]]
    syk = sy[k, np.arange(k.shape[1])[None, :]]
    slope = r * sx[:, None] / np.where(syk > 0, syk, np.nan)
    icpt = mx[:, None] - slope * myk
    return r, lags[k], slope, icpt


# ----------------------------------------------------------------------------- data access
def kept_and_targets(data, sid):
    import torch
    d = torch.load(os.path.join(data, "primary", f"{sid}.pt"), weights_only=False)
    return d


def raw_names_map(names, t1005):
    from s2i.scales17 import scalp_name
    out = {}
    for c in names:
        nm, _ = scalp_name(c, t1005)
        if nm is not None and nm not in out:
            out[nm] = c
    return out


def raw_segments(dat, ades, scalp, contacts):
    """Raw segments of the scalp channels (header type EEG) and contacts (header type SEEG), columns looked up by
    (name, type) as in the Phase-14 preprocessing: a scalp channel and a contact may share a name (e.g. P10)."""
    from s2i.scales17 import read_header
    fs, n, hdr = read_header(ades)
    col = {}
    for i, (c, typ) in enumerate(hdr):
        col.setdefault((c, typ.strip().upper()), i)
    chans = [(c, "EEG") for c in scalp] + [(c, "SEEG") for c in contacts]
    miss = [c for c in chans if c not in col]
    assert not miss, f"channels not found with their type in {ades}: {miss[:5]}"
    X = np.memmap(dat, dtype="<f4", mode="r", shape=(n, len(hdr)))
    seg = int(RAW_SEG[1] * fs)
    lo, hi = int(5 * fs), n - int(5 * fs) - seg
    starts = np.linspace(lo, hi, RAW_SEG[0]).astype(int) if hi > lo else np.array([0])
    idx = [col[c] for c in chans]
    parts = [np.asarray(X[s:s + seg])[:, idx].astype(np.float64).T for s in starts]
    del X
    return parts, fs


# ----------------------------------------------------------------------------- B1
def audit_patient(a, num, pid, t1005):
    from s2i.scales17 import load_block
    sid = f"sub-P{num:02d}"
    fp = os.path.join(a.data, "primary", f"{sid}.pt")
    if not os.path.exists(fp):
        log(f"[{sid}] not in the primary level, skipped")
        return None
    d = kept_and_targets(a.data, sid)
    kept = list(d["eeg_names"])
    bips = [tuple(n.split("-")) for n in d["bip_names"]]
    files = sorted(glob.glob(os.path.join(a.sc_root, "proc", f"SEEG{pid}_Block*.npz")),
                   key=lambda p: int(re.search(r"Block(\d+)", p).group(1)))
    res = {"raw": [], "proc": [], "input": []}            # per block: dict(signals=[...], r, lag, slope, icpt)
    sig_names = {}
    for f in files:
        k = int(re.search(r"Block(\d+)", f).group(1))
        b = load_block(f)
        cmap = raw_names_map(b["eeg_names"], t1005)
        scalp_raw = [cmap[c] for c in kept]
        ct = [c for c in b["seeg_names"] if c not in b["bad"]]
        # ---- proc stage
        E = np.stack([b["eeg"][b["eeg_names"].index(c)] for c in scalp_raw]).astype(np.float64)
        M = np.stack([b["seeg"][b["seeg_names"].index(c)] for c in ct]).astype(np.float64)
        L = min(E.shape[1], M.shape[1])
        E, M = E[:, EDGE:L - EDGE], M[:, EDGE:L - EDGE]
        ci = {c: i for i, c in enumerate(ct)}
        B = np.stack([M[ci[p]] - M[ci[q]] for p, q in bips if p in ci and q in ci])
        bnames = [f"{p}-{q}" for p, q in bips if p in ci and q in ci]
        Y = np.concatenate([M, B, M.mean(0, keepdims=True)])
        names = [("contact", c) for c in ct] + [("bipolar", n) for n in bnames] + [("cm", "iEEG common mode")]
        lags, R, xs, ys = lagcorr([E], [Y], MAXLAG["proc"])
        r, lg, sl, ic = best_lag(lags, R, xs, ys)
        res["proc"].append({"block": k, "signals": names, "r": r, "lag": lg, "slope": sl, "icpt": ic})
        # ---- raw stage (contacts only)
        if not a.skip_raw:
            base = os.path.join(a.sc_root, "raw", f"SEEG{pid}", f"Block{k}", f"SEEG{pid}Block{k}")
            parts, fs = raw_segments(base + ".dat", base + ".ades", scalp_raw, ct)
            Xs = [p[:len(scalp_raw)] for p in parts]
            Ys = [p[len(scalp_raw):] for p in parts]
            ok = np.all([np.isfinite(y).all(1) for y in Ys], 0)
            Ys = [y[ok] for y in Ys]
            lags, R, xs, ys = lagcorr(Xs, Ys, MAXLAG["raw"])
            r, lg, sl, ic = best_lag(lags, R, xs, ys)
            res["raw"].append({"block": k, "signals": [("contact", c) for c, o in zip(ct, ok) if o], "r": r,
                               "lag": lg, "slope": sl, "icpt": ic})
    # ---- input stage: CAR scalp vs bipolar targets over the primary windows, per block
    st = d["norm_stats"]
    car = d["eeg_car"].float().numpy().astype(np.float64)
    bip = d["ieeg_bip"].float().numpy().astype(np.float64)
    tb = d["trial_bounds"].numpy()
    wb = np.asarray(d["window_block"])
    for k in sorted(set(wb.tolist())):
        idx = np.where(wb == k)[0]
        Xs = [car[:, tb[i, 0]:tb[i, 1]] for i in idx]
        Ys = [bip[:, tb[i, 0]:tb[i, 1]] for i in idx]
        lags, R, xs, ys = lagcorr(Xs, Ys, MAXLAG["input"])
        r, lg, sl, ic = best_lag(lags, R, xs, ys)
        res["input"].append({"block": int(k), "signals": [("bipolar", n) for n in d["bip_names"]], "r": r, "lag": lg,
                             "slope": sl, "icpt": ic, "note": "normalised units (slope/intercept not in uV)"})
    del st
    return sid, kept, res


def aggregate(sid, kept, res):
    """Per (stage, scalp, signal): r over blocks, lags, affine fit; copy-like test at the proc stage."""
    rows = []
    for stage, blocks in res.items():
        if not blocks:
            continue
        common = set(blocks[0]["signals"])
        for b in blocks[1:]:
            common &= set(b["signals"])
        sigs = [s for s in blocks[0]["signals"] if s in common]
        cm = ("cm", "iEEG common mode")
        for i, c in enumerate(kept):
            for s in sigs:
                rr, lg, sl, ic, mg = [], [], [], [], []
                for b in blocks:
                    j = b["signals"].index(s)
                    rr.append(float(abs(b["r"][i, j])))
                    lg.append(int(b["lag"][i, j]))
                    sl.append(float(b["slope"][i, j]))
                    ic.append(float(b["icpt"][i, j]))
                    if stage == "proc" and s != cm:
                        mg.append(rr[-1] - float(abs(b["r"][i, b["signals"].index(cm)])))
                med = float(np.median(lg))
                copy_like = (stage == "proc" and s != cm and min(rr) >= CRIT["r"]
                             and max(abs(x - med) for x in lg) <= CRIT["lag_tol"]
                             and len(mg) == len(rr) and min(mg) >= CRIT["cm_margin"])
                rows.append({"subject": sid, "stage": stage, "scalp": c, "kind": s[0], "signal": s[1],
                             "n_blocks": len(rr), "r_min": min(rr), "r_mean": float(np.mean(rr)), "r_max": max(rr),
                             "lags": lg, "slope_mean": float(np.nanmean(sl)), "intercept_mean": float(np.nanmean(ic)),
                             "cm_margin_min": min(mg) if mg else None, "copy_like": bool(copy_like),
                             "flagged_input_like": c in SEVEN.get(sid, [])})
    return rows


def audit(a):
    from s2i.scales17 import PATIENTS, template_1005
    t1005 = template_1005()
    os.makedirs(a.out, exist_ok=True)
    allrows = []
    pats = [p for p in PATIENTS if not a.patients or f"sub-P{p[0]:02d}" in a.patients]
    for num, pid, setup, _ in pats:
        t0 = time.time()
        out = audit_patient(a, num, pid, t1005)
        if out is None:
            continue
        rows = aggregate(*out)
        for r in rows:
            r["setup"] = setup
        allrows += rows
        top = {st: max((r for r in rows if r["stage"] == st and r["kind"] != "cm"), key=lambda r: r["r_mean"],
                       default=None) for st in ("raw", "proc", "input")}
        log(f"[{out[0]}] " + " | ".join(f"{st}: max mean|r| {t['r_mean']:.3f} ({t['scalp']} vs {t['signal']})"
                                        for st, t in top.items() if t) + f" | {time.time() - t0:.0f}s")
    if a.patients and os.path.exists(os.path.join(a.out, "leak23_pairs.csv.gz")):
        done = {r["subject"] for r in allrows}
        old = [r for r in load_rows(a.out) if r["subject"] not in done]
        log(f"merged with the existing table: {len({r['subject'] for r in old})} other patients kept")
        allrows = old + allrows
    write_outputs(a.out, allrows)


def load_rows(out):
    rows = []
    with gzip.open(os.path.join(out, "leak23_pairs.csv.gz"), "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("r_min", "r_mean", "r_max", "slope_mean", "intercept_mean"):
                r[k] = float(r[k])
            r["n_blocks"] = int(r["n_blocks"])
            r["lags"] = [int(x) for x in r["lags"].split()]
            r["cm_margin_min"] = float(r["cm_margin_min"]) if r["cm_margin_min"] not in ("", "None") else None
            r["copy_like"] = r["copy_like"] == "True"
            r["flagged_input_like"] = r["flagged_input_like"] == "True"
            rows.append(r)
    return rows


def inspect(a):
    """Bipolar targets that contain the given contacts, and how every scalp channel relates to them."""
    d = kept_and_targets(a.data, a.patient)
    rows = [r for r in load_rows(a.out) if r["subject"] == a.patient]
    L = [f"# {a.patient}: contacts {', '.join(a.contacts)}", ""]
    for c in a.contacts:
        tg = [n for n in d["bip_names"] if c in n.split("-")]
        L.append(f"- {c}: in {len(tg)} of {len(d['bip_names'])} bipolar targets of the primary level"
                 + (f" ({', '.join(tg)})" if tg else ""))
        for n in tg:
            best = sorted((r for r in rows if r["stage"] in ("proc", "input") and r["signal"] == n),
                          key=lambda r: -r["r_mean"])[:3]
            for r in best:
                L.append(f"    {r['stage']}: {r['scalp']} vs {n} mean |r| {r['r_mean']:.3f} [min {r['r_min']:.3f}]")
        best = sorted((r for r in rows if r["stage"] == "proc" and r["signal"] == c), key=lambda r: -r["r_mean"])[:5]
        L.append(f"    proc, scalp channels most correlated with {c}: " +
                 "; ".join(f"{r['scalp']} {r['r_mean']:.3f}" for r in best))
    tg = list(dict.fromkeys(list(a.targets) + [n for n in d["bip_names"] for c in a.contacts if c in n.split("-")]))
    if tg:
        L += ["", "## Input stage (CAR scalp vs target, whole primary windows, largest |r| over +-50 ms): "
                  "registered channels vs without the seven input-like channels", ""]
        for lev in ("primary", "primary_drop7"):
            fp = os.path.join(a.data, lev, f"{a.patient}.pt")
            if not os.path.exists(fp):
                L.append(f"- {lev}: no data for {a.patient}")
                continue
            import torch
            dd = torch.load(fp, weights_only=False)
            car = dd["eeg_car"].float().numpy().astype(np.float64)
            bip = dd["ieeg_bip"].float().numpy().astype(np.float64)
            tb = dd["trial_bounds"].numpy()
            j = [dd["bip_names"].index(n) for n in tg if n in dd["bip_names"]]
            Xs = [car[:, s0:s1] for s0, s1 in tb]
            Ys = [bip[j, s0:s1] for s0, s1 in tb]
            lags, R, _, _ = lagcorr(Xs, Ys, MAXLAG["input"])
            r = np.abs(R).max(0)
            for jj, n in enumerate([tg[k] for k in range(len(tg)) if tg[k] in dd["bip_names"]]):
                o = np.argsort(-r[:, jj])[:3]
                L.append(f"- {lev}, {n}: " + "; ".join(f"{dd['eeg_names'][i]} {r[i, jj]:.3f}" for i in o)
                         + f" ({len(dd['eeg_names'])} scalp channels)")
    print("\n".join(L))


def write_outputs(out, rows):
    cols = ["subject", "setup", "stage", "scalp", "kind", "signal", "n_blocks", "r_min", "r_mean", "r_max", "lags",
            "slope_mean", "intercept_mean", "cm_margin_min", "copy_like", "flagged_input_like"]
    with gzip.open(os.path.join(out, "leak23_pairs.csv.gz"), "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "lags": " ".join(map(str, r["lags"]))})
    subs = sorted({r["subject"] for r in rows})
    copy = [r for r in rows if r["copy_like"]]
    L = ["# Phase 23 - Addendum B1: input-target audit (post hoc, descriptive)", "",
         f"Copy-like criterion (proc stage, every block): |r| >= {CRIT['r']}, lag within +-{CRIT['lag_tol']} sample of "
         f"the median lag, |r| - |r with iEEG common mode| >= {CRIT['cm_margin']}.", "",
         f"**Copy-like scalp channel / intracranial signal pairs: {len(copy)}**", ""]
    for r in copy:
        L.append(f"- {r['subject']} {r['scalp']} vs {r['kind']} {r['signal']}: |r| {r['r_min']:.3f}-{r['r_max']:.3f}, "
                 f"lags {r['lags']}, margin over common mode {r['cm_margin_min']:.3f}")
    L += ["", "## Largest correlation per patient and stage (mean |r| over blocks; min over blocks in brackets)", "",
          "| patient | setup | raw: contact | proc: contact | proc: bipolar | proc: iEEG common mode | input: bipolar |",
          "|---|---|---|---|---|---|---|"]

    def top(sub, stage, kind):
        c = [r for r in rows if r["subject"] == sub and r["stage"] == stage and r["kind"] == kind]
        if not c:
            return "–"
        t = max(c, key=lambda r: r["r_mean"])
        return f"{t['r_mean']:.3f} [{t['r_min']:.3f}] {t['scalp']}/{t['signal'] if kind != 'cm' else 'CM'}"
    for s in subs:
        setup = next(r["setup"] for r in rows if r["subject"] == s)
        L.append(f"| {s} | {setup} | {top(s, 'raw', 'contact')} | {top(s, 'proc', 'contact')} | "
                 f"{top(s, 'proc', 'bipolar')} | {top(s, 'proc', 'cm')} | {top(s, 'input', 'bipolar')} |")
    L += ["", "## The seven input-like channels of Phase 20 (three highest signals per stage)", ""]
    for s, chans in SEVEN.items():
        for c in chans:
            for stage in ("raw", "proc", "input"):
                cand = sorted((r for r in rows if r["subject"] == s and r["scalp"] == c and r["stage"] == stage),
                              key=lambda r: -r["r_mean"])[:3]
                if cand:
                    L.append(f"- {s} {c}, {stage}: " + "; ".join(
                        f"{r['kind']} {r['signal'] if r['kind'] != 'cm' else 'CM'} {r['r_mean']:.3f} "
                        f"[{r['r_min']:.3f}, lag {r['lags']}]" for r in cand))
    L += ["", "## Distribution over all pairs (proc stage, contacts and bipolar targets)", "",
          "| patient | pairs | mean |r| >= 0.5 | >= 0.7 | >= 0.9 |", "|---|---|---|---|---|"]
    for s in subs:
        v = np.array([r["r_mean"] for r in rows if r["subject"] == s and r["stage"] == "proc" and r["kind"] != "cm"])
        L.append(f"| {s} | {len(v)} | {(v >= .5).sum()} | {(v >= .7).sum()} | {(v >= .9).sum()} |")
    md = "\n".join(L)
    print(md)
    open(os.path.join(out, "leak23_report.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump({"criterion": CRIT, "maxlag": MAXLAG, "raw_segments": RAW_SEG, "n_pairs": len(rows),
               "copy_like": [{k: r[k] for k in ("subject", "scalp", "kind", "signal", "r_min", "r_max", "lags",
                                                "cm_margin_min")} for r in copy]},
              open(os.path.join(out, "leak23_summary.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- B2
def drop(a):
    """DATA17/primary_drop7/sub-P05..07.pt: the seven input-like channels removed, CAR recomputed (as setch)."""
    import torch
    from s2i.scales17 import robust
    src, dst = os.path.join(a.data, "primary"), os.path.join(a.data, "primary_drop7")
    os.makedirs(dst, exist_ok=True)
    for sid, chans in SEVEN.items():
        d = torch.load(os.path.join(src, f"{sid}.pt"), weights_only=False)
        kept = list(d["eeg_names"])
        miss = [c for c in chans if c not in kept]
        assert not miss, f"{sid}: {miss} not among the kept channels {kept}"
        keep = [c for c in kept if c not in chans]
        idx = [kept.index(c) for c in keep]
        st = d["norm_stats"]["eeg_orig"]["all"]
        z = d["eeg_orig"].float().numpy().astype(np.float64)
        v = (z * np.asarray(st["mad"])[:, None] + np.asarray(st["med"])[:, None])[idx]       # back to microvolts
        car = v - v.mean(0, keepdims=True)
        out = dict(d)
        out["eeg_names"] = keep
        out["norm_stats"] = dict(d["norm_stats"])
        for k, x in (("eeg_orig", v), ("eeg_car", car)):
            med, mad = robust(x)
            out[k] = torch.from_numpy(np.clip((x - med) / mad, -6e4, 6e4).astype(np.float16))
            out["norm_stats"][k] = {"all": {"med": med[:, 0].tolist(), "mad": mad[:, 0].tolist()}}
        out["channel_rule"] = f"v1.6 kept minus the seven input-like channels {chans} (Addendum B2, post hoc)"
        torch.save(out, os.path.join(dst, f"{sid}.pt"))
        torch.save({k: out[k] for k in ("subject", "norm_stats", "norm_clip_z", "trial_bounds", "trial_session")},
                   os.path.join(dst, f"{sid}_normstats.pt"))
        log(f"[{sid}] {len(kept)} -> {len(keep)} scalp channels (removed {', '.join(chans)})")


def report_drop(a):
    from s2i.scales17 import _ex, _load, compare
    rng = np.random.default_rng(0)
    R = a.runs
    L = ["# Phase 23 - Addendum B2: H1 / H2 without the seven input-like scalp channels (post hoc)", "",
         "Patients 05, 06 and 07 recomputed without the seven channels (CAR recomputed); the other nine patients keep "
         "their registered results. p-values unadjusted.", "",
         "| analysis | comparison | n | mean A | mean B | median A − B | mean A − B [95 % CI] | A > B | p |",
         "|---|---|---|---|---|---|---|---|---|"]
    out = {}

    def load(tag):
        return (_load(os.path.join(R, "main"), f"final_{tag}"), _load(os.path.join(R, "no_env"), f"learned3_{tag}"),
                _load(os.path.join(R, "no_env"), f"fixed3_{tag}"))
    reg = load("primary")
    dr = load("primary_drop7")
    assert all(sorted(x) == sorted(SEVEN) for x in dr), f"drop7 readouts incomplete: {[sorted(x) for x in dr]}"
    comb = tuple({**r, **d} for r, d in zip(reg, dr))
    st = load("primary_setch")
    for tag, lab, (F, L3, F3) in (("primary", "registered (v1.6 channels)", reg),
                                  ("drop7", "without the seven input-like channels", comb),
                                  ("setch", "Addendum A (authors' file channels)", st)):
        if not F:
            continue
        for name, A, B in (("H1: Final − Base", _ex(F, "combo_nosess"), _ex(F, "base")),
                           ("H2: Learned-3 − Fixed-3", _ex(L3, "combo_nosess"), _ex(F3, "combo_nosess"))):
            h = compare(A, B, rng)
            if h is None:
                continue
            out[f"{tag}|{name}"] = h
            L.append(f"| {lab} | {name} | {h['n']} | {h['mean_a']:.4f} | {h['mean_b']:.4f} | {h['median_diff']:+.4f} | "
                     f"{h['mean_diff']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] | {h['wins']}/{h['n']} | "
                     f"{h['p']:.5f} |")
    for name in ("H1: Final − Base", "H2: Learned-3 − Fixed-3"):
        a_, b_ = out.get(f"primary|{name}"), out.get(f"drop7|{name}")
        if a_ and b_:
            L.append("")
            L.append(f"- {name}, patients 05-07 (registered → without the seven channels): " + ", ".join(
                f"{s} {a_['per_patient'][s]:+.4f} → {b_['per_patient'][s]:+.4f}" for s in SEVEN))
    L += ["", "| patient | model | registered r | without the seven channels |", "|---|---|---|---|"]
    for s in SEVEN:
        for lab, D_reg, D_dr, m in (("Base", reg[0], dr[0], "base"), ("Final", reg[0], dr[0], "combo_nosess"),
                                    ("Learned-3", reg[1], dr[1], "combo_nosess"), ("Fixed-3", reg[2], dr[2], "combo_nosess")):
            x, y = _ex(D_reg, m).get(s), _ex(D_dr, m).get(s)
            if x is not None and y is not None:
                L.append(f"| {s} | {lab} | {x:.4f} | {y:.4f} |")
    md = "\n".join(L)
    print(md)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "leak23_drop7_report.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(os.path.join(a.out, "leak23_drop7_report.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- smoke test
def smoke(a):
    """Synthetic patient with a planted copy (scalp C3 = 0.8 * contact A2 + small noise) and a pure common mode."""
    rng = np.random.default_rng(1)
    rows = []
    kept = ["C3", "Cz", "Pz", "F3"]
    for planted in (True, False):
        res = {"raw": [], "proc": [], "input": []}
        for k in range(3):
            n = 6000
            cm = np.cumsum(rng.normal(size=n)) * 0.05
            M = rng.normal(size=(6, n)) + cm
            E = rng.normal(size=(4, n)) + 0.6 * cm
            if planted:
                E[0] = 0.8 * np.roll(M[1], 2) + 0.05 * rng.normal(size=n)
            B = M[:-1] - M[1:]
            Y = np.concatenate([M, B, M.mean(0, keepdims=True)])
            names = ([("contact", f"A{i}") for i in range(6)] + [("bipolar", f"A{i}-A{i + 1}") for i in range(5)]
                     + [("cm", "iEEG common mode")])
            lags, R, xs, ys = lagcorr([E], [Y], MAXLAG["proc"])
            r, lg, sl, ic = best_lag(lags, R, xs, ys)
            res["proc"].append({"block": k, "signals": names, "r": r, "lag": lg, "slope": sl, "icpt": ic})
        rows += aggregate("sub-PSMOKE" if planted else "sub-PCLEAN", kept, res)
    for r in rows:
        r["setup"] = "synthetic"
    os.makedirs(a.out, exist_ok=True)
    write_outputs(a.out, rows)
    hit = [r for r in rows if r["copy_like"]]
    assert len(hit) >= 1 and all(r["subject"] == "sub-PSMOKE" and r["scalp"] == "C3" for r in hit), hit
    assert any(r["signal"] == "A1" and r["lags"] == [-2, -2, -2] for r in hit), [(r["signal"], r["lags"]) for r in hit]
    print("smoke test passed: the planted copy is found (C3 vs A1, lag -2), nothing else is copy-like")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("audit", "drop", "report_drop", "smoke", "inspect"))
    ap.add_argument("--patient", help="inspect: subject, e.g. sub-P07")
    ap.add_argument("--contacts", nargs="*", default=[], help="inspect: contact names, e.g. K'04")
    ap.add_argument("--targets", nargs="*", default=[], help="inspect: bipolar targets, e.g. D'10-D'11")
    ap.add_argument("--sc_root")
    ap.add_argument("--data", help="DATA17 (build output root with primary/)")
    ap.add_argument("--runs", help="RUNS17")
    ap.add_argument("--out")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--skip_raw", action="store_true", help="audit without the raw stage")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    {"audit": audit, "drop": drop, "report_drop": report_drop, "smoke": smoke, "inspect": inspect}[a.cmd](a)


if __name__ == "__main__":
    main()
