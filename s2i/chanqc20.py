"""
Phase 20 - descriptive audit of the ScaleOLDNEW scalp channels (focus: patients 01, 02, 03, 05).

Why: under protocol v1.6 the number of kept scalp channels exceeded Table 1 of Barborica et al. 2023 in patients 01
(31 vs 30), 02 (38 vs 30), 03 (35 vs 30) and 05 (38 vs 35). The Addendum-A sensitivity analysis does not touch
patient 02, because the authors' EEGLAB file also lists all 38 of its channels. This audit collects quantitative
evidence on every scalp channel of these patients, next to reference patients whose kept count matched Table 1.

What it is NOT:
  * it never removes a channel, never writes into a Phase-17 folder and never re-runs a model;
  * flags are warnings with thresholds fixed in THRESH below, not labels. Signal statistics cannot prove whether a
    physical electrode was attached: a valid electrode can correlate strongly with its neighbours, and a channel that
    looks normal does not prove that an electrode was present.

Inputs (read only)
  <sc_root>/raw/SEEG<id>/Block<k>/SEEG<id>Block<k>.ades / .dat / .ades.bad   raw ADES (float32, sample-major, uV)
  <sc_root>/raw/SEEG<id>/SEEG<id>.set                                       authors' EEGLAB scalp file (metadata only)
  <sc_root>/proc/SEEG<id>_Block<k>.npz                                     Phase-14 output (notch, 0.5-70 Hz, 200 Hz,
                                                                           original reference = before CAR)
  <build_json> (= DATA17/p17_build.json)                                   channels kept under v1.6 + exclusion reasons
Outputs (<out>/)
  chanqc20_channels.csv   one row per patient x EEG-type header channel (metrics aggregated over blocks, flags)
  chanqc20_blocks.csv     the same metrics per block
  chanqc20_report.md      summary tables
  chanqc20_config.json    thresholds, options, input hashes, code hash, library versions

Sections
  A metadata    header index and type, 10-05 name and template position, membership and coordinates in the authors'
                .set, bad in which blocks, kept under v1.6 (or the exclusion reason)
  B quality     raw (sampled segments, unfiltered): mean, SD, MAD, exactly-zero share, flat-run share, non-finite,
                share at the segment extremes (saturation), ADC step (software-derived signals are off the lattice),
                50 Hz and 100-1000 Hz power ratios; preprocessed (whole blocks): SD, SD relative to the kept channels,
                MAD, excess kurtosis, share beyond 10 MAD
  C dependence  before CAR (preprocessed, original reference): max |r| with any other EEG-type channel and the
                relative RMS difference to it (uV), R^2 from the k nearest kept channels and from all other kept
                channels, max |r| with the EEG-type inputs that have no 10-05 name
  D spatial     median r with the nearest kept channels, correlation of the log spectrum with the neighbours' median log
                spectrum, relative band power, CAR impact (relative change of the CAR signal when the channel is left out)
  E report      CSV / Markdown, configuration and hashes

Version 20.1 (post hoc, 11 Oct 2026, after the first real-data run of version 20.0; report layer only)
  * 'outside reference range' now uses only the reference patients' kept channels that are also listed in the
    authors' .set (in 20.0 the range included kept channels absent from the .set, some of which had the very
    signature under study, which widened the range);
  * new composite flag INPUT_LIKE_SIGNAL (raw SD > 500 uV and max |r| >= 0.99 with an input without a 10-05 name),
    written to its own column 'posthoc_flags'. Its thresholds were chosen after seeing the 20.0 output; it is
    descriptive, is not a protocol criterion, and never changes a channel list or a model.

Usage
  python -m s2i.chanqc20 audit --sc_root SC_ROOT --build_json DATA17/p17_build.json --out RUNS/chanqc20
  python -m s2i.chanqc20 smoke --out /tmp/chanqc20_smoke          (synthetic data with planted problems)
  python -m s2i.chanqc20 rereport --out RUNS/chanqc20             (v20.1 report from an existing v20.0 output,
                                                                   no data re-read; writes *_v20_1 files)
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

from s2i.scales17 import PATIENTS, read_header, scalp_name, template_1005      # read-only reuse of the v1.6 rules

FOCUS = ["sub-P01", "sub-P02", "sub-P03", "sub-P05"]
REFERENCE = ["sub-P04", "sub-P06", "sub-P07"]          # two-amplifier patients whose kept count matched Table 1
K_NB, K_PRED = 4, 6                                    # neighbours for median r / spectrum, and for the R^2
THRESH = {
    "flat_share": 0.01,      # FLAT: share of raw samples in runs of identical values >= flat_run_s (zero runs included;
    "flat_run_s": 0.01,      #       the exactly-zero share alone is reported, not flagged: quantised signals hit 0 often)
    "extreme_share": 0.001,  # EXTREME: share of raw samples equal to the channel's segment minimum or maximum
    "step_rel_tol": 0.01,    # STEP: ADC step differs from the patient's modal step by more than 1 %
    "sd_low": 0.25,          # SD_LOW / SD_HIGH: preprocessed SD relative to the median of the kept channels
    "sd_high": 4.0,
    "dup_r": 0.98,           # DUPLICATE: max |r| with another channel above this ...
    "dup_rel_rms": 0.2,      # ... or relative RMS difference (uV) to the most correlated channel below this
    "pred_r2": 0.98,         # PREDICTABLE: R^2 from the K_PRED nearest kept channels
    "spec_r": 0.8,           # SPECTRUM: log-spectrum correlation with the neighbours' median below this
    "line_rel": 5.0,         # LINE50: raw 50 Hz ratio above this multiple of the kept channels' median
    "unnamed_r": 0.9,        # LIKE_UNNAMED: max |r| with an EEG-type input without a 10-05 name
}
VERSION = "20.1"
THRESH_POSTHOC = {           # added in 20.1 after the 20.0 results (post hoc, descriptive, 10-05 named channels only)
    "input_like_sd_uV": 500.0,   # INPUT_LIKE_SIGNAL: raw SD above this (uV) ...
    "input_like_r": 0.99,        # ... and max |r| with an input without a 10-05 name at least this
}
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 70)}
REF_RANGE_METRICS = ("sd_rel", "r_nb_med", "r2_nb", "spec_r", "raw_hf_ratio")
LIMITATION = ("Signal statistics give evidence about channel quality and relative independence only. They cannot "
              "prove whether a physical electrode was attached: a valid electrode can correlate strongly with its "
              "neighbours, and a channel that looks normal does not prove that an electrode was present. Flags are "
              "warnings with pre-stated thresholds, not labels; no channel is removed and no model is re-run.")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)


# ----------------------------------------------------------------------------- inputs
def file_hash(path, full_limit=64e6, part=8 * 2 ** 20):
    """sha256 of the whole file (<= 64 MB) or of its first and last 8 MB plus its size (stated as partial)."""
    if not os.path.exists(path):
        return {"path": path, "missing": True}
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= full_limit:
            for chunk in iter(lambda: f.read(2 ** 20), b""):
                h.update(chunk)
            kind = "full"
        else:
            h.update(f.read(part))
            f.seek(size - part)
            h.update(f.read(part))
            h.update(str(size).encode())
            kind = "partial (first + last 8 MB + size)"
    return {"path": path, "size": size, "sha256": h.hexdigest(), "kind": kind}


def template_positions():
    """10-05 template positions (m) keyed by the template's own spelling."""
    import warnings
    import mne
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            mont = mne.channels.make_standard_montage("standard_1005")
        except ValueError:
            mont = mne.channels.make_standard_montage("colin27_1005")
    return {k: np.asarray(v, float) for k, v in mont.get_positions()["ch_pos"].items()}


def _s(v):
    if isinstance(v, np.ndarray):
        return " ".join(str(x).strip() for x in v.ravel()).strip()
    return str(v).strip() if v is not None else ""


def set_meta(path, t1005):
    """Metadata of the authors' EEGLAB .set (labels, coordinates, reference, epoching, comments, history keywords)."""
    out = {"path": path, "exists": os.path.exists(path)}
    if not out["exists"]:
        return out
    from scipy.io import loadmat
    try:
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:                              # MATLAB v7.3: labels only
        from s2i.scales17 import set_channels
        labels = set_channels(path)
        out.update(labels=labels, format="v7.3 (labels only)")
        out["canon"] = {lb: scalp_name(lb, t1005)[0] for lb in labels}
        return out
    E = m["EEG"] if "EEG" in m else None
    get = (lambda k: getattr(E, k, None)) if E is not None else (lambda k: m.get(k))      # noqa: E731
    locs = np.atleast_1d(get("chanlocs"))
    labels, xyz, urchan = [], {}, {}
    for c in locs:
        lb = str(getattr(c, "labels", "")).strip()
        labels.append(lb)
        v = [getattr(c, k, None) for k in ("X", "Y", "Z")]
        xyz[lb] = all(isinstance(q, (int, float, np.floating, np.integer)) and np.isfinite(q) for q in v)
        u = getattr(c, "urchan", None)
        urchan[lb] = int(u) if isinstance(u, (int, np.integer, float, np.floating)) and np.isfinite(u) else None
    hist = _s(get("history"))
    ci = get("chaninfo")
    nodat = getattr(ci, "nodatchans", None) if ci is not None and not isinstance(ci, np.ndarray) else None
    nodat_labels = [str(getattr(c, "labels", "")).strip() for c in np.atleast_1d(nodat)] \
        if nodat is not None and not (isinstance(nodat, np.ndarray) and nodat.size == 0) else []
    data = get("data")
    out.update(format="v5/v7", labels=labels, has_xyz=xyz, urchan=urchan, nbchan=_s(get("nbchan")),
               srate=_s(get("srate")), trials=_s(get("trials")), pnts=_s(get("pnts")), ref=_s(get("ref")),
               comments=_s(get("comments")), setname=_s(get("setname")),
               data_location=data if isinstance(data, str) else "embedded",
               history_keywords={k: hist.count(k) for k in ("pop_interp", "pop_select", "pop_chanedit", "pop_reref",
                                                             "pop_rejchan", "clean_rawdata", "pop_runica",
                                                             "pop_subcomp", "pop_resample", "pop_epoch")},
               history_chars=len(hist), nodatchans=nodat_labels,
               icachansind=int(np.atleast_1d(get("icachansind")).size) if get("icachansind") is not None else None)
    out["canon"] = {lb: scalp_name(lb, t1005)[0] for lb in labels}
    return out


def read_bad(path):
    return open(path, encoding="latin-1").read().split() if os.path.exists(path) else []


# ----------------------------------------------------------------------------- B: raw metrics
def _flat_samples(x, min_run):
    """Number of samples in runs of >= min_run identical consecutive values."""
    eq = np.r_[False, np.diff(x) == 0, False]
    d = np.diff(eq.astype(np.int8))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    L = en - st + 1
    return int(L[L >= min_run].sum())


def _band(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(P[..., m].sum(-1)) if np.ndim(P) == 1 else P[..., m].sum(-1)


def raw_metrics(dat, ades, chans, n_seg, seg_sec):
    """Unfiltered raw statistics of `chans` on n_seg evenly spaced segments of seg_sec s (5 s from the edges)."""
    from scipy.signal import welch
    fs, n, hdr = read_header(ades)
    names = [c for c, _ in hdr]
    col = {}
    for i, c in enumerate(names):
        col.setdefault(c, i)
    assert os.path.getsize(dat) == 4 * len(hdr) * n, f"size mismatch {dat}"
    X = np.memmap(dat, dtype="<f4", mode="r", shape=(n, len(hdr)))
    seg = int(seg_sec * fs)
    lo, hi = int(5 * fs), n - int(5 * fs) - seg
    starts = np.linspace(lo, hi, n_seg).astype(int) if hi > lo else np.array([0])
    seg = min(seg, n)
    idx = [col[c] for c in chans]
    parts = [np.asarray(X[s:s + seg])[:, idx].astype(np.float64) for s in starts]
    del X
    min_run = max(2, int(round(THRESH["flat_run_s"] * fs)))
    out = {}
    for j, c in enumerate(chans):
        xs = [p[:, j] for p in parts]
        x = np.concatenate(xs)
        fin = np.isfinite(x)
        xf = x[fin]
        r = {"raw_nonfinite": int((~fin).sum()), "raw_n": int(x.size)}
        if xf.size == 0:
            out[c] = r
            continue
        med = np.median(xf)
        r.update(raw_mean_uV=float(xf.mean()), raw_sd_uV=float(xf.std()),
                 raw_mad_uV=float(np.median(np.abs(xf - med)) * 1.4826),
                 raw_zero_share=float(np.mean(xf == 0)),
                 raw_flat_share=float(sum(_flat_samples(s[np.isfinite(s)], min_run) for s in xs) / x.size))
        mn, mx = xf.min(), xf.max()
        r["raw_extreme_share"] = float(np.mean((xf == mn) | (xf == mx))) if mx > mn else 1.0
        u = np.unique(xf.astype(np.float32).astype(np.float64))
        du = np.diff(u)
        r["raw_step_uV"] = float(du[du > 0].min()) if (du > 0).any() else float("nan")
        r["raw_unique_share"] = float(u.size / xf.size)
        P = []
        for s in xs:
            s = s[np.isfinite(s)]
            if s.size >= int(fs):
                f, p = welch(s - s.mean(), fs=fs, nperseg=int(fs))
                P.append(p)
        if P:
            P = np.mean(P, 0)
            base = _band(f, P, 2, 40)
            r["raw_line50_ratio"] = _band(f, P, 48, 52) / base if base > 0 else float("nan")
            r["raw_hf_ratio"] = _band(f, P, 100, min(1000, fs / 2)) / base if base > 0 else float("nan")
        out[c] = r
    return out, {"fs": fs, "n": n, "segments": len(starts), "seg_s": seg / fs}


# ----------------------------------------------------------------------------- B-D: preprocessed metrics (one block)
def _r2(y, X):
    """R^2 of an OLS fit of y on X (+ intercept)."""
    A = np.c_[np.ones(len(y)), X]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    vy = np.var(y)
    return float(1 - np.var(res) / vy) if vy > 0 else float("nan")


def proc_metrics(npz, chans, pool, unnamed, pos, max_samples=100_000):
    """
    Metrics of `chans` on one preprocessed block (original reference, i.e. before CAR).
    pool    = kept channels (v1.6) present in the block: neighbours, R^2 predictors, SD reference and the CAR
    unnamed = EEG-type inputs without a 10-05 name
    pos     = header name -> template position (10-05 channels only)
    """
    from scipy.signal import welch
    z = np.load(npz, allow_pickle=True)
    names = [str(x).strip() for x in z["eeg_names"]]
    fs = float(z["fs"]) if "fs" in z.files else 200.0
    bad = [str(x).strip() for x in z["bad"]] if "bad" in z.files else []
    have = [c for c in chans if c in names]
    E = np.stack([z["eeg"][names.index(c)] for c in have]).astype(np.float64)
    L = E.shape[1]
    step = max(1, L // max_samples)
    S = E[:, ::step]
    sd = S.std(1)
    Zs = (S - S.mean(1, keepdims=True)) / np.where(sd > 0, sd, np.nan)[:, None]
    with np.errstate(invalid="ignore"):
        R = (Zs @ Zs.T) / Zs.shape[1]
    f, P = welch(E - E.mean(1, keepdims=True), fs=fs, nperseg=int(2 * fs))
    edges = np.geomspace(1, 70, 25)                       # 24 log-spaced bands, notch region excluded
    logP = np.stack([np.log10(P[:, (f >= lo) & (f < hi) & ~((f >= 48) & (f <= 52))].mean(1) + 1e-30)
                     for lo, hi in zip(edges[:-1], edges[1:]) if ((f >= lo) & (f < hi) & ~((f >= 48) & (f <= 52))).any()], 1)
    tot = _band(f, P, 1, 70)
    ix = {c: i for i, c in enumerate(have)}
    pool = [c for c in pool if c in ix]
    unnamed = [c for c in unnamed if c in ix]
    sd_pool = float(np.median([E[ix[c]].std() for c in pool])) if pool else float("nan")
    car = E[[ix[c] for c in pool]].mean(0) if pool else None
    out = {}
    for c in have:
        i = ix[c]
        x = E[i]
        med = np.median(x)
        mad = np.median(np.abs(x - med)) * 1.4826
        s_ = x.std()
        r = {"sd_uV": float(s_), "sd_rel": float(s_ / sd_pool) if sd_pool > 0 else float("nan"),
             "mad_uV": float(mad),
             "kurtosis": float(np.mean(((x - x.mean()) / s_) ** 4) - 3) if s_ > 0 else float("nan"),
             "big_share": float(np.mean(np.abs(x - med) > 10 * mad)) if mad > 0 else float("nan"),
             "bad_in_block": c in bad}
        for b, (lo, hi) in BANDS.items():
            r[f"rel_{b}"] = float(_band(f, P[i], lo, hi) / tot[i]) if tot[i] > 0 else float("nan")
        others = [j for j in range(len(have)) if j != i and np.isfinite(R[i, j])]
        if others:
            j = max(others, key=lambda q: abs(R[i, q]))
            r["r_max"], r["r_max_with"] = float(abs(R[i, j])), have[j]
            y = E[j] * np.sign(R[i, j])
            r["rel_rms_diff"] = float(np.sqrt(np.mean((x - y) ** 2)) / s_) if s_ > 0 else float("nan")
        un = [ix[u] for u in unnamed if u != c and np.isfinite(R[i, ix[u]])]
        r["r_unnamed_max"] = float(np.max(np.abs(R[i, un]))) if un else float("nan")
        if c in pos and pool:
            cand = [p for p in pool if p != c and p in pos]
            nb = sorted(cand, key=lambda p: np.linalg.norm(pos[p] - pos[c]))
            nb4, nb6 = nb[:K_NB], nb[:K_PRED]
            if nb4:
                r["neighbours"] = " ".join(nb4)
                r["r_nb_med"] = float(np.nanmedian([R[i, ix[p]] for p in nb4]))
                r["spec_r"] = float(np.corrcoef(logP[i], np.median(logP[[ix[p] for p in nb4]], 0))[0, 1])
            if len(nb6) >= 2 and np.isfinite(Zs[i]).all():
                r["r2_nb"] = _r2(Zs[i], Zs[[ix[p] for p in nb6]].T)
            if len(cand) >= 2 and np.isfinite(Zs[i]).all():
                r["r2_all"] = _r2(Zs[i], Zs[[ix[p] for p in cand]].T)
        if car is not None and c in pool and len(pool) > 1:
            wo = E[[ix[p] for p in pool if p != c]].mean(0)
            rc = np.sqrt(np.mean(car ** 2))
            r["car_delta"] = float(np.sqrt(np.mean((car - wo) ** 2)) / rc) if rc > 0 else float("nan")
        out[c] = r
    return out, {"fs": fs, "samples": L, "bad": bad}


# ----------------------------------------------------------------------------- one patient
def audit_patient(a, num, pid, setup, t1, t1005, pos_t, build):
    sid = f"sub-P{num:02d}"
    raw_dir = os.path.join(a.sc_root, "raw", f"SEEG{pid}")
    procs = sorted(glob.glob(os.path.join(a.sc_root, "proc", f"SEEG{pid}_Block*.npz")),
                   key=lambda p: int(re.search(r"Block(\d+)", p).group(1)))
    blocks = [int(re.search(r"Block(\d+)", p).group(1)) for p in procs]
    if a.blocks:
        keep_b = [(b, p) for b, p in zip(blocks, procs) if b in a.blocks]
        blocks, procs = [b for b, _ in keep_b], [p for _, p in keep_b]
    hashes = []
    if not procs:
        log(f"[{sid}] no preprocessed block -> skipped")
        return None, [], {"subject": sid, "skipped": "no preprocessed block"}, hashes
    # ---- A: header channels (first block), names, positions, v1.6 status, .set
    ades0 = os.path.join(raw_dir, f"Block{blocks[0]}", f"SEEG{pid}Block{blocks[0]}.ades")
    _, _, hdr = read_header(ades0)
    eeg_hdr = [(i, c) for i, (c, t) in enumerate(hdr) if t == "EEG"]
    B = build.get("patients", {}).get(sid, {})
    kept_canon = set(B.get("scalp_kept", []))
    excluded = B.get("scalp_excluded", {})
    sm = set_meta(os.path.join(raw_dir, f"SEEG{pid}.set"), t1005)
    set_canon = {v for v in sm.get("canon", {}).values() if v}
    set_xyz = {sm["canon"][lb]: ok for lb, ok in sm.get("has_xyz", {}).items() if sm["canon"].get(lb)}
    rows, named, unnamed, pos = {}, [], [], {}
    for k, (i, c) in enumerate(eeg_hdr):
        nm, why = scalp_name(c, t1005)
        kept = bool(nm) and nm in kept_canon and c not in excluded
        if nm is None:
            group = "no 10-05 name"
            unnamed.append(c)
        elif kept:
            group = "kept, in .set" if nm in set_canon else "kept, not in .set"
        else:
            group = f"not kept: {excluded.get(c, 'not in v1.6 kept list')}"
        if nm:
            named.append(c)
            if nm in pos_t:
                pos[c] = pos_t[nm]
        rows[c] = {"subject": sid, "id": pid, "setup": setup, "table1_scalp": t1,
                   "role": "focus" if sid in a.focus else "reference", "channel": c, "name_1005": nm or "",
                   "name_reason": why, "header_index": i, "eeg_input_index": k, "group": group, "kept_v16": kept,
                   "in_set": bool(nm) and nm in set_canon, "set_has_xyz": set_xyz.get(nm, "") if nm else ""}
    pool = [c for c in named if rows[c]["kept_v16"]]
    chans = named + unnamed
    # ---- B-D per block
    per_block = []
    bad_blocks = {c: [] for c in chans}
    for b, npz in zip(blocks, procs):
        base = os.path.join(raw_dir, f"Block{b}", f"SEEG{pid}Block{b}")
        hashes += [file_hash(base + ".ades"), file_hash(base + ".ades.bad"), file_hash(base + ".dat"), file_hash(npz)]
        _, _, hdr_b = read_header(base + ".ades")
        hb = {c for c, t in hdr_b if t == "EEG"}
        ch_b = [c for c in chans if c in hb]
        t0 = time.time()
        rm, rinfo = raw_metrics(base + ".dat", base + ".ades", ch_b, a.n_seg, a.seg_sec)
        pm, pinfo = proc_metrics(npz, ch_b, [c for c in pool if c in hb], unnamed, pos)
        for c in set(read_bad(base + ".ades.bad")) | set(pinfo["bad"]):
            if c in bad_blocks:
                bad_blocks[c].append(b)
        for c in ch_b:
            per_block.append({"subject": sid, "block": b, "channel": c, **rm.get(c, {}), **pm.get(c, {})})
        log(f"[{sid}] block {b}: {len(ch_b)} EEG-type channels, raw {rinfo['segments']} x {rinfo['seg_s']:.0f} s "
            f"at {rinfo['fs']:.0f} Hz, preprocessed {pinfo['samples'] / pinfo['fs'] / 60:.1f} min | "
            f"{time.time() - t0:.0f} s")
    hashes += [file_hash(os.path.join(raw_dir, f"SEEG{pid}.set")),
               file_hash(os.path.join(raw_dir, f"SEEG{pid}.fdt"))]
    # ---- aggregate over blocks: median; worst block for shares and counts
    worst = {"raw_nonfinite": np.sum, "raw_zero_share": np.max, "raw_flat_share": np.max,
             "raw_extreme_share": np.max, "big_share": np.max}
    for c in chans:
        rb = [r for r in per_block if r["channel"] == c]
        row = rows[c]
        row["blocks"] = len(rb)
        row["bad_blocks"] = " ".join(str(b) for b in sorted(set(bad_blocks[c])))
        keys = sorted({k for r in rb for k in r if k not in ("subject", "block", "channel")})
        for k in keys:
            vals = [r[k] for r in rb if k in r]
            if not vals:
                continue
            if isinstance(vals[0], str):
                row[k] = max(set(vals), key=vals.count)
            elif isinstance(vals[0], (bool, np.bool_)):
                row[k] = int(sum(vals))
            else:
                v = np.asarray(vals, float)
                row[k] = float(worst[k](v)) if k in worst else float(np.nanmedian(v)) if np.isfinite(v).any() \
                    else float("nan")
    steps = [rows[c].get("raw_step_uV", np.nan) for c in pool]
    steps = np.asarray([s for s in steps if np.isfinite(s)])
    modal = float(np.median(steps)) if steps.size else float("nan")
    line = np.asarray([rows[c].get("raw_line50_ratio", np.nan) for c in pool], float)
    line_med = float(np.nanmedian(line)) if np.isfinite(line).any() else float("nan")
    for c in chans:
        rows[c]["patient_modal_step_uV"] = modal
        rows[c]["flags"] = " ".join(flags_of(rows[c], modal, line_med))
        rows[c]["posthoc_flags"] = " ".join(posthoc_flags_of(rows[c]))
    info = {"subject": sid, "id": pid, "setup": setup, "table1_scalp": t1, "blocks": blocks,
            "n_eeg_type": len(eeg_hdr), "n_named": len(named), "n_unnamed": len(unnamed), "n_kept": len(pool),
            "kept_v16_from_build": len(kept_canon), "set": {k: v for k, v in sm.items() if k not in ("canon",)},
            "set_not_kept": sorted(n for n in set_canon if n not in {rows[c]["name_1005"] for c in pool}),
            "modal_step_uV": modal}
    return list(rows.values()), per_block, info, hashes


def flags_of(r, modal, line_med):
    T, f = THRESH, []
    g = lambda k: r.get(k, float("nan"))                 # noqa: E731
    if g("raw_flat_share") > T["flat_share"]:
        f.append("FLAT")
    if g("raw_nonfinite") > 0:
        f.append("NONFINITE")
    if g("raw_extreme_share") > T["extreme_share"]:
        f.append("EXTREME")
    if np.isfinite(g("raw_step_uV")) and np.isfinite(modal) and abs(g("raw_step_uV") / modal - 1) > T["step_rel_tol"]:
        f.append("STEP")
    if g("sd_rel") < T["sd_low"]:
        f.append("SD_LOW")
    if g("sd_rel") > T["sd_high"]:
        f.append("SD_HIGH")
    if g("r_max") > T["dup_r"] or g("rel_rms_diff") < T["dup_rel_rms"]:
        f.append("DUPLICATE")
    if g("r2_nb") > T["pred_r2"]:
        f.append("PREDICTABLE")
    if g("spec_r") < T["spec_r"]:
        f.append("SPECTRUM")
    if np.isfinite(line_med) and line_med > 0 and g("raw_line50_ratio") > T["line_rel"] * line_med:
        f.append("LINE50")
    if g("r_unnamed_max") > T["unnamed_r"]:
        f.append("LIKE_UNNAMED")
    return f


def posthoc_flags_of(r):
    """Post-hoc composite flag (version 20.1). Descriptive only; 10-05 named channels only."""
    T, f = THRESH_POSTHOC, []
    if not r.get("name_1005"):
        return f
    sd, ru = r.get("raw_sd_uV", float("nan")), r.get("r_unnamed_max", float("nan"))
    if np.isfinite(sd) and np.isfinite(ru) and sd > T["input_like_sd_uV"] and ru >= T["input_like_r"]:
        f.append("INPUT_LIKE_SIGNAL")
    return f


# ----------------------------------------------------------------------------- E: report
def _fmt(v, d=3):
    if v is None or v == "":
        return "–"
    if isinstance(v, (float, np.floating)):
        return "–" if not np.isfinite(v) else f"{v:.{d}f}"
    return str(v)


def _q(vals):
    v = np.asarray([x for x in vals if isinstance(x, (int, float, np.floating)) and np.isfinite(x)], float)
    if v.size == 0:
        return "–"
    return f"{np.median(v):.3f} [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}] (n={v.size})"


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def report(a, rows, infos):
    # 20.1: reference range from the reference patients' kept channels that are also listed in the authors' .set
    ref_rows = [r for r in rows if r["role"] == "reference" and r["kept_v16"] and r["in_set"]]
    ref_rng = {}
    for m in REF_RANGE_METRICS:
        v = np.asarray([r.get(m, np.nan) for r in ref_rows], float)
        v = v[np.isfinite(v)]
        if v.size:
            ref_rng[m] = (float(v.min()), float(v.max()))
    for r in rows:
        out = []
        if r["name_1005"]:
            for m, (lo, hi) in ref_rng.items():
                v = r.get(m, np.nan)
                if isinstance(v, float) and np.isfinite(v) and not lo <= v <= hi:
                    out.append(m)
        r["outside_reference_range"] = " ".join(out)
    L = [f"# Phase 20: audit of the ScaleOLDNEW scalp channels (descriptive; chanqc20 v{VERSION})", "",
         f"> {LIMITATION}", "",
         f"- focus: {', '.join(a.focus)}; reference (kept count matched Table 1): {', '.join(a.reference)}",
         f"- raw: {a.n_seg} segments x {a.seg_sec:g} s per block (unfiltered, from 5 s after the start to 5 s before "
         "the end); preprocessed: whole blocks, original reference (before CAR)",
         "- aggregation over blocks: median; worst block for zero / flat / extreme / >10 MAD shares and non-finite "
         "counts",
         f"- thresholds: {json.dumps(THRESH)}",
         f"- post-hoc thresholds (v20.1, chosen after the v20.0 output; column posthoc_flags; descriptive, not a "
         f"protocol criterion): {json.dumps(THRESH_POSTHOC)}",
         f"- neighbours: {K_NB} nearest kept channels (median r, spectrum), {K_PRED} for R^2 (10-05 template "
         "positions); 'outside reference range' = outside the min-max over the reference patients' kept channels "
         f"that are also listed in the authors' .set (v20.1), for {', '.join(REF_RANGE_METRICS)}", ""]
    L += ["## Patients", "", "| patient | ID | setup | Table 1 | EEG-type inputs | 10-05 named | kept (v1.6) | kept, not in "
          ".set | .set, not kept | .set: ref / srate / trials / comments | .set labels without coordinates | "
          "ADC step (uV) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i in infos:
        if i.get("skipped"):
            L.append(f"| {i['subject']} | | | | | | | | | {i['skipped']} | | |")
            continue
        s = i["set"]
        pr = [r for r in rows if r["subject"] == i["subject"]]
        nks = [r["channel"] for r in pr if r["group"] == "kept, not in .set"]
        noxyz = [lb for lb, ok in s.get("has_xyz", {}).items() if not ok]
        L.append(f"| {i['subject']} | {i['id']} | {i['setup']} | {i['table1_scalp']} | {i['n_eeg_type']} | "
                 f"{i['n_named']} | {i['n_kept']} | {', '.join(nks) or '–'} | {', '.join(i['set_not_kept']) or '–'} | "
                 f"{s.get('ref', '–')} / {s.get('srate', '–')} / {s.get('trials', '–')} / {s.get('comments', '–')} | "
                 f"{', '.join(noxyz) or '–'} | {_fmt(i['modal_step_uV'], 4)} |")
    L += ["", "Authors' .set history keywords (counts) and channels without data (chaninfo.nodatchans):", ""]
    for i in infos:
        s = i.get("set", {})
        if s.get("exists"):
            kw = {k: v for k, v in s.get("history_keywords", {}).items() if v}
            L.append(f"- {i['subject']}: history {kw or 'no listed keyword'} ({s.get('history_chars', 0)} chars); "
                     f"nodatchans: {', '.join(s.get('nodatchans', [])) or 'none'}; data: {s.get('data_location', '–')}")
        else:
            L.append(f"- {i['subject']}: no .set file")
    cols = [("channel", 0), ("header_index", 0), ("group", 0), ("bad_blocks", 0), ("raw_sd_uV", 1), ("sd_uV", 1),
            ("sd_rel", 2), ("raw_flat_share", 3), ("raw_step_uV", 4), ("r_max", 3), ("r_max_with", 0),
            ("r2_nb", 3), ("r_nb_med", 3), ("spec_r", 3), ("r_unnamed_max", 3), ("car_delta", 3), ("flags", 0),
            ("posthoc_flags", 0), ("outside_reference_range", 0)]
    for i in infos:
        if i.get("skipped"):
            continue
        pr = [r for r in rows if r["subject"] == i["subject"]]
        order = {"kept, not in .set": 0, "kept, in .set": 1}
        pr_named = sorted([r for r in pr if r["name_1005"]], key=lambda r: (order.get(r["group"], 2), r["header_index"]))
        L += ["", f"## {i['subject']} (ID {i['id']}, {i.get('role', '')})", "",
              "| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
        for r in pr_named:
            L.append("| " + " | ".join(_fmt(r.get(c, ""), d) if d else _fmt(r.get(c, "")) for c, d in cols) + " |")
        un = [r for r in pr if not r["name_1005"]]
        if un:
            L.append("")
            L.append(f"EEG-type inputs without a 10-05 name ({len(un)}): " + ", ".join(
                f"{r['channel']} (raw SD {_fmt(r.get('raw_sd_uV', ''), 0)} uV, flags: {r['flags'] or '–'})" for r in un))
    groups = [("focus: kept, in .set", lambda r: r["role"] == "focus" and r["group"] == "kept, in .set"),
              ("focus: kept, not in .set", lambda r: r["role"] == "focus" and r["group"] == "kept, not in .set"),
              ("reference: kept", lambda r: r["role"] == "reference" and r["kept_v16"]),
              ("all: not kept (10-05 named)", lambda r: r["name_1005"] and not r["kept_v16"]),
              ("all: no 10-05 name", lambda r: not r["name_1005"])]
    mets = ("raw_sd_uV", "sd_rel", "r2_nb", "r_nb_med", "spec_r", "r_unnamed_max", "raw_hf_ratio")
    L += ["", "## Channel groups: median [IQR] (n channels)", "", "| group | " + " | ".join(mets) + " |",
          "|---|" + "---|" * len(mets)]
    for g, sel in groups:
        rr = [r for r in rows if sel(r)]
        L.append(f"| {g} | " + " | ".join(_q([r.get(m, np.nan) for r in rr]) for m in mets) + " |")
    L += ["", "## Flag counts (10-05 named channels)", "", "| patient | group | channels | flagged | flags |",
          "|---|---|---|---|---|"]
    for i in infos:
        if i.get("skipped"):
            continue
        pr = [r for r in rows if r["subject"] == i["subject"] and r["name_1005"]]
        for g in sorted({r["group"] for r in pr}):
            rr = [r for r in pr if r["group"] == g]
            fl = {}
            for r in rr:
                for x in r["flags"].split():
                    fl[x] = fl.get(x, 0) + 1
            L.append(f"| {i['subject']} | {g} | {len(rr)} | {sum(bool(r['flags']) for r in rr)} | "
                     f"{', '.join(f'{k}: {v}' for k, v in sorted(fl.items())) or '–'} |")
    # ---- post-hoc composite flag (v20.1)
    named = [r for r in rows if r["name_1005"]]
    hit = [r for r in named if "INPUT_LIKE_SIGNAL" in r.get("posthoc_flags", "").split()]
    L += ["", "## Post-hoc composite flag INPUT_LIKE_SIGNAL (v20.1, descriptive)", "",
          f"Raw SD > {THRESH_POSTHOC['input_like_sd_uV']:g} uV and max |r| >= {THRESH_POSTHOC['input_like_r']:g} with "
          "an EEG-type input without a 10-05 name. The thresholds were chosen after seeing the v20.0 output. The flag "
          "describes a signal resembling those inputs; it does not establish whether an electrode was attached or "
          "how the signal was generated, and it changes no channel list or model.", "",
          "| group (10-05 named, all audited patients) | channels | flagged | max raw SD (uV) among unflagged |",
          "|---|---|---|---|"]
    for g in ("kept, in .set", "kept, not in .set"):
        rr = [r for r in named if r["group"] == g]
        un_sd = [r.get("raw_sd_uV", np.nan) for r in rr if r not in hit]
        un_sd = [v for v in un_sd if isinstance(v, float) and np.isfinite(v)]
        L.append(f"| {g} | {len(rr)} | {sum(r in hit for r in rr)} | {_fmt(max(un_sd) if un_sd else '', 1)} |")
    rr = [r for r in named if not r["kept_v16"]]
    L.append(f"| not kept (any reason) | {len(rr)} | {sum(r in hit for r in rr)} | – |")
    L += ["", "| patient | channel | group | raw_sd_uV | r_unnamed_max | r_max_with | r_nb_med | spec_r | car_delta |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in hit:
        L.append(f"| {r['subject']} | {r['channel']} | {r['group']} | {_fmt(r.get('raw_sd_uV', ''), 1)} | "
                 f"{_fmt(r.get('r_unnamed_max', ''))} | {_fmt(r.get('r_max_with', ''))} | {_fmt(r.get('r_nb_med', ''))} | "
                 f"{_fmt(r.get('spec_r', ''))} | {_fmt(r.get('car_delta', ''))} |")
    if not hit:
        L.append("| – | – | – | – | – | – | – | – | – |")
    return "\n".join(L)


# ----------------------------------------------------------------------------- commands
def audit(a):
    os.makedirs(a.out, exist_ok=True)
    t1005 = template_1005()
    pos_t = template_positions()
    build = json.load(open(a.build_json)) if a.build_json and os.path.exists(a.build_json) else {}
    if not build:
        log(f"WARNING: {a.build_json} not found -> v1.6 kept status unknown (every channel 'not kept')")
    want = a.patients or (a.focus + a.reference)
    rows, blocks, infos, hashes = [], [], [], []
    for num, pid, setup, t1 in PATIENTS:
        sid = f"sub-P{num:02d}"
        if sid not in want:
            continue
        r, b, info, h = audit_patient(a, num, pid, setup, t1, t1005, pos_t, build)
        info["role"] = "focus" if sid in a.focus else "reference"
        infos.append(info)
        hashes += h
        if r:
            rows += r
            blocks += b
    write_csv(os.path.join(a.out, "chanqc20_blocks.csv"), blocks)
    md = report(a, rows, infos)
    write_csv(os.path.join(a.out, "chanqc20_channels.csv"), rows)
    open(os.path.join(a.out, "chanqc20_report.md"), "w", encoding="utf-8").write(md + "\n")
    import scipy
    try:
        import mne
        mne_v = mne.__version__
    except ImportError:
        mne_v = None
    cfg = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "version": VERSION, "args": vars(a), "thresholds": THRESH,
           "thresholds_posthoc": THRESH_POSTHOC,
           "k_neighbours": K_NB, "k_predictors": K_PRED, "bands": BANDS, "limitation": LIMITATION,
           "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "mne": mne_v},
           "code": file_hash(os.path.abspath(__file__)),
           "inputs": [file_hash(a.build_json)] + hashes if a.build_json else hashes,
           "patients": infos}
    json.dump(cfg, open(os.path.join(a.out, "chanqc20_config.json"), "w"), indent=1, default=str)
    print(md)
    log(f"written: {a.out}/chanqc20_report.md, chanqc20_channels.csv, chanqc20_blocks.csv, chanqc20_config.json")


def smoke(a):
    """Synthetic ScaleOLDNEW-like data with planted problems; checks that the matching flags fire."""
    from scipy.io import savemat
    root = os.path.join(a.out, "sc")
    fs, fs_p, sec, step = 4000.0, 200.0, 60, 0.265839
    named = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "Cz", "Pz"]
    hdr = [(c, "EEG") for c in named] + [("C57", "EEG"), ("C58", "EEG"), ("ECG", "ECG"), ("A01", "SEEG"),
                                         ("A02", "SEEG")]
    pos_t = template_positions()
    t1005 = template_1005()
    rng = np.random.default_rng(0)
    build = {"patients": {}}
    for sid, pid in (("sub-P01", 89), ("sub-P04", 96)):
        n = int(fs * sec)
        n_p = int(fs_p * sec)
        for b in (1, 2):
            src = rng.standard_normal((6, n_p)).cumsum(1)
            src -= src.mean(1, keepdims=True)
            src /= src.std(1, keepdims=True)
            hum = 800 * rng.standard_normal(n_p)                         # large interference on floating inputs
            P = np.stack([pos_t[t1005[c.upper()]] for c in named])
            Wm = np.exp(-np.linalg.norm(P[:, None] - P[None], axis=-1) / 0.05)[:, :6] @ rng.standard_normal((6, 6))
            Ep = 10 * (Wm @ src) / np.abs(Wm).sum(1, keepdims=True)
            loc = rng.standard_normal((len(named), n_p)).cumsum(1)
            loc = (loc - loc.mean(1, keepdims=True)) / loc.std(1, keepdims=True)
            Ep += 3 * loc + 0.5 * rng.standard_normal((len(named), n_p))   # independent local activity (1/f-like)
            un = np.stack([hum + 50 * rng.standard_normal(n_p) for _ in range(2)])
            if sid == "sub-P01":
                Ep[named.index("F4")] = hum + 50 * rng.standard_normal(n_p)               # floating input
                Ep[named.index("O2")] = Ep[named.index("O1")] + 0.05 * rng.standard_normal(n_p)   # duplicate
                nb = ["C4", "P3", "O2", "Pz"]
                Ep[named.index("P4")] = np.mean([Ep[named.index(q)] for q in nb], 0)      # software-derived
            Ep[named.index("C3"), : n_p // 5] = 0 if sid == "sub-P01" else Ep[named.index("C3"), : n_p // 5]
            tp, tr = np.arange(n_p) / fs_p, np.arange(n) / fs
            up = lambda x: np.stack([np.interp(tr, tp, v) for v in np.atleast_2d(x)])   # noqa: E731
            raw = np.concatenate([up(Ep), up(un), np.zeros((3, n))], 0)
            raw[: len(named) + 2] += 0.3 * rng.standard_normal((len(named) + 2, n))     # sub-LSB noise
            q = np.round(raw / step) * step                                               # ADC lattice
            if sid == "sub-P01":
                q[named.index("P4")] = up(Ep[named.index("P4")])[0] + 1e-3 * rng.standard_normal(n)  # off lattice
                q[named.index("C3"), : n // 5] = 0
            d = os.path.join(root, "raw", f"SEEG{pid}", f"Block{b}")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, f"SEEG{pid}Block{b}.ades"), "w").write(
                f"#ADES header file \nsamplingRate = {fs:g}\nnumberOfSamples = {n}\n" + "".join(f"{c} = {t}\n"
                                                                                                for c, t in hdr))
            q.T.astype("<f4").tofile(os.path.join(d, f"SEEG{pid}Block{b}.dat"))
            open(os.path.join(d, f"SEEG{pid}Block{b}.ades.bad"), "w").write("F4\n" if sid == "sub-P01" else "")
            os.makedirs(os.path.join(root, "proc"), exist_ok=True)
            ep = np.concatenate([Ep, un], 0).astype(np.float32)
            np.savez(os.path.join(root, "proc", f"SEEG{pid}_Block{b}.npz"), eeg=ep,
                     eeg_names=np.array(named + ["C57", "C58"], dtype=object), fs=fs_p,
                     bad=np.array(["F4"] if sid == "sub-P01" else [], dtype=object))
        set_names = [c for c in named if not (sid == "sub-P01" and c in ("Fp2", "F4"))]
        cl = np.zeros((len(set_names),), dtype=[("labels", "O"), ("X", "O"), ("Y", "O"), ("Z", "O"), ("urchan", "O")])
        for k, c in enumerate(set_names):
            xyz = pos_t[t1005[c.upper()]] * 1000
            cl[k] = (c, xyz[0], xyz[1], xyz[2], k + 1)
        savemat(os.path.join(root, "raw", f"SEEG{pid}", f"SEEG{pid}.set"),
                {"EEG": {"chanlocs": cl, "nbchan": len(set_names), "srate": 256.0, "trials": 10, "pnts": 410,
                         "ref": "averef", "comments": "synthetic", "history": "EEG = pop_select(EEG);",
                         "data": f"SEEG{pid}.fdt"}})
        kept = [c for c in named if not (sid == "sub-P01" and c == "F4")]
        build["patients"][sid] = {"scalp_kept": kept, "scalp_excluded": {"F4": "bad"} if sid == "sub-P01" else {}}
    bj = os.path.join(a.out, "p17_build.json")
    json.dump(build, open(bj, "w"))
    b = argparse.Namespace(sc_root=root, build_json=bj, out=os.path.join(a.out, "out"), patients=None,
                           focus=["sub-P01"], reference=["sub-P04"], n_seg=4, seg_sec=5.0, blocks=None)
    audit(b)
    rows = {(r["subject"], r["channel"]): r for r in csv.DictReader(open(os.path.join(b.out, "chanqc20_channels.csv"),
                                                                           encoding="utf-8"))}
    expect = [(("sub-P01", "O2"), "DUPLICATE"), (("sub-P01", "P4"), "STEP"), (("sub-P01", "P4"), "PREDICTABLE"),
              (("sub-P01", "F4"), "LIKE_UNNAMED"), (("sub-P01", "C3"), "FLAT")]
    ok = True
    for key, fl in expect:
        got = rows[key]["flags"].split()
        print(f"  {key[0]} {key[1]}: expected {fl}, flags = {got} -> {'OK' if fl in got else 'MISSING'}")
        ok &= fl in got
    got = rows[("sub-P01", "F4")]["posthoc_flags"].split()
    print(f"  sub-P01 F4: expected INPUT_LIKE_SIGNAL (post hoc), posthoc_flags = {got} -> "
          f"{'OK' if 'INPUT_LIKE_SIGNAL' in got else 'MISSING'}")
    ok &= "INPUT_LIKE_SIGNAL" in got
    for key in (("sub-P01", "Cz"), ("sub-P04", "Cz"), ("sub-P04", "C4")):
        got = rows[key]["flags"].split() + rows[key]["posthoc_flags"].split()
        print(f"  {key[0]} {key[1]}: expected no flag, flags = {got} -> {'OK' if not got else 'UNEXPECTED'}")
        ok &= not got
    unn = [k for k, r in rows.items() if not r["name_1005"] and r["posthoc_flags"]]
    print(f"  inputs without a 10-05 name carrying a post-hoc flag: {unn} -> {'OK' if not unn else 'UNEXPECTED'}")
    ok &= not unn
    g = rows[("sub-P01", "Fp2")]["group"]
    print(f"  sub-P01 Fp2 group = {g!r} -> {'OK' if g == 'kept, not in .set' else 'WRONG'}")
    ok &= g == "kept, not in .set"
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    if not ok:
        sys.exit(1)


def rereport(a):
    """Rebuild the report with the v20.1 rules from an existing audit output (chanqc20_channels.csv + config).

    No data file is read again. The v20.0 files are left untouched; new files get the suffix _v20_1."""
    txt_cols = {"subject", "setup", "role", "channel", "name_1005", "name_reason", "group", "set_has_xyz",
                "bad_blocks", "r_max_with", "flags", "posthoc_flags", "outside_reference_range"}

    def conv(k, v):
        if k in txt_cols:
            return v
        if v in ("True", "False"):
            return v == "True"
        if v == "":
            return float("nan")
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    src_csv = os.path.join(a.out, "chanqc20_channels.csv")
    src_cfg = os.path.join(a.out, "chanqc20_config.json")
    with open(src_csv, encoding="utf-8") as f:
        rows = [{k: conv(k, v) for k, v in r.items()} for r in csv.DictReader(f)]
    cfg = json.load(open(src_cfg, encoding="utf-8"))
    for r in rows:
        r["posthoc_flags"] = " ".join(posthoc_flags_of(r))
    args = cfg.get("args", {})
    b = argparse.Namespace(focus=args.get("focus", FOCUS), reference=args.get("reference", REFERENCE),
                           n_seg=args.get("n_seg", 6), seg_sec=args.get("seg_sec", 10.0))
    infos = cfg.get("patients", [])
    md = report(b, rows, infos)
    write_csv(os.path.join(a.out, "chanqc20_channels_v20_1.csv"), rows)
    open(os.path.join(a.out, "chanqc20_report_v20_1.md"), "w", encoding="utf-8").write(md + "\n")
    cfg2 = dict(cfg, created=time.strftime("%Y-%m-%d %H:%M:%S"), version=VERSION, thresholds_posthoc=THRESH_POSTHOC,
                rereport_of={"channels_csv": file_hash(src_csv), "config": file_hash(src_cfg),
                             "version_of_source": cfg.get("version", "20.0")},
                code=file_hash(os.path.abspath(__file__)))
    json.dump(cfg2, open(os.path.join(a.out, "chanqc20_config_v20_1.json"), "w"), indent=1, default=str)
    print(md)
    log(f"written: {a.out}/chanqc20_report_v20_1.md, chanqc20_channels_v20_1.csv, chanqc20_config_v20_1.json "
        "(v20.0 files untouched)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("audit", "smoke", "rereport"))
    ap.add_argument("--sc_root", help="ScaleOLDNEW root of Phase 14 (proc/, raw/)")
    ap.add_argument("--build_json", help="Phase-17 build log (DATA17/p17_build.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", nargs="*", default=None, help="default: focus + reference")
    ap.add_argument("--focus", nargs="*", default=FOCUS)
    ap.add_argument("--reference", nargs="*", default=REFERENCE)
    ap.add_argument("--n_seg", type=int, default=6, help="raw segments per block")
    ap.add_argument("--seg_sec", type=float, default=10.0, help="length of each raw segment (s)")
    ap.add_argument("--blocks", nargs="*", type=int, default=None, help="default: every block")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    {"audit": audit, "smoke": smoke, "rereport": rereport}[a.cmd](a)


if __name__ == "__main__":
    main()
