"""
Phase 17 - external validation on ScaleOLDNEW, exactly as in paper/external_validation_protocol.md (v1.6, locked).

Commands
  build   Phase-14 output (proc/<patient>_Block<k>.npz: notch 50/100 Hz, 0.5-70 Hz, 200 Hz, microvolts) + the raw
          ADES files and spreadsheets on Drive -> one data folder per evaluation level, in the s2i format read by
          s2i.readout --strict and s2i.review13 volts:
            primary      all windows, no automatic artefact rejection, no target clipping
            secondary    windows with |z| > 15 (CAR scalp EEG) removed, targets clipped at |z| = 10, median / MAD of
                         the whole block
            sensitivity  the same thresholds, median / MAD of the first 20 % of the windows
          plus the execution log (p17_build.md / p17_build.json): channel rules and the Table-1 consistency check,
          non-finite counts (raw data), recording gaps, windows, split, inclusion, target validity, markers.
  runs    copy each final encoder (s2i.shared --final -> <variant>/enc_final.pt) to one file per patient
          (<runs>/<variant>/enc_sub-Pxx.pt), so that s2i.readout runs on ScaleOLDNEW unchanged
  report  H1 / H2 (paired two-sided Wilcoxon, Holm over H1-H2 only, median direction, patient bootstrap CI of the
          mean), descriptive analyses, constant-prediction counts, common mode by recording setup
          -> p17_report.md / p17_report.json

Rules implemented here (protocol sections in brackets):
  scalp channel [4]   EEG type in the header; name (after the dataset's lookup table) is an MNE 10-05 position; not
                      A1/A2/M1/M2; not bad in any block; non-zero SD in every block; no non-finite raw sample
  target [4, 7]       pair of the BipolarChannels sheet; both contacts recorded in every block, never bad, no
                      non-finite raw sample; bipolar = A - B (microvolts); non-zero variance in calibration and test
  windows [5]         8 s, consecutive, inside each block (file) only, 3.4 s from the block edges; a window
                      overlapping a recording gap (>= 50 ms where all kept scalp channels, or all target contacts, are
                      exactly 0 or non-finite in the RAW data) or its 3.4 s margins is dropped; the last piece of a
                      block is kept if >= 2 s
  split [5]           windows ordered by block and time; n_cal = max(3, round(0.2 N)); 1-window gap; rest = test;
                      patient included only if n_cal >= 5, >= 3 scalp channels and >= 3 valid targets
  CAR [4]             mean of the kept scalp channels subtracted at every sample (microvolts), before normalisation
Storage: every signal is robust-normalised with the median / MAD of ALL kept samples of the patient and stored as
float16 together with these statistics (norm_stats, one pseudo-session "all"); s2i.readout --strict undoes this and
re-normalises with the calibration windows only, exactly as for ds004752.

Usage
-----
python -m s2i.scales17 build  --sc_root SC_ROOT --out DATA17
python -m s2i.scales17 runs   --enc_root ENC17 --data DATA17/primary --runs RUNS17
python -m s2i.scales17 report --data DATA17 --runs RUNS17
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np

FS = 200
WIN, MIN_REM, EDGE = 8 * FS, 2 * FS, 680              # 8 s windows, >= 2 s remainder, 3.4 s margins (samples)
GAP_SEC = 0.05
CAL_FRAC, MIN_CAL, MIN_SCALP, MIN_TGT = 0.2, 5, 3, 3
REJECT_Z, CLIP_Z = 15.0, 10.0
LEVELS = ("primary", "secondary", "sensitivity")
LOOKUP = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "FB1": "FT9", "FB2": "FT10", "TP1": "TP9", "TP2": "TP10"}
EXCLUDED_NAMES = {"A1", "A2", "M1", "M2"}
# Table 1 of Barborica et al. 2023: (patient number, ID, recording setup, number of scalp electrodes).
# Setup from the Methods: patients 1-8 two 128-channel amplifiers, patients 9-12 one 256-channel amplifier.
# Scalp counts from the published Table 1 (ID 90 = 30 also checked against the PDF). They disagree with the authors'
# EEGLAB scalp files for some patients (see Addendum A); the cause is unknown.
PATIENTS = [(1, 89, "two amplifiers", 30), (2, 90, "two amplifiers", 30), (3, 92, "two amplifiers", 30),
            (4, 96, "two amplifiers", 38), (5, 97, "two amplifiers", 35), (6, 98, "two amplifiers", 38),
            (7, 99, "two amplifiers", 38), (8, 101, "two amplifiers", 40), (9, 102, "one amplifier", 40),
            (10, 104, "one amplifier", 40), (11, 105, "one amplifier", 37), (12, 107, "one amplifier", 40)]
CONTACT = re.compile(r"^[A-Z]{1,2}'?\d{1,2}$")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)


def n_cal_of(n):
    return max(3, int(round(n * CAL_FRAC)))


def robust(x):
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True) * 1.4826
    return med, np.where(mad < 1e-30, 1.0, mad)


# ----------------------------------------------------------------------------- inputs
def template_1005():
    import warnings
    import mne
    with warnings.catch_warnings():                   # renamed colin27_1005 from MNE 1.14 (same positions)
        warnings.simplefilter("ignore", FutureWarning)
        try:
            mont = mne.channels.make_standard_montage("standard_1005")
        except ValueError:
            mont = mne.channels.make_standard_montage("colin27_1005")
    return {n.upper(): n for n in mont.ch_names}


def scalp_name(name, t1005):
    """-> (10-05 name or None, reason)."""
    up = LOOKUP.get(name.strip().upper(), name.strip().upper()).upper()
    if up in EXCLUDED_NAMES:
        return None, "A1/A2/M1/M2"
    if up not in t1005:
        return None, "not a 10-05 position"
    return t1005[up], ""


def load_block(path):
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in ("eeg", "seeg")}
    for k in ("eeg_names", "seeg_names", "bad"):
        out[k] = [str(x).strip() for x in z[k]]
    out["markers"] = [(str(m[0]).strip(), float(m[1])) for m in z["markers"]]
    out["fs_raw"], out["n_raw"] = float(z["fs_raw"]), int(z["n_raw"])
    return out


def read_header(path):
    fs, n, chans = None, None, []
    for ln in open(path, encoding="latin-1").read().splitlines():
        if ln.startswith("#") or "=" not in ln:
            continue
        k, v = [x.strip() for x in ln.split("=", 1)]
        if k == "samplingRate":
            fs = float(v)
        elif k == "numberOfSamples":
            n = int(v)
        else:
            chans.append((k, v))
    return fs, n, chans


def raw_scan(dat, ades, count_names, scalp, contacts, chunk=32768):
    """
    One pass over a raw block (float32, sample-major, microvolts): non-finite samples per channel of count_names,
    and recording gaps = runs of >= 50 ms in which all `scalp` channels, or all `contacts`, are exactly zero or
    non-finite. Gaps are returned as [start, stop) intervals at 200 Hz.
    """
    fs, n, chans = read_header(ades)
    col = {}
    for i, (c, _) in enumerate(chans):
        col.setdefault(c, i)
    assert os.path.getsize(dat) == 4 * len(chans) * n, f"size mismatch {dat}"
    X = np.memmap(dat, dtype="<f4", mode="r", shape=(n, len(chans)))
    cc = [col[c] for c in count_names]
    pos = {c: j for j, c in enumerate(count_names)}
    si, ti = [pos[c] for c in scalp], [pos[c] for c in contacts]
    nonfin = np.zeros(len(cc), np.int64)
    dead = np.zeros(n, bool)
    for k in range(0, n, chunk):
        x = np.asarray(X[k:k + chunk])[:, cc]
        bad = ~np.isfinite(x)
        nonfin += bad.sum(0)
        z = bad | (x == 0)
        dead[k:k + chunk] = (z[:, si].all(1) if si else False) | (z[:, ti].all(1) if ti else False)
    del X
    gaps = []
    d = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for s0, s1 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if s1 - s0 >= int(np.ceil(GAP_SEC * fs)):
            gaps.append((int(np.floor(s0 * FS / fs)), int(np.ceil(s1 * FS / fs))))
    return dict(zip(count_names, nonfin.tolist())), gaps, fs


def windows_of(L, gaps):
    """Consecutive 8-s windows of one block -> (kept [(s0, s1)], number dropped for gaps, remainder kept / dropped)."""
    blocked = np.zeros(L, bool)
    blocked[:EDGE] = blocked[L - EDGE:] = True
    for g0, g1 in gaps:
        blocked[max(0, g0 - EDGE):min(L, g1 + EDGE)] = True
    kept, n_gap, rem = [], 0, "none"
    s, end = EDGE, L - EDGE
    while s < end:
        e = min(s + WIN, end)
        if e - s < WIN:
            if e - s < MIN_REM:
                rem = f"dropped ({(e - s) / FS:.1f} s)"
                break
            rem = f"kept ({(e - s) / FS:.1f} s)"
        if blocked[s:e].any():
            n_gap += 1
        else:
            kept.append((s, e))
        s = e
    return kept, n_gap, rem


# ----------------------------------------------------------------------------- one patient
def build_patient(a, num, pid, setup, t1005_n, t1005):
    sid = f"sub-P{num:02d}"
    raw_dir = os.path.join(a.sc_root, "raw", f"SEEG{pid}")
    files = sorted(glob.glob(os.path.join(a.sc_root, "proc", f"SEEG{pid}_Block*.npz")),
                   key=lambda p: int(re.search(r"Block(\d+)", p).group(1)))
    info = {"subject": sid, "patient": num, "id": pid, "setup": setup, "table1_scalp": t1005_n, "excluded": None}
    if not files:
        info["excluded"] = "no preprocessed block"
        return info, None
    blocks, nums = [], []
    for f in files:
        try:
            blocks.append(load_block(f))
            nums.append(int(re.search(r"Block(\d+)", f).group(1)))
        except Exception as e:                                         # unreadable block: skipped and reported
            info.setdefault("unreadable_blocks", []).append(f"{os.path.basename(f)}: {type(e).__name__}")
    info["blocks"] = nums
    bad_all = sorted({c for b in blocks for c in b["bad"]})
    info["bad_any_block"] = bad_all

    # ---- scalp channels (rules 1-5; non-finite from the raw scan below)
    cand = [c for c in blocks[0]["eeg_names"] if all(c in b["eeg_names"] for b in blocks)]
    reasons, keep, canon = {}, [], []
    for c in cand:
        nm, why = scalp_name(c, t1005)
        if nm is None:
            reasons[c] = why
        elif c in bad_all:
            reasons[c] = "bad"
        elif nm in canon:
            reasons[c] = f"duplicate of {nm}"
        elif any(np.std(b["eeg"][b["eeg_names"].index(c)].astype(np.float64)) == 0 for b in blocks):
            reasons[c] = "zero SD in a block"
        else:
            keep.append(c)
            canon.append(nm)

    # ---- targets: BipolarChannels pairs, contacts recorded in every block and never bad
    import pandas as pd
    xl = sorted(glob.glob(os.path.join(raw_dir, "*ChannelData.xlsx")))
    sheet = pd.read_excel(xl[0], sheet_name="BipolarChannels")
    recorded = set.intersection(*[set(b["seeg_names"]) for b in blocks])
    pairs, t_reason = [], {}
    for _, row in sheet.iterrows():
        nm = str(row["name"]).strip()
        parts = nm.split("-")
        if len(parts) != 2:
            t_reason[nm] = "name not A-B"
            continue
        p0, p1 = parts[0].strip(), parts[1].strip()
        if p0 not in recorded or p1 not in recorded:
            t_reason[nm] = "contact not recorded in every block"
        elif p0 in bad_all or p1 in bad_all:
            t_reason[nm] = "contact bad"
        else:
            pairs.append({"name": nm, "a": p0, "b": p1, "mni": [float(row[k]) for k in ("x_mni", "y_mni", "z_mni")],
                          "structure": str(row.get("structure", ""))})

    # ---- raw scan: non-finite counts and recording gaps (repeated if the non-finite rule changes the sets)
    seeg_all = [c for c in blocks[0]["seeg_names"] if c in recorded]
    count_names = cand + seeg_all
    nonfin, gaps, fs_raw = {c: 0 for c in count_names}, {}, {}
    for it in range(3):
        contacts = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
        nonfin = {c: 0 for c in count_names}
        for k, b in zip(nums, blocks):
            base = os.path.join(raw_dir, f"Block{k}", f"SEEG{pid}Block{k}")
            nf, gaps[k], fs_raw[k] = raw_scan(base + ".dat", base + ".ades", count_names, keep, contacts)
            for c, v in nf.items():
                nonfin[c] += v
        bad_scalp = [c for c in keep if nonfin[c] > 0]
        bad_ct = {c for c in contacts if nonfin[c] > 0}
        if not bad_scalp and not bad_ct:
            break
        for c in bad_scalp:
            reasons[c] = "non-finite raw sample"
            canon.pop(keep.index(c))
            keep.remove(c)
        for p in [p for p in pairs if p["a"] in bad_ct or p["b"] in bad_ct]:
            t_reason[p["name"]] = "contact non-finite"
            pairs.remove(p)
    info.update(scalp_candidates=len(cand), scalp_kept=canon, n_scalp=len(keep), scalp_excluded=reasons,
                nonfinite={c: v for c, v in nonfin.items() if v > 0}, nonfinite_channels_checked=len(count_names),
                n_pairs_sheet=int(len(sheet)), pairs_excluded=t_reason, fs_raw=fs_raw,
                gaps={k: v for k, v in gaps.items()})
    contacts = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})

    # ---- signals per block (microvolts) and windows
    S, W, wl = {}, [], {}
    for k, b in zip(nums, blocks):
        e = np.stack([b["eeg"][b["eeg_names"].index(c)] for c in keep]).astype(np.float64)
        m = np.stack([b["seeg"][b["seeg_names"].index(c)] for c in contacts]).astype(np.float64) if contacts \
            else np.zeros((0, e.shape[1]))
        L = min(e.shape[1], m.shape[1])
        e, m = e[:, :L], m[:, :L]
        ci = {c: i for i, c in enumerate(contacts)}
        bip = np.stack([m[ci[p["a"]]] - m[ci[p["b"]]] for p in pairs]) if pairs else np.zeros((0, L))
        S[k] = {"orig": e, "car": e - e.mean(0, keepdims=True), "mono": m, "bip": bip}
        kw, n_gap, rem = windows_of(L, gaps[k])
        stim = [(c, t) for c, t in b["markers"] if c.isdigit()]
        codes = {}
        for c, _ in stim:
            codes[c] = codes.get(c, 0) + 1
        wl[k] = {"minutes": L / FS / 60, "windows": len(kw), "dropped_gap": n_gap, "remainder": rem,
                 "gaps": len(gaps[k]), "gap_seconds": sum(g1 - g0 for g0, g1 in gaps[k]) / FS, "stim_markers": codes,
                 "other_markers": sorted({c for c, _ in b["markers"] if not c.isdigit()})}
        onsets = np.array([int(round(t * FS)) for _, t in stim], int)
        for s0, s1 in kw:
            W.append({"block": k, "s0": s0, "s1": s1, "onsets": (onsets[(onsets >= s0) & (onsets < s1)] - s0).tolist()})
    info["block_windows"] = wl
    info["n_windows"] = len(W)
    if len(keep) < MIN_SCALP:
        info["excluded"] = f"{len(keep)} scalp channels (< {MIN_SCALP})"
    meta = dict(sid=sid, pid=pid, setup=setup, canon=canon, pairs=pairs, contacts=contacts)
    return info, (S, W, meta)


def level_windows(level, S, W):
    """-> kept windows and their signals for one evaluation level (rejection / clipping as in the protocol)."""
    sig = [{k: S[w["block"]][k][:, w["s0"]:w["s1"]] for k in ("orig", "car", "mono", "bip")} for w in W]
    if level == "primary":
        return W, sig, {}
    if level == "secondary":                          # whole-block median / MAD
        st = {k: {b: robust(S[b][k]) for b in S} for k in ("car", "bip")}
        get = lambda k, w: st[k][w["block"]]                                    # noqa: E731
    else:                                             # first 20 % of the windows, fixed for all later windows
        nc = n_cal_of(len(W))
        st = {k: robust(np.concatenate([s[k] for s in sig[:nc]], 1)) for k in ("car", "bip")}
        get = lambda k, w: st[k]                                                # noqa: E731
    kw, ks = [], []
    for w, s in zip(W, sig):
        med, mad = get("car", w)
        if np.abs((s["car"] - med) / mad).max() > REJECT_Z:
            continue
        s = dict(s)                                   # monopolar signals are used at the primary level only
        med, mad = get("bip", w)
        s["bip"] = np.clip(s["bip"], med - CLIP_Z * mad, med + CLIP_Z * mad)
        kw.append(w)
        ks.append(s)
    return kw, ks, {"rejected_windows": len(W) - len(kw)}


def write_level(a, level, info, data):
    import torch
    S, W, meta = data
    kw, ks, extra = level_windows(level, S, W)
    N = len(kw)
    nc = n_cal_of(N)
    out = {"n_windows": N, "n_cal": nc, "n_test": max(N - nc - 1, 0), **extra,
           "calib_minutes": sum(w["s1"] - w["s0"] for w in kw[:nc]) / FS / 60,
           "test_minutes": sum(w["s1"] - w["s0"] for w in kw[nc + 1:]) / FS / 60, "excluded": info["excluded"]}
    if out["excluded"] is None and nc < MIN_CAL:
        out["excluded"] = f"n_cal = {nc} (< {MIN_CAL}; N = {N})"
    valid, why = [], {}
    for j, p in enumerate(meta["pairs"]):
        vc = np.var(np.concatenate([s["bip"][j] for s in ks[:nc]])) if nc and ks else 0.0
        vt = np.var(np.concatenate([s["bip"][j] for s in ks[nc + 1:]])) if N > nc + 1 else 0.0
        if vc > 0 and vt > 0:
            valid.append(j)
        else:
            why[p["name"]] = "zero variance in calibration or test windows"
    out["n_valid_targets"], out["targets_zero_variance"] = len(valid), why
    if out["excluded"] is None and len(valid) < MIN_TGT:
        out["excluded"] = f"{len(valid)} valid targets (< {MIN_TGT})"
    if out["excluded"] is not None:
        return out
    pairs = [meta["pairs"][j] for j in valid]
    cts = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    ci = [meta["contacts"].index(c) for c in cts]
    arrays = {"eeg_orig": np.concatenate([s["orig"] for s in ks], 1),
              "eeg_car": np.concatenate([s["car"] for s in ks], 1),
              "ieeg_bip": np.concatenate([s["bip"][valid] for s in ks], 1)}
    if level == "primary":                            # monopolar only for the common-mode analysis (Section 8)
        arrays["ieeg_mono"] = np.concatenate([s["mono"][ci] for s in ks], 1)
    d = {"subject": meta["sid"], "dataset": "ScaleOLDNEW", "id": meta["pid"], "setup": meta["setup"], "fs": FS,
         "level": level, "eeg_names": meta["canon"], "contact_names": cts,
         "bip_names": [p["name"] for p in pairs], "bip_structure": [p["structure"] for p in pairs],
         "bip_mni": torch.tensor([p["mni"] for p in pairs], dtype=torch.float32),
         "bip_pairs": [(cts.index(p["a"]), cts.index(p["b"])) for p in pairs],
         "window_block": [w["block"] for w in kw], "window_start": [w["s0"] for w in kw],
         "stim_events": [(i, o) for i, w in enumerate(kw) for o in w["onsets"]],
         "norm_clip_z": float("inf") if level == "primary" else CLIP_Z, "norm_stats": {}}
    bounds, c = [], 0
    for w in kw:
        bounds.append((c, c + w["s1"] - w["s0"]))
        c += w["s1"] - w["s0"]
    d["trial_bounds"] = torch.tensor(bounds, dtype=torch.long)
    d["trial_session"] = ["all"] * N
    n_inf = 0
    for k, x in arrays.items():                       # storage normalisation (undone by readout --strict)
        med, mad = robust(x)
        z = (x - med) / mad
        n_inf += int((np.abs(z) > 6e4).sum())
        d[k] = torch.from_numpy(np.clip(z, -6e4, 6e4).astype(np.float16))
        d["norm_stats"][k] = {"all": {"med": med[:, 0].tolist(), "mad": mad[:, 0].tolist()}}
    out["float16_overflow_samples"] = n_inf
    od = os.path.join(a.out, level)
    os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, f"{meta['sid']}.pt")
    torch.save(d, fp + ".tmp")
    os.replace(fp + ".tmp", fp)
    torch.save({k: d[k] for k in ("subject", "norm_stats", "norm_clip_z", "trial_bounds", "trial_session")},
               os.path.join(od, f"{meta['sid']}_normstats.pt"))
    out["stim_events_in_windows"] = len(d["stim_events"])
    return out


def filter_record():
    """Filter / resampling configuration of the installed MNE (the Phase-14 preprocessing used the same functions)."""
    import inspect
    import mne
    from s2i.data.preprocess_ds004752 import BAND, NOTCH
    h = mne.filter.create_filter(None, 4096.0, BAND[0], BAND[1], verbose=False)
    sig = lambda f: {k: str(v.default) for k, v in inspect.signature(f).parameters.items()      # noqa: E731
                     if v.default is not inspect.Parameter.empty}
    return {"mne": mne.__version__, "notch_hz": list(NOTCH), "band_hz": list(BAND),
            "bandpass_length_at_4096_hz_s": len(h) / 4096.0,
            "notch_filter_defaults": sig(mne.io.BaseRaw.notch_filter), "filter_defaults": sig(mne.io.BaseRaw.filter),
            "resample_defaults": sig(mne.io.BaseRaw.resample)}


def build(a):
    os.makedirs(a.out, exist_ok=True)
    t1005 = template_1005()
    jp = os.path.join(a.out, "p17_build.json")
    res = json.load(open(jp)) if os.path.exists(jp) else {"patients": {}}
    res["filters"] = filter_record()
    lt = os.path.join(a.sc_root, "raw", "ElectrodeLabelsLookupTable.txt")
    res["lookup_table_file"] = open(lt, encoding="latin-1").read() if os.path.exists(lt) else "not found"
    res["rules"] = {"window_s": WIN / FS, "edge_margin_s": EDGE / FS, "min_remainder_s": MIN_REM / FS,
                    "gap_min_s": GAP_SEC, "cal_frac": CAL_FRAC, "min_cal": MIN_CAL, "min_scalp": MIN_SCALP,
                    "min_targets": MIN_TGT, "reject_z": REJECT_Z, "clip_z": CLIP_Z, "lookup": LOOKUP}
    for num, pid, setup, t1 in PATIENTS:
        sid = f"sub-P{num:02d}"
        if a.patients and sid not in a.patients and str(pid) not in a.patients:
            continue
        if sid in res["patients"] and all(lv in res["patients"][sid].get("levels", {}) for lv in LEVELS):
            log(f"[{sid}] already built")
            continue
        t0 = time.time()
        info, data = build_patient(a, num, pid, setup, t1, t1005)
        info["levels"] = {}
        if data is not None:
            for lv in LEVELS:
                info["levels"][lv] = write_level(a, lv, info, data)
        del data
        res["patients"][sid] = info
        json.dump(res, open(jp + ".tmp", "w"), indent=1, default=str)
        os.replace(jp + ".tmp", jp)
        lv = info["levels"].get("primary", {})
        log(f"[{sid}] ID {pid}: {info.get('n_scalp', 0)} scalp (Table 1: {t1}), "
            f"{lv.get('n_valid_targets', 0)} targets, {lv.get('n_windows', 0)} windows "
            f"(calib {lv.get('n_cal', 0)}, test {lv.get('n_test', 0)}), excluded: {lv.get('excluded', info['excluded'])} | "
            f"{(time.time() - t0) / 60:.1f} min")
    write_build_md(a, res)


def write_build_md(a, res):
    P = res["patients"]
    f = res["filters"]
    L = ["# Phase 17 execution log: data (protocol v1.6, Sections 4, 5, 9)", "",
         f"- MNE {f['mne']}; notch {f['notch_hz']} Hz, band-pass {f['band_hz']} Hz (band-pass FIR length at 4096 Hz: "
         f"{f['bandpass_length_at_4096_hz_s']:.2f} s); defaults: notch {f['notch_filter_defaults']}, "
         f"filter {f['filter_defaults']}, resample {f['resample_defaults']}",
         f"- rules: {res['rules']}", "", "## Scalp channels (consistency check against Table 1; not used to change rules)", "",
         "| patient | ID | setup | blocks | candidates (EEG in every block) | kept | Table 1 | excluded (reason: n) |",
         "|---|---|---|---|---|---|---|---|"]
    for sid, i in P.items():
        rs = {}
        for _, why in i.get("scalp_excluded", {}).items():
            rs[why] = rs.get(why, 0) + 1
        L.append(f"| {sid} | {i['id']} | {i['setup']} | {len(i.get('blocks', []))} | {i.get('scalp_candidates', 0)} | "
                 f"{i.get('n_scalp', 0)} | {i['table1_scalp']} | " + ", ".join(f"{k}: {v}" for k, v in rs.items()) + " |")
    L += ["", "Kept scalp channels per patient:", ""]
    L += [f"- {sid}: {', '.join(i.get('scalp_kept', []))}" for sid, i in P.items()]
    L += ["", "## Non-finite raw samples (all scalp candidates and recorded contacts; channels with 0 not listed)", ""]
    for sid, i in P.items():
        nf = i.get("nonfinite", {})
        L.append(f"- {sid}: {i.get('nonfinite_channels_checked', 0)} channels checked; "
                 + (", ".join(f"{c}: {v}" for c, v in nf.items()) if nf else "0 non-finite samples"))
    L += ["", "## Targets", "", "| patient | pairs in sheet | after recording / bad / non-finite rules | valid (primary) | "
          "excluded (reason: n) |", "|---|---|---|---|---|"]
    for sid, i in P.items():
        rs = {}
        for _, why in i.get("pairs_excluded", {}).items():
            rs[why] = rs.get(why, 0) + 1
        lv = i["levels"].get("primary", {})
        for why in lv.get("targets_zero_variance", {}).values():
            rs[why] = rs.get(why, 0) + 1
        L.append(f"| {sid} | {i.get('n_pairs_sheet', 0)} | {i.get('n_pairs_sheet', 0) - len(i.get('pairs_excluded', {}))} | "
                 f"{lv.get('n_valid_targets', 0)} | " + ", ".join(f"{k}: {v}" for k, v in rs.items()) + " |")
    L += ["", "## Blocks, gaps, windows and markers", "",
          "| patient | block | min | raw Hz | gaps (n, s) | windows kept | dropped (gap) | last piece | stimulus markers | other markers |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for sid, i in P.items():
        for k, b in i.get("block_windows", {}).items():
            L.append(f"| {sid} | {k} | {b['minutes']:.1f} | {i['fs_raw'].get(k, i['fs_raw'].get(str(k), ''))} | "
                     f"{b['gaps']}, {b['gap_seconds']:.1f} | {b['windows']} | {b['dropped_gap']} | {b['remainder']} | "
                     f"{b['stim_markers']} | {len(b['other_markers'])} |")
    for lv in LEVELS:
        L += ["", f"## Split and inclusion: {lv}", "",
              "| patient | windows | rejected | calib | test | calib min | test min | valid targets | float16 clipped | excluded |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for sid, i in P.items():
            x = i["levels"].get(lv)
            if x is None:
                L.append(f"| {sid} | | | | | | | | | {i.get('excluded')} |")
                continue
            L.append(f"| {sid} | {x['n_windows']} | {x.get('rejected_windows', 0)} | {x['n_cal']} | {x['n_test']} | "
                     f"{x['calib_minutes']:.2f} | {x['test_minutes']:.1f} | {x['n_valid_targets']} | "
                     f"{x.get('float16_overflow_samples', '')} | {x['excluded'] or ''} |")
        inc = [s for s, i in P.items() if i["levels"].get(lv, {}).get("excluded") is None and i["levels"].get(lv)]
        L.append(f"\nincluded: {len(inc)} patients")
    out = "\n".join(L)
    open(os.path.join(a.out, "p17_build.md"), "w", encoding="utf-8").write(out + "\n")
    print(out)


# ----------------------------------------------------------------------------- runs
def runs(a):
    subs = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(a.data, "sub-P*.pt"))
                  if not p.endswith("_normstats.pt"))
    for v in a.variants:
        src = os.path.join(a.enc_root, v, "enc_final.pt")
        assert os.path.exists(src), f"missing final encoder {src} (run s2i.shared --final first)"
        od = os.path.join(a.runs, v)
        os.makedirs(od, exist_ok=True)
        for s in subs:
            dst = os.path.join(od, f"enc_{s}.pt")
            if not os.path.exists(dst):
                shutil.copy(src, dst)
        log(f"{v}: encoder copied for {len(subs)} patients -> {od}")


# ----------------------------------------------------------------------------- report
def _load(run, name):
    out = {}
    for f in glob.glob(os.path.join(run, name, "readout_sub-P*.json")):
        d = json.load(open(f))
        out[d["subject"]] = d
    return out


def _ex(D, m, k="r"):
    return {s: r["explicit"][m]["mean"][k] for s, r in D.items() if m in r.get("explicit", {})
            and k in r["explicit"][m]["mean"]}


def compare(a_, b_, rng):
    from scipy.stats import wilcoxon
    s = sorted(set(a_) & set(b_))
    x, y = np.array([a_[q] for q in s]), np.array([b_[q] for q in s])
    d = x - y
    if len(d) == 0:
        return None
    bt = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    try:
        p = float(wilcoxon(x, y, alternative="two-sided").pvalue)
    except ValueError:
        p = float("nan")
    return {"n": len(d), "mean_a": float(x.mean()), "mean_b": float(y.mean()), "median_diff": float(np.median(d)),
            "mean_diff": float(d.mean()), "ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
            "wins": int((d > 0).sum()), "p": p, "per_patient": dict(zip(s, d.round(5).tolist()))}


def report(a):
    rng = np.random.default_rng(0)
    R = a.runs
    B = json.load(open(os.path.join(a.data, "p17_build.json")))["patients"]
    M = {}
    for lv in LEVELS:
        M[lv] = {"final": _load(os.path.join(R, "main"), f"final_{lv}"),
                 "learned3": _load(os.path.join(R, "no_env"), f"learned3_{lv}"),
                 "fixed3": _load(os.path.join(R, "no_env"), f"fixed3_{lv}")}
    fixed1 = _load(os.path.join(R, "no_env"), "fixed1_primary")
    f10 = _load(os.path.join(R, "main"), "final_primary_f10")
    P = M["primary"]
    included = sorted(s for s, i in B.items() if i["levels"].get("primary", {}).get("excluded") is None
                      and i["levels"].get("primary"))
    out, L = {}, ["# Phase 17: external validation on ScaleOLDNEW (protocol v1.6)", ""]
    have = {k: sorted(v) for k, v in P.items()}
    L.append(f"- included patients (primary level): {len(included)}: {', '.join(included)}")
    for k, v in have.items():
        if v != included:
            L.append(f"- **WARNING**: results for {k} exist for {len(v)} patients ({', '.join(v)}): run incomplete")
    # ---- confirmatory
    h1 = compare(_ex(P["final"], "combo_nosess"), _ex(P["final"], "base"), rng)
    h2 = compare(_ex(P["learned3"], "combo_nosess"), _ex(P["fixed3"], "combo_nosess"), rng)
    ps = [h1["p"], h2["p"]]
    order = np.argsort(ps)
    adj = [0.0, 0.0]
    adj[order[0]] = min(1.0, 2 * ps[order[0]])
    adj[order[1]] = max(adj[order[0]], min(1.0, ps[order[1]]))
    L += ["", "## Primary hypotheses (primary level, Pearson r; two-sided Wilcoxon, Holm over H1 and H2)", "",
          "| hypothesis | A | B | n | mean A | mean B | median A − B | mean A − B [95 % CI] | A > B | p | p (Holm) | supported |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, (la, lb), h, pa in (("H1 (system)", ("Final", "Base"), h1, adj[0]),
                                  ("H2 (learning)", ("Learned-3", "Fixed-3"), h2, adj[1])):
        sup = pa < 0.05 and h["median_diff"] > 0
        h["p_holm"], h["supported"] = pa, bool(sup)
        L.append(f"| {name} | {la} | {lb} | {h['n']} | {h['mean_a']:.4f} | {h['mean_b']:.4f} | {h['median_diff']:+.4f} | "
                 f"{h['mean_diff']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] | {h['wins']}/{h['n']} | {h['p']:.5f} | "
                 f"{pa:.5f} | {'**yes**' if sup else 'no'} |")
    out["H1"], out["H2"] = h1, h2
    L += ["", "Per-patient differences (A − B):", ""]
    for name, h in (("H1", h1), ("H2", h2)):
        L.append(f"- {name}: " + ", ".join(f"{s} {v:+.4f}" for s, v in h["per_patient"].items()))
    # ---- descriptive
    L += ["", "## Descriptive analyses (no confirmatory claim; unadjusted p)", "",
          "| level | comparison | metric | n | mean A | mean B | mean A − B [95 % CI] | median A − B | A > B | p |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    desc = []

    def row(lv, lab, A, B, metric):
        h = compare(A, B, rng)
        if h is None:
            return
        desc.append({"level": lv, "comparison": lab, "metric": metric, **h})
        L.append(f"| {lv} | {lab} | {metric} | {h['n']} | {h['mean_a']:.4f} | {h['mean_b']:.4f} | {h['mean_diff']:+.4f} "
                 f"[{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] | {h['median_diff']:+.4f} | {h['wins']}/{h['n']} | {h['p']:.4f} |")
    row("primary", "Final − Fixed-1", _ex(P["final"], "combo_nosess"), _ex(fixed1, "combo_nosess"), "r")
    for lv in ("secondary", "sensitivity"):
        row(lv, "H1: Final − Base", _ex(M[lv]["final"], "combo_nosess"), _ex(M[lv]["final"], "base"), "r")
        row(lv, "H2: Learned-3 − Fixed-3", _ex(M[lv]["learned3"], "combo_nosess"), _ex(M[lv]["fixed3"], "combo_nosess"), "r")
    for metric in ("r_event", "r_no50", "r2"):
        row("primary", "Final − Base", _ex(P["final"], "combo_nosess", metric), _ex(P["final"], "base", metric), metric)
        row("primary", "Learned-3 − Fixed-3", _ex(P["learned3"], "combo_nosess", metric),
            _ex(P["fixed3"], "combo_nosess", metric), metric)
    mins = lambda D, f: np.mean([r["fracs"][f]["minutes"] for r in D.values()]) if D else float("nan")   # noqa: E731
    row("primary", f"Final, 10 % ({mins(f10, '0.1'):.1f} min) − Base, 20 % ({mins(P['final'], '0.2'):.1f} min)",
        _ex(f10, "combo_nosess"), _ex(P["final"], "base"), "r")
    out["descriptive"] = desc
    L += ["", "Mean over patients, primary level:", "", "| model | r | R² | r_event | r_no50 | constant predictions |",
          "|---|---|---|---|---|---|"]
    for lab, D, m in (("Base", P["final"], "base"), ("Fixed-1", fixed1, "combo_nosess"),
                      ("Fixed-3", P["fixed3"], "combo_nosess"), ("Learned-3", P["learned3"], "combo_nosess"),
                      ("Final", P["final"], "combo_nosess"), ("Final, 10 % calibration", f10, "combo_nosess")):
        v = {k: _ex(D, m, k) for k in ("r", "r2", "r_event", "r_no50")}
        nc = sum(r["explicit"][m].get("n_const", 0) for r in D.values() if m in r.get("explicit", {}))
        L.append(f"| {lab} | " + " | ".join(f"{np.mean(list(x.values())):.4f}" if x else "" for x in v.values())
                 + f" | {nc} |")
    # ---- common mode (Section 8)
    cm = {}
    for f in glob.glob(os.path.join(R, "review13", "volts_sub-P*.json")):
        d = json.load(open(f))
        cm[d["subject"]] = d
    if cm:
        L += ["", "## Reference / common-mode analysis (descriptive; lagged ridge, primary level, volts)", "",
              "| patient | setup | scalp-cm vs iEEG-cm r | mono, orig | mono − cm, orig | bip, orig | bip, CAR | cm, orig | "
              "removal of iEEG cm | interaction |", "|---|---|---|---|---|---|---|---|---|---|"]
        grp = {}
        for s in sorted(cm):
            q, p_ = cm[s], cm[s]["pred"]
            rem = p_["mono|orig"] - p_["mono_minus_cm|orig"]
            inter = p_["mono|orig"] - p_["mono|car"] - p_["bip|orig"] + p_["bip|car"]
            setup = B[s]["setup"]
            grp.setdefault(setup, []).append((q["r_scalp_cm_vs_ieeg_cm"], p_["mono|orig"], rem, inter))
            L.append(f"| {s} | {setup} | {q['r_scalp_cm_vs_ieeg_cm']:+.2f} | {p_['mono|orig']:.3f} | "
                     f"{p_['mono_minus_cm|orig']:.3f} | {p_['bip|orig']:.3f} | {p_['bip|car']:.3f} | {p_['cm|orig']:.3f} | "
                     f"{-rem:+.3f} | {inter:+.3f} |")
        L += ["", "| setup | n | mean scalp-cm vs iEEG-cm r | mean abs r | mean mono, orig | mean removal of iEEG cm | "
              "mean interaction |", "|---|---|---|---|---|---|---|"]
        for g, v in sorted(grp.items()):
            v = np.array(v)
            L.append(f"| {g} | {len(v)} | {v[:, 0].mean():+.3f} | {np.abs(v[:, 0]).mean():.3f} | {v[:, 1].mean():.3f} | "
                     f"{-v[:, 2].mean():+.3f} | {v[:, 3].mean():+.3f} |")
        out["common_mode"] = {s: {"setup": B[s]["setup"], **cm[s]} for s in cm}
    md = "\n".join(L)
    print(md)
    open(os.path.join(R, "p17_report.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(os.path.join(R, "p17_report.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- post-hoc channel sensitivity
def set_channels(path):
    """Channel labels of an EEGLAB .set file (the original authors' preprocessed scalp data)."""
    from scipy.io import loadmat
    try:
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        E = m["EEG"] if "EEG" in m else None
        locs = E.chanlocs if E is not None else m["chanlocs"]
        return [str(c.labels).strip() for c in np.atleast_1d(locs)]
    except NotImplementedError:                         # MATLAB v7.3 (HDF5)
        import h5py
        with h5py.File(path, "r") as f:
            g = f["EEG"]["chanlocs"]["labels"] if "EEG" in f else f["chanlocs"]["labels"]
            return ["".join(chr(int(c)) for c in np.ravel(f[r[0]][()])) for r in g]


def setch(a):
    """
    Post-hoc sensitivity analysis (protocol v1.6, Addendum A). Scalp channels = channels kept under v1.6 that also
    appear in the original authors' EEGLAB scalp file (raw/SEEGxx/SEEGxx.set) (channels used by the authors; not proof of
    physical placement). CAR is
    recomputed on these channels; targets, windows, split and everything else are those of the primary level.
    -> <out>/primary_setch/sub-Pxx.pt (+ _normstats.pt) and <out>/p17_setch.md (channel comparison table).
    """
    import torch
    t1005 = template_1005()
    src = os.path.join(a.out, "primary")
    dst = os.path.join(a.out, "primary_setch")
    os.makedirs(dst, exist_ok=True)
    L = ["# Post-hoc channel sensitivity (Addendum A): scalp channels listed in the original authors' EEGLAB file", "",
         "| patient | ID | Table 1 | kept (v1.6) | authors' .set channels (10-05 names) | kept ∩ .set | kept, not in .set | "
         ".set, not kept | status |", "|---|---|---|---|---|---|---|---|---|"]
    info = {}
    for num, pid, setup, t1 in PATIENTS:
        sid = f"sub-P{num:02d}"
        fp = os.path.join(src, f"{sid}.pt")
        sp = os.path.join(a.sc_root, "raw", f"SEEG{pid}", f"SEEG{pid}.set")
        if not os.path.exists(fp):
            L.append(f"| {sid} | {pid} | {t1} | – | – | – | – | – | not in primary |")
            continue
        if not os.path.exists(sp):
            L.append(f"| {sid} | {pid} | {t1} | – | – | – | – | – | no .set file |")
            continue
        d = torch.load(fp, weights_only=False)
        kept = list(d["eeg_names"])
        raw_set = set_channels(sp)
        canon_set = []
        for c in raw_set:
            nm, _ = scalp_name(c, t1005)
            if nm is not None and nm not in canon_set:
                canon_set.append(nm)
        keep = [c for c in kept if c in canon_set]
        status = "ok" if len(keep) >= MIN_SCALP else f"excluded (< {MIN_SCALP} channels)"
        L.append(f"| {sid} | {pid} | {t1} | {len(kept)} | {len(raw_set)} ({len(canon_set)}) | {len(keep)} | "
                 f"{', '.join(c for c in kept if c not in canon_set) or '–'} | "
                 f"{', '.join(c for c in canon_set if c not in kept) or '–'} | {status} |")
        info[sid] = {"kept": kept, "set": raw_set, "used": keep}
        if len(keep) < MIN_SCALP:
            continue
        idx = [kept.index(c) for c in keep]
        st = d["norm_stats"]["eeg_orig"]["all"]
        z = d["eeg_orig"].float().numpy().astype(np.float64)
        v = z * np.asarray(st["mad"])[:, None] + np.asarray(st["med"])[:, None]          # back to microvolts
        v = v[idx]
        car = v - v.mean(0, keepdims=True)
        out = dict(d)
        out["eeg_names"] = keep
        out["norm_stats"] = dict(d["norm_stats"])
        for k, x in (("eeg_orig", v), ("eeg_car", car)):
            med, mad = robust(x)
            out[k] = torch.from_numpy(np.clip((x - med) / mad, -6e4, 6e4).astype(np.float16))
            out["norm_stats"][k] = {"all": {"med": med[:, 0].tolist(), "mad": mad[:, 0].tolist()}}
        out["channel_rule"] = "v1.6 kept ∩ authors' EEGLAB .set (Addendum A, post hoc)"
        torch.save(out, os.path.join(dst, f"{sid}.pt"))
        torch.save({k: out[k] for k in ("subject", "norm_stats", "norm_clip_z", "trial_bounds", "trial_session")},
                   os.path.join(dst, f"{sid}_normstats.pt"))
    md = "\n".join(L)
    print(md)
    open(os.path.join(a.out, "p17_setch.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(info, open(os.path.join(a.out, "p17_setch.json"), "w"), indent=1)


def report_setch(a):
    """H1 / H2 with the author-file channel sets (post hoc), next to the registered primary result."""
    rng = np.random.default_rng(0)
    R = a.runs
    L = ["# Post-hoc channel sensitivity (Addendum A): H1 / H2 with channels listed in the authors' scalp files", "",
         "Not part of the pre-registered analysis; p-values unadjusted. The registered result (v1.6 channel rules) is "
         "shown for comparison.", "",
         "| analysis | comparison | n | mean A | mean B | median A − B | mean A − B [95 % CI] | A > B | p |",
         "|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for tag, lab in (("primary", "registered (v1.6 channels)"), ("primary_setch", "post hoc (authors' file channels)")):
        F = _load(os.path.join(R, "main"), f"final_{tag}")
        L3 = _load(os.path.join(R, "no_env"), f"learned3_{tag}")
        F3 = _load(os.path.join(R, "no_env"), f"fixed3_{tag}")
        for name, A, B in (("H1: Final − Base", _ex(F, "combo_nosess"), _ex(F, "base")),
                           ("H2: Learned-3 − Fixed-3", _ex(L3, "combo_nosess"), _ex(F3, "combo_nosess"))):
            h = compare(A, B, rng)
            if h is None:
                continue
            out[f"{tag}|{name}"] = h
            L.append(f"| {lab} | {name} | {h['n']} | {h['mean_a']:.4f} | {h['mean_b']:.4f} | {h['median_diff']:+.4f} | "
                     f"{h['mean_diff']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] | {h['wins']}/{h['n']} | {h['p']:.5f} |")
    for name in ("H1: Final − Base", "H2: Learned-3 − Fixed-3"):
        a_, b_ = out.get(f"primary|{name}"), out.get(f"primary_setch|{name}")
        if a_ and b_:
            L.append("")
            L.append(f"- {name}, per patient (registered → post hoc): " + ", ".join(
                f"{s} {a_['per_patient'][s]:+.4f} → {b_['per_patient'][s]:+.4f}" for s in b_["per_patient"]
                if s in a_["per_patient"]))
    md = "\n".join(L)
    print(md)
    open(os.path.join(R, "p17_setch_report.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(os.path.join(R, "p17_setch_report.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "runs", "report", "setch", "report_setch"))
    ap.add_argument("--sc_root", help="ScaleOLDNEW root of Phase 14 (proc/, raw/)")
    ap.add_argument("--out", help="build: output root (one folder per level)")
    ap.add_argument("--data", help="runs: primary data folder; report: build output root")
    ap.add_argument("--enc_root", help="runs: folder with <variant>/enc_final.pt")
    ap.add_argument("--runs", help="runs / report: Phase-17 run root")
    ap.add_argument("--variants", nargs="*", default=["main", "seed1", "seed2", "short", "no_env"])
    ap.add_argument("--patients", nargs="*", default=None)
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    {"build": build, "runs": runs, "report": report, "setch": setch, "report_setch": report_setch}[a.cmd](a)


if __name__ == "__main__":
    main()
