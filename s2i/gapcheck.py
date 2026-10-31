"""
Guard gap between calibration and test (read-only check for the manuscript, Section 2.1).

The preprocessing chain (notch -> band-pass -> resampling) has an effective impulse response of at most +-3.36 s
(|h| > 1e-6 of the peak; measured on the executed function). Encoder and lagged features are computed within each
trial / window. Calibration and test are therefore separated if, in recording time,
  > 3.36 s  no filtered test sample depends on a raw calibration sample, and vice versa;
  > 6.72 s  no raw sample influences both a calibration and a test sample.
For every patient this reports the separation between the end of the last calibration trial and the start of the
first test trial at 20 % calibration (the smallest separation: smaller calibration fractions keep the same test set
and end calibration earlier).

  ds004752   exact separation from the BIDS event files (onset, duration of the trials, matched by session and
             nTrial) if --bids is given; otherwise the duration of the skipped trial(s), which is a lower bound when
             trials do not overlap in time. Calibration and test in different sessions = separate recordings.
  ScaleOLDNEW (--scales) exact: windows are consecutive within a block; different blocks = separate files.

Usage
-----
python -m s2i.gapcheck --data DATA_DIR [DATA_DIR ...] [--bids BIDS_DIR] [--scales SCALES_PRIMARY_DIR] --out OUT.md
"""
import argparse
import glob
import os
import sys


import torch

FS = 200
IR = 3.36                                # max effective half-support of the chain over the input rates (filtlog)


def n_cal(n):
    return max(3, int(round(n * 0.2)))


def bids_events(bids, sub):
    import pandas as pd
    out = {}
    for ses_dir in sorted(glob.glob(os.path.join(bids, sub, "ses-*"))):
        ses = os.path.basename(ses_dir)
        f = glob.glob(os.path.join(ses_dir, "ieeg", "*events.tsv")) or glob.glob(os.path.join(ses_dir, "eeg", "*events.tsv"))
        if not f:
            continue
        ev = pd.read_csv(f[0], sep="\t")
        if "nTrial" not in ev:
            continue
        for _, r in ev.iterrows():
            if pd.notna(r["nTrial"]):
                out[(ses, int(r["nTrial"]))] = (float(r["onset"]), float(r["duration"]))
    return out


def ds_rows(data, bids):
    rows = []
    for p in sorted(glob.glob(os.path.join(data, "sub-[0-9][0-9].pt"))):
        sub = os.path.basename(p)[:-3]
        d = torch.load(p, weights_only=False)
        b = d["trial_bounds"].numpy()
        ses = list(d.get("trial_session", ["ses-01"] * len(b)))
        tid = list(d.get("trial_id", [None] * len(b)))
        n = len(b)
        nc = n_cal(n)
        last, first = nc - 1, nc + 1
        if first >= n:
            continue
        skipped = (b[last + 1:first, 1] - b[last + 1:first, 0]).sum() / FS
        same = ses[last] == ses[first]
        exact, how = None, "lower bound (skipped trial duration)"
        if bids and same and tid[last] is not None:
            ev = bids_events(bids, sub)
            a, c = ev.get((ses[last], int(tid[last]))), ev.get((ses[first], int(tid[first])))
            if a and c:
                exact, how = c[0] - (a[0] + a[1]), "exact (BIDS events)"
        if not same:
            sep, how = None, "different sessions (separate recordings)"
        else:
            sep = exact if exact is not None else skipped
        rows.append({"patient": sub, "trials": n, "calib": nc, "same_session": same,
                     "gap_trial_s": skipped, "separation_s": sep, "how": how})
    return rows


def scales_rows(data):
    rows = []
    for p in sorted(glob.glob(os.path.join(data, "sub-P*.pt"))):
        if p.endswith("_normstats.pt"):
            continue
        d = torch.load(p, weights_only=False)
        blk, st = list(d["window_block"]), list(d["window_start"])
        b = d["trial_bounds"].numpy()
        n = len(b)
        nc = n_cal(n)
        last, first = nc - 1, nc + 1
        same = blk[last] == blk[first]
        sep = (st[first] - (st[last] + b[last, 1] - b[last, 0])) / FS if same else None
        rows.append({"patient": os.path.basename(p)[:-3], "trials": n, "calib": nc, "same_session": same,
                     "gap_trial_s": (b[nc, 1] - b[nc, 0]) / FS, "separation_s": sep,
                     "how": "exact (window positions)" if same else "different blocks (separate files)"})
    return rows


def table(title, rows):
    L = [f"## {title}", "", "| patient | trials / windows | calibration | same session / block | skipped trial (s) | "
         "separation (s) | > 3.36 s | > 6.72 s | how |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["separation_s"]
        ok1 = "yes" if (s is None or s > IR) else "**NO**"
        ok2 = "yes" if (s is None or s > 2 * IR) else "**NO**"
        L.append(f"| {r['patient']} | {r['trials']} | {r['calib']} | {'yes' if r['same_session'] else 'no'} | "
                 f"{r['gap_trial_s']:.2f} | {'separate recordings' if s is None else f'{s:.2f}'} | {ok1} | {ok2} | {r['how']} |")
    same = [r["separation_s"] for r in rows if r["separation_s"] is not None]
    lb = sum(r["how"].startswith("lower bound") for r in rows)
    if same:
        L += ["", f"- minimum separation within one recording: {min(same):.2f} s ({len(same)} patients"
              + (f"; {lb} of them lower bounds from the skipped trial" if lb else "") + "); "
              f"{len(rows) - len(same)} patients have calibration and test in different recordings"]
    else:
        L += ["", "- calibration and test lie in different recordings for every patient"]
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[], help="ds004752 data folders (e.g. s2i_v1_noqc s2i_v1 s2i_qc_calib)")
    ap.add_argument("--bids", default=None)
    ap.add_argument("--scales", default=None, help="ScaleOLDNEW primary data folder (sub-P*.pt)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    L = ["# Guard gap between calibration and test (20 % calibration)", "",
         f"Effective impulse response of the preprocessing chain: +-{IR} s (|h| > 1e-6 of peak).", ""]
    for dd in a.data:
        L += table(f"ds004752: {os.path.basename(dd.rstrip('/'))}", ds_rows(dd, a.bids if a.bids and os.path.isdir(a.bids) else None))
    if a.scales:
        L += table("ScaleOLDNEW: " + os.path.basename(a.scales.rstrip("/")), scales_rows(a.scales))
    out = "\n".join(L)
    print(out)
    open(a.out, "w", encoding="utf-8").write(out + "\n")


if __name__ == "__main__":
    main()
