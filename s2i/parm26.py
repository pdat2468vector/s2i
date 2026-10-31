"""
Phase 26: download and inventory of the third cohort (Parmigiani et al. 2022, Brain Stimulation 15:664-675;
OSF wsgzp, CC BY-NC-SA 4.0). No model is fitted here: this phase only establishes what the data contain, so that the
protocol for this cohort can be written before any model is applied.

Dataset facts known before this phase (read from the archive's central directory and small files, 21 Oct 2026):
  - BIDS derivative "epochs" only (no continuous recordings): per run, EEG and SEEG epochs of 1001 samples at 1000 Hz
    (-0.3 ... +0.7 s around each single-pulse stimulation), float64, baseline-corrected (mean of -0.3 ... 0 s);
  - EEG: 185 channels, average reference, 0.5-45 Hz (3rd-order Butterworth), ICA for blinks;
    SEEG: bipolar, 0.5-300 Hz;
  - EEG epochs were rejected separately: additional_info/eeg_badtrials.json flags, per run, the SEEG epochs whose
    EEG counterpart was removed (1 = removed), which defines the EEG-SEEG pairing.

  python -m s2i.parm26 download  --dst /content/parm
  python -m s2i.parm26 extract   --dst /content/parm
  python -m s2i.parm26 inventory --dst /content/parm --out <runs>/parm26
"""
import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

PARTS = [("ccepcoreg-bids.z01", 4294967296, "https://osf.io/download/7h5pj/"),
         ("ccepcoreg-bids.z02", 4294967296, "https://osf.io/download/cktnz/"),
         ("ccepcoreg-bids.z03", 4294967296, "https://osf.io/download/9g7kw/"),
         ("ccepcoreg-bids.z04", 4294967296, "https://osf.io/download/d4g9j/"),
         ("ccepcoreg-bids.z05", 4294967296, "https://osf.io/download/ghtb2/"),
         ("ccepcoreg-bids.zip", 2698060289, "https://osf.io/download/agw23/")]
BADTRIALS = ("eeg_badtrials.json", 37419, "https://osf.io/download/4qn3s/")
FS, T0 = 1000, 300                       # sampling rate; index of the stimulation pulse (t = 0) in each epoch
WINDOWS = [(-300, -10), (-10, -2), (-2, 0), (0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 300),
           (300, 701)]                   # ms relative to the pulse


# ----------------------------------------------------------------------------- download / extract
def download(a):
    os.makedirs(a.dst, exist_ok=True)
    for name, size, url in PARTS + [BADTRIALS]:
        f = os.path.join(a.dst, name)
        if os.path.exists(f) and os.path.getsize(f) == size:
            print("ok (present)", name)
            continue
        for attempt in range(5):
            subprocess.run(["curl", "-L", "-sS", "--retry", "5", "-C", "-", "-o", f, url])
            if os.path.exists(f) and os.path.getsize(f) == size:
                break
            print(f"  {name}: {os.path.getsize(f) if os.path.exists(f) else 0} of {size} bytes, retrying")
        if not (os.path.exists(f) and os.path.getsize(f) == size):
            sys.exit(f"download of {name} failed")
        print("ok", name, size)


def extract(a):
    root = os.path.join(a.dst, "ccepcoreg")
    if os.path.isdir(os.path.join(root, "derivatives", "epochs")):
        print("already extracted:", root)
        return
    z = os.path.join(a.dst, "ccepcoreg-bids.zip")
    r = subprocess.run(["7z", "x", "-y", f"-o{a.dst}", z], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isdir(os.path.join(root, "derivatives", "epochs")):
        print("7z could not read the split archive; joining it with zip -s 0 instead\n", r.stdout[-2000:], r.stderr[-2000:])
        joined = os.path.join(a.dst, "joined.zip")
        subprocess.run(["zip", "-s", "0", z, "--out", joined], check=True)
        for name, _, _ in PARTS:
            os.remove(os.path.join(a.dst, name))
        subprocess.run(["unzip", "-q", "-o", joined, "-d", a.dst], check=True)
        os.remove(joined)
    n = len(glob.glob(os.path.join(root, "derivatives", "epochs", "sub-*", "*", "*_epochs.npy")))
    print("extracted", n, "epoch files to", root)
    if a.delete_parts:
        for name, _, _ in PARTS:
            f = os.path.join(a.dst, name)
            if os.path.exists(f):
                os.remove(f)


# ----------------------------------------------------------------------------- reading
def read_tsv(f):
    lines = open(f, encoding="utf-8").read().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]


def runs_of(root):
    """[(subject, run, eeg_npy, ieeg_npy)] for every run that has both modalities."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, "derivatives", "epochs", "sub-*", "ieeg", "*_epochs.npy"))):
        m = re.search(r"(sub-\d+)_task-ccepcoreg_(run-\d+)_epochs\.npy$", f)
        e = f.replace(os.sep + "ieeg" + os.sep, os.sep + "eeg" + os.sep)
        out.append((m.group(1), m.group(2), e if os.path.exists(e) else None, f))
    return out


def stim_pair(desc):
    """'Stimulation of I'6-7 1ma ...' -> ("I'", 6, 7)."""
    m = re.search(r"Stimulation of\s+(\S+?)(\d+)-(\d+)\s", desc)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


def near_stim(ch, sp):
    """SEEG bipolar channel 'I'5-6' shares a contact with the stimulated pair, or is the pair itself."""
    m = re.match(r"(\S+?)(\d+)-(\d+)$", ch)
    if not (m and sp) or m.group(1) != sp[0]:
        return False
    return bool({int(m.group(2)), int(m.group(3))} & {sp[1], sp[2]})


def win_rms(x, w):
    a, b = T0 + w[0], T0 + w[1]
    return np.sqrt(np.mean(x[..., a:b] ** 2, axis=-1))          # (epochs, channels)


# ----------------------------------------------------------------------------- inventory
def inventory(a):
    root = os.path.join(a.dst, "ccepcoreg")
    bad = json.load(open(os.path.join(a.dst, BADTRIALS[0])))
    os.makedirs(a.out, exist_ok=True)
    rows, prof = [], collections.defaultdict(list)
    for s, r, fe, fi in runs_of(root):
        base = fi[:-len("_epochs.npy")]
        js = json.load(open(base + "_epochs.json"))
        ch_i = read_tsv(base + "_channels.tsv")
        ch_e = read_tsv(fe[:-len("_epochs.npy")] + "_channels.tsv") if fe else []
        Xi = np.load(fi, mmap_mode="r")
        Xe = np.load(fe, mmap_mode="r") if fe else None
        flags = bad.get(s, {}).get(r)
        row = dict(sub=s, run=r, desc=js.get("Description", ""), shape_ieeg=list(Xi.shape),
                   shape_eeg=list(Xe.shape) if Xe is not None else None, n_flags=len(flags) if flags else None,
                   n_kept_flags=(len(flags) - int(sum(flags))) if flags else None,
                   ch_ieeg_listed=len(ch_i), ch_eeg_listed=len(ch_e),
                   bad_ieeg=sum(c.get("status") == "bad" for c in ch_i), bad_eeg=sum(c.get("status") == "bad" for c in ch_e),
                   units=sorted({c.get("units") for c in ch_i + ch_e}),
                   cut_ieeg=sorted({(c.get("low_cutoff"), c.get("high_cutoff")) for c in ch_i}),
                   cut_eeg=sorted({(c.get("low_cutoff"), c.get("high_cutoff")) for c in ch_e}),
                   ref_eeg=sorted({c.get("reference") for c in ch_e}))
        row["paired"] = (Xe is not None and flags is not None and len(flags) == Xi.shape[0]
                         and row["n_kept_flags"] == Xe.shape[0] and Xi.shape[2] == Xe.shape[2] == 1001
                         and len(ch_i) == Xi.shape[1] and len(ch_e) == Xe.shape[1])
        sp = stim_pair(row["desc"])
        names = [c["name"] for c in ch_i]
        row["stim_pair"] = f"{sp[0]}{sp[1]}-{sp[2]}" if sp else None
        row["stim_in_seeg"] = bool(sp) and f"{sp[0]}{sp[1]}-{sp[2]}" in names
        row["n_seeg_near_stim"] = sum(near_stim(n, sp) for n in names)
        m = re.search(r"(\d+(?:\.\d+)?)hz", row["desc"])
        row["stim_hz"] = float(m.group(1)) if m else None
        if row["paired"]:
            keep = np.array(flags) == 0
            xi = np.asarray(Xi[keep], dtype=np.float64) * 1e6          # microvolts
            xe = np.asarray(Xe, dtype=np.float64) * 1e6
            good_i = np.array([c.get("status") != "bad" for c in ch_i])
            good_e = np.array([c.get("status") != "bad" for c in ch_e])
            near = np.array([near_stim(n, sp) for n in names])
            groups = {"eeg": xe[:, good_e], "seeg_far": xi[:, good_i & ~near], "seeg_near": xi[:, good_i & near]}
            row["baseline_mean_over_sd"] = {k: float(np.median(np.abs(v[..., :T0].mean(-1)) /
                                                               (v[..., :T0].std(-1) + 1e-12))) if v.shape[1] else None
                                            for k, v in groups.items()}
            row["baseline_sd_uv"] = {k: float(np.median(v[..., 10:290].std(-1))) if v.shape[1] else None
                                     for k, v in groups.items()}
            for k, v in groups.items():
                if not v.shape[1]:
                    continue
                b = win_rms(v, WINDOWS[0])
                prof[k].append([float(np.median(win_rms(v, w) / (b + 1e-12))) for w in WINDOWS])
            # share of post-stimulus variance (10-700 ms) that is time-locked: variance of the run average / total
            for k in ("eeg", "seeg_far"):
                v = groups[k][..., T0 + 10:]
                if v.shape[1] and v.shape[0] > 1:
                    ev = v.mean(0).var(-1)
                    tot = v.var(-1).mean(0)
                    row[f"evoked_share_{k}"] = float(np.median(ev / (tot + 1e-12)))
            # timing of the largest EEG deflection around the pulse (artefact alignment)
            pk = np.abs(groups["eeg"][..., T0 - 10:T0 + 20]).mean(0).mean(0).argmax() - 10
            row["eeg_peak_ms"] = int(pk)
            row["n_pairs"] = int(keep.sum())
        rows.append(row)
        print(s, r, "paired" if row["paired"] else "NOT PAIRED", row["shape_ieeg"], row["shape_eeg"], row["stim_pair"],
              row.get("n_pairs"), flush=True)
    json.dump(rows, open(os.path.join(a.out, "runs.json"), "w"), indent=1)

    # ---- coordinates and participants
    coords = {}
    for s in sorted({r["sub"] for r in rows}):
        d = os.path.join(root, "derivatives", "epochs", s)
        mni = glob.glob(os.path.join(d, "ieeg", "*space-MNI152*_electrodes.tsv"))
        eeg_el = glob.glob(os.path.join(d, "eeg", "*_electrodes.tsv"))
        coords[s] = dict(ieeg_mni=len(read_tsv(mni[0])) if mni else 0, eeg_electrodes=len(read_tsv(eeg_el[0])) if eeg_el else 0)
    scans = sorted(glob.glob(os.path.join(root, "derivatives", "epochs", "sub-*", "*_scans.tsv")))
    scan_cols = sorted({k for f in scans for k in (read_tsv(f)[0].keys() if read_tsv(f) else [])})

    # ---- report
    L = ["# Phase 26: inventory of the Parmigiani et al. 2022 dataset (no model fitted)", ""]
    P = [r for r in rows if r["paired"]]
    subs = sorted({r["sub"] for r in rows})
    L += [f"Runs with both modalities: {len(rows)}; EEG-SEEG pairing consistent: {len(P)} "
          f"(flags length = SEEG epochs, kept flags = EEG epochs, 1001 samples, channel lists match the arrays). "
          f"Subjects: {len(subs)}.", ""]
    bad_runs = [r for r in rows if not r["paired"]]
    if bad_runs:
        L += ["Runs that could not be paired:", ""] + [f"- {r['sub']} {r['run']}: SEEG {r['shape_ieeg']}, EEG "
                                                        f"{r['shape_eeg']}, flags {r['n_flags']}/{r['n_kept_flags']}"
                                                        for r in bad_runs] + [""]
    L += ["| subject | runs | paired epochs | paired minutes | SEEG ch (min-max) | EEG ch (min-max) | stim Hz | "
          "SEEG MNI rows | EEG electrodes |", "|" + "---|" * 9]
    for s in subs:
        R = [r for r in P if r["sub"] == s]
        if not R:
            L.append(f"| {s} | 0 | – | – | – | – | – | {coords[s]['ieeg_mni']} | {coords[s]['eeg_electrodes']} |")
            continue
        n = sum(r["n_pairs"] for r in R)
        ci = [r["shape_ieeg"][1] - r["bad_ieeg"] for r in R]
        ce = [r["shape_eeg"][1] - r["bad_eeg"] for r in R]
        hz = sorted({r["stim_hz"] for r in R})
        L.append(f"| {s} | {len(R)} | {n} | {n / 60:.1f} | {min(ci)}-{max(ci)} | {min(ce)}-{max(ce)} | {hz} | "
                 f"{coords[s]['ieeg_mni']} | {coords[s]['eeg_electrodes']} |")
    tot = [sum(r["n_pairs"] for r in P if r["sub"] == s) for s in subs]
    L += ["", f"Paired epochs per subject: median {int(np.median(tot))}, range {min(tot)}-{max(tot)}; total {sum(tot)} "
              f"({sum(tot) / 60:.0f} min of 1-s epochs).", ""]
    L += ["**Filters / reference / units** (from channels.tsv): SEEG cut-offs "
          f"{sorted({tuple(x) for r in rows for x in r['cut_ieeg']})}, EEG cut-offs "
          f"{sorted({tuple(x) for r in rows for x in r['cut_eeg']})}, EEG reference {sorted({x for r in rows for x in r['ref_eeg']})}, "
          f"units {sorted({x for r in rows for x in r['units']})}.", ""]
    L += [f"**Stimulated pair among the SEEG channels:** {sum(r['stim_in_seeg'] for r in rows)} of {len(rows)} runs; "
          f"SEEG channels sharing a contact with the stimulated pair per run: median "
          f"{int(np.median([r['n_seeg_near_stim'] for r in rows]))}, max {max(r['n_seeg_near_stim'] for r in rows)}.", ""]
    L += [f"**Run order:** scans.tsv columns {scan_cols} (no acquisition time means the run number is the only order "
          "available).", ""]
    for k in ("eeg", "seeg_far", "seeg_near"):
        bm = [r["baseline_mean_over_sd"][k] for r in P if r["baseline_mean_over_sd"].get(k) is not None]
        sd = [r["baseline_sd_uv"][k] for r in P if r["baseline_sd_uv"].get(k) is not None]
        if bm:
            L.append(f"- {k}: |pre-stimulus mean| / SD median {np.median(bm):.3f} (0 = baseline-corrected); "
                     f"pre-stimulus SD median {np.median(sd):.1f} µV (runs: {len(bm)})")
    L += ["", "**Artefact profile**: median over runs of the RMS in each window relative to the pre-stimulus RMS "
              "(-300 to -10 ms).", "", "| window (ms) | " + " | ".join(prof) + " |", "|" + "---|" * (len(prof) + 1)]
    for j, w in enumerate(WINDOWS):
        L.append(f"| {w[0]} to {w[1]} | " + " | ".join(f"{np.median([p[j] for p in prof[k]]):.2f}" for k in prof) + " |")
    for k in ("eeg", "seeg_far"):
        v = [r[f"evoked_share_{k}"] for r in P if f"evoked_share_{k}" in r]
        if v:
            L.append(f"\nTime-locked share of post-stimulus variance (10-700 ms; variance of the run average / mean "
                     f"single-epoch variance), {k}: median {np.median(v):.3f}, IQR {np.percentile(v, 25):.3f}-"
                     f"{np.percentile(v, 75):.3f}.")
    pk = collections.Counter(r["eeg_peak_ms"] for r in P)
    L.append(f"\nLatency of the largest mean |EEG| deflection between -10 and +20 ms: {dict(sorted(pk.items()))}.")
    open(os.path.join(a.out, "inventory.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("download", "extract", "inventory"))
    ap.add_argument("--dst", required=True, help="local working folder (Colab disk), about 55 GB")
    ap.add_argument("--out", help="inventory: output folder")
    ap.add_argument("--delete_parts", action="store_true", help="extract: delete the archive parts afterwards")
    a = ap.parse_args()
    {"download": download, "extract": extract, "inventory": inventory}[a.cmd](a)


if __name__ == "__main__":
    main()
