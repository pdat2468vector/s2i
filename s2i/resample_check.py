"""
Timing of the executed resampling for every ds004752 recording (read-only check for the manuscript, Section 2.1).

MNE's FFT resampling (Raw.resample, default npad="auto") does not keep the nominal time axis exactly: the output is
shifted by a few milliseconds, by an amount that depends on the exact number of input samples, and the shift drifts
slowly along the record. Scalp EEG and SEEG of ds004752 are separate EDF files (scalp EEG of patients 1-9 already at
200 Hz, not resampled), so the two modalities can be offset relative to each other by a session-specific amount.

For every session this script
  1. reads only the EDF headers from OpenNeuro (HTTP range requests, a few kB per file): sampling rate and the exact
     number of samples (records x samples per record, as read by MNE);
  2. measures, for that exact (rate, length), the delay of a 10 Hz sinusoid after Raw.resample(200) at 10 %, 50 % and
     90 % of the record (filtering is zero-phase and adds no delay; resampling alone is measured);
  3. reports the SEEG - scalp offset at the three positions.
Negative delay = the output leads the nominal time axis.

Usage
-----
python -m s2i.resample_check --out OUT.md [--subjects sub-01 ...]
"""
import argparse
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

BUCKET = "https://s3.amazonaws.com/openneuro.org"
PREFIX = "ds004752/"
FS = 200
POS = (0.10, 0.50, 0.90)


def list_edf():
    keys, token = [], None
    while True:
        url = f"{BUCKET}?list-type=2&prefix={PREFIX}" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        root = ET.fromstring(urllib.request.urlopen(url, timeout=120).read())
        ns = {"s": root.tag.split("}")[0].strip("{")}
        for c in root.findall("s:Contents", ns):
            k = c.find("s:Key", ns).text
            if k.endswith(".edf") and "/derivatives/" not in k:
                keys.append((k, int(c.find("s:Size", ns).text)))
        if root.find("s:IsTruncated", ns).text != "true":
            return keys
        token = root.find("s:NextContinuationToken", ns).text


def get_range(key, a, b):
    req = urllib.request.Request(f"{BUCKET}/{key}", headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req, timeout=120).read()


def edf_info(key, size):
    h = get_range(key, 0, 255)
    ns = int(h[252:256].decode("ascii").strip())
    full = get_range(key, 0, 256 * (ns + 1) - 1)
    n_rec = int(full[236:244].decode("ascii").strip())
    dur = float(full[244:252].decode("ascii").strip())
    off = 256 + ns * 216
    spr = [int(full[off + 8 * i:off + 8 * (i + 1)].decode("ascii").strip()) for i in range(ns)]
    if n_rec < 0:                                       # unknown record count: from the file size (int16 samples)
        n_rec = (size - 256 * (ns + 1)) // (2 * sum(spr))
    smax = max(spr)
    return smax / dur, n_rec * smax, sorted(set(spr))


_cache = {}


def delays(sf, n):
    """Delay (ms) of a 10 Hz sinusoid after Raw.resample(200) at the positions POS; 0 if no resampling."""
    if abs(sf - FS) < 1e-9:
        return (0.0,) * len(POS)
    key = (sf, n)
    if key not in _cache:
        import mne
        mne.set_log_level("ERROR")
        t = np.arange(n) / sf
        z = mne.io.RawArray(np.sin(2 * np.pi * 10 * t)[None], mne.create_info(["s"], sf, "eeg")).resample(FS).get_data()[0]
        to = np.arange(len(z)) / FS
        out = []
        for p in POS:
            sl = slice(int((p - 0.02) * len(z)), int((p + 0.02) * len(z)))
            ph = np.arctan2(z[sl] @ np.cos(2 * np.pi * 10 * to[sl]), z[sl] @ np.sin(2 * np.pi * 10 * to[sl]))
            out.append(float(-1e3 * ph / (2 * np.pi * 10)))
        _cache[key] = tuple(out)
    return _cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    sess = {}
    for key, size in list_edf():
        m = re.search(r"(sub-\d+)/(ses-\d+)/(eeg|ieeg)/", key)
        if not m or (a.subjects and m.group(1) not in a.subjects):
            continue
        sess.setdefault((m.group(1), m.group(2)), {})[m.group(3)] = (key, size)
    L = ["# Timing of the executed resampling, per ds004752 session",
         "", "Delay of a 10 Hz sinusoid after Raw.resample(200) for the exact rate and length of each EDF file, at 10 / 50 / "
         "90 % of the record (ms; negative = leads the nominal time axis). Offset = SEEG delay − scalp delay.", "",
         "| patient | session | scalp Hz | scalp samples | scalp delay | SEEG Hz | SEEG samples | SEEG delay | SEEG − scalp |",
         "|---|---|---|---|---|---|---|---|---|"]
    offs = []
    for (sub, ses), mods in sorted(sess.items()):
        if "eeg" not in mods or "ieeg" not in mods:
            continue
        fe, ne, _ = edf_info(*mods["eeg"])
        fi, ni, _ = edf_info(*mods["ieeg"])
        de, di = delays(fe, ne), delays(fi, ni)
        off = [x - y for x, y in zip(di, de)]
        offs.append((sub, off))
        f3 = lambda v: " / ".join(f"{x:+.2f}" for x in v)                       # noqa: E731
        L.append(f"| {sub} | {ses} | {fe:g} | {ne} | {f3(de)} | {fi:g} | {ni} | {f3(di)} | {f3(off)} |")
        print(L[-1], flush=True)
    allv = np.array([x for _, o in offs for x in o])
    if len(allv):
        L += ["", f"- SEEG − scalp offset over all sessions and positions: {allv.min():+.2f} to {allv.max():+.2f} ms "
              f"(median |offset| {np.median(np.abs(allv)):.2f} ms, {len(offs)} sessions)"]
        for sub in sorted({s for s, _ in offs}):
            v = np.array([x for s, o in offs if s == sub for x in o])
            L.append(f"  - {sub}: {v.min():+.2f} to {v.max():+.2f} ms")
    out = "\n".join(L)
    print(out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(out + "\n")


if __name__ == "__main__":
    main()
