"""
Filter log for the manuscript (Section 2.1, Supplementary Table S2): the MNE design messages of the exact calls made by
s2i.data.preprocess_ds004752.filt_resample, and the measured impulse responses of each step and of the whole chain.

For each input rate (4096 Hz = ScaleOLDNEW, 4000 / 2000 Hz = ds004752 SEEG, 200 Hz = ds004752 scalp EEG of patients
1-9) a unit impulse in the middle of a 60-s signal is passed through
  notch      Raw.notch_filter(NOTCH < Nyquist - 1)      (as in filt_resample)
  band-pass  Raw.filter(0.5, min(70, Nyquist - 1))
  chain      filt_resample itself (notch -> band-pass -> resampling to 200 Hz)
and the support where |h| > 1e-6 of the peak, and the energy beyond +-3.3 s, are reported.

Usage
-----
python -m s2i.filtlog --out OUT.md
"""
import argparse
import logging
import sys

import numpy as np

RATES = (4096.0, 4000.0, 2000.0, 200.0)
THR = 1e-6


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, rec):
        self.lines.append(rec.getMessage())


def extent(h, fs):
    a = np.abs(h) / np.abs(h).max()
    c = int(np.argmax(np.abs(h)))
    idx = np.where(a > THR)[0]
    t = (np.arange(len(h)) - c) / fs
    e = h ** 2 / (h ** 2).sum()
    sym = float(np.max(np.abs(h[c - 50:c + 51] - h[c - 50:c + 51][::-1])) / np.abs(h).max())
    return t[idx.min()], t[idx.max()], float(e[np.abs(t) > 3.3].sum()), sym


def sidecar_rates(bids):
    """{subject: (scalp EEG Hz, SEEG Hz)} from *_eeg.json / *_ieeg.json (SamplingFrequency); sets of rates if mixed."""
    import glob
    import json
    import os
    out = {}
    for sd in sorted(glob.glob(os.path.join(bids, "sub-*"))):
        rates = {}
        for mod in ("eeg", "ieeg"):
            fs = set()
            for f in glob.glob(os.path.join(sd, "ses-*", mod, f"*_{mod}.json")):
                v = json.load(open(f)).get("SamplingFrequency")
                if v is not None:
                    fs.add(float(v))
            rates[mod] = fs.pop() if len(fs) == 1 else (None if not fs else sorted(fs))
        out[os.path.basename(sd)] = (rates["eeg"] if not isinstance(rates["eeg"], list) else None,
                                     rates["ieeg"] if not isinstance(rates["ieeg"], list) else None)
        if any(isinstance(v, list) for v in rates.values()):
            print(f"{os.path.basename(sd)}: several rates across sessions {rates}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bids", default=None, help="BIDS root with the *_eeg.json / *_ieeg.json sidecars (optional)")
    ap.add_argument("--extra_rates", type=float, nargs="*", default=[],
                    help="further input rates to measure (rates found in the sidecars are added automatically)")
    a = ap.parse_args()
    if a.bids:
        a.extra_rates = list(a.extra_rates) + [r for fe_fi in sidecar_rates(a.bids).values() for r in fe_fi if r]
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    import mne
    from s2i.data.preprocess_ds004752 import BAND, FS, NOTCH, filt_resample
    log = logging.getLogger("mne")
    L = [f"# Filter log (MNE {mne.__version__}; the calls of filt_resample)", "",
         f"Support = where |h| > {THR:g} of the peak; energy = share of h² beyond ±3.3 s; asymmetry = max |h(t) − h(−t)| "
         f"near the peak / peak.", "",
         "| input rate | step | output rate | support (s) | energy beyond ±3.3 s | asymmetry |", "|---|---|---|---|---|---|"]
    design = []
    for sf in RATES:
        n = int(sf * 60)
        x = np.zeros((1, n))
        x[0, n // 2] = 1.0
        nyq = sf / 2
        notch = [f for f in NOTCH if f < nyq - 1]
        steps = []
        for name in ("notch", "band-pass", "chain"):
            raw = mne.io.RawArray(x.copy(), mne.create_info(["imp"], sf, "eeg"), verbose=False)
            h = _Collect()
            log.addHandler(h)
            old = log.level
            log.setLevel(logging.INFO)
            try:
                if name == "notch":
                    if not notch:
                        steps.append((name, None, sf, ["not applied (100 Hz and 50 Hz handled as in filt_resample)"]))
                        continue
                    y = raw.notch_filter(notch, picks="all", verbose=True).get_data()[0]
                    fs_out = sf
                elif name == "band-pass":
                    y = raw.filter(BAND[0], min(BAND[1], nyq - 1.0), picks="all", verbose=True).get_data()[0]
                    fs_out = sf
                else:
                    y = filt_resample(raw)[0]
                    fs_out = FS
            finally:
                log.removeHandler(h)
                log.setLevel(old)
            steps.append((name, y, fs_out, h.lines))
        for name, y, fs_out, lines in steps:
            if y is None:
                L.append(f"| {sf:g} Hz | {name} | – | not applied (notch {list(NOTCH)} Hz ≥ Nyquist − 1) | – | – |")
                continue
            lo, hi, e, sym = extent(y, fs_out)
            L.append(f"| {sf:g} Hz | {name} | {fs_out:g} Hz | {lo:+.2f} / {hi:+.2f} | {e:.1e} | {sym:.1e} |")
            design.append((sf, name, [ln for ln in lines if ln.strip()]))
    # ---- timing: delay of a 10 Hz sinusoid through the whole chain, per input rate
    delays = {}
    for sf in RATES + tuple(sorted(set(a.extra_rates) - set(RATES))):
        n = int(sf * 60)
        t = np.arange(n) / sf
        z = filt_resample(mne.io.RawArray(np.sin(2 * np.pi * 10 * t)[None], mne.create_info(["s"], sf, "eeg"),
                                          verbose=False))[0]
        to = np.arange(len(z)) / FS
        mid = slice(len(z) // 4, 3 * len(z) // 4)
        ph = np.arctan2(z[mid] @ np.cos(2 * np.pi * 10 * to[mid]), z[mid] @ np.sin(2 * np.pi * 10 * to[mid]))
        delays[sf] = float(-1e3 * ph / (2 * np.pi * 10))
    L += ["", "## Timing of the chain output for a 60-s test signal only (10 Hz sinusoid; negative = output leads the input)",
          "", "The shift of MNE's FFT resampling depends on the exact record length; these values hold for 60-s "
          "signals only. Per-recording values: `python -m s2i.resample_check`.", "",
          "| input rate | delay (ms) |", "|---|---|"]
    L += [f"| {sf:g} Hz | {d:+.3f} |" for sf, d in delays.items()]
    if a.bids:
        L += ["", "## Per patient: sampling rates (BIDS sidecars); offsets below assume 60-s records "
              "(NOT the real recordings; see s2i.resample_check)", "",
              "| patient | scalp EEG Hz | SEEG Hz | scalp delay (ms) | SEEG delay (ms) | SEEG − scalp (ms) |",
              "|---|---|---|---|---|---|"]
        for sub, (fe, fi) in sidecar_rates(a.bids).items():
            de = delays.get(fe) if fe else None
            di = delays.get(fi) if fi else None
            off = f"{di - de:+.3f}" if de is not None and di is not None else "rate not measured"
            L.append(f"| {sub} | {fe or '?'} | {fi or '?'} | {'' if de is None else f'{de:+.3f}'} | "
                     f"{'' if di is None else f'{di:+.3f}'} | {off} |")
    L += ["", "## MNE design messages", ""]
    for sf, name, lines in design:
        L.append(f"**{sf:g} Hz, {name}**")
        L += ["```"] + lines + ["```", ""]
    out = "\n".join(L)
    print(out)
    open(a.out, "w", encoding="utf-8").write(out + "\n")


if __name__ == "__main__":
    main()
