"""
10-20 scalp channel canonicalisation.

Old pipeline bug: EEG channels were zero-padded *by index* to 21. Subjects have
different montages (6, 8, 17, 18, 19, 21 ch) and different orderings
(e.g. sub-05 starts with Fp1, sub-01 with F3), so input slot i meant a different
electrode for each subject -> the "universal" encoder could not be universal.

Here every scalp channel is identified by its canonical 10-20 name and a 3D
position, and the model treats channels as a *set* of tokens with positional
embeddings, so any montage works.
"""
import re

# Old 10-20 nomenclature -> modern (MNE standard_1020 uses modern names)
_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Non-EEG / reference channels to drop
NON_EEG_TOKENS = ("EKG", "ECG", "EMG", "EOG", "A1", "A2", "M1", "M2",
                  "STATUS", "TRIG", "PHOTIC", "RESP", "SPO2", "PULSE")


def canonical_eeg_name(raw_name: str):
    """'EEG F3-Ref' -> 'F3', 'T3' -> 'T7', 'FP1' -> 'Fp1'. Returns None if not EEG."""
    n = raw_name.strip()
    n = re.sub(r"^(EEG|eeg)[\s_:-]*", "", n)
    n = re.split(r"[-\s/]", n)[0]          # drop reference suffix
    if not n:
        return None
    up = n.upper()
    if any(tok == up or up.startswith(tok) for tok in NON_EEG_TOKENS):
        return None
    # restore canonical casing: Fp1, Fz, Cz, AF3, FC1 ...
    n = "Fp" + up[2:] if up.startswith("FP") else up
    if n.endswith("Z"):
        n = n[:-1] + "z"
    return _ALIASES.get(n, n)


def standard_positions_mm():
    """Canonical-name -> (x, y, z) in mm (MNE standard_1020, head frame).

    Head frame is close to (not identical to) MNI; the model learns a relative
    geometry bias, so a constant offset between frames is absorbed in training.
    """
    import mne
    mont = mne.channels.make_standard_montage("standard_1020")
    pos = mont.get_positions()["ch_pos"]
    out = {}
    for name, xyz in pos.items():
        c = canonical_eeg_name(name)
        if c is not None:
            out[c] = [float(v) * 1000.0 for v in xyz]
    return out
