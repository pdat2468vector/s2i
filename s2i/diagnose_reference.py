"""
Common-reference contamination test  (run this BEFORE trusting any number).

Hypothesis: in the old monopolar setup, scalp EEG and every iEEG contact were
recorded against a shared reference R:
    EEG_c  = e_c - R           iEEG_k = s_k - R
so -R is present in every input AND every target. Predicting R is trivial and
says nothing about intracranial sources. Signatures in the old logs:
  * bipolar Hipp/Amyg, WITHIN-subject (easiest setting)   -> mean r = 0.14
  * monopolar, CROSS-subject (harder setting)             -> sub-13 r = 0.78
  * "observable" sets contain 45-67 of 64-68 contacts incl. hippocampus,
    and the same patients are "good" in both GIN and ds004752 (06, 09, 13, 15)
    -> subject (recording-setup) dependent, not anatomy dependent.

This script fits the same lagged ridge on 4 input/target combinations:
                    target=mono     target=bipolar
  input=orig ref        A                 B
  input=CAR             C                 D
If A >> B and A >> C, the monopolar "reconstruction" is mostly the shared
reference. D is the honest number for intracranial reconstruction.

python -m s2i.diagnose_reference --data c:\\iEEG_EEG\\data\\s2i_v1
"""
import argparse
import os
import sys

import numpy as np
import torch

from s2i.baselines import LaggedRidge
from s2i.data.dataset import chrono_split
from s2i.metrics import per_contact_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--calib_frac", type=float, default=0.5)
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for f in sorted(x for x in os.listdir(a.data) if x.endswith(".pt") and "_normstats" not in x):
        d = torch.load(os.path.join(a.data, f), weights_only=False)
        b = d["trial_bounds"].numpy()
        if len(b) < 10:
            continue
        tr, te = chrono_split(len(b), a.calib_frac, gap=1)

        def seg(key, idx):
            return [d[key][:, s0:s1].float().numpy() for s0, s1 in b[idx]]

        res = {}
        for ein in ("eeg_orig", "eeg_car"):
            for tgt in ("ieeg_mono", "ieeg_bip"):
                if d[tgt].shape[0] == 0:
                    res[(ein, tgt)] = np.nan
                    continue
                rg = LaggedRidge().fit(seg(ein, tr), seg(tgt, tr))
                m = per_contact_metrics(rg.predict(seg(ein, te)), seg(tgt, te), bands=False)
                res[(ein, tgt)] = float(np.mean(m["r"]))
        dg = d.get("diagnostics", {})
        rows.append((d["subject"], res, dg))
        print(f"{d['subject']}: A(orig->mono)={res[('eeg_orig', 'ieeg_mono')]:.3f} "
              f"B(orig->bip)={res[('eeg_orig', 'ieeg_bip')]:.3f} "
              f"C(car->mono)={res[('eeg_car', 'ieeg_mono')]:.3f} "
              f"D(car->bip)={res[('eeg_car', 'ieeg_bip')]:.3f} | "
              f"mono PC1={dg.get('mono_pc1_ratio', np.nan):.2f} "
              f"|corr(mean iEEG, EEG)|={dg.get('mean_abs_corr_iEEGmean_vs_EEG', np.nan):.2f}")

    def col(key):
        return np.nanmean([r[1][key] for r in rows])

    A, B = col(("eeg_orig", "ieeg_mono")), col(("eeg_orig", "ieeg_bip"))
    C, D = col(("eeg_car", "ieeg_mono")), col(("eeg_car", "ieeg_bip"))
    print("\nMEAN  A={:.3f}  B={:.3f}  C={:.3f}  D={:.3f}".format(A, B, C, D))
    if A > 1.5 * max(B, 1e-3):
        print("=> Strong evidence of shared-reference contamination in monopolar targets.\n"
              "   Report bipolar (or locally re-referenced) results as the primary outcome.")
    else:
        print("=> No strong contamination signature; monopolar results are less suspect.")


if __name__ == "__main__":
    main()
