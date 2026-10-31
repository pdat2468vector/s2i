# S2I: scalp EEG to bipolar SEEG prediction

Code, notebooks, and analysis protocols for the manuscript *Scalp-to-bipolar SEEG prediction under deployment-oriented evaluation: protocol sensitivity and multi-cohort validation*.

The study predicts bipolar stereo-EEG (SEEG) signals from simultaneously recorded scalp EEG. A temporal filter bank shared across patients is meta-trained through a closed-form ridge readout, and a ridge decoder is fitted to the calibration data of each new patient. Evaluation uses bipolar targets, common-average-referenced scalp EEG, chronological calibration on the first 20 % of trials, and normalisation statistics from the calibration data only.

## Contents

| Folder | Content |
|---|---|
| `s2i/` | Python package with every model, baseline, readout, metric, and analysis script. |
| `notebooks/` | Google Colab notebooks, one per analysis phase, in the order they were run. The notebook instructions are in Vietnamese. Code cells and printed results are in English. |
| `protocols/` | The analysis protocols, unchanged from the versions used. |

## Data

No patient data are included. The three cohorts are public:

| Cohort | Source | Licence |
|---|---|---|
| ds004752 (15 patients, Zurich, verbal working memory) | OpenNeuro, doi:10.18112/openneuro.ds004752.v1.0.1 | as stated on OpenNeuro |
| ScaleOLDNEW (12 patients, Bucharest, visual recognition memory) | http://epi.fizica.unibuc.ro/scalesoldnew/ | CC BY |
| Parmigiani et al. 2022 (31 patients analysed, Milan, single-pulse stimulation) | Open Science Framework, doi:10.17605/OSF.IO/WSGZP | CC BY-NC-SA 4.0 |

## Running the analyses

The analyses were run on Google Colab with a T4 GPU. Every notebook mounts Google Drive and uses `MyDrive/iEEG_S2I/` as its root, with the code in `code/s2i/`, the data in `data/`, and the results in `runs/`. Each notebook first writes the `s2i` modules it needs to Drive (`%%writefile` cells), so the copy in `s2i/` and the copy in the notebooks are the same code.

1. `LeadFormer_Colab.ipynb` downloads ds004752 from the public OpenNeuro bucket and preprocesses it (`s2i/data/preprocess_ds004752.py`). Its later cells train the LeadFormer baseline.
2. `S2I_Phase14_ScalesDownload_Colab.ipynb` downloads and preprocesses ScaleOLDNEW (`s2i/data/scales_fetch.py`).
3. `S2I_Phase26_ThirdCohort_Inventory_Colab.ipynb` downloads the third cohort from OSF and builds its inventory (`s2i/parm26.py`).
4. The other notebooks run the analyses, in phase order.

Dependencies are listed in `requirements.txt`. Colab provides most of them, and the notebooks install the rest (`mne`, `nilearn`, `statsmodels`).

## Where the main results come from

| Result | Notebook | Module |
|---|---|---|
| Main analysis on ds004752, primary level (no additional automatic QC) | Phase 16 | `s2i/shared.py`, `s2i/readout.py`, `s2i/review11.py` |
| Fixed and learned filter banks, five-encoder model | Phase 9 | `s2i/shared.py` |
| Protocol sensitivity | Phases 5 and 8 | `s2i/protocol.py` |
| External validation on ScaleOLDNEW (Table 5) | Phase 17 | `s2i/scales17.py` |
| Addenda A, B, and C of the external-validation protocol, and the scalp-channel audit | Phases 19, 20, 23, and 25 | `s2i/scales17.py`, `s2i/chanqc20.py`, `s2i/leak23.py`, `s2i/dense24.py` |
| Dense-lag ridge controls | Phases 24 and 25 | `s2i/dense24.py` |
| Third cohort (Table 6) | Phase 27 | `s2i/parm27.py` |
| Statistical sensitivity analyses (Table S14) and recorded and predicted traces (Fig. 6) | Phase 28 | `s2i/sens28.py` |
| EEG source imaging baseline (Table S15) | Phase 29 | `s2i/srcimg29.py` |
| Working-memory load effects (Table S16) | Phase 30 | `s2i/load30.py` |
| Training-objective ablation (Table S17a, b) | Phase 31 | `s2i/abl31.py` |
| Per-patient fine-tuning (Table S17c) | Phase 35 | `s2i/ft35.py` |
| Montage density (Table S18) | Phase 33 | `s2i/mont33.py` |
| Leave-one-cohort-out training and training-set size (Table S19) | Phase 34 | `s2i/loco34.py` |
| Deep baselines at the primary level with strict normalisation (Table S20) | Phase 36 | `s2i/strict36.py`, `s2i/train_loso.py`, `s2i/nfn_baseline.py` |
| CAST re-run under the present protocol, with positive control (Table S21) | Phase 37 | `s2i/cast37.py` |
| Figures | Phases 22, 32, and 33 | `s2i/figs11.py`, `s2i/paperfigs.py`, `s2i/render3d.py` |

## Protocols

- `protocols/external_validation_protocol.md`: the protocol for ScaleOLDNEW (version 1.6, 11 Oct 2026), with Addenda A–C and its revision history. It was written and fixed before any model was applied to ScaleOLDNEW. Its heading says "pre-registered", but it was not deposited in a public registry before the analysis. The manuscript therefore calls it pre-specified.
- `protocols/cohort3_protocol.md`: the protocol for the third cohort (version 1.0, 21 Oct 2026) and its Addendum A (montage density, 23 Oct 2026).
- `protocols/review6_experiments_plan.md`: the plan of the post-hoc analyses added during the revision, each section written before its analysis was run.

The dates of these files are those written in them. This repository was created after the analyses, so its commit history does not date them.

## Licence

The code is released under the MIT licence (`LICENSE`). The fonts in `s2i/fonts/` (Federo, Josefin Sans) are distributed under the SIL Open Font License 1.1.

## Citation

[Citation and DOI to be added.]
