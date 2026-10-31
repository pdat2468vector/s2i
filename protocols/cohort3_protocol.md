# Cohort-3 protocol: generalisation to a stimulation paradigm (Parmigiani et al. 2022)

Version 1.0, written 21 Oct 2026, before any model was applied to this cohort. Not signed and not deposited in a public registry (an OSF registration is planned; until then the date, this file and its revision history are the record).

## 1. Purpose

The two cohorts of the main analysis (ds004752, ScaleOLDNEW) are memory tasks recorded during wakefulness. This protocol tests whether the two main findings hold in a different paradigm, site and montage: simultaneous high-density scalp EEG and SEEG during single-pulse electrical stimulation. It is a test of generalisation, not a replication: the epochs are 1 s long, so only the ±200-ms ("short") encoder of the main analysis can be used.

## 2. Data and what was inspected before this protocol

Dataset: Parmigiani et al. 2022, *Brain Stimulation* 15:664–675, OSF project wsgzp (BIDS derivative `ccepcoreg`, licence CC BY-NC-SA 4.0). 36 patients, 323 stimulation runs. Each run has EEG and SEEG epochs from −0.3 to +0.7 s around each pulse (1001 samples at 1000 Hz). EEG: 185 channels, average reference, 0.5–45 Hz, ICA for blinks. SEEG: bipolar, 0.5–300 Hz. The dataset contains no continuous recordings.

Before this protocol was written, only the Phase 26 inventory (`s2i/parm26.py`; `runs/parm26/inventory.md`) was inspected. It covers:
- epoch counts and the EEG–SEEG pairing;
- channel lists, filters and coordinates;
- the stimulation description of each run;
- the RMS profile around the pulse;
- the share of post-stimulus variance that is time-locked.

No model, no prediction and no correlation between scalp EEG and SEEG was computed. The inventory showed:
- consistent pairing in 278 of 323 runs;
- a pulse artefact already attenuated in the shared data (RMS around the pulse below the pre-stimulus RMS in both modalities);
- a time-locked share of 7 % (SEEG) and 18 % (EEG) of the post-stimulus variance;
- 0.6–8.6 min of paired epochs per patient.

## 3. Runs, pairing and inclusion

- **Pairing.** EEG epochs were rejected separately from SEEG epochs; `additional_info/eeg_badtrials.json` flags, per run, the SEEG epochs whose EEG counterpart was removed. A run is used only if the number of flags equals the number of SEEG epochs and the number of zero flags equals the number of EEG epochs; the k-th zero flag then pairs a SEEG epoch with the k-th EEG epoch. Runs failing this check are excluded and listed.
- **Order.** Runs carry no acquisition time, so they are ordered by run number. Epochs keep their order within a run.
- **Inclusion.** A patient is included if, after these rules, there are ≥ 100 paired epochs, ≥ 3 EEG channels and ≥ 3 valid targets (non-zero variance in calibration and in test). From the inventory, 31 patients are expected to meet the epoch rule (excluded: 06, 09, 13, 16, 19); the executed counts are reported, whatever they are.

## 4. Signals

- **EEG:** channels with status "good" in every used run of the patient; common average reference over these channels.
- **SEEG targets:** the bipolar channels as shared. A channel is kept if its status is "good" in every used run and it shares no contact with the stimulated pair of any used run.
- **Resampling:** every epoch is resampled from 1000 to 200 Hz (polyphase, factor 1/5); units are microvolts.
- No further filtering or artefact rejection. The pulse region is not removed, because the inventory shows no residual pulse artefact.
- **Normalisation:** strict, as in the main analysis. Robust statistics (median, MAD) come from the calibration epochs only.

## 5. Split

Epochs in run order. Calibration is the first round(0.3 N) epochs, then one epoch is skipped, and the rest form the test set. Test runs therefore include stimulation sites that were never in the calibration data.

30 % is used instead of the 20 % of the other cohorts because the epochs are short. With 20 %, the median calibration would be about 47 s, below the shortest calibration evaluated in the main analysis (1.4 min).

All hyper-parameters (ridge penalties, mixture weight, amplitude gains) are chosen by blocked 5-fold cross-validation within the calibration epochs, as in the main analysis.

## 6. Models

All encoders are fixed. The short encoder is the final encoder trained on ds004752 for the ScaleOLDNEW validation (13 patients for optimisation, 2 for early stopping). No data from this cohort or from ScaleOLDNEW were used for training, architecture choice, hyper-parameters or checkpoint selection.

- **Base:** LAG9 lagged ridge (lags 0, ±25, ±50, ±100, ±200 ms), penalty by calibration cross-validation.
- **Fixed-short:** the short encoder at its initialisation, ridge readout, mixed with LAG9 (weight by cross-validation).
- **Learned-short:** the trained short encoder, ridge readout, mixed with LAG9.
- **Final-short:** identical to Learned-short; the name is used for the system comparison.

Code: `s2i/parm27.py`, then `s2i.readout --strict --only_combo --test_frac 0.3 --fracs 0.3 --explicit --explicit_frac 0.3 --induced_by_session`; the untrained bank uses `--untrained`.

## 7. Metrics

- **r:** Pearson correlation per target over the concatenated test epochs (explicit predictions; a constant prediction scores 0), averaged over targets, then over patients.
- **r_induced_ses:** r after subtracting, from each side, its own average over the test epochs of the same run. This is the single-trial part, without the response time-locked to the stimulation of that site.

## 8. Hypotheses

Unit of inference: the patient.

- **H1:** Final-short > Base, on r.
- **H2:** Learned-short > Fixed-short, on r.
- **H3:** Final-short > Base, on r_induced_ses.

Each hypothesis uses a two-sided Wilcoxon signed-rank test on patient means, with Holm correction over H1–H3. A hypothesis is supported if its adjusted p < 0.05 and the median paired difference is positive. Means are reported with a patient-bootstrap 95 % CI (10,000 resamples).

## 9. Descriptive analyses (no confirmatory claim; unadjusted)

- r_induced (average over all test epochs), r_no50 and R² for Final-short − Base.
- r_induced_ses for Learned-short − Fixed-short.
- Fixed-short − Base, and the learned encoder alone − Base.
- 20 % calibration on the same test epochs (Final-short and Base).
- Per-patient table.

Any further analysis (montage density 185 → 64 → 19 channels, dense-lag ridge, source imaging with the stimulation sites as ground truth) will be defined in a dated addendum before it is run.

## 10. Reporting

All included patients are reported. Results are reported whatever their direction, together with the excluded runs, channels and patients.

## Revision history

- v1.0 (21 Oct 2026): first version, written after the Phase 26 inventory and before any model was applied.
- Addendum A (23 Oct 2026): montage density, written before it was run.

## Addendum A (23 Oct 2026): montage density (post hoc, descriptive)

Written after the H1–H3 results and before the analysis it defines was run. It answers review 6 (Section 3.2, item 4) and tests the "sparse montage" interpretation of the Discussion. No confirmatory claim; p-values are unadjusted.

- **Densities.** 64, 32, 19, and 8 scalp channels per patient, in addition to all good channels of the main analysis (115–176).
- **Channel selection.** Farthest-point sampling on the 3-D electrode positions of the patient's good EEG channels (`*_electrodes.tsv`), starting from the channel closest to the centroid of these positions, so that each subset covers the scalp as evenly as possible. Channels without a position are not eligible. The selection uses positions only, never a signal.
- **Signals.** The stored EEG (CAR over all good channels) is de-normalised with its stored statistics and re-referenced to the average of the selected channels, which equals CAR computed over the subset alone. The targets are unchanged.
- **Models and evaluation.** Unchanged from Sections 5–7: the same short encoder, Base, Fixed-short, and Final-short, strict normalisation, 30 % calibration, one-epoch gap, explicit predictions, and r.
- **Reported.** For each density: the mean r of Base, Fixed-short, and Final-short; Final-short − Base and Learned-short − Fixed-short (mean [patient-bootstrap 95 % CI], patients improved, two-sided Wilcoxon p); and Final-short at each density − Final-short with all channels.
- **Expected direction, stated before running.** If sparse montages limit the information available, r decreases with fewer channels. Whether the gain over Base changes with density is open.
