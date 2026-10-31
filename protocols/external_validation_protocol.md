# External validation protocol: ScaleOLDNEW (pre-registered)

*Version 1.6, 11 Oct 2026. No model has been applied to ScaleOLDNEW. So far the raw data were downloaded and filtered (Phase 14), and only channel lists, block lengths, marker counts and the original publication were inspected.*

*Final wording edits before sign-off (no change to data, models, criteria or tests):*
- Section 5 no longer says that every sample is included; it says which parts of a block are eligible and that the exclusion rules decide which parts are used;
- Section 3 separates the ridge penalty learned during encoder training on ds004752 from the readout penalty chosen by 5-fold CV for each ScaleOLDNEW patient, and states the readout grids;
- Section 3 states that Fixed-3 and Fixed-1 keep frozen encoders but fit their readouts like the other models.

*Changes from v1.5 (algorithmic consistency only):*
- the cross-validation folds used for every calibration-based choice are fixed;
- the patient inclusion rule is written as a minimum number of calibration windows (≥ 5), so that 5 folds always exist.

*Changes from v1.4 (fifth internal review, wording and execution details only):*
- the encoder-training wording now states the 13/2 split exactly;
- the edge-margin statement is limited to what the impulse-response test shows, and the 200 Hz measurement is explained;
- it is stated that encoder and lagged features are computed separately within each window, with zero padding;
- non-finite values in individual channels are handled by a fixed technical rule.

*Changes from v1.3 (fourth internal review, before any model run):*
- the impulse response of the whole preprocessing chain is measured;
- windows now start and end at least 3.4 s from the block edges, so that no evaluated sample depends on edge padding;
- the full filter-design parameters are given;
- r = 0 for constant predictions is described as a scoring convention.

*Changes from v1.2 (third internal review, before any model run):*
- target validity no longer depends on any model output; a constant prediction scores r = 0;
- the frozen weights of Fixed-3 are specified;
- the full filter configuration is given;
- the direction of the effect is tied to the median, consistent with the Wilcoxon test.

*Changes from v1.1 (second internal review, before any model run):*
- the description of the ID 99 header is corrected (280 channel entries; 38 scalp electrodes after the channel rules, matching Table 1 of the publication);
- a pre-specified consistency check against Table 1 is added;
- the input representation and the CAR computation are fixed;
- the window boundary rules are completed;
- valid targets are defined by technical criteria only;
- the bootstrap, effect size and scope of the Holm correction are fixed;
- the three branches of Learned-3 and Fixed-3 are named.

*Changes from v1.0, made after an internal review and before any model run:*
- the amplifier grouping now uses the publication's patient table, not header channel counts (the v1.0 rule was wrong);
- the 8-s windows are defined as signal segments, with explicit inclusion and boundary rules;
- the statistical unit and aggregation are fixed;
- H2 is now a learning test matched for architecture and ensemble size;
- CAR is fixed as the primary scalp reference;
- the checkpoint rule is fixed.

*After sign-off this document is locked. Any later deviation will be reported as a deviation, with its reason, and kept separate from the pre-registered analysis.*

## 1. Aim and type of validation

Test, on an independent cohort, the paper's main claim: under a deployment-faithful evaluation, the final model predicts bipolar SEEG from scalp EEG better than the strongest linear baseline.

This is an **external validation with pre-specified patient-specific calibration**, not a zero-shot test:
- ScaleOLDNEW is not used to train encoders, to choose architectures, encoder variants, hyper-parameters or checkpoints, or to select models.
- For each patient, only the pre-specified calibration windows (Section 5) are used, to fit that patient's readout as defined in Section 3.

## 2. Data

**Source.** ScaleOLDNEW (Barborica et al. 2023, *Front. Hum. Neurosci.* 17:1154038, CC BY; http://epi.fizica.unibuc.ro/scalesoldnew/).

**Cohort and recordings.**
- 12 patients with drug-resistant epilepsy, visual recognition-memory task.
- Per patient: 7 blocks, each one continuous raw recording (AnyWave ADES file), with scalp EEG and SEEG in the same file at 4096 Hz.

**Independence from ds004752.** The site, task, amplifier (Natus Quantum) and patients all differ.

## 3. Locked model

**Encoders.** Five encoders (main, seed 1, seed 2, short, no-env) are developed with the 15-patient ds004752 cohort, with everything else as in the paper (Section 2.4):
- **Data:** for each variant, 13 patients contribute to parameter optimisation, and 2 patients, drawn with the variant's seed as in the paper's leave-one-patient-out procedure, are used only for early stopping and checkpoint selection. No refitting on the validation patients is performed.
- **Hyper-parameters:** exactly as in the paper.
- **Seeds:** the same as for each variant in the paper.
- **Checkpoint rule:** validation every 150 steps, patience 6 validations, at most 3000 steps; the checkpoint with the best validation r is kept.

All five are trained once, before any ScaleOLDNEW data are read by a model.

**Readout.** The paper's final-model recipe, unchanged:
- per-patient ridge on each encoder's features, with a single penalty chosen by cross-validated r on calibration data;
- predictions averaged over encoders;
- a mixture with the lagged-ridge (LAG9) prediction, weight w ∈ {0, 0.25, 0.5, 0.75, 1} chosen by cross-validated r on calibration data;
- a per-contact amplitude gain fitted on out-of-fold calibration predictions.

**Calibration folds.** Every calibration-based choice uses the **same 5 folds**:
- the ridge penalties of the encoder readouts and of LAG9, for all models including Base;
- the mixture weight;
- the out-of-fold predictions for the amplitude gain.

The folds are **5 contiguous, non-shuffled blocks** of the calibration windows in time order, with sizes as equal as possible: boundaries at round(i·n_cal/5), i = 0…5. This is the ds004752 code's rule K = min(5, n_cal). With the inclusion rule of Section 5 (n_cal ≥ 5), K is always 5. No other number of folds is used.

**Models compared.** All use the same inputs, windows, calibration and test sets.

| Name | Definition |
|---|---|
| **Final** | 5 learned encoders + LAG9 mixture |
| **Base** | lagged ridge (LAG9), penalty chosen on the same 5 calibration folds |
| **Learned-3** | three learned encoders + LAG9 mixture: main (support ±1.2 s, with envelopes), short (±0.2 s, with envelopes), no-env (±1.2 s, no envelopes), each trained as above |
| **Fixed-3** | the same three architectures, branch by branch (main ↔ main, short ↔ short, no-env ↔ no-env), at their initialisation (below), with no weight updated by any data + LAG9 mixture |
| **Fixed-1** | the untrained no-env bank + LAG9 mixture |

**Initialisation shared by Learned-3 and Fixed-3.** In every architecture the filter bank is initialised deterministically, independently of the training seed:
- 12 quadrature pairs of complex Morlet wavelets, 5 cycles, centre frequencies log-spaced from 2 to 60 Hz;
- 1 identity kernel (broadband, lag 0);
- 3 random kernels, Gaussian-windowed with σ = 50 ms, drawn with a fixed generator seed (0);
- all kernels unit-norm, truncated to the architecture's support (±1.2 s or ±0.2 s);
- in the architectures with envelopes, the 12 envelopes are the magnitudes of the quadrature pairs and have no parameters.

The learned encoders start from exactly these weights and are trained as in Section 3. During this training on ds004752, all kernels (including the identity and random kernels) are updated, together with the penalty of the inner closed-form ridge used for meta-training. In Fixed-3 every kernel stays at its initial value. Fixed-3 is therefore a fixed bank of Morlet pairs plus one identity and three fixed random kernels, not a pure Morlet bank.

**Two different ridge penalties.**
- **Training penalty.** The penalty learned during encoder training is used only inside meta-training on ds004752. It is not used on ScaleOLDNEW.
- **Readout penalty.** For every ScaleOLDNEW patient and every model, the readout penalty is chosen on the 5 calibration folds (Section 3, *Calibration folds*) from a fixed grid:
  - encoder readouts: {10⁻³, 10⁻², …, 10⁵};
  - LAG9: {10⁻³, 10⁻², …, 10³}.

**Same readout for every model.** Fixed-3 and Fixed-1 keep their encoder weights frozen. Their readouts are fitted to each patient's calibration data exactly like those of Final and Learned-3: ridge penalty, LAG9 mixture weight and amplitude gain. The models differ only in the encoder weights.

## 4. Signals

**Preprocessing.** Done in Phase 14 with the ds004752 functions and order, on each block at its native 4096 Hz:
1. notch 50/100 Hz;
2. band-pass 0.5–70 Hz (MNE FIR, `phase="zero"`, `filter_length="auto"`); the 70 Hz low-pass also prevents aliasing at the new rate (Nyquist 100 Hz);
3. resampling to **200 Hz** with MNE's default method.

Signals are kept in µV. No ICA or other artefact removal or spatial processing.

**Filter configuration** (MNE 1.13.2 defaults, as for ds004752). Each block (ADES file) is filtered as one continuous signal, separately from the other blocks.

| Step | Setting |
|---|---|
| Notch | `Raw.notch_filter([50, 100])`: FIR, `fir_design="firwin"`, `fir_window="hamming"`, `phase="zero"`, `filter_length="auto"`, `notch_widths` = f/200 (0.25 and 0.5 Hz), transition bandwidth 1 Hz (0.5 Hz each side), default edge padding (`pad="reflect_limited"`) |
| Band-pass | `Raw.filter(0.5, 70)`: one FIR, `fir_design="firwin"`, `fir_window="hamming"`, `phase="zero"`, `filter_length="auto"`, default edge padding. High-pass 0.5 Hz with a 0.5 Hz transition band (filter length 6.6 s); low-pass 70 Hz with a 17.5 Hz transition band (stop-band edge 87.5 Hz, below the 100 Hz Nyquist frequency of the output) |
| Resampling | `Raw.resample(200)` with the MNE 1.13.2 defaults (FFT method); the exact parameter values are printed in the Phase-17 log |
| Impulse response of the **whole chain** | Measured by passing an impulse through the exact preprocessing function. **The test relevant to ScaleOLDNEW is 4096 Hz in → 200 Hz out** (notch 50 and 100 Hz → band-pass → resampling): the output is symmetric, \|h\| exceeds 10⁻⁶ of its peak only within ±3.35 s, and the energy beyond ±3.3 s is < 10⁻⁹ of the total. The same holds for 2000 Hz in (the ds004752 SEEG rate). A separate test with 200 Hz in covers the ds004752 scalp EEG of patients 1–9: there the function applies only the 50 Hz notch (100 Hz is at the Nyquist frequency and is skipped) and does not resample |

**What these choices imply.**
- Zero-phase filtering uses samples on both sides in time. This, and the non-causal encoders, make the evaluation an **offline/buffered** one (latency up to about 4.5 s, as for ds004752), not a demonstration of causal online operation.
- The pipeline differs on purpose from the ERP pipeline of the original ScaleOLDNEW publication (256 Hz, 0.5–45 Hz, CAR, ICA). This is an **external validation with a pre-specified pipeline** that matches the encoders' training data. It is not a reproduction of the original preprocessing.

**Encoder input representation** (identical to the encoders' training data):
- 200 Hz;
- one scalp channel at a time, after CAR and strict robust normalisation.

The encoders and the lagged ridge act on each scalp channel separately and the readout is fitted per patient, so no channel needs to be mapped to a training montage. No model in H1–H2 uses electrode positions.

**Scalp channels.** A channel is kept if all of the following hold:
1. It is labelled EEG in the ADES header.
2. Its name, after the dataset's lookup table (T3→T7, T4→T8, T5→P7, T6→P8, FB1/FB2→FT9/FT10, TP1/TP2→TP9/TP10), is a position of the MNE 10-05 template.
3. It is not A1, A2, M1 or M2.
4. It is not listed as bad in any block of the patient.
5. Its standard deviation is non-zero in every block.

This excludes unused amplifier inputs (e.g. C41…C128), EOG (LOC1, ROC2), OSAT, PR, Pleth, AF11 and AF12.

**Consistency check (reported, not used to change the rules).** Headers list the full cap template, including positions where no electrode was placed. For every patient, the number of kept scalp channels is therefore reported next to the number of scalp electrodes in Table 1 of the publication. For example, the ID 99 header has 40 named 10-20/10-10 entries; 38 remain after rule 2, and Table 1 lists 38. Positions never placed are expected to be removed by rules 4–5 (bad or flat).

**Scalp reference.**
- **Primary: common average (CAR).** At every sample, the mean of the kept scalp channels (in µV, after filtering and resampling) is subtracted, before any normalisation. This is the same order as in the ds004752 pipeline.
- The original recording reference (Fpz for patients 1–8) is used only in the reference analysis (Section 8).

**Targets.**
- **Bipolar (primary):** the pairs of the dataset's *BipolarChannels* sheet whose two contacts are recorded and not bad in any block, computed as contact A − contact B.
- **Monopolar:** the contacts of those pairs against the common intracranial reference. Used only in Section 8.

**Coordinates.** MNI coordinates and structure labels come from the *SEEGContacts* and *BipolarChannels* sheets.

## 5. Windows and temporal split

**What windows are.** The task has no fixed trial structure: image sequences, 1.5-s response windows and 1-min distractor videos. The analysis unit is therefore a fixed-length **signal window** of 8 s. It is a segmentation unit for evaluating scalp-to-SEEG translation, and is not equivalent to an experimental trial.

**Inclusion.**
- A block is one ADES file.
- All task phases are eligible for analysis: image sequences, distractor videos and pauses alike. No segment is selected by task phase.
- The usable parts of each block are then set only by the pre-specified edge-margin, recording-gap and block-remainder rules below.
- A block that cannot be read is skipped and reported.

**Boundaries.**
- **Blocks are separate recordings.** Each block is a separate file, and the publication does not state that consecutive blocks are contiguous. Each block is therefore filtered on its own (Section 4).
- **Edge margins.** The first and last **3.4 s** of every block are excluded from all windows. Windows are cut from the remaining span of each block. Under the pre-specified impulse-response test (Section 4), the effective response of the preprocessing chain outside this margin is negligible at the stated numerical threshold (\|h\| < 10⁻⁶ of the peak). The FFT-based resampling has no strictly finite support, so this is a bound from the test, not a proof of zero dependence on padding.
- Windows are consecutive and non-overlapping, and never cross a block (file) boundary.
- **Recording gaps.** A gap is any stretch of at least 50 ms (10 samples) in which all kept scalp channels, or all target contacts, are exactly zero or non-finite. Windows overlapping a gap are dropped, and their number is reported.
- **Block remainders.** The last segment of a block's usable span (after the edge margins) is kept as a shorter window if it lasts at least 2 s (the minimum trial length used for ds004752), and dropped otherwise.
- **Gaps inside a block.** The same 3.4 s margin is applied on both sides of every recording gap.

**Non-finite values in single channels.** Before any model is run:
- non-finite samples are counted per scalp channel and per target contact, in every block;
- a scalp channel or contact with any non-finite sample in any block is marked invalid for that patient and excluded for all models;
- counts are reported, including zero counts.

This is a data-integrity rule, not artefact rejection by amplitude or by model output.

**How features are computed within windows.** Each window is processed like one ds004752 trial:
- the encoders (1-D convolutions with `padding` = kernel half-width) and the lagged features (shifts filled with zeros) are applied to each window separately;
- samples outside the window are therefore treated as zeros, and features near a window's edges are influenced by this zero padding;
- this is exactly how features were computed during encoder training and in every ds004752 analysis (trials of about 8 s), and it is applied identically to every model compared;
- windows only define the units of the calibration/gap/test split and of feature extraction; no feature uses samples from another window.

**Split, per patient.**
- Windows are ordered by block (1–7) and time.
- With N windows in total, **calibration** = the first n_cal = max(3, round(0.2·N)) windows.
- **Inclusion: n_cal ≥ 5**, so that the 5 calibration folds of Section 3 exist. With this formula, that means N ≥ 23 windows, about 3 min of usable signal. A patient with n_cal < 5 is excluded (Section 9).
- The smallest included case is N = 23: 5 calibration, 1 gap and 17 test windows.
- **Gap** = the next window.
- **Test** = all remaining windows.

**Normalisation (strict).** Every scalp channel and target is robust-standardised with the median and MAD of the calibration windows only.

## 6. Evaluation levels

| Level | Windows used |
|---|---|
| **Primary** | **All windows of Section 5. No automatic artefact rejection, no target clipping.** |
| Secondary (artefact-screened) | Windows whose CAR scalp EEG exceeds \|z\| = 15 are removed and targets are clipped at \|z\| = 10, with the median/MAD of the whole block. The split of Section 5 is then applied to the remaining windows. |
| Sensitivity (calibration-only thresholds) | The same thresholds, with median/MAD from the first 20 % of the windows, kept fixed for all later windows. The split is then applied to the remaining windows. |

## 7. Metrics, aggregation and statistics

**Contact level.** For each bipolar target, Pearson r between prediction and target over the concatenated test windows.

**Valid targets.** A target is valid only by technical criteria on the data, fixed here. Validity never depends on any model output or accuracy:
1. both contacts are recorded and the pair is listed in *BipolarChannels*;
2. neither contact is marked bad in any block;
3. the target has non-zero variance in both the calibration and the test windows.

Invalid targets are excluded for all models; they and their reasons are reported per patient.

**Constant predictions.** If a model's prediction for a valid target is constant over the test windows, Pearson r is mathematically undefined. The target is **not** excluded. Instead it receives a **score of 0** for that model, as a pre-specified scoring convention for a prediction failure; this is not a true Pearson correlation of 0. It is the convention of the evaluation code already used for ds004752, where the numerator is zero in that case. The number of such cases is reported separately for each model.

**Patient level.** The patient's score is the mean r over its included targets. The same set of targets, windows and calibration is used for every model in a patient.

**Unit of inference: the patient** (n = 12). Contacts and windows are never treated as independent observations.

**Other metrics** (same aggregation):
- **R²**, after the amplitude gain;
- **r_no50**, with 46–54 Hz removed from both signals;
- **r_event**: r after subtracting from each side its own average over the test stimulus epochs (−0.5 to +2.5 s around each stimulus marker), which removes the stimulus-locked component.

**Primary hypotheses, primary level, Pearson r.**
- **H1 (system).** Final > Base.
- **H2 (learning, matched for architecture and ensemble size).** Learned-3 > Fixed-3.

**Decision rule.**
- **Test:** paired **two-sided** Wilcoxon signed-rank test on the patient-level scores.
- **Multiplicity:** the Holm correction is applied **only to H1 and H2**. Every other comparison is labelled secondary or descriptive.
- **Supported:** Holm-adjusted p < 0.05 **and** median of the paired patient-level differences > 0. The median is the direction measure consistent with the Wilcoxon test.

**Also reported for H1 and H2 (supplementary estimates):**
- the mean of the paired patient-level differences, and all per-patient differences;
- the unadjusted 95 % CI of the mean difference from a patient-level bootstrap: 10 000 resamples of whole patients with replacement, each keeping that patient's pair of scores;
- the number of patients improved.

A non-significant result will not be described as evidence that the methods are equivalent.

**Descriptive analyses (no confirmatory claim):**
- Final vs Fixed-1 (system-level effect of the learned ensemble);
- H1 and H2 at the secondary and sensitivity levels;
- r_event, r_no50 and R²;
- calibration efficiency: Final with 10 % calibration vs Base with 20 %, on the same test windows.

**Reporting.** A hypothesis that is not supported is reported with the same prominence as one that is. No further model, configuration, hyper-parameter, reference or segmentation will be tried on ScaleOLDNEW to change an outcome.

## 8. Reference / common-mode analysis (descriptive, pre-specified)

The analysis of Section 3.9 of the paper, repeated with the lagged ridge baseline at the primary level:
- targets: monopolar, monopolar − iEEG common mode, bipolar, and the iEEG common mode itself;
- scalp inputs: original reference or CAR;
- per-patient correlation between the scalp and iEEG common modes, formed in volts.

**Recording setup, from Table 1 and the Methods of the original publication** (not from header channel counts):

| Setup | Patients (ID) |
|---|---|
| **Two 128-channel amplifiers**, separate references (scalp Fpz; SEEG a white-matter contact) | 1–8 (89, 90, 92, 96, 97, 98, 99, 101) |
| **One 256-channel amplifier** | 9–12 (102, 104, 105, 107) |

Header channel counts cannot separate the two setups. Headers list cap template positions and unused amplifier inputs as well as the recorded electrodes; for example, the ID 99 header has 280 channel entries although this patient belongs to the two-amplifier group. A patient whose file metadata contradicts the publication is marked *undetermined* and left out of the group summary.

**Prediction.** If the cross-modal common mode enters through a shared amplifier or reference path, the scalp–iEEG common-mode correlation and the monopolar/original-reference inflation should be larger in the one-amplifier group.

**Interpretation limits.** With 8 and 4 patients, this is reported descriptively (per-patient values and group means, no test). It is not used to make causal claims about the hardware.

## 9. Fixed before running

- **Segmentation and split:** window length 8 s; inclusion of whole blocks; calibration max(3, round(0.2·N)) windows, patient included only if this is ≥ 5; one-window gap.
- **Inputs:** channel rules; CAR as the primary scalp reference; the *BipolarChannels* target list.
- **Models:** the five encoders and the checkpoint rule (Section 3); model definitions Final, Base, Learned-3, Fixed-3, Fixed-1; readout penalty grids (Section 3).
- **Analysis:** primary level and metric; aggregation; decision rule.
- **Calibration folds:** 5 contiguous blocks of the calibration windows, used for every calibration-based choice (Section 3).
- **Patient exclusion:** only for fewer than 5 calibration windows (n_cal < 5, i.e. N < 23), fewer than 3 kept scalp channels or fewer than 3 valid targets; every exclusion is reported.
- **Execution log:** filter and resampling parameters, kept scalp channels per patient against Table 1, non-finite counts, excluded windows and targets, and constant-prediction counts are written to the Phase-17 log. A discrepancy between the log and this protocol is handled and reported as a deviation; it never leads to a silent rule change after results are seen.

**Sign-off.** Authors: ________  Date: ________

*Note added 15 Oct 2026 (wording only): this sign-off line was never completed. The protocol was fixed by the authors as dated above but was not formally signed. "Sign-off" and "locked" elsewhere in this document refer to that fixed version.*

---

## Addendum A (post hoc; written 11 Oct 2026, after the registered results, before this analysis was run)

**Status.** This addendum does not change the pre-registered analysis above. H1 and H2 were evaluated and reported exactly as specified (both supported). What follows is a post-hoc sensitivity analysis. It is reported as such and has no confirmatory weight.

**Reason.** The execution log showed more kept scalp channels than scalp electrodes listed in Table 1 of the original publication for patients 01 (31 vs 30), 02 (38 vs 30), 03 (35 vs 30) and 05 (38 vs 35). The expectation stated in Section 4 was that template positions without an electrode would be removed as bad or flat. That expectation was not met. Such channels enter both the model input and the CAR.

**Channel rule.** For every patient, the scalp channels are the channels kept under Section 4 that also appear in the original authors' EEGLAB scalp file (`SEEGxx.set`, distributed with the dataset). Labels are matched after the same lookup table and 10-05 naming.
- This uses verifiable metadata. It does not select channels to match the counts of Table 1.
- The authors' file reflects their own analysis and may omit electrodes that were placed but discarded. The rule therefore yields a conservative set of verified electrodes, not an exact list of placed electrodes.
- A patient is analysed only if at least 3 channels remain.

**Everything else is unchanged:**
- windows, split, targets and target validity of the primary level;
- the five encoders;
- the model definitions (Final, Base, Learned-3, Fixed-3);
- the 5 calibration folds;
- strict normalisation.

CAR is recomputed on the retained channels.

**Reporting.**
- H1 and H2 on these channel sets are reported next to the registered result, with unadjusted two-sided Wilcoxon p-values, patient-bootstrap CIs and per-patient differences. A per-patient table compares kept, verified and Table 1 channel counts.
- If the two analyses disagree, both are reported. The registered result remains the primary result.

**Terminology note (added 11 Oct 2026, after the Addendum A results; wording only).** In this addendum, "verified electrodes" means *channels listed in the original authors' scalp files*. These files show which channels the authors used. They do not prove which electrodes were physically placed. The published Table 1 and these files disagree on the number of scalp channels for some patients (for example ID 90: 30 in Table 1, 38 channels with 10-05 names in the file). The cause of this disagreement is unknown.

**Terminology note on the title (added 11 Oct 2026; wording only).** "Pre-registered" in the title means *written and signed by the authors before any model was applied to ScaleOLDNEW*. The protocol was not deposited in a public registry. The manuscript therefore calls it *pre-specified*. *(Correction added 15 Oct 2026: the protocol was written and fixed but not formally signed; see the note under Sign-off.)*

---

## Addendum B (post hoc; written 13 Oct 2026, after Addendum A and the Phase-20 channel audit, before this analysis was run)

**Status.** This addendum does not change the pre-specified analysis or Addendum A. It defines two post-hoc checks, B1 and B2, and how their results are used. The criteria below are fixed before any of these computations is run.

**Reason.** The Phase-20 audit found seven kept scalp channels that are absent from the authors' files and have raw standard deviations of 875–1,986 µV and |r| ≥ 0.99 with amplifier inputs that have no 10-05 name: patient 05 (C3, FC1, FC5, CP5), patient 06 (CP6) and patient 07 (AF1, FC1). That audit compared scalp channels with other scalp-amplifier inputs, not with the intracranial signals. If a scalp input carried a copy of an intracranial signal, the external result could be inflated by input–target leakage. A high correlation alone is not proof of a copy, because the scalp and intracranial signals can share a common mode through the references, grounding or amplifier (Section 8).

### B1. Input–target audit (all 12 patients, descriptive)

**Signals**, per block, for every kept scalp channel (including the seven channels above):
1. **Raw:** the unfiltered ADES samples at 4096 Hz in the same six 10-s segments per block as Phase 20, against every recorded SEEG contact; lags of −8 to +8 raw samples (about ±2 ms).
2. **Preprocessed:** the Phase-14 output (notch, 0.5–70 Hz, 200 Hz, original reference, whole blocks), against every recorded SEEG contact (monopolar), every bipolar target and the iEEG common mode (mean of the recorded contacts); lags of −10 to +10 samples (±50 ms).
3. **Model input:** CAR scalp EEG (kept channels) against every bipolar target, whole blocks, lags of ±50 ms.

**Statistics** for every scalp channel × intracranial signal pair and block: the largest |r| over the lags, its lag, and the affine fit (slope, intercept, R²) of the scalp signal on the intracranial signal at that lag.

**Copy-like criterion** (fixed here). A scalp channel is *copy-like* with respect to an intracranial signal s (a contact or a bipolar target) if, at signal stage 2, all three conditions hold in every block of the patient:
- the largest lagged |r| is at least 0.90;
- the lag at which it occurs is the same within ±1 sample across blocks;
- |r| with s exceeds |r| with the iEEG common mode by at least 0.10.

The last condition separates a contact-specific copy from a common mode shared across modalities. Stages 1 and 3 are reported for every pair that meets the criterion.

**Reporting.**
- Per patient: the largest |r| over all scalp × intracranial pairs at each stage, with the pair and lag.
- For the seven channels: their full results.
- The full pair table as a CSV file.

### B2. Sensitivity without the seven channels

H1 and H2 are recomputed with only the seven channels above removed. CAR is recomputed on the remaining channels and the readouts are refitted on the calibration windows. Everything else is unchanged: windows, split, targets, target validity, the five encoders, model definitions, calibration folds and strict normalisation. Only patients 05, 06 and 07 change. For the other nine patients, the input data and the deterministic readout are identical to the registered analysis, so their registered results are used. The results are reported next to the registered result and the Addendum A result, with unadjusted two-sided Wilcoxon p-values, patient-bootstrap CIs and per-patient differences.

### Use of the results

- **No copy-like channel.** The registered result remains the primary result, and B1 and B2 are reported as post-hoc checks.
- **Any copy-like channel.**
  - The channel is reported.
  - H1 and H2 are recomputed without all copy-like channels, as in B2.
  - The external-validation conclusions of the manuscript are restated on that basis, and the registered result is reported as affected by input–target leakage.
  - This holds whatever the direction of the change.

---

## Addendum C (post hoc; written 15 Oct 2026, after Addenda A and B and the dense-lag analysis on ds004752, before this analysis was run)

**Status.** This addendum does not change the pre-specified analysis or Addenda A and B. It defines one post-hoc, descriptive comparison with no confirmatory weight. Its settings are fixed here, before it is run.

**Reason.** On ds004752 (development data, Phase 24), a ridge on every lag within ±200 ms recovered 0.016 of the 0.037 gain of the final model over Base (LAG9). It reached almost the same accuracy as a ridge on every lag within ±1.2 s (0.183 vs 0.184). This addendum asks whether the final model and the fixed bank also exceed this stronger linear control on ScaleOLDNEW.

**Models.** All models use the primary level, the v1.6 channels, windows, split, targets and target validity, the five calibration folds and strict normalisation of the registered analysis.
- **Dense-200:** ridge on every lag from −200 to +200 ms (81 lags at 200 Hz). The penalty is chosen by pooled out-of-fold r from 10⁻³ to 10⁷ in half-decade steps. It is evaluated alone and mixed with LAG9, with the weight w ∈ {0, 0.25, 0.5, 0.75, 1} chosen by pooled out-of-fold r, as for the encoder pipelines.
- **Base** and **Fixed-1** (alone and with LAG9) are recomputed with the same code only as consistency checks against the registered values. For Final, the registered result is used.

**Not run.** The ridge on every lag within ±1.2 s would have up to 38 × 481 = 18,278 features per patient in this cohort, which is computationally prohibitive in this analysis. It is therefore not run, and this is stated in the manuscript.

**Comparisons** (patient-bootstrap 95 % CI, two-sided Wilcoxon, unadjusted, 12 patients):
- C1: Final − Dense-200 + LAG9;
- C2: Fixed-1 + LAG9 − Dense-200 + LAG9;
- C3: Dense-200 − Base.

**Use of the results.** C1–C3 are reported in the manuscript and the supplement whatever their direction, as post-hoc descriptive results. They do not change the status of H1 and H2.
