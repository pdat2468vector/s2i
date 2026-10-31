# Analysis plan: remaining experiments requested by review 6 (post hoc)

Written on 24 Oct 2026, before any of the analyses below was run. All three are post hoc. They are reported as descriptive analyses with patient-bootstrap 95 % CIs, patients improved, and unadjusted two-sided Wilcoxon p. Results are reported whatever their direction.

## A. Leave-one-cohort-out training and training-set size (Phase 34; review 6, Section 3.2, items 2–3)

**Question.** Does the learned temporal filtering transfer across sites, amplifiers, and paradigms, and does the learning effect grow with the number of training patients?

**Encoder.** The short (±200 ms) architecture, the only one that applies to all three cohorts (the third cohort has 1-s epochs). It is trained exactly as the external-validation encoders (episodic ridge-in-the-loop objective, learnable λ, seed 42, 2 early-stopping patients drawn from the training pool, patience 6, at most 3000 steps), but on the patients of the other cohorts only. The support of a training episode is 240 s, except in the third cohort, where it is 30 % of the patient's epochs (its calibration fraction). Third-cohort patients enter training with their 32-channel montages (Phase 33, Addendum A of the cohort-3 protocol).

**Held-out cohorts and comparisons.** In each held-out cohort, the encoder is evaluated with the readout of the main analysis of that cohort (encoder mixed with the LAG9 ridge, strict normalisation, explicit predictions).

| Held-out cohort | Training cohorts | Evaluation | Compared with |
|---|---|---|---|
| Third cohort (31) | ds004752 + ScaleOLDNEW (27) | 32 channels, 30 % calibration | the encoder trained on ds004752 only (the encoder of the pre-specified analysis), and the fixed bank |
| ScaleOLDNEW (12) | ds004752 + third cohort (46) | primary level, 20 % calibration | the encoder trained on ds004752 only, and the fixed bank |
| ds004752 (15) | ScaleOLDNEW + third cohort (43) | primary level, strict, 20 % calibration | the leave-one-patient-out encoders trained within ds004752, and the fixed bank |

**Training-set size.** Pool = ds004752 + ScaleOLDNEW (27 patients), held-out = third cohort (32 channels). The 2 early-stopping patients are those of the full run. From the remaining 25, random subsets of 4, 8, and 16 optimisation patients are drawn with two subset seeds each, and the full 25 form the last point. Reported: learned − fixed for every encoder, and the mean over the two seeds of each size.

## B. Per-patient fine-tuning of the filters (Phase 35; review 6, M1(c))

**Question.** Does adapting the learned filters to the new patient's calibration data improve on the encoder trained on other patients?

**Procedure** (ds004752, primary level, strict normalisation, main-architecture fold encoders of the main analysis). Only the calibration trials of the patient are used. The last max(2, round(0.25 n_cal)) calibration trials are an inner validation set and the earlier ones an inner training set. Each step draws a query of 2 consecutive inner-training trials at a random position, with every earlier inner-training trial except the one just before the query as support. The loss is 1 − r of the closed-form ridge with the encoder's own λ, fixed. AdamW with learning rate 10⁻⁴ updates the filters. Every 5 steps, a ridge fitted on all inner-training trials is scored on the inner validation trials. The best state is kept (step 0 means no fine-tuning), with patience 4 evaluations and at most 100 steps. The fine-tuned encoder is then evaluated with the readout of the main analysis.

**Comparison.** Fine-tuned − not fine-tuned (the same fold encoder), mixed with the LAG9 ridge and alone, and fine-tuned − LAG9 ridge. Since the readout's penalty is chosen by cross-validation within calibration trials that the fine-tuning has seen, this selection is slightly optimistic for the fine-tuned encoder. The test trials are never used.

## C. Deep baselines under the main-analysis conditions (Phase 36; review 6, M3)

**Data.** ds004752 at the primary level, with each patient's scalp EEG and targets normalised using the median and MAD of the patient's own calibration trials (strict normalisation). The split (first 20 % for calibration, one-trial gap, the rest for testing) is that of the main analysis.

**Models.** All with their previous hyper-parameters, which were not tuned again for this analysis.
- *LeadFormer*: pre-trained leave-one-patient-out on 12 other patients, with 2 more for early stopping, and its decoder and per-contact adapter fine-tuned on the calibration trials.
- *NeuroFlowNet*: trained per patient on the calibration trials (zero-latent output, single sample, and mean of 8 samples).
- *ShallowConvNet-style front end* [Schirrmeister et al. 2017]: 40 temporal filters of 25 samples (±60 ms) with random Kaiming initialisation, applied to every scalp channel separately, with the filter outputs and the logarithm of their squared output smoothed over 75 samples as features. The spatial convolution of ShallowConvNet is replaced by the per-patient ridge decoder, since montages differ between patients. It is evaluated untrained and after the same episodic training as our encoder (leave-one-patient-out), each alone and mixed with the LAG9 ridge.

**Comparison.** Every model against the LAG9 ridge and against the final model, on the same 15 patients and test trials.

## D. CAST re-run under the present protocol (Phase 37; review 6, C2). Addendum of 27 Oct 2026

Written after 3 of the 15 folds had been completed (patients 01, 02, and 06), because a complete run would take about 40 GPU hours. The remaining folds are run in a fixed order: 07 and 08 (already started), then 09, 13, and 15 (the remaining patients with a strong common-mode signal, Section 4.5, where the earlier report had its highest values), then 10, 11, 12, and 14. Every fold completed when the analysis is stopped is reported, together with the number of patients. Before these folds, a positive control runs the same code with the earlier report's own protocol (monopolar targets, original scalp reference, random split of 2-s windows with a 1-s stride, its metric) on patients 01 and 13, whose values in that report were 0.268 and 0.784. The present-protocol results are interpreted only if the positive control approximately reproduces these values.
