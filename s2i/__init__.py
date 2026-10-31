"""
s2i — Scalp-to-Intracranial translation (rebuilt from scratch).

Package layout
--------------
s2i/montage.py                 10-20 channel canonicalisation + 3D positions
s2i/data/preprocess_ds004752.py BIDS -> per-subject trial-level .pt files
s2i/data/dataset.py            trial-level storage, leakage-free splits, windowing, collate
s2i/model/leadformer.py        LeadFormer: geometry-aware encoder-decoder
s2i/losses.py                  NLL (observability) + Pearson + multi-resolution STFT
s2i/metrics.py                 per-contact r / R^2 (unclipped) / band-limited r
s2i/baselines.py               lagged ridge regression baseline (must be beaten)
s2i/train_loso.py              full LOSO protocol: pretrain -> zero-shot -> calibration -> ridge
s2i/diagnose_reference.py      common-reference contamination test
s2i/smoke_test.py              synthetic end-to-end test (no real data needed)
"""
