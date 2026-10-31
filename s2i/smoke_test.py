"""
End-to-end smoke test on SYNTHETIC data (no real recordings, CPU is fine).

It simulates a toy volume-conduction world that deliberately contains a shared
recording reference R in both scalp EEG and monopolar iEEG, writes subjects in
the exact format produced by preprocess_ds004752.py, then runs:
  1. diagnose_reference  (should flag A >> D, i.e. reference contamination)
  2. train_loso          (tiny model, 2 folds, a few steps) -> checks the whole pipeline

python -m s2i.smoke_test
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

from s2i.data.preprocess_ds004752 import common_mode_stats, make_bipolar, robust_norm

FS = 200


def _sphere_points(n, r, rng, upper=True):
    v = rng.normal(size=(n, 3))
    if upper:
        v[:, 2] = np.abs(v[:, 2])
    return r * v / np.linalg.norm(v, axis=1, keepdims=True)


def _gain(a, b, p=2.0, eps=5.0):
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return 1.0 / (d + eps) ** p


def make_subject(sid, rng, n_trials=30, L=8 * FS):
    n_src = 40
    src_pos = _sphere_points(n_src, 1.0, rng, upper=False) * rng.uniform(20, 70, (n_src, 1))
    all_scalp = _sphere_points(21, 90.0, rng)
    n_eeg = int(rng.choice([6, 8, 17, 19, 21]))
    eeg_pos = all_scalp[:n_eeg]
    # 3 shafts x 8 contacts going from the surface inwards
    contacts, names = [], []
    for s in range(3):
        entry = _sphere_points(1, 75.0, rng)[0]
        direction = -entry / np.linalg.norm(entry)
        for c in range(8):
            contacts.append(entry + direction * 3.5 * c)
            names.append(f"S{s}C{c + 1}")
    contacts = np.asarray(contacts)
    T = n_trials * L
    t = np.arange(T) / FS
    freqs = rng.uniform(3, 30, n_src)
    src = np.sin(2 * np.pi * freqs[:, None] * t + rng.uniform(0, 6.28, (n_src, 1)))
    src *= 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5, (n_src, 1)) * t)
    src += 0.5 * rng.normal(size=src.shape)
    R = 3.0 * np.convolve(rng.normal(size=T), np.ones(20) / 20, mode="same")   # shared reference
    G_e = _gain(eeg_pos, src_pos)
    G_i = _gain(contacts, src_pos, p=3.0)
    e_v = (G_e / G_e.sum(1, keepdims=True)) @ src - R + 0.3 * rng.normal(size=(n_eeg, T))
    i_v = (G_i / G_i.sum(1, keepdims=True)) @ src - R + 0.3 * rng.normal(size=(len(names), T))
    pairs = make_bipolar(names)
    bip_v = np.stack([i_v[a] - i_v[b] for a, b in pairs])
    diag = common_mode_stats(i_v, e_v)
    depth = ["Cortical" if c % 8 < 3 else ("Mid" if c % 8 < 5 else "Deep") for c in range(len(names))]
    d = {
        "subject": sid, "fs": FS, "eeg_names": [f"E{i}" for i in range(n_eeg)],
        "eeg_pos": torch.tensor(eeg_pos, dtype=torch.float32),
        "contact_names": names, "contact_mni": torch.tensor(contacts, dtype=torch.float32),
        "contact_region": depth, "contact_cat": depth, "contact_depth": depth,
        "bip_pairs": pairs, "bip_names": [f"{names[a]}-{names[b]}" for a, b in pairs],
        "bip_mni": torch.tensor(np.stack([(contacts[a] + contacts[b]) / 2 for a, b in pairs]),
                                dtype=torch.float32),
        "bip_region": [depth[a] for a, _ in pairs], "bip_cat": [depth[a] for a, _ in pairs],
        "bip_depth": [depth[a] for a, _ in pairs],
        "trial_bounds": torch.tensor([(i * L, (i + 1) * L) for i in range(n_trials)]),
        "trial_session": ["ses-01"] * n_trials, "trial_id": list(range(n_trials)),
        "diagnostics": diag, "n_rejected_trials": 0,
        "eeg_orig": torch.from_numpy(robust_norm(e_v).astype(np.float16)),
        "eeg_car": torch.from_numpy(robust_norm(e_v - e_v.mean(0, keepdims=True)).astype(np.float16)),
        "ieeg_mono": torch.from_numpy(robust_norm(i_v).astype(np.float16)),
        "ieeg_bip": torch.from_numpy(robust_norm(bip_v).astype(np.float16)),
    }
    return d


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    root = tempfile.mkdtemp(prefix="s2i_smoke_")
    data = os.path.join(root, "data")
    os.makedirs(data)
    for i in range(5):
        sid = f"sub-{i + 1:02d}"
        torch.save(make_subject(sid, rng), os.path.join(data, f"{sid}.pt"))
    print(f"synthetic data -> {data}")

    from s2i import diagnose_reference, train_loso
    print("\n--- diagnose_reference ---")
    sys.argv = ["diag", "--data", data]
    diagnose_reference.main()

    print("\n--- train_loso (tiny) ---")
    sys.argv = ["train", "--data", data, "--out", os.path.join(root, "run"), "--target", "bip",
                "--device", "cpu", "--no_amp", "--d", "32", "--enc_layers", "1", "--dec_layers", "1",
                "--epochs", "2", "--steps", "3", "--bs", "4", "--win", "400", "--k_max", "12",
                "--val_trials", "4", "--n_val_subj", "1", "--calib_epochs", "2",
                "--folds", "sub-01", "sub-02", "--min_trials", "10"]
    train_loso.main()
    print(f"\nSMOKE TEST PASSED. Outputs in {root}")
    if "--keep" not in sys.argv:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
