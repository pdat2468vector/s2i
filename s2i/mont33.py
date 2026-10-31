"""
Phase 33 - montage density on the third cohort (paper/cohort3_protocol.md, Addendum A, 23 Oct 2026; post hoc,
descriptive). Each patient's scalp montage is reduced to 64 / 32 / 19 / 8 channels by farthest-point sampling on the
electrode positions; the stored EEG (CAR over all good channels) is re-referenced to the average of the subset, which
equals CAR over the subset alone. Targets, encoder, split and readout are those of the main cohort-3 analysis.

Commands
  build   DATA27/sub-XX.pt -> OUT/n{N}/sub-XX.pt (+ _normstats.pt) for every density, and mont33_build.json
  report  readouts RUNS27/short/{learned,fixed}_n{N} and the main readouts -> mont33_report.md / .json

python -m s2i.mont33 build  --data DATA27 --out DATA33
python -m s2i.mont33 report --data DATA27 --runs RUNS27
"""
import argparse
import glob
import json
import os

import numpy as np

DENS = (64, 32, 19, 8)


def fps(pos, n):
    """Farthest-point sampling: start at the point closest to the centroid, then add the point farthest from the set."""
    pos = np.asarray(pos, float)
    sel = [int(np.argmin(np.linalg.norm(pos - pos.mean(0), axis=1)))]
    d = np.linalg.norm(pos - pos[sel[0]], axis=1)
    while len(sel) < min(n, len(pos)):
        k = int(np.argmax(d))
        sel.append(k)
        d = np.minimum(d, np.linalg.norm(pos - pos[k], axis=1))
    return sorted(sel)


def _robust(x):
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True) * 1.4826
    return med, np.where(mad < 1e-12, 1.0, mad)


def build(a):
    import torch
    log = {}
    for fp in sorted(glob.glob(os.path.join(a.data, "sub-*.pt"))):
        if fp.endswith("_normstats.pt"):
            continue
        sid = os.path.basename(fp)[:-3]
        d = torch.load(fp, weights_only=False)
        st0 = next(iter(d["norm_stats"]["eeg_car"].values()))     # one identical copy per run
        z = d["eeg_car"].float().numpy()
        raw = z * np.asarray(st0["mad"], np.float32)[:, None] + np.asarray(st0["med"], np.float32)[:, None]
        pos = np.asarray(d["eeg_pos"], float)
        ok = np.where(np.isfinite(pos).all(1))[0]
        info = {"n_good": int(len(pos)), "n_with_position": int(len(ok)), "subsets": {}}
        for n in DENS:
            sel = [int(ok[k]) for k in fps(pos[ok], n)]
            x = raw[sel] - raw[sel].mean(0, keepdims=True)          # CAR over the subset
            med, mad = _robust(x)
            e = dict(d)
            e["eeg_car"] = torch.from_numpy(np.clip((x - med) / mad, -6e4, 6e4).astype(np.float16))
            e["eeg_names"] = [d["eeg_names"][k] for k in sel]
            e["eeg_pos"] = [d["eeg_pos"][k] for k in sel]
            stats = {"med": med[:, 0].tolist(), "mad": mad[:, 0].tolist()}
            e["norm_stats"] = {"eeg_car": {r_: stats for r_ in d["norm_stats"]["eeg_car"]},
                               "ieeg_bip": d["norm_stats"]["ieeg_bip"]}
            od = os.path.join(a.out, f"n{n}")
            os.makedirs(od, exist_ok=True)
            torch.save(e, os.path.join(od, f"{sid}.pt.tmp"))
            os.replace(os.path.join(od, f"{sid}.pt.tmp"), os.path.join(od, f"{sid}.pt"))
            torch.save({k: e[k] for k in ("subject", "norm_stats", "norm_clip_z", "trial_bounds", "trial_session")},
                       os.path.join(od, f"{sid}_normstats.pt"))
            info["subsets"][n] = {"n": len(sel), "channels": e["eeg_names"]}
        log[sid] = info
        print(f"{sid}: {info['n_good']} good EEG channels, {info['n_with_position']} with a position -> "
              + ", ".join(f"{n}: {v['n']}" for n, v in info["subsets"].items()), flush=True)
    json.dump(log, open(os.path.join(a.out, "mont33_build.json"), "w"), indent=1)
    miss = [s for s, i in log.items() if i["n_with_position"] < i["n_good"]]
    print(f"\n{len(log)} patients; patients with channels lacking a position: {miss or 'none'}")


def report(a):
    from s2i.parm27 import _ex, _load, compare
    rng = np.random.default_rng(0)
    R = os.path.join(a.runs, "short")
    B = json.load(open(os.path.join(a.data, "p27_build.json")))["patients"]
    inc = sorted(s for s, i in B.items() if i["excluded"] is None)
    sets = [("all", _load(R, "learned_primary"), _load(R, "fixed_primary"))]
    sets += [(str(n), _load(R, f"learned_n{n}"), _load(R, f"fixed_n{n}")) for n in DENS]
    full = _ex(sets[0][1], "combo_nosess")
    L = ["# Phase 33: montage density, third cohort (Addendum A; post hoc, descriptive, unadjusted p)", "",
         "| channels | n | Base | Fixed-short | Final-short | Final-short − Base [95 % CI] | improved | p | "
         "Learned − Fixed [95 % CI] | improved | p | Final-short − Final-short (all channels) [95 % CI] | improved |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for lab, Ld, Fd in sets:
        if sorted(Ld) != inc or sorted(Fd) != inc:
            L.append(f"| {lab} | incomplete: learned {len(Ld)}, fixed {len(Fd)} of {len(inc)} |")
            continue
        base, fin, fix = _ex(Ld, "base"), _ex(Ld, "combo_nosess"), _ex(Fd, "combo_nosess")
        h1, h2 = compare(fin, base, rng), compare(fin, fix, rng)
        hf = compare(fin, full, rng) if lab != "all" else None
        out[lab] = {"base": float(np.mean(list(base.values()))), "fixed": float(np.mean(list(fix.values()))),
                    "final": float(np.mean(list(fin.values()))), "final_minus_base": h1,
                    "learned_minus_fixed": h2, "final_minus_full": hf}
        c = lambda h: f"{h['mean_diff']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}]"  # noqa: E731
        L.append(f"| {lab} | {h1['n']} | {out[lab]['base']:.4f} | {out[lab]['fixed']:.4f} | {out[lab]['final']:.4f} | "
                 f"{c(h1)} | {h1['wins']}/{h1['n']} | {h1['p']:.2e} | {c(h2)} | {h2['wins']}/{h2['n']} | {h2['p']:.2e} | "
                 + (f"{c(hf)} | {hf['wins']}/{hf['n']} |" if hf else "– | – |"))
    md = "\n".join(L)
    print(md)
    open(os.path.join(a.runs, "mont33_report.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(os.path.join(a.runs, "mont33_report.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "report"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--runs")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else report(a)


if __name__ == "__main__":
    main()
