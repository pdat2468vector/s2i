"""
Phase 35 (review 6, M1(c); plan paper/review6_experiments_plan.md, 24 Oct 2026; post hoc) - per-patient fine-tuning
of the learned filters on the patient's own calibration data, ds004752, primary level, strict normalisation.

For each patient, the main-architecture fold encoder (trained on other patients, RUN/enc_<sub>.pt) is fine-tuned with
the same ridge-in-the-loop objective, using only the calibration trials (first max(3, round(0.2 n)), as in s2i.readout):
  inner validation  the last max(2, round(0.25 n_cal)) calibration trials
  inner training    the earlier calibration trials
  step              query = 2 consecutive inner-training trials at a random position (at least 3 trials before it);
                    support = every inner-training trial before the query except the one just before it (gap);
                    loss 1 - r of the closed-form ridge (the encoder's own lambda, fixed); AdamW, lr 1e-4 on the filters
  selection         every 5 steps: ridge on all inner-training trials -> r on the inner validation trials; the best
                    state is kept (step 0 = no fine-tuning), patience 4 evaluations, at most 100 steps
The fine-tuned encoder is written to OUT/enc_<sub>.pt, so that s2i.readout evaluates it exactly like the other models.
No test trial is used at any stage.

python -m s2i.ft35 --data NOQC --run RUNS/shared_bip_car --out RUNS/ft35
python -m s2i.ft35 report --runs RUNS
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def report(runs):
    """Fine-tuned vs the same fold encoders without fine-tuning, the untrained bank, and the LAG9 ridge."""
    from s2i.parm27 import _ex, _load, compare
    rng = np.random.default_rng(0)
    rd = lambda rel: _load(os.path.join(runs, os.path.dirname(rel)), os.path.basename(rel))  # noqa: E731
    V = {"fine-tuned": rd("ft35/readout_combo1_strict_qc_none"),
         "not fine-tuned": rd("shared_bip_car/readout_combo1_strict_qc_none"),
         "untrained": rd("shared_bip_car/readout_untrained1_strict_qc_none")}
    L, out = ["# Phase 35: per-patient fine-tuning (ds004752, primary, strict; post hoc, unadjusted p)", ""], {}
    for model, lab in (("combo_nosess", "mixed with the LAG9 ridge"), ("enc", "encoder alone")):
        x = {k: _ex(v, model) for k, v in V.items()}
        base = _ex(V["not fine-tuned"], "base")
        L += [f"## {lab}", "", "| A | B | n | mean A | mean B | A − B [95 % CI] | A > B | p |", "|---|---|---|---|---|---|---|---|"]
        for A, B, xa, xb in (("fine-tuned", "not fine-tuned", x["fine-tuned"], x["not fine-tuned"]),
                             ("fine-tuned", "untrained", x["fine-tuned"], x["untrained"]),
                             ("not fine-tuned", "untrained", x["not fine-tuned"], x["untrained"]),
                             ("fine-tuned", "LAG9 ridge", x["fine-tuned"], base)):
            if not xa or not xb:
                L.append(f"| {A} | {B} | missing | | | | | |")
                continue
            h = compare(xa, xb, rng)
            out[f"{model}|{A}-{B}"] = h
            L.append(f"| {A} | {B} | {h['n']} | {h['mean_a']:.4f} | {h['mean_b']:.4f} | {h['mean_diff']:+.4f} "
                     f"[{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] | {h['wins']}/{h['n']} | {h['p']:.2e} |")
        L.append("")
    lp = os.path.join(runs, "ft35", "ft35_log.json")
    if os.path.exists(lp):
        lg = json.load(open(lp))
        L.append(f"Fine-tuning kept (best inner-validation step > 0) in {sum(v['best_step'] > 0 for v in lg.values())} "
                 f"of {len(lg)} patients; inner validation r {np.mean([v['inner_val_r0'] for v in lg.values()]):.4f} -> "
                 f"{np.mean([v['inner_val_best'] for v in lg.values()]):.4f} (mean).")
    md = "\n".join(L)
    print(md)
    open(os.path.join(runs, "ft35_report.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(out, open(os.path.join(runs, "ft35_report.json"), "w"), indent=1)


def main():
    import torch
    from s2i.readout import load_subject, strict_renorm
    from s2i.shared import TemporalEncoder, col_r, ridge_torch, rows
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        return report(sys.argv[sys.argv.index("--runs") + 1])
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--n_query", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    log = {}
    lp = os.path.join(a.out, "ft35_log.json")
    if os.path.exists(lp):
        log = json.load(open(lp))
    for ep in sorted(glob.glob(os.path.join(a.run, "enc_sub-*.pt"))):
        sid = os.path.basename(ep)[4:-3]
        dst = os.path.join(a.out, f"enc_{sid}.pt")
        if os.path.exists(dst) and sid in log:
            print(f"[{sid}] done", flush=True)
            continue
        t0 = time.time()
        ck = torch.load(ep, map_location=dev, weights_only=False)
        enc = TemporalEncoder(**ck["cfg"]).to(dev)
        enc.load_state_dict(ck["state"])
        lam = float(ck["lam"])
        E, Y, ses, _ = load_subject(os.path.join(a.data, f"{sid}.pt"))
        st = torch.load(os.path.join(a.data, f"{sid}_normstats.pt"), weights_only=False)
        n_cal = max(3, int(round(len(E) * a.frac)))
        E, Y = strict_renorm(E, Y, ses, st, n_cal)
        n_val = max(2, int(round(0.25 * n_cal)))
        tr, va = list(range(n_cal - n_val)), list(range(n_cal - n_val, n_cal))
        rng = np.random.default_rng(a.seed)
        torch.manual_seed(a.seed)
        opt = torch.optim.AdamW(enc.parameters(), lr=a.lr, weight_decay=0.0)

        @torch.no_grad()
        def inner_val():
            enc.eval()
            Xs, Ys = rows(enc, E, Y, tr, dev, amp)
            Xq, Yq = rows(enc, E, Y, va, dev, amp)
            return col_r(ridge_torch(Xs, Ys, Xq, lam), Yq).mean().item()
        best = inner_val()
        hist, best_state, bad, step = [(0, best)], {k: v.clone() for k, v in enc.state_dict().items()}, 0, 0
        q_lo = 4                                              # >= 3 support trials + 1 gap trial before the query
        if len(tr) >= q_lo + a.n_query:
            while step < a.max_steps and bad < a.patience:
                enc.train()
                q0 = int(rng.integers(q_lo, len(tr) - a.n_query + 1))
                sup, qry = tr[:q0 - 1], tr[q0:q0 + a.n_query]
                Xs, Ys = rows(enc, E, Y, sup, dev, amp)
                Xq, Yq = rows(enc, E, Y, qry, dev, amp)
                loss = 1 - col_r(ridge_torch(Xs, Ys, Xq, lam), Yq).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()
                step += 1
                if step % a.eval_every == 0:
                    v = inner_val()
                    hist.append((step, v))
                    if v > best:
                        best, bad, best_state = v, 0, {k: t.clone() for k, t in enc.state_dict().items()}
                    else:
                        bad += 1
        enc.load_state_dict(best_state)
        best_step = max(hist, key=lambda h: h[1])[0]
        torch.save({"state": enc.state_dict(), "cfg": enc.cfg, "lam": lam, "best_val": best, "hist": hist,
                    "fine_tuned_from": ep, "best_step": best_step}, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        log[sid] = {"n_cal": n_cal, "n_inner_train": len(tr), "n_inner_val": len(va), "inner_val_r0": hist[0][1],
                    "inner_val_best": best, "best_step": best_step, "steps": step}
        json.dump(log, open(lp, "w"), indent=1)
        print(f"[{sid}] calibration {n_cal} trials (inner {len(tr)} + {len(va)}) | inner val r {hist[0][1]:.4f} -> "
              f"{best:.4f} at step {best_step} of {step} | {time.time() - t0:.0f}s", flush=True)
        del enc, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    kept = sum(v["best_step"] > 0 for v in log.values())
    print(f"\n{len(log)} patients; fine-tuning kept (best step > 0) in {kept}")


if __name__ == "__main__":
    main()
