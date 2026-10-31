"""Inference helpers: full-trial reconstruction with weighted overlap-add, streaming val r."""
import numpy as np
import torch


@torch.no_grad()
def predict_trials(model, subj, trial_idx, win, stride, device, adapter=None, batch=16, amp=True):
    """Returns (preds, tgts, logvar_mean) with preds/tgts lists of (K, L) float32 arrays."""
    model.eval()
    w = torch.from_numpy((0.1 + 0.9 * np.hanning(win)).astype(np.float32)).to(device)
    K = subj.n_targets
    eeg_pos = subj.eeg_pos.to(device)[None]
    tgt_pos = subj.tgt_pos.to(device)[None]
    C = eeg_pos.shape[1]
    eeg_mask = torch.ones(1, C, dtype=torch.bool, device=device)
    tgt_mask = torch.ones(1, K, dtype=torch.bool, device=device)
    cidx = torch.arange(K, device=device)[None]
    preds, tgts, lv_sum, lv_n = [], [], torch.zeros(K, device=device), 0
    for ti in trial_idx:
        e, y = subj.trial(ti)
        e = e.float().to(device)
        L = e.shape[1]
        Lp = max(L, win)
        if Lp > L:
            e = torch.nn.functional.pad(e, (0, Lp - L))
        starts = list(range(0, Lp - win + 1, stride))
        if starts[-1] != Lp - win:
            starts.append(Lp - win)
        acc = torch.zeros(K, Lp, device=device)
        wsum = torch.zeros(Lp, device=device)
        for b0 in range(0, len(starts), batch):
            st = starts[b0:b0 + batch]
            xb = torch.stack([e[:, s:s + win] for s in st])
            n = len(st)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                pr, lv = model(xb, eeg_pos.expand(n, -1, -1), eeg_mask.expand(n, -1),
                               tgt_pos.expand(n, -1, -1), tgt_mask.expand(n, -1),
                               cidx.expand(n, -1), adapter)
            pr, lv = pr.float(), lv.float()
            for j, s in enumerate(st):
                acc[:, s:s + win] += pr[j] * w
                wsum[s:s + win] += w
            lv_sum += lv.sum(0)
            lv_n += n
        preds.append((acc / wsum)[:, :L].cpu().numpy())
        tgts.append(y.float().numpy())
    return preds, tgts, (lv_sum / max(lv_n, 1)).cpu().numpy()


@torch.no_grad()
def streaming_r(model, loader, n_subjects, n_targets, device, amp=True, adapter=None):
    """Mean (over subjects) of mean (over contacts) Pearson r, accumulated over windows."""
    model.eval()
    stats = [torch.zeros(k, 5, dtype=torch.float64, device=device) for k in n_targets]
    counts = [0] * n_subjects
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            pr, _ = model(b["eeg"], b["eeg_pos"], b["eeg_mask"], b["tgt_pos"], b["tgt_mask"],
                          b["cidx"], adapter)
        pr = pr.double()
        tg = b["tgt"].double()
        L = pr.shape[-1]
        for i in range(pr.shape[0]):
            s = int(b["subj"][i])
            k = int(b["tgt_mask"][i].sum())
            p, t, ci = pr[i, :k], tg[i, :k], b["cidx"][i, :k]
            v = torch.stack([p.sum(-1), t.sum(-1), (p * p).sum(-1), (t * t).sum(-1), (p * t).sum(-1)], -1)
            stats[s].index_add_(0, ci, v)
            counts[s] += L
    per_subj = []
    for s, st in enumerate(stats):
        n = counts[s]
        if n == 0:
            continue
        sp, sq, spp, sqq, spq = st.unbind(-1)
        num = n * spq - sp * sq
        den = torch.sqrt((n * spp - sp ** 2).clamp_min(1e-12) * (n * sqq - sq ** 2).clamp_min(1e-12))
        per_subj.append((num / den).mean().item())
    return float(np.mean(per_subj)) if per_subj else float("nan")
