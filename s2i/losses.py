"""
Losses.

L = NLL(observability)  +  w_r * (1 - Pearson)  +  w_s * multi-resolution STFT

* NLL: per-contact heteroscedastic Gaussian on the window MSE. Contacts the model
  cannot predict get a large variance and contribute small gradients, so model
  capacity is not wasted on scalp-invisible sources.
* Multi-resolution STFT replaces CAST's single full-window |FFT| loss (which had
  no time resolution and was dominated by noise bins).
"""
import torch


def masked_mean(x, mask):
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def pearson(pred, tgt, eps=1e-6):
    """(B, K, L) -> (B, K)."""
    p = pred - pred.mean(-1, keepdim=True)
    t = tgt - tgt.mean(-1, keepdim=True)
    return (p * t).sum(-1) / torch.sqrt((p * p).sum(-1) * (t * t).sum(-1) + eps)


def mr_stft(pred, tgt, mask, ffts=(32, 64, 128)):
    x, y = pred[mask], tgt[mask]
    if x.numel() == 0:
        return pred.new_zeros(())
    loss = 0.0
    for n in ffts:
        w = torch.hann_window(n, device=x.device)
        X = torch.stft(x, n, hop_length=n // 4, window=w, return_complex=True).abs()
        Y = torch.stft(y, n, hop_length=n // 4, window=w, return_complex=True).abs()
        sc = torch.linalg.norm(Y - X, dim=(-2, -1)) / torch.linalg.norm(Y, dim=(-2, -1)).clamp_min(1e-6)
        lm = (torch.log(X + 1e-4) - torch.log(Y + 1e-4)).abs().mean(dim=(-2, -1))
        loss = loss + (sc + lm).mean()
    return loss / len(ffts)


def total_loss(pred, logvar, tgt, mask, w_r=1.0, w_s=0.1):
    pred, logvar, tgt = pred.float(), logvar.float(), tgt.float()
    mse = ((pred - tgt) ** 2).mean(-1)
    lv = logvar.clamp(-5.0, 3.0)
    nll = 0.5 * (mse * torch.exp(-lv) + lv)
    r = pearson(pred, tgt)
    l_nll = masked_mean(nll, mask)
    l_r = masked_mean(1.0 - r, mask)
    l_s = mr_stft(pred, tgt, mask) if w_s > 0 else pred.new_zeros(())
    loss = l_nll + w_r * l_r + w_s * l_s
    return loss, {"nll": l_nll.item(), "r": 1.0 - l_r.item(), "stft": float(l_s),
                  "mse": masked_mean(mse, mask).item()}
