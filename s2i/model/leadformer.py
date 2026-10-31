"""
LeadFormer — geometry-aware encoder-decoder for scalp-EEG -> iEEG translation.

Design goals (each one targets a concrete weakness of CAST):

1. Montage-agnostic encoder.
   CAST flattened (21 channels x patch) into one vector, so channel identity was
   tied to an array index that differed between subjects. LeadFormer tokenises
   every scalp channel separately (shared multi-scale conv tokenizer) and adds a
   3D positional embedding of the electrode. Attention alternates between the
   spatial axis (across electrodes, padded electrodes masked) and the temporal
   axis. Works for 6..21 electrodes without zero-padding hacks.

2. Subject-transferable decoder.
   CAST's decoder used `nn.Embedding(channel_index)` -> re-initialised from scratch
   for every new patient, so nothing about the output side transferred. Here every
   target contact is a query built from its MNI coordinate, so the *same* decoder
   is trained jointly on all training subjects and can be applied zero-shot to an
   unseen patient's electrode layout.

3. Lead-field-inspired attention prior.
   Volume conduction is instantaneous and decays with distance. Decoder queries
   cross-attend to scalp tokens at the *same* time step with an additive bias
   b_h(k, c) = MLP(r_kc) - softplus(a_h) * |r_kc|, where r_kc is the vector from
   contact k to electrode c. Different heads learn different spatial scales.

4. Observability head.
   A per-contact log-variance is predicted (heteroscedastic Gaussian NLL). It
   (a) down-weights unpredictable contacts during training, and (b) gives a
   principled, ground-truth-free ranking of reconstructable contacts, replacing
   the post-hoc "r >= 0.15 on calibration" threshold.

5. Light per-subject calibration.
   Optional `SubjectAdapter`: per-contact query offset + output gain/bias. Only
   these (and optionally the decoder) are tuned on a few minutes of data.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoid_table(n, d, device=None):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class FourierXYZ(nn.Module):
    """Random-Fourier-feature embedding of 3D coordinates (mm)."""

    def __init__(self, d, n_freq=32, scale_mm=80.0, sigma=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(3, n_freq, generator=g) * sigma)
        self.scale = scale_mm
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq + 3, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, xyz):
        x = xyz / self.scale
        proj = 2 * math.pi * x @ self.B
        return self.mlp(torch.cat([proj.sin(), proj.cos(), x], -1))


class MHA(nn.Module):
    """Multi-head attention on (N, L, d) with additive bias and key-padding mask."""

    def __init__(self, d, h, dropout=0.0):
        super().__init__()
        assert d % h == 0
        self.h, self.dh, self.p = h, d // h, dropout
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, ctx=None, bias=None, key_mask=None):
        ctx = x if ctx is None else ctx
        N, Lq, d = x.shape
        Lk = ctx.shape[1]
        q = self.q(x).view(N, Lq, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(ctx).view(N, Lk, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        mask = None
        if bias is not None:
            mask = bias.to(q.dtype)
        if key_mask is not None:
            km = torch.zeros(N, 1, 1, Lk, dtype=q.dtype, device=x.device)
            km = km.masked_fill(~key_mask[:, None, None, :], -1e4)
            mask = km if mask is None else mask + km
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=self.p if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(N, Lq, d))


class FFN(nn.Module):
    def __init__(self, d, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * mult), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d * mult, d))

    def forward(self, x):
        return self.net(x)


class Tokenizer(nn.Module):
    """Shared per-channel multi-scale conv stem + non-overlapping patching."""

    def __init__(self, d, patch, kernels=(7, 15, 31), width=32):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(1, width, k, padding=k // 2) for k in kernels])
        self.norm = nn.GroupNorm(1, width * len(kernels))
        self.proj = nn.Conv1d(width * len(kernels), d, kernel_size=patch, stride=patch)

    def forward(self, x):                      # (N, L)
        x = x.unsqueeze(1)
        h = torch.cat([F.gelu(b(x)) for b in self.branches], 1)
        return self.proj(self.norm(h)).transpose(1, 2)   # (N, T, d)


class EncoderLayer(nn.Module):
    """Criss-cross block: spatial attention (across electrodes) then temporal attention."""

    def __init__(self, d, h, dropout):
        super().__init__()
        self.n_s, self.att_s = nn.LayerNorm(d), MHA(d, h, dropout)
        self.n_t, self.att_t = nn.LayerNorm(d), MHA(d, h, dropout)
        self.ff = FFN(d, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, eeg_mask):            # x (B, C, T, d)
        B, C, T, d = x.shape
        xs = x.permute(0, 2, 1, 3).reshape(B * T, C, d)
        xs = xs + self.drop(self.att_s(self.n_s(xs), key_mask=eeg_mask.repeat_interleave(T, 0)))
        xt = xs.view(B, T, C, d).permute(0, 2, 1, 3).reshape(B * C, T, d)
        xt = xt + self.drop(self.att_t(self.n_t(xt)))
        xt = xt + self.drop(self.ff(xt))
        return xt.view(B, C, T, d)


class GeoBias(nn.Module):
    """Lead-field-inspired additive attention bias (B, H, K, C)."""

    def __init__(self, h, hidden=64, scale_mm=80.0):
        super().__init__()
        self.scale = scale_mm
        self.mlp = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, h))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.decay = nn.Parameter(torch.linspace(0.5, 3.0, h))

    def forward(self, tgt_pos, eeg_pos):
        rel = (tgt_pos[:, :, None, :] - eeg_pos[:, None, :, :]) / self.scale
        dist = rel.norm(dim=-1, keepdim=True)
        b = self.mlp(torch.cat([rel, dist], -1)) - F.softplus(self.decay) * dist
        return b.permute(0, 3, 1, 2)


class DecoderLayer(nn.Module):
    """geo cross-attention (same t) -> contact self-attention (same t) -> temporal self-attention."""

    def __init__(self, d, h, dropout):
        super().__init__()
        self.h = h
        self.n_x, self.att_x = nn.LayerNorm(d), MHA(d, h, dropout)
        self.n_k, self.att_k = nn.LayerNorm(d), MHA(d, h, dropout)
        self.n_t, self.att_t = nn.LayerNorm(d), MHA(d, h, dropout)
        self.ff = FFN(d, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, mem, geo, eeg_mask, tgt_mask):
        B, K, T, d = q.shape
        C = mem.shape[1]
        qx = q.permute(0, 2, 1, 3).reshape(B * T, K, d)
        kv = mem.permute(0, 2, 1, 3).reshape(B * T, C, d)
        bias = geo.unsqueeze(1).expand(B, T, self.h, K, C).reshape(B * T, self.h, K, C)
        qx = qx + self.drop(self.att_x(self.n_x(qx), ctx=kv, bias=bias,
                                       key_mask=eeg_mask.repeat_interleave(T, 0)))
        qx = qx + self.drop(self.att_k(self.n_k(qx), key_mask=tgt_mask.repeat_interleave(T, 0)))
        qt = qx.view(B, T, K, d).permute(0, 2, 1, 3).reshape(B * K, T, d)
        qt = qt + self.drop(self.att_t(self.n_t(qt)))
        qt = qt + self.drop(self.ff(qt))
        return qt.view(B, K, T, d)


class SubjectAdapter(nn.Module):
    """Per-contact calibration parameters (zero/identity initialised)."""

    def __init__(self, n_contacts, d):
        super().__init__()
        self.q_delta = nn.Parameter(torch.zeros(n_contacts, d))
        self.gain = nn.Parameter(torch.ones(n_contacts))
        self.bias = nn.Parameter(torch.zeros(n_contacts))


class LeadFormer(nn.Module):
    def __init__(self, d=128, heads=4, enc_layers=4, dec_layers=3, patch=16,
                 max_tokens=512, dropout=0.1):
        super().__init__()
        self.d, self.patch = d, patch
        self.tokenizer = Tokenizer(d, patch)
        self.eeg_pos = FourierXYZ(d, seed=1)
        self.tgt_pos = FourierXYZ(d, seed=2)
        self.register_buffer("time_pe", sinusoid_table(max_tokens, d), persistent=False)
        self.encoder = nn.ModuleList([EncoderLayer(d, heads, dropout) for _ in range(enc_layers)])
        self.enc_norm = nn.LayerNorm(d)
        self.geo = GeoBias(heads)
        self.query_type = nn.Parameter(torch.zeros(d))
        self.decoder = nn.ModuleList([DecoderLayer(d, heads, dropout) for _ in range(dec_layers)])
        self.dec_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, patch)
        self.refine = nn.Sequential(nn.Conv1d(1, 16, 9, padding=4), nn.GELU(), nn.Conv1d(16, 1, 9, padding=4))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        self.logvar_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    # ------------------------------------------------------------------ parts
    def encode(self, eeg, eeg_pos, eeg_mask):
        B, C, L = eeg.shape
        assert L % self.patch == 0, "window length must be a multiple of patch"
        T = L // self.patch
        x = self.tokenizer(eeg.reshape(B * C, L)).view(B, C, T, self.d)
        x = x + self.eeg_pos(eeg_pos)[:, :, None, :] + self.time_pe[:T][None, None]
        for layer in self.encoder:
            x = layer(x, eeg_mask)
        return self.enc_norm(x)

    def decode(self, mem, eeg_pos, eeg_mask, tgt_pos, tgt_mask, cidx=None, adapter=None):
        B, C, T, d = mem.shape
        K = tgt_pos.shape[1]
        q0 = self.tgt_pos(tgt_pos) + self.query_type
        if adapter is not None:
            q0 = q0 + adapter.q_delta[cidx]
        q = q0[:, :, None, :] + self.time_pe[:T][None, None]
        geo = self.geo(tgt_pos, eeg_pos)
        for layer in self.decoder:
            q = layer(q, mem, geo, eeg_mask, tgt_mask)
        q = self.dec_norm(q)
        y = self.head(q).reshape(B * K, 1, T * self.patch)
        y = (y + self.refine(y)).view(B, K, T * self.patch)
        logvar = self.logvar_head(q.mean(2)).squeeze(-1)
        if adapter is not None:
            y = y * adapter.gain[cidx][..., None] + adapter.bias[cidx][..., None]
        return y, logvar

    def forward(self, eeg, eeg_pos, eeg_mask, tgt_pos, tgt_mask, cidx=None, adapter=None):
        mem = self.encode(eeg, eeg_pos, eeg_mask)
        return self.decode(mem, eeg_pos, eeg_mask, tgt_pos, tgt_mask, cidx, adapter)

    def param_groups(self):
        enc = [p for n, p in self.named_parameters()
               if n.startswith(("tokenizer.", "eeg_pos.", "encoder.", "enc_norm."))]
        enc_ids = {id(p) for p in enc}
        dec = [p for p in self.parameters() if id(p) not in enc_ids]
        return enc, dec

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
