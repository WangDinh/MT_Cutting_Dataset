"""A small DDPM for spectrogram patches, with RePaint-style inpainting at sampling time.

The model is unconditional: it learns the distribution of clean ``(1, N_BINS, N_FRAMES)`` patches
(normalised to [-1, 1], see ``masking.normalize_db``). Missing frames are filled at sampling time by
RePaint (Lugmayr et al., CVPR 2022): at every step the known region is replaced by a noised copy of
the observed data, and the model only has to denoise the unknown region consistently with it.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x))) + self.t(F.silu(te))[:, :, None, None]
        return self.c2(F.silu(self.n2(h))) + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        z = self.norm(x).flatten(2).transpose(1, 2)
        z, _ = self.attn(z, z, z, need_weights=False)
        return x + z.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    """Noise-prediction U-Net. Three 2x downsamplings, attention at the lowest resolution."""

    def __init__(self, base: int = 32, mult: tuple[int, ...] = (1, 2, 3, 4)):
        super().__init__()
        self.base, tdim = base, base * 4
        self.temb = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        chs = [base * m for m in mult]
        self.inc = nn.Conv2d(1, chs[0], 3, padding=1)
        self.down, self.down_ds, skip_chs, c = nn.ModuleList(), nn.ModuleList(), [], chs[0]
        for i, co in enumerate(chs):
            self.down.append(nn.ModuleList([ResBlock(c, co, tdim), ResBlock(co, co, tdim)]))
            skip_chs.append(co)
            c = co
            self.down_ds.append(nn.Conv2d(co, co, 3, stride=2, padding=1) if i < len(chs) - 1 else nn.Identity())
        self.mid1, self.mid_attn, self.mid2 = ResBlock(c, c, tdim), Attention(c), ResBlock(c, c, tdim)
        self.up, self.up_us = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(chs))):
            co = chs[i]
            self.up.append(nn.ModuleList([ResBlock(c + skip_chs[i], co, tdim), ResBlock(co, co, tdim)]))
            c = co
            self.up_us.append(nn.Upsample(scale_factor=2, mode="nearest") if i > 0 else nn.Identity())
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t):
        te = self.temb(time_embedding(t, self.base))
        h, skips = self.inc(x), []
        for blocks, ds in zip(self.down, self.down_ds):
            for b in blocks:
                h = b(h, te)
            skips.append(h)
            h = ds(h)
        h = self.mid2(self.mid_attn(self.mid1(h, te)), te)
        for blocks, us in zip(self.up, self.up_us):
            h = torch.cat([h, skips.pop()], dim=1)
            for b in blocks:
                h = b(h, te)
            h = us(h)
        return self.out(h)


def cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    abar = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    return (1 - abar[1:] / abar[:-1]).clamp(1e-5, 0.999).float()


class Diffusion:
    """DDPM schedule with epsilon-prediction, ancestral sampling and RePaint inpainting."""

    def __init__(self, T: int = 250, device: str | torch.device = "cpu"):
        self.T, self.device = T, torch.device(device)
        self.betas = cosine_betas(T).to(self.device)
        self.alphas = 1 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)
        self.abar_prev = torch.cat([torch.ones(1, device=self.device), self.abar[:-1]])
        self.post_var = self.betas * (1 - self.abar_prev) / (1 - self.abar)

    def q_sample(self, x0, t, noise):
        a = self.abar[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, model, x0, t=None, noise=None):
        t = torch.randint(0, self.T, (len(x0),), device=x0.device) if t is None else t
        noise = torch.randn_like(x0) if noise is None else noise
        return F.mse_loss(model(self.q_sample(x0, t, noise), t), noise)

    def p_sample(self, model, x, t: int):
        """One ancestral step x_t -> x_{t-1} (index ``t`` in 0..T-1)."""
        tt = torch.full((len(x),), t, device=x.device, dtype=torch.long)
        eps = model(x, tt)
        x0 = ((x - (1 - self.abar[t]).sqrt() * eps) / self.abar[t].sqrt()).clamp(-1, 1)
        mean = (self.abar_prev[t].sqrt() * self.betas[t] / (1 - self.abar[t])) * x0 \
            + (self.alphas[t].sqrt() * (1 - self.abar_prev[t]) / (1 - self.abar[t])) * x
        return mean if t == 0 else mean + self.post_var[t].sqrt() * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, model, shape):
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x

    @torch.no_grad()
    def repaint(self, model, x_known, known, n_resample: int = 2):
        """Fill the unknown region of ``x_known`` ([-1, 1], ``(B, 1, F, T)``); ``known`` is 1 where observed."""
        x = torch.randn_like(x_known)
        for t in reversed(range(self.T)):
            for u in range(n_resample):
                if t > 0:
                    x_kn = self.abar[t - 1].sqrt() * x_known + (1 - self.abar[t - 1]).sqrt() * torch.randn_like(x)
                else:
                    x_kn = x_known
                x_prev = known * x_kn + (1 - known) * self.p_sample(model, x, t)
                if u < n_resample - 1 and t > 0:  # jump back to level t and redo the step (harmonises the seam)
                    x = self.alphas[t].sqrt() * x_prev + self.betas[t].sqrt() * torch.randn_like(x)
                else:
                    x = x_prev
        return x


class EMA:
    """Exponential moving average of the weights; ``.model`` is the copy used for sampling."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model, self.decay = copy.deepcopy(model).eval(), decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p, 1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def to_tensor(db_patches: np.ndarray, device) -> torch.Tensor:
    """``(N, F, T)`` dB array -> normalised ``(N, 1, F, T)`` float tensor."""
    from .masking import normalize_db

    return torch.from_numpy(normalize_db(db_patches)).float().unsqueeze(1).to(device)


def inpaint(model, diffusion: Diffusion, db_patches: np.ndarray, time_mask: np.ndarray, *, n_resample: int = 2,
            batch_size: int = 64) -> np.ndarray:
    """RePaint-inpaint dB patches whose ``time_mask`` frames are missing. Returns dB patches (observed
    frames are returned exactly as given)."""
    from .masking import DB_HI, DB_LO, denormalize_db

    out = []
    for i in range(0, len(db_patches), batch_size):
        x = to_tensor(db_patches[i:i + batch_size], diffusion.device)
        known = torch.from_numpy(~time_mask[i:i + batch_size]).to(diffusion.device).float()[:, None, None, :]
        filled = diffusion.repaint(model, x * known, known, n_resample)
        out.append(denormalize_db(filled.squeeze(1).cpu().numpy()))
    pred = np.clip(np.concatenate(out), DB_LO, DB_HI)
    return np.where(time_mask[:, None, :], pred, db_patches)
