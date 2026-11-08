"""Loading the masked patches, classical inpainting baselines and the evaluation metrics.

All arrays here are log-magnitude spectrograms in dB with shape ``(N, N_BINS, N_FRAMES)`` and
boolean time masks ``(N, N_FRAMES)`` where True = missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .config import PROCESSED_DIR
from .masking import DB_HI, DB_LO, FRAME_S

MASKED_DIR = PROCESSED_DIR / "masked"
LOW_BAND_BINS = 64  # first 64 bins = 0-3 kHz, where almost all the energy is


def load_patches(split: str | None = None, root: Path = MASKED_DIR) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """``(clean_db, time_mask, meta)`` for one split (``train|val|test|krpm``) or all patches."""
    npz = np.load(root / "patches.npz")
    meta = pd.read_csv(root / "patches.csv", index_col="patch_idx")
    sel = npz["split"] == split if split else np.ones(len(meta), dtype=bool)
    return npz["clean"][sel].astype(np.float32), npz["mask"][sel], meta[sel]


def longest_span_ms(mask: np.ndarray) -> np.ndarray:
    """Length in ms of the longest missing run in each ``(N, T)`` mask."""
    out = np.zeros(len(mask))
    for i, m in enumerate(mask):
        d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
        runs = np.where(d == -1)[0] - np.where(d == 1)[0]
        out[i] = runs.max() * FRAME_S * 1000 if len(runs) else 0.0
    return out


def fill_mean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace missing frames with the per-bin mean of the observed frames of the same patch."""
    out = x.copy()
    for i in range(len(x)):
        out[i][:, mask[i]] = x[i][:, ~mask[i]].mean(axis=1, keepdims=True)
    return out


def fill_linear(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-bin linear interpolation across the gap (edge value held at the patch borders)."""
    out = x.copy()
    t = np.arange(x.shape[-1])
    for i in range(len(x)):
        obs = np.flatnonzero(~mask[i])
        f = interp1d(obs, x[i][:, obs], kind="linear", axis=1, bounds_error=False,
                     fill_value=(x[i][:, obs[0]], x[i][:, obs[-1]]))  # hold the edge value outside the data
        out[i][:, mask[i]] = f(t[mask[i]])
    return np.clip(out, DB_LO, DB_HI)


BASELINES = {"mean": fill_mean, "linear": fill_linear}


def masked_errors(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, bins: slice = slice(None)) -> pd.DataFrame:
    """Per-patch MAE and RMSE (dB) over the missing frames only, optionally restricted to a bin range."""
    rows = []
    for p, t, m in zip(pred, target, mask):
        d = (p - t)[bins][:, m]
        rows.append((np.abs(d).mean(), np.sqrt((d**2).mean())))
    return pd.DataFrame(rows, columns=["mae_db", "rmse_db"])


def gap_stats(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, bins: slice = slice(0, LOW_BAND_BINS)) -> pd.DataFrame:
    """Per-patch statistics that error metrics miss, over the missing frames of the low band (0-3 kHz).

    * ``profile_corr``: correlation across frequency between the predicted and true gap-averaged spectrum,
      i.e. whether the harmonic stripes are recovered at the right frequencies.
    * ``std_ratio``: temporal std of the prediction over the gap divided by that of the truth, averaged over
      bins. ~1 means a realistic texture; a blur (mean fill) gives ~0.
    """
    rows = []
    for p, t, m in zip(pred, target, mask):
        pg, tg = p[bins][:, m], t[bins][:, m]
        pm, tm = pg.mean(axis=1), tg.mean(axis=1)
        corr = np.corrcoef(pm, tm)[0, 1] if pm.std() > 0 and tm.std() > 0 else np.nan
        ratio = pg.std(axis=1).mean() / max(tg.std(axis=1).mean(), 1e-6)
        rows.append((corr, ratio))
    return pd.DataFrame(rows, columns=["profile_corr", "std_ratio"])
