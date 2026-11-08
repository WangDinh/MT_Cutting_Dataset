"""Spectrogram patches and time-span masks for the missing-data (inpainting) task.

A *patch* is a fixed-length waveform window turned into a log-magnitude
spectrogram of shape ``(N_BINS, N_FRAMES)`` (frequency x time). A *mask* marks
the time frames that are "missing" (dropouts), shared across all frequencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from .config import SAMPLE_RATE
from .features import frame_rms

N_FFT = 1024
HOP = 256
N_FRAMES = 64
# Keep only the lowest bins (0-6 kHz). Sensor 2 (base ISS) has no energy above ~6 kHz, so higher
# bins are digital silence (>80% of values at the dB floor) and would swamp the model.
N_BINS = 128
BIN_HZ = SAMPLE_RATE / N_FFT
PATCH_SAMPLES = (N_FRAMES - 1) * HOP + N_FFT  # 17152 samples, ~0.357 s at 48 kHz
FRAME_S = HOP / SAMPLE_RATE

# Log-magnitude (dB re full scale) is clipped to this range, then mapped to [-1, 1].
DB_LO, DB_HI = -100.0, 0.0


def patch_spectrogram_db(x_int16: np.ndarray) -> np.ndarray:
    """Log-magnitude STFT in dB of one ``PATCH_SAMPLES``-long int16 window -> ``(N_BINS, N_FRAMES)``."""
    assert len(x_int16) == PATCH_SAMPLES, len(x_int16)
    x = np.asarray(x_int16, dtype=np.float32) / 32768.0
    _, _, Z = signal.stft(x, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=False)
    db = 20 * np.log10(np.abs(Z[:N_BINS]) + 1e-6)
    return np.clip(db, DB_LO, DB_HI).astype(np.float32)


def normalize_db(db: np.ndarray) -> np.ndarray:
    """dB -> [-1, 1]."""
    return (db - DB_LO) / (DB_HI - DB_LO) * 2.0 - 1.0


def denormalize_db(x: np.ndarray) -> np.ndarray:
    """[-1, 1] -> dB."""
    return (x + 1.0) / 2.0 * (DB_HI - DB_LO) + DB_LO


def random_span_mask(
    rng: np.random.Generator,
    n_frames: int = N_FRAMES,
    n_spans: tuple[int, int] = (1, 2),
    span_ms: tuple[float, float] = (10.0, 100.0),
) -> np.ndarray:
    """Boolean ``(n_frames,)`` mask, True = missing. Spans may merge if they overlap."""
    lo = max(1, int(round(span_ms[0] / 1000 / FRAME_S)))
    hi = max(lo, int(round(span_ms[1] / 1000 / FRAME_S)))
    mask = np.zeros(n_frames, dtype=bool)
    for _ in range(rng.integers(n_spans[0], n_spans[1] + 1)):
        length = int(rng.integers(lo, hi + 1))
        start = int(rng.integers(0, n_frames - length + 1))
        mask[start : start + length] = True
    return mask


def apply_mask(spec: np.ndarray, time_mask: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace masked time frames of a ``(..., N_BINS, N_FRAMES)`` spectrogram with ``fill``."""
    out = np.array(spec, copy=True)
    out[..., time_mask] = fill
    return out


def idle_level_db(x: np.ndarray, intervals: pd.DataFrame, guard_s: float = 0.2) -> float:
    """Median frame RMS (dBFS) of the recording outside the cutting paths (+/- ``guard_s``)."""
    t, r = frame_rms(x.astype(np.float32) / 32768.0)
    idle = np.ones(len(t), dtype=bool)
    for _, row in intervals.iterrows():
        idle &= ~((t >= row["start"] - guard_s) & (t <= row["end"] + guard_s))
    return float(np.median(20 * np.log10(r[idle] + 1e-9)))
