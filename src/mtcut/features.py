"""Signal-processing helpers used for exploration and feature extraction."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .config import SAMPLE_RATE


def welch_psd(x: np.ndarray, sr: int = SAMPLE_RATE, nperseg: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density via Welch's method. Returns ``(freqs_hz, psd)``."""
    nperseg = min(nperseg, len(x))
    return signal.welch(x, fs=sr, nperseg=nperseg)


def spectrogram_db(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    nperseg: int = 2048,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT magnitude in dB. Returns ``(freqs_hz, times_s, S_db)``.

    The default ``nperseg=2048`` matches the MTConnect chunk size the data
    was streamed with (~42.7 ms at 48 kHz).
    """
    f, t, S = signal.spectrogram(x, fs=sr, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")
    return f, t, 10 * np.log10(S + 1e-12)


def frame_rms(x: np.ndarray, sr: int = SAMPLE_RATE, frame_s: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping frame RMS envelope. Returns ``(times_s, rms)``."""
    n = max(1, int(frame_s * sr))
    k = len(x) // n
    frames = np.asarray(x[: k * n], dtype=np.float32).reshape(k, n)
    rms = np.sqrt((frames**2).mean(axis=1))
    return (np.arange(k) + 0.5) * n / sr, rms


def rms_db(x: np.ndarray) -> float:
    """Overall RMS level in dBFS."""
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def tooth_passing_harmonics(tpf_hz: float, fmax: float = SAMPLE_RATE / 2) -> np.ndarray:
    """Frequencies of the tooth-passing frequency and its harmonics up to ``fmax``.

    Forced vibration in milling concentrates energy at these harmonics;
    chatter shows up as strong peaks *between* them.
    """
    if not np.isfinite(tpf_hz) or tpf_hz <= 0:
        return np.array([])
    return np.arange(1, int(fmax // tpf_hz) + 1) * tpf_hz


def harmonic_energy_ratio(
    freqs: np.ndarray,
    psd: np.ndarray,
    tpf_hz: float,
    fmin: float = 100.0,
    fmax: float = 10_000.0,
    rel_bw: float = 0.02,
) -> float:
    """Share of band energy lying *off* the tooth-passing harmonics.

    A simple physics-informed chatter indicator: values near 0 mean the
    spectrum is dominated by forced vibration; larger values mean energy
    at non-harmonic (chatter) frequencies.
    """
    band = (freqs >= fmin) & (freqs <= fmax)
    on = np.zeros_like(band)
    for h in tooth_passing_harmonics(tpf_hz, fmax):
        on |= np.abs(freqs - h) <= max(rel_bw * tpf_hz, freqs[1] - freqs[0])
    total = psd[band].sum()
    return float(psd[band & ~on].sum() / total) if total > 0 else np.nan
