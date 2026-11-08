"""Plotting helpers shared by notebooks and scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CHATTER_LABELS, SAMPLE_RATE, SENSORS
from .dataset import Session
from .features import frame_rms, spectrogram_db

LABEL_COLORS = {0: "#2a9d8f", 1: "#e9c46a", 2: "#e76f51", None: "#8d99ae"}
SENSOR_COLORS = {0: "#264653", 1: "#e76f51", 2: "#2a9d8f"}


def label_legend_handles():
    from matplotlib.patches import Patch

    return [Patch(color=LABEL_COLORS[k], alpha=0.35, label=f"{k}: {v}") for k, v in CHATTER_LABELS.items()]


def shade_intervals(ax, segments: pd.DataFrame, label_col: str | None = None, alpha: float = 0.25) -> None:
    """Shade each ``start``-``end`` span, coloured by ``label_col`` if given."""
    for _, row in segments.iterrows():
        lab = row.get(label_col) if label_col else None
        lab = None if lab is None or pd.isna(lab) else int(lab)
        ax.axvspan(row["start"], row["end"], color=LABEL_COLORS[lab], alpha=alpha, lw=0)


def plot_session_overview(
    session: Session,
    segments: pd.DataFrame | None = None,
    label_col: str | None = None,
    frame_s: float = 0.05,
    figsize=(14, 6),
):
    """RMS envelope of every sensor for a whole session with cutting paths shaded."""
    if segments is None:
        segments = session.intervals()
    fig, axes = plt.subplots(len(session.sensors), 1, figsize=figsize, sharex=True)
    for ax, s in zip(np.atleast_1d(axes), session.sensors):
        t, rms = frame_rms(session.load(s), frame_s=frame_s)
        shade_intervals(ax, segments, label_col)
        ax.plot(t, 20 * np.log10(rms + 1e-9), lw=0.7, color=SENSOR_COLORS.get(s, "k"))
        ax.set_ylabel(f"S{s} dBFS")
        ax.set_title(SENSORS.get(s, f"sensor {s}"), loc="left", fontsize=9)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{session.id}: frame RMS ({frame_s * 1000:.0f} ms) with cutting paths shaded")
    if label_col:
        axes[0].legend(handles=label_legend_handles(), loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def plot_spectrogram(
    ax,
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    t0: float = 0.0,
    fmax: float = 12_000,
    nperseg: int = 2048,
    vmin_pct: float = 5,
    cmap: str = "magma",
):
    """Draw a dB spectrogram of ``x`` on ``ax`` (time axis offset by ``t0``)."""
    f, t, S = spectrogram_db(x, sr, nperseg=nperseg)
    keep = f <= fmax
    S = S[keep]
    mesh = ax.pcolormesh(
        t + t0, f[keep] / 1000, S, shading="auto", cmap=cmap,
        vmin=np.percentile(S, vmin_pct), vmax=S.max(),
    )
    ax.set_ylabel("freq [kHz]")
    return mesh
