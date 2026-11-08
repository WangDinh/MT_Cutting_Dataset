"""Integrity checks on the raw dataset.

Checks, per session:
  * three sensor WAVs exist, are mono 48 kHz int16, and have equal length;
  * cutting intervals are well-formed (start < end) and inside the recording;
  * consecutive intervals do not overlap (reported as a warning);
  * for IMI, cutting.csv matches the start/end columns of its label sheet.

Usage:
    python scripts/validate_dataset.py
Exit code is 1 if any error (not warning) is found.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.io import wavfile

from mtcut import SAMPLE_RATE, discover_sessions, read_labels


def main() -> int:
    sessions = discover_sessions()
    labels = read_labels()
    errors: list[str] = []
    warnings: list[str] = []

    for s in sessions:
        tag = s.id
        if s.sensors != [0, 1, 2]:
            errors.append(f"{tag}: expected sensors [0, 1, 2], found {s.sensors}")

        lengths = set()
        for sensor, path in s.wav_paths.items():
            sr, x = wavfile.read(path, mmap=True)
            if sr != SAMPLE_RATE or x.ndim != 1 or x.dtype != np.int16:
                errors.append(f"{tag} S{sensor}: sr={sr} ndim={x.ndim} dtype={x.dtype}")
            lengths.add(x.shape[0])
        if len(lengths) != 1:
            errors.append(f"{tag}: sensor lengths differ {sorted(lengths)}")

        iv = s.intervals()
        if (iv["end"] <= iv["start"]).any():
            errors.append(f"{tag}: {int((iv['end'] <= iv['start']).sum())} interval(s) with end <= start")
        if iv["end"].max() > s.duration:
            errors.append(f"{tag}: interval ends at {iv['end'].max():.2f}s > recording {s.duration:.2f}s")
        overlap = np.flatnonzero(iv["start"].to_numpy()[1:] < iv["end"].to_numpy()[:-1])
        for i in overlap:
            warnings.append(
                f"{tag}: paths {i} and {i + 1} overlap by "
                f"{iv['end'].iloc[i] - iv['start'].iloc[i + 1]:.3f}s"
            )

        if s.dataset == "IMI":
            lab = labels[labels["experiment"] == s.name]
            if len(lab) == 0:
                errors.append(f"{tag}: no label sheet")
            elif len(lab) != len(iv) or not np.allclose(
                lab[["start", "end"]].to_numpy(), iv[["start", "end"]].to_numpy()
            ):
                errors.append(f"{tag}: cutting.csv and label sheet disagree ({len(iv)} vs {len(lab)} rows)")

    print(f"Checked {len(sessions)} sessions.")
    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
