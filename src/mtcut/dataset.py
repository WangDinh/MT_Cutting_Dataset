"""Session discovery and dataset-level tables.

A *session* is one continuous recording: three synchronized WAVs (sensor
0/1/2), a ``cutting.csv`` of cutting-path intervals, and optionally a video.
IMI has one session per ``IMI/Exp*`` folder; KRPM is a single session.
A *segment* is one cutting path (one row of ``cutting.csv``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .config import IMI_DIR, KRPM_DIR, LABELS_XLSX, SAMPLE_RATE
from .io import read_cutting_csv, read_labels, read_wav, wav_num_frames

# IMI: ..._VM20i_sensor0.wav   KRPM: ..._Mazak_sound0.wav
_WAV_RE = re.compile(r"^(?P<ts>\d{8}_\d{6}\.\d+)Z?_.+_(?:sensor|sound)(?P<sensor>\d)\.wav$")


@dataclass(frozen=True)
class Session:
    name: str
    dataset: str
    directory: Path
    wav_paths: dict[int, Path]
    cutting_csv: Path
    video: Path | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        return f"Session({self.dataset}/{self.name}, sensors={sorted(self.wav_paths)})"

    @property
    def id(self) -> str:
        return f"{self.dataset}/{self.name}"

    @property
    def sensors(self) -> list[int]:
        return sorted(self.wav_paths)

    @property
    def recorded_at(self) -> datetime | None:
        m = _WAV_RE.match(self.wav_paths[self.sensors[0]].name)
        return datetime.strptime(m["ts"], "%Y%m%d_%H%M%S.%f") if m else None

    @cached_property
    def num_frames(self) -> int:
        return wav_num_frames(self.wav_paths[self.sensors[0]])

    @property
    def duration(self) -> float:
        return self.num_frames / SAMPLE_RATE

    def intervals(self) -> pd.DataFrame:
        """Cutting-path intervals (seconds) from ``cutting.csv``."""
        return read_cutting_csv(self.cutting_csv)

    def load(self, sensor: int, *, as_float: bool = True) -> np.ndarray:
        """Full-length signal of one sensor."""
        sr, x = read_wav(self.wav_paths[sensor], as_float=as_float)
        assert sr == SAMPLE_RATE, f"{self.wav_paths[sensor]}: unexpected sample rate {sr}"
        return x

    def load_all(self, *, as_float: bool = True) -> np.ndarray:
        """All sensors stacked as ``(n_sensors, n_samples)``."""
        return np.stack([self.load(s, as_float=as_float) for s in self.sensors])

    def segment(self, sensor: int, path_idx: int, *, pad: float = 0.0) -> np.ndarray:
        """Samples of one cutting path, optionally padded by ``pad`` seconds each side."""
        row = self.intervals().iloc[path_idx]
        return self.slice(sensor, row["start"] - pad, row["end"] + pad)

    def slice(self, sensor: int, start: float, end: float) -> np.ndarray:
        """Samples between ``start`` and ``end`` seconds (clipped to the recording)."""
        x = self.load(sensor)
        i0 = max(0, int(round(start * SAMPLE_RATE)))
        i1 = min(len(x), int(round(end * SAMPLE_RATE)))
        return np.asarray(x[i0:i1])


def _session_from_dir(directory: Path, dataset: str, name: str) -> Session:
    wavs: dict[int, Path] = {}
    for p in sorted(directory.glob("*.wav")):
        m = _WAV_RE.match(p.name)
        if m:
            wavs[int(m["sensor"])] = p
    videos = sorted(directory.glob("*.mp4"))
    # Empty .txt files are used as annotations: their *name* is the note.
    notes = tuple(p.stem.replace("_", " ") for p in sorted(directory.glob("*.txt")))
    return Session(
        name=name,
        dataset=dataset,
        directory=directory,
        wav_paths=wavs,
        cutting_csv=directory / "cutting.csv",
        video=videos[0] if videos else None,
        notes=notes,
    )


def discover_sessions(imi_dir: Path = IMI_DIR, krpm_dir: Path = KRPM_DIR) -> list[Session]:
    """All recording sessions: the IMI experiments (sorted) followed by KRPM."""
    sessions = [
        _session_from_dir(d, "IMI", d.name)
        for d in sorted(imi_dir.iterdir())
        if d.is_dir() and d.name.startswith("Exp")
    ]
    if krpm_dir.exists():
        sessions.append(_session_from_dir(krpm_dir, "KRPM", "KRPM"))
    return sessions


def get_session(name: str, sessions: list[Session] | None = None) -> Session:
    """Look up a session by name, e.g. ``"Exp1-2"`` or ``"KRPM"``."""
    for s in sessions or discover_sessions():
        if s.name == name or s.id == name:
            return s
    raise KeyError(name)


def session_table(sessions: list[Session] | None = None) -> pd.DataFrame:
    """One row per session: duration, number of cutting paths, cutting time, video, notes."""
    rows = []
    for s in sessions or discover_sessions():
        iv = s.intervals()
        rows.append(
            {
                "dataset": s.dataset,
                "session": s.name,
                "recorded_at": s.recorded_at,
                "n_sensors": len(s.wav_paths),
                "duration_s": s.duration,
                "n_paths": len(iv),
                "cutting_s": iv["duration"].sum(),
                "cutting_frac": iv["duration"].sum() / s.duration,
                "has_video": s.video is not None,
                "notes": "; ".join(s.notes),
            }
        )
    return pd.DataFrame(rows)


def segment_table(
    sessions: list[Session] | None = None,
    labels_xlsx: Path | None = LABELS_XLSX,
) -> pd.DataFrame:
    """One row per cutting path across both datasets.

    IMI rows are joined with the label workbook on ``(session, path_idx)``;
    KRPM rows have NaN in the label / cutting-parameter columns.
    """
    sessions = sessions or discover_sessions()
    frames = []
    for s in sessions:
        iv = s.intervals().reset_index()
        iv.insert(0, "dataset", s.dataset)
        iv.insert(1, "session", s.name)
        frames.append(iv)
    seg = pd.concat(frames, ignore_index=True)
    seg.insert(0, "segment_id", seg["dataset"] + "/" + seg["session"] + "/" + seg["path_idx"].map("{:03d}".format))

    if labels_xlsx is not None:
        labels = read_labels(labels_xlsx).rename(columns={"experiment": "session"})
        labels = labels.drop(columns=["start", "end", "duration"])
        seg = seg.merge(labels, on=["session", "path_idx"], how="left")
    return seg
