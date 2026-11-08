"""Low-level readers for the raw files (WAV, cutting.csv, label workbook).

The raw files have a few inconsistencies that these readers absorb so the
rest of the code does not have to:

* ``cutting.csv`` is usually UTF-8 *with* a BOM, but not always
  (``IMI/Exp1-4``), and the header is ``start,end`` in most folders but
  ``Start,End`` in ``IMI/Exp3-2`` and ``IMI/Exp4-3``.
* The label workbook spells cutting direction as ``Up``/``Down`` and
  occasionally ``Up (conventional)``/``Down (climb)``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from .config import LABELS_XLSX

_LABEL_COLUMNS = {
    "No.": "path_no",
    "Machine": "machine",
    "Tool": "tool",
    "Tool setup": "tool_setup",
    "Tool Cond.": "tool_condition",
    "Workpiece": "workpiece",
    "Spindle speed [RPM]": "spindle_rpm",
    "Feedrate [IPM]": "feed_ipm",
    "Cutting direction": "direction",
    "DOC [in.]": "doc_in",
    "WOC [in.]": "woc_in",
    "start": "start",
    "end": "end",
    "Chatter (operator 1)": "chatter_op1",
    "Chatter (operator 2)": "chatter_op2",
}

_TOOL_RE = re.compile(r'(?P<diam>\d+/\d+|\d*\.?\d+)"\s*(?P<flutes>\d+)-flute', re.I)


def read_wav(path: str | Path, *, mmap: bool = True, as_float: bool = True) -> tuple[int, np.ndarray]:
    """Read a mono WAV file.

    Returns ``(sample_rate, samples)``. With ``as_float`` the int16 samples
    are scaled to float32 in [-1, 1); otherwise the raw int16 array is
    returned (memory-mapped when ``mmap`` is true, which is cheap for the
    multi-minute IMI recordings).
    """
    sr, x = wavfile.read(path, mmap=mmap)
    if as_float:
        x = x.astype(np.float32) / 32768.0
    return sr, x


def wav_num_frames(path: str | Path) -> int:
    """Number of samples in a WAV file without reading the data."""
    _, x = wavfile.read(path, mmap=True)
    return int(x.shape[0])


def read_cutting_csv(path: str | Path) -> pd.DataFrame:
    """Read a ``cutting.csv`` into a ``start``/``end``/``duration`` frame (seconds)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["start", "end"]].astype(float)
    df["duration"] = df["end"] - df["start"]
    df.index.name = "path_idx"
    return df


def _parse_tool(tool: str) -> tuple[float, int]:
    m = _TOOL_RE.search(str(tool))
    if not m:
        return np.nan, -1
    num, _, den = m["diam"].partition("/")
    diam = float(num) / float(den) if den else float(num)
    return diam, int(m["flutes"])


def read_labels(path: str | Path = LABELS_XLSX) -> pd.DataFrame:
    """Read every per-experiment sheet of ``labeling_all_details.xlsx``.

    Returns one row per cutting path with snake_case columns, an
    ``experiment`` column (matching the IMI folder name), a 0-based
    ``path_idx`` aligned with ``cutting.csv`` rows, and parsed tool geometry
    (``tool_diameter_in``, ``flutes``) plus the tooth-passing frequency.
    """
    sheets = pd.read_excel(path, sheet_name=None)
    frames = []
    for name, df in sheets.items():
        if name == "Summary":
            continue
        df = df.rename(columns=_LABEL_COLUMNS)
        df.insert(0, "experiment", name)
        df.insert(1, "path_idx", np.arange(len(df)))
        frames.append(df)
    labels = pd.concat(frames, ignore_index=True)

    labels["machine"] = labels["machine"].ffill()
    labels["direction"] = (
        labels["direction"].str.strip().str.split().str[0].str.lower()
    )
    geom = labels["tool"].map(_parse_tool)
    labels["tool_diameter_in"] = geom.str[0]
    labels["flutes"] = geom.str[1].astype(int)
    labels["tooth_passing_hz"] = labels["spindle_rpm"] / 60.0 * labels["flutes"]
    labels["duration"] = labels["end"] - labels["start"]
    return labels


def read_label_summary(path: str | Path = LABELS_XLSX) -> pd.DataFrame:
    """The ``Summary`` sheet (experiment-level setup; lists 15 of 18 experiments)."""
    df = pd.read_excel(path, sheet_name="Summary")
    df = df[df["Experiment"].astype(str).str.startswith("Exp")]
    df["Machine"] = df["Machine"].ffill()
    return df.reset_index(drop=True)
