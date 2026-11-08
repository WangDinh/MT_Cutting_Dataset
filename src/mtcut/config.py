"""Project-wide paths and dataset constants."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The raw dataset lives at the repository root (IMI/, KRPM/) because the
# repository is also the published dataset. Override with MTCUT_DATA_DIR to
# point at a copy elsewhere.
DATA_ROOT = Path(os.environ.get("MTCUT_DATA_DIR", PROJECT_ROOT))
IMI_DIR = DATA_ROOT / "IMI"
KRPM_DIR = DATA_ROOT / "KRPM"
LABELS_XLSX = IMI_DIR / "labeling_all_details.xlsx"

# Derived artefacts (git-ignored).
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SAMPLE_RATE = 48_000
BIT_DEPTH = 16

# Sensor index -> description. Same physical layout on both machines.
SENSORS = {
    0: "Microphone (working area)",
    1: "ISS (spindle)",
    2: "ISS (base)",
}

# Operator chatter annotation scale (IMI only).
CHATTER_LABELS = {0: "normal", 1: "uncertain", 2: "chatter"}

MACHINES = {
    "IMI": "Hurco VM20i (3-axis vertical, lab)",
    "KRPM": "Mazak HCN-6000 (4-axis horizontal, industry)",
}
