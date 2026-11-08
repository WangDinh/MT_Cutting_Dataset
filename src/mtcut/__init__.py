"""mtcut: tooling for the CNC milling cutting-sound (chatter) dataset."""

from .config import CHATTER_LABELS, SAMPLE_RATE, SENSORS
from .dataset import Session, discover_sessions, get_session, segment_table, session_table
from .io import read_cutting_csv, read_labels, read_wav

__all__ = [
    "CHATTER_LABELS",
    "SAMPLE_RATE",
    "SENSORS",
    "Session",
    "discover_sessions",
    "get_session",
    "read_cutting_csv",
    "read_labels",
    "read_wav",
    "segment_table",
    "session_table",
]

__version__ = "0.1.0"
