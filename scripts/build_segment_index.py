"""Build flat metadata tables for downstream modelling.

Writes to data/processed/:
  * sessions.csv - one row per recording session
  * segments.csv - one row per cutting path (IMI rows carry cutting
    parameters, both operators' labels and consensus labels)

Usage:
    python scripts/build_segment_index.py [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mtcut import discover_sessions, segment_table, session_table
from mtcut.config import PROCESSED_DIR
from mtcut.labels import consensus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions()
    sess = session_table(sessions)
    seg = segment_table(sessions)
    for strategy in ("max", "mean", "agree"):
        seg[f"chatter_{strategy}"] = consensus(seg["chatter_op1"], seg["chatter_op2"], strategy)

    sess.to_csv(args.out / "sessions.csv", index=False)
    seg.to_csv(args.out / "segments.csv", index=False)
    print(f"Wrote {len(sess)} sessions and {len(seg)} segments to {args.out}")


if __name__ == "__main__":
    main()
