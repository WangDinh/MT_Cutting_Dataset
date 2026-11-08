"""Build spectrogram patches + fixed time-span masks for the inpainting proof of concept.

Takes one sensor (default 2, the least-clipped one), cuts every cutting path into
non-overlapping ~0.36 s windows, drops windows with clipped samples, and writes log-magnitude
spectrograms plus seeded masks. IMI only (KRPM is left out to keep the story simple). Splits are
by session. The session's idle level is stored per patch for reference but is not used to filter.

Writes to data/processed/masked/:
  * patches.npz - clean (N, 128, 64) float16 dB, mask (N, 64) bool, split (N,) str
  * patches.csv - one row per patch (segment_id, session, start_s, clip_frac, chatter labels)

Usage:
    python scripts/build_segment_index.py      # once, for the labels
    python scripts/make_masked_pairs.py [--sensor 2] [--max-clip 0.0] [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mtcut import discover_sessions, segment_table
from mtcut.config import PROCESSED_DIR, SAMPLE_RATE
from mtcut.features import rms_db
from mtcut.labels import consensus
from mtcut.masking import PATCH_SAMPLES, idle_level_db, patch_spectrogram_db, random_span_mask

FULL_SCALE = 32767


def assign_splits(imi_sessions: list[str], rng: np.random.Generator, n_val: int, n_test: int) -> dict[str, str]:
    order = list(rng.permutation(sorted(imi_sessions)))
    split = {s: "train" for s in order}
    for s in order[:n_val]:
        split[s] = "val"
    for s in order[n_val : n_val + n_test]:
        split[s] = "test"
    return split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", type=int, default=2)
    ap.add_argument("--max-clip", type=float, default=0.0, help="max fraction of full-scale samples per patch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=3)
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR / "masked")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sessions = discover_sessions()
    seg = segment_table(sessions)
    seg["chatter_max"] = consensus(seg["chatter_op1"], seg["chatter_op2"], "max")
    seg = seg.set_index(["session", "path_idx"])

    split_of = assign_splits([s.name for s in sessions if s.dataset == "IMI"], rng, args.n_val, args.n_test)

    specs, masks, rows = [], [], []
    for s in sessions:
        if s.dataset != "IMI":
            continue
        x = s.load(args.sensor, as_float=False)
        intervals = s.intervals()
        idle_db = idle_level_db(x, intervals)
        for path_idx, iv in intervals.iterrows():
            i0 = int(round(iv["start"] * SAMPLE_RATE))
            i1 = min(len(x), int(round(iv["end"] * SAMPLE_RATE)))
            lab = seg.loc[(s.name, path_idx)]
            for start in range(i0, i1 - PATCH_SAMPLES + 1, PATCH_SAMPLES):
                w = np.asarray(x[start : start + PATCH_SAMPLES])
                clip_frac = float(np.mean(np.abs(w.astype(np.int32)) >= FULL_SCALE))
                if clip_frac > args.max_clip:
                    continue
                level_db = rms_db(w.astype(np.float32) / 32768.0)
                specs.append(patch_spectrogram_db(w).astype(np.float16))
                masks.append(random_span_mask(rng))
                rows.append(
                    {
                        "segment_id": lab["segment_id"],
                        "dataset": s.dataset,
                        "session": s.name,
                        "path_idx": path_idx,
                        "start_s": start / SAMPLE_RATE,
                        "clip_frac": clip_frac,
                        "rms_db": level_db,
                        "idle_db": idle_db,
                        "split": split_of[s.name],
                        "chatter_max": lab["chatter_max"],
                    }
                )
        print(f"{s.id}: {sum(r['session'] == s.name for r in rows)} patches")

    meta = pd.DataFrame(rows)
    np.savez_compressed(
        args.out / "patches.npz",
        clean=np.stack(specs),
        mask=np.stack(masks),
        split=meta["split"].to_numpy().astype(str),
    )
    meta.to_csv(args.out / "patches.csv", index_label="patch_idx")
    print(f"\nWrote {len(meta)} patches to {args.out}")
    print(meta.groupby("split").agg(patches=("session", "size"), sessions=("session", "nunique")))
    print("sessions per split:", {k: sorted(v) for k, v in meta.groupby("split").session.unique().items()})


if __name__ == "__main__":
    main()
