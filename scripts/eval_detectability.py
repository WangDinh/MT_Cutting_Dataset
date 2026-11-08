"""Score how easily a detector tells filled gaps from real signal (see mtcut.detect), on the test split.

Reads the predictions written by ``eval_inpainting.py`` and writes data/processed/masked/detectability.csv
(one row per method: pooled AUC plus one column per held-out session).

Usage:
    python scripts/eval_inpainting.py     # once, for the predictions
    python scripts/eval_detectability.py [--methods mean linear diffusion]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mtcut.detect import loso_detectability
from mtcut.inpaint import MASKED_DIR, load_patches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["mean", "linear", "diffusion"])
    args = ap.parse_args()

    clean, mask, meta = load_patches("test")
    preds = np.load(MASKED_DIR / "inpaint_preds.npz")
    sessions = meta.session.to_numpy()
    rows = []
    for name in args.methods:
        res = loso_detectability(clean, preds[f"test_{name}"].astype(np.float32), mask, sessions)
        rows.append({"method": name, **res})
        print(name, round(res["auc"], 3), flush=True)
    pd.DataFrame(rows).to_csv(MASKED_DIR / "detectability.csv", index=False)
    print(f"wrote {MASKED_DIR / 'detectability.csv'}")


if __name__ == "__main__":
    main()
