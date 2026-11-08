"""Score inpainting methods on the fixed masks of the held-out splits.

Methods: ``mean`` / ``linear`` (classical, see mtcut.inpaint) and ``diffusion`` (RePaint with the trained
DDPM). The diffusion model is stochastic, so ``diffusion`` is scored on one sample (the first of ``--k``, what
you get from a single run); the other samples are only saved, to show that the gap is filled differently
each time.

Writes to data/processed/masked/: inpaint_metrics.csv (one row per patch x method) and
inpaint_preds.npz (predictions in dB, float16, keyed ``{split}_{method}``; ``{split}_diffusion_samples``
holds all k samples).

Usage:
    python scripts/eval_inpainting.py [--splits test] [--k 3] [--resample 2] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# torch and Anaconda's numpy/scipy each bundle an OpenMP runtime; without this, the baselines abort on Windows.
# The numpy/scipy work here is single-threaded and light, and the baseline numbers are cross-checked in tests.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from mtcut.config import PROCESSED_DIR  # noqa: E402
from mtcut.diffusion import Diffusion, UNet, inpaint  # noqa: E402
from mtcut.inpaint import (  # noqa: E402
    BASELINES,
    LOW_BAND_BINS,
    MASKED_DIR,
    gap_stats,
    load_patches,
    longest_span_ms,
    masked_errors,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--ckpt", type=Path, default=PROCESSED_DIR / "diffusion" / "ckpt.pt")
    ap.add_argument("--k", type=int, default=3, help="diffusion samples per patch (the first is scored)")
    ap.add_argument("--resample", type=int, default=2, help="RePaint resampling steps per timestep")
    ap.add_argument("--limit", type=int, default=None, help="only the first N patches per split (quick test)")
    ap.add_argument("--out", type=Path, default=MASKED_DIR)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev)
    model = UNet(base=ck["args"]["base"]).to(dev).eval()
    model.load_state_dict(ck["ema"])
    diff = Diffusion(ck["args"]["T"], dev)
    print(f"checkpoint: step {ck['step']}, T={diff.T}, base={ck['args']['base']}, device {dev}")

    rows, preds = [], {}
    for split in args.splits:
        clean, mask, meta = load_patches(split)
        if args.limit:
            clean, mask, meta = clean[:args.limit], mask[:args.limit], meta.iloc[:args.limit]
        span = longest_span_ms(mask)
        observed = np.where(mask[:, None, :], 0.0, clean)  # the model/baselines never see the missing frames

        outputs = {name: fn(observed, mask) for name, fn in BASELINES.items()}
        t0 = time.time()
        samples = []
        for j in range(args.k):
            torch.manual_seed(j)
            samples.append(inpaint(model, diff, observed, mask, n_resample=args.resample))
            print(f"{split}: diffusion sample {j + 1}/{args.k}  ({time.time() - t0:.0f} s)", flush=True)
        outputs["diffusion"] = samples[0]
        preds[f"{split}_diffusion_samples"] = np.stack(samples).astype(np.float16)  # (k, N, bins, frames)

        for name, pred in outputs.items():
            preds[f"{split}_{name}"] = pred.astype(np.float16)
            full = masked_errors(pred, clean, mask)
            low = masked_errors(pred, clean, mask, bins=slice(0, LOW_BAND_BINS))
            gap = gap_stats(pred, clean, mask)
            df = pd.DataFrame({"split": split, "patch_idx": meta.index, "session": meta.session.values,
                               "method": name, "mae_db": full.mae_db, "rmse_db": full.rmse_db,
                               "mae_low_db": low.mae_db, "rmse_low_db": low.rmse_db, "profile_corr": gap.profile_corr, "std_ratio": gap.std_ratio,
                               "longest_span_ms": span, "masked_frac": mask.mean(axis=1)})
            rows.append(df)
        summ = pd.concat(rows)
        summ = summ[summ.split == split].groupby("method")[["mae_db", "rmse_db", "mae_low_db", "profile_corr", "std_ratio"]].mean().round(2)
        print(f"\n{split}:\n{summ}\n")

    pd.concat(rows).to_csv(args.out / "inpaint_metrics.csv", index=False)
    np.savez_compressed(args.out / "inpaint_preds.npz", **preds)
    print(f"wrote {args.out / 'inpaint_metrics.csv'} and inpaint_preds.npz")


if __name__ == "__main__":
    main()
