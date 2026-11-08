"""Does filling gaps keep the chatter evidence? Normal-vs-chatter classification on filled patches.

For each held-out session that has both classes (``Exp3-2``, ``Exp4-3-2``):
  1. train a logistic-regression classifier (see mtcut.chatter) on the CLEAN patches of every other session;
  2. take the held-out session's patches with their fixed masks and fill the gaps with mean / linear /
     diffusion (a model trained WITHOUT that session, see ``train_diffusion.py --exclude-session``);
  3. score each version of the patches with it.
Reference rows: ``clean`` (no gap) and ``gap_left`` (masked frames left at the floor, nothing filled).

Writes to data/processed/chatter/: scores.csv (one row per fold x input x patch) and
fills_{session}.npz (the filled patches, for figures).

Usage:
    python scripts/eval_chatter_fills.py [--sessions Exp3-2 Exp4-3-2] [--skip-diffusion]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# torch and Anaconda's numpy/scipy each bundle an OpenMP runtime; see scripts/eval_inpainting.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mtcut.chatter import binary_labels, make_classifier, patch_features  # noqa: E402
from mtcut.config import PROCESSED_DIR  # noqa: E402
from mtcut.inpaint import BASELINES, load_patches  # noqa: E402
from mtcut.masking import DB_LO  # noqa: E402


def diffusion_fill(session: str, observed: np.ndarray, mask: np.ndarray, resample: int) -> np.ndarray:
    import torch

    from mtcut.diffusion import Diffusion, UNet, inpaint

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(PROCESSED_DIR / f"diffusion_fold_{session}" / "ckpt.pt", map_location=dev)
    assert session in ck["args"]["exclude_session"], "fold model was not trained without this session"
    model = UNet(base=ck["args"]["base"]).to(dev).eval()
    model.load_state_dict(ck["ema"])
    diff = Diffusion(ck["args"]["T"], dev)
    torch.manual_seed(0)
    print(f"{session}: diffusion model step {ck['step']}", flush=True)
    return inpaint(model, diff, observed, mask, n_resample=resample)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", default=["Exp3-2", "Exp4-3-2"])
    ap.add_argument("--resample", type=int, default=2)
    ap.add_argument("--skip-diffusion", action="store_true")
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR / "chatter")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    clean, mask, meta = load_patches()
    keep_all, _ = binary_labels(meta.chatter_max)
    rows = []
    for held in args.sessions:
        is_held = (meta.session == held).to_numpy()
        tr = keep_all & ~is_held
        te = keep_all & is_held
        y_tr = (meta.chatter_max[tr] == 2).astype(int).to_numpy()
        y_te = (meta.chatter_max[te] == 2).astype(int).to_numpy()
        print(f"{held}: train {len(y_tr)} patches ({y_tr.sum()} chatter), test {len(y_te)} ({y_te.sum()} chatter)")

        x_clean, m_te = clean[te], mask[te]
        observed = np.where(m_te[:, None, :], 0.0, x_clean)
        inputs = {"clean": x_clean, "gap_left": np.where(m_te[:, None, :], DB_LO, x_clean)}
        for name, fn in BASELINES.items():
            inputs[name] = fn(observed, m_te)
        if not args.skip_diffusion:
            t0 = time.time()
            inputs["diffusion"] = diffusion_fill(held, observed, m_te, args.resample)
            print(f"{held}: diffusion fill took {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(args.out / f"fills_{held}.npz", patch_idx=meta.index[te].to_numpy(), mask=m_te,
                            y=y_te, **{k: v.astype(np.float16) for k, v in inputs.items()})

        clf = make_classifier().fit(patch_features(clean[tr]), y_tr)
        for iname, x in inputs.items():
            score = clf.predict_proba(patch_features(x))[:, 1]
            rows.append(pd.DataFrame({"fold": held, "input": iname, "patch_idx": meta.index[te].to_numpy(),
                                      "y": y_te, "score": score}))
    pd.concat(rows).to_csv(args.out / "scores.csv", index=False)
    print(f"wrote {args.out / 'scores.csv'}")


if __name__ == "__main__":
    main()
