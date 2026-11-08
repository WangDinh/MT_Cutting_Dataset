"""Train the unconditional spectrogram DDPM on the training split (clean patches only).

Stops at ``--steps`` or ``--max-minutes`` (whichever comes first) and saves the EMA weights.
``--exclude-session`` leaves whole sessions out of the training set (used for the held-out folds of
the chatter-classification test, notebook 04, so the generator never sees the sessions it fills).

Writes to data/processed/diffusion/: ckpt.pt (EMA weights + config), train_log.csv (step, train/val loss).

Usage:
    python scripts/train_diffusion.py [--steps 20000] [--max-minutes 40] [--batch 64]
    python scripts/train_diffusion.py --exclude-session Exp3-2 --out data/processed/diffusion_fold_Exp3-2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mtcut.config import PROCESSED_DIR
from mtcut.diffusion import EMA, Diffusion, UNet, to_tensor
from mtcut.inpaint import load_patches

GAIN_JITTER_DB = 3.0
DB_RANGE = 100.0  # DB_HI - DB_LO; one dB is 2/DB_RANGE in normalised units


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random time flip and +/- 3 dB gain jitter (the silent floor stays at the floor)."""
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    shift = (torch.rand(len(x), 1, 1, 1, device=x.device) * 2 - 1) * GAIN_JITTER_DB * 2 / DB_RANGE
    return torch.where(x > -0.999, (x + shift).clamp(-1, 1), x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=int, default=250)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-session", nargs="*", default=[], help="sessions to drop from the training set")
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR / "diffusion")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    clean, _, meta = load_patches("train")
    keep = ~meta.session.isin(args.exclude_session).to_numpy()
    train = to_tensor(clean[keep], dev)
    val = to_tensor(load_patches("val")[0], dev)
    print(f"train {tuple(train.shape)}, val {tuple(val.shape)}, device {dev}")

    model = UNet(base=args.base).to(dev)
    ema = EMA(model)
    diff = Diffusion(args.T, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))
    print(f"UNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    g = torch.Generator(device=dev).manual_seed(123)  # fixed val noise/timesteps -> comparable val loss
    val_t = torch.randint(0, args.T, (len(val),), device=dev, generator=g)
    val_n = torch.randn(val.shape, device=dev, generator=g)

    log, t0, run = [], time.time(), 0.0
    for step in range(1, args.steps + 1):
        x = augment(train[torch.randint(0, len(train), (args.batch,), device=dev)])
        loss = diff.loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); ema.update(model)
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()

        elapsed = (time.time() - t0) / 60
        if step % 500 == 0 or step == args.steps or elapsed > args.max_minutes:
            with torch.no_grad():
                vl = diff.loss(ema.model, val, val_t, val_n).item()
            log.append({"step": step, "train_loss": run, "val_loss_ema": vl, "minutes": elapsed})
            print(f"step {step:6d}  train {run:.4f}  val(ema) {vl:.4f}  {elapsed:5.1f} min", flush=True)
            pd.DataFrame(log).to_csv(args.out / "train_log.csv", index=False)
            torch.save({"ema": ema.model.state_dict(), "args": vars(args), "step": step}, args.out / "ckpt.pt")
        if elapsed > args.max_minutes:
            print(f"time budget of {args.max_minutes} min reached at step {step}")
            break


if __name__ == "__main__":
    main()
