# Results and next steps

Current numbers are from the second run: IMI only, no loudness gate, 1461 patches (see note 04). Notebook: `notebooks/03_inpainting_baselines_vs_diffusion.ipynb`. The first run (with the 3 dB gate and KRPM) is backed up in `data/processed/diffusion_old/`; its numbers were within 0.1 dB of these.

## Training
35.0 min, 8643 steps, on an RTX 4060 Laptop. Validation loss flattened at about 0.046 from step 4500 with no rise. (In the first run a launch survived as a duplicate process and competed for the GPU; check for leftovers before relaunching.)

## Headline: mean-fill beats diffusion on accuracy
0-3 kHz MAE in dB, lower is better, on the test split (3 held-out IMI sessions, 144 patches):

| mean | linear | diffusion (1 sample) |
|---|---|---|
| **3.75** | 4.34 | 5.30 |

- Mean-fill wins in all 3 test sessions and in every gap-length bucket, including gaps over 100 ms.
- Linear is close for gaps under 30 ms (3.82 vs 3.75) and falls behind as gaps grow.

## Where diffusion wins: texture
`std_ratio` is 0.88 for diffusion (1 sample), against 0 for mean-fill and 0.40 for linear. `profile_corr` is 0.97 for diffusion against 0.99 for mean-fill.

## Detectability: can you tell the gap was filled?
MAE misses what the eye sees (mean-fill and linear gaps are obvious smooth blocks). A logistic regression is trained to tell the real frame from the filled frame of each missing frame (it is told where the gap is; leave-one-session-out over the 3 test sessions). AUC: 0.5 = cannot tell, 1.0 = every filled frame spotted. Code: `src/mtcut/detect.py`, `scripts/eval_detectability.py`; shown in notebook 03 section 7.

| Method | detector AUC |
|---|---|
| mean | 0.98 |
| linear | 1.00 |
| diffusion (1 sample) | **0.76** |

- Diffusion with a single sample is the hardest to spot, but not undetectable. A stronger detector finds more: in a trial a small CNN on a 48 ms window gave 0.83 for it (and an earlier gradient-boosting detector 0.72), so read these as lower bounds. The CNN was dropped from the repo to keep one simple detector.
- Per-session spread is large for diffusion (0.87 on Exp0-1, 0.67 on Exp1-1-A, 0.79 on Exp1-8).
- Mean-fill and linear score about 1.0 trivially: their gaps are exactly flat or smooth.

## Why
Within 0.36 s the cutting sound is close to stationary: harmonic stripes plus random noise. The mean of the observed frames already predicts the gap well. Fine texture is noise-like and cannot be matched frame by frame, and error metrics punish realistic-but-different texture.

## Things that did not change the conclusion
- Removing the 3 dB loudness gate and retraining: all numbers moved by under 0.1 dB.
- Gap-length table: it first stopped at 101 ms and silently dropped 37 of 250 patches (merged overlapping dropouts reach about 200 ms). Fixed with an open-ended last bucket.

## Caveats
- Three test sessions, all normal (no chatter), and `Exp0-1` supplies 103 of 144 patches. The ranking holds per session, but sizes are rough.
- One small, short training run; no tuning of architecture, T or learning rate.
- Masks are an easy setting (stationary signal, short gaps). This does not show diffusion loses on harder cases.
- The loudest chatter cuts are excluded because they clip on S2.

## Not yet tested (where a generative model could earn its cost)
1. Longer gaps (200 ms and more) or longer context windows.
2. Non-stationary events (tool entry/exit, chatter onset) where an average is wrong.
3. Real clipped samples (declipping), the original motivation.
4. A downstream test: does diffusion-filled data improve a chatter classifier compared with mean-fill? Done in a first form: see [07_chatter_with_filled_gaps.md](07_chatter_with_filled_gaps.md) (no difference; the task is at its ceiling).
5. KRPM as an out-of-domain test (first run: diffusion 5.69 vs mean 4.37 dB; not repeated).
