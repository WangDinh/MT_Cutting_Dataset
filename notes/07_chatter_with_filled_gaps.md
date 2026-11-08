# Chatter classification on filled gaps

Question: if a patch has a gap, does filling it let a normal-vs-chatter classifier still tell the cut type? This replaces "how close is the fill" (note 05) with the original task. Notebook: `notebooks/04_chatter_with_filled_gaps.ipynb`. Code: `src/mtcut/chatter.py`, `scripts/eval_chatter_fills.py`, `tests/test_chatter.py`.

## Why a separate setup from note 05
Chatter patches exist almost only in three sessions (`Exp3-2`: 314 chatter / 20 normal; `Exp4-3`: 272 chatter / 0 normal; `Exp4-3-2`: 287 chatter / 18 normal). The note 05 test and val sessions have no chatter, so they cannot score a classifier. Only `Exp3-2` and `Exp4-3-2` have both classes, so they are the held-out folds.

## Design
- **Labels:** normal (0) vs chatter (2), uncertain (1) dropped, stricter-of-two-operators (`max`).
- **Features:** per-frequency mean and temporal std of the dB patch (256 numbers).
- **Classifier:** logistic regression (L2, standardised features), trained on the clean patches of every other session. I first also ran gradient boosting and kept logistic because it is simpler and had the higher clean AUC (0.999 vs 0.955 on `Exp4-3-2`); the AUC conclusion was the same with both.
- **Inputs:** the held-out patches with the same fixed gaps as note 05 (about 23% of frames), filled by mean / linear / diffusion. References: `clean` and `gap_left` (gap frames left at the floor).
- **Diffusion:** one sample per patch, from a model trained without the held-out session (`train_diffusion.py --exclude-session`; 22 min, about 5.9k steps each, in `data/processed/diffusion_fold_<session>/`). The note 05 model saw these sessions in training, so it was not used.
- **Score:** ROC AUC with a stratified bootstrap 95% interval; balanced accuracy, chatter recall and normal recall at the fixed 0.5 threshold.

## Result: same ranking quality, but the fills shift the decision threshold
| | clean | gap_left | mean | linear | diffusion |
|---|---|---|---|---|---|
| AUC, Exp3-2 | 0.999 | 0.841 | 1.000 | 0.999 | 0.998 |
| AUC, Exp4-3-2 | 0.999 | 0.678 | 1.000 | 1.000 | 0.998 |
| Balanced accuracy, Exp3-2 | 0.978 | 0.500 | 0.906 | 0.954 | 0.992 |
| Balanced accuracy, Exp4-3-2 | 0.806 | 0.500 | 0.889 | 0.917 | 0.778 |

- AUC: all three fills are within 0.002 of clean. Only the empty gap hurts.
- Accuracy does not show diffusion as better: it is best on Exp3-2 and worst on Exp4-3-2, and on Exp3-2 it beats the clean patches (0.992 vs 0.978), which a fill cannot truly do. This is a threshold effect.
- Mean-fill and linear flatten the gap (lower std features), so they call fewer patches chatter (chatter recall 0.81 and 0.91 on Exp3-2). Diffusion adds texture, so it calls more (chatter recall 0.98, but normal recall 0.56 on Exp4-3-2). Which helps depends on which error the clean classifier already makes (on Exp4-3-2 it has normal recall 0.61).
- An empty gap puts floor values into the features and inflates the std, so every patch looks like chatter (chatter recall 1.00, normal recall 0.00). Mean-fill is the fair no-effort baseline.
- On individual long-gap patches the probability can move a lot (mean-fill: chatter 0.79 to 0.24; diffusion: normal 0.65 to 0.93), but not on average.

## Caveats
- Ceiling effect: clean AUC is already 0.999, so a better fill has no room to show a gain in ranking, and averages cannot see texture.
- Two folds, 20 and 18 normal patches (one patch moves normal recall by 5 points); the accuracy interval ignores the threshold shift; chatter is tied to the session; loudest chatter excluded (clipping); one mask and one diffusion sample per patch.

## Not yet tested
Heavier masks (60-80% missing), a classifier that uses temporal structure (small CNN), and filled patches as training data (augmentation).
