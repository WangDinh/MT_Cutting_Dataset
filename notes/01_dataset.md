# The dataset

CNC milling cutting-sound dataset. It accompanies two papers:
- Kim et al., Manufacturing Letters, 2024: chatter detection.
- Mostafiz et al., INDIN 2024: lab-to-industry domain adaptation.

The public layout (`IMI/`, `KRPM/`, `figure/`, `README.md`) sits at the repository root and must not be moved or modified. About 1 GB, committed without Git LFS. `README.md` documents it; keep its footnote references (`[^1]` to `[^6]`) consistent when editing.

## Two parts
- **IMI**: lab recordings, 18 sessions (`IMI/Exp*`), with chatter labels.
- **KRPM**: an industrial machine, a single session, cutting intervals but **no chatter labels**. Quieter (10-20 dB) and barely clipped. Not used in the inpainting PoC (left out to keep it simple).

## Vocabulary
- **Session**: one recording. Three synchronized mono WAVs (48 kHz, int16) plus `cutting.csv`, optionally an `.mp4`.
- **Segment**: one cutting path, i.e. one row of `cutting.csv` (start and end in seconds).
- `segment_id` = `"{dataset}/{session}/{path_idx:03d}"`.

## Sensors
| Id | Sensor | Clipped samples (mean) |
|---|---|---|
| 0 | working-area microphone | about 24% |
| 1 | spindle ISS (internal sound sensor) | about 24% |
| 2 | base ISS | about 6% |

Segments with at most 0.1% clipping: 256 of 597 on S2, against 27 on S0 and 10 on S1. This is why S2 was chosen for the inpainting work.

## Labels (IMI only)
`IMI/labeling_all_details.xlsx`: one sheet per experiment, rows aligned one-to-one with `cutting.csv`. Holds the cutting parameters and two operator scores (0 normal, 1 uncertain, 2 chatter). Cohen's kappa between operators is about 0.69, and nearly all disagreement is about class 1, so always state the merge strategy (`consensus`: max/min/mean/agree in `labels.py`).
