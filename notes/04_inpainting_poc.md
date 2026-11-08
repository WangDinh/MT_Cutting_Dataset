# Inpainting proof of concept: design and data

Goal: a small demo of diffusion for **missing data** (inpainting). Scope is a proof of concept, not a benchmark.

## Decisions
| Question | Choice | Why |
|---|---|---|
| Channels | Sensor 2 only (base ISS) | Least clipped (about 6%); one channel keeps the demo simple |
| Missingness | Artificial time-span masking of spectrogram patches | Gives ground truth for scoring |
| Where | Windows inside cutting paths only | The machine idles much of the time; idle windows are trivial |
| Data | IMI only, KRPM left out | Simpler story; no domain-gap question |
| Training budget | At most 1-2 hours | User request; actual run was 35 min |

## Patches
- 17152 samples (about 0.357 s), non-overlapping, inside cutting paths.
- Log-magnitude STFT: n_fft 1024, hop 256, 64 frames, 128 bins (0-6 kHz; S2 has no energy above about 6 kHz).
- dB range [-100, 0] normalised to [-1, 1].

## Keep rule
A patch is kept if it has no full-scale (clipped) samples. There is no loudness gate.

History: the first version also required RMS at least 3 dB above the session's idle median (median 50 ms RMS outside the paths, 0.2 s guard). It was removed after looking at notebook 02: some cutting windows fell below the threshold but were clearly louder than their local surroundings. `idle_db` is still stored per patch for reference. Clipped samples are those at |x| >= 32767; they distort the spectrum, so the ground truth would be wrong.

## Result: 1461 patches
| Split | Patches | Sessions |
|---|---|---|
| train | 1230 | 11 sessions |
| val | 87 | Exp0-3, Exp1-1-2, Exp1-6 |
| test | 144 | Exp0-1 (103 patches), Exp1-1-A, Exp1-8 |

`Exp2-1` yields 0 patches (every window clips). Splits are by session. All three test sessions are scored normal (no chatter), and `Exp0-1` dominates the test average, so check the per-session table.

## Masks
Seeded, over time frames and all frequencies: 1-2 spans of 10-100 ms. Overlapping spans merge, so gaps reach about 197 ms. About 23% of frames are masked on average. Files: `data/processed/masked/patches.npz` (`clean`, `mask`, `split`) and `patches.csv`.

## Models
- **Baselines**: mean-fill (per-bin mean of observed frames) and per-bin linear interpolation. (Cubic interpolation and a mean of 4 diffusion samples were tried and removed: cubic overshoots badly at 13 dB, and the 4-sample average blurs the texture that is diffusion's point.)
- **Diffusion**: unconditional DDPM, cosine schedule, T=250, epsilon-prediction, 3.56M-parameter U-Net (base 32, mult 1/2/3/4, attention at lowest resolution), EMA, RePaint sampling with 2 resamples. Trained with time-flip and +/-3 dB gain-jitter augmentation.

## Metrics (over missing frames only)
MAE and RMSE in dB; low-band (0-3 kHz) MAE; `profile_corr` (do harmonic stripes land at the right frequencies); `std_ratio` (predicted over true temporal std; 1 is realistic, 0 is a blur).
