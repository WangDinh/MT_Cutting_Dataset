# Code layout and commands

## Commands
```bash
pip install -e .                          # install mtcut (src/ layout); or: pip install -r requirements.txt
pip install -e ".[dev,notebook]"          # adds pytest + ruff, jupyter + seaborn
pip install -e ".[diffusion]"             # adds torch (needed for diffusion.py only)
pip install -e ".[chatter]"               # adds scikit-learn (needed for chatter.py only)
ruff check src scripts tests              # lint, line length 110
python -m pytest                          # tests (pytest adds src/ to the path; no install needed)
python -m pytest tests/test_dataset.py::test_label_parsing   # single test
python scripts/validate_dataset.py        # raw-data integrity checks; exits 1 on errors
python scripts/build_segment_index.py     # writes data/processed/{sessions,segments}.csv
python scripts/make_masked_pairs.py       # writes data/processed/masked/
python scripts/train_diffusion.py --steps 9000 --max-minutes 35
python scripts/eval_inpainting.py         # test split; writes inpaint_metrics.csv, inpaint_preds.npz
python scripts/eval_detectability.py      # real-vs-filled detector (logistic regression) on the test split; needs eval_inpainting.py first
python scripts/train_diffusion.py --exclude-session Exp3-2 --out data/processed/diffusion_fold_Exp3-2 --max-minutes 22   # per-fold generator (also Exp4-3-2)
python scripts/eval_chatter_fills.py      # chatter classification on filled gaps; writes data/processed/chatter/
jupyter nbconvert --to notebook --execute --inplace notebooks/01_explore_dataset.ipynb
```
Scripts import `mtcut`: install the package or prefix with `PYTHONPATH=src`. Notebooks fall back to `../src`.

## `src/mtcut/` (the only library code)
Notebooks and scripts should import from here rather than re-parse the raw files.

| Module | Role |
|---|---|
| `config.py` | Paths and constants. `DATA_ROOT` defaults to the repo root; override with `MTCUT_DATA_DIR`. Sensor and label maps. |
| `io.py` | Raw readers; absorb the file quirks. |
| `dataset.py` | `Session` (lazy memory-mapped WAV), `discover_sessions()`, `session_table()`, `segment_table()` (one row per cutting path, joined with labels). |
| `labels.py` | Merge the two operators' scores (`consensus`), binarize, Cohen's kappa. |
| `features.py` | PSD, spectrogram, RMS, tooth-passing-harmonic features. |
| `viz.py` | Shared plots. |
| `masking.py` | Spectrogram patches, dB normalisation, seeded time-span masks, `idle_level_db`. |
| `inpaint.py` | Patch loading, baselines (`fill_mean/linear`), metrics (`masked_errors`, `gap_stats`). |
| `diffusion.py` | Small DDPM U-Net, `Diffusion`, RePaint inpainting, `EMA`. Needs torch. |
| `detect.py` | Real-vs-filled detector: logistic regression, leave-one-session-out AUC (needs scikit-learn). |
| `chatter.py` | Normal-vs-chatter features, simple classifiers, bootstrap AUC (needs scikit-learn). |

## Other folders
- `scripts/`: CLI entry points. `tests/`: run against the real data (`test_masking.py`, `test_inpaint.py` added for the PoC).
- `notebooks/`: numbered analyses.
  - `01_explore_dataset`: dataset tour and the findings in note 02.
  - `02_masked_dataset`: patches, masks, cutting-path examples.
  - `03_inpainting_baselines_vs_diffusion`: training curve, results, examples, detectability (section 7), slide figures (section 8).
  - `04_chatter_with_filled_gaps`: chatter classification on filled gaps (note 07).
- `data/processed/`: generated outputs (patches, model, metrics), git-ignored.
