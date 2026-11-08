# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository. This file is only an overview and a map; detail lives in [`notes/`](notes/README.md).

## What this repository is

The **published CNC milling cutting-sound dataset** (Kim et al., Manufacturing Letters 2024; Mostafiz et al., INDIN 2024) plus an ML project built on it: the `mtcut` package, scripts, notebooks, and a diffusion-based missing-data (inpainting) proof of concept.

The raw data (`IMI/`, `KRPM/`, `figure/`, `README.md`) stays at the repository root because it is the public dataset layout. **Do not move or modify the raw files.** `README.md` documents the dataset; keep its footnote references (`[^1]` to `[^6]`) consistent.

## Where to look

| Need | Note |
|---|---|
| What the dataset is: sessions, segments, sensors, labels | [notes/01_dataset.md](notes/01_dataset.md) |
| Raw-file quirks and findings that shape modelling | [notes/02_data_quirks_and_findings.md](notes/02_data_quirks_and_findings.md) |
| Package layout, scripts, notebooks, all commands | [notes/03_code_layout.md](notes/03_code_layout.md) |
| Inpainting PoC: decisions, patches, masks, models | [notes/04_inpainting_poc.md](notes/04_inpainting_poc.md) |
| Results, caveats, untested ideas | [notes/05_results_and_next_steps.md](notes/05_results_and_next_steps.md) |
| Windows / OpenMP / torch gotchas | [notes/06_environment.md](notes/06_environment.md) |
| Downstream test: chatter classification on filled gaps | [notes/07_chatter_with_filled_gaps.md](notes/07_chatter_with_filled_gaps.md) |

When you finish meaningful work, update the matching note (add a row to `notes/README.md` for a new file) rather than growing this file.

## Essentials

- Common commands: `pip install -e ".[dev,notebook]"`, `python -m pytest`, `ruff check src scripts tests` (line length 110). Full list in note 03.
- `src/mtcut/` is the only library code; notebooks and scripts import from it and never re-parse raw files.
- Generated outputs (`data/processed/`) are git-ignored.
- Sensors: 0 microphone, 1 spindle ISS, 2 base ISS. Labels exist for IMI only; KRPM has none.

## Rules that matter

- **Split by session, never by window** (chatter rate is confounded with session).
- IMI is heavily clipped; state any clipping filter, since it removes the loudest chatter.
- State the operator-merge strategy whenever labels are used (kappa is about 0.69).
- Set `KMP_DUPLICATE_LIB_OK=TRUE` when mixing torch with the Anaconda numpy on Windows.
- Current headline: on the inpainting PoC, mean-fill beats the diffusion model on every error metric; diffusion only wins on realism: a detector finds mean-fill gaps almost always but a single diffusion sample much less (note 05). For chatter classification all fills match clean patches and diffusion is no better (note 07).
