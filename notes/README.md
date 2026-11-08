# Notes

Working notes for this repository. `CLAUDE.md` at the root is only the overview and a map; the detail lives here, one topic per file.

| File | What it covers |
|---|---|
| [01_dataset.md](01_dataset.md) | What the dataset is: sessions, segments, sensors, labels |
| [02_data_quirks_and_findings.md](02_data_quirks_and_findings.md) | Raw-file inconsistencies and findings that should shape modelling |
| [03_code_layout.md](03_code_layout.md) | The `mtcut` package, scripts, notebooks, tests and commands |
| [04_inpainting_poc.md](04_inpainting_poc.md) | Missing-data proof of concept: design decisions and data preparation |
| [05_results_and_next_steps.md](05_results_and_next_steps.md) | Baselines vs diffusion results, caveats, ideas not yet tested |
| [06_environment.md](06_environment.md) | Machine and tooling gotchas (Windows, OpenMP, torch, Python) |
| [07_chatter_with_filled_gaps.md](07_chatter_with_filled_gaps.md) | Downstream test: normal-vs-chatter classification on filled gaps |

Keep each file to one topic. Add a row here when you add a file.
