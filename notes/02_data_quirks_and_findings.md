# Data quirks and findings

## Raw-file quirks (already absorbed by `mtcut.io` and `mtcut.dataset`)
- WAV names: IMI uses `*_VM20i_sensor{k}.wav`, KRPM uses `*_Mazak_sound{k}.wav`.
- `cutting.csv` is mostly UTF-8 with a BOM, but `Exp1-4` has no BOM. The header is `Start,End` in `Exp3-2` and `Exp4-3`.
- Interval pairs can overlap: `Exp0-1` paths 17 and 18 overlap by 0.42 s.
- Label workbook: direction is sometimes "Up (conventional)" or "Down (climb)", normalised to `up`/`down`. Machine names are filled only in the first row. The `Summary` sheet omits `Exp3-2`, `Exp4-3` and `Exp4-3-2`.
- Empty `*.txt` files are notes whose filename is the message, e.g. `No_video_clip.txt`.

## Findings that shape modelling (notebook 01)
- **Split by session, never by window.** Chatter rate per session runs from 0% (`Exp0-1`) to 100% (`Exp4-3`) and is confounded with tool, material and gain.
- **IMI is heavily clipped**: up to about 17% of samples at full scale, and clipping correlates with loud chatter cuts. Any clipping filter therefore removes the loudest chatter.
- **KRPM** is barely clipped, 10-20 dB quieter, and has a constant tone near 6.7 kHz on S1.
- **Most of each recording is not cutting.** The machine sits idle for long stretches, so analyses focus on windows inside cutting paths.
- **Operator agreement**: kappa about 0.69; disagreements are almost all class 1.
