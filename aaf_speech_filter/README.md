# aaf_speech_filter

Timeline-level AAF processing package used by the main toolkit pipeline.

It handles:

- quiet clip detection and removal;
- duplicate timeline block removal;
- optional YAMNet-based lane layout;
- shared timeline traversal, timing conversion, and media resolution.

## Install

From the toolkit repository root, install `aaf_io` first, then this package:

```powershell
pip install -e .\aaf_io
pip install -e .\aaf_speech_filter
```

## CLI

```powershell
python -m aaf_speech_filter.cli C:\path\to\input.aaf
```

Useful options:

- `--remove-quiet` / `--no-remove-quiet`
- `--quiet-peak-dbfs`
- `--experimental-yamnet-lanes`
- `--yamnet-score-threshold`
- `--yamnet-frame-aggregate`
