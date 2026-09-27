# AGENTS.md

Guidance for AI agents and maintainers working in this repository.

## Project Overview

This is a Windows-focused AAF speech cleanup toolkit. The main entry point is
`aaf_pipeline.py`, which runs the GUI when started without CLI arguments and
produces `<input_stem>_processed.aaf` next to the source AAF.

The project is split into three main layers:

- `aaf_pipeline.py`, `aaf_gui.py`, `aaf_workflow.py`: user-facing orchestration,
  GUI, progress reporting, preparation, SDK/PyAAF2 fallback flow.
- `aaf_speech_filter/aaf_speech_filter/`: timeline analysis and mutations:
  quiet-clip removal, duplicate removal, YAMNet lane layout, media resolution,
  timing conversion, VAD/peak measurement.
- `aaf_io/aaf_io/`: low-level AAF I/O: lenient PyAAF2 open, embedded essence
  extraction, AAF SDK XML conversion, Premiere/DaVinci healing helpers.

Bundled tools and generated outputs:

- `sdk_bin/`: optional local-only AAF SDK tools such as `aaffmtconv.exe` and `ComAAFInfo.exe`.
- `dist_pyinstaller/`: built standalone app output.
- `build_pyinstaller/`, `.venv-build/`: build artifacts.
- `scripts/`: diagnostic and regression helpers for real-world AAF cases.
- `tests/`: unit tests.

## Hard Safety Rules

- Never overwrite the input AAF. The pipeline must always write a separate
  processed AAF or a temporary work copy.
- Treat real AAF media as user data. Do not delete files in user media folders unless the user explicitly asks.
- Temporary work should stay under the repository or the pipeline work folder
  (`__aaf_tool_work/` near the input AAF). Clean only temp folders you created.
- Do not use destructive git commands such as `git reset --hard` or checkout
  rewrites unless the user explicitly requested them.
- Be careful with `dist_pyinstaller/`: it is generated, huge, and should not be
  searched blindly. Prefer scoped `rg` searches in source folders.
- The repository may be used without `git` available. Do not assume git commands
  work in this environment.

## Coding Conventions

- Follow existing Python style: straightforward functions, local helpers,
  dataclasses for simple config/plan objects, and conservative changes.
- Avoid compatibility wrappers for internal code. Import the concrete module
  that owns the behavior, and keep old public APIs only when there is an
  explicit compatibility requirement.
- Prefer shared helpers in `timeline_timing.py`, `timeline_sourceclips.py`,
  `media_resolve.py`, and `timeline_walk.py` over duplicating AAF traversal or
  timing logic.
- Avoid broad refactors while fixing a specific AAF behavior. Many branches are
  deliberate fallbacks for malformed or application-specific AAF exports.
- Do not add sample-specific patches or one-off workarounds silently. For AAF
  compatibility bugs, first identify the general rule or shared invariant behind
  the failure and fix that systemically. If a systemic fix is not practical, say
  so explicitly before adding a narrow workaround, including why it is isolated
  and how it avoids regressing other hosts such as Nuendo.
- Never use clip display names, mob names, locator basenames, file stems, or
  filename substrings/extensions as production decision inputs for deletion,
  classification, lane layout, compatibility healing, or timing. Names are only
  for UI, logs, diagnostics, and user-facing reports. If media type matters,
  derive it from explicit media metadata, resolved media descriptors, decoded
  audio, or a documented host/export invariant.
- Do not encode behavior for individual AAF filenames, clip names, or observed
  sample names in production code or unit tests. Regression samples may appear
  only in diagnostic scripts or external/manual test harnesses.
- Keep comments short and only where they explain non-obvious AAF quirks.
- Use ASCII in new docs/code unless there is a clear reason to use Cyrillic or
  other Unicode. Some existing docs have encoding issues.

## AAF and Audio Behavior Notes

- Timeline SourceClips can point to `MasterMob` objects that contain nested
  `SourceMob` essence. Resolve media through `_resolve_wave_path_for_sourceclip`.
- Embedded essence is usually extracted read-only into `runtime_essence_paths`;
  the original embedded AAF must not be rewritten just for analysis.
- Do not trust slot edit rates blindly. Some sound timeline slots use fps-like
  rates such as `25`, while SourceClip `StartPosition` and `Length` may still be
  essence samples. Use `sourceclip_audio_timing()` and
  `sourceclip_duration_sec()`.
- Quiet-clip removal is based on the measured digital sample peak of the edited
  media segment versus `quiet_peak_dbfs`. Do not shift this threshold by clip
  gain. A gain of zero may still be treated as mute.
- For stereo/multichannel peak checks, current behavior downmixes channels before
  measuring. Preserve this unless the user asks for a different loudness model.
- YAMNet is optional/experimental and TensorFlow-heavy. Avoid importing it in
  code paths that do not need it.
- SDK XML removal keys must remain stable between PyAAF2 traversal and AAF SDK
  XML rewriting. Be careful when changing `sourceclip_sdk_key_tuple()`.

## Common Commands

Install editable packages from the repository root:

```powershell
pip install -e .\aaf_io
pip install -e .\aaf_speech_filter
pip install -r requirements-dev.txt
```

Run focused unit tests:

```powershell
python -m unittest tests.test_speech_vad tests.test_aaf_workflow
```

Run all unit tests:

```powershell
python -m unittest discover tests
```

If imports fail because local packages are not installed, set `PYTHONPATH`:

```powershell
$env:PYTHONPATH="$PWD;$PWD\aaf_io;$PWD\aaf_speech_filter"
python -m unittest discover tests
```

Run the CLI:

```powershell
python aaf_pipeline.py C:\path\to\input.aaf
```

Build the standalone PyInstaller app:

```powershell
cmd /c build_pyinstaller.bat
```

Check the built app starts:

```powershell
dist_pyinstaller\aaf_pipeline\aaf_pipeline.exe --help
```

## Testing Guidance

- For narrow audio logic changes, run the relevant unit tests first:
  `tests.test_speech_vad`, `tests.test_aaf_workflow`, and
  `tests.test_refactor_compat`.
- For AAF conversion or SDK XML changes, also run `tests.test_aaf_converter` and
  any diagnostic script that matches the reported sample.
- Real AAF regressions often require a targeted probe script under `scripts/`.
  Prefer adding or updating a focused diagnostic helper over printing from
  production code.
- If a test needs local packages but no editable install exists, use the
  `PYTHONPATH` command above instead of changing package imports.
- TensorFlow/YAMNet tests can be slow or environment-sensitive. Keep unit tests
  mocked where possible, as in `tests/test_speech_yamnet.py`.

## Build and Distribution Notes

- `build_pyinstaller.bat` creates/uses `.venv-build`, installs runtime
  dependencies, runs PyInstaller, and copies `sdk_bin/` into the distribution.
- TensorFlow makes `dist_pyinstaller/` very large. Do not commit generated build
  churn unless the user explicitly wants the distribution updated.
- If behavior changes in source and the user runs the standalone GUI/exe, rebuild
  `dist_pyinstaller\aaf_pipeline\aaf_pipeline.exe`.
- Optional ffmpeg bundling is controlled by `FFMPEG_EXE`; optional external AAF
  SDK tools are controlled by `AAF_TOOLS_DIR`.

## Diagnostic Workflow for Reported AAF Bugs

1. Reproduce the decision on a work copy, not the original AAF.
2. Identify the exact timeline SourceClip(s): mob name, SDK key, start units,
   length units, source track, resolved media path, chosen timing rate.
3. Measure the same media window the pipeline uses.
4. Compare the production decision path with the expected user-visible behavior.
5. Fix both SDK XML and PyAAF2 fallback paths when they share the same rule.
6. Run focused unit tests and, when practical, a direct probe on the reported AAF.
7. Remove any temporary debug folders created during investigation.

## Files Worth Knowing

- `aaf_workflow.py`: pipeline stage helpers, work-copy preparation, SDK build
  path, PyAAF2 fallback, lane-layout post-processing.
- `aaf_speech_filter/aaf_speech_filter/sdk_removals.py`: read-only pass that
  collects SDK XML replacement keys for quiet removals.
- `aaf_speech_filter/aaf_speech_filter/pyaaf2_filter.py`: PyAAF2 fallback
  filtering path used when SDK XML rebuild cannot complete.
- `aaf_speech_filter/aaf_speech_filter/speech_vad.py`: media segment decoding
  and peak-based quiet decision.
- `aaf_speech_filter/aaf_speech_filter/timeline_timing.py`: conversion from AAF
  units to media seconds.
- `aaf_speech_filter/aaf_speech_filter/media_resolve.py`: locator,
  MasterMob/SourceMob, and runtime essence path resolution.
- `aaf_io/aaf_io/converter.py`: embedded essence extraction and work-copy
  preparation.
- `aaf_io/aaf_io/sdk_tools.py`: AAF SDK tool discovery and subprocess runners.
- `aaf_io/aaf_io/sdk_xml.py`: AAF SDK XML validation and mutation helpers.

## Review Checklist

Before handing off a change, verify:

- The original AAF is never modified in place.
- SDK and PyAAF2 paths still agree for the changed behavior.
- Missing media, malformed descriptors, and fps-like edit rates are handled
  conservatively.
- Temporary files are cleaned or intentionally left in a documented work folder.
- Focused tests or a documented manual probe were run.
