# AAF Speech Toolkit

[![Tests](https://github.com/apchernavsky/aaf-speech-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/apchernavsky/aaf-speech-toolkit/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Clean and organize audio timelines in Advanced Authoring Format (AAF) files.**

A Windows-focused Python GUI and CLI for audio post-production: remove quiet clips, remove duplicate occurrences, and optionally organize speech, noise and music into track groups with local YAMNet inference.

Processing creates a separate `<input_stem>_processed.aaf` beside the input. The original AAF is never a processing destination. Retained clips must preserve their source windows and visible edit positions.

## What it does

| Capability | Behavior |
| --- | --- |
| Quiet-clip removal | Measures digital sample peak over the edited source window; default threshold is **-40 dBFS**. |
| Duplicate removal | Uses source identity and timeline/rendering semantics to identify duplicate occurrences. |
| Optional track layout | Classifies decoded audio with YAMNet and arranges speech above noise, with music below, subject to structural constraints. |
| Media resolution | Follows MasterMob/SourceMob references to embedded essence or linked files and configured media search roots. |
| Compatibility | Uses PyAAF2 plus optional native AAF SDK conversion/validation, with compatibility handling for several host exports. |
| Safe output lifecycle | Uses run-owned work files, cancellation checks and transactional output publication. Failed runs preserve any previous output. |
| Diagnostics | Includes structural, timing, source-window, channel-order and independent reviewed-label checks. |

Names of clips and media files are not speech/music/noise classification rules. AAF objects that cannot be interpreted safely constrain processing; missing or ambiguous media is reported rather than treated as silence.

## Requirements

- **Windows**, with Python **3.10-3.12**. Python 3.12 was used for the local pre-publication validation; GitHub Actions tests all three versions.
- Core Python dependencies: PyAAF2, NumPy and packaging.
- Tkinter for the GUI (normally included with the Windows Python installer).
- Optional TensorFlow/TensorFlow Hub and a YAMNet model for classification.
- Optional AAF SDK tools and FFmpeg/FFprobe for additional conversion, validation and media decoding.

This repository distributes source code. Native executables, model weights, downloaded fixtures and private production media are not included. See [third-party components and licenses](THIRD_PARTY.md).

## Quick start

From PowerShell:

```powershell
git clone https://github.com/apchernavsky/aaf-speech-toolkit.git
cd aaf-speech-toolkit
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ./aaf_io -e ./aaf_speech_filter
.\.venv\Scripts\python.exe aaf_pipeline.py
```

Starting without arguments opens the GUI. Select an AAF, choose processing options, and start processing. The GUI currently uses Russian labels.

**The GUI enables YAMNet track layout by default.** For a core-only installation, uncheck the track-layout option. To use it, install the optional dependencies and model as described below. Quiet removal and duplicate removal are also enabled by default.

### Command line

Core cleanup, with classification disabled by default:

```powershell
.\.venv\Scripts\python.exe aaf_pipeline.py 'C:\path\to\input.aaf'
```

Change the quiet threshold or disable individual operations:

```powershell
.\.venv\Scripts\python.exe aaf_pipeline.py 'C:\path\to\input.aaf' --quiet-peak-dbfs -45
.\.venv\Scripts\python.exe aaf_pipeline.py 'C:\path\to\input.aaf' --no-remove-quiet
.\.venv\Scripts\python.exe aaf_pipeline.py 'C:\path\to\input.aaf' --no-remove-duplicates
.\.venv\Scripts\python.exe aaf_pipeline.py --help
```

Quiet removal compares sample peak against the configured threshold. Clip gain does not offset that threshold; zero gain can still represent mute. Multichannel peak analysis currently downmixes channels before measurement.

### YAMNet track layout

Install optional dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e './aaf_speech_filter[yamnet]'
$env:AAF_YAMNET_MODEL_DIR='C:\path\to\yamnet-savedmodel'
.\.venv\Scripts\python.exe aaf_pipeline.py 'C:\path\to\input.aaf' --experimental-yamnet-lanes --no-download-yamnet-model
```

`AAF_YAMNET_MODEL_DIR` points to an extracted YAMNet SavedModel. Without a local/cached model, the GUI may download one when layout is selected. The CLI also permits model download by default when classification is requested; use `--no-download-yamnet-model` for a local-model-only run. The model loader additionally recognizes `AAF_YAMNET_ALLOW_DOWNLOAD=1` as permission to download.

Audio analysis runs locally. The pipeline does not upload audio to a classification service. Installing dependencies and fetching models or the optional public corpus require network access. TensorFlow and model downloads can be large.

### Native tools and linked media

Follow [native tool setup](sdk_bin/README.md) to provide `aaffmtconv.exe`, `ComAAFInfo.exe`, FFmpeg or FFprobe. Configure an external SDK directory through `AAF_TOOLS_DIR`, or place the tools in `sdk_bin/` with their required DLLs and notices.

For linked audio, copy `aaf_tool_config.example.json` to `aaf_tool_config.json` and set local search roots:

```json
{
  "media_search_roots": ["C:/path/to/media"]
}
```

The local configuration is ignored by Git. `AAF_MEDIA_ROOTS` also accepts semicolon-separated search roots. The pipeline searches near the input AAF as well. Native tools and configuration paths are resolved for each run; see the [architecture and processing contract](docs/architecture.md).

## Processing limits

- **Classification is heuristic.** Mixed speech/music, short fragments and effects can be ambiguous. A successful structural check does not prove every label is correct.
- **Structure can constrain layout.** Effects, transitions, incompatible clocks, protected tracks and unknown material may prevent an otherwise desirable move. Preserving rendering and timing takes precedence over compact packing.
- **Python 3.10 has a narrower AIFF decoder.** Its standard `aifc` module rejects little-endian AIFC (`sowt`); direct analysis reports the limitation and retains the clip. Python 3.11-3.12 can analyze 16-bit `sowt`. Use Python 3.12 for the documented installation.
- **Host compatibility varies.** PyAAF2 fallback and AAF SDK conversion have different limits. Malformed containers and unsupported formats may be rejected explicitly.
- **Keep original media and inspect important results in your DAW.** Automated AAF checks do not replace listening or a target-application import check.
- **Metadata is not promised to remain byte-for-byte identical.** Processing changes timeline objects, and conversion or compatibility repair may rewrite container representation. See the preservation checks and their limits in [validation coverage](docs/validation.md).

See [silent-audio layout behavior](docs/silent-audio-layout.md) for empty/silent cases.

## Testing and validation

Install development dependencies and run the core suite:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover tests
```

Core tests create synthetic fixtures and mock model inference. No test downloads a model or corpus automatically.

The optional public corpus contains **135 distinct compatibility cases**: 134 CFB AAFs and one expected unsupported-format rejection. Assets are fetched separately from pinned upstream sources and checked by size and hashes:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_aaf_corpus.py
.\.venv\Scripts\python.exe scripts/fetch_aaf_corpus.py --verify-only
.\.venv\Scripts\python.exe -m unittest discover tests
.\.venv\Scripts\python.exe scripts/check_aaf_corpus.py --stage structure --report-dir __manual_checks/public-structure
```

An absent corpus causes one explicit skip; a partial or corrupt cache fails. Use a new report directory for each corpus run. Downloaded third-party fixtures retain their own terms and must not be committed to this repository.

GitHub Actions runs core tests on Windows/Python 3.10, 3.11 and 3.12. Its manual `public_corpus` option also provisions and verifies the public structural corpus. CI does not build an EXE or access private media.

The local validation also includes private production AAFs, independent source/timing comparisons, channel ordering and exhaustive track-permutation checks. Private sample identities and reports are excluded from this repository. See [validation coverage and limitations](docs/validation.md) and [public corpus documentation](docs/public-aaf-corpus.md) for exact scope and reproduction commands.

## Architecture

```text
aaf_pipeline.py / aaf_gui.py       CLI, GUI, progress and cancellation
                 |
aaf_workflow.py                   Pipeline stages and fallback policy
                 |
aaf_speech_filter/               Audio analysis, timing and mutation plans
                 |
aaf_io/                          AAF access, SDK adapters, extraction,
                                 output transactions and cleanup
```

Shared timing and media-resolution helpers keep the SDK XML and PyAAF2 paths aligned. Tests cover both writers when they implement a common invariant.

| Location | Contents |
| --- | --- |
| `aaf_io/` | Low-level AAF package and SDK integration. |
| `aaf_speech_filter/` | Peak analysis, classification and lane-layout planning. |
| `scripts/` | Corpus runners and focused diagnostic tools. |
| `tests/` | Synthetic and optional public-corpus regression tests. |
| `docs/` | Architecture, validation scope and corpus instructions. |
| `sdk_bin/` | Setup instructions for separately supplied native tools. |

## Contributing and reporting problems

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing processing behavior. Report reproducible issues through the [bug report template](https://github.com/apchernavsky/aaf-speech-toolkit/issues/new?template=bug_report.md), including versions, options, expected behavior and a minimal synthetic example where possible.

Remove credentials, personal paths and private clip identifiers from reports. Only share media you have permission to redistribute. Local configuration, models, native binaries, diagnostic reports and production media belong outside version control.

[Build instructions](BUILD_PYINSTALLER.md) are available for maintainers who need a standalone application. This initial source publication does not include an executable release.

## License

Original project code is available under the [MIT License](LICENSE). The PyAAF2 compatibility adapter contains adapted upstream code with its [retained MIT notice](aaf_io/aaf_io/compat/PYAAF2_LICENSE.txt). Dependencies, native tools, models and external fixtures retain their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).
