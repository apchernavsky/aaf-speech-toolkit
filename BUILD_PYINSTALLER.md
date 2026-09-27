# PyInstaller build (Windows)

This repository can be distributed as a self-contained Windows folder using **PyInstaller**.

## One-command build

On a fresh checkout, first copy `aaf_tool_config.example.json` to `aaf_tool_config.json` and install the native tools described below. The build bundles that local configuration; use empty/default search roots for a public distribution.

Run:

```bat
build_pyinstaller.bat
```

Output will be placed under:

- `dist_pyinstaller\aaf_pipeline\`

Run GUI:

- `dist_pyinstaller\aaf_pipeline\aaf_pipeline.exe`

## Native build inputs

Install the native tools first; they are not part of the public source snapshot. The build validates the required SDK tools before publication. See [native tool setup](sdk_bin/README.md).

The script copies local `.\sdk_bin\` into:

- `dist_pyinstaller\aaf_pipeline\sdk_bin\`

You can also override with an external SDK folder:

```bat
set AAF_TOOLS_DIR=C:\path\to\AAF_Tools
build_pyinstaller.bat
```

## Bundling ffmpeg (optional)

If you want DaVinci MXF audio decoding on target machines, bundle ffmpeg:

```bat
set FFMPEG_EXE=C:\path\to\ffmpeg.exe
build_pyinstaller.bat
```

It will be copied into:

- `dist_pyinstaller\aaf_pipeline\bin\ffmpeg.exe`

## Notes / caveats

- TensorFlow makes the distribution large.
- If TensorFlow Hub needs to download YAMNet at runtime, target machines will require Internet or a pre-populated TF Hub cache.


## Verified dependencies and publication

The build uses `requirements-build-lock.txt` as pip constraints, captured from the tested Windows/Python 3.12 build environment. Update the constraints deliberately and rerun the full test suite when upgrading dependencies. The standalone distribution includes the optional YAMNet dependencies.

An exclusive `.pyinstaller_build.lock` directory prevents simultaneous builds from sharing the build cache and environment. If a build is interrupted, verify that no build process is running before removing the empty lock directory.

Output is assembled in a unique `dist_pyinstaller.stage-*` directory. After the staged executable passes `--help`, `scripts/publish_distribution.py` promotes it to `dist_pyinstaller`. The previous distribution is retained as `dist_pyinstaller.previous-*`; a failed promotion restores the previous destination. Backups require deliberate manual cleanup after acceptance.

Executable builds require separate end-to-end acceptance with the bundled SDK, FFmpeg and YAMNet model. Source-only CI does not establish frozen application compatibility.
