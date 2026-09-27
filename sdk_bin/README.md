# Optional native tools

The public source distribution includes setup instructions only. Local executables and DLLs are ignored by Git.

Obtain compatible Windows tools from the [AAF SDK project](https://aaf.sourceforge.net/) and keep required DLLs beside them. The pipeline uses `aaffmtconv.exe` and `ComAAFInfo.exe`. Place them here or configure:

~~~powershell
$env:AAF_TOOLS_DIR='C:\path\to\AAF-SDK-tools'
~~~

Obtain FFmpeg/FFprobe from sources linked by the [FFmpeg download page](https://ffmpeg.org/download.html). Place them on PATH or here. For a later executable build, `FFMPEG_EXE` specifies FFmpeg to bundle; see [build instructions](../BUILD_PYINSTALLER.md).

Keep upstream notices/provenance with separately distributed native packages. Tool versions affect compatibility; the Python fallback does not replace native validation.
