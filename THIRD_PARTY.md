# Third-party components and notices

This repository does not redistribute native binaries, model weights or test media. The root MIT license covers original project code; it does not relicense dependencies or separately downloaded material.

## Adapted code included in this repository

`aaf_io/aaf_io/compat/pyaaf2_lenient.py` adapts portions of PyAAF2 stream writing and AAF file initialization. PyAAF2 is Copyright (c) 2017 Mark Reid, under the MIT License. The complete upstream notice is retained in [PYAAF2_LICENSE.txt](aaf_io/aaf_io/compat/PYAAF2_LICENSE.txt), adjacent to the adapter so it also accompanies package source.

Upstream: [PyAAF2](https://github.com/markreidvfx/pyaaf2), [license](https://github.com/markreidvfx/pyaaf2/blob/main/LICENSE).

## Separately installed dependencies

| Component | Purpose | Upstream and terms |
| --- | --- | --- |
| PyAAF2 | Python AAF access | [Repository and MIT license](https://github.com/markreidvfx/pyaaf2) |
| NumPy | PCM analysis and numerical operations | [Repository and license notices](https://github.com/numpy/numpy) |
| packaging | Version handling | [Repository and license notices](https://github.com/pypa/packaging) |
| TensorFlow / TensorFlow Hub / tf-keras | Optional local model execution | [TensorFlow](https://github.com/tensorflow/tensorflow), [Hub](https://github.com/tensorflow/hub), [tf-keras](https://github.com/keras-team/tf-keras) |
| YAMNet | Optional speech/music/noise evidence | [Model implementation and documentation](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet), [model distribution](https://tfhub.dev/google/yamnet/1) |
| AAF SDK | Optional native conversion and validation | [AAF SDK project](https://aaf.sourceforge.net/) |
| FFmpeg / FFprobe | Optional media decoding and inspection | [Downloads](https://ffmpeg.org/download.html), [licensing information](https://ffmpeg.org/legal.html) |
| Build and development tools | Packaging and tests | Versions are listed in `requirements-dev.txt` and `requirements-build-lock.txt`; consult each installed distribution's notices. |

Review the exact version/build terms when redistributing an executable bundle. Native builds can have different obligations depending on included components. Model terms and third-party fixture permissions must be checked separately from the Python code license. No claim is made that a bundle of all optional components can be redistributed under MIT alone.

## External test fixtures

The [public corpus manifest](scripts/corpora/public_aaf.json) records origins, pinned revisions, file hashes, available license assets and repository-level license metadata. The downloader keeps assets in an ignored local cache. No fixture binaries are part of the Git history published here.

Repository-level licensing does not automatically cover media supplied in issue attachments. A public download link is not a license grant. Keep the cache local and review individual terms before redistributing any asset. Private reviewed-label manifests and production media are excluded; the shipped example uses invented identities.
