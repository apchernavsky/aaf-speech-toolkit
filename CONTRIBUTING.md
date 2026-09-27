# Contributing

Use Python 3.10-3.12 on Windows. Follow [installation and testing](README.md). Start with synthetic AAF/audio fixtures; keep real media outside version control.

Before changing timeline behavior, identify the invariant, owner and failure modes. Add a regression that fails for old behavior. Preserve original AAFs, source windows, visible edit times, transitions, effects and channel order. Check both SDK XML and PyAAF2 writers when a shared rule changes.

Do not use names or sample identities as production decision inputs. Keep optional TensorFlow imports out of core paths. Avoid unnecessary dependencies.

Run `python -m unittest discover tests` and relevant corpus probes. Distinguish structural compatibility, classifier evidence and actual DAW verification. See [coverage](docs/validation.md) and [corpus setup](docs/public-aaf-corpus.md).

Do not commit environments, native binaries, models, downloaded corpora, private media, local configuration or diagnostic output. Bug reports should include tool versions, options and a minimized synthetic reproduction. Remove personal paths and identifiers from logs. Only share media you are authorized to redistribute.

See [THIRD_PARTY.md](THIRD_PARTY.md) before distributing executable bundles.
