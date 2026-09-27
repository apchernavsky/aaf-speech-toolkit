# Validation coverage

The pre-publication Windows/Python 3.12 processing audit passed 728 tests with the separately provisioned public fixture cache. A fresh checkout runs fewer tests because 135 public cases are replaced by one explicit skip until provisioned.

Additional local evidence covered 23 private AAF inputs, 9,623 input occurrences and 8,795 retained clips, with zero visible edit shifts or aligned-channel inversions. Source hashes were unchanged. SDK reopening, sequence lengths and removals were checked independently. Private media, paths and per-case reports are not distributed.

The track-permutation contract was exhaustively checked over 1,082,400 class/protection combinations with up to four tracks, plus seeded larger cases and both writer adapters.

Classification validation is separate from structural validation. Reviewed expectations check known labels and relative placement independently. An automated audit used 36 seeded windows and two additional changed full clips with a separate local ASR model. Some short windows remained inconclusive; this was not human listening or visual DAW verification.

All 135 public compatibility scenarios passed with the YAMNet option enabled: 134 valid CFB AAFs and one expected unsupported-format rejection. Missing media, empty timelines and too few eligible tracks limit classification coverage. This does not establish universal classifier accuracy.

## Reproduce

~~~powershell
python -m unittest discover tests
python scripts/fetch_aaf_corpus.py
python scripts/fetch_aaf_corpus.py --verify-only
python scripts/check_aaf_corpus.py --stage structure --report-dir __manual_checks/public-structure
python scripts/verify_local_aafs.py C:\path\to\private-corpus --report-dir __manual_checks/local-check --jobs 2
~~~

Report directories must be new. Local processing checks require native tools and a locally available YAMNet model. They use owned copies and never replace corpus inputs. Private reviewed-label manifests are not distributed. Use `scripts/corpora/reviewed_audio_cases.example.json` as a synthetic schema example, then run `python scripts/check_reviewed_audio_cases.py <report-dir> --cases <local-manifest.json>`. The previous implicit default was removed to prevent accidental use or publication of private sample identities. The local `scripts/corpora/reviewed_audio_cases.json` path is ignored by Git.
