# Public AAF regression corpus

The external corpus is defined by scripts/corpora/public_aaf.json. It contains all 135 unique candidates from the 2026-09-23 research: 134 CFB AAFs and one intentionally unsupported MXF-style .AAF. Duplicate paths retain their provenance but do not create duplicate cases. Available sidecar media, upstream expected-output files and license texts bring the manifest to 237 assets (87,906,441 bytes).

Binary fixtures are downloaded into the ignored .cache/aaf-corpus directory, rather than committed. Repository objects use commit-pinned URLs and exact Git blob hashes. Issue ZIPs additionally pin archive SHA-256, size and exact member. No test downloads anything automatically.

## Setup and commands

Run from the repository root, using the project's Python environment:

~~~powershell
python scripts/fetch_aaf_corpus.py
python scripts/fetch_aaf_corpus.py --verify-only
python -m unittest discover tests
~~~

Once provisioned, ordinary unittest discovery includes 135 individual external structure tests. An entirely absent cache produces one explicit skipped test. A partial or corrupt cache fails discovery. For CI or a required corpus gate, always run the verify-only command first; it fails if any asset is missing.

Full independent structure and processing runs:

~~~powershell
python scripts/check_aaf_corpus.py --stage structure --report-dir __manual_checks/public-structure
python scripts/check_aaf_corpus.py --stage process --report-dir __manual_checks/public-processing
~~~

Report directories must be new. Each contains results.json, summary.json and a per-case result.json/run.log. The command exits nonzero for any unexpected failure or timeout. Use --case <full-git-blob-sha> to repeat selected cases; --case can be repeated. --cache and --manifest permit explicit alternative inputs. Default timeout is 180 seconds per case and can be changed with --timeout.

An optional --stage yamnet enables lane layout as well as cleanup, using an already installed model with downloads disabled. It is not part of the default tests because TensorFlow is optional and these fixtures have no speech/music/noise ground truth. Some cases have missing media, no composition or no audio; an all-green structural corpus is not classifier validation.

## Assertions and ownership

Structure checks traverse the content object graph, record mob/slot/edit-rate/length and audio SourceClip parameters, hash every embedded essence stream, save a separate writable copy, then reopen it and compare the snapshot. Mob properties are serialized, so this is more than a file-copy check. The MXF-style case must fail with the specific container-signature error; other errors are failures.

Processing checks invoke the real run_aaf_pipeline with quiet removal and duplicate removal enabled. They require a separate readable output, unchanged source bytes, unchanged composition count and embedded essence, and retained audio source windows drawn from the original multiset. These are compatibility and preservation assertions, not a complete oracle for which clips should be removed. Absolute placement, automation curves, target-DAW import and classifier quality need additional semantic tests. Downloaded LibAAF expected output is retained for future adapter-specific assertions; it is not claimed as fully asserted by this runner.

Missing external media remains observable in each pipeline log. The pipeline may explicitly skip quiet analysis and continue duplicate removal; that is not counted as successful audio decoding. Fixture media roots override local configured search roots inside the isolated worker. The test harness uses scoped dependency injection for this; production source startup/import behavior is still used and no PYTHONPATH is injected.

The parent owns each worker's temporary directory. Timeouts terminate the worker tree and wait for known descendant processes before cleanup. The parent independently checks source integrity even when a worker fails. Original fixtures are never processing destinations. Work copies are removed; evidence logs remain.

## Corpus inventory

| Source | Distinct AAF candidates contributed |
| --- | ---: |
| LibAAF | 63 (65 paths; three identical fade fixtures share one blob) |
| OTIO AAF adapter | 37 |
| PyAAF2 | 15 |
| AAF SDK mirrors | 6 (DNEG/nexgenta copies deduplicated) |
| Legacy PyAAF | 1 additional |
| rust-aaf | 2 |
| Issue attachments | 11 additional, including the MXF-style negative case |
| Total | 135 |

Upstream origins, exact commits, archive links and duplicate aliases are preserved in the manifest. The sources were compared with the existing 21-file local corpus: no exact matches. Those user AAFs remain a separate corpus and are not redistributed.

Repository-level license metadata is recorded, and available license files are downloaded. This does not assign a license to third-party media or issue attachments. Keep the cache local; review individual terms before redistributing binaries. No automatic upstream code execution is involved.

## Adding a fixture

Add a unique case and its asset to the manifest, pinning provenance, byte size and Git blob hash. An archive source also needs archive_bytes, archive_sha256 and member. Add supplied sidecar files as media assets with their own hashes and set the case's scoped media_roots. If the AAF hash already exists, add an origin to that existing case instead. Never add filename-based behavior to production code or classify an unexpected processing failure as an expected rejection.

Use synthetic unit fixtures for harness contracts and newly discovered general behavior. Regression sample names belong in this external manifest, not production decisions or sample-specific unit tests.

## Behavior fixed by the corpus

Valid AAF containers can contain only source mobs, empty compositions or picture-only timelines. The cleanup pipeline previously attempted SDK XML/timeline filtering anyway, causing CompositionPackage/CompositionMob errors. After successful preparation, it now detects the absence of audio SourceClips through the shared timeline walker and publishes an unchanged separate copy. Invalid input is still rejected; cancellation and atomic output publication remain owned by the existing pipeline lifecycle.

The original rollback test used an empty mocked work copy to reach an SDK failure. It now creates a real nonempty audio timeline so it continues to test rollback at that boundary.

A second regression exposed by preservation checks involved older AAF dictionaries: OperationGroup stores the operation under OperationDefinition instead of Operation, with the same property PID 0x0b01. The shared gain reader now handles that spelling, preserving gain and zero-gain mute interpretation before and after SDK normalization. Source windows, parameters and media are not rewritten by this fix.
