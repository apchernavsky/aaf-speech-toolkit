# Architecture and processing contract

## Ownership

The GUI/CLI layer owns options, cancellation and progress. `aaf_workflow.py` owns stage orchestration and SDK/PyAAF2 fallback. `aaf_speech_filter` owns audio decisions and timeline mutation plans. `aaf_io` owns container access, native processes, extraction, atomic publication and cleanup.

Each run has an exclusive workspace. Failed or cancelled runs preserve their input and any previous output. Cleanup affects only that run. Search roots and SDK paths are captured per run rather than written into shared process state.

## Audio decisions

Quiet removal measures digital sample peak in the edited source window. Clip gain does not shift the threshold; mute may still apply. Multichannel peak checks currently downmix before measurement.

Resolve SourceClips through MasterMob/SourceMob references and explicit descriptors. Display names and filenames are diagnostic information, not classification/removal rules. Timing uses shared exact-rate helpers; unresolved windows are retained and reported.

YAMNet is optional and loaded only when needed. Stronger sustained music is not overridden by a moderate generic Speech spike without concrete dialog or decisive speech evidence. This is heuristic, not proof that a clip contains no speech.

## Layout preservation

Retained source windows and visible montage positions must not change. Inter-track moves require matching clocks/origins and compatible rendering effects. Constant pan is independent of total track length; automation is not.

Whole-track ordering preserves SlotIDs, effects and stable channel order. Protected, mixed-class and unknown-containing tracks keep their positions. They bound same-class peers; unrelated known-only tracks may cross them. Final ordering also applies to plain sequences. Aligned groups remain ordered, and compaction repeats until no permitted improvement remains.

Raw serialized offsets can differ when transitions are represented on different tracks; preservation is judged by visible edit positions, source windows and transition semantics.

Metadata is not universally byte-for-byte immutable: processing changes timeline structure and may update presentation track numbers, locators or compatibility descriptors when required. Untouched content and protected structures should be preserved. Compare intended invariants rather than whole-file hashes of input versus output.

## Configuration

`FilterConfig.media_search_roots=()` explicitly disables environment fallback; `None` retains fallback lookup. Local configuration and model/native-tool locations belong outside versioned defaults. Original media is never a cleanup target.
