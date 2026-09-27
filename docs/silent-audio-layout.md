# Safe silence in lane layout

The layout layer treats supported gain-of-silence trees as gaps, using an
explicit plan shared by the PyAAF2 and SDK writers. The rule is implemented in
aaf_io/audio_silence.py and also used by final PyAAF2 workflow cleanup.

## Supported trees and boundaries

A sound Filler is silent. MonoAudioGain or StereoAudioGain over exactly one
same-duration silent input is also silent. Nested supported gain operations
are accepted. Recognition uses OperationDef AUIDs, not names, source filenames,
clip labels, inferred classes or audio peak thresholds. Negative/mismatched
lengths, non-sound inputs, live SourceClips, unknown operations, missing inputs,
invalid operation metadata and excessive/cyclic nesting do not prove silence.

Replacement additionally requires that the top-level wrapper has no adjacent
Transition. An operation can be silent while its surrounding transition still
owns timing or effects. These transition-dependent structures remain protected
until their normalization has an explicit transition migration contract.
The lane protection is reported in the processing log. This conservative
boundary also applies in final cleanup, so a protected effect cannot be removed
by a later stage.

## Plan and write contract

Read-only analysis appends a LaneLayoutEvent with kind=silence, its lane,
top-level component index, raw position, duration, and a structural signature.
This does not classify or delete any additional audible clip.

Both writers require the addressed component to match the signature, length,
position, lane and transition context. A stale or unsupported replacement
fails explicitly. PyAAF2 interprets the validated occurrence as a gap; SDK
marks the occurrence covered and emits ordinary Sound Fillers around retained
events. Unknown/unplanned operations remain structural payload.

SDK uses canonical dictionary identifiers for standard gain definitions.
Its adapter and the PyAAF2 adapter share the recursive signature and context
rules. SDK can create a standard Sound Filler if no template exists.

Existing rollback and work-copy ownership remain unchanged. Caller-owned
input AAFs are not modified by the pipeline.

## Duration accounting

PyAAF2 padding now uses the declared visible duration after serializing the
actual transitions. It no longer pads to the original sum of raw components,
which becomes incorrect when transitions move. A regression transferring a
five-unit transition and ten-unit clip between two ninety-unit lanes previously
produced declared lengths 100 and 80; it now preserves 90 and 90.

## Verification

tests/test_silent_audio_layout.py covers the original frozen-lane failure,
operation identity versus display name, nested/malformed/live inputs, explicit
writer proofs, stale-plan rejection, transition protection, unknown payload
preservation, final cleanup, lack of a Filler template, declared durations, and
quiet removal followed by layout and a stable repeated run.

The real-sample investigation and later validation are documented separately:
docs/xfades-speech-lane-analysis-2026-09-23.md and
docs/silent-audio-layout-validation-2026-09-23.md.
