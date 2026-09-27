from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
import urllib.error
import builtins
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
for p in (repo_root, repo_root / "aaf_io", repo_root / "aaf_speech_filter"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from aaf_speech_filter import speech_yamnet as speech_yamnet_mod  # noqa: E402
from aaf_speech_filter import aaf_yamnet_lane_layout as lane_layout_mod  # noqa: E402
from aaf_speech_filter.lane_layout_model import (  # noqa: E402
    TimelinePosition,
    event_owned_raw_transition_spans,
    event_owned_transition_history_spans,
    events_share_same_kind_transition_edge,
    planned_overlap_is_allowed_transition_overlap,
)
from aaf_speech_filter.lane_layout_writer import (  # noqa: E402
    write_lane_layout_with_sdk_or_pyaaf2_fallback,
)
from aaf_speech_filter.lane_layout_zones import (  # noqa: E402
    class_zone_lane_order,
    class_zone_lane_preference,
    lane_zone_preferences,
)
from aaf_speech_filter.lane_layout_timing import (  # noqa: E402
    raw_from_visible_with_transition_spans,
    spans_without_exact_matches,
)
from aaf_speech_filter.lane_layout_plan import (  # noqa: E402
    repair_lane_layout_plan_overlaps,
    validate_lane_layout_plan,
)
from aaf_speech_filter.lane_layout_groups import (  # noqa: E402
    aligned_kind_group_key,
    aligned_layout_group_key,
    cross_lane_event_groups,
    aligned_group_source_order,
    aligned_group_target_lanes,
    aligned_group_target_order,
    dominant_aligned_group_kind,
    event_source_window_key,
    group_events_by_optional_key,
)
from aaf_speech_filter.lane_layout_candidates import candidate_group_lane_sequences  # noqa: E402
from aaf_speech_filter.lane_layout_placement import (  # noqa: E402
    EventPlacementSpan,
    event_owned_visible_transition_spans,
    event_placement_length,
    event_placement_start,
    event_target_raw_start_for_lane,
    event_visible_occupancy_length,
)
from aaf_speech_filter.lane_layout_occupancy import (  # noqa: E402
    LaneOccupancyState,
    SourceSpanTracker,
    lane_occupancy_conflicts,
)
from aaf_speech_filter.lane_layout_snapshot import build_lane_occupancy_snapshot  # noqa: E402
from aaf_speech_filter.lane_layout_xfade import (  # noqa: E402
    preserve_same_kind_xfade_chains,
    same_kind_chain_candidate_lanes,
    same_kind_transition_chains,
)
from aaf_speech_filter.lane_layout_constraints import target_lane_allowed_for_event  # noqa: E402
from aaf_speech_filter.lane_layout_conflicts import target_lane_visible_conflicts  # noqa: E402
from aaf_speech_filter.lane_layout_compaction import compact_events_to_preferred_lanes  # noqa: E402
from aaf_speech_filter.lane_layout_initial_placement import LaneInitialPlacementPlanner  # noqa: E402
from aaf_speech_filter.lane_layout_speech_swap import promote_longer_speech_events  # noqa: E402
from aaf_speech_filter.lane_layout_aligned_compaction import (  # noqa: E402
    compact_aligned_groups_to_visible_gaps,
)
from aaf_speech_filter.lane_layout_aligned_order import (  # noqa: E402
    preserve_aligned_target_order_after_compaction,
)
from aaf_speech_filter.lane_layout_final_compaction import (  # noqa: E402
    final_bounded_compact_class_events,
)
from aaf_speech_filter.speech_yamnet import YamnetConfig, yamnet_clip_kind_with_scores  # noqa: E402
from aaf_speech_filter.config import FilterConfig  # noqa: E402
from aaf_speech_filter.aaf_yamnet_lane_layout import (  # noqa: E402
    _assign_target_lanes,
    _build_sdk_lane_layout_events,
    _classify_timeline_block,
    _class_zone_lane_order,
    _clip_name,
    _compact_rebuilt_parts_to_class_zones_without_visible_time_shift,
    _event_owned_transition_indices,
    _lane_layout_mutable,
    _lane_layout_signature,
    _rebalance_rebuilt_event_parts_to_visible_times,
    _rebuilt_lane_declared_length,
    _rebuilt_lane_total_length,
    _resolve_rebuilt_event_overlaps_without_visible_time_shift,
    _sequence_transition_protected_groups,
    _sequence_transition_protected_parts,
    _sdk_xml_event_preferred_transition_offset,
    _trim_leading_rebuilt_transitions_to_cursor,
    _write_lane_layout_with_sdk_or_pyaaf2_fallback,
    _transition_part_is_serializable,
    _validate_pyaaf2_lane_layout_output_for_sdk_open,
    _validate_lane_layout_plan,
    apply_experimental_yamnet_lane_layout,
)


class _Scores:
    def __init__(self, values: np.ndarray):
        self._values = values

    def numpy(self) -> np.ndarray:
        return self._values


class _FakeModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 24] = 0.90  # Music class.
        scores[0, 0] = 0.001
        scores[0, 65] = 0.002
        return _Scores(scores), None, None


class _FakeCarSignalModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 494] = 1.00  # Silence / signal gap.
        scores[0, 498] = 0.43  # Sound effect.
        scores[0, 5] = 0.001  # Sub-threshold speech-like false-positive.
        scores[0, 132] = 0.015
        return _Scores(scores), None, None


class _FakeGenericSpeechSfxModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.99  # Speech evidence belongs to this clip only.
        scores[0, 3] = 0.055  # Concrete speech evidence supports generic Speech.
        scores[0, 504] = 0.032  # Non-speech scene evidence.
        scores[0, 67] = 0.026
        return _Scores(scores), None, None


class _FakeGenericSpeechWithDetailModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.99
        scores[0, 3] = 0.055  # Concrete speech subclass supports generic Speech.
        scores[0, 504] = 0.032
        return _Scores(scores), None, None


class _FakeMusicWithTraceSpeechModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.008
        scores[0, 132] = 0.93
        scores[0, 264] = 0.07
        scores[0, 498] = 0.02
        return _Scores(scores), None, None


class _FakeMusicSfxWithTraceSpeechModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.007
        scores[0, 132] = 0.40
        scores[0, 498] = 0.24
        scores[0, 294] = 0.23
        return _Scores(scores), None, None


class _FakeMusicWithGenericOnlySpeechModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.048
        scores[0, 132] = 0.43
        return _Scores(scores), None, None


class _FakeDialogOverMusicModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.20
        scores[0, 3] = 0.055  # Concrete speech subclass preserves dialog ownership.
        scores[0, 132] = 0.72
        return _Scores(scores), None, None


class _FakeVocalMusicWithSpeechModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.22
        scores[0, 24] = 0.72  # Singing.
        scores[0, 249] = 0.60  # Vocal music.
        scores[0, 132] = 0.50
        return _Scores(scores), None, None


class _FakeShortGenericSpeechNoiseModel:
    def __call__(self, _waveform):
        scores = np.zeros((4, 521), dtype=np.float32)
        scores[:, 0] = 0.97
        scores[0, 3] = 0.039  # Sparse concrete subclass support is not stable speech evidence.
        scores[:, 500] = 0.016
        return _Scores(scores), None, None


class _FakeStableLowDetailSpeechModel:
    def __call__(self, _waveform):
        scores = np.zeros((3, 521), dtype=np.float32)
        scores[:, 0] = 0.97
        scores[:2, 3] = 0.018  # Stable low-level subclass evidence in most frames.
        scores[:, 500] = 0.016
        return _Scores(scores), None, None


class _FakeShortGenericSpeechSilenceModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.116
        scores[0, 494] = 0.99
        scores[0, 500] = 0.24
        scores[0, 469] = 0.22
        return _Scores(scores), None, None


class _FakeGenericOnlySpeechNoiseModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.38
        scores[0, 48] = 0.62  # Walk, footsteps.
        scores[0, 67] = 0.45  # Animal.
        scores[0, 500] = 0.29  # Inside, small room.
        scores[0, 132] = 0.006
        return _Scores(scores), None, None


class _FakeSparseDialogSubclassNoiseModel:
    def __call__(self, _waveform):
        scores = np.zeros((14, 521), dtype=np.float32)
        scores[:, 500] = 0.44  # Inside, small room.
        scores[:, 67] = 0.17  # Animal.
        scores[:, 36] = 0.14  # Breathing-like noise.
        scores[:, 41] = 0.12  # Snort.
        scores[:, 0] = np.asarray(
            [
                0.0,
                0.016,
                0.0397,
                0.0163,
                0.0426,
                0.0047,
                0.0039,
                0.0111,
                0.0304,
                0.0051,
                0.0146,
                0.0013,
                0.0,
                0.0061,
            ],
            dtype=np.float32,
        )
        scores[2, 3] = 0.0262  # One weak dialog-like false-positive frame.
        return _Scores(scores), None, None


class _FakeSparseMusicSubclassNoiseModel:
    def __call__(self, _waveform):
        scores = np.zeros((12, 521), dtype=np.float32)
        scores[:, 500] = 0.38
        scores[:, 48] = 0.22
        scores[:, 494] = 0.18
        scores[:, 0] = 0.003
        scores[4, 132] = 0.09  # One weak music-like false-positive frame.
        return _Scores(scores), None, None


class _FakeStrongNonDialogSpeechSubclassNoiseModel:
    def __call__(self, _waveform):
        scores = np.zeros((5, 521), dtype=np.float32)
        scores[:, 500] = 0.28
        scores[:, 504] = 0.22
        scores[:, 277] = 0.18
        scores[:, 0] = 0.018
        scores[2, 11] = 0.35  # Screaming-like non-dialog speech-class false-positive.
        return _Scores(scores), None, None


class _FakeBorderlineGenericSpeechNoiseModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.35
        scores[0, 500] = 0.23
        scores[0, 504] = 0.17
        scores[0, 277] = 0.16
        return _Scores(scores), None, None


class _FakeStrongGenericSpeechVehicleNoiseModel:
    def __call__(self, _waveform):
        scores = np.zeros((3, 521), dtype=np.float32)
        scores[:, 294] = 0.188  # Vehicle.
        scores[:, 301] = 0.16  # Car.
        scores[:, 373] = 0.14  # Keys jangling / mechanical detail.
        scores[1, 0] = 0.6208  # Generic Speech false-positive spike.
        scores[1, 3] = 0.0074  # Trace dialog subclass is not stable dialog evidence.
        return _Scores(scores), None, None


class _FakeShortGenericSpeechWithMechanicalNoiseModel:
    def __call__(self, _waveform):
        scores = np.zeros((3, 521), dtype=np.float32)
        scores[:, 378] = 0.204  # Typing / mechanical detail.
        scores[:, 380] = 0.170  # Computer keyboard.
        scores[:, 485] = 0.060  # Clicking.
        scores[:, 0] = np.asarray([0.0123, 0.7724, 0.4826], dtype=np.float32)
        scores[:, 3] = np.asarray([0.0057, 0.0015, 0.0021], dtype=np.float32)
        return _Scores(scores), None, None


class _FakeDominantGenericSpeechAnimalNoiseModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.956
        scores[0, 82] = 0.42  # Animal/horse-like background.
        scores[0, 83] = 0.38
        scores[0, 132] = 0.003
        return _Scores(scores), None, None


class _FakeDecisiveGenericSpeechWithSilenceNoiseModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.91
        scores[0, 494] = 0.53  # Silence / signal gap can dominate the non-speech group.
        scores[0, 3] = 0.008
        scores[0, 500] = 0.02
        return _Scores(scores), None, None


class _FakeWindowMarkerModel:
    def __call__(self, waveform):
        marker = float(np.asarray(waveform)[0])
        scores = np.zeros((1, 521), dtype=np.float32)
        if marker < 1000.0:
            scores[0, 0] = 0.98
            scores[0, 3] = 0.035
            scores[0, 500] = 0.02
        else:
            scores[0, 48] = 0.62
            scores[0, 500] = 0.42
            scores[0, 0] = 0.04
        return _Scores(scores), None, None


class _FakeMixedSpeechWithStrongNoiseModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.255
        scores[0, 3] = 0.055  # Dialog-specific evidence survives background dominance.
        scores[0, 494] = 0.65
        scores[0, 500] = 0.01
        return _Scores(scores), None, None


class _FakeVocalizationNoiseModel:
    def __call__(self, _waveform):
        scores = np.zeros((1, 521), dtype=np.float32)
        scores[0, 0] = 0.085
        scores[0, 11] = 0.032  # Screaming-like false-positive.
        scores[0, 12] = 0.025  # Whispering-like false-positive.
        scores[0, 48] = 0.35  # Walk, footsteps.
        scores[0, 372] = 0.33  # Zipper (clothing).
        return _Scores(scores), None, None


class Sequence:
    def __init__(self, components=None) -> None:
        self.components = components if components is not None else []


class Filler:
    def __init__(self, length: int) -> None:
        self.length = length


class OperationGroup:
    def __init__(self, length: int) -> None:
        self.length = length


class SourceClip:
    def __init__(
        self,
        length: int,
        *,
        source_id: str = "sid",
        source_track_id: int = 0,
        start: int = 0,
    ) -> None:
        self.length = length
        self.source_id = source_id
        self.source_track_id = source_track_id
        self.start = start


class Transition:
    def __init__(self, length: int) -> None:
        self.length = length


class _Slot:
    media_kind = "Sound"

    def __init__(self) -> None:
        self.segment = Sequence()


class _PanOperationGroup:
    def __init__(self, seq: Sequence, operation_name: str = "Mono Audio Pan") -> None:
        self.segments = [seq]
        self.operation = operation_name


class _Composition:
    mob_id = "synthetic-composition"
    usage = None

    def __init__(self, lanes: int) -> None:
        self.slots = [_Slot() for _ in range(lanes)]


class _Content:
    def __init__(self, lanes: int) -> None:
        self._lanes = lanes

    def compositionmobs(self):
        return [_Composition(self._lanes)]


class _Aaf:
    def __init__(self, lanes: int) -> None:
        self.content = _Content(lanes)


@contextmanager
def _fake_open_aaf_lenient(_path, mode):
    if mode != "r":
        raise AssertionError(f"unexpected write open: {mode}")
    yield _Aaf(2)


class SpeechYamnetTests(unittest.TestCase):
    def test_clip_name_diagnostic_helper_never_raises(self) -> None:
        with mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout.resolve_clip_display_name",
            side_effect=RuntimeError("bad node"),
        ):
            self.assertEqual(_clip_name(object(), object()), "")

    def test_yamnet_model_missing_local_model_does_not_touch_network(self) -> None:
        fake_hub = types.SimpleNamespace(load=mock.Mock())

        with mock.patch.dict(
            os.environ,
            {
                speech_yamnet_mod.YAMNET_MODEL_DIR_ENV: "",
                speech_yamnet_mod.YAMNET_ALLOW_DOWNLOAD_ENV: "",
            },
        ), mock.patch.dict(sys.modules, {"tensorflow_hub": fake_hub}), mock.patch(
            "aaf_speech_filter.speech_yamnet._tfhub_cached_model_dir",
            return_value=None,
        ), mock.patch(
            "aaf_speech_filter.speech_yamnet._clear_corrupt_tfhub_cache",
            return_value=False,
        ):
            speech_yamnet_mod._model = None
            speech_yamnet_mod._model_url = None
            with self.assertRaisesRegex(RuntimeError, "AAF_YAMNET_MODEL_DIR"):
                speech_yamnet_mod._get_yamnet_model(YamnetConfig())

        fake_hub.load.assert_not_called()

    def test_yamnet_model_explicit_download_uses_tfhub_url_when_local_missing(self) -> None:
        fake_model = object()
        fake_hub = types.SimpleNamespace(load=mock.Mock(return_value=fake_model))

        with mock.patch.dict(
            os.environ,
            {
                speech_yamnet_mod.YAMNET_MODEL_DIR_ENV: "",
                speech_yamnet_mod.YAMNET_ALLOW_DOWNLOAD_ENV: "",
            },
        ), mock.patch.dict(sys.modules, {"tensorflow_hub": fake_hub}), mock.patch(
            "aaf_speech_filter.speech_yamnet._ensure_pkg_resources_for_tensorflow_hub"
        ), mock.patch(
            "aaf_speech_filter.speech_yamnet._tfhub_cached_model_dir",
            return_value=None,
        ), mock.patch(
            "aaf_speech_filter.speech_yamnet._clear_corrupt_tfhub_cache",
            return_value=False,
        ):
            speech_yamnet_mod._model = None
            speech_yamnet_mod._model_url = None
            model = speech_yamnet_mod._get_yamnet_model(
                YamnetConfig(allow_download=True)
            )

        self.assertIs(model, fake_model)
        fake_hub.load.assert_called_once_with(YamnetConfig().hub_url)

    def test_yamnet_model_network_failure_reports_local_model_requirement(self) -> None:
        fake_hub = types.SimpleNamespace(
            load=mock.Mock(side_effect=urllib.error.URLError("ssl eof"))
        )

        with mock.patch.dict(
            os.environ,
            {
                speech_yamnet_mod.YAMNET_MODEL_DIR_ENV: "",
                speech_yamnet_mod.YAMNET_ALLOW_DOWNLOAD_ENV: "1",
            },
        ), mock.patch.dict(sys.modules, {"tensorflow_hub": fake_hub}), mock.patch(
            "aaf_speech_filter.speech_yamnet._ensure_pkg_resources_for_tensorflow_hub"
        ), mock.patch(
            "aaf_speech_filter.speech_yamnet._tfhub_cached_model_dir",
            return_value=None,
        ), mock.patch(
            "aaf_speech_filter.speech_yamnet._clear_corrupt_tfhub_cache",
            return_value=False,
        ):
            speech_yamnet_mod._model = None
            speech_yamnet_mod._model_url = None
            with self.assertRaisesRegex(RuntimeError, "AAF_YAMNET_MODEL_DIR"):
                speech_yamnet_mod._get_yamnet_model(YamnetConfig())

    def test_clip_kind_checks_model_handle_before_importing_tensorflow(self) -> None:
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "tensorflow":
                raise AssertionError("tensorflow imported before model availability check")
            return real_import(name, *args, **kwargs)

        with mock.patch(
            "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
            return_value=(b"\x00\x00" * 16000, 16000),
        ), mock.patch(
            "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
            return_value=np.ones(16000, dtype=np.float32),
        ), mock.patch(
            "aaf_speech_filter.speech_yamnet._get_yamnet_model",
            side_effect=speech_yamnet_mod.YamnetModelUnavailableError(
                "YAMNet model is not available locally."
            ),
        ), mock.patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(RuntimeError, "YAMNet model is not available"):
                yamnet_clip_kind_with_scores(
                    Path("dialog.wav"),
                    0.0,
                    1.0,
                    YamnetConfig(clip_samples=1),
                )

    def test_yamnet_scores_are_used_directly_not_softmaxed(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeModel(),
            ):
                kind, speech, music, noise = yamnet_clip_kind_with_scores(
                    Path("music.wav"),
                    0.0,
                    1.0,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "music")
        self.assertGreater(music, 0.80)
        self.assertLess(speech, 0.05)
        self.assertLess(noise, 0.05)

    def test_strong_sfx_noise_beats_subthreshold_speech_score(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeCarSignalModel(),
            ):
                kind, speech, music, noise = yamnet_clip_kind_with_scores(
                    Path("car-signal.wav"),
                    0.0,
                    1.0,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "noise")
        self.assertGreater(noise, 0.90)
        self.assertLess(speech, YamnetConfig().score_threshold)

    def test_reliable_above_threshold_speech_keeps_clip_in_speech(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeGenericSpeechSfxModel(),
            ):
                kind, speech, music, noise = yamnet_clip_kind_with_scores(
                    Path("sfx.wav"),
                    0.0,
                    1.0,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "speech")
        self.assertGreater(speech, 0.90)
        self.assertGreater(noise, 0.02)

    def test_music_dominance_beats_trace_speech_score(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeMusicWithTraceSpeechModel(),
            ):
                kind, speech, music, noise = yamnet_clip_kind_with_scores(
                    Path("music.wav"),
                    0.0,
                    1.0,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "music")
        self.assertGreater(speech, YamnetConfig().score_threshold)
        self.assertGreater(music, 0.80)

    def test_music_sfx_dominance_beats_trace_speech_score(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeMusicSfxWithTraceSpeechModel(),
            ):
                kind, speech, music, noise = yamnet_clip_kind_with_scores(
                    Path("accent.wav"),
                    0.0,
                    1.0,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "music")
        self.assertGreater(speech, YamnetConfig().score_threshold)
        self.assertGreater(music, 0.15)
        self.assertGreater(noise, 0.15)

    def test_music_dominance_beats_generic_only_speech_score(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeMusicWithGenericOnlySpeechModel(),
            ):
                kind, speech, music, _noise = yamnet_clip_kind_with_scores(
                    Path("music.wav"),
                    0.0,
                    1.0,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "music")
        self.assertGreater(speech, YamnetConfig().score_threshold)
        self.assertGreater(music, 5.0 * speech)

    def test_concrete_dialog_over_music_remains_speech(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeDialogOverMusicModel(),
            ):
                kind, speech, music, _noise = yamnet_clip_kind_with_scores(
                    Path("dialog-over-music.wav"),
                    0.0,
                    1.0,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "speech")
        self.assertGreater(music, speech)

    def test_vocal_music_with_generic_speech_score_stays_music(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeVocalMusicWithSpeechModel(),
            ):
                kind, speech, music, noise = yamnet_clip_kind_with_scores(
                    Path("song.wav"),
                    0.0,
                    1.0,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "music")
        self.assertGreater(speech, 0.20)
        self.assertGreater(music, 0.60)

    def test_short_generic_only_speech_is_not_downgraded_to_noise(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeShortGenericSpeechNoiseModel(),
            ):
                kind, speech, music, noise = yamnet_clip_kind_with_scores(
                    Path("short-sfx.wav"),
                    0.0,
                    1.96,
                    YamnetConfig(),
                )

        self.assertEqual(kind, "speech")
        self.assertGreater(speech, 0.90)
        self.assertLess(music, 0.08)

    def test_short_stable_low_detail_speech_remains_speech(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeStableLowDetailSpeechModel(),
            ):
                kind, speech, music, _noise = yamnet_clip_kind_with_scores(
                    Path("short-speech.wav"),
                    0.0,
                    1.96,
                    YamnetConfig(),
                )

        self.assertEqual(kind, "speech")
        self.assertGreater(speech, 0.90)
        self.assertLess(music, 0.08)

    def test_short_stable_generic_speech_does_not_depend_on_secondary_vad(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeStableLowDetailSpeechModel(),
            ):
                kind, speech, music, _noise = yamnet_clip_kind_with_scores(
                    Path("short-no-voice.wav"),
                    0.0,
                    1.96,
                    YamnetConfig(),
                )

        self.assertEqual(kind, "speech")
        self.assertGreater(speech, 0.90)
        self.assertLess(music, 0.08)

    def test_trace_generic_speech_under_strong_noise_stays_noise(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeShortGenericSpeechSilenceModel(),
            ):
                kind, speech, music, noise = yamnet_clip_kind_with_scores(
                    Path("short-silence-sfx.wav"),
                    0.0,
                    3.72,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "noise")
        self.assertGreater(noise, 0.25)

    def test_strong_generic_only_speech_under_noise_stays_noise(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeGenericOnlySpeechNoiseModel(),
            ):
                kind, speech, music, noise = yamnet_clip_kind_with_scores(
                    Path("generic-only-noise.wav"),
                    0.0,
                    3.64,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "noise")
        self.assertGreater(speech, 0.20)
        self.assertGreater(noise, 0.25)

    def test_sparse_dialog_subclass_under_dominant_noise_stays_noise(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeSparseDialogSubclassNoiseModel(),
            ):
                kind, speech, _music, noise = yamnet_clip_kind_with_scores(
                    Path("sparse-dialog-subclass-noise.wav"),
                    0.0,
                    7.08,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "noise")
        self.assertGreater(noise, 0.25)
        self.assertGreater(speech, YamnetConfig().score_threshold)

    def test_sparse_music_subclass_under_dominant_noise_stays_noise(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeSparseMusicSubclassNoiseModel(),
            ):
                kind, _speech, music, noise = yamnet_clip_kind_with_scores(
                    Path("sparse-music-subclass-noise.wav"),
                    0.0,
                    6.0,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "noise")
        self.assertGreater(noise, 0.25)
        self.assertLess(music, 0.02)

    def test_strong_non_dialog_speech_subclass_under_noise_stays_noise(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeStrongNonDialogSpeechSubclassNoiseModel(),
            ):
                kind, speech, _music, noise = yamnet_clip_kind_with_scores(
                    Path("strong-non-dialog-speech-class-noise.wav"),
                    0.0,
                    2.5,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "noise")
        self.assertGreater(speech, 0.20)
        self.assertGreater(noise, 0.20)

    def test_borderline_generic_speech_over_noise_stays_noise_without_dialog_support(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeBorderlineGenericSpeechNoiseModel(),
            ):
                kind, speech, _music, noise = yamnet_clip_kind_with_scores(
                    Path("borderline-generic-speech-noise.wav"),
                    0.0,
                    1.0,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "noise")
        self.assertGreater(speech, 0.20)
        self.assertGreater(noise, 0.20)

    def test_strong_generic_speech_over_vehicle_noise_stays_noise_without_dialog_support(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeStrongGenericSpeechVehicleNoiseModel(),
            ):
                kind, speech, _music, noise = yamnet_clip_kind_with_scores(
                    Path("generic-speech-vehicle-noise.wav"),
                    0.0,
                    6.54,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "noise")
        self.assertGreater(speech, 0.35)
        self.assertGreater(noise, 0.15)

    def test_decisive_generic_speech_over_silence_noise_is_speech(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeDecisiveGenericSpeechWithSilenceNoiseModel(),
            ):
                kind, speech, _music, noise = yamnet_clip_kind_with_scores(
                    Path("decisive-generic-speech-with-silence-noise.wav"),
                    0.0,
                    1.72,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "speech")
        self.assertGreater(speech, 0.80)
        self.assertGreater(noise, 0.50)

    def test_short_dominant_generic_speech_over_moderate_mechanical_noise_is_speech(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeShortGenericSpeechWithMechanicalNoiseModel(),
            ):
                kind, speech, _music, noise = yamnet_clip_kind_with_scores(
                    Path("short-generic-speech-mechanical-noise.wav"),
                    0.0,
                    2.12,
                    YamnetConfig(),
                )

        self.assertEqual(kind, "speech")
        self.assertGreater(speech, 0.70)
        self.assertGreater(noise, 0.15)

    def test_dominant_generic_speech_over_noise_stays_speech(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeDominantGenericSpeechAnimalNoiseModel(),
            ):
                kind, speech, _music, noise = yamnet_clip_kind_with_scores(
                    Path("dominant-generic-speech-with-background.wav"),
                    0.0,
                    3.64,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "speech")
        self.assertGreater(speech, noise)

    def test_long_clip_speech_detection_uses_temporal_coverage_not_only_peak_windows(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        def fake_read(_path: Path, start: float, _duration: float):
            marker = 100 if 4.0 <= float(start) <= 6.0 else 20000
            return np.asarray([marker, 0, 0, 0], dtype=np.int16).tobytes(), 16000

        def fake_to_float32(pcm: bytes, _sample_rate: int):
            marker = int(np.frombuffer(pcm, dtype=np.int16)[0])
            return np.asarray([float(marker)] * 16000, dtype=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                side_effect=fake_read,
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                side_effect=fake_to_float32,
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeWindowMarkerModel(),
            ):
                kind, speech, _music, noise = yamnet_clip_kind_with_scores(
                    Path("quiet-speech-among-louder-noise.wav"),
                    0.0,
                    10.0,
                    YamnetConfig(clip_samples=3, clip_edge_trim_sec=0.0),
                )

        self.assertEqual(kind, "speech")
        self.assertGreater(speech, noise)

    def test_mixed_speech_with_strong_noise_remains_speech(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeMixedSpeechWithStrongNoiseModel(),
            ):
                kind, speech, music, noise = yamnet_clip_kind_with_scores(
                    Path("long-speech-with-noise.wav"),
                    0.0,
                    4.24,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "speech")
        self.assertGreater(noise, 0.25)

    def test_vocalization_subclasses_do_not_prove_dialog_over_dominant_noise(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeVocalizationNoiseModel(),
            ):
                kind, speech, _music, noise = yamnet_clip_kind_with_scores(
                    Path("vocalization-noise.wav"),
                    0.0,
                    5.16,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "noise")
        self.assertGreater(speech, YamnetConfig().score_threshold)
        self.assertGreater(noise, speech)

    def test_generic_speech_with_detail_remains_speech(self) -> None:
        fake_tf = types.SimpleNamespace(convert_to_tensor=lambda audio, dtype=None: audio, float32=np.float32)

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            with mock.patch(
                "aaf_speech_filter.speech_yamnet.read_media_segment_pcm16_mono",
                return_value=(b"\x00\x00" * 16000, 16000),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._pcm16_mono_to_float32_16k",
                return_value=np.ones(16000, dtype=np.float32),
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet._get_yamnet_model",
                return_value=_FakeGenericSpeechWithDetailModel(),
            ):
                kind, speech, music, noise = yamnet_clip_kind_with_scores(
                    Path("dialog.wav"),
                    0.0,
                    1.0,
                    YamnetConfig(clip_samples=1),
                )

        self.assertEqual(kind, "speech")
        self.assertGreater(speech, 0.90)

    def test_lane_block_classification_preserves_speech_from_clip_classifier(self) -> None:
        class SourceClip:
            pass

        with mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout._resolve_wave_path_for_sourceclip",
            return_value=Path(__file__),
        ), mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout._sourceclip_audio_timing",
            return_value=(2.0, 52.0),
        ), mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout.yamnet_clip_kind_with_scores",
            return_value=("speech", 0.16, 0.01, 0.50),
        ):
            kind = _classify_timeline_block(
                object(),
                SourceClip(),
                {},
                YamnetConfig(),
                25.0,
                lambda: None,
            )

        self.assertEqual(kind, "speech")

    def test_lane_block_classification_does_not_downgrade_speech_to_music(self) -> None:
        class SourceClip:
            pass

        with mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout._resolve_wave_path_for_sourceclip",
            return_value=Path(__file__),
        ), mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout._sourceclip_audio_timing",
            return_value=(2.0, 52.0),
        ), mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout.yamnet_clip_kind_with_scores",
            return_value=("speech", 0.04, 0.05, 0.01),
        ):
            kind = _classify_timeline_block(
                object(),
                SourceClip(),
                {},
                YamnetConfig(),
                25.0,
                lambda: None,
            )

        self.assertEqual(kind, "speech")

    def test_lane_block_classification_without_media_does_not_require_yamnet_model(self) -> None:
        class SourceClip:
            pass

        with mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout._resolve_wave_path_for_sourceclip",
            return_value=None,
        ), mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout.yamnet_clip_kind_with_scores",
            side_effect=RuntimeError("YAMNet model unavailable"),
        ) as classify:
            kind = _classify_timeline_block(
                object(),
                SourceClip(),
                {},
                YamnetConfig(),
                25.0,
                lambda: None,
            )

        self.assertEqual(kind, "unknown")
        classify.assert_not_called()

    def test_lane_block_classification_with_media_requires_yamnet_model(self) -> None:
        class SourceClip:
            pass

        with mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout._resolve_wave_path_for_sourceclip",
            return_value=Path(__file__),
        ), mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout._sourceclip_audio_timing",
            return_value=(2.0, 52.0),
        ), mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout.yamnet_clip_kind_with_scores",
            side_effect=RuntimeError("YAMNet model unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "YAMNet model unavailable"):
                _classify_timeline_block(
                    object(),
                    SourceClip(),
                    {},
                    YamnetConfig(),
                    25.0,
                    lambda: None,
                )

    def test_lane_block_classification_clamps_analysis_to_outer_visible_duration(self) -> None:
        class InnerSourceClip:
            start = 50
            length = 100

        inner = InnerSourceClip()
        outer = OperationGroup(25)
        outer.segments = [inner]

        with mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout._resolve_wave_path_for_sourceclip",
            return_value=Path(__file__),
        ), mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout._sourceclip_audio_timing",
            return_value=(2.0, 1.0),
        ) as timing, mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout.yamnet_clip_kind_with_scores",
            return_value=("noise", 0.0, 0.0, 1.0),
        ):
            kind = _classify_timeline_block(
                object(),
                outer,
                {},
                YamnetConfig(),
                25.0,
                lambda: None,
            )

        self.assertEqual(kind, "noise")
        self.assertEqual(timing.call_args.kwargs["visible_duration_sec"], 1.0)

    def test_lane_assignment_keeps_aligned_channel_kinds_independent(self) -> None:
        events = [
            {"src_lane": 5, "T": 100, "T_edit": 100, "L": 40, "kind": "speech"},
            {"src_lane": 6, "T": 100, "T_edit": 100, "L": 40, "kind": "noise"},
            {"src_lane": 7, "T": 100, "T_edit": 100, "L": 40, "kind": "music"},
        ]

        _assign_target_lanes(events, 9)

        self.assertEqual([e["kind"] for e in events], ["speech", "noise", "music"])
        self.assertEqual([e["target_lane"] for e in events], [0, 4, 8])
        self.assertEqual([e["target_T"] for e in events], [100, 100, 100])

    def test_sdk_layout_events_preserve_visible_edit_start(self) -> None:
        events = [
            {
                "src_lane": 2,
                "top_idx": 5,
                "T": 1200,
                "T_edit": 1188,
                "target_T": 1200,
                "L": 40,
                "target_lane": 0,
                "kind": "speech",
            }
        ]

        layout_events = _build_sdk_lane_layout_events(events, [])

        self.assertEqual(len(layout_events), 1)
        self.assertEqual(layout_events[0].T_edit, 1200)
        self.assertEqual(layout_events[0].visible_t, 1188)

    def test_lane_layout_uses_pyaaf2_writer_before_sdk_xml(self) -> None:
        events = [{"src_lane": 1, "top_idx": 0, "T": 10, "L": 5, "target_lane": 0}]
        result: dict[str, object] = {}

        with tempfile.TemporaryDirectory() as tmp:
            input_aaf = Path(tmp) / "input.aaf"
            output_aaf = Path(tmp) / "output.aaf"
            input_aaf.write_bytes(b"fake-aaf")

            with mock.patch(
                "aaf_speech_filter.aaf_yamnet_lane_layout.apply_lane_layout_via_aaf_sdk_xml",
                side_effect=RuntimeError("aaffmtconv -xml failed (3221226519): (no output)"),
            ) as sdk_writer, mock.patch(
                "aaf_speech_filter.aaf_yamnet_lane_layout._apply_lane_layout_via_pyaaf2_rebuild",
                return_value=1,
            ) as pyaaf2_writer, mock.patch(
                "aaf_speech_filter.aaf_yamnet_lane_layout._validate_pyaaf2_lane_layout_output_for_sdk_open",
                return_value=None,
            ):
                moved = _write_lane_layout_with_sdk_or_pyaaf2_fallback(
                    input_aaf=input_aaf,
                    output_aaf=output_aaf,
                    events=events,
                    structural_events=[],
                    n_lanes=3,
                    cfg=FilterConfig(experimental_yamnet_lane_layout=True),
                    runtime_essence_paths={},
                    work_dir=None,
                    cancel_check=lambda: None,
                    log_callback=None,
                    result_out=result,
                )

        self.assertEqual(moved, 1)
        sdk_writer.assert_not_called()
        pyaaf2_writer.assert_called_once()
        self.assertTrue(result["pyaaf2_lane_layout_primary"])

    def test_lane_layout_logs_before_pyaaf2_writer_starts(self) -> None:
        events = [{"src_lane": 1, "top_idx": 0, "T": 10, "L": 5, "target_lane": 0}]
        logs: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            input_aaf = Path(tmp) / "input.aaf"
            output_aaf = Path(tmp) / "output.aaf"
            input_aaf.write_bytes(b"fake-aaf")

            with mock.patch(
                "aaf_speech_filter.aaf_yamnet_lane_layout.apply_lane_layout_via_aaf_sdk_xml",
                return_value=None,
            ) as sdk_writer, mock.patch(
                "aaf_speech_filter.aaf_yamnet_lane_layout._apply_lane_layout_via_pyaaf2_rebuild",
                return_value=1,
            ) as pyaaf2_writer, mock.patch(
                "aaf_speech_filter.aaf_yamnet_lane_layout._validate_pyaaf2_lane_layout_output_for_sdk_open",
                return_value=None,
            ):
                moved = _write_lane_layout_with_sdk_or_pyaaf2_fallback(
                    input_aaf=input_aaf,
                    output_aaf=output_aaf,
                    events=events,
                    structural_events=[],
                    n_lanes=3,
                    cfg=FilterConfig(experimental_yamnet_lane_layout=True),
                    runtime_essence_paths={},
                    work_dir=None,
                    cancel_check=lambda: None,
                    log_callback=logs.append,
                    result_out={},
                )

        self.assertEqual(moved, 1)
        sdk_writer.assert_not_called()
        pyaaf2_writer.assert_called_once()
        self.assertTrue(any("PyAAF2" in line for line in logs))

    def test_writer_facade_prefers_pyaaf2_to_preserve_container(self) -> None:
        events = [{"src_lane": 1, "top_idx": 0, "T": 10, "L": 5, "target_lane": 0}]
        result: dict[str, object] = {}
        sdk_calls: list[dict[str, object]] = []
        pyaaf2_calls: list[dict[str, object]] = []

        def sdk_writer(**kwargs):
            sdk_calls.append(kwargs)

        def pyaaf2_writer(**kwargs):
            pyaaf2_calls.append(kwargs)
            return 1

        with tempfile.TemporaryDirectory() as tmp:
            input_aaf = Path(tmp) / "in.aaf"
            output_aaf = Path(tmp) / "out.aaf"
            input_aaf.write_bytes(b"source")

            moved = write_lane_layout_with_sdk_or_pyaaf2_fallback(
                input_aaf=input_aaf,
                output_aaf=output_aaf,
                events=events,
                structural_events=[],
                n_lanes=3,
                cfg=FilterConfig(experimental_yamnet_lane_layout=True),
                runtime_essence_paths={},
                work_dir=None,
                cancel_check=lambda: None,
                log_callback=lambda _msg: None,
                result_out=result,
                progress_callback=None,
                build_sdk_events=lambda ev, structural: list(ev) + list(structural),
                class_zone_lane_order=lambda kind, n: list(range(n)),
                pyaaf2_rebuild=pyaaf2_writer,
                sdk_xml_writer=sdk_writer,
                sdk_exporter_unavailable_error=lambda _exc: False,
            )
            output_bytes = output_aaf.read_bytes()

        self.assertEqual(moved, 1)
        self.assertEqual(len(pyaaf2_calls), 1)
        self.assertEqual(len(sdk_calls), 0)
        self.assertEqual(output_bytes, b"source")
        self.assertFalse(result["sdk_xml_primary"])
        self.assertTrue(result["pyaaf2_lane_layout_primary"])

    def test_writer_facade_passes_immutable_lanes_to_writers(self) -> None:
        events = [{"src_lane": 1, "top_idx": 0, "T": 10, "L": 5, "target_lane": 0}]
        result: dict[str, object] = {}
        sdk_calls: list[dict[str, object]] = []
        pyaaf2_calls: list[dict[str, object]] = []

        def sdk_writer(**kwargs):
            sdk_calls.append(kwargs)

        def pyaaf2_writer(**kwargs):
            pyaaf2_calls.append(kwargs)
            raise RuntimeError("pyaaf2 rebuild failed")

        with tempfile.TemporaryDirectory() as tmp:
            input_aaf = Path(tmp) / "in.aaf"
            output_aaf = Path(tmp) / "out.aaf"
            input_aaf.write_bytes(b"source")

            write_lane_layout_with_sdk_or_pyaaf2_fallback(
                input_aaf=input_aaf,
                output_aaf=output_aaf,
                events=events,
                structural_events=[],
                n_lanes=3,
                cfg=FilterConfig(experimental_yamnet_lane_layout=True),
                runtime_essence_paths={},
                work_dir=None,
                cancel_check=lambda: None,
                log_callback=lambda _msg: None,
                result_out=result,
                progress_callback=None,
                build_sdk_events=lambda ev, structural: list(ev) + list(structural),
                class_zone_lane_order=lambda kind, n: list(range(n)),
                pyaaf2_rebuild=pyaaf2_writer,
                sdk_xml_writer=sdk_writer,
                sdk_exporter_unavailable_error=lambda exc: "aaffmtconv -xml failed" in str(exc),
                immutable_lanes={2},
            )

        self.assertEqual(sdk_calls[0]["immutable_lanes"], {2})
        self.assertEqual(pyaaf2_calls[0]["immutable_lanes"], {2})

    def test_writer_facade_falls_back_to_sdk_when_pyaaf2_fails(self) -> None:
        events = [{"src_lane": 1, "top_idx": 0, "T": 10, "L": 5, "target_lane": 0}]
        result: dict[str, object] = {}
        sdk_calls: list[dict[str, object]] = []

        def sdk_writer(**kwargs):
            sdk_calls.append(kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            input_aaf = Path(tmp) / "in.aaf"
            output_aaf = Path(tmp) / "out.aaf"
            input_aaf.write_bytes(b"source")

            moved = write_lane_layout_with_sdk_or_pyaaf2_fallback(
                input_aaf=input_aaf,
                output_aaf=output_aaf,
                events=events,
                structural_events=[],
                n_lanes=3,
                cfg=FilterConfig(experimental_yamnet_lane_layout=True),
                runtime_essence_paths={},
                work_dir=None,
                cancel_check=lambda: None,
                log_callback=lambda _msg: None,
                result_out=result,
                progress_callback=None,
                build_sdk_events=lambda ev, structural: list(ev) + list(structural),
                class_zone_lane_order=lambda kind, n: list(range(n)),
                pyaaf2_rebuild=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("pyaaf2 failed")),
                sdk_xml_writer=sdk_writer,
                sdk_exporter_unavailable_error=lambda exc: "aaffmtconv -xml failed" in str(exc),
            )

        self.assertEqual(moved, 1)
        self.assertEqual(len(sdk_calls), 1)
        self.assertFalse(result["sdk_xml_primary"])
        self.assertTrue(result["sdk_xml_fallback"])

    def test_writer_facade_falls_back_to_sdk_when_pyaaf2_output_is_invalid(self) -> None:
        events = [{"src_lane": 1, "top_idx": 0, "T": 10, "L": 5, "target_lane": 0}]
        result: dict[str, object] = {}
        sdk_calls: list[dict[str, object]] = []
        logs: list[str] = []

        def pyaaf2_writer(**kwargs):
            Path(kwargs["aaf_path"]).write_bytes(b"corrupt-pyaaf2-output")
            return 1

        def reject_pyaaf2_output(path: Path) -> None:
            self.assertEqual(path.name, "out.aaf")
            raise RuntimeError("ComAAFInfo exit 1")

        def sdk_writer(**kwargs):
            sdk_calls.append(kwargs)
            Path(kwargs["output_aaf"]).write_bytes(b"sdk-output")

        with tempfile.TemporaryDirectory() as tmp:
            input_aaf = Path(tmp) / "in.aaf"
            output_aaf = Path(tmp) / "out.aaf"
            input_aaf.write_bytes(b"source")

            moved = write_lane_layout_with_sdk_or_pyaaf2_fallback(
                input_aaf=input_aaf,
                output_aaf=output_aaf,
                events=events,
                structural_events=[],
                n_lanes=3,
                cfg=FilterConfig(experimental_yamnet_lane_layout=True),
                runtime_essence_paths={},
                work_dir=None,
                cancel_check=lambda: None,
                log_callback=logs.append,
                result_out=result,
                progress_callback=None,
                build_sdk_events=lambda ev, structural: list(ev) + list(structural),
                class_zone_lane_order=lambda kind, n: list(range(n)),
                pyaaf2_rebuild=pyaaf2_writer,
                sdk_xml_writer=sdk_writer,
                sdk_exporter_unavailable_error=lambda exc: "aaffmtconv -xml failed" in str(exc),
                pyaaf2_output_validator=reject_pyaaf2_output,
            )
            output_bytes = output_aaf.read_bytes()

        self.assertEqual(moved, 1)
        self.assertEqual(len(sdk_calls), 1)
        self.assertEqual(output_bytes, b"sdk-output")
        self.assertFalse(result["pyaaf2_lane_layout_primary"])
        self.assertTrue(result["sdk_xml_fallback"])
        self.assertIn("ComAAFInfo exit 1", result["pyaaf2_failure"])
        self.assertTrue(any("PyAAF2 writer failed" in line for line in logs))

    def test_pyaaf2_output_validator_rejects_when_sdk_binary_conversion_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_aaf = root / "out.aaf"
            work_dir = root / "work"
            output_aaf.write_bytes(b"fake-aaf")

            with mock.patch("aaf_io.sdk_tools.find_comaafinfo", return_value=root / "ComAAFInfo.exe"), mock.patch(
                "aaf_io.sdk_tools.run_comaafinfo",
                return_value=(0, "ok"),
            ), mock.patch("aaf_io.sdk_tools.find_aaffmtconv", return_value=root / "aaffmtconv.exe"), mock.patch(
                "aaf_io.sdk_tools.run_aaffmtconv_to_structured_storage",
                side_effect=RuntimeError("aaffmtconv -ss failed (801201d1)"),
            ) as binary_conv:
                with self.assertRaisesRegex(RuntimeError, "aaffmtconv -ss failed"):
                    _validate_pyaaf2_lane_layout_output_for_sdk_open(
                        output_aaf,
                        work_dir=work_dir,
                        cancel_check=lambda: None,
                    )

        binary_conv.assert_called_once()

    def test_lane_layout_skips_when_fewer_than_three_sound_lanes(self) -> None:
        logs: list[str] = []

        with mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout.open_aaf_lenient",
            side_effect=_fake_open_aaf_lenient,
        ):
            moved = apply_experimental_yamnet_lane_layout(
                Path("two-lanes.aaf"),
                FilterConfig(experimental_yamnet_lane_layout=True),
                runtime_essence_paths={},
                log_callback=logs.append,
            )

        self.assertEqual(moved, 0)
        self.assertTrue(any("≥3" in line for line in logs))

    def test_rebuilt_lane_length_extends_to_moved_clip_end(self) -> None:
        self.assertEqual(_rebuilt_lane_total_length(250, 70733), 70733)
        self.assertEqual(_rebuilt_lane_total_length(70733, 250), 70733)

    def test_rebuilt_lane_declared_length_accounts_for_transition_overlap(self) -> None:
        components = [OperationGroup(20), Transition(2), OperationGroup(30)]

        self.assertEqual(_rebuilt_lane_declared_length(52, components), 48)

    def test_lane_assignment_respects_reserved_transition_intervals(self) -> None:
        events = [
            {"src_lane": 1, "T": 100, "T_edit": 100, "L": 20, "kind": "speech"},
            {"src_lane": 2, "T": 200, "T_edit": 200, "L": 20, "kind": "music"},
        ]

        _assign_target_lanes(events, 3, reserved_by_lane={0: [(90, 130)], 1: [(190, 230)]})

        self.assertEqual(events[0]["target_lane"], 1)
        self.assertEqual(events[1]["target_lane"], 2)

    def test_lane_assignment_respects_original_lane_capacity(self) -> None:
        events = [
            {"src_lane": 1, "T": 100, "T_edit": 100, "L": 20, "kind": "speech"},
            {"src_lane": 2, "T": 200, "T_edit": 200, "L": 20, "kind": "speech"},
        ]

        _assign_target_lanes(events, 3, lane_capacity_by_lane={0: 110, 1: 150, 2: 250})

        self.assertEqual(events[0]["target_lane"], 1)
        self.assertEqual(events[1]["target_lane"], 2)

    def test_lane_assignment_preserves_source_slot_when_no_target_fits(self) -> None:
        events = [
            {"src_lane": 1, "T": 100, "T_edit": 100, "L": 40, "kind": "speech"},
            {"src_lane": 2, "T": 100, "T_edit": 100, "L": 40, "kind": "speech"},
        ]

        _assign_target_lanes(events, 3, lane_capacity_by_lane={0: 90, 1: 200, 2: 200})

        self.assertEqual(events[0]["target_lane"], 1)
        self.assertEqual(events[1]["target_lane"], 2)

    def test_lane_assignment_respects_lane_signature(self) -> None:
        events = [
            {"src_lane": 1, "T": 100, "T_edit": 100, "L": 20, "kind": "speech"},
        ]

        _assign_target_lanes(
            events,
            3,
            lane_signature_by_lane={
                0: ("OperationGroup", "Mono Audio Pan", 200),
                1: ("Sequence", "", 200),
                2: ("Sequence", "", 200),
            },
        )

        self.assertEqual(events[0]["target_lane"], 1)

    def test_lane_assignment_uses_capacity_not_signature_length(self) -> None:
        events = [
            {"src_lane": 1, "T": 100, "T_edit": 100, "L": 20, "kind": "speech"},
        ]

        _assign_target_lanes(
            events,
            3,
            lane_capacity_by_lane={0: 110, 1: 200, 2: 200},
            lane_signature_by_lane={
                0: ("Sequence", ""),
                1: ("Sequence", ""),
                2: ("Sequence", ""),
            },
        )

        self.assertEqual(events[0]["target_lane"], 1)

    def test_name_only_pan_wrapper_is_protected(self) -> None:
        seq = Sequence([SourceClip(20)])
        wrapped_slot = _Slot()
        wrapped_slot.segment = _PanOperationGroup(seq)
        plain_slot = _Slot()
        plain_seq = plain_slot.segment

        self.assertEqual(_lane_layout_signature(wrapped_slot, seq), ("UnsupportedOperation", None, None, 0))
        self.assertEqual(_lane_layout_signature(plain_slot, plain_seq), ("Sequence", "", None, 0))
        self.assertFalse(_lane_layout_mutable(wrapped_slot, seq))
        self.assertTrue(_lane_layout_mutable(plain_slot, plain_seq))

    def test_unknown_wrapped_sequence_lane_remains_immutable_for_layout(self) -> None:
        seq = Sequence([OperationGroup(20)])
        wrapped_slot = _Slot()
        wrapped_slot.segment = _PanOperationGroup(seq, operation_name="Unknown Audio Effect")

        self.assertEqual(
            _lane_layout_signature(wrapped_slot, seq),
            ("UnsupportedOperation", None, None, 0),
        )
        self.assertFalse(_lane_layout_mutable(wrapped_slot, seq))

    def test_lane_assignment_cannot_move_under_unproven_pan_wrapper(self) -> None:
        wrapped_seq = Sequence([SourceClip(20)])
        wrapped_slot = _Slot()
        wrapped_slot.segment = _PanOperationGroup(wrapped_seq)
        plain_slot = _Slot()
        plain_seq = plain_slot.segment
        events = [
            {"src_lane": 1, "T": 100, "T_edit": 100, "L": 20, "kind": "speech"},
        ]

        _assign_target_lanes(
            events,
            2,
            lane_signature_by_lane={
                0: _lane_layout_signature(wrapped_slot, wrapped_seq),
                1: _lane_layout_signature(plain_slot, plain_seq),
            },
            lane_mutable_by_lane={
                0: _lane_layout_mutable(wrapped_slot, wrapped_seq),
                1: _lane_layout_mutable(plain_slot, plain_seq),
            },
        )

        self.assertEqual(events[0]["target_lane"], 1)

    def test_lane_assignment_fallback_uses_source_position_occupancy(self) -> None:
        events = [
            {"src_lane": 0, "T": 100, "T_edit": 90, "L": 20, "kind": "speech"},
            {"src_lane": 0, "T": 120, "T_edit": 110, "L": 20, "kind": "speech"},
        ]

        _assign_target_lanes(events, 1)

        self.assertEqual([e["target_lane"] for e in events], [0, 0])

    def test_lane_assignment_does_not_reserve_movable_source_positions(self) -> None:
        events = [
            {"src_lane": 19, "T": 0, "T_edit": 0, "L": 100, "kind": "speech"},
            {"src_lane": 0, "T": 0, "T_edit": 0, "L": 100, "kind": "music"},
        ]

        _assign_target_lanes(events, 21)

        self.assertEqual(events[0]["target_lane"], 0)
        self.assertEqual(events[1]["target_lane"], 20)

    def test_lane_assignment_keeps_classes_in_explicit_lane_bands(self) -> None:
        events = [
            {"src_lane": 20, "T": 0, "T_edit": 0, "L": 100, "kind": "noise"},
            {"src_lane": 7, "T": 0, "T_edit": 0, "L": 100, "kind": "speech"},
            {"src_lane": 0, "T": 0, "T_edit": 0, "L": 100, "kind": "music"},
        ]

        _assign_target_lanes(events, 21)

        self.assertLess(events[1]["target_lane"], 7)
        self.assertGreaterEqual(events[0]["target_lane"], 7)
        self.assertLess(events[0]["target_lane"], 14)
        self.assertGreaterEqual(events[2]["target_lane"], 14)

    def test_class_zone_order_does_not_fallback_to_opposite_class_zone(self) -> None:
        self.assertEqual(_class_zone_lane_order("speech", 9), [0, 1, 2, 3, 4, 5])
        self.assertEqual(_class_zone_lane_order("noise", 9), [4, 3, 5, 2, 1, 0])
        self.assertEqual(_class_zone_lane_order("music", 9), [8, 7, 6, 5, 4, 3])

    def test_lane_zone_policy_module_matches_planner_zone_contract(self) -> None:
        speech, noise, music, unknown, lanes = lane_zone_preferences(9)

        self.assertEqual(lanes, list(range(9)))
        self.assertEqual(class_zone_lane_order("speech", 9), speech)
        self.assertEqual(class_zone_lane_order("noise", 9), noise)
        self.assertEqual(class_zone_lane_order("music", 9), music)
        self.assertEqual(class_zone_lane_order("unknown", 9), unknown)
        self.assertEqual(class_zone_lane_preference("music", 9, 7)[-1], 7)
        self.assertEqual(_class_zone_lane_order("music", 9), class_zone_lane_order("music", 9))

    def test_candidate_group_lane_sequences_prefers_class_physical_band(self) -> None:
        self.assertEqual(
            candidate_group_lane_sequences("music", list(range(9)), 2, lane_count=9)[0],
            [7, 8],
        )
        self.assertEqual(
            candidate_group_lane_sequences("speech", list(range(9)), 2, lane_count=9)[0],
            [0, 1],
        )
        self.assertEqual(candidate_group_lane_sequences("speech", [0], 2, lane_count=9), [])

    def test_lane_assignment_does_not_fallback_noise_to_music_source_lane(self) -> None:
        events = [
            {"src_lane": 20, "T": 0, "T_edit": 0, "L": 100, "kind": "noise"},
        ]

        _assign_target_lanes(events, 21)

        self.assertGreaterEqual(events[0]["target_lane"], 7)
        self.assertLess(events[0]["target_lane"], 14)

    def test_lane_assignment_uses_raw_position_for_packing(self) -> None:
        events = [
            {"src_lane": 5, "T": 500, "T_edit": 100, "L": 50, "kind": "speech"},
            {"src_lane": 11, "T": 100, "T_edit": 500, "L": 50, "kind": "speech"},
        ]
        reserved = {i: [(100, 150)] for i in list(range(5)) + list(range(6, 11))}

        _assign_target_lanes(events, 18, reserved_by_lane=reserved)

        self.assertEqual(events[1]["target_lane"], 5)
        self.assertEqual([e["target_T"] for e in events], [500, 100])

    def test_lane_assignment_keeps_raw_position_out_of_reserved_raw_space(self) -> None:
        events = [
            {"src_lane": 2, "T": 100, "T_edit": 50, "L": 20, "kind": "speech"},
        ]

        _assign_target_lanes(events, 6, reserved_by_lane={0: [(50, 70)]})

        self.assertNotEqual(events[0]["target_lane"], 3)
        self.assertEqual(events[0]["target_T"], 100)

    def test_lane_assignment_compacts_music_to_lowest_free_lane(self) -> None:
        events = [
            {"src_lane": 8, "T": 100, "T_edit": 100, "L": 20, "kind": "music"},
        ]

        _assign_target_lanes(events, 11)

        self.assertEqual(events[0]["target_lane"], 10)

    def test_lane_assignment_preserves_stereo_music_vertical_order(self) -> None:
        events = [
            {"src_lane": 6, "T": 100, "T_edit": 100, "L": 20, "kind": "music"},
            {"src_lane": 7, "T": 100, "T_edit": 100, "L": 20, "kind": "music"},
        ]

        _assign_target_lanes(events, 11)

        self.assertEqual(events[0]["target_lane"], 9)
        self.assertEqual(events[1]["target_lane"], 10)

    def test_lane_assignment_preserves_aligned_noise_vertical_order(self) -> None:
        events = [
            {"src_lane": 3, "T": 100, "T_edit": 100, "L": 20, "kind": "noise"},
            {"src_lane": 4, "T": 100, "T_edit": 100, "L": 20, "kind": "noise"},
            {"src_lane": 5, "T": 100, "T_edit": 100, "L": 20, "kind": "noise"},
        ]

        _assign_target_lanes(events, 18)

        self.assertEqual([e["target_lane"] for e in events], [7, 8, 9])
        self.assertEqual([e["target_T"] for e in events], [100, 100, 100])

    def test_lane_assignment_keeps_polywav_channel_kinds_independent(self) -> None:
        events = [
            {
                "src_lane": 2,
                "T": 100,
                "T_edit": 100,
                "L": 200,
                "source_start": 96000,
                "source_length": 200,
                "kind": "speech",
            },
            {
                "src_lane": 3,
                "T": 100,
                "T_edit": 100,
                "L": 200,
                "source_start": 96000,
                "source_length": 200,
                "kind": "speech",
            },
            {
                "src_lane": 4,
                "T": 100,
                "T_edit": 100,
                "L": 200,
                "source_start": 96000,
                "source_length": 200,
                "kind": "noise",
            },
            {
                "src_lane": 5,
                "T": 100,
                "T_edit": 100,
                "L": 200,
                "source_start": 96000,
                "source_length": 200,
                "kind": "music",
            },
        ]

        _assign_target_lanes(events, 18)

        self.assertEqual([e["kind"] for e in events], ["speech", "speech", "noise", "music"])
        self.assertEqual([e["target_lane"] for e in events[:2]], [0, 1])
        self.assertGreaterEqual(events[2]["target_lane"], 7)
        self.assertGreaterEqual(events[3]["target_lane"], 12)
        self.assertEqual([e["target_T"] for e in events], [100, 100, 100, 100])

    def test_lane_assignment_does_not_promote_aligned_take_when_any_channel_has_speech(self) -> None:
        events = [
            {
                "src_lane": 2,
                "T": 100,
                "T_edit": 100,
                "L": 200,
                "source_start": 96000,
                "source_length": 200,
                "kind": "noise",
                "yamnet_s": 0.99,
                "yamnet_m": 0.0,
                "yamnet_n": 0.66,
            },
            {
                "src_lane": 4,
                "T": 100,
                "T_edit": 100,
                "L": 200,
                "source_start": 96000,
                "source_length": 200,
                "kind": "noise",
                "yamnet_s": 0.01,
                "yamnet_m": 0.0,
                "yamnet_n": 0.80,
            },
            {
                "src_lane": 5,
                "T": 100,
                "T_edit": 100,
                "L": 200,
                "source_start": 96000,
                "source_length": 200,
                "kind": "speech",
                "yamnet_s": 0.99,
                "yamnet_m": 0.0,
                "yamnet_n": 0.48,
            },
        ]

        _assign_target_lanes(events, 18)

        self.assertEqual([e["kind"] for e in events], ["noise", "noise", "speech"])
        self.assertGreaterEqual(events[0]["target_lane"], 7)
        self.assertGreaterEqual(events[1]["target_lane"], 7)
        self.assertEqual(events[2]["target_lane"], 0)

    def test_lane_assignment_does_not_promote_strong_noise_channel_to_speech(self) -> None:
        events = [
            {
                "src_lane": 8,
                "T": 10367,
                "T_edit": 10367,
                "L": 47,
                "source_start": 10,
                "source_length": 47,
                "kind": "noise",
                "yamnet_s": 0.324461,
                "yamnet_m": 0.000389,
                "yamnet_n": 0.753148,
            },
            {
                "src_lane": 9,
                "T": 10367,
                "T_edit": 10367,
                "L": 47,
                "source_start": 10,
                "source_length": 47,
                "kind": "speech",
                "yamnet_s": 0.98652,
                "yamnet_m": 0.000364,
                "yamnet_n": 0.005361,
            },
        ]

        _assign_target_lanes(events, 26)

        self.assertEqual([event["kind"] for event in events], ["noise", "speech"])
        self.assertGreaterEqual(events[0]["target_lane"], 10)
        self.assertLessEqual(events[0]["target_lane"], 15)
        self.assertEqual(events[1]["target_lane"], 0)


    def test_lane_assignment_allows_adjacent_speech_groups_to_share_transition_edge(self) -> None:
        events: list[dict[str, object]] = []
        for lane in range(6):
            events.append(
                {
                    "src_lane": lane,
                    "T": 35792640,
                    "T_edit": 35792640,
                    "L": 195840,
                    "source_start": 576000,
                    "source_length": 195840,
                    "kind": "speech",
                    "post_transition_len": 21120,
                }
            )
        for lane in range(6):
            events.append(
                {
                    "src_lane": lane,
                    "T": 36009600,
                    "T_edit": 35967360,
                    "L": 178560,
                    "source_start": 576000,
                    "source_length": 178560,
                    "kind": "speech",
                    "pre_transition_len": 21120,
                }
            )

        _assign_target_lanes(events, 17, blocked_target_lanes=set(range(11, 17)))

        self.assertEqual([e["target_lane"] for e in events[:6]], [0, 1, 2, 3, 4, 5])
        self.assertEqual([e["target_lane"] for e in events[6:]], [0, 1, 2, 3, 4, 5])

    def test_lane_assignment_uses_target_lane_transition_history_for_upper_gap(self) -> None:
        events: list[dict[str, object]] = []
        for lane in range(5):
            events.append(
                {
                    "src_lane": lane,
                    "T": 648960,
                    "T_edit": 606720,
                    "L": 178560,
                    "source_start": 576000,
                    "source_length": 178560,
                    "kind": "speech",
                }
            )
        for lane in range(6, 11):
            events.append(
                {
                    "src_lane": lane,
                    "T": 827520,
                    "T_edit": 785280,
                    "L": 186240,
                    "source_start": 576000,
                    "source_length": 186240,
                    "kind": "speech",
                }
            )

        _assign_target_lanes(
            events,
            17,
            transition_spans_by_lane={lane: [(627840, 21120)] for lane in range(5)},
        )

        self.assertEqual([e["target_lane"] for e in events[5:]], [0, 1, 2, 3, 4])
        self.assertEqual([e["target_T"] for e in events[5:]], [827520] * 5)

    def test_lane_assignment_does_not_push_adjacent_speech_down_for_owned_pre_transition(self) -> None:
        events: list[dict[str, object]] = []
        for lane in range(6):
            events.append(
                {
                    "src_lane": lane,
                    "T": 648960,
                    "T_edit": 606720,
                    "L": 178560,
                    "source_start": 576000,
                    "source_length": 178560,
                    "kind": "speech",
                    "pre_transition_len": 21120,
                }
            )
        for lane in range(5):
            events.append(
                {
                    "src_lane": lane,
                    "T": 827520,
                    "T_edit": 785280,
                    "L": 186240,
                    "source_start": 576000,
                    "source_length": 186240,
                    "kind": "speech",
                }
            )
        for lane in range(6):
            events.append(
                {
                    "src_lane": lane,
                    "T": 1013760,
                    "T_edit": 971520,
                    "L": 336000,
                    "source_start": 576000,
                    "source_length": 336000,
                    "kind": "speech",
                }
            )

        _assign_target_lanes(
            events,
            17,
            transition_spans_by_lane={lane: [(627840, 21120)] for lane in range(5)},
        )

        self.assertEqual([e["target_T"] for e in events[:6]], [648960] * 6)
        self.assertEqual([e["target_lane"] for e in events[6:11]], [0, 1, 2, 3, 4])
        self.assertEqual([e["target_T"] for e in events[6:11]], [827520] * 5)
        self.assertEqual([e["target_lane"] for e in events[11:]], [0, 1, 2, 3, 4, 5])

    def test_lane_plan_validation_allows_adjacent_events_to_share_transition_edge(self) -> None:
        events = [
            {
                "target_lane": 15,
                "target_T": 100,
                "T": 100,
                "L": 40,
                "post_transition_len": 10,
                "display_name": "first",
            },
            {
                "target_lane": 15,
                "target_T": 150,
                "T": 150,
                "L": 20,
                "pre_transition_len": 10,
                "display_name": "second",
            },
        ]

        _validate_lane_layout_plan(events, 17)

    def test_lane_plan_repair_moves_overlapping_noise_without_visible_shift(self) -> None:
        events = [
            {
                "src_lane": 6,
                "target_lane": 7,
                "T": 15423360,
                "T_edit": 14693760,
                "target_T": 15953280,
                "L": 238080,
                "kind": "noise",
                "pre_transition_len": 86400,
                "source_start": 576000,
                "source_length": 238080,
                "name": "first",
            },
            {
                "src_lane": 2,
                "target_lane": 7,
                "T": 16279680,
                "T_edit": 14931840,
                "target_T": 16018560,
                "L": 120960,
                "kind": "noise",
                "source_start": 576000,
                "source_length": 120960,
                "name": "second",
            },
        ]

        moved = lane_layout_mod._repair_lane_layout_plan_overlaps(events, 17)

        self.assertEqual(moved, 0)
        self.assertEqual([e["T_edit"] for e in events], [14693760, 14931840])
        self.assertEqual(events[0]["target_lane"], events[1]["target_lane"])
        self.assertFalse(any(e.get("raw_t_authoritative") for e in events))
        _validate_lane_layout_plan(events, 17)

    def test_lane_layout_plan_module_repairs_and_validates_legacy_event_dicts(self) -> None:
        events = [
            {
                "src_lane": 0,
                "target_lane": 0,
                "T": 100,
                "T_edit": 100,
                "target_T": 100,
                "L": 50,
                "kind": "speech",
            },
            {
                "src_lane": 1,
                "target_lane": 0,
                "T": 120,
                "T_edit": 120,
                "target_T": 120,
                "L": 50,
                "kind": "speech",
            },
        ]

        moved = repair_lane_layout_plan_overlaps(events, 3)

        self.assertEqual(moved, 1)
        validate_lane_layout_plan(events, 3)
        self.assertEqual([e["T_edit"] for e in events], [100, 120])

    def test_lane_assignment_uses_visible_gap_when_raw_transition_history_overlaps(self) -> None:
        events = [
            {
                "src_lane": 0,
                "T": 0,
                "T_edit": 0,
                "L": 200,
                "kind": "speech",
            },
            {
                "src_lane": 1,
                "T": 100,
                "T_edit": 0,
                "L": 100,
                "kind": "speech",
            },
            {
                "src_lane": 2,
                "T": 150,
                "T_edit": 100,
                "L": 50,
                "kind": "speech",
            },
        ]

        _assign_target_lanes(
            events,
            6,
            transition_spans_by_lane={1: [(90, 25)]},
        )

        self.assertEqual(events[2]["target_lane"], 1)
        self.assertEqual(events[2]["T_edit"], 100)

    def test_lane_assignment_packs_visible_aligned_group_despite_raw_history_gap(self) -> None:
        events = [
            {
                "src_lane": 1,
                "top_idx": 35,
                "T": 12906240,
                "T_edit": 12906240,
                "L": 349440,
                "kind": "speech",
                "source_start": 1,
                "source_length": 349440,
            },
            {
                "src_lane": 0,
                "top_idx": 38,
                "T": 13593600,
                "T_edit": 13255680,
                "L": 142080,
                "kind": "speech",
                "source_start": 576000,
                "source_length": 142080,
            },
            {
                "src_lane": 1,
                "top_idx": 36,
                "T": 13255680,
                "T_edit": 13255680,
                "L": 142080,
                "kind": "speech",
                "source_start": 576000,
                "source_length": 142080,
            },
            {
                "src_lane": 2,
                "top_idx": 39,
                "T": 14430720,
                "T_edit": 13255680,
                "L": 142080,
                "kind": "speech",
                "source_start": 576000,
                "source_length": 142080,
            },
        ]

        _assign_target_lanes(
            events,
            17,
            transition_spans_by_lane={
                1: [(12900000, 100000), (14400000, 100000)],
                2: [(14400000, 100000)],
            },
        )

        self.assertEqual([e["target_lane"] for e in events[1:]], [0, 1, 2])

    def test_lane_plan_validation_allows_same_kind_visible_adjacent_raw_overlap(self) -> None:
        events = [
            {
                "target_lane": 0,
                "target_T": 100,
                "T": 100,
                "T_edit": 0,
                "L": 100,
                "kind": "speech",
                "display_name": "first",
            },
            {
                "target_lane": 0,
                "target_T": 150,
                "T": 150,
                "T_edit": 100,
                "L": 50,
                "kind": "speech",
                "display_name": "second",
            },
        ]

        _validate_lane_layout_plan(events, 2)

    def test_timeline_position_model_preserves_visible_and_raw_spans(self) -> None:
        event = {
            "T": 9409920,
            "T_edit": 7720320,
            "target_T": 8138880,
            "L": 6963840,
            "pre_transition_len": 209280,
            "post_transition_len": 629760,
        }

        pos = TimelinePosition.from_event(event)

        self.assertEqual(pos.raw_start, 8138880)
        self.assertEqual(pos.visible_start, 7720320)
        self.assertEqual(pos.length, 6963840)
        self.assertEqual(pos.visible_span, (7720320, 14684160))
        self.assertEqual(pos.raw_span_with_transitions, (7929600, 15732480))
        self.assertEqual(
            event_owned_raw_transition_spans(event, pos.raw_start),
            [(7929600, 8138880), (15102720, 15732480)],
        )
        self.assertEqual(
            event_owned_transition_history_spans(event, event["T"]),
            [(9200640, 209280), (16373760, 629760)],
        )

    def test_lane_layout_timing_raw_from_visible_excludes_owned_transition_history(self) -> None:
        transition_spans = [(90, 25), (90, 25), (150, 10), (200, 10)]

        filtered = spans_without_exact_matches(transition_spans, [(90, 25)])

        self.assertEqual(filtered, [(150, 10), (200, 10)])
        self.assertEqual(raw_from_visible_with_transition_spans(filtered, 160, 5), 190)

    def test_lane_layout_placement_helpers_preserve_transition_span_semantics(self) -> None:
        event = {
            "T": 100,
            "T_edit": 80,
            "L": 40,
            "pre_transition_len": 10,
            "post_transition_len": 20,
        }

        self.assertEqual(event_placement_length(event), 70)
        self.assertEqual(event_visible_occupancy_length(event), 40)
        self.assertEqual(event_placement_start(event, 130), 120)
        self.assertEqual(
            event_owned_visible_transition_spans(event),
            [(80, 100), (110, 150)],
        )

    def test_lane_layout_placement_target_raw_ignores_owned_transition_history(self) -> None:
        event = {
            "T": 100,
            "T_edit": 80,
            "L": 40,
            "pre_transition_len": 10,
            "post_transition_len": 20,
        }

        target_t = event_target_raw_start_for_lane(
            event,
            1,
            {
                1: [
                    (90, 10),  # Owned pre transition from original raw T.
                    (140, 20),  # Owned post transition from original raw T.
                    (120, 5),  # Foreign transition on the target lane.
                ],
            },
        )

        self.assertEqual(target_t, 100)

    def test_lane_layout_event_placement_span_uses_target_or_lane_raw_start(self) -> None:
        event = {
            "src_lane": 1,
            "target_lane": 2,
            "target_T": 130,
            "T": 100,
            "T_edit": 80,
            "L": 40,
            "pre_transition_len": 10,
            "post_transition_len": 20,
        }

        current = EventPlacementSpan.from_event(
            event,
            transition_spans_by_lane=None,
        )
        candidate = EventPlacementSpan.from_event(
            event,
            lane=3,
            transition_spans_by_lane=None,
        )

        self.assertEqual(current.lane, 2)
        self.assertEqual(current.event_t, 130)
        self.assertEqual(current.raw_span, (120, 190))
        self.assertEqual(current.visible_span, (80, 120))
        self.assertEqual(candidate.lane, 3)
        self.assertEqual(candidate.event_t, 100)

    def test_lane_layout_constraints_enforce_blocked_mutable_signature_and_raw_time(self) -> None:
        event = {"src_lane": 1, "T": 10, "T_edit": 10, "L": 5}
        signatures = {1: ("slot", "sound", 48000), 2: ("slot", "sound", 48000), 3: ("slot", "other", 48000)}

        self.assertFalse(
            target_lane_allowed_for_event(
                event,
                2,
                blocked_target_lanes={2},
            )
        )
        self.assertTrue(
            target_lane_allowed_for_event(
                event,
                1,
                blocked_target_lanes={1},
            )
        )
        self.assertFalse(
            target_lane_allowed_for_event(
                event,
                2,
                lane_mutable_by_lane={1: False, 2: True},
            )
        )
        self.assertFalse(
            target_lane_allowed_for_event(
                event,
                2,
                lane_mutable_by_lane={1: True, 2: False},
            )
        )
        self.assertFalse(
            target_lane_allowed_for_event(
                event,
                3,
                lane_signature_by_lane=signatures,
            )
        )
        self.assertFalse(
            target_lane_allowed_for_event(
                {**event, "T_edit": -1},
                2,
                transition_spans_by_lane={2: []},
            )
        )

    def test_lane_layout_visible_conflict_helper_restores_candidate_assignment(self) -> None:
        candidate = {
            "src_lane": 0,
            "target_lane": 5,
            "target_T": 500,
            "T": 10,
            "T_edit": 100,
            "L": 20,
            "kind": "speech",
        }
        allowed_other = {
            "src_lane": 1,
            "target_lane": 2,
            "T": 20,
            "T_edit": 110,
            "L": 5,
            "kind": "speech",
        }
        blocking_other = {
            "src_lane": 2,
            "target_lane": 2,
            "T": 30,
            "T_edit": 115,
            "L": 5,
            "kind": "noise",
        }

        blocked = target_lane_visible_conflicts(
            candidate,
            2,
            [candidate, allowed_other, blocking_other],
            target_raw_for_lane=lambda _event, lane: 200 + lane,
            allowed_overlap=lambda other, _candidate: other is allowed_other,
        )

        self.assertTrue(blocked)
        self.assertEqual(candidate["target_lane"], 5)
        self.assertEqual(candidate["target_T"], 500)

    def test_lane_layout_compaction_moves_event_to_first_non_conflicting_preferred_lane(self) -> None:
        blocker = {
            "src_lane": 0,
            "target_lane": 0,
            "target_T": 100,
            "T": 100,
            "T_edit": 100,
            "L": 20,
            "kind": "speech",
        }
        candidate = {
            "src_lane": 2,
            "target_lane": 2,
            "target_T": 100,
            "T": 100,
            "T_edit": 100,
            "L": 20,
            "kind": "speech",
        }

        state = compact_events_to_preferred_lanes(
            [blocker, candidate],
            [candidate],
            lane_count=3,
            preferred_lanes=[0, 1],
            fallback_lanes=[0, 1, 2],
            target_allowed=lambda _event, _lane: True,
            reserved_by_lane=None,
            transition_spans_by_lane=None,
            lane_capacity_by_lane=None,
        )

        self.assertEqual(candidate["target_lane"], 1)
        self.assertEqual(candidate["target_T"], 100)
        self.assertIn((100, 120), state.raw[1])

    def test_lane_layout_initial_placement_releases_source_before_preferred_lane(self) -> None:
        event = {"src_lane": 1, "T": 100, "T_edit": 100, "L": 20, "kind": "speech"}
        state = LaneOccupancyState(raw=[[], []], visible=[[], []])
        source_spans = SourceSpanTracker.from_events([event], 2, state)
        planner = LaneInitialPlacementPlanner(
            lane_count=2,
            raw_state=state,
            source_spans=source_spans,
            lane_pref_all=[0, 1],
            target_allowed=lambda _event, _lane: True,
            transition_spans_by_lane=None,
            aligned_kind_group_key=lambda _event: None,
            aligned_layout_group_key=lambda _event: None,
        )

        planner.place_events_preserving_aligned_order(
            [event],
            [0],
            kind="speech",
            sort_key=lambda item: int(item["T"]),
        )

        self.assertEqual(event["target_lane"], 0)
        self.assertEqual(event["target_T"], 100)
        self.assertEqual(state.raw[0], [(100, 120)])
        self.assertEqual(state.raw[1], [])

    def test_lane_layout_speech_swap_promotes_longer_overlapping_lower_event(self) -> None:
        upper = {"src_lane": 0, "target_lane": 0, "target_T": 100, "T": 100, "T_edit": 100, "L": 10}
        lower = {"src_lane": 1, "target_lane": 1, "target_T": 100, "T": 100, "T_edit": 100, "L": 20}
        state = LaneOccupancyState(
            raw=[[(100, 110)], [(100, 120)]],
            visible=[[(100, 110)], [(100, 120)]],
        )

        promote_longer_speech_events(
            [upper, lower],
            lane_count=2,
            occupancy=state,
            target_allowed=lambda _event, _lane: True,
            transition_spans_by_lane=None,
        )

        self.assertEqual(lower["target_lane"], 0)
        self.assertEqual(upper["target_lane"], 1)
        self.assertEqual(state.raw[0], [(100, 120)])
        self.assertEqual(state.raw[1], [(100, 110)])

    def test_lane_layout_aligned_compaction_moves_group_to_upper_visible_gap(self) -> None:
        group = [
            {
                "src_lane": 4,
                "target_lane": 2,
                "T": 100,
                "T_edit": 100,
                "L": 20,
                "kind": "speech",
                "source_start": 1000,
                "source_length": 20,
            },
            {
                "src_lane": 5,
                "target_lane": 3,
                "T": 100,
                "T_edit": 100,
                "L": 20,
                "kind": "speech",
                "source_start": 1000,
                "source_length": 20,
            },
        ]

        compact_aligned_groups_to_visible_gaps(
            group,
            lane_count=6,
            target_allowed=lambda _event, _lane: True,
            target_raw_for_lane=lambda event, _lane: int(event["T"]),
            aligned_kind_group_key=aligned_kind_group_key,
            allowed_overlap=lambda _other, _candidate: False,
        )

        self.assertEqual([event["target_lane"] for event in group], [0, 1])
        self.assertEqual([event["target_T"] for event in group], [100, 100])

    def test_lane_layout_aligned_order_restores_source_order_on_unique_target_lanes(self) -> None:
        group = [
            {
                "src_lane": 1,
                "target_lane": 3,
                "T": 100,
                "T_edit": 100,
                "L": 20,
                "source_start": 1000,
                "source_length": 20,
            },
            {
                "src_lane": 2,
                "target_lane": 2,
                "T": 100,
                "T_edit": 100,
                "L": 20,
                "source_start": 1000,
                "source_length": 20,
            },
        ]

        preserve_aligned_target_order_after_compaction(
            group,
            target_allowed=lambda _event, _lane: True,
            target_raw_for_lane=lambda event, _lane: int(event["T"]),
            aligned_kind_group_key=aligned_kind_group_key,
        )

        self.assertEqual([event["target_lane"] for event in group], [2, 3])

    def test_lane_layout_occupancy_conflicts_respects_capacity_and_owned_spans(self) -> None:
        raw_occ = [[(10, 30)]]
        visible_occ = [[(100, 130)]]

        self.assertTrue(
            lane_occupancy_conflicts(
                raw_occ,
                0,
                25,
                35,
                visible_occupancy=visible_occ,
                visible_t0=200,
                visible_t1=210,
            )
        )
        self.assertFalse(
            lane_occupancy_conflicts(
                raw_occ,
                0,
                25,
                35,
                visible_occupancy=visible_occ,
                visible_t0=200,
                visible_t1=210,
                ignored_raw_spans=[(25, 30)],
            )
        )
        self.assertTrue(
            lane_occupancy_conflicts(
                raw_occ,
                0,
                40,
                50,
                lane_capacity_by_lane={0: 45},
                visible_occupancy=visible_occ,
                visible_t0=200,
                visible_t1=210,
            )
        )
        self.assertTrue(
            lane_occupancy_conflicts(
                raw_occ,
                0,
                40,
                50,
                visible_occupancy=visible_occ,
                visible_t0=120,
                visible_t1=140,
            )
        )
        self.assertFalse(
            lane_occupancy_conflicts(
                raw_occ,
                0,
                40,
                50,
                visible_occupancy=visible_occ,
                visible_t0=120,
                visible_t1=140,
                ignored_visible_spans=[(120, 130)],
            )
        )

    def test_lane_layout_occupancy_state_add_remove_and_conflict_checks(self) -> None:
        state = LaneOccupancyState(
            raw=[[(10, 30)], []],
            visible=[[(100, 130)], []],
            lane_capacity_by_lane={0: 50},
        )

        self.assertTrue(state.conflicts(0, 25, 35, visible_t0=200, visible_t1=210))
        self.assertFalse(
            state.conflicts(
                0,
                25,
                35,
                visible_t0=200,
                visible_t1=210,
                ignored_raw_spans=[(25, 30)],
            )
        )
        self.assertTrue(state.conflicts(0, 45, 55, visible_t0=200, visible_t1=210))

        state.add(1, 40, 60, visible_t0=140, visible_t1=160)
        self.assertEqual(state.raw[1], [(40, 60)])
        self.assertEqual(state.visible[1], [(140, 160)])
        self.assertTrue(state.remove(1, 40, 60, visible_t0=140, visible_t1=160))
        self.assertFalse(state.remove(1, 40, 60, visible_t0=140, visible_t1=160))
        self.assertEqual(state.raw[1], [])
        self.assertEqual(state.visible[1], [])

    def test_lane_layout_source_span_tracker_releases_and_restores_raw_source_spans(self) -> None:
        state = LaneOccupancyState(raw=[[], []], visible=[[], []])
        event = {"src_lane": 1, "T": 100, "L": 20, "pre_transition_len": 5}
        tracker = SourceSpanTracker.from_events([event], 2, state)

        self.assertEqual(state.raw[1], [(95, 120)])
        released = tracker.release(event)

        self.assertEqual(released, (1, 95, 120))
        self.assertEqual(state.raw[1], [])
        self.assertIsNone(tracker.release({"src_lane": 0, "T": 1, "L": 1}))
        tracker.restore(released)
        self.assertEqual(state.raw[1], [(95, 120)])

    def test_lane_layout_snapshot_uses_current_target_and_visible_spans(self) -> None:
        events = [
            {
                "src_lane": 1,
                "target_lane": 99,
                "T": 100,
                "T_edit": 80,
                "L": 20,
                "pre_transition_len": 5,
                "post_transition_len": 10,
            },
            {
                "src_lane": 0,
                "target_lane": 2,
                "target_T": 200,
                "T": 100,
                "T_edit": 50,
                "L": 10,
            },
        ]

        snapshot = build_lane_occupancy_snapshot(
            events,
            3,
            reserved_by_lane={0: [(1, 2)]},
            transition_spans_by_lane=None,
        )

        self.assertEqual(snapshot.raw[0], [(1, 2)])
        self.assertEqual(snapshot.raw[1], [(95, 130)])
        self.assertEqual(snapshot.raw[2], [(200, 210)])
        self.assertEqual(snapshot.visible[0], [])
        self.assertEqual(snapshot.visible[1], [(80, 100)])
        self.assertEqual(snapshot.visible[2], [(50, 60)])

    def test_lane_layout_xfade_chains_group_only_same_kind_shared_edges(self) -> None:
        first = {
            "src_lane": 3,
            "top_idx": 1,
            "kind": "music",
            "post_transition_len": 10,
            "post_transition_top_idx": 2,
        }
        second = {
            "src_lane": 3,
            "top_idx": 3,
            "kind": "music",
            "pre_transition_len": 10,
            "pre_transition_top_idx": 2,
            "post_transition_len": 5,
            "post_transition_top_idx": 4,
        }
        different_kind = {
            "src_lane": 3,
            "top_idx": 5,
            "kind": "speech",
            "pre_transition_len": 5,
            "pre_transition_top_idx": 4,
        }
        other_lane = {
            "src_lane": 4,
            "top_idx": 1,
            "kind": "music",
            "pre_transition_len": 10,
            "pre_transition_top_idx": 2,
        }

        chains = same_kind_transition_chains([different_kind, second, other_lane, first])

        self.assertEqual(chains, [[first, second]])

    def test_lane_layout_xfade_candidate_lanes_keep_current_then_class_zone_order(self) -> None:
        chain = [
            {"kind": "music", "target_lane": 4, "src_lane": 10},
            {"kind": "music", "target_lane": 6, "src_lane": 11},
            {"kind": "music", "target_lane": 4, "src_lane": 12},
        ]

        candidates = same_kind_chain_candidate_lanes(chain, 8)

        self.assertEqual(candidates[:2], [4, 6])
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertTrue(all(0 <= lane < 8 for lane in candidates))

    def test_lane_layout_xfade_preserve_moves_chain_to_single_candidate_lane(self) -> None:
        first = {
            "src_lane": 2,
            "top_idx": 1,
            "target_lane": 4,
            "T": 100,
            "T_edit": 100,
            "L": 20,
            "kind": "music",
            "post_transition_len": 10,
            "post_transition_top_idx": 2,
        }
        second = {
            "src_lane": 2,
            "top_idx": 3,
            "target_lane": 5,
            "T": 120,
            "T_edit": 120,
            "L": 20,
            "kind": "music",
            "pre_transition_len": 10,
            "pre_transition_top_idx": 2,
        }

        preserve_same_kind_xfade_chains(
            [first, second],
            lane_count=8,
            target_allowed=lambda _event, _lane: True,
            target_raw_for_lane=lambda event, _lane: int(event["T"]),
            allowed_overlap=lambda _other, _candidate: False,
        )

        self.assertEqual(first["target_lane"], 4)
        self.assertEqual(second["target_lane"], 4)

    def test_lane_layout_final_compaction_moves_xfade_chain_as_atom_to_upper_gap(self) -> None:
        blockers = [
            {"src_lane": 0, "target_lane": lane, "T": 100, "T_edit": 100, "L": 50, "kind": "speech"}
            for lane in (0, 1, 2)
        ]
        first = {
            "src_lane": 6,
            "top_idx": 1,
            "target_lane": 4,
            "T": 100,
            "T_edit": 100,
            "L": 20,
            "kind": "speech",
            "post_transition_len": 10,
            "post_transition_top_idx": 2,
        }
        second = {
            "src_lane": 6,
            "top_idx": 3,
            "target_lane": 4,
            "T": 120,
            "T_edit": 110,
            "L": 40,
            "kind": "speech",
            "pre_transition_len": 10,
            "pre_transition_top_idx": 2,
        }
        events = [*blockers, first, second]

        moved = final_bounded_compact_class_events(
            events,
            lane_count=11,
            target_allowed=lambda _event, _lane: True,
            target_raw_for_lane=lambda event, _lane: int(event["T"]),
        )

        self.assertEqual(moved, 2)
        self.assertEqual(first["target_lane"], 3)
        self.assertEqual(second["target_lane"], 3)
        self.assertEqual(first["target_T"], 100)
        self.assertEqual(second["target_T"], 120)

    def test_lane_layout_final_compaction_removes_speech_gap_after_late_promote(self) -> None:
        blockers = [
            {"src_lane": 0, "target_lane": lane, "T": 100, "T_edit": 100, "L": 50, "kind": "speech"}
            for lane in (0, 1, 2)
        ]
        lower = {
            "src_lane": 2,
            "target_lane": 5,
            "T": 105,
            "T_edit": 105,
            "L": 30,
            "kind": "speech",
            "source_start": 10,
            "source_length": 30,
        }
        gapped = {
            "src_lane": 1,
            "target_lane": 7,
            "T": 106,
            "T_edit": 106,
            "L": 30,
            "kind": "speech",
            "source_start": 50,
            "source_length": 30,
        }
        events = [*blockers, lower, gapped]

        moved = final_bounded_compact_class_events(
            events,
            lane_count=11,
            target_allowed=lambda _event, _lane: True,
            target_raw_for_lane=lambda event, _lane: int(event["T"]),
        )

        self.assertEqual(moved, 2)
        self.assertEqual([lower["target_lane"], gapped["target_lane"]], [3, 4])
        self.assertEqual([lower["target_T"], gapped["target_T"]], [105, 106])

    def test_lane_layout_final_compaction_fills_visible_gap_between_upper_speech_blocks(self) -> None:
        upper_before_after = []
        for lane in range(4):
            upper_before_after.extend(
                [
                    {
                        "src_lane": lane,
                        "top_idx": lane * 10,
                        "target_lane": lane,
                        "T": 68638,
                        "T_edit": 68638,
                        "L": 62,
                        "kind": "speech",
                    },
                    {
                        "src_lane": lane,
                        "top_idx": lane * 10 + 2,
                        "target_lane": lane,
                        "T": 68753,
                        "T_edit": 68753,
                        "L": 139,
                        "kind": "speech",
                    },
                ]
            )
        middle_group = [
            {
                "src_lane": src_lane,
                "top_idx": src_lane,
                "target_lane": target_lane,
                "T": 68700,
                "T_edit": 68700,
                "L": 47,
                "kind": "speech",
                "source_start": 1000,
                "source_length": 47,
            }
            for src_lane, target_lane in ((4, 4), (5, 5), (6, 6), (7, 7))
        ]
        events = [*upper_before_after, *middle_group]

        moved = final_bounded_compact_class_events(
            events,
            lane_count=21,
            target_allowed=lambda _event, _lane: True,
            target_raw_for_lane=lambda event, _lane: int(event["T_edit"]),
            aligned_group_key=lambda event: (
                int(event["T_edit"]),
                int(event["L"]),
                int(event.get("source_start", -1)),
                int(event.get("source_length", -1)),
            )
            if "source_start" in event
            else None,
        )

        self.assertEqual(moved, 4)
        self.assertEqual([event["target_lane"] for event in middle_group], [0, 1, 2, 3])
        self.assertEqual([event["target_T"] for event in middle_group], [68700] * 4)

    def test_lane_layout_final_compaction_does_not_invert_aligned_group_order(self) -> None:
        blockers = [
            {"src_lane": 9, "target_lane": lane, "T": 100, "T_edit": 100, "L": 50, "kind": "speech"}
            for lane in (0, 1)
        ]
        earlier_channel = {
            "src_lane": 2,
            "target_lane": 3,
            "T": 100,
            "T_edit": 100,
            "L": 30,
            "kind": "music",
            "source_start": 10,
            "source_length": 30,
        }
        later_channel = {
            "src_lane": 3,
            "target_lane": 5,
            "T": 100,
            "T_edit": 100,
            "L": 30,
            "kind": "speech",
            "source_start": 10,
            "source_length": 30,
        }
        events = [*blockers, earlier_channel, later_channel]

        moved = final_bounded_compact_class_events(
            events,
            lane_count=11,
            target_allowed=lambda _event, _lane: True,
            target_raw_for_lane=lambda event, _lane: int(event["T"]),
            aligned_group_key=lambda event: (
                int(event["T_edit"]),
                int(event["L"]),
                int(event.get("source_start", -1)),
                int(event.get("source_length", -1)),
            )
            if "source_start" in event
            else None,
        )

        self.assertEqual(moved, 1)
        self.assertEqual(later_channel["target_lane"], 4)

    def test_lane_layout_final_compaction_moves_aligned_noise_group_as_atom(self) -> None:
        group = [
            {
                "src_lane": src_lane,
                "top_idx": src_lane,
                "target_lane": target_lane,
                "T": 100 + src_lane,
                "T_edit": 100,
                "L": 20,
                "kind": "noise",
                "source_start": 10,
                "source_length": 20,
            }
            for src_lane, target_lane in ((0, 1), (1, 2), (2, 7))
        ]

        moved = final_bounded_compact_class_events(
            group,
            lane_count=17,
            target_allowed=lambda _event, _lane: True,
            target_raw_for_lane=lambda event, _lane: int(event["T"]),
            aligned_group_key=lambda event: (
                int(event["T_edit"]),
                int(event["L"]),
                int(event.get("source_start", -1)),
                int(event.get("source_length", -1)),
            ),
        )

        self.assertEqual(moved, 3)
        self.assertEqual([event["target_lane"] for event in group], [6, 7, 8])
        self.assertEqual([event["target_T"] for event in group], [100, 101, 102])

    def test_lane_layout_final_compaction_places_earlier_music_pair_on_bottom_lanes(self) -> None:
        first_left = {
            "src_lane": 24,
            "top_idx": 3,
            "target_lane": 22,
            "T": 3134,
            "T_edit": 3048,
            "target_T": 3134,
            "L": 1188,
            "kind": "music",
            "source_start": 7,
            "source_length": 1188,
            "pre_transition_len": 43,
            "post_transition_len": 4,
            "pre_transition_top_idx": 2,
            "post_transition_top_idx": 4,
        }
        first_right = {
            "src_lane": 25,
            "top_idx": 3,
            "target_lane": 25,
            "T": 3134,
            "T_edit": 3048,
            "target_T": 3134,
            "L": 1188,
            "kind": "music",
            "source_start": 7,
            "source_length": 1188,
            "pre_transition_len": 43,
            "post_transition_len": 4,
            "pre_transition_top_idx": 2,
            "post_transition_top_idx": 4,
        }
        later_left = {
            "src_lane": 24,
            "top_idx": 5,
            "target_lane": 22,
            "T": 4326,
            "T_edit": 4232,
            "target_T": 4240,
            "L": 933,
            "kind": "music",
            "source_start": 2755,
            "source_length": 933,
            "pre_transition_len": 4,
            "post_transition_len": 43,
            "pre_transition_top_idx": 4,
            "post_transition_top_idx": 6,
        }
        later_right = {
            "src_lane": 25,
            "top_idx": 5,
            "target_lane": 25,
            "T": 4326,
            "T_edit": 4232,
            "target_T": 4326,
            "L": 933,
            "kind": "music",
            "source_start": 2755,
            "source_length": 933,
            "pre_transition_len": 4,
            "post_transition_len": 43,
            "pre_transition_top_idx": 4,
            "post_transition_top_idx": 6,
        }
        events = [first_left, later_left, first_right, later_right]

        final_bounded_compact_class_events(
            events,
            lane_count=26,
            target_allowed=lambda _event, _lane: True,
            target_raw_for_lane=lambda event, _lane: int(event.get("target_T", event["T"])),
            aligned_group_key=lambda event: (
                int(event["T_edit"]),
                int(event["L"]),
                int(event["source_start"]),
                int(event["source_length"]),
            ),
        )

        self.assertEqual([first_left["target_lane"], first_right["target_lane"]], [24, 25])
        self.assertEqual([later_left["target_lane"], later_right["target_lane"]], [24, 25])

    def test_transition_relation_requires_same_kind_lane_and_shared_transition_index(self) -> None:
        prev = {
            "src_lane": 11,
            "kind": "music",
            "post_transition_len": 209280,
            "post_transition_top_idx": 5,
        }
        item = {
            "src_lane": 11,
            "kind": "music",
            "pre_transition_len": 209280,
            "pre_transition_top_idx": 5,
        }

        self.assertTrue(events_share_same_kind_transition_edge(prev, item))
        self.assertFalse(events_share_same_kind_transition_edge({**prev, "kind": "speech"}, item))
        self.assertFalse(events_share_same_kind_transition_edge(prev, {**item, "src_lane": 12}))
        self.assertFalse(events_share_same_kind_transition_edge(prev, {**item, "pre_transition_top_idx": 6}))
        self.assertFalse(events_share_same_kind_transition_edge({**prev, "post_transition_len": 0}, item))

    def test_planned_overlap_model_allows_only_transition_safe_overlaps(self) -> None:
        visible_disjoint_prev = {
            "target_T": 100,
            "T": 100,
            "T_edit": 0,
            "L": 100,
            "kind": "speech",
        }
        visible_disjoint_item = {
            "target_T": 150,
            "T": 150,
            "T_edit": 100,
            "L": 50,
            "kind": "speech",
        }

        self.assertTrue(
            planned_overlap_is_allowed_transition_overlap(
                visible_disjoint_prev,
                visible_disjoint_item,
            )
        )

        visible_overlap_item = {
            "target_T": 150,
            "T": 150,
            "T_edit": 50,
            "L": 50,
            "kind": "speech",
        }

        self.assertFalse(
            planned_overlap_is_allowed_transition_overlap(
                visible_disjoint_prev,
                visible_overlap_item,
            )
        )

        shared_edge_prev = {
            "src_lane": 11,
            "target_T": 100,
            "T": 100,
            "L": 40,
            "kind": "music",
            "post_transition_len": 10,
            "post_transition_top_idx": 5,
        }
        shared_edge_item = {
            "src_lane": 11,
            "target_T": 145,
            "T": 145,
            "L": 20,
            "kind": "music",
            "pre_transition_len": 10,
            "pre_transition_top_idx": 5,
        }

        self.assertTrue(
            planned_overlap_is_allowed_transition_overlap(
                shared_edge_prev,
                shared_edge_item,
            )
        )

    def test_lane_layout_group_helpers_use_source_window_and_visible_start(self) -> None:
        event = {
            "T": 200,
            "T_edit": 180,
            "L": 40,
            "source_start": 1000,
            "source_length": 40,
        }

        self.assertEqual(event_source_window_key(event), (1000, 40))
        self.assertEqual(aligned_layout_group_key(event), (200, 40, 1000, 40))
        self.assertEqual(aligned_kind_group_key(event), (180, 40, 1000, 40))
        self.assertIsNone(aligned_kind_group_key({"T": 1, "L": 2}))
        self.assertEqual(
            dominant_aligned_group_kind(
                [
                    {"kind": "noise"},
                    {"kind": "speech"},
                    {"kind": "noise"},
                ]
            ),
            "speech",
        )

    def test_lane_layout_group_helpers_collect_only_cross_lane_groups(self) -> None:
        events = [
            {"src_lane": 0, "key": "a"},
            {"src_lane": 1, "key": "a"},
            {"src_lane": 2, "key": "b"},
            {"src_lane": 2, "key": "b"},
            {"src_lane": 3, "key": None},
        ]

        grouped = group_events_by_optional_key(
            events,
            lambda event: None if event["key"] is None else (event["key"],),
        )
        cross_lane = cross_lane_event_groups(
            events,
            lambda event: None if event["key"] is None else (event["key"],),
        )

        self.assertEqual(list(grouped), [("a",), ("b",)])
        self.assertEqual(grouped[("a",)], events[:2])
        self.assertEqual(cross_lane, [events[:2]])

    def test_lane_layout_group_helpers_report_source_and_target_order(self) -> None:
        low_source = {"src_lane": 1, "top_idx": 3, "target_lane": 4}
        high_source = {"src_lane": 2, "top_idx": 1, "target_lane": 3}
        duplicate_target = {"src_lane": 3, "top_idx": 2, "target_lane": 3}
        group = [high_source, duplicate_target, low_source]

        self.assertEqual(aligned_group_source_order(group), [low_source, high_source, duplicate_target])
        self.assertEqual(aligned_group_target_order(group), [high_source, duplicate_target, low_source])
        self.assertEqual(aligned_group_target_lanes(group), [3, 3, 4])

    def test_lane_assignment_preserves_aligned_order_when_visible_times_differ(self) -> None:
        events = [
            {
                "src_lane": 0,
                "top_idx": 11,
                "T": 1836,
                "T_edit": 1836,
                "L": 97,
                "source_start": 278,
                "source_length": 97,
                "kind": "noise",
            },
            {
                "src_lane": 1,
                "top_idx": 11,
                "T": 1836,
                "T_edit": 1808,
                "L": 97,
                "source_start": 278,
                "source_length": 97,
                "kind": "noise",
            },
        ]

        _assign_target_lanes(events, 18)

        self.assertLess(events[0]["target_lane"], events[1]["target_lane"])
        self.assertEqual([e["target_T"] for e in events], [1836, 1836])

    def test_lane_assignment_preserves_three_channel_music_vertical_order(self) -> None:
        events = [
            {"src_lane": 3, "T": 100, "T_edit": 100, "L": 20, "kind": "music"},
            {"src_lane": 4, "T": 100, "T_edit": 100, "L": 20, "kind": "music"},
            {"src_lane": 5, "T": 100, "T_edit": 100, "L": 20, "kind": "music"},
        ]

        _assign_target_lanes(events, 18)

        self.assertEqual([e["target_lane"] for e in events], [15, 16, 17])
        self.assertEqual([e["target_T"] for e in events], [100, 100, 100])

    def test_lane_assignment_keeps_same_kind_xfade_chain_on_one_lane(self) -> None:
        events = [
            {
                "src_lane": 11,
                "top_idx": 4,
                "T": 7703040,
                "T_edit": 6432000,
                "L": 1497600,
                "kind": "music",
                "source_start": 1,
                "source_length": 1497600,
                "pre_transition_len": 635520,
                "pre_transition_top_idx": 3,
                "post_transition_len": 209280,
                "post_transition_top_idx": 5,
            },
            {
                "src_lane": 11,
                "top_idx": 6,
                "T": 9409920,
                "T_edit": 7720320,
                "L": 6963840,
                "kind": "music",
                "source_start": 2,
                "source_length": 6963840,
                "pre_transition_len": 209280,
                "pre_transition_top_idx": 5,
                "post_transition_len": 629760,
                "post_transition_top_idx": 7,
            },
            {
                "src_lane": 12,
                "top_idx": 4,
                "T": 7703040,
                "T_edit": 6432000,
                "L": 1497600,
                "kind": "music",
                "source_start": 1,
                "source_length": 1497600,
                "pre_transition_len": 635520,
                "pre_transition_top_idx": 3,
                "post_transition_len": 209280,
                "post_transition_top_idx": 5,
            },
            {
                "src_lane": 12,
                "top_idx": 6,
                "T": 9409920,
                "T_edit": 7720320,
                "L": 6963840,
                "kind": "music",
                "source_start": 2,
                "source_length": 6963840,
                "pre_transition_len": 209280,
                "pre_transition_top_idx": 5,
                "post_transition_len": 629760,
                "post_transition_top_idx": 7,
            },
        ]

        _assign_target_lanes(
            events,
            17,
            transition_spans_by_lane={
                11: [(7067520, 635520), (9200640, 209280), (16377600, 629760)],
                12: [(7067520, 635520), (9200640, 209280), (16377600, 629760)],
            },
        )

        self.assertEqual(events[0]["target_lane"], events[1]["target_lane"])
        self.assertEqual(events[2]["target_lane"], events[3]["target_lane"])
        self.assertNotEqual(events[0]["target_lane"], events[2]["target_lane"])

    def test_lane_assignment_avoids_transition_lanes_as_new_targets(self) -> None:
        events = [
            {"src_lane": 8, "T": 100, "T_edit": 100, "L": 20, "kind": "music"},
        ]

        _assign_target_lanes(events, 11, blocked_target_lanes={10})

        self.assertEqual(events[0]["target_lane"], 9)

    def test_lane_assignment_allows_event_to_stay_on_own_transition_lane(self) -> None:
        events = [
            {"src_lane": 10, "T": 100, "T_edit": 100, "L": 20, "kind": "music"},
        ]

        _assign_target_lanes(events, 11, blocked_target_lanes={10})

        self.assertEqual(events[0]["target_lane"], 10)

    def test_lane_assignment_preserves_visible_time_from_transition_source_lane(self) -> None:
        events = [
            {"src_lane": 0, "T": 120, "T_edit": 100, "L": 20, "kind": "music"},
        ]

        _assign_target_lanes(
            events,
            4,
            transition_spans_by_lane={0: [(50, 10)]},
        )

        self.assertEqual(events[0]["target_lane"], 3)
        self.assertEqual(events[0]["target_T"], 100)

    def test_lane_assignment_preserves_visible_time_onto_transition_target_lane(self) -> None:
        events = [
            {"src_lane": 0, "T": 100, "T_edit": 100, "L": 20, "kind": "music"},
        ]

        _assign_target_lanes(
            events,
            4,
            transition_spans_by_lane={3: [(50, 10)]},
        )

        self.assertEqual(events[0]["target_lane"], 3)
        self.assertEqual(events[0]["target_T"], 120)

    def test_lane_assignment_allows_transition_history_compatible_target_lane(self) -> None:
        events = [
            {"src_lane": 0, "T": 120, "T_edit": 100, "L": 20, "kind": "music"},
        ]

        _assign_target_lanes(
            events,
            4,
            transition_spans_by_lane={0: [(50, 10)], 3: [(50, 10)]},
        )

        self.assertEqual(events[0]["target_lane"], 3)
        self.assertEqual(events[0]["target_T"], 120)

    def test_lane_assignment_uses_target_lane_transition_history_for_speech_compaction(self) -> None:
        events = [
            {"src_lane": 2, "T": 90, "T_edit": 90, "L": 20, "kind": "speech"},
            {"src_lane": 3, "T": 111, "T_edit": 100, "L": 100, "kind": "speech"},
        ]

        _assign_target_lanes(events, 4, transition_spans_by_lane={0: [(50, 10)]})

        self.assertEqual([e["target_lane"] for e in events], [1, 0])
        self.assertEqual([e["target_T"] for e in events], [90, 120])

    def test_lane_assignment_uses_upper_raw_solution_for_post_transition_clip(self) -> None:
        events = [
            {
                "src_lane": 0,
                "T": 120,
                "T_nuendo": 100,
                "T_edit": 100,
                "transition_offset_before": 10,
                "L": 20,
                "kind": "speech",
            },
        ]

        _assign_target_lanes(
            events,
            4,
            reserved_by_lane={0: [(50, 60)]},
            transition_spans_by_lane={0: [(50, 10)]},
        )

        self.assertEqual(events[0]["target_lane"], 0)
        self.assertEqual(events[0]["target_T"], 120)

    def test_lane_assignment_moves_transition_island_without_changing_visible_time(self) -> None:
        events = [
            {
                "src_lane": 2,
                "T": 5016,
                "T_nuendo": 5012,
                "T_edit": 5012,
                "L": 184,
                "kind": "speech",
                "pre_transition_len": 2,
            },
        ]

        _assign_target_lanes(events, 4)

        self.assertEqual(events[0]["target_lane"], 0)
        self.assertEqual(events[0]["target_T"], 5016)

    def test_lane_assignment_avoids_visible_overlap_on_transition_lane(self) -> None:
        events = [
            {
                "src_lane": 7,
                "T": 1854,
                "T_nuendo": 1816,
                "T_edit": 1816,
                "transition_offset_before": 6,
                "L": 18,
                "kind": "music",
            },
            {"src_lane": 6, "T": 1820, "T_edit": 1820, "L": 2, "kind": "music"},
        ]

        _assign_target_lanes(
            events,
            8,
            transition_spans_by_lane={7: [(1848, 6)]},
        )

        self.assertEqual(events[0]["target_lane"], 7)
        self.assertNotEqual(events[1]["target_lane"], 7)

    def test_lane_assignment_does_not_block_event_with_its_own_transition_reservations(self) -> None:
        events = [
            {
                "src_lane": 6,
                "T": 1391,
                "T_edit": 1379,
                "L": 18,
                "kind": "noise",
                "pre_transition_len": 6,
                "post_transition_len": 7,
            },
        ]

        _assign_target_lanes(
            events,
            17,
            reserved_by_lane={7: [(1385, 1391), (1409, 1416)]},
            transition_spans_by_lane={7: [(1385, 6), (1409, 7)]},
        )

        self.assertEqual(events[0]["target_lane"], 7)

    def test_lane_assignment_keeps_owned_pre_transition_groups_serializable(self) -> None:
        events = [
            {
                "src_lane": lane,
                "T": 708,
                "T_edit": 706,
                "L": 81,
                "kind": "speech",
                "pre_transition_len": 1,
            }
            for lane in range(4)
        ]
        _assign_target_lanes(
            events,
            17,
        )

        self.assertEqual([e["target_lane"] for e in events], [0, 1, 2, 3])
        self.assertEqual([e["target_T"] for e in events], [708, 708, 708, 708])

    def test_lane_assignment_reserves_full_owned_pre_transition_span(self) -> None:
        events = [
            {
                "src_lane": 0,
                "T": 100,
                "T_edit": 100,
                "L": 50,
                "kind": "speech",
            },
            {
                "src_lane": 1,
                "T": 150,
                "T_edit": 150,
                "L": 20,
                "kind": "speech",
                "pre_transition_len": 10,
            },
        ]

        _assign_target_lanes(events, 6)

        self.assertEqual(events[0]["target_lane"], 0)
        self.assertNotEqual(events[1]["target_lane"], 0)

    def test_lane_assignment_does_not_double_count_owned_pre_transition(self) -> None:
        events = [
            {
                "src_lane": 7,
                "T": 1854,
                "T_nuendo": 1816,
                "T_edit": 1816,
                "transition_offset_before": 19,
                "L": 18,
                "kind": "noise",
                "pre_transition_len": 6,
                "post_transition_len": 7,
            },
        ]

        _assign_target_lanes(
            events,
            17,
            reserved_by_lane={7: [(1385, 1391), (1409, 1416)]},
            transition_spans_by_lane={7: [(1385, 6), (1409, 7)]},
        )

        self.assertEqual(events[0]["target_lane"], 7)
        self.assertEqual(events[0]["target_T"], 1854)

    def test_lane_assignment_compacts_speech_to_highest_free_lanes(self) -> None:
        events = [
            {"src_lane": 3, "T": 100, "T_edit": 100, "L": 20, "kind": "speech"},
            {"src_lane": 4, "T": 100, "T_edit": 100, "L": 20, "kind": "speech"},
            {"src_lane": 5, "T": 100, "T_edit": 100, "L": 20, "kind": "speech"},
        ]
        reserved = {0: [(100, 120)]}

        _assign_target_lanes(events, 11, reserved_by_lane=reserved)

        self.assertEqual([e["target_lane"] for e in events], [1, 2, 3])

    def test_lane_assignment_prioritizes_longer_overlapping_speech_on_top_lanes(self) -> None:
        events = [
            {
                "src_lane": 2,
                "T": 90,
                "T_edit": 90,
                "L": 20,
                "kind": "speech",
                "source_start": 10,
                "source_length": 20,
            },
            {
                "src_lane": 3,
                "T": 90,
                "T_edit": 90,
                "L": 20,
                "kind": "speech",
                "source_start": 30,
                "source_length": 20,
            },
            {
                "src_lane": 4,
                "T": 100,
                "T_edit": 100,
                "L": 100,
                "kind": "speech",
                "source_start": 50,
                "source_length": 100,
            },
            {
                "src_lane": 5,
                "T": 100,
                "T_edit": 100,
                "L": 100,
                "kind": "speech",
                "source_start": 150,
                "source_length": 100,
            },
        ]

        _assign_target_lanes(events, 9)

        self.assertEqual([e["target_lane"] for e in events], [2, 3, 0, 1])
        self.assertEqual([e["target_T"] for e in events], [90, 90, 100, 100])

    def test_rebuilt_class_zone_compaction_preserves_visible_time_and_blocks_overlap(self) -> None:
        speech = object()
        blocked_speech = object()
        blocker = object()
        rebuilt = {
            0: [(50, 20, blocker, "structural", None, None, 0)],
            1: [(100, 20, speech, "event", 100, "speech", 0)],
            2: [(55, 10, blocked_speech, "event", 55, "speech", 0)],
        }

        moved = _compact_rebuilt_parts_to_class_zones_without_visible_time_shift(rebuilt, 3)

        self.assertEqual(moved, 2)
        self.assertIn((100, 20, speech, "event", 100, "speech", 0), rebuilt[0])
        self.assertIn((55, 10, blocked_speech, "event", 55, "speech", 0), rebuilt[1])
        self.assertNotIn((100, 20, speech, "event", 100, "speech", 0), rebuilt[1])

    def test_rebuilt_class_zone_compaction_moves_noise_to_center_past_transition_only_parts(self) -> None:
        noise = object()
        rebuilt = {
            6: [(1868, 18, noise, "event", 1854, "noise", 0)],
            7: [
                (1848, 6, object(), "transition", None, None, 0),
                (1872, 7, object(), "transition", None, None, 0),
            ],
            8: [],
        }

        moved = _compact_rebuilt_parts_to_class_zones_without_visible_time_shift(
            rebuilt,
            17,
            lambda lane, visible, event_offset: 1854
            if lane == 7 and visible == 1854
            else visible + (2 * event_offset),
        )

        self.assertEqual(moved, 1)
        self.assertIn((1854, 18, noise, "event", 1854, "noise", 0), rebuilt[7])
        self.assertNotIn((1868, 18, noise, "event", 1854, "noise", 0), rebuilt[6])

    def test_rebuilt_class_zone_compaction_keeps_event_already_best_for_class(self) -> None:
        noise = object()
        rebuilt = {
            7: [(100, 20, noise, "event", 100, "noise", 0)],
            8: [],
        }

        moved = _compact_rebuilt_parts_to_class_zones_without_visible_time_shift(rebuilt, 17)

        self.assertEqual(moved, 0)
        self.assertIn((100, 20, noise, "event", 100, "noise", 0), rebuilt[7])

    def test_rebuilt_visible_rebalance_accounts_for_prior_and_owned_transitions(self) -> None:
        event = OperationGroup(82)
        owned_pre = Transition(2)
        rebuilt = {
            5: [
                (100, 2, Transition(2), "transition", None, None, 0),
                (200, 2, Transition(2), "transition", None, None, 0),
                (57043, 84, [owned_pre, event], "event", 57037, "speech", 2),
            ],
        }

        _rebalance_rebuilt_event_parts_to_visible_times(rebuilt, 6)

        self.assertIn((57047, 84, [owned_pre, event], "event", 57037, "speech", 2), rebuilt[5])

    def test_rebuilt_overlap_resolver_moves_event_without_visible_shift(self) -> None:
        event_a = OperationGroup(20)
        event_b = OperationGroup(10)
        rebuilt = {
            0: [
                (90, 5, Transition(5), "transition", None, None, 0),
                (100, 20, event_a, "event", 100, "speech", 0),
                (108, 10, event_b, "event", 108, "speech", 0),
            ],
            1: [],
            2: [],
            3: [],
        }

        _rebalance_rebuilt_event_parts_to_visible_times(rebuilt, 4)
        moved = _resolve_rebuilt_event_overlaps_without_visible_time_shift(rebuilt, 4)

        self.assertEqual(moved, 1)
        self.assertIn((110, 20, event_a, "event", 100, "speech", 0), rebuilt[0])
        self.assertIn((108, 10, event_b, "event", 108, "speech", 0), rebuilt[1])

    def test_rebuilt_overlap_resolver_prefers_non_speech_lane_for_music(self) -> None:
        event_a = OperationGroup(20)
        event_b = OperationGroup(20)
        rebuilt = {
            0: [],
            1: [],
            2: [],
            3: [(100, 20, object(), "structural", None, None, 0)],
            4: [(100, 20, object(), "structural", None, None, 0)],
            5: [],
            6: [(100, 20, object(), "structural", None, None, 0)],
            7: [(100, 20, object(), "structural", None, None, 0)],
            8: [
                (100, 20, event_a, "event", 100, "music", 0),
                (100, 20, event_b, "event", 100, "music", 0),
            ],
        }

        moved = _resolve_rebuilt_event_overlaps_without_visible_time_shift(rebuilt, 9)

        self.assertEqual(moved, 1)
        self.assertEqual(
            [
                item
                for lane in (0, 1, 2)
                for item in rebuilt[lane]
                if item[3] == "event"
            ],
            [],
        )
        self.assertEqual(
            sum(
                1
                for lane in (5, 8)
                for item in rebuilt[lane]
                if item[3] == "event"
            ),
            2,
        )

    def test_rebuilt_overlap_resolver_uses_any_free_lane_before_leaving_overlap(self) -> None:
        event_a = OperationGroup(20)
        event_b = OperationGroup(20)
        rebuilt = {
            0: [],
            1: [],
            2: [],
            3: [(100, 20, object(), "structural", None, None, 0)],
            4: [(100, 20, object(), "structural", None, None, 0)],
            5: [(100, 20, object(), "structural", None, None, 0)],
            6: [(100, 20, object(), "structural", None, None, 0)],
            7: [(100, 20, object(), "structural", None, None, 0)],
            8: [
                (100, 20, event_a, "event", 100, "music", 0),
                (100, 20, event_b, "event", 100, "music", 0),
            ],
        }

        moved = _resolve_rebuilt_event_overlaps_without_visible_time_shift(rebuilt, 9)

        self.assertEqual(moved, 1)
        event_parts = [
            (lane, item)
            for lane in range(9)
            for item in rebuilt[lane]
            if item[3] == "event"
        ]
        self.assertEqual(len(event_parts), 2)
        spans_by_lane: dict[int, list[tuple[int, int]]] = {}
        for lane, item in event_parts:
            t0 = int(item[0])
            t1 = t0 + int(item[1])
            for other_t0, other_t1 in spans_by_lane.setdefault(lane, []):
                self.assertFalse(_intervals_overlap(t0, t1, other_t0, other_t1))
            spans_by_lane[lane].append((t0, t1))
        self.assertTrue(any(lane in {0, 1, 2} for lane, _item in event_parts))

    def test_rebuilt_final_pass_compacts_class_zone_gaps_without_visible_shift(self) -> None:
        noise = OperationGroup(20)
        blocker = object()
        rebuilt = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [(100, 20, blocker, "structural", None, None, 0)],
            7: [],
            8: [],
            9: [],
            10: [(100, 20, noise, "event", 100, "noise", 0)],
            11: [],
            12: [],
            13: [],
            14: [],
            15: [],
            16: [],
        }

        finalize = getattr(
            lane_layout_mod,
            "_finalize_rebuilt_parts_without_visible_time_shift",
            None,
        )
        self.assertIsNotNone(finalize)
        moved = finalize(rebuilt, 17)

        self.assertEqual(moved, 1)
        self.assertIn((100, 20, noise, "event", 100, "noise", 0), rebuilt[7])
        self.assertNotIn((100, 20, noise, "event", 100, "noise", 0), rebuilt[10])

    def test_rebuilt_final_pass_compacts_unknown_gaps_without_visible_shift(self) -> None:
        unknown = OperationGroup(20)
        blocker = object()
        rebuilt = {i: [] for i in range(17)}
        rebuilt[6].append((100, 20, blocker, "structural", None, None, 0))
        rebuilt[10].append((100, 20, unknown, "event", 100, "unknown", 0))

        moved = lane_layout_mod._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 17)

        self.assertEqual(moved, 1)
        self.assertIn((100, 20, unknown, "event", 100, "unknown", 0), rebuilt[7])
        self.assertNotIn((100, 20, unknown, "event", 100, "unknown", 0), rebuilt[10])

    def test_rebuilt_final_pass_preserves_equal_visible_group_order(self) -> None:
        first = OperationGroup(20)
        second = OperationGroup(20)
        rebuilt = {i: [] for i in range(17)}
        rebuilt[9].append((100, 20, first, "event", 100, "noise", 0))
        rebuilt[10].append((100, 20, second, "event", 100, "noise", 0))

        moved = lane_layout_mod._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 17)

        self.assertEqual(moved, 2)
        self.assertIn((100, 20, first, "event", 100, "noise", 0), rebuilt[7])
        self.assertIn((100, 20, second, "event", 100, "noise", 0), rebuilt[8])

    def test_rebuilt_final_pass_uses_serialized_clock_for_upper_gaps(self) -> None:
        aligned = [OperationGroup(20) for _ in range(4)]
        rebuilt = {i: [] for i in range(21)}
        for lane in range(3):
            rebuilt[lane].extend(
                [
                    (50, 10, Transition(10), "transition", None, None, 0),
                    (90, 30, OperationGroup(30), "event", 70, "speech", 0),
                    (140, 20, OperationGroup(20), "event", 120, "speech", 0),
                ]
            )
        for lane, node in zip(range(3, 7), aligned):
            rebuilt[lane].append((100, 20, node, "event", 100, "speech", 0))

        moved = lane_layout_mod._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 21)

        self.assertGreaterEqual(moved, 4)
        for lane, node in enumerate(aligned):
            expected_t = 100
            self.assertIn((expected_t, 20, node, "event", 100, "speech", 0), rebuilt[lane])

    def test_rebuilt_final_pass_keeps_music_group_on_bottom_lanes_with_owned_transitions(self) -> None:
        left = OperationGroup(811)
        right = OperationGroup(811)
        later_left = OperationGroup(231)
        later_right = OperationGroup(231)
        left_pre = Transition(95)
        left_post = Transition(120)
        right_pre = Transition(95)
        right_post = Transition(120)
        rebuilt = {i: [] for i in range(21)}
        left_item = (
            15401,
            1026,
            [left_pre, left, left_post],
            "event",
            15274,
            "music",
            95,
        )
        right_item = (
            15401,
            1026,
            [right_pre, right, right_post],
            "event",
            15274,
            "music",
            95,
        )
        rebuilt[19].append(
            left_item
        )
        rebuilt[20].append(
            right_item
        )
        rebuilt[19].append((16723, 231, later_left, "event", 16119, "music", 0))
        rebuilt[20].append((16723, 231, later_right, "event", 16119, "music", 0))

        lane_layout_mod._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 21)

        self.assertTrue(
            any(part[2] == left_item[2] and part[4] == 15274 for part in rebuilt[19])
        )
        self.assertTrue(
            any(part[2] == right_item[2] and part[4] == 15274 for part in rebuilt[20])
        )
        self.assertFalse(
            any(
                part[3] == "event"
                and part[4] == 15274
                and isinstance(part[2], list)
                and (left in part[2] or right in part[2])
                for lane in range(0, 19)
                for part in rebuilt[lane]
            )
        )

    def test_rebuilt_final_pass_preserves_three_item_aligned_group_order(self) -> None:
        first = OperationGroup(20)
        second = OperationGroup(20)
        third = OperationGroup(20)
        rebuilt = {i: [] for i in range(17)}
        rebuilt[5].append((100, 20, first, "event", 100, "noise", 0))
        rebuilt[6].append((100, 20, second, "event", 100, "noise", 0))
        rebuilt[7].append((100, 20, third, "event", 100, "noise", 0))

        moved = lane_layout_mod._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 17)

        self.assertGreaterEqual(moved, 1)
        self.assertIn((100, 20, first, "event", 100, "noise", 0), rebuilt[6])
        self.assertIn((100, 20, second, "event", 100, "noise", 0), rebuilt[7])
        self.assertIn((100, 20, third, "event", 100, "noise", 0), rebuilt[8])

    def test_rebuilt_final_pass_restores_aligned_group_order_from_source_track(self) -> None:
        first = SourceClip(20, source_track_id=1, start=300)
        second = SourceClip(20, source_track_id=2, start=300)
        third = SourceClip(20, source_track_id=3, start=300)
        rebuilt = {i: [] for i in range(17)}
        rebuilt[6].append((100, 20, second, "event", 100, "noise", 0))
        rebuilt[7].append((100, 20, third, "event", 100, "noise", 0))
        rebuilt[8].append((100, 20, first, "event", 100, "noise", 0))

        moved = lane_layout_mod._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 17)

        self.assertGreaterEqual(moved, 1)
        self.assertIn((100, 20, first, "event", 100, "noise", 0), rebuilt[6])
        self.assertIn((100, 20, second, "event", 100, "noise", 0), rebuilt[7])
        self.assertIn((100, 20, third, "event", 100, "noise", 0), rebuilt[8])

    def test_rebuilt_final_pass_restores_aligned_group_order_from_plan_source_lanes(self) -> None:
        first = OperationGroup(20)
        second = OperationGroup(20)
        third = OperationGroup(20)
        rebuilt = {i: [] for i in range(17)}
        rebuilt[6].append((100, 20, second, "event", 100, "noise", 0))
        rebuilt[7].append((100, 20, third, "event", 100, "noise", 0))
        rebuilt[8].append((100, 20, first, "event", 100, "noise", 0))

        moved = lane_layout_mod._finalize_rebuilt_parts_without_visible_time_shift(
            rebuilt,
            17,
            event_order_by_node_id={id(first): 0, id(second): 1, id(third): 2},
        )

        self.assertGreaterEqual(moved, 1)
        self.assertIn((100, 20, first, "event", 100, "noise", 0), rebuilt[6])
        self.assertIn((100, 20, second, "event", 100, "noise", 0), rebuilt[7])
        self.assertIn((100, 20, third, "event", 100, "noise", 0), rebuilt[8])

    def test_rebuilt_final_pass_checks_overlap_at_owned_transition_adjusted_raw_time(self) -> None:
        owned_pre = Transition(5)
        event = OperationGroup(20)
        rebuilt = {i: [] for i in range(17)}
        rebuilt[6].append((100, 25, object(), "structural", None, None, 0))
        rebuilt[7].append((95, 5, object(), "structural", None, None, 0))
        rebuilt[8].append((100, 25, object(), "structural", None, None, 0))
        rebuilt[9].append((200, 25, [owned_pre, event], "event", 100, "noise", 5))

        moved = lane_layout_mod._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 17)

        self.assertGreaterEqual(moved, 1)
        self.assertIn((105, 25, [owned_pre, event], "event", 100, "noise", 5), rebuilt[7])
        self.assertNotIn((200, 25, [owned_pre, event], "event", 100, "noise", 5), rebuilt[9])

    def test_rebuilt_final_pass_discards_pruned_transition_timing_history(self) -> None:
        event = OperationGroup(20)
        rebuilt = {i: [] for i in range(17)}
        rebuilt[7].append((90, 5, Transition(5), "transition", None, None, 0))
        rebuilt[9].append((200, 20, event, "event", 100, "noise", 0))

        moved = lane_layout_mod._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 17)

        self.assertGreaterEqual(moved, 1)
        self.assertIn((100, 20, event, "event", 100, "noise", 0), rebuilt[7])
        self.assertNotIn((110, 20, event, "event", 100, "noise", 0), rebuilt[7])
        self.assertEqual([item for item in rebuilt[7] if item[3] == "transition"], [])

    def test_rebuilt_final_pass_ignores_empty_audio_wrappers_pruned_after_layout(self) -> None:
        empty_wrapper = OperationGroup(80)
        event = OperationGroup(20)
        rebuilt = {i: [] for i in range(17)}
        rebuilt[7].append((80, 80, empty_wrapper, "empty_audio_wrapper", None, None, 0))
        rebuilt[9].append((100, 20, event, "event", 100, "noise", 0))

        moved = lane_layout_mod._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 17)

        self.assertGreaterEqual(moved, 1)
        self.assertIn((100, 20, event, "event", 100, "noise", 0), rebuilt[7])
        self.assertEqual(
            [item for item in rebuilt[7] if item[3] == "empty_audio_wrapper"],
            [],
        )

    def test_lane_layout_plan_match_prefers_stable_clip_identity_over_top_index(self) -> None:
        expected = {
            "src_lane": 0,
            "top_idx": 10,
            "T_edit": 100,
            "L": 20,
            "source_id": "sid-a",
            "source_track": 1,
            "source_start": 300,
            "source_length": 20,
        }
        stale_top_index_collision = {
            "src_lane": 0,
            "top_idx": 5,
            "T_edit": 100,
            "L": 20,
            "source_id": "sid-b",
            "source_track": 1,
            "source_start": 900,
            "source_length": 20,
        }
        planned_by_identity: dict[tuple[object, ...], list[dict[str, object]]] = {}
        planned_by_top: dict[tuple[int, int], dict[str, object]] = {}
        for ev in (expected, stale_top_index_collision):
            lane_layout_mod._index_lane_layout_plan_event(
                ev,
                planned_by_identity=planned_by_identity,
                planned_by_top=planned_by_top,
            )
        matched_ids: set[int] = set()

        matched = lane_layout_mod._match_lane_layout_plan_event(
            src_lane=0,
            top_idx=5,
            visible_t=100,
            length=20,
            source_id="sid-a",
            source_track=1,
            source_start=300,
            source_length=20,
            planned_by_identity=planned_by_identity,
            planned_by_top=planned_by_top,
            matched_event_ids=matched_ids,
        )

        self.assertIs(matched, expected)
        self.assertIn(id(expected), matched_ids)
        self.assertNotIn(id(stale_top_index_collision), matched_ids)

    def test_sdk_xml_event_preferred_offset_tracks_owned_pre_transition(self) -> None:
        event = {
            "T": 1200,
            "T_edit": 1000,
            "T_nuendo": 1000,
            "pre_transition_len": 3,
        }

        self.assertEqual(_sdk_xml_event_preferred_transition_offset(event), 3)

    def test_sdk_xml_layout_event_plan_preserves_event_context(self) -> None:
        events = [
            {
                "src_lane": 4,
                "top_idx": 12,
                "T": 1200,
                "target_T": 1210,
                "T_edit": 1000,
                "L": 80,
                "target_lane": 1,
                "kind": "speech",
                "source_start": 333,
                "source_length": 80,
                "pre_transition_len": 3,
                "pre_transition_top_idx": 11,
                "post_transition_len": 5,
                "post_transition_top_idx": 13,
            }
        ]

        planned = _build_sdk_lane_layout_events(events, [])

        self.assertEqual(len(planned), 1)
        ev = planned[0]
        self.assertEqual(ev.src_lane, 4)
        self.assertEqual(ev.top_idx, 12)
        self.assertEqual(ev.T_edit, 1210)
        self.assertEqual(ev.visible_t, 1000)
        self.assertEqual(ev.target_lane, 1)
        self.assertEqual(ev.class_kind, "speech")
        self.assertEqual(ev.pre_top_idx, 11)
        self.assertEqual(ev.pre_len, 3)
        self.assertEqual(ev.post_top_idx, 13)
        self.assertEqual(ev.post_len, 5)
        self.assertEqual(ev.source_start, 333)
        self.assertEqual(ev.source_length, 80)

    def test_lane_assignment_keeps_short_speech_as_speech(self) -> None:
        events = [
            {"src_lane": 5, "T": 100, "T_edit": 100, "L": 10, "kind": "speech", "dur_sec": 0.2},
        ]

        _assign_target_lanes(events, 11)

        self.assertEqual(events[0]["target_lane"], 0)

    def test_transition_part_must_be_adjacent_and_non_overlapping(self) -> None:
        event_edges = {100, 150}
        event_intervals = [(100, 150)]

        self.assertTrue(_transition_part_is_serializable(90, 10, event_edges, event_intervals))
        self.assertTrue(_transition_part_is_serializable(150, 10, event_edges, event_intervals))
        self.assertFalse(_transition_part_is_serializable(100, 10, event_edges, event_intervals))
        self.assertFalse(_transition_part_is_serializable(80, 5, event_edges, event_intervals))

    def test_rebuilt_serialization_trims_fully_consumed_leading_transition(self) -> None:
        transition = Transition(5)
        event = OperationGroup(20)

        out_t, out_l, nodes = _trim_leading_rebuilt_transitions_to_cursor(
            95,
            25,
            [transition, event],
            100,
            emitted_transitions={95: transition},
        )

        self.assertEqual(out_t, 100)
        self.assertEqual(out_l, 20)
        self.assertEqual(nodes, [event])

    def test_rebuilt_rebalance_counts_duplicate_transition_span_once(self) -> None:
        shared_transition = Transition(21120)
        previous_event = OperationGroup(195840)
        shifted_event = OperationGroup(336000)
        rebuilt = {i: [] for i in range(17)}
        rebuilt[5].extend(
            [
                (
                    432000,
                    216960,
                    [previous_event, shared_transition],
                    "event",
                    432000,
                    "speech",
                    0,
                ),
                (
                    627840,
                    21120,
                    shared_transition,
                    "transition",
                    None,
                    None,
                    0,
                ),
                (
                    1013760,
                    336000,
                    shifted_event,
                    "event",
                    971520,
                    "speech",
                    0,
                ),
            ]
        )

        _rebalance_rebuilt_event_parts_to_visible_times(rebuilt, 17)

        self.assertIn(
            (1013760, 336000, shifted_event, "event", 971520, "speech", 0),
            rebuilt[5],
        )
        self.assertNotIn(
            (1056000, 336000, shifted_event, "event", 971520, "speech", 0),
            rebuilt[5],
        )

    def test_unavailable_media_stays_unknown_for_lane_layout(self) -> None:
        class SourceClip:
            pass

        self.assertEqual(
            _classify_timeline_block(
                None,
                SourceClip(),
                None,
                YamnetConfig(),
                48000,
                lambda: None,
            ),
            "unknown",
        )

    def test_lane_layout_entrypoint_resolves_media_roots_for_input_aaf(self) -> None:
        class EmptyContent:
            def compositionmobs(self):
                return []

        class EmptyAaf:
            content = EmptyContent()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as td:
            aaf_path = Path(td) / "project.aaf"
            aaf_path.write_bytes(b"fake")
            cfg = FilterConfig(experimental_yamnet_lane_layout=True)

            with mock.patch.object(lane_layout_mod, "open_aaf_lenient", return_value=EmptyAaf()):
                with mock.patch.object(lane_layout_mod, "_configured_media_roots", return_value=()) as resolve:
                    moved = lane_layout_mod.apply_experimental_yamnet_lane_layout(
                        aaf_path,
                        cfg,
                        runtime_essence_paths={},
                    )

        self.assertEqual(moved, 0)
        resolve.assert_called_once_with(aaf_path)

    def test_lane_assignment_keeps_locked_lanes_in_place(self) -> None:
        events = [
            {"src_lane": 0, "T": 100, "T_edit": 100, "L": 20, "kind": "speech"},
            {"src_lane": 1, "T": 100, "T_edit": 100, "L": 20, "kind": "speech"},
        ]

        _assign_target_lanes(events, 3, lane_mutable_by_lane={0: False, 1: True, 2: False})

        self.assertEqual(events[0]["target_lane"], 0)
        self.assertEqual(events[1]["target_lane"], 1)

    def test_music_can_leave_lane_that_contains_separate_transition_island(self) -> None:
        events = [
            {"src_lane": 2, "T": 0, "T_edit": 0, "L": 125, "kind": "music"},
        ]

        _assign_target_lanes(
            events,
            21,
            reserved_by_lane={2: [(1000, 1100)]},
            lane_capacity_by_lane={i: 2000 for i in range(21)},
            lane_signature_by_lane={i: ("Sequence", "") for i in range(21)},
        )

        self.assertEqual(events[0]["target_lane"], 20)

    def test_transition_protection_keeps_neighboring_components(self) -> None:
        seq = Sequence([Filler(10), OperationGroup(20), Transition(2), OperationGroup(30), Filler(40)])

        anchors, indices = _sequence_transition_protected_parts(seq)

        self.assertEqual(indices, {1, 2, 3})
        self.assertEqual([(t, l, type(node).__name__) for t, l, node in anchors], [
            (10, 20, "OperationGroup"),
            (30, 2, "Transition"),
            (32, 30, "OperationGroup"),
        ])

    def test_transition_protection_groups_island_for_atomic_layout(self) -> None:
        seq = Sequence([Filler(985), OperationGroup(585), Transition(152), Filler(152), Filler(10)])

        groups = _sequence_transition_protected_groups(seq)

        self.assertEqual(len(groups), 1)
        start, length, nodes, indices = groups[0]
        self.assertEqual(start, 985)
        self.assertEqual(length, 889)
        self.assertEqual(indices, {1, 2, 3})
        self.assertEqual([type(n).__name__ for n in nodes], ["OperationGroup", "Transition", "Filler"])

    def test_boundary_transitions_are_owned_by_adjacent_events(self) -> None:
        nodes = [Filler(10), Transition(3), OperationGroup(20), Transition(4), OperationGroup(30)]

        self.assertEqual(_event_owned_transition_indices(nodes), {1: 2, 3: 4})

    def test_boundary_transition_is_available_to_both_adjacent_events(self) -> None:
        nodes = [OperationGroup(20), Transition(4), OperationGroup(30)]

        pre_owner, post_owner = lane_layout_mod._event_transition_edge_owners(nodes)

        self.assertEqual(pre_owner, {1: 2})
        self.assertEqual(post_owner, {1: 0})

if __name__ == "__main__":
    unittest.main()
