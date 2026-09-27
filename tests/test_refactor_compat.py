from __future__ import annotations

import unittest

from aaf_speech_filter.config import FilterConfig
from aaf_speech_filter.exceptions import SpeechFilterCancelled
from aaf_speech_filter.progress import (
    pipeline_progress_to_pair_sink,
    progress_set_global,
    progress_stage_wrap,
)
from aaf_speech_filter.timeline_format import format_timecode_from_units
from aaf_speech_filter import (
    FilterConfig as PackageFilterConfig,
    PipelineProgress as PackagePipelineProgress,
    SpeechFilterCancelled as PackageSpeechFilterCancelled,
)


class RefactorCompatTests(unittest.TestCase):
    def test_package_exports_are_preserved(self) -> None:
        self.assertIs(PackageFilterConfig, FilterConfig)
        self.assertIs(PackageSpeechFilterCancelled, SpeechFilterCancelled)
        self.assertIsNotNone(PackagePipelineProgress)

    def test_progress_stage_wrap_maps_local_progress(self) -> None:
        calls: list[tuple[float, float]] = []
        wrapped = progress_stage_wrap(lambda processed, total: calls.append((processed, total)), 0.25, 0.5)
        self.assertIsNotNone(wrapped)

        assert wrapped is not None
        wrapped(1, 4)

        self.assertEqual(calls, [(37.5, 100.0)])

    def test_timeline_format_matches_existing_timecode_shape(self) -> None:
        self.assertEqual(format_timecode_from_units(50, 25.0), "00:02:00")
        self.assertEqual(format_timecode_from_units(48000, 48000.0), "00:01.000")


if __name__ == "__main__":
    unittest.main()
