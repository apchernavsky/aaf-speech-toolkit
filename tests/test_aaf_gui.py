from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
AAF_IO_ROOT = ROOT / "aaf_io"
AAF_FILTER_ROOT = ROOT / "aaf_speech_filter"
for path in (ROOT, AAF_IO_ROOT, AAF_FILTER_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

import aaf_gui  # noqa: E402


class _FakeProgressbar:
    def __init__(self) -> None:
        self.stopped = False
        self.configured: list[dict[str, object]] = []
        self.items: dict[str, object] = {}

    def stop(self) -> None:
        self.stopped = True

    def configure(self, **kwargs) -> None:
        self.configured.append(dict(kwargs))

    def __setitem__(self, key: str, value: object) -> None:
        self.items[key] = value


class _FakeLabel:
    def __init__(self) -> None:
        self.configured: list[dict[str, object]] = []

    def config(self, **kwargs) -> None:
        self.configured.append(dict(kwargs))


class AafGuiTests(unittest.TestCase):
    def test_reset_progress_display_clears_stale_file_progress_before_processing(self) -> None:
        bar = _FakeProgressbar()
        pct = _FakeLabel()
        stage = _FakeLabel()
        timer = _FakeLabel()

        aaf_gui._reset_progress_display(bar, pct, stage, timer, timer_text="")

        self.assertTrue(bar.stopped)
        self.assertIn({"mode": "determinate"}, bar.configured)
        self.assertEqual(bar.items["value"], 0)
        self.assertEqual(pct.configured[-1], {"text": "0%"})
        self.assertEqual(stage.configured[-1], {"text": ""})
        self.assertEqual(timer.configured[-1], {"text": ""})

    def test_path_entry_remains_copyable_but_not_editable_while_busy(self) -> None:
        self.assertEqual(aaf_gui._path_entry_state_for_busy(True), "readonly")
        self.assertEqual(aaf_gui._path_entry_state_for_busy(False), "normal")

    def test_lane_layout_checkbox_is_enabled_by_default(self) -> None:
        self.assertTrue(aaf_gui._default_experimental_lane_layout_enabled())

    def test_yamnet_download_policy_disabled_when_lanes_disabled(self) -> None:
        allow_download = aaf_gui._resolve_yamnet_download_policy(
            experimental_lane_layout=False,
        )

        self.assertFalse(allow_download)

    def test_yamnet_download_policy_disabled_when_model_is_local(self) -> None:

        with mock.patch(
            "aaf_speech_filter.speech_yamnet.yamnet_model_available_locally",
            return_value=True,
        ):
            allow_download = aaf_gui._resolve_yamnet_download_policy(
                experimental_lane_layout=True,
            )

        self.assertFalse(allow_download)

    def test_yamnet_download_policy_enabled_when_model_missing(self) -> None:
        with mock.patch(
            "aaf_speech_filter.speech_yamnet.yamnet_model_available_locally",
            return_value=False,
        ):
            allow_download = aaf_gui._resolve_yamnet_download_policy(
                experimental_lane_layout=True,
            )

        self.assertTrue(allow_download)


if __name__ == "__main__":
    unittest.main()
