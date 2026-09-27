from __future__ import annotations

import sys
import tempfile
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

import aaf_pipeline  # noqa: E402


class AafPipelineTests(unittest.TestCase):
    def test_yamnet_layout_without_local_model_downloads_before_work_copy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_aaf = Path(td) / "input.aaf"
            input_aaf.write_bytes(b"not a real aaf")

            with mock.patch(
                "aaf_speech_filter.speech_yamnet.yamnet_model_available_locally",
                return_value=False,
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet.ensure_yamnet_model_available",
            ) as ensure_model, mock.patch(
                "aaf_pipeline.emit_comaafinfo_header"
            ), mock.patch(
                "aaf_pipeline.make_work_dir_near_input",
                return_value=Path(td) / "__work",
            ), mock.patch(
                "aaf_pipeline.prepare_work_copy_for_processing",
                side_effect=RuntimeError("stop after model download boundary"),
            ) as prepare_work_copy:
                _user_error, _processed, error, _removed, _hint = aaf_pipeline._run_aaf_pipeline_impl(
                    input_aaf,
                    experimental_yamnet_lane_layout=True,
                )

        self.assertIn("stop after model download boundary", str(error))
        ensure_model.assert_called_once()
        self.assertTrue(ensure_model.call_args.args[0].allow_download)
        prepare_work_copy.assert_called_once()

    def test_yamnet_layout_explicit_download_checks_model_before_work_copy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_aaf = Path(td) / "input.aaf"
            input_aaf.write_bytes(b"not a real aaf")

            with mock.patch(
                "aaf_speech_filter.speech_yamnet.yamnet_model_available_locally",
                return_value=False,
            ), mock.patch(
                "aaf_speech_filter.speech_yamnet.ensure_yamnet_model_available",
            ) as ensure_model, mock.patch(
                "aaf_pipeline.emit_comaafinfo_header"
            ), mock.patch(
                "aaf_pipeline.make_work_dir_near_input",
                return_value=Path(td) / "__work",
            ), mock.patch(
                "aaf_pipeline.prepare_work_copy_for_processing",
                side_effect=RuntimeError("stop after model download boundary"),
            ) as prepare_work_copy:
                _user_error, _processed, error, _removed, _hint = aaf_pipeline._run_aaf_pipeline_impl(
                    input_aaf,
                    experimental_yamnet_lane_layout=True,
                    allow_yamnet_download=True,
                )

        self.assertIn("stop after model download boundary", str(error))
        ensure_model.assert_called_once()
        self.assertTrue(ensure_model.call_args.args[0].allow_download)
        prepare_work_copy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
