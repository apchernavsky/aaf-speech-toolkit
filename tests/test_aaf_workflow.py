from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
AAF_IO_ROOT = ROOT / "aaf_io"
if str(AAF_IO_ROOT) not in sys.path:
    sys.path.insert(0, str(AAF_IO_ROOT))

import aaf_workflow
from aaf_workflow_logging import StageLogger
from aaf_io import sdk_tools
import aaf_speech_filter.pyaaf2_filter
from aaf_speech_filter.config import FilterConfig


class _NoSlotContent:
    def compositionmobs(self):
        return [SimpleNamespace(mob_id="comp", slots=[])]


class _NoSlotAaf:
    content = _NoSlotContent()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class SdkApplyRemovalsTests(unittest.TestCase):
    def test_stage_logger_sets_progress_stage_before_emitting_message(self) -> None:
        calls: list[tuple[str, str]] = []

        class Progress:
            def set_stage(self, value: str) -> None:
                calls.append(("stage", value))

            def set_indeterminate(self, value: bool) -> None:
                calls.append(("indeterminate", str(value)))

        logger = StageLogger(emit=lambda msg: calls.append(("emit", msg)), progress=Progress())

        logger.start("AAF SDK XML: сборка выходного AAF через aaffmtconv -ss…", stage="AAF SDK: сборка AAF", indeterminate=True)

        self.assertEqual(
            calls,
            [
                ("indeterminate", "True"),
                ("stage", "AAF SDK: сборка AAF"),
                ("emit", "AAF SDK XML: сборка выходного AAF через aaffmtconv -ss…"),
            ],
        )

    def test_workflow_plan_keeps_non_media_modes_when_media_missing(self) -> None:
        plan = aaf_workflow.make_workflow_plan(
            remove_quiet_clips=True,
            remove_duplicates=True,
            experimental_yamnet_lane_layout=False,
        )

        updated = aaf_workflow.workflow_plan_after_missing_media(plan)

        self.assertFalse(updated.remove_quiet_clips)
        self.assertTrue(updated.remove_duplicates)
        self.assertFalse(updated.experimental_yamnet_lane_layout)

    def test_workflow_plan_fails_if_only_media_dependent_mode_selected(self) -> None:
        plan = aaf_workflow.make_workflow_plan(
            remove_quiet_clips=True,
            remove_duplicates=False,
            experimental_yamnet_lane_layout=False,
        )

        with self.assertRaisesRegex(RuntimeError, "Анализ невозможен"):
            aaf_workflow.workflow_plan_after_missing_media(plan)

    def test_lane_layout_result_already_sdk_safe_uses_lane_result_flags(self) -> None:
        self.assertFalse(aaf_workflow.lane_layout_result_already_sdk_safe(None))
        self.assertFalse(aaf_workflow.lane_layout_result_already_sdk_safe({"moved": 0}))
        self.assertTrue(aaf_workflow.lane_layout_result_already_sdk_safe({"sdk_normalized": True}))
        self.assertTrue(aaf_workflow.lane_layout_result_already_sdk_safe({"sdk_xml_fallback": True}))
        self.assertTrue(aaf_workflow.lane_layout_result_already_sdk_safe({"pyaaf2_lane_layout_primary": True}))
        self.assertTrue(aaf_workflow.lane_layout_result_already_sdk_safe({"pyaaf2_lane_layout_fallback": True}))
        self.assertTrue(
            aaf_workflow.lane_layout_result_already_sdk_safe(
                {"sdk_structured_storage_normalized": True}
            )
        )
        self.assertTrue(
            aaf_workflow.lane_layout_result_already_sdk_safe(
                {"sdk_final_structured_storage_normalized": True}
            )
        )

    def test_structured_storage_normalization_accepts_aaf_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_aaf = root / "layout.aaf"
            output_aaf = root / "normalized.aaf"
            input_aaf.write_bytes(b"source")

            def fake_run(args, **kwargs):
                Path(args[-1]).write_bytes(b"0" * 128)
                return 0, ""

            with mock.patch("aaf_io.sdk_tools._run_subprocess_cancellable", side_effect=fake_run) as run:
                sdk_tools.run_aaffmtconv_to_structured_storage(
                    Path("aaffmtconv.exe"),
                    input_aaf,
                    output_aaf,
                )

        self.assertEqual(run.call_args.args[0][1], "-ss")
        self.assertEqual(Path(run.call_args.args[0][2]), input_aaf.resolve())
        self.assertNotEqual(Path(run.call_args.args[0][3]), output_aaf.resolve())
        self.assertEqual(Path(run.call_args.args[0][3]).parent, output_aaf.resolve().parent)

    def test_finalizes_pyaaf2_lane_layout_container_with_aaf_to_aaf_ss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processed = root / "out.aaf"
            processed.write_bytes(b"pyaaf2")
            result: dict[str, object] = {"pyaaf2_lane_layout_primary": True}
            logs: list[str] = []

            def normalize_stub(_conv, input_aaf, output_aaf, **_kwargs):
                self.assertEqual(Path(input_aaf), processed)
                Path(output_aaf).write_bytes(b"normalized" * 16)

            with (
                mock.patch("aaf_io.sdk_tools.find_aaffmtconv", return_value=Path("aaffmtconv.exe")) as find,
                mock.patch(
                    "aaf_io.sdk_tools.run_aaffmtconv_to_structured_storage",
                    side_effect=normalize_stub,
                ) as normalize,
                mock.patch("aaf_io.heal.cfb.sync_fat_sector_count_header") as sync_fat,
            ):
                changed = aaf_workflow.finalize_pyaaf2_lane_layout_container(
                    processed_aaf=processed,
                    work_dir=root,
                    lane_layout_result_out=result,
                    tools_root=Path("sdk_bin"),
                    emit=logs.append,
                )

            self.assertTrue(changed)
            self.assertEqual(processed.read_bytes(), b"normalized" * 16)
            self.assertFalse((root / "out.aaf.__pyaaf2_lane_layout_sdk_ss.aaf").exists())

        self.assertEqual(find.call_args.args[0], Path("sdk_bin"))
        self.assertEqual(normalize.call_count, 1)
        self.assertEqual(sync_fat.call_count, 1)
        self.assertTrue(result["sdk_final_structured_storage_normalized"])
        self.assertTrue(any("structured-storage" in line for line in logs))

    def test_returns_analysis_removal_count_not_xml_replacement_count(self) -> None:
        removals = {
            ("mob-a", 10, 0, 1),
        }

        def collect_stub(*args, **kwargs):
            kwargs["removal_stats"]["removed_decisions"] = 2
            kwargs["removal_stats"]["removed_quiet"] = 2
            kwargs["removal_stats"]["unique_sdk_keys"] = 1
            return removals

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml = root / "work.xml"
            xml.write_text("<xml />", encoding="utf-8")

            with (
                mock.patch(
                    "aaf_speech_filter.sdk_removals.collect_timeline_sourceclip_removals_for_sdk_xml",
                    side_effect=collect_stub,
                ),
                mock.patch("aaf_io.sdk_xml.apply_removals_in_composition_xml", return_value=1),
                mock.patch("aaf_io.sdk_tools.run_aaffmtconv_to_aaf"),
            ):
                removed, built, replaced, removals_len, fallback = (
                    aaf_workflow.sdk_apply_removals_and_build_aaf(
                        tools=Path("aaffmtconv.exe"),
                        work_aaf=root / "input.aaf",
                        xml_work=xml,
                        processed_aaf=root / "out.aaf",
                        cfg=SimpleNamespace(remove_quiet_clips=True),
                        runtime_essence_paths={},
                    )
                )

        self.assertEqual(removed, 2)
        self.assertTrue(built)
        self.assertEqual(replaced, 1)
        self.assertEqual(removals_len, 1)
        self.assertFalse(fallback)

    def test_sdk_build_preserves_original_locator_urls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml = root / "work.xml"
            xml.write_text(
                "<URL>file:///Users/example/Desktop/audio%20project/A1.wav</URL>",
                encoding="utf-8",
            )

            with (
                mock.patch(
                    "aaf_speech_filter.sdk_removals.collect_timeline_sourceclip_removals_for_sdk_xml",
                    return_value=set(),
                ),
                mock.patch("aaf_io.sdk_xml.apply_removals_in_composition_xml", return_value=0),
                mock.patch("aaf_io.sdk_xml.localize_existing_file_locators_in_xml") as localize,
                mock.patch("aaf_io.sdk_tools.run_aaffmtconv_to_aaf"),
            ):
                aaf_workflow.sdk_apply_removals_and_build_aaf(
                    tools=Path("aaffmtconv.exe"),
                    work_aaf=root / "input.aaf",
                    xml_work=xml,
                    processed_aaf=root / "out.aaf",
                    cfg=SimpleNamespace(remove_quiet_clips=False),
                    runtime_essence_paths={},
                )

        localize.assert_not_called()

    def test_applies_duplicate_removal_after_sdk_build(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml = root / "work.xml"
            xml.write_text("<xml />", encoding="utf-8")

            with (
                mock.patch(
                    "aaf_speech_filter.sdk_removals.collect_timeline_sourceclip_removals_for_sdk_xml",
                    return_value=set(),
                ),
                mock.patch("aaf_io.sdk_xml.apply_removals_in_composition_xml", return_value=0),
                mock.patch("aaf_io.sdk_tools.run_aaffmtconv_to_aaf"),
                mock.patch(
                    "aaf_speech_filter.duplicate_filter.remove_duplicate_timeline_blocks_inplace",
                    side_effect=[3, 3],
                ) as dupes,
            ):
                removed, built, replaced, removals_len, fallback = (
                    aaf_workflow.sdk_apply_removals_and_build_aaf(
                        tools=Path("aaffmtconv.exe"),
                        work_aaf=root / "input.aaf",
                        xml_work=xml,
                        processed_aaf=root / "out.aaf",
                        cfg=SimpleNamespace(remove_quiet_clips=False, remove_duplicates=True),
                        runtime_essence_paths={},
                    )
        )

        self.assertEqual(removed, 3)
        self.assertTrue(built)
        self.assertEqual(replaced, 0)
        self.assertEqual(removals_len, 0)
        self.assertFalse(fallback)
        self.assertEqual(dupes.call_count, 2)
        self.assertTrue(dupes.call_args_list[0].kwargs["dry_run"])
        self.assertFalse(dupes.call_args_list[1].kwargs.get("dry_run", False))

    def test_falls_back_to_pyaaf2_when_sdk_rebuild_rejects_xml(self) -> None:
        removals = {
            ("mob-a", 10, 0, 1),
        }

        def collect_stub(*args, **kwargs):
            kwargs["removal_stats"]["removed_decisions"] = 2
            return removals

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            xml = root / "work.xml"
            xml.write_text("<xml />", encoding="utf-8")

            with (
                mock.patch(
                    "aaf_speech_filter.sdk_removals.collect_timeline_sourceclip_removals_for_sdk_xml",
                    side_effect=collect_stub,
                ),
                mock.patch("aaf_io.sdk_xml.apply_removals_in_composition_xml", return_value=1),
                mock.patch(
                    "aaf_io.sdk_tools.run_aaffmtconv_to_aaf",
                    side_effect=RuntimeError("aaffmtconv -ss failed (1): XML parser position"),
                ),
                mock.patch("aaf_workflow.run_pyaaf2_filter_only", return_value=4) as pyaaf2,
            ):
                removed, built, replaced, removals_len, fallback = (
                    aaf_workflow.sdk_apply_removals_and_build_aaf(
                        tools=Path("aaffmtconv.exe"),
                        work_aaf=root / "input.aaf",
                        xml_work=xml,
                        processed_aaf=root / "out.aaf",
                        cfg=SimpleNamespace(remove_quiet_clips=True),
                        runtime_essence_paths={},
                    )
                )

        self.assertEqual(removed, 4)
        self.assertFalse(built)
        self.assertEqual(replaced, 1)
        self.assertEqual(removals_len, 1)
        self.assertTrue(fallback)
        self.assertEqual(pyaaf2.call_count, 1)

    def test_pyaaf2_filter_restores_cfb_bookkeeping_after_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logs: list[str] = []
            with (
                mock.patch(
                    "aaf_speech_filter.pyaaf2_filter.filter_aaf_speech_only",
                    return_value=7,
                ),
                mock.patch(
                    "aaf_io.heal.cfb.restore_pyAAF2_noop_bookkeeping_from_source",
                    return_value=SimpleNamespace(
                        applied=True,
                        restored_ranges=2,
                        restored_bytes=10,
                        skipped_ranges=0,
                        truncated_to_source_size=True,
                    ),
                ) as restore,
            ):
                removed = aaf_workflow.run_pyaaf2_filter_only(
                    work_aaf=root / "work.aaf",
                    processed_aaf=root / "out.aaf",
                    cfg=SimpleNamespace(),
                    runtime_essence_paths={},
                    work_dir=root,
                    emit=logs.append,
                    cfb_bookkeeping_source_aaf=root / "original.aaf",
                )

        self.assertEqual(removed, 7)
        self.assertEqual(restore.call_count, 1)
        self.assertEqual(restore.call_args.args[0], root / "original.aaf")
        self.assertTrue(any("CFB/OLE" in line for line in logs))

    def test_pyaaf2_filter_logs_final_completion_after_post_processing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logs: list[str] = []

            def filter_stub(_work, _out, _cfg, **kwargs):
                Path(_out).touch()
                kwargs["log_callback"]("PyAAF2: основной проход завершён.")
                return 7

            with (
                mock.patch(
                    "aaf_speech_filter.pyaaf2_filter.filter_aaf_speech_only",
                    side_effect=filter_stub,
                ),
                mock.patch("aaf_workflow.flatten_top_level_empty_sound_operationgroups", return_value=0),
                mock.patch(
                    "aaf_io.heal.cfb.restore_pyAAF2_noop_bookkeeping_from_source",
                    return_value=SimpleNamespace(applied=False),
                ),
            ):
                removed = aaf_workflow.run_pyaaf2_filter_only(
                    work_aaf=root / "work.aaf",
                    processed_aaf=root / "out.aaf",
                    cfg=SimpleNamespace(),
                    runtime_essence_paths={},
                    work_dir=root,
                    emit=logs.append,
                )

        self.assertEqual(removed, 7)
        main_done_idx = logs.index("PyAAF2: основной проход завершён.")
        final_done_idx = logs.index("PyAAF2: обработка завершена.")
        self.assertLess(main_done_idx, final_done_idx)
        self.assertTrue(any("постобработка" in line for line in logs[main_done_idx + 1 : final_done_idx]))
        self.assertTrue(any("OperationGroup" in line for line in logs[main_done_idx + 1 : final_done_idx]))
        self.assertTrue(any("CFB/OLE" in line for line in logs[main_done_idx + 1 : final_done_idx]))
        self.assertTrue(any("постобработка завершена" in line for line in logs[main_done_idx + 1 : final_done_idx]))
        self.assertFalse(any("постобработка" in line for line in logs[final_done_idx + 1 :]))

    def test_pyaaf2_filter_finalizes_sdk_normalized_lane_result_after_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logs: list[str] = []
            result: dict[str, object] = {}

            def filter_stub(_work, _out, _cfg, **kwargs):
                Path(_out).touch()
                kwargs["lane_layout_result_out"]["sdk_structured_storage_normalized"] = True
                return 7

            def normalize_stub(_conv, _input_aaf, output_aaf, **_kwargs):
                Path(output_aaf).write_bytes(b"0" * 128)

            with (
                mock.patch(
                    "aaf_speech_filter.pyaaf2_filter.filter_aaf_speech_only",
                    side_effect=filter_stub,
                ),
                mock.patch("aaf_workflow.flatten_top_level_empty_sound_operationgroups", return_value=4) as flatten,
                mock.patch("aaf_io.heal.cfb.restore_pyAAF2_noop_bookkeeping_from_source") as restore,
                mock.patch("aaf_io.sdk_tools.find_aaffmtconv", return_value=Path("aaffmtconv.exe")),
                mock.patch(
                    "aaf_io.sdk_tools.run_aaffmtconv_to_structured_storage",
                    side_effect=normalize_stub,
                ) as normalize,
                mock.patch("aaf_io.heal.cfb.sync_fat_sector_count_header") as sync_fat,
            ):
                removed = aaf_workflow.run_pyaaf2_filter_only(
                    work_aaf=root / "work.aaf",
                    processed_aaf=root / "out.aaf",
                    cfg=SimpleNamespace(),
                    runtime_essence_paths={},
                    work_dir=root,
                    emit=logs.append,
                    lane_layout_result_out=result,
                    cfb_bookkeeping_source_aaf=root / "original.aaf",
                )

        self.assertEqual(removed, 7)
        self.assertEqual(flatten.call_count, 1)
        self.assertEqual(normalize.call_count, 1)
        self.assertEqual(sync_fat.call_count, 1)
        self.assertEqual(restore.call_count, 0)
        self.assertTrue(result["sdk_final_structured_storage_normalized"])
        self.assertTrue(any("finalized post-layout cleanup" in line for line in logs))

    def test_pyaaf2_filter_keeps_lane_layout_when_transitions_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logs: list[str] = []
            result: dict[str, object] = {}
            seen_cfg: list[object] = []

            def filter_stub(_work, _out, cfg, **_kwargs):
                seen_cfg.append(cfg)
                Path(_out).touch()
                return 3

            with (
                mock.patch(
                    "aaf_speech_filter.pyaaf2_filter.filter_aaf_speech_only",
                    side_effect=filter_stub,
                ),
                mock.patch(
                    "aaf_io.heal.cfb.restore_pyAAF2_noop_bookkeeping_from_source",
                    return_value=SimpleNamespace(applied=False),
                ),
                mock.patch(
                    "aaf_workflow.flatten_top_level_empty_sound_operationgroups",
                    return_value=4,
                ) as flatten,
            ):
                removed = aaf_workflow.run_pyaaf2_filter_only(
                    work_aaf=root / "work.aaf",
                    processed_aaf=root / "out.aaf",
                    cfg=FilterConfig(
                        experimental_yamnet_lane_layout=True,
                        remove_quiet_clips=True,
                        remove_duplicates=True,
                    ),
                    runtime_essence_paths={},
                    work_dir=root,
                    emit=logs.append,
                    lane_layout_result_out=result,
                )

        self.assertEqual(removed, 3)
        self.assertEqual(len(seen_cfg), 1)
        self.assertTrue(seen_cfg[0].experimental_yamnet_lane_layout)
        self.assertNotIn("skipped", result)
        self.assertEqual(flatten.call_count, 1)
        self.assertTrue(any("OperationGroup" in line for line in logs))

    def test_post_sdk_lane_layout_skips_when_pyaaf2_cannot_open_sdk_output(self) -> None:
        result: dict[str, object] = {}
        logs: list[str] = []
        progress: list[tuple[float, float]] = []

        with mock.patch(
            "aaf_io.compat.pyaaf2_lenient.open_aaf_lenient",
            side_effect=IndexError("list index out of range"),
        ):
            aaf_workflow.post_sdk_lane_layout_if_enabled(
                cfg=SimpleNamespace(experimental_yamnet_lane_layout=True),
                processed_aaf=Path("processed.aaf"),
                work_dir=Path("."),
                runtime_essence_paths={},
                progress_callback=lambda done, total: progress.append((done, total)),
                emit=logs.append,
                lane_layout_result_out=result,
            )

        self.assertTrue(result["skipped"])
        self.assertEqual(result["skip_reason"], "pyaaf2_open_failed_after_sdk")
        self.assertIn("проверка промежуточного AAF", logs[0])
        self.assertTrue(any("PyAAF2" in line for line in logs[1:]))
        self.assertEqual(progress, [(1.0, 1.0)])

    def test_post_sdk_lane_layout_fails_when_yamnet_model_unavailable(self) -> None:
        result: dict[str, object] = {}
        logs: list[str] = []
        progress: list[tuple[float, float]] = []

        with mock.patch(
            "aaf_io.compat.pyaaf2_lenient.open_aaf_lenient",
            return_value=_NoSlotAaf(),
        ), mock.patch(
            "aaf_speech_filter.aaf_yamnet_lane_layout.apply_experimental_yamnet_lane_layout",
            side_effect=RuntimeError("YAMNet model is not available locally."),
        ) as apply_layout:
            with self.assertRaisesRegex(RuntimeError, "YAMNet model is not available"):
                aaf_workflow.post_sdk_lane_layout_if_enabled(
                    cfg=SimpleNamespace(experimental_yamnet_lane_layout=True),
                    processed_aaf=Path("processed.aaf"),
                    work_dir=Path("."),
                    runtime_essence_paths={},
                    progress_callback=lambda done, total: progress.append((done, total)),
                    emit=logs.append,
                    lane_layout_result_out=result,
                )

        apply_layout.assert_called_once()
        self.assertNotIn("skipped", result)
        self.assertEqual(progress, [])

    def test_pyaaf2_lane_layout_fails_when_yamnet_model_unavailable(self) -> None:
        result: dict[str, object] = {}
        logs: list[str] = []

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_aaf = root / "input.aaf"
            output_aaf = root / "output.aaf"
            input_aaf.write_bytes(b"fake-aaf")

            with mock.patch(
                "aaf_speech_filter.pyaaf2_filter.open_aaf_lenient",
                return_value=_NoSlotAaf(),
            ), mock.patch(
                "aaf_speech_filter.pyaaf2_filter._total_timeline_sourceclips",
                return_value=1,
            ), mock.patch(
                "aaf_speech_filter.pyaaf2_filter.remove_duplicate_timeline_blocks_inplace",
            ), mock.patch(
                "aaf_speech_filter.pyaaf2_filter._output_aaf_lightly_readable",
                return_value=True,
            ), mock.patch(
                "aaf_speech_filter.aaf_yamnet_lane_layout.apply_experimental_yamnet_lane_layout",
                side_effect=RuntimeError("YAMNet model is not available locally."),
            ) as apply_layout:
                with self.assertRaisesRegex(RuntimeError, "YAMNet model is not available"):
                    aaf_speech_filter.pyaaf2_filter.filter_aaf_speech_only(
                        input_aaf,
                        output_aaf,
                        FilterConfig(
                            remove_quiet_clips=False,
                            remove_duplicates=True,
                            experimental_yamnet_lane_layout=True,
                        ),
                        log_callback=logs.append,
                        lane_layout_result_out=result,
                    )

            self.assertFalse(output_aaf.exists())
        apply_layout.assert_called_once()
        self.assertNotIn("skipped", result)

    def test_cleanup_work_dir_retains_shared_tool_parent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            work_dir = parent / "__aaf_tool_work" / "sample" / "run"
            work_dir.mkdir(parents=True)
            (work_dir / "temp.txt").write_text("x", encoding="utf-8")

            aaf_workflow.cleanup_work_dir(work_dir, input_parent=parent)

            self.assertFalse(work_dir.exists())
            self.assertTrue((parent / "__aaf_tool_work").is_dir())
            self.assertTrue(parent.exists())

    def test_cleanup_aaf_tool_work_root_preserves_populated_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            root = parent / "__aaf_tool_work" / "sample"
            root.mkdir(parents=True)
            (root / "temp.txt").write_text("x", encoding="utf-8")
            keep = parent / "keep"
            keep.mkdir()

            aaf_workflow.cleanup_aaf_tool_work_root(parent)

            self.assertEqual((root / "temp.txt").read_text(encoding="utf-8"), "x")
            self.assertTrue(keep.exists())


if __name__ == "__main__":
    unittest.main()
