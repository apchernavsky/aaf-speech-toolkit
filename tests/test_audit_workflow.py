from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import aaf_workflow


class WorkflowInvariantTests(unittest.TestCase):
    def test_lane_fastpath_never_skips_duplicate_removal(self):
        cfg = SimpleNamespace(experimental_yamnet_lane_layout=True,
                              remove_quiet_clips=False, remove_duplicates=True)
        with mock.patch('shutil.copy2'), mock.patch('aaf_speech_filter.aaf_yamnet_lane_layout.apply_experimental_yamnet_lane_layout'):
            used = aaf_workflow.maybe_run_lane_layout_only_fastpath(cfg=cfg,
                work_aaf=Path('source'), processed_aaf=Path('output'),
                runtime_essence_paths={}, work_dir=Path('.'), emit=lambda _: None)
        self.assertFalse(used)

    def test_partial_and_excess_sdk_matches_fallback(self):
        for matched in (2, 4):
            with self.subTest(matched=matched), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                xml = root / 'work.xml'
                xml.write_text('<xml/>')
                removals = {('a', 10, 0, 1), ('b', 10, 0, 1), ('c', 10, 0, 1)}
                with mock.patch('aaf_speech_filter.sdk_removals.collect_timeline_sourceclip_removals_for_sdk_xml', return_value=removals), \
                     mock.patch('aaf_io.sdk_xml.apply_removals_in_composition_xml', return_value=matched), \
                     mock.patch('aaf_io.sdk_tools.run_aaffmtconv_to_aaf') as build, \
                     mock.patch('aaf_workflow.run_pyaaf2_filter_only', return_value=3):
                    result = aaf_workflow.sdk_apply_removals_and_build_aaf(
                        tools=Path('fake'), work_aaf=root/'source.aaf', xml_work=xml,
                        processed_aaf=root/'output.aaf',
                        cfg=SimpleNamespace(remove_quiet_clips=True), runtime_essence_paths={})
                self.assertTrue(result[4])
                self.assertEqual(result[0], 3)
                build.assert_not_called()

if __name__ == '__main__':
    unittest.main()
