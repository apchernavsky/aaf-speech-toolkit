from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for package in (ROOT, ROOT / 'aaf_io', ROOT / 'aaf_speech_filter'):
    sys.path.insert(0, str(package))

import aaf2
import aaf2.cfb
import aaf2.file
import aaf2.metadict
import aaf_pipeline
import aaf_workflow
from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
from aaf_io.roundtrip import sdk_roundtrip
from aaf_io.sdk_tools import run_aaffmtconv_to_structured_storage
from aaf_speech_filter.exceptions import SpeechFilterCancelled


class AuditIoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT, prefix='audit_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.aaf'
        with aaf2.open(str(self.source), 'w'):
            pass

    def test_guard_rejects_resolved_alias(self):
        (self.root / 'sub').mkdir()
        with self.assertRaises(RuntimeError):
            aaf_workflow.assert_not_overwriting_source(self.source, self.root / 'sub' / '..' / self.source.name)

    def test_guard_rejects_hardlink(self):
        alias = self.root / 'alias.aaf'
        alias.hardlink_to(self.source)
        with self.assertRaises(RuntimeError):
            aaf_workflow.assert_not_overwriting_source(self.source, alias)

    def test_roundtrip_rejects_same_file_before_removal(self):
        before = self.source.read_bytes()
        with mock.patch('aaf_io.roundtrip.find_aaffmtconv', return_value=Path('fake.exe')), mock.patch('aaf_io.roundtrip.run_aaffmtconv_to_xml', side_effect=RuntimeError('export failed')):
            with self.assertRaises(RuntimeError):
                sdk_roundtrip(self.source, self.source, try_cfb_fat_header_heal=False, on_xml_crash="raise")
        self.assertEqual(self.source.read_bytes(), before)

    def test_structured_storage_rejects_same_file_before_removal(self):
        before = self.source.read_bytes()
        with mock.patch('aaf_io.sdk_tools._run_subprocess_cancellable', return_value=(0, '')):
            with self.assertRaisesRegex(RuntimeError, 'source|input'):
                run_aaffmtconv_to_structured_storage(Path('missing.exe'), self.source, self.source)
        self.assertEqual(self.source.read_bytes(), before)

    def test_invalid_input_does_not_clean_another_run(self):
        active = self.root / '__aaf_tool_work' / 'active'
        active.mkdir(parents=True)
        sentinel = active / 'work.bin'
        sentinel.write_bytes(b'other run')
        result = aaf_pipeline.run_aaf_pipeline(self.root / 'missing.aaf')
        self.assertIsNotNone(result[2])
        self.assertTrue(sentinel.exists(), 'another run was deleted')
        self.assertEqual(sentinel.read_bytes(), b'other run')

    def test_workspace_allocations_are_exclusive(self):
        with mock.patch('time.strftime', return_value='fixed'):
            first = aaf_workflow.make_work_dir_near_input(self.source)
            second = aaf_workflow.make_work_dir_near_input(self.source)
        self.assertNotEqual(first, second)

    def test_save_failure_is_propagated(self):
        opened = None
        try:
            with mock.patch.object(aaf2.file.AAFFile, 'save', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    with open_aaf_lenient(self.source, 'r+') as opened:
                        pass
            self.assertTrue(opened.f.closed)
            self.assertFalse(opened.is_open)
        finally:
            if opened is not None:
                opened.is_open = False
                opened.f.close()

    def test_compatibility_does_not_mutate_library_classes(self):
        originals = (aaf2.cfb.Stream.write, aaf2.cfb.CompoundFileBinary.read_dir_entry,
                     aaf2.metadict.MetaDictionary.read_properties)
        with open_aaf_lenient(self.source, 'r'):
            self.assertEqual(originals, (aaf2.cfb.Stream.write,
                aaf2.cfb.CompoundFileBinary.read_dir_entry,
                aaf2.metadict.MetaDictionary.read_properties))

    def test_sdk_cancellation_is_not_successful_copy(self):
        output = self.root / 'out.aaf'
        def cancel():
            raise SpeechFilterCancelled()
        def exporter(*args, cancel_check=None, **kwargs):
            cancel_check()
        with mock.patch('aaf_io.roundtrip.find_aaffmtconv', return_value=Path('fake.exe')), \
             mock.patch('aaf_io.roundtrip.run_aaffmtconv_to_xml', side_effect=exporter):
            with self.assertRaises(SpeechFilterCancelled):
                sdk_roundtrip(self.source, output, cancel_check=cancel,
                              try_cfb_fat_header_heal=False)
        self.assertFalse(output.exists())

    def test_pipeline_failure_preserves_previous_output(self):
        output = self.root / 'input_processed.aaf'
        output.write_bytes(b'previous successful output')
        # Use a real audio timeline: an empty AAF is now a successful no-op.
        with aaf2.open(str(self.source), 'r+') as aaf:
            comp = aaf.create.CompositionMob()
            aaf.content.mobs.append(comp)
            slot = comp.create_timeline_slot(edit_rate=48000)
            slot.segment = aaf.create.SourceClip(media_kind='sound', length=100)
        with mock.patch('aaf_pipeline.emit_comaafinfo_header'), \
             mock.patch('aaf_pipeline.find_aaffmtconv', side_effect=RuntimeError('SDK missing')):
            result = aaf_pipeline.run_aaf_pipeline(self.source, remove_quiet_clips=False,
                                                  remove_duplicates=True, log_callback=lambda _: None)
        self.assertIn('SDK missing', result[2])
        self.assertTrue(output.exists(), 'previous output was deleted')
        self.assertEqual(output.read_bytes(), b'previous successful output')

    def test_roundtrip_publication_is_local_to_new_destination_parent(self):
        from aaf_io.roundtrip import RoundtripResult, RoundtripMethod
        output = self.root / 'new-output' / 'result.aaf'
        xml = self.root / 'external-scratch' / 'requested.xml'
        def build(source, candidate, **kwargs):
            self.assertTrue(candidate.is_relative_to(output.parent))
            self.assertTrue(kwargs['xml_sibling'].is_relative_to(xml.parent))
            candidate.write_bytes(b'completed')
            return RoundtripResult(True, RoundtripMethod.sdk_xml_raw, 'ok')
        with mock.patch('aaf_io.roundtrip._sdk_roundtrip_work', side_effect=build):
            result = sdk_roundtrip(self.source, output, xml_sibling=xml)
        self.assertTrue(result.ok)
        self.assertEqual(output.read_bytes(), b'completed')

    def test_roundtrip_failure_preserves_previous_output(self):
        output = self.root / 'roundtrip.aaf'
        output.write_bytes(b'previous')
        with mock.patch('aaf_io.roundtrip.find_aaffmtconv', return_value=Path('fake.exe')), \
             mock.patch('aaf_io.roundtrip.run_aaffmtconv_to_xml', side_effect=RuntimeError('SDK failed')):
            result = sdk_roundtrip(self.source, output, on_xml_crash='raise',
                                   try_cfb_fat_header_heal=False)
        self.assertFalse(result.ok)
        self.assertTrue(output.exists(), 'previous roundtrip output deleted')
        self.assertEqual(output.read_bytes(), b'previous')

    def test_cleanup_root_preserves_live_work(self):
        work = aaf_workflow.make_work_dir_near_input(self.source)
        marker = work / 'active'
        marker.touch()
        aaf_workflow.cleanup_aaf_tool_work_root(self.root)
        self.assertTrue(marker.exists())

    def test_roundtrip_replacement_failure_preserves_old_result(self):
        output = self.root / 'out.aaf'
        output.write_bytes(b'previous')
        def roundtrip(source, destination, **kwargs):
            destination.write_bytes(b'new')
            from aaf_io.roundtrip import RoundtripResult, RoundtripMethod
            return RoundtripResult(True, RoundtripMethod.sdk_xml_raw, 'ok')
        with mock.patch('aaf_io.roundtrip.sdk_roundtrip', side_effect=roundtrip), \
             mock.patch.object(Path, 'replace', side_effect=OSError('rename failed')):
            with self.assertRaises(OSError):
                aaf_workflow.nuendo_safe_roundtrip_inplace(processed_aaf=output,
                    work_dir=self.root, aaf_tools_dir=None)
        self.assertEqual(output.read_bytes(), b'previous')


    def test_lenient_stream_writes_across_sector_boundaries(self):
        for length in (1, 63, 64, 65, 511, 512, 513, 4095, 4096, 4097):
            with self.subTest(length=length):
                data = bytes(index % 251 for index in range(length))
                with open_aaf_lenient(self.source, 'r+') as aaf:
                    stream = aaf.cfb.open('/audit-stream', 'w')
                    split = length // 2
                    stream.write(data[:split])
                    stream.write(data[split:])
                with aaf2.open(str(self.source), 'r') as aaf:
                    self.assertEqual(aaf.cfb.open('/audit-stream', 'r').read(), data)

    def test_pipeline_publishes_only_completed_candidate(self):
        output = self.root / 'input_processed.aaf'
        output.write_bytes(b'previous')
        original = self.source.read_bytes()
        def execute(source, *, work_dir, processed_path, **kwargs):
            self.assertEqual(output.read_bytes(), b'previous')
            processed_path.write_bytes(original)
            return None, processed_path, None, 4, {'verified': True}
        with mock.patch('aaf_pipeline._execute_aaf_pipeline', side_effect=execute):
            result = aaf_pipeline.run_aaf_pipeline(self.source)
        self.assertEqual(result[1], output)
        self.assertEqual(result[3], 4)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertTrue((self.root / '__aaf_tool_work').is_dir())
        self.assertEqual(list((self.root / '__aaf_tool_work').iterdir()), [])

    def test_cancellation_before_commit_preserves_previous_output(self):
        output = self.root / 'input_processed.aaf'
        output.write_bytes(b'previous')
        cancel = threading.Event()
        def execute(source, *, processed_path, **kwargs):
            processed_path.write_bytes(b'new')
            cancel.set()
            return None, processed_path, None, 0, None
        with mock.patch('aaf_pipeline._execute_aaf_pipeline', side_effect=execute):
            result = aaf_pipeline.run_aaf_pipeline(self.source, cancel_event=cancel)
        self.assertEqual(result[2], aaf_pipeline.MSG_PIPELINE_STOPPED)
        self.assertEqual(output.read_bytes(), b'previous')

    def test_prepare_copy_rejects_normalized_input_alias(self):
        original = self.source.read_bytes()
        (self.root / 'nested').mkdir()
        alias = self.root / 'nested' / '..' / self.source.name
        with self.assertRaisesRegex(RuntimeError, 'overwrite'):
            aaf_workflow.prepare_work_copy_for_processing(input_aaf=self.source,
                work_dir=self.root, work_aaf=alias, emit=lambda _: None)
        self.assertEqual(self.source.read_bytes(), original)

    def test_duplicate_save_failure_preserves_previous_pipeline_output(self):
        from aaf_speech_filter.config import FilterConfig
        from aaf_speech_filter.pyaaf2_filter import filter_aaf_speech_only
        with aaf2.open(str(self.source), 'r+') as aaf:
            comp = aaf.create.CompositionMob()
            aaf.content.mobs.append(comp)
            comp.create_timeline_slot(48000).segment = aaf.create.Filler(media_kind='sound', length=100)
        output = self.root / 'input_processed.aaf'
        output.write_bytes(b'previous')
        def execute(source, *, processed_path, **kwargs):
            filter_aaf_speech_only(source, processed_path,
                FilterConfig(remove_quiet_clips=False, remove_duplicates=True))
            return None, processed_path, None, 0, None
        with mock.patch('aaf_pipeline._execute_aaf_pipeline', side_effect=execute), \
             mock.patch('aaf_speech_filter.pyaaf2_filter.remove_duplicate_timeline_blocks_inplace',
                        side_effect=OSError('disk full on duplicate close')):
            result = aaf_pipeline.run_aaf_pipeline(self.source)
        self.assertIn('disk full', result[2] or '')
        self.assertEqual(output.read_bytes(), b'previous')

    def test_long_input_name_does_not_expand_private_sdk_paths(self):
        source = self.root / ('a' * 160 + '.aaf')
        source.write_bytes(self.source.read_bytes())
        expected = source.with_name(source.stem + '_processed.aaf')
        def execute(input_path, *, work_dir, processed_path, **kwargs):
            # Reserve room for the nested SDK writer and stream sidecars.
            self.assertLess(len(str(processed_path.relative_to(source.parent))), 60)
            self.assertTrue(str(processed_path.relative_to(source.parent)).isascii())
            processed_path.write_bytes(source.read_bytes())
            return None, processed_path, None, 0, None
        with mock.patch('aaf_pipeline._execute_aaf_pipeline', side_effect=execute):
            result = aaf_pipeline.run_aaf_pipeline(source)
        self.assertIsNone(result[2], result[2])
        self.assertEqual(result[1], expected)
        self.assertTrue(expected.is_file())

    def test_commit_failure_preserves_previous_output(self):
        output = self.root / 'input_processed.aaf'
        output.write_bytes(b'previous')
        def execute(source, *, processed_path, **kwargs):
            processed_path.write_bytes(b'new')
            return None, processed_path, None, 0, None
        with mock.patch('aaf_pipeline._execute_aaf_pipeline', side_effect=execute), \
             mock.patch.object(Path, 'replace', side_effect=OSError('publish failed')):
            result = aaf_pipeline.run_aaf_pipeline(self.source)
        self.assertIn('publish failed', result[2])
        self.assertEqual(output.read_bytes(), b'previous')

if __name__ == '__main__':
    unittest.main()
