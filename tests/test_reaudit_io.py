"""Ownership and commit-boundary regressions from the repeat audit."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import aaf2
import aaf_workflow
from aaf_io import roundtrip, sdk_tools, temp_cleanup
from aaf_io.errors import OperationCancelled


class ReauditIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.aaf'
        with aaf2.open(str(self.source), 'w') as aaf:
            aaf.content.mobs.append(aaf.create.CompositionMob())
        self.output = self.root / 'output.aaf'
        self.output.write_bytes(b'previous complete output')
        self.previous = self.output.read_bytes()

    def test_cleanup_metacharacters_never_launches_shell(self):
        for dirname in ('R&D', 'two words', '(group)', '%PATH%'):
            with self.subTest(dirname=dirname):
                parent = self.root / dirname
                parent.mkdir()
                xml = parent / 'export.xml'
                xml.write_text('xml')
                streams = parent / 'export_streams'
                streams.mkdir()
                (streams / 'data').write_bytes(b'owned')
                sentinel = self.root / 'R'
                sentinel.mkdir(exist_ok=True)
                with mock.patch('aaf_io.subprocess_hidden.run_hidden') as launched:
                    aaf_workflow.sdk_cleanup_xml_side_artifacts(xml)
                launched.assert_not_called()
                self.assertFalse(xml.exists())
                self.assertFalse(streams.exists())
                self.assertTrue(sentinel.exists())

    def test_cleanup_failure_is_reported_without_shell_fallback(self):
        target = self.root / 'R&D'
        target.mkdir()
        with mock.patch.object(temp_cleanup, 'robust_rmtree', return_value=False), mock.patch('aaf_io.subprocess_hidden.run_hidden') as launched:
            self.assertFalse(temp_cleanup.robust_rmtree_with_windows_fallback(target))
        launched.assert_not_called()

    def test_workflow_cleanup_failure_is_explicit(self):
        xml = self.root / 'export.xml'
        xml.write_text('xml')
        (self.root / 'export_streams').mkdir()
        with mock.patch('aaf_io.temp_cleanup.robust_rmtree', return_value=False):
            with self.assertRaises(OSError):
                aaf_workflow.sdk_cleanup_xml_side_artifacts(xml)

    def test_roundtrip_cleanup_failure_preserves_previous_output(self):
        cleanup = tempfile.TemporaryDirectory.cleanup
        def failed_cleanup(directory):
            cleanup(directory)
            raise PermissionError('workspace cleanup failed')
        def build(source, destination, **kwargs):
            shutil.copyfile(source, destination)
            return roundtrip.RoundtripResult(True, roundtrip.RoundtripMethod.sdk_xml_raw, 'ok')
        with mock.patch.object(tempfile.TemporaryDirectory, 'cleanup', failed_cleanup), mock.patch.object(roundtrip, '_sdk_roundtrip_work', side_effect=build):
            with self.assertRaisesRegex(PermissionError, 'cleanup failed'):
                roundtrip.sdk_roundtrip(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), self.previous)
        self.assertFalse(list(self.root.glob('.aaf-candidate-*')))

    def test_precancelled_sdk_calls_preserve_destinations(self):
        def cancel():
            raise OperationCancelled('cancelled before start')
        for function in (sdk_tools.run_aaffmtconv_to_xml, sdk_tools.run_aaffmtconv_to_structured_storage):
            with self.subTest(function=function.__name__):
                self.output.write_bytes(self.previous)
                with mock.patch.object(sdk_tools, 'popen_hidden') as launched:
                    with self.assertRaises(OperationCancelled):
                        function(self.root / 'tool.exe', self.source, self.output, cancel_check=cancel)
                launched.assert_not_called()
                self.assertEqual(self.output.read_bytes(), self.previous)

    def test_failed_sdk_binary_preserves_output(self):
        def fail(argv, **kwargs):
            Path(argv[-1]).write_bytes(b'partial')
            return 1, 'conversion failed'
        with mock.patch.object(sdk_tools, '_run_subprocess_cancellable', side_effect=fail):
            with self.assertRaises(RuntimeError):
                sdk_tools.run_aaffmtconv_to_structured_storage(self.root / 'tool.exe', self.source, self.output)
        self.assertEqual(self.output.read_bytes(), self.previous)

    def test_xml_rejects_existing_destination_and_sidecars(self):
        for occupied in ('xml', 'streams'):
            output = self.root / (occupied + '.xml')
            owned = output if occupied == 'xml' else output.with_name(output.stem + '_streams')
            if occupied == 'xml':
                owned.write_bytes(b'caller XML')
            else:
                owned.mkdir()
                (owned / 'sentinel').write_bytes(b'caller essence')
            with mock.patch.object(sdk_tools, '_run_subprocess_cancellable') as launched:
                with self.assertRaises(FileExistsError):
                    sdk_tools.run_aaffmtconv_to_xml(self.root / 'tool.exe', self.source, output)
            launched.assert_not_called()
            self.assertTrue(owned.exists())

    def test_failed_fresh_xml_cleans_only_created_artifacts(self):
        output = self.root / 'fresh.xml'
        def fail(argv, **kwargs):
            xml = Path(argv[-1])
            xml.write_bytes(b'partial')
            xml.with_name(xml.stem + '_streams').mkdir(exist_ok=True)
            return 1, 'conversion failed'
        with mock.patch.object(sdk_tools, '_run_subprocess_cancellable', side_effect=fail):
            with self.assertRaises(RuntimeError):
                sdk_tools.run_aaffmtconv_to_xml(self.root / 'tool.exe', self.source, output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.stem + '_streams').exists())

    def test_invalid_sdk_binary_preserves_previous_output(self):
        def invalid(argv, **kwargs):
            Path(argv[-1]).write_bytes(b'tiny')
            return 0, ''
        with mock.patch.object(sdk_tools, '_run_subprocess_cancellable', side_effect=invalid):
            with self.assertRaises(RuntimeError):
                sdk_tools.run_aaffmtconv_to_structured_storage(self.root / 'tool.exe', self.source, self.output)
        self.assertEqual(self.output.read_bytes(), self.previous)

    def test_failed_stream_reservation_preserves_other_export(self):
        output = self.root / 'pair.other'
        streams = self.root / 'pair_streams'
        original_mkdir = Path.mkdir
        def race(directory, *args, **kwargs):
            if directory.resolve() == streams.resolve():
                original_mkdir(directory)
                (directory / 'other-export').write_bytes(b'completed essence')
                raise FileExistsError('another export owns the same stem')
            return original_mkdir(directory, *args, **kwargs)
        with mock.patch.object(Path, 'mkdir', race), mock.patch.object(sdk_tools, '_run_subprocess_cancellable') as launched:
            with self.assertRaises(FileExistsError):
                sdk_tools.run_aaffmtconv_to_xml(self.root/'tool.exe', self.source, output)
        launched.assert_not_called()
        self.assertEqual((streams/'other-export').read_bytes(), b'completed essence')
        self.assertFalse(output.exists())

    def test_xml_reservation_failure_preserves_foreign_xml(self):
        output = self.root / 'pair.xml'
        original_open = Path.open
        def race(file, mode='r', *args, **kwargs):
            if mode == 'xb' and file.resolve() == output.resolve():
                with original_open(file,'wb') as stream:
                    stream.write(b'other export')
                raise FileExistsError('another XML was created')
            return original_open(file,mode,*args,**kwargs)
        with mock.patch.object(Path, 'open', race):
            with self.assertRaises(FileExistsError):
                sdk_tools.run_aaffmtconv_to_xml(self.root/'tool.exe', self.source, output)
        self.assertEqual(output.read_bytes(), b'other export')
        self.assertFalse(output.with_name(output.stem+'_streams').exists())
