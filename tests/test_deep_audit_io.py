from __future__ import annotations

import io
import os
import shutil
import struct
import sys
import time
import subprocess
import tempfile
import threading
import unittest
import wave
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import aaf2
import aaf_workflow
from aaf_io.converter import AAFConverter
from aaf_io.errors import OperationCancelled
from aaf_io import sdk_tools
from aaf_io.session import prepare_work_copy
from aaf_speech_filter.config import FilterConfig
from aaf_speech_filter.exceptions import SpeechFilterCancelled
from aaf_speech_filter.pyaaf2_filter import filter_aaf_speech_only


class DeepAuditIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.aaf'
        self.output = self.root / 'output.aaf'
        with aaf2.open(str(self.source), 'w') as aaf:
            comp = aaf.create.CompositionMob()
            aaf.content.mobs.append(comp)
            slot = comp.create_timeline_slot(48000)
            slot.segment = aaf.create.Filler('sound', 100)
        self.original = self.source.read_bytes()
        self.output.write_bytes(b'previous output')
        self.cfg = FilterConfig(remove_quiet_clips=False, remove_duplicates=False)

    def assert_preserved(self):
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(self.output.read_bytes(), b'previous output')

    def test_converter_rejects_identity_before_open_or_cleanup(self):
        for alias in (self.source, self.root / 'sub' / '..' / 'source.aaf', self.root / 'hardlink.aaf'):
            if alias.name == 'hardlink.aaf':
                os.link(self.source, alias)
            converter = AAFConverter(self.source, work_dir=self.root / 'owned')
            with mock.patch('aaf_io.converter.open_aaf_lenient') as opened:
                with self.assertRaises(RuntimeError):
                    converter.prepare_external_media_copy(alias)
                opened.assert_not_called()
            self.assertEqual(self.source.read_bytes(), self.original)
            self.assertFalse((self.root / 'owned').exists())

    def test_prepare_work_copy_rejects_hardlink(self):
        link = self.root / 'link.aaf'
        os.link(self.source, link)
        with self.assertRaises(RuntimeError):
            prepare_work_copy(self.source, link, sync_cfb_fat_header=False)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_converter_failure_preserves_unowned_essence_and_destination(self):
        essence = self.root / 'essence'
        essence.mkdir()
        sentinel = essence / 'unowned.txt'
        sentinel.write_text('caller data')
        converter = AAFConverter(self.source)
        with mock.patch('aaf_io.converter.open_aaf_lenient', side_effect=ValueError('invalid')):
            self.assertFalse(converter.prepare_external_media_copy(self.output))
        self.assertTrue(sentinel.is_file())
        self.assert_preserved()

    def test_converter_copy_failure_preserves_destination(self):
        converter = AAFConverter(self.source)
        def partial_copy(_source, destination):
            Path(destination).write_bytes(b'partial')
            raise OSError('disk full')
        with mock.patch('aaf_io.converter.shutil.copyfile', side_effect=partial_copy):
            self.assertFalse(converter.prepare_external_media_copy(self.output))
        self.assert_preserved()

    def test_converter_concurrent_extraction_directories_are_exclusive(self):
        first, second = AAFConverter(self.source), AAFConverter(self.source)
        self.assertTrue(first.prepare_external_media_copy(self.root / 'one.aaf'))
        self.assertTrue(second.prepare_external_media_copy(self.root / 'two.aaf'))
        self.assertNotEqual(first.essence_dir, second.essence_dir)
        sentinel = second.essence_dir / 'owned.txt'
        sentinel.write_text('second')
        first._unlink_mapped_essence()
        self.assertTrue(sentinel.is_file())
        second._unlink_mapped_essence()

    def test_filter_invalid_composition_preserves_destination(self):
        with aaf2.open(str(self.source), 'w'):
            pass
        self.original = self.source.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'CompositionMob'):
            filter_aaf_speech_only(self.source, self.output, self.cfg)
        self.assert_preserved()

    def test_filter_copy_failure_preserves_destination(self):
        def partial_copy(_source, destination):
            Path(destination).write_bytes(b'partial')
            raise OSError('disk full')
        with mock.patch('shutil.copyfile', side_effect=partial_copy):
            with self.assertRaisesRegex(OSError, 'disk full'):
                filter_aaf_speech_only(self.source, self.output, self.cfg)
        self.assert_preserved()

    def test_filter_save_failure_preserves_destination(self):
        from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
        @contextmanager
        def failing_save(path, mode):
            with open_aaf_lenient(path, mode) as aaf:
                yield aaf
            if mode == 'r+':
                raise OSError('save failed')
        with mock.patch('aaf_speech_filter.pyaaf2_filter.open_aaf_lenient', side_effect=failing_save):
            with self.assertRaisesRegex(OSError, 'save failed'):
                filter_aaf_speech_only(self.source, self.output, self.cfg)
        self.assert_preserved()

    def test_filter_cancel_before_publication_preserves_destination(self):
        event = threading.Event()
        def progress(done, total):
            if done >= total:
                event.set()
        with self.assertRaises(SpeechFilterCancelled):
            filter_aaf_speech_only(self.source, self.output, self.cfg, progress_callback=progress, cancel_event=event)
        self.assert_preserved()

    def test_filter_failed_replace_preserves_destination(self):
        with mock.patch.object(Path, 'replace', side_effect=OSError('publish failed')):
            with self.assertRaisesRegex(OSError, 'publish failed'):
                filter_aaf_speech_only(self.source, self.output, self.cfg)
        self.assert_preserved()

    def test_filter_failed_validation_preserves_destination(self):
        with mock.patch('aaf_speech_filter.pyaaf2_filter._output_aaf_lightly_readable', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'read'):
                filter_aaf_speech_only(self.source, self.output, self.cfg)
        self.assert_preserved()

    def test_cleanup_cannot_remove_shared_root_between_mkdir_and_allocation(self):
        actual = tempfile.mkdtemp
        def interleave(*args, **kwargs):
            aaf_workflow.cleanup_aaf_tool_work_root(self.root)
            return actual(*args, **kwargs)
        with mock.patch('tempfile.mkdtemp', side_effect=interleave):
            work = aaf_workflow.make_work_dir_near_input(self.source)
        aaf_workflow.cleanup_work_dir(work, input_parent=self.root)
        self.assertTrue(work.parent.is_dir())

    def test_aifc_pcm_representation(self):
        for width in (1, 2, 3, 4):
            for compression in (b'NONE', b'twos', b'sowt'):
                with self.subTest(width=width, compression=compression):
                    values = (0, -(1 << (8 * width - 1)), (1 << (8 * width - 1)) - 1)
                    endian = 'little' if compression == b'sowt' else 'big'
                    raw = b''.join(v.to_bytes(width, endian, signed=True) for v in values)
                    expected = bytes(v + 128 for v in values) if width == 1 else b''.join(v.to_bytes(width, 'little', signed=True) for v in values)
                    self.extract_aifc(raw, width, compression)
                    with wave.open(str(self.root / 'audio.wav'), 'rb') as wav:
                        self.assertEqual(wav.readframes(3), expected)

    def extract_aifc(self, raw, width, compression):
        comm = struct.pack('>hIh', 1, 3, width * 8) + bytes(10) + compression
        summary = b'FORM' + struct.pack('>I', 4 + 8 + len(comm)) + b'AIFC' + b'COMM' + struct.pack('>I', len(comm)) + comm
        class AIFCDescriptor:
            sample_rate = 48000
            channels = 1
            quantization_bits = width * 8
            def __getitem__(self, key):
                if key == 'Summary':
                    return mock.Mock(value=summary)
                raise KeyError(key)
        essence = mock.Mock()
        essence.open.return_value = io.BytesIO(raw)
        return AAFConverter(self.source)._extract_audio(essence, self.root / 'audio.wav', AIFCDescriptor(), None)

    def test_unknown_aifc_compression_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'compression'):
            self.extract_aifc(b'\0' * 6, 2, b'ulaw')
        self.assertFalse((self.root / 'audio.wav').exists())

    def test_comaafinfo_cancellation_reaps_process(self):
        process = mock.Mock(returncode=None)
        process.communicate.side_effect = [subprocess.TimeoutExpired('tool', .25), ('', '')]
        def cancel():
            if process.communicate.call_count:
                raise OperationCancelled()
        with mock.patch('aaf_io.sdk_tools.popen_hidden', return_value=process):
            with self.assertRaises(OperationCancelled):
                sdk_tools.run_comaafinfo(self.root / 'tool.exe', self.source, cancel_check=cancel)
        process.terminate.assert_called_once()
        self.assertGreaterEqual(process.communicate.call_count, 2)

    def test_comaafinfo_timeout_reaps_process(self):
        process = mock.Mock(returncode=None)
        process.communicate.side_effect = [subprocess.TimeoutExpired('tool', .25), ('', '')]
        with mock.patch('aaf_io.sdk_tools.popen_hidden', return_value=process), mock.patch('aaf_io.sdk_tools.time.monotonic', side_effect=[0, 0, 2]):
            with self.assertRaises(subprocess.TimeoutExpired):
                sdk_tools.run_comaafinfo(self.root / 'tool.exe', self.source, timeout_sec=1)
        process.terminate.assert_called_once()

    def test_comaafinfo_workflow_does_not_downgrade_cancellation(self):
        for func, kwargs in ((aaf_workflow.emit_comaafinfo_header, {'input_aaf': self.source}), (aaf_workflow.warn_if_strict_aaf_validation_fails, {'processed_aaf': self.source})):
            with mock.patch('aaf_io.sdk_tools.find_comaafinfo', return_value=self.root / 'tool.exe'), mock.patch('aaf_io.sdk_tools.run_comaafinfo', side_effect=OperationCancelled()):
                with self.assertRaises(OperationCancelled):
                    func(**kwargs, tools_root=self.root, emit=lambda message: None)

    def test_converter_cancel_after_copy_preserves_destination_and_cleans_child(self):
        converter = AAFConverter(self.source)
        event = threading.Event()
        original_copy = shutil.copyfile
        def cancel_after_copy(source, destination):
            result = original_copy(source, destination)
            event.set()
            return result
        with mock.patch('aaf_io.converter.shutil.copyfile', side_effect=cancel_after_copy):
            with self.assertRaises(OperationCancelled):
                converter.prepare_external_media_copy(self.output, cancel_event=event)
        self.assert_preserved()
        self.assertFalse(converter.essence_dir.exists())
        self.assertEqual(list(self.root.glob('.aaf-candidate-*')), [])

    def test_converter_failed_publish_preserves_destination_and_cleans_child(self):
        converter = AAFConverter(self.source)
        with mock.patch.object(Path, 'replace', side_effect=OSError('publish failed')):
            self.assertFalse(converter.prepare_external_media_copy(self.output))
        self.assert_preserved()
        self.assertFalse(converter.essence_dir.exists())
        self.assertIn('publish failed', converter.last_prepare_error)

    def test_extraction_write_failure_is_observable(self):
        with mock.patch('aaf_io.converter._write_wav', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.extract_aifc(bytes(6), 2, b'NONE')

    def test_filter_success_publishes_and_releases_candidate(self):
        self.assertEqual(filter_aaf_speech_only(self.source, self.output, self.cfg), 0)
        with aaf2.open(str(self.output), 'r') as aaf:
            self.assertEqual(len(list(aaf.content.compositionmobs())), 1)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(list(self.root.glob('.aaf-candidate-*')), [])

    def test_public_filter_cli_failure_preserves_destination(self):
        with aaf2.open(str(self.source), 'w'):
            pass
        self.original = self.source.read_bytes()
        self.output = self.source.with_name('source_processed.aaf')
        self.output.write_bytes(b'previous output')
        result = subprocess.run(
            [sys.executable, '-m', 'aaf_speech_filter.cli', str(self.source)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'CompositionMob', result.stderr)
        self.assert_preserved()

    def test_comaafinfo_kills_and_reaps_child_ignoring_termination(self):
        process = mock.Mock(returncode=None)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired('tool', .25),
            subprocess.TimeoutExpired('tool', 1), ('', ''),
        ]
        def cancel():
            if process.communicate.call_count:
                raise OperationCancelled()
        with mock.patch('aaf_io.sdk_tools.popen_hidden', return_value=process):
            with self.assertRaises(OperationCancelled):
                sdk_tools.run_comaafinfo(self.root / 'tool.exe', self.source, cancel_check=cancel)
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.call_count, 3)

    def test_actual_blocking_process_is_reaped_on_cancel_and_deadline(self):
        for mode in ('cancel', 'deadline'):
            with self.subTest(mode=mode):
                children = []
                started = time.monotonic()
                def start_child(*args, **kwargs):
                    child = subprocess.Popen(*args, **kwargs)
                    children.append(child)
                    return child
                def cancel():
                    if time.monotonic() - started > 0.15:
                        raise OperationCancelled()
                with mock.patch('aaf_io.sdk_tools.popen_hidden', side_effect=start_child):
                    with self.assertRaises(OperationCancelled if mode == 'cancel' else subprocess.TimeoutExpired):
                        sdk_tools._run_subprocess_cancellable(
                            [sys.executable, '-c', 'import threading; threading.Event().wait()'],
                            cwd=str(self.root), cancel_check=cancel if mode == 'cancel' else None,
                            timeout_sec=0.2 if mode == 'deadline' else 3,
                        )
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll())
                self.assertLess(time.monotonic() - started, 5)


    def test_complete_aiff_and_aifc_signed_silence_and_fullscale(self):
        import aifc
        from aaf_speech_filter.speech_vad import measure_segment_peak_dbfs
        for container in ('AIFF', 'AIFC'):
            for width in (1, 2, 3, 4):
                for value in (0, -(1 << (width * 8 - 1)), (1 << (width * 8 - 1)) - 1):
                    with self.subTest(container=container, width=width, value=value):
                        source_audio = self.root / 'encoded.aifc'
                        with aifc.open(str(source_audio), 'wb') as stream:
                            if container == 'AIFF':
                                stream.aiff()
                            else:
                                stream.aifc()
                            stream.setnchannels(1)
                            stream.setsampwidth(width)
                            stream.setframerate(48000)
                            stream.writeframes(value.to_bytes(width, 'big', signed=True) * 480)
                        raw = source_audio.read_bytes()
                        essence = mock.Mock()
                        essence.open.return_value = io.BytesIO(raw)
                        with aaf2.open(str(self.root / 'descriptor.aaf'), 'w') as aaf:
                            desc = aaf.create.AIFCDescriptor()
                            desc['Summary'].value = list(raw)
                            desc['SampleRate'].value = 48000
                            desc['Length'].value = 480
                            target = self.root / 'decoded.wav'
                            self.assertTrue(AAFConverter(self.source)._extract_audio(essence, target, desc, aaf))
                        with wave.open(str(target), 'rb') as stream:
                            expected = bytes([value + 128]) if width == 1 else value.to_bytes(width, 'little', signed=True)
                            self.assertEqual(stream.readframes(480), expected * 480)
                        # Existing decoder supports signed AIFF samples at 8/16/24/32 bits.
                        self.assertAlmostEqual(
                            measure_segment_peak_dbfs(source_audio, 0, .01),
                            measure_segment_peak_dbfs(target, 0, .01), places=5,
                        )


    def test_cleanup_refuses_shared_allocation_root(self):
        shared = self.root / '__aaf_tool_work'
        shared.mkdir()
        with self.assertRaises(ValueError):
            aaf_workflow.cleanup_work_dir(shared, input_parent=self.root)
        self.assertTrue(shared.is_dir())

    def test_comaafinfo_rejects_nonfinite_deadline_before_start(self):
        for duration in (float('nan'), float('inf'), -1, 0):
            with self.subTest(duration=duration):
                with mock.patch('aaf_io.sdk_tools.popen_hidden') as started:
                    with self.assertRaises(ValueError):
                        sdk_tools.run_comaafinfo(self.root / 'tool.exe', self.source, timeout_sec=duration)
                    started.assert_not_called()



    def test_successful_publication_has_no_fallible_directory_cleanup(self):
        from aaf_io.output_transaction import output_candidate
        with mock.patch.object(tempfile.TemporaryDirectory, 'cleanup', side_effect=OSError('cleanup failed')):
            with output_candidate(self.source, self.output) as candidate:
                candidate.write_bytes(b'validated new output')
        self.assertEqual(self.output.read_bytes(), b'validated new output')
        self.assertEqual(list(self.root.glob('.aaf-candidate-*')), [])

    def test_conflicting_aifc_descriptor_and_payload_are_rejected(self):
        import aifc
        encoded = []
        for width in (1, 2):
            path = self.root / f'encoded-{width}.aifc'
            with aifc.open(str(path), 'wb') as stream:
                stream.setnchannels(1)
                stream.setsampwidth(width)
                stream.setframerate(48000)
                stream.writeframes(bytes(width * 4))
            encoded.append(path.read_bytes())
        essence = mock.Mock()
        essence.open.return_value = io.BytesIO(encoded[0])
        with aaf2.open(str(self.root / 'descriptor.aaf'), 'w') as aaf:
            desc = aaf.create.AIFCDescriptor()
            desc['Summary'].value = list(encoded[1])
            desc['SampleRate'].value = 48000
            desc['Length'].value = 4
            with self.assertRaisesRegex(ValueError, 'conflict'):
                AAFConverter(self.source)._extract_audio(essence, self.root/'decoded.wav', desc, aaf)
        self.assertFalse((self.root/'decoded.wav').exists())


    def test_raw_pcm_summary_rate_conflict_is_rejected(self):
        import aifc
        path = self.root/'summary.aifc'
        with aifc.open(str(path), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(44100)
            stream.writeframes(bytes(44100 * 2))
        essence = mock.Mock()
        essence.open.return_value = io.BytesIO(bytes(44100 * 2))
        with aaf2.open(str(self.root/'descriptor.aaf'), 'w') as aaf:
            desc = aaf.create.AIFCDescriptor()
            desc['Summary'].value = list(path.read_bytes())
            desc['SampleRate'].value = 48000
            desc['Length'].value = 44100
            with self.assertRaisesRegex(ValueError, 'conflict'):
                AAFConverter(self.source)._extract_audio(essence, self.root/'decoded.wav', desc, aaf)

    def test_child_cleanup_error_preserves_exception_without_add_note(self):
        class LegacyFailure(Exception):
            add_note = None
        process = mock.Mock(returncode=None)
        process.terminate.side_effect = OSError('cannot terminate')
        process.poll.return_value = None
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if started.called:
                raise LegacyFailure('original cancellation')
        with mock.patch('aaf_io.sdk_tools.popen_hidden', return_value=process) as started:
            with self.assertLogs('aaf_io.sdk_tools', level='ERROR'):
                with self.assertRaisesRegex(LegacyFailure, 'original cancellation'):
                    sdk_tools.run_comaafinfo(self.root/'tool', self.source, cancel_check=cancel)


if __name__ == '__main__':
    unittest.main()
