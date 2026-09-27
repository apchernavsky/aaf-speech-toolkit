from __future__ import annotations

import io
import tempfile
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import aaf2
import aaf_pipeline
from aaf_io.heal.premiere import heal_import_descriptors_from_linked_media


class PipelineLogTests(unittest.TestCase):
    def run_logged_pipeline(self, emit_messages, *, failure=None, callback=None):
        def execute(source, *, processed_path, log_callback, **options):
            emit_messages(log_callback)
            if isinstance(failure, Exception):
                raise failure
            if failure:
                return None, None, failure, None, None
            processed_path.write_bytes(b'result')
            return None, processed_path, None, 0, None

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            source = Path(temp) / 'input.aaf'
            source.write_bytes(b'input')
            with mock.patch.object(aaf_pipeline, '_execute_aaf_pipeline', side_effect=execute):
                return aaf_pipeline.run_aaf_pipeline(source, log_callback=callback)

    def messages(self, emit, count):
        from aaf_io.diagnostic_log import DiagnosticMessage
        for index in range(count):
            emit(DiagnosticMessage(f'clip {index}: unresolved window', category='media-window'))

    def test_limit_is_per_category_and_keeps_progress_and_fatal_error(self):
        from aaf_io.diagnostic_log import DiagnosticMessage
        lines = []
        def send(emit):
            self.messages(emit, 15)
            emit(DiagnosticMessage('different warning', category='decode'))
            emit('Final stage')
        result = self.run_logged_pipeline(send, failure='Fatal save error', callback=lines.append)
        self.assertEqual(sum(line.startswith('clip ') for line in lines), 10)
        self.assertIn('different warning', lines)
        self.assertIn('Final stage', lines)
        self.assertEqual(result[2], 'Fatal save error')
        self.assertTrue(any('5' in line and 'скрыто' in line for line in lines), lines)
        self.assertEqual(sum('дальнейшие' in line for line in lines), 1)

    def test_exact_limit_does_not_claim_truncation(self):
        lines = []
        self.run_logged_pipeline(lambda emit: self.messages(emit, 10), callback=lines.append)
        self.assertEqual(len(lines), 10)

    def test_new_run_gets_new_limit(self):
        for _ in range(2):
            lines = []
            self.run_logged_pipeline(lambda emit: self.messages(emit, 11), callback=lines.append)
            self.assertEqual(sum(line.startswith('clip ') for line in lines), 10)

    def test_exception_and_cancellation_flush_counts(self):
        for failure in (RuntimeError('failed'), aaf_pipeline.OperationCancelled()):
            with self.subTest(failure=type(failure).__name__):
                lines = []
                result = self.run_logged_pipeline(lambda emit: self.messages(emit, 12),
                                                  failure=failure, callback=lines.append)
                self.assertIsNotNone(result[2])
                self.assertTrue(any('2' in line and 'скрыто' in line for line in lines), lines)

    def test_cli_stdout_uses_same_limit(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.run_logged_pipeline(lambda emit: self.messages(emit, 13))
        self.assertEqual(sum(line.startswith('clip ') for line in output.getvalue().splitlines()), 10)
        self.assertIn('скрыто 3', output.getvalue())

    def test_concurrent_diagnostics_have_exact_counts(self):
        lines = []
        def send(emit):
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: self.messages(emit, 25), range(4)))
        self.run_logged_pipeline(send, callback=lines.append)
        self.assertEqual(sum(line.startswith('clip ') for line in lines), 10)
        self.assertTrue(any('скрыто 90' in line for line in lines), lines)


class HealLogTests(unittest.TestCase):
    def heal(self, *, successes=0, missing=0):
        lines = []
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            folder = Path(temp)
            wav = folder / 'audio.wav'
            with wave.open(str(wav), 'wb') as out:
                out.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
                out.writeframes(b'\0\0' * 48)
            with aaf2.open(str(folder / 'input.aaf'), 'w') as aaf:
                for index in range(successes + missing):
                    mob = aaf.create.SourceMob()
                    mob.name = f'Audio {index}'
                    desc = aaf.create.ImportDescriptor()
                    if index < successes:
                        locator = aaf.create.NetworkLocator()
                        locator['URLString'].value = wav.as_uri()
                        desc['Locator'].append(locator)
                    mob.descriptor = desc
                    aaf.content.mobs.append(mob)
                stats = heal_import_descriptors_from_linked_media(aaf, log_callback=lines.append)
        return stats, lines

    def test_successes_do_not_spam_or_consume_warning_budget(self):
        stats, lines = self.heal(successes=12, missing=1)
        self.assertEqual(stats.pcm_replaced, 12)
        self.assertFalse(any('[heal] OK' in line for line in lines), lines)
        self.assertEqual(sum('нет URL' in line for line in lines), 1)

    def test_heal_warnings_cap_and_report_actual_suppressed_count(self):
        stats, lines = self.heal(missing=14)
        self.assertEqual(stats.skipped_missing_media, 14)
        self.assertEqual(sum('нет URL' in line for line in lines), 10)
        self.assertTrue(any('скрыто 4' in line for line in lines), lines)

    def test_heal_exact_limit_does_not_claim_truncation(self):
        _, lines = self.heal(missing=10)
        self.assertEqual(len(lines), 10)


class HealSummaryTests(unittest.TestCase):
    def test_summary_does_not_repeat_already_emitted_error_details(self):
        from aaf_io.heal.premiere import HealStats, format_heal_stats_lines
        stats = HealStats(errors=[f'detail {index}' for index in range(20)])
        lines = format_heal_stats_lines(stats, include_error_samples=False)
        self.assertFalse(any('detail ' in line for line in lines), lines)
        self.assertTrue(any('20' in line for line in lines), lines)

    def test_standalone_summary_bounds_error_samples(self):
        from aaf_io.heal.premiere import HealStats, format_heal_stats_lines
        lines = format_heal_stats_lines(HealStats(errors=[f'detail {i}' for i in range(20)]))
        self.assertEqual(sum('detail ' in line for line in lines), 10)


class RealDiagnosticSourceTests(unittest.TestCase):
    def test_real_unresolved_windows_are_bounded_and_retained(self):
        from aaf_io.diagnostic_log import BoundedDiagnosticLog
        from aaf_speech_filter.config import FilterConfig
        from aaf_speech_filter.sdk_removals import collect_timeline_sourceclip_removals_for_sdk_xml
        from aaf_speech_filter.aaf_yamnet_lane_layout import _classify_timeline_block_with_scores
        from aaf_speech_filter.speech_yamnet import YamnetConfig
        from tests.test_audit_audio import add_comp, add_source

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            folder = Path(temp)
            media = folder / 'media.wav'
            source_path = folder / 'input.aaf'
            with wave.open(str(media), 'wb') as stream:
                stream.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
                stream.writeframes(bytes(4800 * 2))
            with aaf2.open(str(source_path), 'w') as aaf:
                source = add_source(aaf, media)
                comp = add_comp(aaf)
                sequence = aaf.create.Sequence(media_kind='sound')
                for _ in range(13):
                    sequence.components.append(source.create_source_clip(1, start=0, length=25, media_kind='sound'))
                comp.create_timeline_slot(25).segment = sequence
            original = source_path.read_bytes()
            lines = []
            with BoundedDiagnosticLog(lines.append) as emit:
                keys = collect_timeline_sourceclip_removals_for_sdk_xml(source_path, FilterConfig(), log_callback=emit)
                self.assertEqual(len(keys), 0)
                with aaf2.open(str(source_path), 'r') as aaf:
                    sequence = next(aaf.content.compositionmobs()).slots[0].segment
                    for node in sequence.components:
                        result = _classify_timeline_block_with_scores(
                            aaf, node, None, YamnetConfig(), 25, lambda: None, log_callback=emit)
                        self.assertEqual(result, ('unknown', 0.0, 0.0, 0.0))
            self.assertEqual(sum(line.startswith('Media window unresolved; clip retained:') for line in lines), 10)
            self.assertEqual(sum(line.startswith('Media window unresolved; classification skipped:') for line in lines), 10)
            self.assertEqual(sum('скрыто 3' in line for line in lines), 2)
            self.assertEqual(source_path.read_bytes(), original)

    def test_incomplete_audio_analysis_is_bounded(self):
        from aaf_io.diagnostic_log import BoundedDiagnosticLog
        from aaf_speech_filter.speech_vad import is_quiet_clip
        lines = []
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            missing = Path(temp) / 'unavailable.wav'
            with BoundedDiagnosticLog(lines.append) as emit:
                for _ in range(14):
                    self.assertFalse(is_quiet_clip(missing, 0, 0.01, -40, log_callback=emit))
        self.assertEqual(sum(line.startswith('Audio analysis incomplete;') for line in lines), 10)
        self.assertTrue(any('скрыто 4' in line for line in lines), lines)
