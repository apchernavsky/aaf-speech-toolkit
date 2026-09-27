from __future__ import annotations

import struct
from fractions import Fraction
import tempfile
import unittest
from pathlib import Path

import aaf2
from tests.test_audit_audio import add_source, add_comp, write_wave
from aaf_speech_filter.config import FilterConfig
from aaf_speech_filter.media_resolve import UnsupportedMediaMapping, _resolve_wave_path_for_sourceclip
from aaf_speech_filter.timeline_timing import sourceclip_audio_timing
from aaf_speech_filter.sdk_removals import collect_timeline_sourceclip_removals_for_sdk_xml


class ResolvedMediaWindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.wav = self.folder / 'media.wav'
        write_wave(self.wav, b'\0\0' * 48000 + struct.pack('<h', 16000) * 96000)
        self.path = self.folder / 'input.aaf'

    def make(self, *, depth=1, master_rate=48000, origin=0, sequence=False, crossing=False, cycle=False):
        with aaf2.open(str(self.path), 'w') as f:
            target = add_source(f, self.wav)
            for _ in range(depth):
                master = f.create.MasterMob()
                f.content.mobs.append(master)
                slot = master.create_timeline_slot(master_rate)
                slot.origin = origin
                inner = target.create_source_clip(1, start=master_rate, length=2 * master_rate, media_kind='sound')
                if cycle:
                    inner.mob = master
                if sequence:
                    slot.segment = f.create.Sequence(media_kind='sound')
                    slot.segment.components.append(f.create.Filler('sound', master_rate))
                    slot.segment.components.append(inner)
                else:
                    slot.segment = inner
                target = master
            comp = add_comp(f)
            comp.create_timeline_slot(48000).segment = target.create_source_clip(1, start=(45600 if crossing else 48000) if sequence else 0, length=4800, media_kind='sound')

    def timing(self):
        with aaf2.open(str(self.path), 'r') as f:
            node = next(f.content.compositionmobs()).slots[0].segment
            return sourceclip_audio_timing(f, 48000, node, self.wav)

    def test_nested_offset_keeps_audible_clip(self):
        self.make()
        self.assertEqual(self.timing(), (Fraction(1), Fraction(1, 10)))
        self.assertEqual(collect_timeline_sourceclip_removals_for_sdk_xml(self.path, FilterConfig()), set())

    def test_multiple_reference_levels_compose_offsets(self):
        self.make(depth=2)
        self.assertEqual(self.timing(), (Fraction(2), Fraction(1, 10)))

    def test_reference_rates_and_origin_are_converted_in_seconds(self):
        self.make(master_rate=24000, origin=12000)
        self.assertEqual(self.timing(), (Fraction(3, 2), Fraction(1, 10)))

    def test_sequence_selects_the_referenced_component(self):
        self.make(sequence=True)
        self.assertEqual(self.timing(), (Fraction(1), Fraction(1, 10)))

    def test_discontinuous_window_is_retained(self):
        self.make(sequence=True, crossing=True)
        self.assertEqual(collect_timeline_sourceclip_removals_for_sdk_xml(self.path, FilterConfig()), set())

    def test_reference_cycle_is_bounded_and_retained(self):
        self.make(cycle=True)
        with aaf2.open(str(self.path), 'r') as f:
            node = next(f.content.compositionmobs()).slots[0].segment
            self.assertIsNone(_resolve_wave_path_for_sourceclip(f, node))

    def test_equal_fractional_duration_fits_exactly(self):
        with aaf2.open(str(self.path), 'w') as f:
            source = add_source(f, self.wav)
            master = f.create.MasterMob()
            f.content.mobs.append(master)
            master.create_timeline_slot(48000).segment = source.create_source_clip(1, start=0, length=40000, media_kind='sound')
            comp = add_comp(f)
            comp.create_timeline_slot(48000).segment = master.create_source_clip(1, start=0, length=40000, media_kind='sound')
        self.assertEqual(self.timing(), (Fraction(0), Fraction(40000, 48000)))
        self.assertEqual(len(collect_timeline_sourceclip_removals_for_sdk_xml(self.path, FilterConfig())), 1)

    def test_preflight_uses_the_owning_slot_rate(self):
        from aaf_speech_filter.timeline_sourceclips import count_resolvable_timeline_wavs
        with aaf2.open(str(self.path), 'w') as f:
            source = add_source(f, self.wav)
            master = f.create.MasterMob()
            f.content.mobs.append(master)
            master.create_timeline_slot(24000).segment = source.create_source_clip(1, start=0, length=24000, media_kind='sound')
            comp = add_comp(f)
            comp.create_timeline_slot(48000).segment = master.create_source_clip(1, start=36000, length=4800, media_kind='sound')
        self.assertEqual(count_resolvable_timeline_wavs(self.path), 1)
        self.assertEqual(len(collect_timeline_sourceclip_removals_for_sdk_xml(self.path, FilterConfig())), 1)

    def test_fps_like_slot_discovers_media_but_retains_unproven_sample_units(self):
        from aaf_speech_filter.timeline_sourceclips import count_resolvable_timeline_wavs
        with aaf2.open(str(self.path), 'w') as f:
            source = add_source(f, self.wav)
            master = f.create.MasterMob()
            f.content.mobs.append(master)
            master.create_timeline_slot(48000).segment = source.create_source_clip(1, start=0, length=144000, media_kind='sound')
            comp = add_comp(f)
            node = master.create_source_clip(1, start=48000, length=48000, media_kind='sound')
            comp.create_timeline_slot(25).segment = node
            self.assertEqual(_resolve_wave_path_for_sourceclip(f, node, edit_rate=25), self.wav)
            with self.assertRaisesRegex(UnsupportedMediaMapping, 'does not prove sample units'):
                sourceclip_audio_timing(f, 25, node, self.wav)
        self.assertEqual(count_resolvable_timeline_wavs(self.path), 1)
        self.assertEqual(collect_timeline_sourceclip_removals_for_sdk_xml(self.path, FilterConfig()), set())

    def test_rational_edit_rate_survives_traversal(self):
        from aaf_speech_filter.timeline_sourceclips import iter_timeline_sourceclip_occurrences
        with aaf2.open(str(self.path), 'w') as f:
            source = add_source(f, self.wav)
            master = f.create.MasterMob()
            f.content.mobs.append(master)
            master.create_timeline_slot('48000000/1001').segment = source.create_source_clip(1, start=0, length=48000, media_kind='sound')
            comp = add_comp(f)
            comp.create_timeline_slot('48000000/1001').segment = master.create_source_clip(1, start=0, length=48000, media_kind='sound')
            occurrence = next(iter_timeline_sourceclip_occurrences(f))
            self.assertEqual(sourceclip_audio_timing(f, occurrence.edit_rate, occurrence.sourceclip, self.wav), (Fraction(0), Fraction(1001, 1000)))

    def test_layout_without_media_preserves_aaf_and_reports_skip(self):
        from aaf_speech_filter.aaf_yamnet_lane_layout import apply_experimental_yamnet_lane_layout
        with aaf2.open(str(self.path), 'w') as f:
            comp = add_comp(f)
            for _ in range(3):
                slot = comp.create_timeline_slot(48000)
                slot.segment = f.create.Sequence('sound')
                slot.segment.components.append(f.create.SourceClip(media_kind='sound', length=4800))
        original = self.path.read_bytes()
        log, report = [], {}
        moved = apply_experimental_yamnet_lane_layout(self.path, FilterConfig(experimental_yamnet_lane_layout=True),
            work_dir=self.folder/'owned', log_callback=log.append, result_out=report)
        self.assertEqual(moved, 0)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(report.get('analysis_skipped'), 'media_unavailable')
        self.assertTrue(any('media' in message.lower() for message in log))
