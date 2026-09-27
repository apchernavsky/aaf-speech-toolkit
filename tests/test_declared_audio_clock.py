from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import aaf2
from tests.test_audit_audio import add_comp, add_source, write_wave
from aaf_speech_filter.timeline_timing import sourceclip_audio_timing
from aaf_speech_filter.media_resolve import UnsupportedMediaMapping
from aaf_speech_filter import aaf_yamnet_lane_layout as layout
from aaf_speech_filter.config import FilterConfig
from aaf_speech_filter.speech_yamnet import YamnetConfig
from aaf_speech_filter.sdk_removals import collect_timeline_sourceclip_removals_for_sdk_xml

ROOT = Path(__file__).resolve().parents[1]


class DeclaredClockTests(unittest.TestCase):
    def test_frame_clock_is_authoritative_even_when_sample_interpretation_also_fits(self):
        for rate in (24, 25):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory(dir=ROOT) as temp:
                folder = Path(temp)
                media, source_path = folder/'media.wav', folder/'input.aaf'
                write_wave(media, bytes(48000*2) + b'\x00\x40'*48000)
                with aaf2.open(str(source_path), 'w') as aaf:
                    source = add_source(aaf, media)
                    comp = add_comp(aaf)
                    slot = comp.create_timeline_slot(rate)
                    slot.segment = source.create_source_clip(1, start=rate, length=rate, media_kind='sound')
                    result = sourceclip_audio_timing(aaf, slot.edit_rate, slot.segment, media)
                    self.assertEqual(result, (Fraction(1), Fraction(1)))

    def test_frame_clock_reaches_classifier_with_correct_window(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp)
            media, source_path = folder/'media.wav', folder/'input.aaf'
            write_wave(media, b'\x00\x40' * (48000*3))
            with aaf2.open(str(source_path), 'w') as aaf:
                source = add_source(aaf, media)
                comp = add_comp(aaf)
                slot = comp.create_timeline_slot(24)
                slot.segment = source.create_source_clip(1, start=24, length=24, media_kind='sound')
                with patch.object(layout, 'yamnet_clip_kind_with_scores', return_value=('speech', .9, .01, .02)) as classifier:
                    result = layout._classify_timeline_block_with_scores(aaf,slot.segment,None,YamnetConfig(),24,lambda:None)
                self.assertEqual(result[0], 'speech')
                self.assertEqual(classifier.call_args.args[1:3], (Fraction(1),Fraction(1)))

    def test_missing_declared_media_does_not_fall_back_to_sample_units(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp)
            media, source_path = folder/'media.wav', folder/'input.aaf'
            write_wave(media, bytes(48000*2))
            with aaf2.open(str(source_path), 'w') as aaf:
                source = add_source(aaf, media)
                slot = add_comp(aaf).create_timeline_slot(24)
                slot.segment = source.create_source_clip(1, start=0,length=48000,media_kind='sound')
                with self.assertRaises(UnsupportedMediaMapping):
                    sourceclip_audio_timing(aaf,24,slot.segment,media)


class UnclassifiedPlacementTests(unittest.TestCase):
    def event(self, lane, kind, start=0):
        return dict(src_lane=lane,top_idx=0,T=start,T_edit=start,L=100,source_start=0,source_length=100,kind=kind,er=48000)

    def test_unknown_keeps_source_lane_while_recognized_speech_moves_up(self):
        unknown = self.event(8,'unknown')
        speech = self.event(7,'speech')
        layout._assign_target_lanes([unknown,speech],9)
        self.assertEqual(unknown['target_lane'],8)
        self.assertEqual(unknown['lane_bounds'],(8,8))
        self.assertEqual(speech['target_lane'],0)

    def test_unknown_occupies_only_its_interval_not_the_entire_lane(self):
        unknown = self.event(0,'unknown')
        overlapping = self.event(7,'speech')
        later = self.event(8,'speech',start=200)
        layout._assign_target_lanes([unknown,overlapping,later],9)
        self.assertEqual(unknown['target_lane'],0)
        self.assertEqual(overlapping['target_lane'],1)
        self.assertEqual(later['target_lane'],0)
        layout._validate_lane_layout_plan([unknown,overlapping,later],9)


class DeclaredClockOracleTests(unittest.TestCase):
    def test_oracle_measures_declared_frame_window_not_unrelated_samples(self):
        from tests.test_reaudit_oracle import RemovalOracleTests
        helper = RemovalOracleTests()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp)
            source, media = helper.fixture(folder, b'\x00\x40'*48000 + bytes(48000*2), [0])
            with aaf2.open(str(source),'r+') as aaf:
                slot = next(aaf.content.compositionmobs()).slots[0]
                slot.edit_rate = 24
                slot.segment.components[0].start = 24
                slot.segment.components[0].length = 24
            output = helper.removed_copy(source,1)
            result = helper.validate(source,output)
            self.assertEqual(result['removed'],1)
            self.assertEqual(result['proofs'][0]['windows'][0]['first_frame'],48000)
