from __future__ import annotations

import struct
from fractions import Fraction
import tempfile
import unittest
import warnings
import wave
from pathlib import Path

import aaf2
from aaf_io.sdk_xml import apply_removals_in_composition_xml
from aaf_speech_filter.config import FilterConfig
from aaf_speech_filter.duplicate_filter import remove_duplicate_timeline_blocks_inplace
from aaf_speech_filter.pyaaf2_filter import filter_aaf_speech_only
from aaf_speech_filter.sdk_removals import collect_timeline_sourceclip_removals_for_sdk_xml
from aaf_speech_filter.speech_vad import (
    _read_aiff_segment_raw_16le, _read_wav_segment_raw_16le, wav_file_info,
)
from aaf_speech_filter.timeline_sourceclips import iter_timeline_sourceclips

ROOT = Path(__file__).resolve().parents[1]


def write_wave(path, samples, width=2):
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, width, 48000, 0, 'NONE', 'not compressed'))
        audio.writeframes(samples)


def add_source(aaf, wav):
    source = aaf.create.SourceMob()
    aaf.content.mobs.append(source)
    source.import_audio_essence(str(wav), offline=True)
    locator = aaf.create.NetworkLocator()
    locator['URLString'].value = wav.resolve().as_uri()
    source.descriptor['Locator'].append(locator)
    return source


def add_comp(aaf):
    comp = aaf.create.CompositionMob()
    aaf.content.mobs.append(comp)
    return comp


def clip(source, length=4800):
    return source.create_source_clip(1, start=0, length=length, media_kind='sound')


def add_gain(aaf, source, value):
    try:
        operation = aaf.dictionary.lookup_operationdef('MonoAudioGain')
    except Exception:
        operation = aaf.create.OperationDef('9d2ea894-0968-11d3-8a38-0050040ef7d2', 'MonoAudioGain')
        operation.media_kind = 'sound'
        operation.number_inputs = 1
        operation['IsTimeWarp'].value = False
        parameter = aaf.create.ParameterDef('e4962321-2267-11d3-8a4c-0050040ef7d2', 'Amplitude', typedef='Rational')
        aaf.dictionary.register_def(parameter)
        operation.parameters.append(parameter)
        aaf.dictionary.register_def(operation)
    group = aaf.create.OperationGroup(operation, length=4800, media_kind='sound')
    group.parameters.append(aaf.create.ConstantValue('Amplitude', value))
    group.segments.append(clip(source))
    return group


class AudioAuditTests(unittest.TestCase):
    def test_pyaaf2_rejects_normalized_input_alias_before_copy(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            original = folder / 'input.aaf'
            original.write_bytes(b'original user bytes')
            (folder / 'nested').mkdir()
            alias = folder / 'nested' / '..' / 'input.aaf'
            with self.assertRaisesRegex(RuntimeError, 'overwrite'):
                filter_aaf_speech_only(original, alias, FilterConfig())
            self.assertEqual(original.read_bytes(), b'original user bytes')

    def test_pyaaf2_rejects_hardlinked_input_before_copy(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            original, alias = folder / 'input.aaf', folder / 'alias.aaf'
            original.write_bytes(b'original user bytes')
            alias.hardlink_to(original)
            with self.assertRaisesRegex(RuntimeError, 'overwrite'):
                filter_aaf_speech_only(original, alias, FilterConfig())
            self.assertEqual(original.read_bytes(), b'original user bytes')

    def test_pyaaf2_reports_removed_duplicates_in_total(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            wav, original, output = folder/'audio.wav', folder/'input.aaf', folder/'out.aaf'
            write_wave(wav, b'\x01\x20' * 4800)
            with aaf2.open(str(original), 'w') as aaf:
                source = add_source(aaf, wav)
                comp = add_comp(aaf)
                for _ in range(2):
                    slot = comp.create_timeline_slot(48000)
                    slot.segment = aaf.create.Sequence(media_kind='sound')
                    slot.segment.components.append(clip(source))
            removed = filter_aaf_speech_only(original, output,
                FilterConfig(remove_quiet_clips=False, remove_duplicates=True))
            with aaf2.open(str(output), 'r') as aaf:
                self.assertEqual(len(list(iter_timeline_sourceclips(aaf))), 1)
            self.assertEqual(removed, 1)

    def test_unsigned_wav8_silence_and_extrema(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            path = Path(td) / 'audio.wav'
            write_wave(path, bytes([128, 0, 255]), width=1)
            raw, channels, rate = _read_wav_segment_raw_16le(path, Fraction(0), Fraction(3, 48000))
            self.assertEqual(struct.unpack('<hhh', raw), (0, -32768, 32512))
            write_wave(path, bytes([128]) * 480, width=1)
            self.assertEqual(wav_file_info(path)[1], -120.0)

    def test_signed_aiff8_is_not_biased(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            import aifc
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            path = Path(td) / 'audio.aiff'
            with aifc.open(str(path), 'wb') as audio:
                audio.setparams((1, 1, 48000, 0, b'NONE', b'not compressed'))
                audio.writeframes(bytes([0, 128, 127]))
            raw, _, _ = _read_aiff_segment_raw_16le(path, Fraction(0), Fraction(3, 48000))
            self.assertEqual(struct.unpack('<hhh', raw), (0, -32768, 32512))

    def test_direct_slot_sourceclip_is_counted_and_removed(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            wav, original, output = folder / 'audio.wav', folder / 'input.aaf', folder / 'output.aaf'
            write_wave(wav, b'\0\0' * 4800)
            with aaf2.open(str(original), 'w') as aaf:
                source = add_source(aaf, wav)
                comp = add_comp(aaf)
                comp.create_timeline_slot(48000).segment = clip(source)
            with aaf2.open(str(original), 'r') as aaf:
                self.assertEqual(len(list(iter_timeline_sourceclips(aaf))), 1)
            self.assertEqual(filter_aaf_speech_only(original, output, FilterConfig()), 1)
            with aaf2.open(str(output), 'r') as aaf:
                segment = next(aaf.content.compositionmobs()).slots[0].segment
                self.assertEqual(type(segment).__name__, 'Filler')
                self.assertEqual(segment.length, 4800)

    def test_muted_occurrence_does_not_remove_audible_reuse_in_pyaaf2(self):
        self._assert_gain_occurrences(use_sdk=False)

    def test_muted_occurrence_does_not_remove_audible_reuse_in_sdk(self):
        self._assert_gain_occurrences(use_sdk=True)

    def _assert_gain_occurrences(self, use_sdk):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            wav, original, output = folder / 'audio.wav', folder / 'input.aaf', folder / 'output.aaf'
            write_wave(wav, struct.pack('<h', 16000) * 4800)
            with aaf2.open(str(original), 'w') as aaf:
                source = add_source(aaf, wav)
                comp = add_comp(aaf)
                sequence = aaf.create.Sequence(media_kind='sound')
                sequence.components.extend([add_gain(aaf, source, 0), add_gain(aaf, source, 1)])
                sequence.length = 9600
                comp.create_timeline_slot(48000).segment = sequence
            if use_sdk:
                from aaf_speech_filter.sdk_removals import sourceclip_sdk_key_tuple
                from xml.etree import ElementTree as ET
                removals = collect_timeline_sourceclip_removals_for_sdk_xml(original, FilterConfig())
                with aaf2.open(str(original), 'r') as aaf:
                    comp = next(aaf.content.compositionmobs())
                    key = sourceclip_sdk_key_tuple(comp.slots[0].segment.components[0].segments[0])
                    clip_xml = (
                        f'<SourceClip><SourcePackageID>{key[0]}</SourcePackageID>'
                        f'<ComponentLength>{key[1]}</ComponentLength>'
                        f'<StartPosition>{key[2]}</StartPosition><SourceTrackID>{key[3]}</SourceTrackID>'
                        '<ComponentDataDefinition>DataDef_Sound</ComponentDataDefinition></SourceClip>'
                    )
                    wrapped = f'<OperationGroup><InputSegments>{clip_xml}</InputSegments></OperationGroup>'
                    xml_text = (
                        f'<Root><CompositionPackage><PackageID>{comp.mob_id}</PackageID><Tracks>'
                        f'<TimelineTrack><TrackID>{comp.slots[0].slot_id}</TrackID><TrackSegment>'
                        f'<Sequence><ComponentObjects>{wrapped}{wrapped}</ComponentObjects></Sequence>'
                        '</TrackSegment></TimelineTrack></Tracks></CompositionPackage></Root>'
                    )
                xml = folder / 'work.xml'
                xml.write_text(xml_text, encoding='utf-8')
                self.assertEqual(apply_removals_in_composition_xml(xml, removals), 1)
                edited = ET.fromstring(xml.read_text(encoding='utf-8'))
                groups = edited.findall('.//OperationGroup')
                self.assertIsNotNone(groups[0].find('./InputSegments/Filler'))
                self.assertIsNotNone(groups[1].find('./InputSegments/SourceClip'))
            else:
                self.assertEqual(filter_aaf_speech_only(original, output, FilterConfig()), 1)
                with aaf2.open(str(output), 'r') as aaf:
                    sequence = next(aaf.content.compositionmobs()).slots[0].segment
                    self.assertEqual(type(sequence.components[0].segments[0]).__name__, 'Filler')
                    self.assertEqual(type(sequence.components[1].segments[0]).__name__, 'SourceClip')
                    self.assertEqual(sequence.length, 9600)

    def test_sdk_addresses_direct_clips_across_compositions_and_slots(self):
        from aaf_io.sdk_xml import RemovalPlanMismatchError
        from dataclasses import replace
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            wav, original = folder / 'audio.wav', folder / 'input.aaf'
            write_wave(wav, b'\0\0' * 4800)
            packages = []
            with aaf2.open(str(original), 'w') as aaf:
                source = add_source(aaf, wav)
                for _ in range(2):
                    comp = add_comp(aaf)
                    tracks = []
                    for _ in range(2):
                        slot = comp.create_timeline_slot(48000)
                        slot.segment = clip(source)
                        tracks.append(
                            f'<TimelineTrack><TrackID>{slot.slot_id}</TrackID><TrackSegment>'
                            f'<SourceClip><SourcePackageID>{source.mob_id}</SourcePackageID>'
                            '<ComponentLength>4800</ComponentLength><StartPosition>0</StartPosition>'
                            '<SourceTrackID>1</SourceTrackID></SourceClip></TrackSegment></TimelineTrack>'
                        )
                    packages.append(f'<CompositionPackage><PackageID>{comp.mob_id}</PackageID>'
                                    + '<Tracks>' + ''.join(tracks) + '</Tracks></CompositionPackage>')
            removals = collect_timeline_sourceclip_removals_for_sdk_xml(original, FilterConfig())
            self.assertEqual(len(removals), 4)
            xml = folder / 'work.xml'
            text = ('<?xml version="1.0"?><!DOCTYPE Root [<!ENTITY label "untouched">]>'
                    '<Root xmlns="http://www.aafassociation.org/aafx/v1.1/20090617">'
                    + ''.join(packages) + '<Label>&label;</Label></Root>')
            xml.write_text(text, encoding='utf-8')
            first = next(iter(removals))
            stale = replace(first, source_key=(first.source_key[0], 999, 0, 1))
            with self.assertRaises(RemovalPlanMismatchError):
                apply_removals_in_composition_xml(xml, (removals - {first}) | {stale})
            self.assertEqual(xml.read_text(encoding='utf-8'), text)
            self.assertEqual(apply_removals_in_composition_xml(xml, removals), 4)
            edited = xml.read_text(encoding='utf-8')
            self.assertEqual(edited.count('<Filler>'), 4)
            self.assertNotIn('<SourceClip>', edited)
            self.assertIn('<Label>&label;</Label>', edited)

    def test_transition_overlap_does_not_create_false_duplicate(self):
        self._assert_duplicate_overlap(first_position=100, expected=0)

    def test_equal_source_position_with_different_transition_is_preserved(self):
        self._assert_duplicate_overlap(first_position=60, expected=0)

    def _assert_duplicate_overlap(self, first_position, expected):
        with aaf2.open() as aaf:
            source = aaf.create.MasterMob()
            aaf.content.mobs.append(source)
            source.create_timeline_slot(48000).segment = aaf.create.Filler(media_kind='sound', length=4800)
            comp = add_comp(aaf)
            for prefix in (first_position, 80):
                sequence = aaf.create.Sequence(media_kind='sound')
                sequence.components.append(aaf.create.Filler(media_kind='sound', length=prefix))
                if prefix == 80:
                    transition = aaf.create.Transition(media_kind='sound', length=20)
                    transition['CutPoint'].value = 10
                    transition['OperationGroup'].value = add_gain(aaf, source, 1)
                    sequence.components.append(transition)
                segment = aaf.create.SourceClip(media_kind='sound', length=10)
                segment.mob_id = source.mob_id
                segment.slot_id = 1
                sequence.components.append(segment)
                comp.create_timeline_slot(48000).segment = sequence
            # The file-open boundary is substituted; the timeline and all components are real AAF objects.
            from unittest.mock import patch
            from contextlib import nullcontext
            with patch('aaf_speech_filter.duplicate_filter.open_aaf_lenient', return_value=nullcontext(aaf)):
                removed = remove_duplicate_timeline_blocks_inplace(Path('unused.aaf'), cfg=FilterConfig(remove_duplicates=True), dry_run=True)
            self.assertEqual(removed, expected)


if __name__ == '__main__':
    unittest.main()
