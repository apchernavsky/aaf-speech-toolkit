from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import aaf2
from tests.test_audit_audio import add_source, add_comp, add_gain, clip, write_wave
from aaf_speech_filter.config import FilterConfig
from aaf_speech_filter.duplicate_filter import remove_duplicate_timeline_blocks_inplace, duplicate_timeline_key
from aaf_speech_filter.aaf_yamnet_lane_layout import _lane_layout_signature, _lane_layout_wrapper_is_rebuildable


def add_pan(f, sequence, value, *, name='Arbitrary display label', auid='9d2ea893-0968-11d3-8a38-0050040ef7d2'):
    try:
        operation = f.dictionary.lookup_operationdef(auid)
    except Exception:
        operation = f.create.OperationDef(auid, name)
        operation.media_kind = 'sound'
        operation.number_inputs = 1
        operation['IsTimeWarp'].value = False
        parameter = f.create.ParameterDef('e4962322-2267-11d3-8a4c-0050040ef7d2', 'Pan', typedef='Rational')
        f.dictionary.register_def(parameter)
        operation.parameters.append(parameter)
        f.dictionary.register_def(operation)
    group = f.create.OperationGroup(operation, length=4800, media_kind='sound')
    group.parameters.append(f.create.ConstantValue('Pan', value))
    group.segments.append(sequence)
    return group


class RenderEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.wav = self.folder / 'media.wav'
        write_wave(self.wav, b'\x00\x40' * 48000)
        self.path = self.folder / 'input.aaf'

    def dedupe(self, *, gains=None, origins=(0, 0), pans=None):
        with aaf2.open(str(self.path), 'w') as f:
            source, comp = add_source(f, self.wav), add_comp(f)
            for i in range(2):
                slot = comp.create_timeline_slot(48000)
                slot.origin = origins[i]
                seq = f.create.Sequence('sound')
                seq.components.append(clip(source) if gains is None else add_gain(f, source, gains[i]))
                slot.segment = seq if pans is None else add_pan(f, seq, pans[i])
        return remove_duplicate_timeline_blocks_inplace(self.path, cfg=FilterConfig(remove_duplicates=True, remove_quiet_clips=False))

    def test_gain_differences_are_preserved_in_both_orders(self):
        for gains in ((0, 1), (1, 0), (0.5, 1), (1, 0.5)):
            with self.subTest(gains=gains):
                self.assertEqual(self.dedupe(gains=gains), 0)

    def test_identical_gain_is_a_duplicate(self):
        self.assertEqual(self.dedupe(gains=(1, 1)), 1)

    def test_distinct_slot_origins_are_preserved(self):
        self.assertEqual(self.dedupe(origins=(0, 48000)), 0)

    def test_track_pan_is_part_of_rendered_equivalence(self):
        self.assertEqual(self.dedupe(pans=(0, 1)), 0)
        self.assertEqual(self.dedupe(pans=(0, 0)), 1)

    def test_edit_rates_are_not_rounded(self):
        args = dict(sourceclip_key=('source', 100, 0, 1), timeline_pos_units=10)
        self.assertNotEqual(duplicate_timeline_key(**args, slot_edit_rate='48000.0001'), duplicate_timeline_key(**args, slot_edit_rate='48000.0002'))

    def test_lane_pan_uses_identity_and_parameter_values(self):
        with aaf2.open(str(self.path), 'w') as f:
            comp = add_comp(f)
            signatures = []
            for value in (0, 1, 0):
                slot = comp.create_timeline_slot(48000)
                seq = f.create.Sequence('sound')
                seq.components.append(f.create.Filler('sound', 4800))
                slot.segment = add_pan(f, seq, value)
                self.assertTrue(_lane_layout_wrapper_is_rebuildable(slot.segment))
                signatures.append(_lane_layout_signature(slot, seq))
            self.assertNotEqual(signatures[0], signatures[1])
            self.assertEqual(signatures[0], signatures[2])

    def test_display_name_does_not_authorize_unknown_operation(self):
        with aaf2.open(str(self.path), 'w') as f:
            seq = f.create.Sequence('sound')
            seq.components.append(f.create.Filler('sound', 4800))
            wrapper = add_pan(f, seq, 0, name='Mono Audio Pan', auid='12345678-1234-1234-1234-123456789012')
            self.assertFalse(_lane_layout_wrapper_is_rebuildable(wrapper))

    def test_sdk_rejects_moves_between_different_pan_parameters(self):
        import xml.etree.ElementTree as ET
        from tests.test_audit_layout import LaneLayoutAuditTests, q, component, xml_document
        from aaf_io.sdk_lane_layout import LaneLayoutEvent
        root = ET.fromstring(xml_document([[component('Filler', 100)], [component('SourceClip', 10), component('Filler', 90)]]))
        for index, track in enumerate(root.findall('.//' + q('TimelineTrack'))):
            segment = track.find(q('TrackSegment'))
            sequence = segment[0]
            segment.remove(sequence)
            group = component('OperationGroup', 100)
            ET.SubElement(group, q('Operation')).text = 'OperationDef_MonoAudioPan'
            parameters = ET.SubElement(group, q('Parameters'))
            constant = ET.SubElement(parameters, q('ConstantValue'))
            ET.SubElement(constant, q('Value')).text = str(index)
            ET.SubElement(group, q('InputSegments')).append(sequence)
            segment.append(group)
        with self.assertRaisesRegex(RuntimeError, 'incompatible'):
            LaneLayoutAuditTests().run_xml(ET.tostring(root, encoding='unicode'), [LaneLayoutEvent(1, 0, 0, 10, 0)])

    def test_identical_track_automation_at_different_local_offsets_is_preserved(self):
        with aaf2.open(str(self.path), 'w') as f:
            source, comp = add_source(f, self.wav), add_comp(f)
            interpolation = f.create.InterpolationDef('5b6c85a4-0ede-11d3-80a9-006008143e6f', 'Linear')
            f.dictionary.register_def(interpolation)
            for offset in (0, 48000):
                slot = comp.create_timeline_slot(48000)
                slot.origin = offset
                seq = f.create.Sequence('sound')
                if offset:
                    seq.components.append(f.create.Filler('sound', offset))
                seq.components.append(clip(source))
                seq.components.append(f.create.Filler('sound', 96000 - offset - 4800))
                group = add_pan(f, seq, 0)
                group.length = 96000
                group.parameters.clear()
                curve = f.create.VaryingValue('Pan', 'Linear')
                curve.add_keyframe(0, 0)
                curve.add_keyframe(1, 1)
                group.parameters.append(curve)
                slot.segment = group
        self.assertEqual(remove_duplicate_timeline_blocks_inplace(self.path, cfg=FilterConfig(remove_duplicates=True), dry_run=True), 0)

    def test_mute_detection_uses_operation_identity_not_label(self):
        from aaf_speech_filter.timeline_sourceclips import _opgroup_gain_multiplier
        with aaf2.open(str(self.path), 'w') as f:
            source = add_source(f, self.wav)
            gain = add_gain(f, source, 0)
            gain.operation.name = 'Renamed operation'
            self.assertEqual(_opgroup_gain_multiplier(gain), 0)
            unknown = f.create.OperationDef('12345678-1234-1234-1234-123456789012', 'MonoAudioGain')
            unknown.media_kind = 'sound'
            unknown.number_inputs = 1
            unknown['IsTimeWarp'].value = False
            f.dictionary.register_def(unknown)
            gain.operation = unknown
            self.assertEqual(_opgroup_gain_multiplier(gain), 1)
