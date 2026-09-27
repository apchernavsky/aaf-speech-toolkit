from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import aaf2
from tests.test_render_equivalence import add_pan
from tests.test_audit_audio import add_comp
from aaf_speech_filter.aaf_yamnet_lane_layout import _lane_layout_signature

ROOT=Path(__file__).resolve().parents[1]


class TrackOrderTests(unittest.TestCase):
    def test_empty_tracks_leave_noise_centered_and_music_at_bottom(self):
        from aaf_io.lane_order import class_ordered_lanes
        order=class_ordered_lanes([set(),{'speech'},set(),set(),set(),set(),{'noise'},set(),{'music'}])
        self.assertEqual(order[0],1)
        self.assertEqual(order[4],6)
        self.assertEqual(order[-1],8)
        self.assertEqual(sorted(order),list(range(9)))

    def test_multichannel_order_and_fixed_mixed_tracks_are_preserved(self):
        from aaf_io.lane_order import class_ordered_lanes
        kinds=[{'speech'},set(),{'speech','noise'},set(),{'noise','unknown'},{'noise'},{'music'},{'music'},set()]
        order=class_ordered_lanes(kinds)
        self.assertEqual(order[2],2)
        self.assertLess(order.index(4),order.index(5))
        self.assertEqual(order[-2:],[6,7])

    def test_unknown_only_and_protected_tracks_are_not_reordered(self):
        from aaf_io.lane_order import class_ordered_lanes
        order=class_ordered_lanes([{'unknown'},set(),{'music'}, {'speech'}],protected={2})
        self.assertEqual(order[0],0)
        self.assertEqual(order[2],2)

    def test_pyaaf2_reorder_keeps_slot_identity_pan_and_non_audio_slot(self):
        from aaf_io.lane_order import reorder_pyaaf2_slots
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            file=Path(temp)/'tracks.aaf'
            with aaf2.open(str(file),'w') as aaf:
                comp=add_comp(aaf)
                picture=comp.create_timeline_slot(24)
                picture.segment=aaf.create.Filler('picture',240)
                slots=[]
                for pan in (0,1):
                    slot=comp.create_timeline_slot(24)
                    slot['PhysicalTrackNumber'].value=len(slots)+1
                    seq=aaf.create.Sequence('sound')
                    seq.components.append(aaf.create.Filler('sound',240))
                    slot.segment=add_pan(aaf,seq,pan)
                    slots.append(slot)
                identities=[s.slot_id for s in slots]
                reorder_pyaaf2_slots(comp,slots,[1,0])
            with aaf2.open(str(file),'r') as aaf:
                comp=next(aaf.content.compositionmobs())
                self.assertEqual(comp.slots[0].segment.media_kind,'Picture')
                self.assertEqual([s.slot_id for s in list(comp.slots)[1:]],identities[::-1])
                self.assertEqual([s['PhysicalTrackNumber'].value for s in list(comp.slots)[1:]],[1,2])
                self.assertEqual([list(s.segment.parameters)[0].value for s in list(comp.slots)[1:]],[1,0])

    def test_xml_reorder_preserves_whole_track_payload_and_ids(self):
        from aaf_io.lane_order import reorder_xml_tracks
        tracks=ET.fromstring('<Tracks><Video/><Audio><Id>7</Id><Pan>0</Pan></Audio><Audio><Id>9</Id><Pan>1</Pan></Audio></Tracks>')
        audio=list(tracks)[1:]
        before=[ET.tostring(t) for t in audio]
        reorder_xml_tracks(tracks,audio,[1,0])
        self.assertEqual(tracks[0].tag,'Video')
        self.assertEqual([ET.tostring(t) for t in list(tracks)[1:]],before[::-1])

    def test_constant_track_pan_equivalence_is_independent_of_track_duration(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp,aaf2.open(str(Path(temp)/'tracks.aaf'),'w') as aaf:
            comp=add_comp(aaf)
            signatures=[]
            for duration in (240,480):
                slot=comp.create_timeline_slot(24)
                seq=aaf.create.Sequence('sound')
                seq.components.append(aaf.create.Filler('sound',duration))
                slot.segment=add_pan(aaf,seq,0)
                slot.segment.length=duration
                signatures.append(_lane_layout_signature(slot,seq))
            self.assertEqual(signatures[0],signatures[1])

    def test_pyaaf2_writer_applies_explicit_track_order_without_changing_pan(self):
        from aaf_speech_filter.aaf_yamnet_lane_layout import _apply_lane_layout_via_pyaaf2_rebuild
        from aaf_speech_filter.config import FilterConfig
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path=Path(temp)/'input.aaf'
            with aaf2.open(str(path),'w') as aaf:
                comp=add_comp(aaf)
                for pan in (0,1):
                    slot=comp.create_timeline_slot(24)
                    seq=aaf.create.Sequence('sound')
                    seq.components.append(aaf.create.SourceClip(media_kind='sound',length=24))
                    slot.segment=add_pan(aaf,seq,pan)
                    slot.segment.length=24
                ids=[slot.slot_id for slot in comp.slots]
            _apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path,
                events=[dict(src_lane=i,top_idx=0,T=0,T_edit=0,L=24,target_lane=i,kind=('music','speech')[i]) for i in range(2)],
                structural_events=[],n_lanes=2,cfg=FilterConfig(),runtime_essence_paths=None,
                work_dir=Path(temp),cancel_check=lambda:None,progress_callback=None,log_callback=None,order_tracks_by_class=True)
            with aaf2.open(str(path),'r') as aaf:
                slots=list(next(aaf.content.compositionmobs()).slots)
                self.assertEqual([slot.slot_id for slot in slots],ids[::-1])
                self.assertEqual([list(slot.segment.parameters)[0].value for slot in slots],[1,0])
                self.assertEqual([slot.segment.segments[0].components[0].length for slot in slots],[24,24])

    def test_sdk_writer_applies_same_explicit_track_order(self):
        from tests.test_audit_layout import LaneLayoutAuditTests, component, xml_document, q
        from aaf_io.sdk_lane_layout import LaneLayoutEvent
        xml=xml_document([[component('SourceClip',100,'left')],[component('SourceClip',100,'right')]])
        written=LaneLayoutAuditTests().run_xml(xml,[LaneLayoutEvent(i,0,0,100,i,class_kind=('music','speech')[i]) for i in range(2)],order_tracks_by_class=True)
        self.assertEqual([node.text for node in written.findall('.//'+q('Payload'))],['right','left'])

    def test_pan_automation_duration_remains_part_of_compatibility(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp,aaf2.open(str(Path(temp)/'tracks.aaf'),'w') as aaf:
            comp=add_comp(aaf)
            aaf.dictionary.register_def(aaf.create.InterpolationDef('5b6c85a4-0ede-11d3-80a9-006008143e6f','Linear'))
            signatures=[]
            for duration in (240,480):
                slot=comp.create_timeline_slot(24)
                seq=aaf.create.Sequence('sound')
                seq.components.append(aaf.create.Filler('sound',duration))
                slot.segment=add_pan(aaf,seq,0)
                slot.segment.length=duration
                slot.segment.parameters.clear()
                curve=aaf.create.VaryingValue('Pan','Linear')
                curve.add_keyframe(0,0)
                curve.add_keyframe(1,1)
                slot.segment.parameters.append(curve)
                signatures.append(_lane_layout_signature(slot,seq))
            self.assertNotEqual(signatures[0],signatures[1])

    def test_whole_track_order_cannot_cross_a_fixed_channel_peer(self):
        from aaf_io.lane_order import class_ordered_lanes
        kinds=[set(),set(),set(),set(),{'noise'},{'noise'},set()]
        order=class_ordered_lanes(kinds,protected={4})
        self.assertEqual(order[4],4)
        self.assertGreater(order.index(5),order.index(4))

    def test_sdk_moves_between_constant_pan_tracks_with_different_durations(self):
        from tests.test_audit_layout import LaneLayoutAuditTests, component, xml_document, q
        from aaf_io.sdk_lane_layout import LaneLayoutEvent
        root=ET.fromstring(xml_document([[component('Filler',100)],
            [component('SourceClip',10,'retained audio'),component('Filler',190)]]))
        for duration,track in zip((100,200),root.findall('.//'+q('TimelineTrack'))):
            segment=track.find(q('TrackSegment'))
            sequence=segment[0]
            sequence.find(q('ComponentLength')).text=str(duration)
            segment.remove(sequence)
            group=component('OperationGroup',duration)
            ET.SubElement(group,q('Operation')).text='OperationDef_MonoAudioPan'
            parameters=ET.SubElement(group,q('Parameters'))
            constant=ET.SubElement(parameters,q('ConstantValue'))
            ET.SubElement(constant,q('Value')).text='0'
            ET.SubElement(group,q('InputSegments')).append(sequence)
            segment.append(group)
        written=LaneLayoutAuditTests().run_xml(ET.tostring(root,encoding='unicode'),[LaneLayoutEvent(1,0,0,10,0)])
        tracks=written.findall('.//'+q('TimelineTrack'))
        self.assertEqual(tracks[0].find('.//'+q('Payload')).text,'retained audio')
        self.assertIsNone(tracks[1].find('.//'+q('Payload')))
        self.assertEqual([t.find('.//'+q('ConstantValue')+'/'+q('Value')).text for t in tracks],['0','0'])
