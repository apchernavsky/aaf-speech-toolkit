import unittest
import xml.etree.ElementTree as ET
from tests import test_audit_layout as fixtures
from tests.test_audit_layout import component, xml_document, q
from aaf_io.sdk_lane_layout import LaneLayoutEvent
from aaf_speech_filter import aaf_yamnet_lane_layout as layout
from tests.test_speech_yamnet import SourceClip


class OrderedLaneAssignmentTests(unittest.TestCase):
    def test_sdk_restores_channel_order_using_compatible_free_tracks(self):
        root=ET.fromstring(xml_document([[component('SourceClip',20,'left')],
            [component('SourceClip',20,'right')],[component('Filler',100)],[component('Filler',100)]]))
        for index,track in enumerate(root.findall('.//'+q('TimelineTrack'))):
            segment=track.find(q('TrackSegment'))
            sequence=segment[0]
            segment.remove(sequence)
            group=component('OperationGroup',100)
            ET.SubElement(group,q('Operation')).text='OperationDef_MonoAudioPan'
            parameter=ET.SubElement(ET.SubElement(group,q('Parameters')),q('ConstantValue'))
            ET.SubElement(parameter,q('Value')).text=str(index%2)
            ET.SubElement(group,q('InputSegments')).append(sequence)
            segment.append(group)
        events=[LaneLayoutEvent(i,0,0,20,target,visible_t=0,class_kind='noise',source_start=5,source_length=20)
                for i,target in enumerate((2,1))]
        written=fixtures.LaneLayoutAuditTests().run_xml(ET.tostring(root,encoding='unicode'),events,n_lanes=4)
        found=[(i,track.findtext('.//'+q('Payload')),track.findtext('.//'+q('Value')))
               for i,track in enumerate(written.findall('.//'+q('TimelineTrack'))) if track.find('.//'+q('Payload')) is not None]
        self.assertEqual([item[1:] for item in found],[('left','0'),('right','1')])
        self.assertLess(found[0][0],found[1][0])

    def test_pyaaf2_restores_channel_order_using_compatible_free_tracks(self):
        left,right=SourceClip(20),SourceClip(20)
        rebuilt={0:[],1:[(0,20,[right],'event',0,'noise',0)],
                 2:[(0,20,[left],'event',0,'noise',0)],3:[]}
        layout._restore_rebuilt_aligned_source_order(rebuilt,{id(left):0,id(right):1},{},set(),
            lane_signatures=[('left',),('right',),('left',),('right',)])
        found=[(lane,layout._rebuilt_event_primary_node(part),part[4])
               for lane,parts in sorted(rebuilt.items()) for part in parts]
        self.assertEqual([item[1] for item in found],[left,right])
        self.assertEqual([item[2] for item in found],[0,0])
        self.assertEqual([item[0]%2 for item in found],[0,1])

    def test_assignment_looks_ahead_instead_of_consuming_only_feasible_lane(self):
        from aaf_io.lane_assignment import ordered_lane_assignment
        self.assertEqual(ordered_lane_assignment([[0,2],[1]],[2,1]),(0,1))

    def test_infeasible_assignment_is_explicit(self):
        from aaf_io.lane_assignment import ordered_lane_assignment
        self.assertIsNone(ordered_lane_assignment([[2],[1]],[2,1]))

    def test_sdk_orders_tracks_after_channel_repair(self):
        lanes=[[component('SourceClip',20,'left')],[component('SourceClip',20,'right')],
               [component('Filler',100)],[component('Filler',100)],
               [component('SourceClip',20,'blocker')],[component('Filler',100)]]
        root=ET.fromstring(xml_document(lanes))
        for index,track in enumerate(root.findall('.//'+q('TimelineTrack'))):
            segment=track.find(q('TrackSegment'))
            sequence=segment[0]
            segment.remove(sequence)
            group=component('OperationGroup',100)
            ET.SubElement(group,q('Operation')).text='OperationDef_MonoAudioPan'
            parameter=ET.SubElement(ET.SubElement(group,q('Parameters')),q('ConstantValue'))
            ET.SubElement(parameter,q('Value')).text=str(index%2)
            ET.SubElement(group,q('InputSegments')).append(sequence)
            segment.append(group)
        events=[LaneLayoutEvent(i,0,0,20,target,visible_t=0,class_kind='noise',source_start=5,source_length=20)
                for i,target in enumerate((2,1))]
        events.append(LaneLayoutEvent(4,0,0,20,0,visible_t=0,class_kind='speech',source_start=0,source_length=20))
        written=fixtures.LaneLayoutAuditTests().run_xml(ET.tostring(root,encoding='unicode'),events,n_lanes=6,order_tracks_by_class=True)
        self.assertEqual([n.text for n in written.findall('.//'+q('Payload'))],['blocker','left','right'])

    def test_pyaaf2_orders_tracks_after_channel_repair(self):
        import tempfile
        from pathlib import Path
        import aaf2
        from tests.test_audit_audio import add_comp
        from tests.test_render_equivalence import add_pan
        from aaf_speech_filter.config import FilterConfig
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'input.aaf'
            with aaf2.open(str(path),'w') as aaf:
                comp=add_comp(aaf)
                for index in range(6):
                    slot=comp.create_timeline_slot(24)
                    sequence=aaf.create.Sequence('sound')
                    if index in (0,1,4):
                        sequence.components.append(aaf.create.SourceClip(start=5,length=20,slot_id=index+1,media_kind='sound'))
                        sequence.components.append(aaf.create.Filler('sound',80))
                    else:
                        sequence.components.append(aaf.create.Filler('sound',100))
                    slot.segment=add_pan(aaf,sequence,index%2)
                    slot.segment.length=100
            events=[dict(src_lane=source,top_idx=0,T=0,T_edit=0,L=20,target_lane=target,kind=kind)
                    for source,target,kind in ((0,2,'noise'),(1,1,'noise'),(4,0,'speech'))]
            details={}
            layout._apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path,events=events,
                structural_events=[],n_lanes=6,cfg=FilterConfig(),runtime_essence_paths=None,
                work_dir=Path(folder),cancel_check=lambda:None,progress_callback=None,log_callback=None,
                order_tracks_by_class=True,result_out=details)
            with aaf2.open(str(path),'r') as aaf:
                found=[]
                for lane,slot in enumerate(next(aaf.content.compositionmobs()).slots):
                    for node in slot.segment.segments[0].components:
                        if type(node).__name__=='SourceClip':
                            found.append((node.slot_id,lane,list(slot.segment.parameters)[0].value,node.start,node.length))
                left=next(item for item in found if item[0]==1)
                right=next(item for item in found if item[0]==2)
                self.assertLess(left[1],right[1])
                self.assertEqual(left[2:],(0,5,20))
                self.assertEqual(right[2:],(1,5,20))
            self.assertIn('lane_order',details)
