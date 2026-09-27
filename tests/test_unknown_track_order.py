import tempfile,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
import aaf2
from tests.test_audit_layout import component,xml_document,q

SCENARIOS=[([['unknown','music'],['unknown'],['speech']],{1}),([['unknown','music'],['unknown','speech']],set())]

class UnknownTrackOrderTests(unittest.TestCase):
    def test_pyaaf2_preserves_unknown_owners_on_otherwise_classified_tracks(self):
        from aaf_speech_filter.aaf_yamnet_lane_layout import _apply_lane_layout_via_pyaaf2_rebuild
        from aaf_speech_filter.config import FilterConfig
        for kinds,protected in SCENARIOS:
            with self.subTest(kinds=kinds),tempfile.TemporaryDirectory() as td:
                path=Path(td)/'input.aaf';events=[];details={}
                with aaf2.open(str(path),'w') as aaf:
                    comp=aaf.create.CompositionMob();aaf.content.mobs.append(comp);ids=[]
                    for lane,roles in enumerate(kinds):
                        slot=comp.create_timeline_slot(24+lane);ids.append(slot.slot_id)
                        slot.segment=aaf.create.Sequence('sound')
                        for index,role in enumerate(roles):
                            slot.segment.components.append(aaf.create.SourceClip(media_kind='sound',length=10))
                            events.append(dict(src_lane=lane,top_idx=index,T=index*10,T_edit=index*10,L=10,target_lane=lane,kind=role))
                _apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path,events=events,structural_events=[],n_lanes=len(kinds),cfg=FilterConfig(),runtime_essence_paths=None,work_dir=Path(td),cancel_check=lambda:None,progress_callback=None,log_callback=None,immutable_lanes=protected,order_tracks_by_class=True,result_out=details)
                with aaf2.open(str(path),'r') as aaf:
                    self.assertEqual([s.slot_id for s in next(aaf.content.compositionmobs()).slots],ids)

    def test_sdk_preserves_unknown_owners_on_otherwise_classified_tracks(self):
        from aaf_io.sdk_lane_layout import LaneLayoutEvent,apply_lane_layout_via_aaf_sdk_xml
        for kinds,protected in SCENARIOS:
            with self.subTest(kinds=kinds),tempfile.TemporaryDirectory() as td:
                root=ET.fromstring(xml_document([[component('SourceClip',10,str(lane)+':'+role) for role in roles] for lane,roles in enumerate(kinds)]))
                for lane,track in enumerate(root.findall('.//'+q('TimelineTrack'))):
                    rate=track.find(q('EditRate'))
                    if rate is None:rate=ET.SubElement(track,q('EditRate'))
                    rate.text=str(24+lane)+'/1'
                written=[];folder=Path(td)
                def export(tool,source,destination,**kwargs):destination.write_text(ET.tostring(root,encoding='unicode'),encoding='utf8')
                def build(tool,source,destination,**kwargs):written.append(ET.fromstring(source.read_text(encoding='utf8')))
                events=[LaneLayoutEvent(lane,index,index*10,10,lane,class_kind=role) for lane,roles in enumerate(kinds) for index,role in enumerate(roles)]
                with patch('aaf_io.sdk_lane_layout.find_aaffmtconv',return_value=folder/'tool'),patch('aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml',side_effect=export),patch('aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf',side_effect=build):
                    apply_lane_layout_via_aaf_sdk_xml(input_aaf=folder/'input.aaf',output_aaf=folder/'output.aaf',events=events,n_lanes=len(kinds),work_dir=folder,immutable_lanes=protected,order_tracks_by_class=True)
                self.assertEqual([node.text for node in written[0].findall('.//'+q('Payload'))],[str(lane)+':'+role for lane,roles in enumerate(kinds) for role in roles])
