import tempfile,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
import aaf2
from tests.test_audit_layout import component,xml_document,q

class FixedTrackPayloadTests(unittest.TestCase):
    def test_pyaaf2_fixed_noise_peer_is_present_in_final_track_order(self):
        from aaf_speech_filter.aaf_yamnet_lane_layout import _apply_lane_layout_via_pyaaf2_rebuild
        from aaf_speech_filter.config import FilterConfig
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'input.aaf'
            with aaf2.open(str(path),'w') as aaf:
                comp=aaf.create.CompositionMob();aaf.content.mobs.append(comp)
                ids=[]
                for lane in range(7):
                    slot=comp.create_timeline_slot(24+lane);ids.append(slot.slot_id)
                    seq=aaf.create.Sequence('sound')
                    seq.components.append(aaf.create.SourceClip(media_kind='sound',length=10) if lane in (4,5) else aaf.create.Filler('sound',10))
                    slot.segment=seq
            events=[dict(src_lane=i,top_idx=0,T=0,T_edit=0,L=10,target_lane=i,kind='noise') for i in (4,5)]
            details={}
            _apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path,events=events,structural_events=[],n_lanes=7,cfg=FilterConfig(),runtime_essence_paths=None,work_dir=Path(td),cancel_check=lambda:None,progress_callback=None,log_callback=None,immutable_lanes={4},order_tracks_by_class=True,result_out=details)
            with aaf2.open(str(path),'r') as aaf:
                output=[s.slot_id for s in next(aaf.content.compositionmobs()).slots]
                self.assertLess(output.index(ids[4]),output.index(ids[5]))
                self.assertEqual(output[4],ids[4])

    def test_sdk_fixed_noise_peer_is_present_in_final_track_order(self):
        from aaf_io.sdk_lane_layout import LaneLayoutEvent,apply_lane_layout_via_aaf_sdk_xml
        root=ET.fromstring(xml_document([[component('SourceClip',10,str(i))] if i in (4,5) else [component('Filler',10)] for i in range(7)]))
        for i,track in enumerate(root.findall('.//'+q('TimelineTrack'))):
            rate=track.find(q('EditRate'))
            if rate is None:rate=ET.SubElement(track,q('EditRate'))
            rate.text=str(24+i)+'/1'
        written=[]
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            def export(tool,source,destination,**kwargs):
                destination.write_text(ET.tostring(root,encoding='unicode'),encoding='utf8')
            def build(tool,source,destination,**kwargs):
                written.append(ET.fromstring(source.read_text(encoding='utf8')))
            with patch('aaf_io.sdk_lane_layout.find_aaffmtconv',return_value=folder/'tool'),patch('aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml',side_effect=export),patch('aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf',side_effect=build):
                apply_lane_layout_via_aaf_sdk_xml(input_aaf=folder/'input.aaf',output_aaf=folder/'output.aaf',events=[LaneLayoutEvent(i,0,0,10,i,class_kind='noise') for i in (4,5)],n_lanes=7,work_dir=folder,immutable_lanes={4},order_tracks_by_class=True)
        self.assertEqual([node.text for node in written[0].findall('.//'+q('Payload'))],['4','5'])
