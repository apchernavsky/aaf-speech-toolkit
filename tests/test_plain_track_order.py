import tempfile,unittest,struct
from pathlib import Path
from unittest.mock import patch
import aaf2
from aaf_speech_filter import aaf_yamnet_lane_layout as layout
from aaf_speech_filter.config import FilterConfig
from tests.test_audit_audio import add_source,write_wave

class PlainTrackOrderTests(unittest.TestCase):
    def test_public_layout_orders_plain_tracks_when_clocks_prevent_clip_transfer(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);path=folder/'input.aaf';wav=folder/'audio.wav'
            write_wave(wav,struct.pack('<h',16000)*48000)
            with aaf2.open(str(path),'w') as aaf:
                second=folder/'second.wav';second.write_bytes(wav.read_bytes())
                sources=[add_source(aaf,wav),add_source(aaf,second)]
                comp=aaf.create.CompositionMob();aaf.content.mobs.append(comp)
                ids=[]
                for source,rate in zip(sources,(48000,24000)):
                    slot=comp.create_timeline_slot(rate)
                    seq=aaf.create.Sequence('sound')
                    seq.components.append(source.create_source_clip(1,start=0,length=rate//10,media_kind='sound'))
                    seq.length=rate//10;slot.segment=seq;ids.append(slot.slot_id)
                empty=comp.create_timeline_slot(32000)
                empty.segment=aaf.create.Sequence('sound')
                empty.segment.components.append(aaf.create.Filler('sound',3200))
                ids.append(empty.slot_id)
            cfg=FilterConfig(experimental_yamnet_lane_layout=True,media_search_roots=(folder,))
            with patch.object(layout,'yamnet_clip_kind_with_scores',side_effect=[('music',0.0,0.9,0.0),('speech',0.9,0.0,0.0)]),patch.object(layout,'_validate_pyaaf2_lane_layout_output_for_sdk_open'):
                layout.apply_experimental_yamnet_lane_layout(path,cfg,runtime_essence_paths={},work_dir=folder/'work')
            with aaf2.open(str(path),'r') as aaf:
                slots=list(next(aaf.content.compositionmobs()).slots)
                self.assertEqual([s.slot_id for s in slots],[ids[1],ids[2],ids[0]])
                self.assertEqual([int(s.edit_rate) for s in slots],[24000,32000,48000])
                self.assertEqual([s.segment.components[0].length for s in slots],[2400,3200,4800])
                self.assertEqual([s.segment.components[0].start for s in (slots[0],slots[2])],[0,0])
