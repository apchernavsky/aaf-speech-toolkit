"""Cross-stage invariants for gain-wrapped silence and lane layout."""
import tempfile
import unittest
from pathlib import Path
import aaf2

from aaf_speech_filter.aaf_yamnet_lane_layout import _lane_layout_mutable

GAIN_ID = '9d2ea894-0968-11d3-8a38-0050040ef7d2'
UNKNOWN_ID = 'b5ab70c8-9d2e-4b29-aafb-79467bc1b188'

def gain(aaf, child, *, operation_id=GAIN_ID, length=None, name='Unrelated label'):
    operation = aaf.create.OperationDef(operation_id, name)
    operation.media_kind = 'sound'
    operation.number_inputs = 1
    operation['IsTimeWarp'].value = False
    aaf.dictionary.register_def(operation)
    node = aaf.create.OperationGroup(operation, length=child.length if length is None else length,
                                     media_kind='sound')
    node.segments.append(child)
    return node

class SilentAudioLayoutTests(unittest.TestCase):
    def lane_with(self, aaf, node):
        comp = aaf.create.CompositionMob()
        aaf.content.mobs.append(comp)
        slot = comp.create_timeline_slot(48000)
        slot.segment = aaf.create.Sequence(media_kind='sound')
        slot.segment.components.append(node)
        slot.segment.components.append(aaf.create.Filler(media_kind='sound', length=100))
        return slot

    def test_gain_over_removed_clip_does_not_freeze_free_intervals(self):
        with tempfile.TemporaryDirectory() as td, aaf2.open(str(Path(td)/'in.aaf'), 'w') as aaf:
            slot = self.lane_with(aaf, gain(aaf, aaf.create.Filler(media_kind='sound', length=50)))
            self.assertTrue(_lane_layout_mutable(slot, slot.segment))

    def test_unknown_operation_with_gain_name_remains_protected(self):
        with tempfile.TemporaryDirectory() as td, aaf2.open(str(Path(td)/'in.aaf'), 'w') as aaf:
            slot = self.lane_with(aaf, gain(aaf, aaf.create.Filler(media_kind='sound', length=50),
                                          operation_id=UNKNOWN_ID, name='Audio Gain'))
            self.assertFalse(_lane_layout_mutable(slot, slot.segment))

    def test_nested_gain_proof_requires_equal_sound_lengths(self):
        from aaf_io.audio_silence import silent_audio_signature
        with tempfile.TemporaryDirectory() as td, aaf2.open(str(Path(td)/'in.aaf'), 'w') as aaf:
            inner = gain(aaf, aaf.create.Filler(media_kind='sound', length=50))
            outer = aaf.create.OperationGroup(inner.operation, length=50, media_kind='sound')
            outer.segments.append(inner)
            self.assertIsNotNone(silent_audio_signature(outer))
            outer.length = 51
            self.assertIsNone(silent_audio_signature(outer))
            outer.length = 50
            inner.segments.value = [aaf.create.Filler(media_kind='picture', length=50)]
            self.assertIsNone(silent_audio_signature(outer))

    def test_live_and_missing_inputs_never_become_silence(self):
        from aaf_io.audio_silence import silent_audio_signature
        with tempfile.TemporaryDirectory() as td, aaf2.open(str(Path(td)/'in.aaf'), 'w') as aaf:
            node = gain(aaf, aaf.create.SourceClip(media_kind='sound', length=50))
            self.assertIsNone(silent_audio_signature(node))
            node.segments.value = []
            self.assertIsNone(silent_audio_signature(node))

    def make_writer_input(self, path):
        with aaf2.open(str(path), 'w') as aaf:
            comp = aaf.create.CompositionMob()
            aaf.content.mobs.append(comp)
            for index in range(2):
                slot = comp.create_timeline_slot(48000)
                slot.segment = aaf.create.Sequence(media_kind='sound')
                if index == 0:
                    slot.segment.components.append(gain(aaf, aaf.create.Filler(media_kind='sound', length=100)))
                else:
                    slot.segment.components.append(aaf.create.Filler(media_kind='sound', length=20))
                    slot.segment.components.append(aaf.create.SourceClip(media_kind='sound', length=10))
                    slot.segment.components.append(aaf.create.Filler(media_kind='sound', length=70))

    def silence_event(self, *, signature=None):
        from aaf_io.sdk_lane_layout import LaneLayoutEvent
        return LaneLayoutEvent(0, 0, 0, 100, 0, kind='silence', silence_signature=signature or
                               ('OperationGroup',100,GAIN_ID,('Filler',100)))

    def test_pyaaf2_writer_replaces_only_planned_silence_preserving_time(self):
        from aaf_speech_filter.aaf_yamnet_lane_layout import _apply_lane_layout_via_pyaaf2_rebuild
        from aaf_speech_filter.config import FilterConfig
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'in.aaf'
            self.make_writer_input(path)
            _apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path,
                events=[dict(src_lane=1,top_idx=1,T=20,T_edit=20,L=10,target_lane=0)],
                structural_events=[self.silence_event()],n_lanes=2,cfg=FilterConfig(),
                runtime_essence_paths=None,work_dir=Path(td),cancel_check=lambda:None,
                progress_callback=None,log_callback=None)
            with aaf2.open(str(path),'r') as aaf:
                comp=next(aaf.content.compositionmobs())
                nodes=list(comp.slots[0].segment.components)
                self.assertEqual([type(n).__name__ for n in nodes], ['Filler','SourceClip','Filler'])
                self.assertEqual([n.length for n in nodes],[20,10,70])

    def test_pyaaf2_writer_rejects_stale_silence_proof(self):
        from aaf_speech_filter.aaf_yamnet_lane_layout import _apply_lane_layout_via_pyaaf2_rebuild
        from aaf_speech_filter.config import FilterConfig
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'in.aaf'
            self.make_writer_input(path)
            with self.assertRaisesRegex(RuntimeError,'silence'):
                _apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path,
                    events=[dict(src_lane=1,top_idx=1,T=20,T_edit=20,L=10,target_lane=0)],
                    structural_events=[self.silence_event(signature=('Filler',100))],
                    n_lanes=2,cfg=FilterConfig(),runtime_essence_paths=None,work_dir=Path(td),
                    cancel_check=lambda:None,progress_callback=None,log_callback=None)

    def test_sdk_writer_replaces_planned_silence_preserving_time(self):
        import xml.etree.ElementTree as ET
        from aaf_io.sdk_lane_layout import LaneLayoutEvent
        from tests.test_audit_layout import LaneLayoutAuditTests,component,xml_document,q
        wrapper=component('OperationGroup',100)
        ET.SubElement(wrapper,q('Operation')).text='OperationDef_MonoAudioGain'
        ET.SubElement(wrapper,q('InputSegments')).append(component('Filler',100))
        xml=xml_document([[wrapper],[component('Filler',20),component('SourceClip',10),component('Filler',70)]])
        result=LaneLayoutAuditTests().run_xml(xml,[self.silence_event(),LaneLayoutEvent(1,1,20,10,0)])
        nodes=list(result.findall('.//'+q('ComponentObjects'))[0])
        self.assertEqual([n.tag for n in nodes],[q('Filler'),q('SourceClip'),q('Filler')])
        self.assertEqual([int(n.findtext(q('ComponentLength'))) for n in nodes],[20,10,70])

    def test_sdk_writer_rejects_stale_silence_proof(self):
        import xml.etree.ElementTree as ET
        from tests.test_audit_layout import LaneLayoutAuditTests,component,xml_document,q
        wrapper=component('OperationGroup',100)
        ET.SubElement(wrapper,q('Operation')).text='OperationDef_MonoAudioGain'
        ET.SubElement(wrapper,q('InputSegments')).append(component('Filler',100))
        xml=xml_document([[wrapper],[component('Filler',100)]])
        with self.assertRaisesRegex(RuntimeError,'silence'):
            LaneLayoutAuditTests().run_xml(xml,[self.silence_event(signature=('Filler',100))])

    def test_silent_only_transition_owner_keeps_lane_protected(self):
        with tempfile.TemporaryDirectory() as td, aaf2.open(str(Path(td)/'in.aaf'),'w') as aaf:
            silent=gain(aaf,aaf.create.Filler(media_kind='sound',length=50))
            slot=self.lane_with(aaf,silent)
            transition=aaf.create.Transition(media_kind='sound',length=5)
            transition['CutPoint'].value=2
            transition['OperationGroup'].value=gain(aaf,aaf.create.Filler(media_kind='sound',length=5))
            slot.segment.components.value=[aaf.create.Filler(media_kind='sound',length=10),transition,
                                          silent,aaf.create.Filler(media_kind='sound',length=35)]
            slot.segment.length=90
            self.assertFalse(_lane_layout_mutable(slot,slot.segment))

    def test_pyaaf2_rejects_silence_plan_that_orphans_transition(self):
        from aaf_io.sdk_lane_layout import LaneLayoutEvent
        from aaf_speech_filter.aaf_yamnet_lane_layout import _apply_lane_layout_via_pyaaf2_rebuild
        from aaf_speech_filter.config import FilterConfig
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'in.aaf';self.make_writer_input(path)
            with aaf2.open(str(path),'r+') as aaf:
                seq=next(aaf.content.compositionmobs()).slots[0].segment
                silent=seq.components[0];silent.length=50;silent.segments[0].length=50
                transition=aaf.create.Transition(media_kind='sound',length=5)
                transition['CutPoint'].value=2
                transition['OperationGroup'].value=gain(aaf,aaf.create.Filler(media_kind='sound',length=5))
                seq.components.value=[aaf.create.Filler(media_kind='sound',length=10),transition,silent,
                                      aaf.create.Filler(media_kind='sound',length=35)]
                seq.length=90
            plan=LaneLayoutEvent(0,2,15,50,0,kind='silence',
                    silence_signature=('OperationGroup',50,GAIN_ID,('Filler',50)))
            with self.assertRaisesRegex(RuntimeError,'silence'):
                _apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path,
                    events=[dict(src_lane=1,top_idx=1,T=20,T_edit=20,L=10,target_lane=0)],
                    structural_events=[plan],n_lanes=2,cfg=FilterConfig(),runtime_essence_paths=None,
                    work_dir=Path(td),cancel_check=lambda:None,progress_callback=None,log_callback=None)

    def test_sdk_rejects_silence_plan_that_orphans_transition(self):
        import xml.etree.ElementTree as ET
        from aaf_io.sdk_lane_layout import LaneLayoutEvent
        from tests.test_audit_layout import LaneLayoutAuditTests,component,xml_document,q
        wrapper=component('OperationGroup',50)
        ET.SubElement(wrapper,q('Operation')).text='OperationDef_MonoAudioGain'
        ET.SubElement(wrapper,q('InputSegments')).append(component('Filler',50))
        xml=xml_document([[component('Filler',10),component('Transition',5),wrapper,component('Filler',35)],
                          [component('Filler',100)]])
        plan=LaneLayoutEvent(0,2,15,50,0,kind='silence',
                    silence_signature=('OperationGroup',50,GAIN_ID,('Filler',50)))
        with self.assertRaisesRegex(RuntimeError,'silence'):
            LaneLayoutAuditTests().run_xml(xml,[plan])

    def test_gain_silence_next_to_live_crossfade_remains_protected(self):
        with tempfile.TemporaryDirectory() as td, aaf2.open(str(Path(td)/'in.aaf'),'w') as aaf:
            silent=gain(aaf,aaf.create.Filler(media_kind='sound',length=50))
            slot=self.lane_with(aaf,silent)
            transition=aaf.create.Transition(media_kind='sound',length=5)
            transition['CutPoint'].value=2
            transition['OperationGroup'].value=gain(aaf,aaf.create.Filler(media_kind='sound',length=5))
            slot.segment.components.value=[silent,transition,aaf.create.SourceClip(media_kind='sound',length=10),
                                          aaf.create.Filler(media_kind='sound',length=35)]
            slot.segment.length=90
            self.assertFalse(_lane_layout_mutable(slot,slot.segment))

    def test_unplanned_unknown_wrapper_is_preserved_by_pyaaf2(self):
        from aaf_speech_filter.aaf_yamnet_lane_layout import _apply_lane_layout_via_pyaaf2_rebuild
        from aaf_speech_filter.config import FilterConfig
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'in.aaf';self.make_writer_input(path)
            with aaf2.open(str(path),'r+') as aaf:
                comp=next(aaf.content.compositionmobs())
                comp.slots[0].segment.components.value=[
                    gain(aaf,aaf.create.Filler(media_kind='sound',length=10),
                         operation_id=UNKNOWN_ID,name='Audio Gain'),
                    aaf.create.Filler(media_kind='sound',length=90)]
            _apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path,
                events=[dict(src_lane=1,top_idx=1,T=20,T_edit=20,L=10,target_lane=0)],
                structural_events=[],n_lanes=2,cfg=FilterConfig(),runtime_essence_paths=None,
                work_dir=Path(td),cancel_check=lambda:None,progress_callback=None,log_callback=None)
            with aaf2.open(str(path),'r') as aaf:
                nodes=list(next(aaf.content.compositionmobs()).slots[0].segment.components)
                self.assertEqual(type(nodes[0]).__name__,'OperationGroup')
                self.assertEqual(str(nodes[0].operation.auid),UNKNOWN_ID)
                self.assertEqual(nodes[0].length,10)

    def test_quiet_removal_then_layout_moves_speech_into_freed_lane(self):
        import struct
        from unittest.mock import patch
        from tests.test_audit_audio import add_source,add_gain,clip,write_wave
        from aaf_speech_filter.pyaaf2_filter import filter_aaf_speech_only
        from aaf_speech_filter import aaf_yamnet_lane_layout as layout
        from aaf_speech_filter.config import FilterConfig
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);original=folder/'original.aaf';output=folder/'output.aaf'
            quiet=folder/'quiet.wav';audible=folder/'audible.wav'
            write_wave(quiet,bytes(9600))
            write_wave(audible,struct.pack('<h',16000)*4800)
            with aaf2.open(str(original),'w') as aaf:
                silent_source=add_source(aaf,quiet);speech_source=add_source(aaf,audible)
                comp=aaf.create.CompositionMob();aaf.content.mobs.append(comp)
                for lane in range(6):
                    slot=comp.create_timeline_slot(48000)
                    slot.segment=aaf.create.Sequence(media_kind='sound')
                    if lane==0:slot.segment.components.append(add_gain(aaf,silent_source,1))
                    elif lane==5:slot.segment.components.append(clip(speech_source))
                    else:slot.segment.components.append(aaf.create.Filler(media_kind='sound',length=4800))
                    slot.segment.length=4800
            original_bytes=original.read_bytes()
            self.assertEqual(filter_aaf_speech_only(original,output,FilterConfig()),1)
            cfg=FilterConfig(experimental_yamnet_lane_layout=True,media_search_roots=(folder,))
            with patch.object(layout,'yamnet_clip_kind_with_scores',return_value=('speech',0.99,0.0,0.0)), \
                 patch.object(layout,'_validate_pyaaf2_lane_layout_output_for_sdk_open'):
                moved=layout.apply_experimental_yamnet_lane_layout(output,cfg,
                    runtime_essence_paths={},work_dir=folder/'work')
            self.assertGreater(moved,0)
            self.assertEqual(original.read_bytes(),original_bytes)
            with aaf2.open(str(output),'r') as aaf:
                comp=next(aaf.content.compositionmobs())
                first=comp.slots[0].segment.components[0]
                self.assertEqual(type(first).__name__,'SourceClip')
                self.assertEqual(first.length,4800)
                self.assertEqual(first.start,0)
                self.assertEqual(type(comp.slots[5].segment.components[0]).__name__,'Filler')
            before_second=output.read_bytes()
            with patch.object(layout,'yamnet_clip_kind_with_scores',return_value=('speech',0.99,0.0,0.0)), \
                 patch.object(layout,'_validate_pyaaf2_lane_layout_output_for_sdk_open'):
                self.assertEqual(layout.apply_experimental_yamnet_lane_layout(output,cfg,
                    runtime_essence_paths={},work_dir=folder/'work'),0)
            self.assertEqual(output.read_bytes(),before_second)

    def test_moved_transition_preserves_declared_lane_durations(self):
        from aaf_speech_filter.aaf_yamnet_lane_layout import _apply_lane_layout_via_pyaaf2_rebuild
        from aaf_speech_filter.config import FilterConfig
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'in.aaf'
            with aaf2.open(str(path),'w') as aaf:
                comp=aaf.create.CompositionMob();aaf.content.mobs.append(comp)
                slot=comp.create_timeline_slot(48000);slot.segment=aaf.create.Sequence(media_kind='sound')
                transition=aaf.create.Transition(media_kind='sound',length=5)
                transition['CutPoint'].value=2
                transition['OperationGroup'].value=gain(aaf,aaf.create.Filler(media_kind='sound',length=5))
                slot.segment.components.value=[aaf.create.Filler(media_kind='sound',length=50),transition,
                    aaf.create.SourceClip(media_kind='sound',length=10),aaf.create.Filler(media_kind='sound',length=35)]
                slot.segment.length=90
                dest=comp.create_timeline_slot(48000);dest.segment=aaf.create.Sequence(media_kind='sound')
                dest.segment.components.append(aaf.create.Filler(media_kind='sound',length=90));dest.segment.length=90
            _apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path,
                events=[dict(src_lane=0,top_idx=2,T=55,T_edit=45,L=10,target_lane=1,target_T=55,kind='music',
                             pre_transition_len=5,pre_transition_top_idx=1)],
                structural_events=[],n_lanes=2,cfg=FilterConfig(),runtime_essence_paths=None,
                work_dir=Path(td),cancel_check=lambda:None,progress_callback=None,log_callback=None)
            with aaf2.open(str(path),'r') as aaf:
                slots=next(aaf.content.compositionmobs()).slots
                self.assertEqual([slot.segment.length for slot in slots],[90,90])
                nodes=list(slots[1].segment.components)
                self.assertEqual([type(n).__name__ for n in nodes],['Filler','Transition','SourceClip','Filler'])
                self.assertEqual([n.length for n in nodes],[50,5,10,35])

    def test_final_cleanup_preserves_unknown_empty_operation(self):
        from aaf_workflow import flatten_top_level_empty_sound_operationgroups
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'in.aaf'
            with aaf2.open(str(path),'w') as aaf:
                self.lane_with(aaf,gain(aaf,aaf.create.Filler(media_kind='sound',length=50),
                                       operation_id=UNKNOWN_ID,name='Audio Gain'))
            self.assertEqual(flatten_top_level_empty_sound_operationgroups(path),0)
            with aaf2.open(str(path),'r') as aaf:
                self.assertEqual(type(next(aaf.content.compositionmobs()).slots[0].segment.components[0]).__name__,
                                 'OperationGroup')

    def test_final_cleanup_normalizes_only_proven_gain_silence(self):
        from aaf_workflow import flatten_top_level_empty_sound_operationgroups
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'in.aaf'
            with aaf2.open(str(path),'w') as aaf:
                self.lane_with(aaf,gain(aaf,aaf.create.Filler(media_kind='sound',length=50)))
            self.assertEqual(flatten_top_level_empty_sound_operationgroups(path),1)
            with aaf2.open(str(path),'r') as aaf:
                node=next(aaf.content.compositionmobs()).slots[0].segment.components[0]
                self.assertEqual(type(node).__name__,'Filler')
                self.assertEqual(node.length,50)

    def test_sdk_can_create_gap_without_top_level_filler_template(self):
        import xml.etree.ElementTree as ET
        from aaf_io.sdk_lane_layout import LaneLayoutEvent
        from tests.test_audit_layout import LaneLayoutAuditTests,component,xml_document,q
        wrapper=component('OperationGroup',100)
        ET.SubElement(wrapper,q('Operation')).text='OperationDef_MonoAudioGain'
        ET.SubElement(wrapper,q('InputSegments')).append(component('Filler',100))
        xml=xml_document([[wrapper],[component('SourceClip',100)]])
        try:
            result=LaneLayoutAuditTests().run_xml(xml,[self.silence_event(),LaneLayoutEvent(1,0,0,100,0)])
        except RuntimeError as exc:
            self.fail(str(exc))
        lanes=result.findall('.//'+q('ComponentObjects'))
        self.assertEqual([n.tag for n in lanes[0]],[q('SourceClip')])
        self.assertEqual([n.tag for n in lanes[1]],[q('Filler')])
        self.assertEqual(lanes[1][0].findtext(q('ComponentLength')),'100')
