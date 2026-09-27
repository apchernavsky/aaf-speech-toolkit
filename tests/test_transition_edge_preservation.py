"""Owned transition edges survive both lane writers and coalesce only once."""
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import aaf2
from aaf_io.sdk_lane_layout import LaneLayoutEvent, _trim_leading_xml_transitions_to_cursor
from aaf_speech_filter.aaf_yamnet_lane_layout import (
    _apply_lane_layout_via_pyaaf2_rebuild, _trim_leading_rebuilt_transitions_to_cursor,
)
from aaf_speech_filter.config import FilterConfig
from tests import test_audit_layout as xml_fixtures

component, q, xml_document = xml_fixtures.component, xml_fixtures.q, xml_fixtures.xml_document

ROOT = Path(__file__).resolve().parents[1]


def xml_transition(length, cut):
    node = component('Transition', length)
    ET.SubElement(node, q('CutPoint')).text = str(cut)
    group = ET.SubElement(node, q('OperationGroup'))
    ET.SubElement(group, q('Operation')).text = '0c3bea41-fc05-11d2-8a29-0050040ef7d2'
    ET.SubElement(group, '{urn:extension}Curve').text = 'nontrivial opaque render data'
    return node


def xml_positions(nodes):
    raw, overlap = 0, 0
    events = []
    for node in nodes:
        length = int(node.findtext(q('ComponentLength'), '0'))
        if node.tag == q('Transition'):
            overlap += length
        elif node.tag == q('SourceClip'):
            events.append(raw - 2 * overlap)
        raw += length
    return events


def aaf_transition(aaf, length, cut):
    try:
        operation = aaf.dictionary.lookup_operationdef('MonoAudioDissolve')
    except Exception:
        operation = aaf.create.OperationDef('0c3bea41-fc05-11d2-8a29-0050040ef7d2', 'MonoAudioDissolve')
        operation.media_kind = 'sound'
        operation.number_inputs = 2
        aaf.dictionary.register_def(operation)
    try:
        aaf.dictionary.lookup_parameterdef('Level')
    except Exception:
        aaf.dictionary.register_def(aaf.create.ParameterDef('e4962320-2267-11d3-8a4c-0050040ef7d2', 'Level', typedef='Rational'))
    try:
        aaf.dictionary.lookup_interperlationdef('CustomCurve')
    except Exception:
        interpolation = aaf.create.InterpolationDef('f7752946-3298-477f-ab65-f84e10f69dbe', 'CustomCurve')
        aaf.dictionary.register_def(interpolation)
    group = aaf.create.OperationGroup(operation, media_kind='sound', length=length)
    curve = aaf.create.VaryingValue('Level', 'CustomCurve')
    curve.add_keyframe(0, 0)
    curve.add_keyframe(1, '1/3')
    group.parameters.append(curve)
    transition = aaf.create.Transition(media_kind='sound', length=length)
    transition['CutPoint'].value = cut
    transition['OperationGroup'].value = group
    return transition


def transition_signature(transition):
    group = transition['OperationGroup'].value
    curve = next(iter(group.parameters))
    return (transition.length, transition['CutPoint'].value, str(group.operation.auid),
            str(curve.interpolation.auid), tuple((str(p.time), str(p.value)) for p in curve.pointlist))


class TransitionPreservationTests(unittest.TestCase):
    def test_sdk_first_event_preserves_owned_leading_and_trailing_edges(self):
        pre, post = xml_transition(7, 0), xml_transition(3, 3)
        document = xml_document([[component('Filler', 7), pre, component('SourceClip', 41), post, component('Filler', 3)]])
        event = LaneLayoutEvent(0, 2, 14, 41, 0, visible_t=0,
            preferred_transition_offset=7, pre_top_idx=1, pre_len=7, post_top_idx=3, post_len=3)
        result = xml_fixtures.LaneLayoutAuditTests().run_xml(document, [event], n_lanes=1)
        nodes = list(result.find('.//' + q('ComponentObjects')))
        self.assertEqual([ET.tostring(node) for node in nodes if node.tag == q('Transition')], [ET.tostring(pre), ET.tostring(post)])
        self.assertEqual(xml_positions(nodes), [0])
        self.assertEqual([(node.tag, node.findtext(q('ComponentLength'))) for node in nodes[:3]],
                         [(q('Filler'), '7'), (q('Transition'), '7'), (q('SourceClip'), '41')])

    def test_sdk_shared_crossfade_has_one_edge_and_unchanged_positions(self):
        edge = xml_transition(5, 2)
        document = xml_document([[component('SourceClip', 20), edge, component('SourceClip', 30)]])
        events = [LaneLayoutEvent(0, 0, 0, 20, 0, visible_t=0, post_top_idx=1, post_len=5),
                  LaneLayoutEvent(0, 2, 25, 30, 0, visible_t=15, preferred_transition_offset=5, pre_top_idx=1, pre_len=5)]
        result = xml_fixtures.LaneLayoutAuditTests().run_xml(document, events, n_lanes=1)
        nodes = list(result.find('.//' + q('ComponentObjects')))
        self.assertEqual([ET.tostring(node) for node in nodes if node.tag == q('Transition')], [ET.tostring(edge)])
        self.assertEqual(xml_positions(nodes), [0, 15])

    def test_sdk_planned_structural_edge_coalesces_only_its_owned_copy(self):
        edge = xml_transition(5, 2)
        document = xml_document([[component('SourceClip', 20), edge, component('SourceClip', 30)]])
        events = [LaneLayoutEvent(0, 0, 0, 20, 0, visible_t=0),
                  LaneLayoutEvent(0, 1, 20, 5, 0, kind='transition'),
                  LaneLayoutEvent(0, 2, 25, 30, 0, visible_t=15, preferred_transition_offset=5, pre_top_idx=1, pre_len=5)]
        result = xml_fixtures.LaneLayoutAuditTests().run_xml(document, events, n_lanes=1)
        nodes = list(result.find('.//' + q('ComponentObjects')))
        self.assertEqual([ET.tostring(node) for node in nodes if node.tag == q('Transition')], [ET.tostring(edge)])
        self.assertEqual(xml_positions(nodes), [0, 15])

    def run_pyaaf(self, shared):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            path = Path(td) / 'work.aaf'
            with aaf2.open(str(path), 'w') as aaf:
                comp = aaf.create.CompositionMob()
                aaf.content.mobs.append(comp)
                sequence = aaf.create.Sequence(media_kind='sound')
                if shared:
                    nodes = [aaf.create.SourceClip(media_kind='sound', length=20), aaf_transition(aaf, 5, 2), aaf.create.SourceClip(media_kind='sound', length=30)]
                    events = [dict(src_lane=0, top_idx=0, T=0, T_edit=0, L=20, target_lane=0, kind='speech'),
                              dict(src_lane=0, top_idx=2, T=25, T_edit=15, L=30, target_lane=0, kind='speech')]
                else:
                    nodes = [aaf.create.Filler('sound', 7), aaf_transition(aaf, 7, 0), aaf.create.SourceClip(media_kind='sound', length=41), aaf_transition(aaf, 3, 3), aaf.create.Filler('sound', 3)]
                    events = [dict(src_lane=0, top_idx=2, T=14, T_edit=0, L=41, target_lane=0, kind='speech')]
                sequence.components.extend(nodes)
                sequence.length = 45 if shared else 41
                comp.create_timeline_slot(25).segment = sequence
                expected = [transition_signature(node) for node in nodes if type(node).__name__ == 'Transition']
            def rebuild():
                _apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path, events=events, structural_events=[], n_lanes=1,
                    cfg=FilterConfig(), runtime_essence_paths=None, work_dir=Path(td), cancel_check=lambda: None,
                    progress_callback=None, log_callback=None)
            if shared:
                with self.assertRaisesRegex(RuntimeError, 'overlap without moving time'):
                    rebuild()
            else:
                rebuild()
            with aaf2.open(str(path), 'r') as aaf:
                sequence = next(aaf.content.compositionmobs()).slots[0].segment
                self.assertEqual([transition_signature(node) for node in sequence.components if type(node).__name__ == 'Transition'], expected)
                self.assertEqual([position for _, position, node in sequence.positions() if type(node).__name__ == 'SourceClip'], [0, 15] if shared else [0])

    def test_pyaaf_first_event_preserves_owned_leading_and_trailing_edges(self):
        self.run_pyaaf(shared=False)

    def test_pyaaf_shared_crossfade_refusal_preserves_edges_and_positions(self):
        self.run_pyaaf(shared=True)

    def test_xml_cursor_overlap_is_not_an_edge_witness(self):
        transition, event = xml_transition(5, 0), component('SourceClip', 20)
        self.assertEqual(_trim_leading_xml_transitions_to_cursor(95, 25, [transition, event], 100), (95, 25, [transition, event]))

    def test_pyaaf_cursor_overlap_is_not_an_edge_witness(self):
        from tests.test_speech_yamnet import Transition, SourceClip
        transition, event = Transition(5), SourceClip(20)
        self.assertEqual(_trim_leading_rebuilt_transitions_to_cursor(95, 25, [transition, event], 100), (95, 25, [transition, event]))

    def test_xml_different_transition_at_same_position_is_not_a_witness(self):
        transition, event = xml_transition(5, 0), component('SourceClip', 20)
        other = xml_transition(5, 2)
        self.assertEqual(_trim_leading_xml_transitions_to_cursor(95, 25, [transition, event], 100,
            emitted_transitions={95: other}), (95, 25, [transition, event]))

    def test_pyaaf_different_transition_at_same_position_is_not_a_witness(self):
        from tests.test_speech_yamnet import Transition, SourceClip
        transition, event = Transition(5), SourceClip(20)
        self.assertEqual(_trim_leading_rebuilt_transitions_to_cursor(95, 25, [transition, event], 100,
            emitted_transitions={95: Transition(5)}), (95, 25, [transition, event]))

    def test_xml_shared_clone_coalesces_with_original_edge_witness(self):
        import copy
        original, event = xml_transition(5, 2), component('SourceClip', 20)
        clone = copy.deepcopy(original)
        self.assertEqual(_trim_leading_xml_transitions_to_cursor(95, 25, [clone, event], 100,
            emitted_transitions={95: original}, transition_sources={id(clone): original}), (100, 20, [event]))

    def test_pyaaf_shared_clone_coalesces_with_original_edge_witness(self):
        from tests.test_speech_yamnet import Transition, SourceClip
        original, clone, event = Transition(5), Transition(5), SourceClip(20)
        self.assertEqual(_trim_leading_rebuilt_transitions_to_cursor(95, 25, [clone, event], 100,
            emitted_transitions={95: original}, transition_sources={id(clone): original}), (100, 20, [event]))


if __name__ == '__main__':
    unittest.main()
