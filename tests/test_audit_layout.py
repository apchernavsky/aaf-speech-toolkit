"""Regression invariants for lane layout ownership and serialization."""
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import aaf2
from aaf_io.errors import OperationCancelled
from aaf_io.sdk_lane_layout import LaneLayoutEvent, apply_lane_layout_via_aaf_sdk_xml
from aaf_speech_filter.aaf_yamnet_lane_layout import (
    _apply_edit_timeline_unification, _lane_layout_signature, _lane_layout_mutable,
    _apply_unknown_opening_music_fallback, _apply_lane_layout_via_pyaaf2_rebuild,
)
from aaf_speech_filter.config import FilterConfig
from aaf_speech_filter.lane_layout_constraints import target_lane_allowed_for_event
from aaf_speech_filter.lane_layout_writer import write_lane_layout_with_sdk_or_pyaaf2_fallback

NS = "http://www.aafassociation.org/aafx/v1.1/20090617"
def q(tag):
    return "{%s}%s" % (NS, tag)

def component(tag, length, payload=None):
    node = ET.Element(q(tag))
    ET.SubElement(node, q("ComponentLength")).text = str(length)
    ET.SubElement(node, q("ComponentDataDefinition")).text = "DataDef_Sound"
    if payload is not None:
        ET.SubElement(node, q("Payload")).text = payload
    return node

def xml_document(lanes, rates=None, origins=None):
    ET.register_namespace("", NS)
    root = ET.Element(q("AAF"))
    comp = ET.SubElement(root, q("CompositionPackage"))
    ET.SubElement(comp, q("PackageUsage")).text = "Usage_TopLevel"
    tracks = ET.SubElement(comp, q("PackageTracks"))
    for i, nodes in enumerate(lanes):
        track = ET.SubElement(tracks, q("TimelineTrack"))
        ET.SubElement(track, q("EditRate")).text = str(rates[i] if rates else "48000/1")
        ET.SubElement(track, q("Origin")).text = str(origins[i] if origins else 0)
        seq = ET.SubElement(ET.SubElement(track, q("TrackSegment")), q("Sequence"))
        ET.SubElement(seq, q("ComponentLength")).text = "100"
        seq.append(ET.Element(q("ComponentObjects")))
        seq[-1].extend(nodes)
    return ET.tostring(root, encoding="unicode")

class LaneLayoutAuditTests(unittest.TestCase):
    def test_reused_source_occurrences_keep_individual_timeline_positions(self):
        with tempfile.TemporaryDirectory() as td, aaf2.open(str(Path(td)/"in.aaf"), "w") as f:
            source = f.create.SourceClip(media_kind="sound", length=2400)
            events = [dict(node=source, src_lane=0, er=48000, T=t, L=2400)
                      for t in (48000, 52800)]
            _apply_edit_timeline_unification(events)
            self.assertEqual([e["T_edit"] for e in events], [48000, 52800])
            events[1]["T_nuendo"] = 50000
            _apply_edit_timeline_unification(events)
            self.assertEqual([e["T_edit"] for e in events], [48000, 50000])

    def test_lane_moves_require_exact_rate_and_origin(self):
        with tempfile.TemporaryDirectory() as td, aaf2.open(str(Path(td)/"in.aaf"), "w") as f:
            mob = f.create.CompositionMob()
            f.content.mobs.append(mob)
            slots = []
            for rate, origin in ((48000, 0), (96000, 0), (48000, 1), ("96000/2", 0)):
                slot = mob.create_timeline_slot(rate)
                slot.origin = origin
                slot.segment = f.create.Sequence(media_kind="sound")
                slot.segment.components.append(f.create.Filler(media_kind="sound", length=100))
                slots.append(slot)
            signatures = {i: _lane_layout_signature(s, s.segment) for i, s in enumerate(slots)}
            self.assertFalse(target_lane_allowed_for_event({"src_lane": 0}, 1, lane_signature_by_lane=signatures))
            self.assertFalse(target_lane_allowed_for_event({"src_lane": 0}, 2, lane_signature_by_lane=signatures))
            self.assertTrue(target_lane_allowed_for_event({"src_lane": 0}, 3, lane_signature_by_lane=signatures))

    def test_opaque_selector_lane_is_immutable(self):
        with tempfile.TemporaryDirectory() as td, aaf2.open(str(Path(td)/"in.aaf"), "w") as f:
            mob = f.create.CompositionMob()
            f.content.mobs.append(mob)
            slot = mob.create_timeline_slot(48000)
            slot.segment = f.create.Sequence(media_kind="sound")
            selector = f.create.Selector(media_kind="sound", length=10)
            selector["Selected"].value = f.create.Filler(media_kind="sound", length=10)
            slot.segment.components.append(selector)
            self.assertFalse(_lane_layout_mutable(slot, slot.segment))

    def run_xml(self, xml, events, n_lanes=2, order_tracks_by_class=False):
        written = []
        def export(_tool, _input, path, **kwargs):
            path.write_text(xml, encoding="utf-8")
        def rebuild(_tool, path, _output, **kwargs):
            written.append(path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with mock.patch("aaf_io.sdk_lane_layout.find_aaffmtconv", return_value=tmp/"tool"), \
                 mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml", side_effect=export), \
                 mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf", side_effect=rebuild):
                apply_lane_layout_via_aaf_sdk_xml(input_aaf=tmp/"in.aaf", output_aaf=tmp/"out.aaf",
                    events=events, n_lanes=n_lanes, work_dir=tmp, order_tracks_by_class=order_tracks_by_class)
        return ET.fromstring(written[0])

    def test_sdk_partial_plan_preserves_opaque_and_unplanned_source_payload(self):
        for tag in ("Selector", "NestedScope", "OperationGroup", "SourceClip"):
            with self.subTest(tag=tag):
                retained = component(tag, 10, "retained")
                xml = xml_document([[component("Filler", 10), retained,
                    component("SourceClip", 10, "moved"), component("Filler", 70)],
                    [component("Filler", 100)]])
                written = self.run_xml(xml, [LaneLayoutEvent(0, 2, 20, 10, 1)])
                lanes = written.findall(".//"+q("ComponentObjects"))
                self.assertIn(ET.tostring(retained), [ET.tostring(node) for node in lanes[0]])
                self.assertEqual(ET.tostring(lanes[0][1]), ET.tostring(retained))
                self.assertEqual(lanes[1][1].find(q("Payload")).text, "moved")
                self.assertEqual(lanes[0][0].find(q("ComponentLength")).text, "10")
                self.assertEqual(lanes[1][0].find(q("ComponentLength")).text, "20")

    def test_sdk_rejects_moving_between_different_clocks(self):
        for rates, origins in ((["48000/1", "96000/1"], [0, 0]), (["48000/1"]*2, [0, 1])):
            xml = xml_document([[component("SourceClip", 10), component("Filler", 90)],
                                [component("Filler", 100)]], rates, origins)
            with self.assertRaisesRegex(RuntimeError, "clock"):
                self.run_xml(xml, [LaneLayoutEvent(0, 0, 0, 10, 1)])

    def test_pyaaf2_rejects_moving_between_different_clocks(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"in.aaf"
            with aaf2.open(str(path), "w") as f:
                mob = f.create.CompositionMob()
                f.content.mobs.append(mob)
                for lane, rate in enumerate((48000, 96000)):
                    slot = mob.create_timeline_slot(rate)
                    slot.segment = f.create.Sequence(media_kind="sound")
                    if lane == 0:
                        slot.segment.components.append(f.create.SourceClip(media_kind="sound", length=10))
                    slot.segment.components.append(f.create.Filler(media_kind="sound", length=90 if lane == 0 else 100))
            with self.assertRaisesRegex(RuntimeError, "clock"):
                _apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path,
                    events=[dict(src_lane=0, top_idx=0, T=0, T_edit=0, L=10, target_lane=1)],
                    structural_events=[], n_lanes=2, cfg=FilterConfig(), runtime_essence_paths=None,
                    work_dir=Path(td), cancel_check=lambda: None, progress_callback=None, log_callback=None)

    def test_pyaaf2_rejects_moves_touching_immutable_lanes(self):
        for immutable in ({0}, {1}):
            with self.subTest(immutable=immutable), tempfile.TemporaryDirectory() as td:
                path = Path(td)/"in.aaf"
                with aaf2.open(str(path), "w") as f:
                    mob = f.create.CompositionMob()
                    f.content.mobs.append(mob)
                    for lane in range(2):
                        slot = mob.create_timeline_slot(48000)
                        slot.segment = f.create.Sequence(media_kind="sound")
                        if lane == 0:
                            slot.segment.components.append(f.create.SourceClip(media_kind="sound", length=10))
                        slot.segment.components.append(f.create.Filler(media_kind="sound", length=90 if lane == 0 else 100))
                with self.assertRaisesRegex(RuntimeError, "immutable"):
                    _apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path,
                        events=[dict(src_lane=0, top_idx=0, T=0, T_edit=0, L=10, target_lane=1)],
                        structural_events=[], n_lanes=2, cfg=FilterConfig(), runtime_essence_paths=None,
                        work_dir=Path(td), cancel_check=lambda: None, progress_callback=None,
                        log_callback=None, immutable_lanes=immutable)
                with aaf2.open(str(path), "r") as f:
                    comp = next(f.content.compositionmobs())
                    self.assertEqual(type(comp.slots[0].segment.components[0]).__name__, "SourceClip")
                    self.assertEqual(len(comp.slots[1].segment.components), 1)

    def test_unknown_opening_music_requires_explicit_opt_in(self):
        self.assertEqual(FilterConfig().lane_layout_unknown_opening_music_sec, 0.0)
        events = [dict(kind="unknown", T=0, T_edit=0, L=48000, er=48000)]
        _apply_unknown_opening_music_fallback(events, 10.0)
        self.assertEqual(events[0]["kind"], "music")

    def test_valid_binary_aaf_does_not_require_sdk_xml_export(self):
        from aaf_speech_filter.aaf_yamnet_lane_layout import _validate_pyaaf2_lane_layout_output_for_sdk_open
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            output = folder / 'output.aaf'
            output.write_bytes(b'persisted output')
            normalized_paths = []
            def normalize(_tool, source, target, **kwargs):
                self.assertEqual(source, output)
                target.write_bytes(b'normalized validation copy')
                normalized_paths.append(target)
            with mock.patch('aaf_io.sdk_tools.find_comaafinfo', return_value=folder/'info'), \
                 mock.patch('aaf_io.sdk_tools.find_aaffmtconv', return_value=folder/'converter'), \
                 mock.patch('aaf_io.sdk_tools.run_comaafinfo', return_value=(0, 'readable')), \
                 mock.patch('aaf_io.sdk_tools.run_aaffmtconv_to_xml', side_effect=RuntimeError('XML exporter unsupported')) as xml, \
                 mock.patch('aaf_io.sdk_tools.run_aaffmtconv_to_structured_storage', side_effect=normalize) as binary:
                _validate_pyaaf2_lane_layout_output_for_sdk_open(output, work_dir=folder/'validation')
            binary.assert_called_once()
            xml.assert_not_called()
            self.assertEqual(output.read_bytes(), b'persisted output')
            self.assertTrue(all(not path.exists() for path in normalized_paths))

    def test_cancellation_does_not_start_sdk_fallback(self):
        def cancelled(**kwargs):
            raise OperationCancelled("stop")
        sdk = mock.Mock()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(OperationCancelled):
                write_lane_layout_with_sdk_or_pyaaf2_fallback(
                    input_aaf=Path(td)/"in.aaf", output_aaf=Path(td)/"out.aaf", events=[],
                    structural_events=[], n_lanes=0, cfg=FilterConfig(), runtime_essence_paths=None,
                    work_dir=Path(td), cancel_check=lambda: None, log_callback=None, result_out=None,
                    build_sdk_events=lambda *args: [], class_zone_lane_order=lambda *args: [],
                    pyaaf2_rebuild=cancelled, sdk_xml_writer=sdk,
                    sdk_exporter_unavailable_error=lambda ex: False)
        sdk.assert_not_called()
