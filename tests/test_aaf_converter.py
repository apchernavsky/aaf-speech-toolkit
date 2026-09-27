from __future__ import annotations

import struct
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

repo_root = Path(__file__).resolve().parents[1]
for p in (repo_root, repo_root / "aaf_io"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from aaf_io.converter import AAFConverter  # noqa: E402
from aaf_io import sdk_xml as aaf_sdk_xml  # noqa: E402
from aaf_io.sdk_lane_layout import (  # noqa: E402
    LaneLayoutEvent,
    apply_lane_layout_via_aaf_sdk_xml,
    _event_candidate_lanes as sdk_event_candidate_lanes,
    _xml_declared_sequence_length_from_children as sdk_declared_sequence_length,
    _xml_lane_occupancy_intervals as sdk_xml_lane_occupancy_intervals,
    _raw_from_visible_for_part as sdk_raw_from_visible_for_part,
    _raw_from_visible_with_transition_spans as sdk_raw_from_visible,
    _rebuilt_lane_total_length as sdk_lane_total_length,
    _trim_leading_xml_transitions_to_cursor as sdk_trim_leading_xml_transitions,
)


class _Prop:
    def __init__(self, value):
        self.value = value


class _SummaryOnlyDescriptor:
    def __init__(self, summary: bytes):
        self._summary = summary

    def __getitem__(self, key: str):
        if key == "Summary":
            return _Prop(list(self._summary))
        raise KeyError(key)


def _wav_summary(*, channels: int, sample_rate: int, bits: int) -> bytes:
    block_align = channels * max(1, bits // 8)
    byte_rate = sample_rate * block_align
    fmt = struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, bits)
    return (
        b"RIFF"
        + struct.pack("<I", 4 + 8 + len(fmt) + 8)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", 0)
    )


class AAFConverterAudioParamsTests(unittest.TestCase):
    def test_get_audio_params_uses_wav_summary_bit_depth(self) -> None:
        desc = _SummaryOnlyDescriptor(_wav_summary(channels=1, sample_rate=48000, bits=24))

        params = AAFConverter("dummy.aaf")._get_audio_params(desc)

        self.assertEqual(params["channels"], 1)
        self.assertEqual(params["sample_rate_float"], 48000.0)
        self.assertEqual(params["quantization_bits"], 24)


class AAFSdkXmlEditTests(unittest.TestCase):
    def _component_xml(self, tag: str, length: int) -> ET.Element:
        node = ET.Element("{http://www.aafassociation.org/aafx/v1.1/20090617}" + tag)
        ln = ET.SubElement(
            node,
            "{http://www.aafassociation.org/aafx/v1.1/20090617}ComponentLength",
        )
        ln.text = str(int(length))
        return node

    def test_sdk_lane_length_extends_to_moved_clip_end(self) -> None:
        self.assertEqual(sdk_lane_total_length(250, 70733), 70733)
        self.assertEqual(sdk_lane_total_length(70733, 250), 70733)

    def test_sdk_xml_declared_length_accounts_for_transition_overlap(self) -> None:
        nodes = [
            self._component_xml("OperationGroup", 20),
            self._component_xml("Transition", 2),
            self._component_xml("OperationGroup", 30),
        ]

        self.assertEqual(sdk_declared_sequence_length(nodes), 48)

    def test_sdk_xml_declared_length_without_transitions_is_component_sum(self) -> None:
        nodes = [
            self._component_xml("Filler", 12),
            self._component_xml("OperationGroup", 20),
        ]

        self.assertEqual(sdk_declared_sequence_length(nodes), 32)

    def test_sdk_lane_raw_from_visible_keeps_preferred_offset(self) -> None:
        raw = sdk_raw_from_visible(
            [(100, 2), (200, 2)],
            visible_t=57037,
            preferred_offset=2,
        )

        self.assertEqual(raw, 57049)

    def test_sdk_lane_raw_from_visible_skips_owned_pre_transition(self) -> None:
        raw = sdk_raw_from_visible(
            [(707, 1)],
            visible_t=706,
            preferred_offset=1,
        )

        self.assertEqual(raw, 708)

    def test_sdk_lane_part_raw_excludes_own_pre_transition_span(self) -> None:
        raw_t = sdk_raw_from_visible_for_part(
            transition_spans=[(100, 2), (196, 2)],
            own_transition_spans=[(196, 2)],
            visible_t=190,
            preferred_offset=2,
            event_offset=2,
        )

        self.assertEqual(raw_t, 196)

    def test_sdk_xml_lane_occupancy_includes_transitions(self) -> None:
        intervals = sdk_xml_lane_occupancy_intervals(
            [
                (100, 5, object(), "transition", None, 0, 0, None),
                (110, 20, object(), "structural", None, 0, 0, None),
            ]
        )

        self.assertEqual(intervals, [(100, 5), (110, 20)])

    def test_sdk_xml_trims_fully_consumed_leading_transition_before_underflow(self) -> None:
        transition = self._component_xml("Transition", 5)
        event = self._component_xml("OperationGroup", 20)

        out_t, out_l, node_xml = sdk_trim_leading_xml_transitions(
            95,
            25,
            [transition, event],
            100,
            emitted_transitions={95: transition},
        )

        self.assertEqual(out_t, 100)
        self.assertEqual(out_l, 20)
        self.assertEqual(node_xml, [event])

    def test_sdk_xml_places_visible_gap_after_transition_on_upper_lane(self) -> None:
        ns = "http://www.aafassociation.org/aafx/v1.1/20090617"
        q = lambda tag: f"{{{ns}}}{tag}"

        def component(tag: str, length: int) -> ET.Element:
            node = ET.Element(q(tag))
            ln = ET.SubElement(node, q("ComponentLength"))
            ln.text = str(int(length))
            dd = ET.SubElement(node, q("ComponentDataDefinition"))
            dd.text = "DataDef_Sound"
            return node

        def track(children: list[ET.Element]) -> ET.Element:
            tt = ET.Element(q("TimelineTrack"))
            seg = ET.SubElement(tt, q("TrackSegment"))
            seq = ET.SubElement(seg, q("Sequence"))
            ln = ET.SubElement(seq, q("ComponentLength"))
            ln.text = "4193280"
            comps = ET.SubElement(seq, q("ComponentObjects"))
            for child in children:
                comps.append(child)
            return tt

        root = ET.Element(q("AAF"))
        comp = ET.SubElement(root, q("CompositionPackage"))
        usage = ET.SubElement(comp, q("PackageUsage"))
        usage.text = "Usage_TopLevel"
        tracks = ET.SubElement(comp, q("PackageTracks"))
        for _lane in range(5):
            tracks.append(
                track(
                    [
                        component("Filler", 432000),
                        component("OperationGroup", 195840),
                        component("Transition", 21120),
                        component("OperationGroup", 178560),
                        component("Filler", 3365760),
                    ]
                )
            )
        tracks.append(track([component("Filler", 4193280)]))
        for _lane in range(5):
            tracks.append(
                track(
                    [
                        component("Filler", 827520),
                        component("OperationGroup", 186240),
                        component("Filler", 3179520),
                    ]
                )
            )
        input_xml = ET.tostring(root, encoding="unicode")
        final_xml: dict[str, str] = {}

        def fake_to_xml(_tool: Path, _input: Path, xml_path: Path, **_kwargs) -> None:
            xml_path.write_text(input_xml, encoding="utf-8")

        def fake_to_aaf(_tool: Path, xml_path: Path, _output: Path, **_kwargs) -> None:
            final_xml["text"] = xml_path.read_text(encoding="utf-8")

        def fake_splice(_xml_text: str, *, new_comp_xml: str, composition_id=None) -> str:
            return new_comp_xml

        events: list[LaneLayoutEvent] = []
        for lane in range(5):
            events.extend(
                [
                    LaneLayoutEvent(
                        src_lane=lane,
                        top_idx=1,
                        T_edit=432000,
                        L=195840,
                        target_lane=lane,
                        visible_t=432000,
                        class_kind="speech",
                        post_top_idx=2,
                        post_len=21120,
                    ),
                    LaneLayoutEvent(
                        src_lane=lane,
                        top_idx=3,
                        T_edit=648960,
                        L=178560,
                        target_lane=lane,
                        visible_t=606720,
                        class_kind="speech",
                        preferred_transition_offset=21120,
                        pre_top_idx=2,
                        pre_len=21120,
                    ),
                    LaneLayoutEvent(
                        src_lane=6 + lane,
                        top_idx=1,
                        T_edit=827520,
                        L=186240,
                        target_lane=lane,
                        visible_t=785280,
                        class_kind="speech",
                    ),
                ]
            )

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with (
                mock.patch("aaf_io.sdk_lane_layout.find_aaffmtconv", return_value=tmp / "aaffmtconv.exe"),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml", side_effect=fake_to_xml),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf", side_effect=fake_to_aaf),
                mock.patch("aaf_io.sdk_lane_layout.splice_composition_package", side_effect=fake_splice),
            ):
                apply_lane_layout_via_aaf_sdk_xml(
                    input_aaf=tmp / "in.aaf",
                    output_aaf=tmp / "out.aaf",
                    n_lanes=11,
                    events=events,
                    candidate_lanes_by_kind={"speech": list(range(6))},
                    work_dir=tmp,
                )

        written = ET.fromstring(final_xml["text"])
        lane_sequences = written.findall(f".//{q('TimelineTrack')}/{q('TrackSegment')}/{q('Sequence')}")
        lane0_components = list(lane_sequences[0].find(q("ComponentObjects")))
        lane6_components = list(lane_sequences[6].find(q("ComponentObjects")))

        self.assertEqual(
            [(node.tag.rsplit('}', 1)[-1], int(node.find(q("ComponentLength")).text)) for node in lane0_components[:5]],
            [
                ("Filler", 432000),
                ("OperationGroup", 195840),
                ("Transition", 21120),
                ("OperationGroup", 178560),
                ("OperationGroup", 186240),
            ],
        )
        self.assertEqual(
            [(node.tag.rsplit('}', 1)[-1], int(node.find(q("ComponentLength")).text)) for node in lane6_components],
            [("Filler", 4193280)],
        )

    def test_sdk_xml_writer_honors_authoritative_planned_raw_time(self) -> None:
        ns = "http://www.aafassociation.org/aafx/v1.1/20090617"
        q = lambda tag: f"{{{ns}}}{tag}"

        def component(tag: str, length: int) -> ET.Element:
            node = ET.Element(q(tag))
            ln = ET.SubElement(node, q("ComponentLength"))
            ln.text = str(int(length))
            dd = ET.SubElement(node, q("ComponentDataDefinition"))
            dd.text = "DataDef_Sound"
            return node

        def track(nodes: list[ET.Element]) -> ET.Element:
            tt = ET.Element(q("TimelineTrack"))
            seg = ET.SubElement(tt, q("TrackSegment"))
            seq = ET.SubElement(seg, q("Sequence"))
            ET.SubElement(seq, q("ComponentLength")).text = str(sum(int(n.find(q("ComponentLength")).text) for n in nodes))
            comps = ET.SubElement(seq, q("ComponentObjects"))
            for node in nodes:
                comps.append(node)
            return tt

        root = ET.Element(q("AAF"))
        comp = ET.SubElement(root, q("CompositionPackage"))
        usage = ET.SubElement(comp, q("PackageUsage"))
        usage.text = "Usage_TopLevel"
        tracks = ET.SubElement(comp, q("PackageTracks"))
        tracks.append(track([component("Filler", 1000)]))
        tracks.append(
            track(
                [
                    component("Filler", 300),
                    component("OperationGroup", 50),
                    component("Filler", 650),
                ]
            )
        )
        input_xml = ET.tostring(root, encoding="unicode")
        final_xml: dict[str, str] = {}

        def fake_to_xml(_tool: Path, _input: Path, xml_path: Path, **_kwargs) -> None:
            xml_path.write_text(input_xml, encoding="utf-8")

        def fake_to_aaf(_tool: Path, xml_path: Path, _output: Path, **_kwargs) -> None:
            final_xml["text"] = xml_path.read_text(encoding="utf-8")

        def fake_splice(_xml_text: str, *, new_comp_xml: str, composition_id=None) -> str:
            return new_comp_xml

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with (
                mock.patch("aaf_io.sdk_lane_layout.find_aaffmtconv", return_value=tmp / "aaffmtconv.exe"),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml", side_effect=fake_to_xml),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf", side_effect=fake_to_aaf),
                mock.patch("aaf_io.sdk_lane_layout.splice_composition_package", side_effect=fake_splice),
            ):
                apply_lane_layout_via_aaf_sdk_xml(
                    input_aaf=tmp / "in.aaf",
                    output_aaf=tmp / "out.aaf",
                    n_lanes=2,
                    events=[
                        LaneLayoutEvent(
                            src_lane=1,
                            top_idx=1,
                            T_edit=300,
                            L=50,
                            target_lane=0,
                            visible_t=100,
                            class_kind="noise",
                            raw_t_authoritative=True,
                        ),
                    ],
                    work_dir=tmp,
                )

        written = ET.fromstring(final_xml["text"])
        lane_sequences = written.findall(f".//{q('TimelineTrack')}/{q('TrackSegment')}/{q('Sequence')}")
        lane0_components = list(lane_sequences[0].find(q("ComponentObjects")))

        self.assertEqual(
            [(node.tag.rsplit('}', 1)[-1], int(node.find(q("ComponentLength")).text)) for node in lane0_components[:2]],
            [("Filler", 300), ("OperationGroup", 50)],
        )

    def test_sdk_xml_writer_leaves_immutable_wrapped_lane_untouched(self) -> None:
        ns = "http://www.aafassociation.org/aafx/v1.1/20090617"
        q = lambda tag: f"{{{ns}}}{tag}"

        def component(tag: str, length: int) -> ET.Element:
            node = ET.Element(q(tag))
            ln = ET.SubElement(node, q("ComponentLength"))
            ln.text = str(int(length))
            dd = ET.SubElement(node, q("ComponentDataDefinition"))
            dd.text = "DataDef_Sound"
            return node

        def plain_track(nodes: list[ET.Element]) -> ET.Element:
            tt = ET.Element(q("TimelineTrack"))
            seg = ET.SubElement(tt, q("TrackSegment"))
            seq = ET.SubElement(seg, q("Sequence"))
            ET.SubElement(seq, q("ComponentLength")).text = str(sum(int(n.find(q("ComponentLength")).text) for n in nodes))
            comps = ET.SubElement(seq, q("ComponentObjects"))
            for node in nodes:
                comps.append(node)
            return tt

        def wrapped_track(nodes: list[ET.Element]) -> ET.Element:
            tt = ET.Element(q("TimelineTrack"))
            seg = ET.SubElement(tt, q("TrackSegment"))
            wrapper = ET.SubElement(seg, q("OperationGroup"))
            ET.SubElement(wrapper, q("ComponentLength")).text = "250"
            inputs = ET.SubElement(wrapper, q("InputSegments"))
            seq = ET.SubElement(inputs, q("Sequence"))
            ET.SubElement(seq, q("ComponentLength")).text = "250"
            comps = ET.SubElement(seq, q("ComponentObjects"))
            for node in nodes:
                comps.append(node)
            return tt

        root = ET.Element(q("AAF"))
        comp = ET.SubElement(root, q("CompositionPackage"))
        usage = ET.SubElement(comp, q("PackageUsage"))
        usage.text = "Usage_TopLevel"
        tracks = ET.SubElement(comp, q("PackageTracks"))
        tracks.append(wrapped_track([component("Filler", 125), component("OperationGroup", 125)]))
        tracks.append(plain_track([component("Filler", 100), component("OperationGroup", 50), component("Filler", 100)]))
        input_xml = ET.tostring(root, encoding="unicode")
        final_xml: dict[str, str] = {}

        def fake_to_xml(_tool: Path, _input: Path, xml_path: Path, **_kwargs) -> None:
            xml_path.write_text(input_xml, encoding="utf-8")

        def fake_to_aaf(_tool: Path, xml_path: Path, _output: Path, **_kwargs) -> None:
            final_xml["text"] = xml_path.read_text(encoding="utf-8")

        def fake_splice(_xml_text: str, *, new_comp_xml: str, composition_id=None) -> str:
            return new_comp_xml

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with (
                mock.patch("aaf_io.sdk_lane_layout.find_aaffmtconv", return_value=tmp / "aaffmtconv.exe"),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml", side_effect=fake_to_xml),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf", side_effect=fake_to_aaf),
                mock.patch("aaf_io.sdk_lane_layout.splice_composition_package", side_effect=fake_splice),
            ):
                apply_lane_layout_via_aaf_sdk_xml(
                    input_aaf=tmp / "in.aaf",
                    output_aaf=tmp / "out.aaf",
                    n_lanes=2,
                    events=[
                        LaneLayoutEvent(
                            src_lane=1,
                            top_idx=1,
                            T_edit=0,
                            L=50,
                            target_lane=1,
                            visible_t=0,
                            class_kind="speech",
                        ),
                    ],
                    immutable_lanes={0},
                    work_dir=tmp,
                )

        written = ET.fromstring(final_xml["text"])
        wrapper = written.find(f".//{q('TimelineTrack')}/{q('TrackSegment')}/{q('OperationGroup')}")
        wrapped_seq = wrapper.find(f"{q('InputSegments')}/{q('Sequence')}")
        wrapped_components = list(wrapped_seq.find(q("ComponentObjects")))

        self.assertEqual(wrapper.find(q("ComponentLength")).text, "250")
        self.assertEqual(wrapped_seq.find(q("ComponentLength")).text, "250")
        self.assertEqual(
            [(node.tag.rsplit('}', 1)[-1], int(node.find(q("ComponentLength")).text)) for node in wrapped_components],
            [("Filler", 125), ("OperationGroup", 125)],
        )

    def test_sdk_xml_event_candidate_lanes_respect_class_policy(self) -> None:
        lanes = sdk_event_candidate_lanes(
            current_lane=4,
            n_lanes=9,
            class_kind="music",
            candidate_lanes_by_kind={"music": [8, 7, 6, 5, 4, 3]},
        )

        self.assertEqual(lanes, [4, 8, 7, 6, 5, 3])
        self.assertNotIn(0, lanes)
        self.assertNotIn(1, lanes)
        self.assertNotIn(2, lanes)

    def test_sdk_xml_event_candidate_lanes_do_not_prefer_current_outside_class_zone(self) -> None:
        lanes = sdk_event_candidate_lanes(
            current_lane=2,
            n_lanes=9,
            class_kind="music",
            candidate_lanes_by_kind={"music": [8, 7, 6, 5, 4, 3]},
        )

        self.assertEqual(lanes, [8, 7, 6, 5, 4, 3])
        self.assertNotIn(2, lanes)

    def test_sdk_xml_writer_serializes_events_at_requested_visible_time(self) -> None:
        ns = "http://www.aafassociation.org/aafx/v1.1/20090617"
        q = lambda tag: f"{{{ns}}}{tag}"

        def component(tag: str, length: int) -> ET.Element:
            node = ET.Element(q(tag))
            ln = ET.SubElement(node, q("ComponentLength"))
            ln.text = str(int(length))
            dd = ET.SubElement(node, q("ComponentDataDefinition"))
            dd.text = "DataDef_Sound"
            return node

        def track(children: list[ET.Element]) -> ET.Element:
            tt = ET.Element(q("TimelineTrack"))
            seg = ET.SubElement(tt, q("TrackSegment"))
            seq = ET.SubElement(seg, q("Sequence"))
            ln = ET.SubElement(seq, q("ComponentLength"))
            ln.text = "400"
            comps = ET.SubElement(seq, q("ComponentObjects"))
            for child in children:
                comps.append(child)
            return tt

        root = ET.Element(q("AAF"))
        comp = ET.SubElement(root, q("CompositionPackage"))
        usage = ET.SubElement(comp, q("PackageUsage"))
        usage.text = "Usage_TopLevel"
        tracks = ET.SubElement(comp, q("PackageTracks"))
        tracks.append(track([component("Filler", 200), component("OperationGroup", 20)]))
        tracks.append(track([component("Filler", 400)]))
        input_xml = ET.tostring(root, encoding="unicode")
        final_xml: dict[str, str] = {}

        def fake_to_xml(_tool: Path, _input: Path, xml_path: Path, **_kwargs) -> None:
            xml_path.write_text(input_xml, encoding="utf-8")

        def fake_to_aaf(_tool: Path, xml_path: Path, _output: Path, **_kwargs) -> None:
            final_xml["text"] = xml_path.read_text(encoding="utf-8")

        def fake_splice(_xml_text: str, *, new_comp_xml: str, composition_id=None) -> str:
            return new_comp_xml

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with (
                mock.patch("aaf_io.sdk_lane_layout.find_aaffmtconv", return_value=tmp / "aaffmtconv.exe"),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml", side_effect=fake_to_xml),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf", side_effect=fake_to_aaf),
                mock.patch("aaf_io.sdk_lane_layout.splice_composition_package", side_effect=fake_splice),
            ):
                apply_lane_layout_via_aaf_sdk_xml(
                    input_aaf=tmp / "in.aaf",
                    output_aaf=tmp / "out.aaf",
                    n_lanes=2,
                    candidate_lanes_by_kind={"speech": [1]},
                    events=[
                        LaneLayoutEvent(
                            src_lane=0,
                            top_idx=1,
                            T_edit=200,
                            L=20,
                            target_lane=1,
                            visible_t=180,
                            class_kind="speech",
                        )
                    ],
                    work_dir=tmp,
                )

        out_root = ET.fromstring(final_xml["text"])
        out_tracks = out_root.findall(".//" + q("TimelineTrack"))
        lane1_components = list(out_tracks[1].find(".//" + q("ComponentObjects")))

        self.assertEqual(lane1_components[0].tag, q("Filler"))
        self.assertEqual(lane1_components[0].find(q("ComponentLength")).text, "180")
        self.assertEqual(lane1_components[1].tag, q("OperationGroup"))

    def test_sdk_xml_writer_compaction_keeps_visible_time_compatible_lane(self) -> None:
        ns = "http://www.aafassociation.org/aafx/v1.1/20090617"
        q = lambda tag: f"{{{ns}}}{tag}"

        def component(tag: str, length: int) -> ET.Element:
            node = ET.Element(q(tag))
            ln = ET.SubElement(node, q("ComponentLength"))
            ln.text = str(int(length))
            dd = ET.SubElement(node, q("ComponentDataDefinition"))
            dd.text = "DataDef_Sound"
            return node

        def track(children: list[ET.Element]) -> ET.Element:
            tt = ET.Element(q("TimelineTrack"))
            seg = ET.SubElement(tt, q("TrackSegment"))
            seq = ET.SubElement(seg, q("Sequence"))
            ln = ET.SubElement(seq, q("ComponentLength"))
            ln.text = "400"
            comps = ET.SubElement(seq, q("ComponentObjects"))
            for child in children:
                comps.append(child)
            return tt

        root = ET.Element(q("AAF"))
        comp = ET.SubElement(root, q("CompositionPackage"))
        usage = ET.SubElement(comp, q("PackageUsage"))
        usage.text = "Usage_TopLevel"
        tracks = ET.SubElement(comp, q("PackageTracks"))
        tracks.append(track([component("Filler", 200), component("OperationGroup", 20)]))
        tracks.append(track([component("Filler", 400)]))
        tracks.append(
            track(
                [
                    component("Filler", 100),
                    component("Transition", 10),
                    component("Filler", 290),
                ]
            )
        )
        input_xml = ET.tostring(root, encoding="unicode")
        final_xml: dict[str, str] = {}

        def fake_to_xml(_tool: Path, _input: Path, xml_path: Path, **_kwargs) -> None:
            xml_path.write_text(input_xml, encoding="utf-8")

        def fake_to_aaf(_tool: Path, xml_path: Path, _output: Path, **_kwargs) -> None:
            final_xml["text"] = xml_path.read_text(encoding="utf-8")

        def fake_splice(_xml_text: str, *, new_comp_xml: str, composition_id=None) -> str:
            return new_comp_xml

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with (
                mock.patch("aaf_io.sdk_lane_layout.find_aaffmtconv", return_value=tmp / "aaffmtconv.exe"),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml", side_effect=fake_to_xml),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf", side_effect=fake_to_aaf),
                mock.patch("aaf_io.sdk_lane_layout.splice_composition_package", side_effect=fake_splice),
            ):
                apply_lane_layout_via_aaf_sdk_xml(
                    input_aaf=tmp / "in.aaf",
                    output_aaf=tmp / "out.aaf",
                    n_lanes=3,
                    candidate_lanes_by_kind={"speech": [1, 2]},
                    events=[
                        LaneLayoutEvent(
                            src_lane=2,
                            top_idx=1,
                            T_edit=100,
                            L=10,
                            target_lane=2,
                            kind="transition",
                        ),
                        LaneLayoutEvent(
                            src_lane=0,
                            top_idx=1,
                            T_edit=200,
                            L=20,
                            target_lane=2,
                            visible_t=180,
                            class_kind="speech",
                        ),
                    ],
                    work_dir=tmp,
                )

        out_root = ET.fromstring(final_xml["text"])
        out_tracks = out_root.findall(".//" + q("TimelineTrack"))
        lane1_tags = [
            node.tag for node in list(out_tracks[1].find(".//" + q("ComponentObjects")))
        ]
        lane2_components = list(out_tracks[2].find(".//" + q("ComponentObjects")))
        lane2_tags = [node.tag for node in lane2_components]

        self.assertNotIn(q("OperationGroup"), lane1_tags)
        self.assertEqual(
            lane2_tags[:4],
            [q("Filler"), q("Transition"), q("Filler"), q("OperationGroup")],
        )
        self.assertEqual(lane2_components[2].find(q("ComponentLength")).text, "90")

    def test_sdk_xml_writer_serializes_compacted_middle_group_into_upper_gap(self) -> None:
        ns = "http://www.aafassociation.org/aafx/v1.1/20090617"
        q = lambda tag: f"{{{ns}}}{tag}"

        def component(tag: str, length: int) -> ET.Element:
            node = ET.Element(q(tag))
            ln = ET.SubElement(node, q("ComponentLength"))
            ln.text = str(int(length))
            dd = ET.SubElement(node, q("ComponentDataDefinition"))
            dd.text = "DataDef_Sound"
            return node

        def track(children: list[ET.Element]) -> ET.Element:
            tt = ET.Element(q("TimelineTrack"))
            seg = ET.SubElement(tt, q("TrackSegment"))
            seq = ET.SubElement(seg, q("Sequence"))
            ln = ET.SubElement(seq, q("ComponentLength"))
            ln.text = "220"
            comps = ET.SubElement(seq, q("ComponentObjects"))
            for child in children:
                comps.append(child)
            return tt

        root = ET.Element(q("AAF"))
        comp = ET.SubElement(root, q("CompositionPackage"))
        usage = ET.SubElement(comp, q("PackageUsage"))
        usage.text = "Usage_TopLevel"
        tracks = ET.SubElement(comp, q("PackageTracks"))
        for _lane in range(4):
            tracks.append(
                track(
                    [
                        component("Filler", 100),
                        component("OperationGroup", 20),
                        component("Filler", 35),
                        component("OperationGroup", 20),
                    ]
                )
            )
        for _lane in range(4):
            tracks.append(track([component("Filler", 130), component("OperationGroup", 15)]))
        input_xml = ET.tostring(root, encoding="unicode")
        final_xml: dict[str, str] = {}

        def fake_to_xml(_tool: Path, _input: Path, xml_path: Path, **_kwargs) -> None:
            xml_path.write_text(input_xml, encoding="utf-8")

        def fake_to_aaf(_tool: Path, xml_path: Path, _output: Path, **_kwargs) -> None:
            final_xml["text"] = xml_path.read_text(encoding="utf-8")

        def fake_splice(_xml_text: str, *, new_comp_xml: str, composition_id=None) -> str:
            return new_comp_xml

        events: list[LaneLayoutEvent] = []
        for lane in range(4):
            events.extend(
                [
                    LaneLayoutEvent(
                        src_lane=lane,
                        top_idx=1,
                        T_edit=100,
                        L=20,
                        target_lane=lane,
                        visible_t=100,
                        class_kind="speech",
                    ),
                    LaneLayoutEvent(
                        src_lane=lane + 4,
                        top_idx=1,
                        T_edit=130,
                        L=15,
                        target_lane=lane,
                        visible_t=130,
                        class_kind="speech",
                    ),
                    LaneLayoutEvent(
                        src_lane=lane,
                        top_idx=3,
                        T_edit=155,
                        L=20,
                        target_lane=lane,
                        visible_t=155,
                        class_kind="speech",
                    ),
                ]
            )

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with (
                mock.patch("aaf_io.sdk_lane_layout.find_aaffmtconv", return_value=tmp / "aaffmtconv.exe"),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml", side_effect=fake_to_xml),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf", side_effect=fake_to_aaf),
                mock.patch("aaf_io.sdk_lane_layout.splice_composition_package", side_effect=fake_splice),
            ):
                apply_lane_layout_via_aaf_sdk_xml(
                    input_aaf=tmp / "in.aaf",
                    output_aaf=tmp / "out.aaf",
                    n_lanes=8,
                    candidate_lanes_by_kind={"speech": list(range(8))},
                    events=events,
                    work_dir=tmp,
                )

        out_root = ET.fromstring(final_xml["text"])
        out_tracks = out_root.findall(".//" + q("TimelineTrack"))
        for lane in range(4):
            components = list(out_tracks[lane].find(".//" + q("ComponentObjects")))
            self.assertEqual(
                [node.tag for node in components[:6]],
                [
                    q("Filler"),
                    q("OperationGroup"),
                    q("Filler"),
                    q("OperationGroup"),
                    q("Filler"),
                    q("OperationGroup"),
                ],
            )
            self.assertEqual(components[0].find(q("ComponentLength")).text, "100")
            self.assertEqual(components[2].find(q("ComponentLength")).text, "10")
            self.assertEqual(components[4].find(q("ComponentLength")).text, "10")

    def test_sdk_xml_writer_rejects_visible_start_overlap_after_moved_transition(self) -> None:
        ns = "http://www.aafassociation.org/aafx/v1.1/20090617"
        q = lambda tag: f"{{{ns}}}{tag}"

        def component(tag: str, length: int) -> ET.Element:
            node = ET.Element(q(tag))
            ln = ET.SubElement(node, q("ComponentLength"))
            ln.text = str(int(length))
            dd = ET.SubElement(node, q("ComponentDataDefinition"))
            dd.text = "DataDef_Sound"
            return node

        def track(children: list[ET.Element]) -> ET.Element:
            tt = ET.Element(q("TimelineTrack"))
            seg = ET.SubElement(tt, q("TrackSegment"))
            seq = ET.SubElement(seg, q("Sequence"))
            ln = ET.SubElement(seq, q("ComponentLength"))
            ln.text = "400"
            comps = ET.SubElement(seq, q("ComponentObjects"))
            for child in children:
                comps.append(child)
            return tt

        root = ET.Element(q("AAF"))
        comp = ET.SubElement(root, q("CompositionPackage"))
        usage = ET.SubElement(comp, q("PackageUsage"))
        usage.text = "Usage_TopLevel"
        tracks = ET.SubElement(comp, q("PackageTracks"))
        tracks.append(
            track(
                [
                    component("Filler", 100),
                    component("OperationGroup", 10),
                    component("Transition", 5),
                    component("OperationGroup", 2),
                ]
            )
        )
        input_xml = ET.tostring(root, encoding="unicode")
        final_xml: dict[str, str] = {}

        def fake_to_xml(_tool: Path, _input: Path, xml_path: Path, **_kwargs) -> None:
            xml_path.write_text(input_xml, encoding="utf-8")

        def fake_to_aaf(_tool: Path, xml_path: Path, _output: Path, **_kwargs) -> None:
            final_xml["text"] = xml_path.read_text(encoding="utf-8")

        def fake_splice(_xml_text: str, *, new_comp_xml: str, composition_id=None) -> str:
            return new_comp_xml

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with (
                mock.patch("aaf_io.sdk_lane_layout.find_aaffmtconv", return_value=tmp / "aaffmtconv.exe"),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml", side_effect=fake_to_xml),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf", side_effect=fake_to_aaf),
                mock.patch("aaf_io.sdk_lane_layout.splice_composition_package", side_effect=fake_splice),
            ):
                with self.assertRaisesRegex(RuntimeError, "layout underflow"):
                    apply_lane_layout_via_aaf_sdk_xml(
                        input_aaf=tmp / "in.aaf",
                        output_aaf=tmp / "out.aaf",
                        n_lanes=1,
                        events=[
                            LaneLayoutEvent(
                                src_lane=0,
                                top_idx=1,
                                T_edit=100,
                                L=10,
                                target_lane=0,
                                post_top_idx=2,
                                post_len=5,
                            ),
                            LaneLayoutEvent(
                                src_lane=0,
                                top_idx=3,
                                T_edit=115,
                                L=2,
                                target_lane=0,
                                visible_t=104,
                                class_kind="noise",
                            ),
                        ],
                        work_dir=tmp,
                    )

    def test_sdk_xml_writer_repairs_transition_underflow_to_earlier_allowed_lane(self) -> None:
        ns = "http://www.aafassociation.org/aafx/v1.1/20090617"
        q = lambda tag: f"{{{ns}}}{tag}"

        def component(tag: str, length: int) -> ET.Element:
            node = ET.Element(q(tag))
            ln = ET.SubElement(node, q("ComponentLength"))
            ln.text = str(int(length))
            dd = ET.SubElement(node, q("ComponentDataDefinition"))
            dd.text = "DataDef_Sound"
            return node

        def track(children: list[ET.Element]) -> ET.Element:
            tt = ET.Element(q("TimelineTrack"))
            seg = ET.SubElement(tt, q("TrackSegment"))
            seq = ET.SubElement(seg, q("Sequence"))
            ln = ET.SubElement(seq, q("ComponentLength"))
            ln.text = "1000"
            comps = ET.SubElement(seq, q("ComponentObjects"))
            for child in children:
                comps.append(child)
            return tt

        root = ET.Element(q("AAF"))
        comp = ET.SubElement(root, q("CompositionPackage"))
        usage = ET.SubElement(comp, q("PackageUsage"))
        usage.text = "Usage_TopLevel"
        tracks = ET.SubElement(comp, q("PackageTracks"))
        tracks.append(track([component("Filler", 1000)]))
        tracks.append(
            track(
                [
                    component("Filler", 100),
                    component("OperationGroup", 100),
                    component("Transition", 40),
                    component("OperationGroup", 20),
                ]
            )
        )
        tracks.append(track([component("OperationGroup", 1000)]))
        input_xml = ET.tostring(root, encoding="unicode")
        final_xml: dict[str, str] = {}

        def fake_to_xml(_tool: Path, _input: Path, xml_path: Path, **_kwargs) -> None:
            xml_path.write_text(input_xml, encoding="utf-8")

        def fake_to_aaf(_tool: Path, xml_path: Path, _output: Path, **_kwargs) -> None:
            final_xml["text"] = xml_path.read_text(encoding="utf-8")

        def fake_splice(_xml_text: str, *, new_comp_xml: str, composition_id=None) -> str:
            return new_comp_xml

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with (
                mock.patch("aaf_io.sdk_lane_layout.find_aaffmtconv", return_value=tmp / "aaffmtconv.exe"),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml", side_effect=fake_to_xml),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf", side_effect=fake_to_aaf),
                mock.patch("aaf_io.sdk_lane_layout.splice_composition_package", side_effect=fake_splice),
            ):
                apply_lane_layout_via_aaf_sdk_xml(
                    input_aaf=tmp / "in.aaf",
                    output_aaf=tmp / "out.aaf",
                    n_lanes=3,
                    events=[
                        LaneLayoutEvent(
                            src_lane=1,
                            top_idx=1,
                            T_edit=100,
                            L=100,
                            target_lane=1,
                            visible_t=100,
                            class_kind="noise",
                        ),
                        LaneLayoutEvent(
                            src_lane=1,
                            top_idx=3,
                            T_edit=180,
                            L=20,
                            target_lane=1,
                            visible_t=110,
                            preferred_transition_offset=40,
                            pre_top_idx=2,
                            pre_len=40,
                            class_kind="noise",
                        ),
                        LaneLayoutEvent(
                            src_lane=2,
                            top_idx=0,
                            T_edit=0,
                            L=1000,
                            target_lane=2,
                            visible_t=0,
                            class_kind="noise",
                        ),
                    ],
                    candidate_lanes_by_kind={"noise": [1, 2, 0]},
                    work_dir=tmp,
                )

        out_root = ET.fromstring(final_xml["text"])
        out_tracks = out_root.findall(".//" + q("TimelineTrack"))
        lane0_components = list(out_tracks[0].find(".//" + q("ComponentObjects")))

        self.assertEqual(
            [(node.tag, node.find(q("ComponentLength")).text) for node in lane0_components[:3]],
            [
                (q("Filler"), "150"),
                (q("Transition"), "40"),
                (q("OperationGroup"), "20"),
            ],
        )

    def test_sdk_xml_writer_preserves_visible_start_after_moved_transition(self) -> None:
        ns = "http://www.aafassociation.org/aafx/v1.1/20090617"
        q = lambda tag: f"{{{ns}}}{tag}"

        def component(tag: str, length: int) -> ET.Element:
            node = ET.Element(q(tag))
            ln = ET.SubElement(node, q("ComponentLength"))
            ln.text = str(int(length))
            dd = ET.SubElement(node, q("ComponentDataDefinition"))
            dd.text = "DataDef_Sound"
            return node

        def track(children: list[ET.Element]) -> ET.Element:
            tt = ET.Element(q("TimelineTrack"))
            seg = ET.SubElement(tt, q("TrackSegment"))
            seq = ET.SubElement(seg, q("Sequence"))
            ln = ET.SubElement(seq, q("ComponentLength"))
            ln.text = "400"
            comps = ET.SubElement(seq, q("ComponentObjects"))
            for child in children:
                comps.append(child)
            return tt

        root = ET.Element(q("AAF"))
        comp = ET.SubElement(root, q("CompositionPackage"))
        usage = ET.SubElement(comp, q("PackageUsage"))
        usage.text = "Usage_TopLevel"
        tracks = ET.SubElement(comp, q("PackageTracks"))
        tracks.append(
            track(
                [
                    component("Filler", 100),
                    component("OperationGroup", 10),
                    component("Transition", 5),
                    component("Filler", 85),
                    component("OperationGroup", 2),
                ]
            )
        )
        input_xml = ET.tostring(root, encoding="unicode")
        final_xml: dict[str, str] = {}

        def fake_to_xml(_tool: Path, _input: Path, xml_path: Path, **_kwargs) -> None:
            xml_path.write_text(input_xml, encoding="utf-8")

        def fake_to_aaf(_tool: Path, xml_path: Path, _output: Path, **_kwargs) -> None:
            final_xml["text"] = xml_path.read_text(encoding="utf-8")

        def fake_splice(_xml_text: str, *, new_comp_xml: str, composition_id=None) -> str:
            return new_comp_xml

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with (
                mock.patch("aaf_io.sdk_lane_layout.find_aaffmtconv", return_value=tmp / "aaffmtconv.exe"),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml", side_effect=fake_to_xml),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf", side_effect=fake_to_aaf),
                mock.patch("aaf_io.sdk_lane_layout.splice_composition_package", side_effect=fake_splice),
            ):
                apply_lane_layout_via_aaf_sdk_xml(
                    input_aaf=tmp / "in.aaf",
                    output_aaf=tmp / "out.aaf",
                    n_lanes=1,
                    events=[
                        LaneLayoutEvent(
                            src_lane=0,
                            top_idx=1,
                            T_edit=100,
                            L=10,
                            target_lane=0,
                            post_top_idx=2,
                            post_len=5,
                        ),
                        LaneLayoutEvent(
                            src_lane=0,
                            top_idx=4,
                            T_edit=200,
                            L=2,
                            target_lane=0,
                            visible_t=200,
                            class_kind="speech",
                        ),
                    ],
                    work_dir=tmp,
                )

        out_root = ET.fromstring(final_xml["text"])
        out_track = out_root.find(".//" + q("TimelineTrack"))
        assert out_track is not None
        out_components = list(out_track.find(".//" + q("ComponentObjects")))

        self.assertEqual(
            [node.tag for node in out_components[:5]],
            [q("Filler"), q("OperationGroup"), q("Transition"), q("Filler"), q("OperationGroup")],
        )
        self.assertEqual(out_components[3].find(q("ComponentLength")).text, "95")

    def test_sdk_xml_writer_does_not_count_future_transition_when_preserving_visible_start(self) -> None:
        ns = "http://www.aafassociation.org/aafx/v1.1/20090617"
        q = lambda tag: f"{{{ns}}}{tag}"

        def component(tag: str, length: int) -> ET.Element:
            node = ET.Element(q(tag))
            ln = ET.SubElement(node, q("ComponentLength"))
            ln.text = str(int(length))
            dd = ET.SubElement(node, q("ComponentDataDefinition"))
            dd.text = "DataDef_Sound"
            return node

        def track(children: list[ET.Element]) -> ET.Element:
            tt = ET.Element(q("TimelineTrack"))
            seg = ET.SubElement(tt, q("TrackSegment"))
            seq = ET.SubElement(seg, q("Sequence"))
            ln = ET.SubElement(seq, q("ComponentLength"))
            ln.text = "500"
            comps = ET.SubElement(seq, q("ComponentObjects"))
            for child in children:
                comps.append(child)
            return tt

        root = ET.Element(q("AAF"))
        comp = ET.SubElement(root, q("CompositionPackage"))
        usage = ET.SubElement(comp, q("PackageUsage"))
        usage.text = "Usage_TopLevel"
        tracks = ET.SubElement(comp, q("PackageTracks"))
        tracks.append(
            track(
                [
                    component("Filler", 100),
                    component("OperationGroup", 10),
                    component("Transition", 5),
                    component("Filler", 85),
                    component("OperationGroup", 2),
                    component("Transition", 7),
                    component("Filler", 91),
                    component("OperationGroup", 3),
                ]
            )
        )
        input_xml = ET.tostring(root, encoding="unicode")
        final_xml: dict[str, str] = {}

        def fake_to_xml(_tool: Path, _input: Path, xml_path: Path, **_kwargs) -> None:
            xml_path.write_text(input_xml, encoding="utf-8")

        def fake_to_aaf(_tool: Path, xml_path: Path, _output: Path, **_kwargs) -> None:
            final_xml["text"] = xml_path.read_text(encoding="utf-8")

        def fake_splice(_xml_text: str, *, new_comp_xml: str, composition_id=None) -> str:
            return new_comp_xml

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with (
                mock.patch("aaf_io.sdk_lane_layout.find_aaffmtconv", return_value=tmp / "aaffmtconv.exe"),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml", side_effect=fake_to_xml),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf", side_effect=fake_to_aaf),
                mock.patch("aaf_io.sdk_lane_layout.splice_composition_package", side_effect=fake_splice),
            ):
                apply_lane_layout_via_aaf_sdk_xml(
                    input_aaf=tmp / "in.aaf",
                    output_aaf=tmp / "out.aaf",
                    n_lanes=1,
                    events=[
                        LaneLayoutEvent(
                            src_lane=0,
                            top_idx=1,
                            T_edit=100,
                            L=10,
                            target_lane=0,
                            post_top_idx=2,
                            post_len=5,
                        ),
                        LaneLayoutEvent(
                            src_lane=0,
                            top_idx=4,
                            T_edit=200,
                            L=2,
                            target_lane=0,
                            visible_t=200,
                            class_kind="speech",
                        ),
                        LaneLayoutEvent(
                            src_lane=0,
                            top_idx=7,
                            T_edit=209,
                            L=3,
                            target_lane=0,
                            visible_t=300,
                            pre_top_idx=5,
                            pre_len=7,
                            class_kind="speech",
                        ),
                    ],
                    work_dir=tmp,
                )

        out_root = ET.fromstring(final_xml["text"])
        out_track = out_root.find(".//" + q("TimelineTrack"))
        assert out_track is not None
        out_components = list(out_track.find(".//" + q("ComponentObjects")))

        self.assertEqual(
            [(node.tag, node.find(q("ComponentLength")).text) for node in out_components[:5]],
            [
                (q("Filler"), "100"),
                (q("OperationGroup"), "10"),
                (q("Transition"), "5"),
                (q("Filler"), "95"),
                (q("OperationGroup"), "2"),
            ],
        )

    def test_sdk_xml_writer_pads_formerly_empty_target_lane_to_composition_extent(self) -> None:
        ns = "http://www.aafassociation.org/aafx/v1.1/20090617"
        q = lambda tag: f"{{{ns}}}{tag}"

        def component(tag: str, length: int) -> ET.Element:
            node = ET.Element(q(tag))
            ln = ET.SubElement(node, q("ComponentLength"))
            ln.text = str(int(length))
            dd = ET.SubElement(node, q("ComponentDataDefinition"))
            dd.text = "DataDef_Sound"
            return node

        def track(children: list[ET.Element], declared_len: int) -> ET.Element:
            tt = ET.Element(q("TimelineTrack"))
            seg = ET.SubElement(tt, q("TrackSegment"))
            seq = ET.SubElement(seg, q("Sequence"))
            ln = ET.SubElement(seq, q("ComponentLength"))
            ln.text = str(int(declared_len))
            comps = ET.SubElement(seq, q("ComponentObjects"))
            for child in children:
                comps.append(child)
            return tt

        root = ET.Element(q("AAF"))
        comp = ET.SubElement(root, q("CompositionPackage"))
        usage = ET.SubElement(comp, q("PackageUsage"))
        usage.text = "Usage_TopLevel"
        tracks = ET.SubElement(comp, q("PackageTracks"))
        tracks.append(track([component("Filler", 0)], 0))
        tracks.append(track([component("Filler", 100), component("OperationGroup", 20)], 400))
        input_xml = ET.tostring(root, encoding="unicode")
        final_xml: dict[str, str] = {}

        def fake_to_xml(_tool: Path, _input: Path, xml_path: Path, **_kwargs) -> None:
            xml_path.write_text(input_xml, encoding="utf-8")

        def fake_to_aaf(_tool: Path, xml_path: Path, _output: Path, **_kwargs) -> None:
            final_xml["text"] = xml_path.read_text(encoding="utf-8")

        def fake_splice(_xml_text: str, *, new_comp_xml: str, composition_id=None) -> str:
            return new_comp_xml

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with (
                mock.patch("aaf_io.sdk_lane_layout.find_aaffmtconv", return_value=tmp / "aaffmtconv.exe"),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml", side_effect=fake_to_xml),
                mock.patch("aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf", side_effect=fake_to_aaf),
                mock.patch("aaf_io.sdk_lane_layout.splice_composition_package", side_effect=fake_splice),
            ):
                apply_lane_layout_via_aaf_sdk_xml(
                    input_aaf=tmp / "in.aaf",
                    output_aaf=tmp / "out.aaf",
                    n_lanes=2,
                    events=[
                        LaneLayoutEvent(
                            src_lane=1,
                            top_idx=1,
                            T_edit=100,
                            L=20,
                            target_lane=0,
                            visible_t=100,
                            class_kind="speech",
                        )
                    ],
                    work_dir=tmp,
                )

        out_root = ET.fromstring(final_xml["text"])
        out_tracks = out_root.findall(".//" + q("TimelineTrack"))
        lane0_seq = out_tracks[0].find(".//" + q("Sequence"))
        assert lane0_seq is not None
        lane0_components = list(lane0_seq.find(".//" + q("ComponentObjects")))

        self.assertEqual(lane0_seq.find(q("ComponentLength")).text, "400")
        self.assertEqual(
            [node.tag for node in lane0_components],
            [q("Filler"), q("OperationGroup"), q("Filler")],
        )
        self.assertEqual(lane0_components[-1].find(q("ComponentLength")).text, "280")

    def test_replaces_inner_gain_operation_without_breaking_outer_pan_operation(self) -> None:
        xml = """<Root>
          <CompositionPackage>
            <TrackSegment>
              <OperationGroup>
                <Parameters><ConstantValue /></Parameters>
                <InputSegments>
                  <Sequence>
                    <ComponentObjects>
                      <OperationGroup>
                        <Parameters><ConstantValue /></Parameters>
                        <InputSegments>
                          <SourceClip>
                            <StartPosition>22</StartPosition>
                            <SourceTrackID>1</SourceTrackID>
                            <SourcePackageID>mob-a</SourcePackageID>
                            <ComponentLength>2</ComponentLength>
                            <ComponentDataDefinition>DataDef_Sound</ComponentDataDefinition>
                          </SourceClip>
                        </InputSegments>
                        <Operation>OperationDef_MonoAudioGain</Operation>
                        <ComponentLength>2</ComponentLength>
                        <ComponentDataDefinition>DataDef_Sound</ComponentDataDefinition>
                      </OperationGroup>
                    </ComponentObjects>
                    <ComponentLength>2</ComponentLength>
                    <ComponentDataDefinition>DataDef_Sound</ComponentDataDefinition>
                  </Sequence>
                </InputSegments>
                <Operation>OperationDef_MonoAudioPan</Operation>
                <ComponentLength>2</ComponentLength>
                <ComponentDataDefinition>DataDef_Sound</ComponentDataDefinition>
              </OperationGroup>
            </TrackSegment>
          </CompositionPackage>
        </Root>"""

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "work.xml"
            path.write_text(xml, encoding="utf-8")

            replaced = aaf_sdk_xml.apply_removals_in_composition_xml(path, {("mob-a", 2, 22, 1)})
            edited = path.read_text(encoding="utf-8")

        self.assertEqual(replaced, 1)
        ET.fromstring(edited)
        self.assertIn("<Operation>OperationDef_MonoAudioPan</Operation>", edited)
        self.assertIn("<Operation>OperationDef_MonoAudioGain</Operation>", edited)
        self.assertNotIn("<SourceClip>", edited)
        self.assertIn("<Filler>", edited)


if __name__ == "__main__":
    unittest.main()
