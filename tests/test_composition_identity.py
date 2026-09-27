from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET
import aaf2

from aaf_speech_filter.aaf_yamnet_lane_layout import _pick_main_composition
from aaf_io.sdk_lane_layout import apply_lane_layout_via_aaf_sdk_xml, LaneLayoutEvent
from tests.test_audit_layout import q, component, xml_document


class CompositionIdentityTests(unittest.TestCase):
    def test_top_level_usage_wins_over_order_and_track_count(self):
        with tempfile.TemporaryDirectory() as td, aaf2.open(str(Path(td)/'in.aaf'), 'w') as f:
            comps = []
            for usage, count in (('Usage_LowerLevel', 5), ('Usage_TopLevel', 3)):
                comp = f.create.CompositionMob()
                comp.usage = usage
                f.content.mobs.append(comp)
                for _ in range(count):
                    comp.create_timeline_slot(48000).segment = f.create.Sequence('sound')
                comps.append(comp)
            self.assertEqual(_pick_main_composition(comps).mob_id, comps[1].mob_id)

    def test_ambiguous_top_level_is_rejected(self):
        with tempfile.TemporaryDirectory() as td, aaf2.open(str(Path(td)/'in.aaf'), 'w') as f:
            comps = [f.create.CompositionMob() for _ in range(2)]
            for comp in comps:
                comp.usage = 'Usage_TopLevel'
            with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
                _pick_main_composition(comps)

    def test_sdk_writer_uses_requested_composition_identity(self):
        document = ET.Element(q('AAF'))
        for identity, usage in (('nested-id', 'Usage_LowerLevel'), ('main-id', 'Usage_TopLevel')):
            package = ET.fromstring(xml_document([[component('Filler', 100)], [component('SourceClip', 10, identity), component('Filler', 90)]])).find(q('CompositionPackage'))
            package.find(q('PackageUsage')).text = usage
            ET.SubElement(package, q('PackageID')).text = identity
            document.append(package)
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            written = []
            def export(tool, source, destination, **kwargs):
                destination.write_text(ET.tostring(document, encoding='unicode'), encoding='utf8')
            def build(tool, source, destination, **kwargs):
                written.append(ET.fromstring(source.read_text(encoding='utf8')))
            with patch('aaf_io.sdk_lane_layout.find_aaffmtconv', return_value=folder/'tool'), patch('aaf_io.sdk_lane_layout.run_aaffmtconv_to_xml', side_effect=export), patch('aaf_io.sdk_lane_layout.run_aaffmtconv_to_aaf', side_effect=build):
                args = dict(input_aaf=folder/'in.aaf', output_aaf=folder/'out.aaf', events=[LaneLayoutEvent(1,0,0,10,0)], n_lanes=2, work_dir=folder)
                apply_lane_layout_via_aaf_sdk_xml(**args, composition_id='nested-id')
                with self.assertRaisesRegex(RuntimeError, 'identity'):
                    apply_lane_layout_via_aaf_sdk_xml(**args, composition_id='missing-id')
            packages = written[0].findall(q('CompositionPackage'))
            self.assertIsNone(packages[0].find(q('PackageTracks'))[1].find('.//' + q('SourceClip')))
            self.assertIsNotNone(packages[1].find(q('PackageTracks'))[1].find('.//' + q('SourceClip')))
