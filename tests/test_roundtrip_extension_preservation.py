from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aaf_io.roundtrip import RoundtripMethod, sdk_roundtrip

ROOT = Path(__file__).resolve().parents[1]

# Extension properties are opaque to the roundtrip, even when they affect render.
EXTENSION_XML = b'''<?xml version="1.0" encoding="utf-8"?>\r
<AAF xmlns="http://www.aafassociation.org/aafx/v1.1/20090617" xmlns:this="urn:vendor:extension">\r
  <Transition><ComponentAttributeList>\r
    <TaggedValue><Name>_ATN_AUDIO_DISSOLVE_CURVETYPE</Name><Value>2</Value></TaggedValue>\r
  </ComponentAttributeList><this:RenderExtension>opaque</this:RenderExtension></Transition>\r
  <ControlPoint><this:ControlPointSource>2</this:ControlPointSource></ControlPoint>\r
  <this:ComponentAttributeList><TaggedValue><Name>_ATN_AUDIO_DISSOLVE_CURVETYPE</Name><Value>2</Value></TaggedValue></this:ComponentAttributeList>\r
  <EssenceDescription> </EssenceDescription>\r
</AAF>\r
'''


class RoundtripExtensionPreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / 'source.aaf'
        self.output = self.folder / 'result.aaf'
        self.original = b'original opaque AAF bytes'
        self.source.write_bytes(self.original)
        self.output.write_bytes(b'previous completed output')

    def run_roundtrip(self, *, reject_extensions=False, **options):
        received = []

        def export(executable, source, xml, **kwargs):
            Path(xml).write_bytes(EXTENSION_XML)

        def rebuild(executable, xml, destination, **kwargs):
            received.append(Path(xml).read_bytes())
            if reject_extensions:
                raise RuntimeError('SDK cannot reconstruct extension property')
            Path(destination).write_bytes(b'rebuilt AAF preserving all XML properties')

        with patch('aaf_io.roundtrip.find_aaffmtconv', return_value=Path('sdk.exe')), \
             patch('aaf_io.roundtrip.run_aaffmtconv_to_xml', side_effect=export), \
             patch('aaf_io.roundtrip.run_aaffmtconv_to_aaf', side_effect=rebuild):
            result = sdk_roundtrip(self.source, self.output, **options)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(received, [EXTENSION_XML])
        return result

    def test_default_roundtrip_preserves_raw_extensions_and_empty_descriptor(self):
        result = self.run_roundtrip()
        self.assertTrue(result.ok)
        self.assertEqual(result.method, RoundtripMethod.sdk_xml_raw)
        self.assertEqual(self.output.read_bytes(), b'rebuilt AAF preserving all XML properties')

    def test_legacy_strip_hint_never_discards_opaque_metadata(self):
        for requested in (True, False):
            with self.subTest(strip_this_namespace=requested):
                result = self.run_roundtrip(strip_this_namespace=requested)
                self.assertEqual(result.method, RoundtripMethod.sdk_xml_raw)

    def test_unsupported_extensions_fall_back_to_identical_input(self):
        result = self.run_roundtrip(reject_extensions=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.method, RoundtripMethod.binary_copy)
        self.assertEqual(self.output.read_bytes(), self.original)

    def test_unsupported_extensions_without_copy_preserve_previous_output(self):
        for policy in ('raise', 'error'):
            with self.subTest(on_xml_crash=policy):
                result = self.run_roundtrip(reject_extensions=True, on_xml_crash=policy)
                self.assertFalse(result.ok)
                self.assertEqual(self.output.read_bytes(), b'previous completed output')


if __name__ == '__main__':
    unittest.main()
