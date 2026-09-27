"""Standard gain identities are specified independently of production constants."""
import tempfile
import unittest
from pathlib import Path

import aaf2
from tests.test_audit_audio import add_source, add_comp, clip, write_wave
from aaf_speech_filter.timeline_sourceclips import iter_timeline_sourceclips
from aaf_speech_filter.sdk_removals import collect_timeline_sourceclip_removals_for_sdk_xml
from aaf_speech_filter.pyaaf2_filter import filter_aaf_speech_only
from aaf_speech_filter.config import FilterConfig


class StandardGainTests(unittest.TestCase):
    def check_parameter(self, parameter_id, label, expected_gain, expected_removed):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            media, source, output = root/'media.wav', root/'input.aaf', root/'output.aaf'
            write_wave(media, b'\x00\x40' * 4800)
            with aaf2.open(str(source), 'w') as aaf:
                mob, comp = add_source(aaf, media), add_comp(aaf)
                op = aaf.create.OperationDef('9d2ea894-0968-11d3-8a38-0050040ef7d2', 'renamed gain')
                op.media_kind = 'sound'
                op.number_inputs = 1
                op['IsTimeWarp'].value = False
                parameter = aaf.create.ParameterDef(parameter_id, label, typedef='Rational')
                aaf.dictionary.register_def(parameter)
                op.parameters.append(parameter)
                aaf.dictionary.register_def(op)
                group = aaf.create.OperationGroup(op, length=4800, media_kind='sound')
                group.parameters.append(aaf.create.ConstantValue(label, 0))
                group.segments.append(clip(mob))
                comp.create_timeline_slot(48000).segment = group
            with aaf2.open(str(source), 'r') as aaf:
                self.assertEqual([gain for _, gain, _ in iter_timeline_sourceclips(aaf)], [expected_gain])
            self.assertEqual(len(collect_timeline_sourceclip_removals_for_sdk_xml(source, FilterConfig())), expected_removed)
            self.assertEqual(filter_aaf_speech_only(source, output, FilterConfig()), expected_removed)
            with aaf2.open(str(output), 'r') as aaf:
                self.assertEqual(len(list(iter_timeline_sourceclips(aaf))), 1 - expected_removed)

    def test_standard_amplitude_zero_is_mute_even_when_renamed(self):
        self.check_parameter('e4962321-2267-11d3-8a4c-0050040ef7d2', 'arbitrary label', 0, 1)

    def test_level_is_not_amplitude_despite_misleading_label(self):
        self.check_parameter('e4962320-2267-11d3-8a4c-0050040ef7d2', 'Amplitude', 1, 0)
