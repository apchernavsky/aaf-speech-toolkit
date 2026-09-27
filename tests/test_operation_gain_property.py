"""OperationGroup gain must not depend on the schema's property display name."""
from types import SimpleNamespace
import unittest

import aaf_pipeline
from aaf_speech_filter.timeline_sourceclips import _opgroup_gain_multiplier


class ConstantValue(SimpleNamespace):
    pass


class LegacyOperationGroup:
    def __init__(self, gain):
        self.parameters = [ConstantValue(auid='e4962321-2267-11d3-8a4c-0050040ef7d2', value=gain)]

    @property
    def operation(self):
        raise KeyError('Operation')

    def get(self, name):
        if name == 'OperationDefinition':
            return SimpleNamespace(value=SimpleNamespace(auid='9d2ea894-0968-11d3-8a38-0050040ef7d2'))
        return None


class OperationGainPropertyTests(unittest.TestCase):
    def test_legacy_operation_definition_reads_gain(self):
        self.assertEqual(_opgroup_gain_multiplier(LegacyOperationGroup(0.5)), 0.5)

    def test_legacy_operation_definition_preserves_mute(self):
        self.assertEqual(_opgroup_gain_multiplier(LegacyOperationGroup(0)), 0)
