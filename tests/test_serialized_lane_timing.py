import unittest
from aaf_speech_filter import aaf_yamnet_lane_layout as layout
from tests.test_speech_yamnet import SourceClip, Transition, Filler


class SerializedLaneTimingTests(unittest.TestCase):
    def test_pruned_transition_cannot_shift_serialized_event(self):
        clip = SourceClip(20)
        rebuilt = {0: [(50, 10, Transition(10), 'transition', None, None, 0),
                       (120, 20, [clip], 'event', 100, 'speech', 0)]}
        layout._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 1)
        self.assertEqual(len(rebuilt[0]), 1)
        self.assertEqual(rebuilt[0][0][0], 100)

    def test_owned_transition_is_counted_once(self):
        clip = SourceClip(20)
        rebuilt = {0: [(90, 30, [Transition(10), clip], 'event', 80, 'speech', 10)]}
        layout._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 1)
        event = next(part for part in rebuilt[0] if part[3] == 'event')
        self.assertEqual(event[0] + event[6], 100)

    def test_serialized_guard_rejects_phantom_transition_shift(self):
        clip = SourceClip(20)
        with self.assertRaisesRegex(RuntimeError, 'visible position'):
            layout._validate_serialized_event_visible_positions([Filler(120), clip], {id(clip): 100})

    def test_serialized_guard_accounts_for_real_transitions(self):
        clip = SourceClip(20)
        layout._validate_serialized_event_visible_positions([Filler(90), Transition(10), clip], {id(clip): 80})

    def test_serialized_guard_rejects_missing_event(self):
        clip = SourceClip(20)
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            layout._validate_serialized_event_visible_positions([Filler(100)], {id(clip): 100})
