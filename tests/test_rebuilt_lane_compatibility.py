import unittest
from aaf_speech_filter import aaf_yamnet_lane_layout as layout
from tests.test_speech_yamnet import SourceClip


class RebuiltLaneCompatibilityTests(unittest.TestCase):
    @staticmethod
    def event(node):
        return (100, 20, [node], 'event', 100, 'speech', 0)

    def test_overlap_repair_preserves_clock_and_wrapper_signature(self):
        node = SourceClip(20)
        rebuilt = {0: [], 1: [self.event(node)], 2: []}
        layout._resolve_rebuilt_event_overlaps_without_visible_time_shift(
            rebuilt, 3, lane_signatures=[('other',), ('source',), ('other',)])
        self.assertEqual(rebuilt[1], [self.event(node)])

    def test_individual_compaction_chooses_first_compatible_lane(self):
        node = SourceClip(20)
        rebuilt = {0: [], 1: [], 2: [self.event(node)]}
        layout._compact_rebuilt_parts_to_class_zones_without_visible_time_shift(
            rebuilt, 3, lane_signatures=[('other',), ('source',), ('source',)])
        self.assertEqual(rebuilt[1], [self.event(node)])
        self.assertEqual(rebuilt[0], [])

    def test_group_compaction_preserves_each_members_signature(self):
        first, second = SourceClip(20), SourceClip(20)
        rebuilt = {lane: [] for lane in range(12)}
        rebuilt[10], rebuilt[11] = [self.event(first)], [self.event(second)]
        layout._compact_rebuilt_aligned_groups_without_visible_time_shift(
            rebuilt, 12, {id(first): 10, id(second): 11},
            lane_signatures=[('other',), ('other',), ('left',), ('right',)] + [('other',)] * 6 + [('left',), ('right',)])
        self.assertEqual(rebuilt[2], [self.event(first)])
        self.assertEqual(rebuilt[3], [self.event(second)])

    def test_finalizer_passes_compatibility_to_every_compaction_stage(self):
        first, second = SourceClip(20), SourceClip(20)
        rebuilt = {lane: [] for lane in range(6)}
        rebuilt[4], rebuilt[5] = [self.event(first)], [self.event(second)]
        layout._finalize_rebuilt_parts_without_visible_time_shift(
            rebuilt, 6, {id(first): 4, id(second): 5},
            lane_signatures=[('other',), ('other',), ('left',), ('right',), ('left',), ('right',)])
        self.assertEqual(rebuilt[2], [self.event(first)])
        self.assertEqual(rebuilt[3], [self.event(second)])
