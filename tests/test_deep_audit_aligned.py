import unittest
from aaf_speech_filter import aaf_yamnet_lane_layout as layout
from tests.test_speech_yamnet import SourceClip, Transition


class AlignedWriterOrderTests(unittest.TestCase):
    def make_fragmented_group(self, *, inverted=True, transition=False):
        first, second = SourceClip(20, start=50), SourceClip(20, start=50)
        rebuilt = {lane: [] for lane in range(9)}
        for lane in range(9):
            if lane not in (3, 5):
                rebuilt[lane].append((80, 70, object(), 'structural', None, None, 0))
        lanes = (5, 3) if inverted else (3, 5)
        first_nodes = [Transition(5), first] if transition else [first]
        rebuilt[lanes[0]].append((105 if transition else 100, 25 if transition else 20, first_nodes, 'event', 100, 'noise', 5 if transition else 0))
        rebuilt[lanes[1]].append((100, 20, [second], 'event', 100, 'noise', 0))
        return rebuilt, first, second

    def lane_for(self, rebuilt, node):
        return next(lane for lane, parts in rebuilt.items() for part in parts if layout._rebuilt_event_primary_node(part) is node)

    def test_final_writer_repairs_inversion_across_occupied_gap(self):
        rebuilt, first, second = self.make_fragmented_group()
        layout._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 9, {id(first): 0, id(second): 1})
        self.assertLess(self.lane_for(rebuilt, first), self.lane_for(rebuilt, second))
        self.assertEqual({part[4] for parts in rebuilt.values() for part in parts if part[3] == 'event'}, {100})

    def test_final_writer_keeps_already_ordered_fragmented_group_unchanged(self):
        rebuilt, first, second = self.make_fragmented_group(inverted=False)
        before = {lane: list(parts) for lane, parts in rebuilt.items()}
        layout._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 9, {id(first): 0, id(second): 1})
        self.assertEqual(rebuilt, before)

    def test_alignment_identity_excludes_owned_transition_padding(self):
        rebuilt, first, second = self.make_fragmented_group(transition=True)
        layout._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 9, {id(first): 0, id(second): 1})
        self.assertLess(self.lane_for(rebuilt, first), self.lane_for(rebuilt, second))
        first_part = next(part for part in rebuilt[self.lane_for(rebuilt, first)] if layout._rebuilt_event_primary_node(part) is first)
        self.assertEqual((first_part[1], first_part[4], first_part[6]), (25, 100, 5))
        self.assertIsInstance(first_part[2][0], Transition)

    def test_distinct_classes_do_not_gain_an_order_constraint(self):
        rebuilt, first, second = self.make_fragmented_group()
        p = rebuilt[3][0]
        rebuilt[3][0] = (*p[:5], 'unknown', p[6])
        before = {lane: list(parts) for lane, parts in rebuilt.items()}
        layout._finalize_rebuilt_parts_without_visible_time_shift(rebuilt, 9, {id(first): 0, id(second): 1})
        self.assertEqual(rebuilt, before)

    def test_unsafe_transition_permutation_leaves_group_unchanged(self):
        rebuilt, first, second = self.make_fragmented_group(transition=True)
        rebuilt[3].append((120, 10, object(), 'structural', None, None, 0))
        before = {lane: list(parts) for lane, parts in rebuilt.items()}
        with self.assertRaisesRegex(RuntimeError, 'overlap'):
            layout._restore_rebuilt_aligned_source_order(rebuilt, {id(first): 0, id(second): 1}, {}, set())
        self.assertEqual(rebuilt, before)

    def test_incompatible_lane_permutation_leaves_group_unchanged(self):
        rebuilt, first, second = self.make_fragmented_group()
        signatures = [('a',)] * 9
        signatures[1] = signatures[3] = ('b',)
        before = {lane: list(parts) for lane, parts in rebuilt.items()}
        with self.assertRaisesRegex(RuntimeError, 'no compatible placement'):
            layout._restore_rebuilt_aligned_source_order(rebuilt, {id(first): 0, id(second): 1}, {}, set(), signatures)
        self.assertEqual(rebuilt, before)

    def test_immutable_lane_permutation_leaves_group_unchanged(self):
        rebuilt, first, second = self.make_fragmented_group()
        before = {lane: list(parts) for lane, parts in rebuilt.items()}
        with self.assertRaisesRegex(RuntimeError, 'no compatible placement'):
            layout._restore_rebuilt_aligned_source_order(rebuilt, {id(first): 0, id(second): 1}, {}, {3})
        self.assertEqual(rebuilt, before)



if __name__ == '__main__':
    unittest.main()
