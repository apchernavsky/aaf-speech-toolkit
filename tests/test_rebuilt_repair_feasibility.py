import unittest

from aaf_speech_filter import aaf_yamnet_lane_layout as layout
from tests.test_speech_yamnet import Filler, SourceClip, Transition


class RebuiltRepairFeasibilityTests(unittest.TestCase):
    def assert_valid(self, rebuilt, bounds):
        for lane, parts in rebuilt.items():
            spans = layout._rebuilt_transition_spans_for_lane(rebuilt, lane)
            occupied = []
            for part in parts:
                raw, length, nodes, kind, visible, _classification, offset = part
                if kind != 'event':
                    continue
                primary = layout._rebuilt_event_primary_node(part)
                if id(primary) in bounds:
                    lower, upper = bounds[id(primary)]
                    self.assertLessEqual(lower, lane)
                    self.assertLessEqual(lane, upper)
                own_spans = layout._rebuilt_part_transition_spans(raw, nodes)
                external_spans = layout._spans_without_exact_matches(spans, own_spans)
                expected_raw = layout._raw_from_visible_with_transition_spans(
                    external_spans, visible, offset,
                ) - offset
                self.assertEqual(raw, expected_raw)
                for start, end in occupied:
                    self.assertFalse(layout._intervals_overlap(raw, raw + length, start, end))
                occupied.append((raw, raw + length))

    def test_feasible_incoming_assignment_is_not_reallocated_into_dead_end(self):
        flexible, constrained = SourceClip(100), SourceClip(100)
        rebuilt = {
            0: [(150, 100, [constrained], 'event', 150, 'speech', 0)],
            1: [(100, 100, [flexible], 'event', 100, 'speech', 0)],
        }
        bounds = {id(constrained): (0, 0)}
        before = {lane: list(parts) for lane, parts in rebuilt.items()}
        self.assert_valid(rebuilt, bounds)

        moved = layout._resolve_rebuilt_event_overlaps_without_visible_time_shift(
            rebuilt, 2, lane_bounds_by_node_id=bounds,
        )

        self.assertEqual(moved, 0)
        self.assertEqual(rebuilt, before)
        self.assert_valid(rebuilt, bounds)

    def test_feasible_owned_transition_assignment_is_not_reallocated(self):
        flexible, constrained = SourceClip(100), SourceClip(100)
        transition = Transition(10)
        rebuilt = {
            0: [(150, 100, [constrained], 'event', 150, 'speech', 0)],
            1: [(110, 110, [transition, flexible], 'event', 100, 'speech', 10)],
        }
        bounds = {id(constrained): (0, 0)}
        before = {lane: list(parts) for lane, parts in rebuilt.items()}
        self.assert_valid(rebuilt, bounds)

        moved = layout._resolve_rebuilt_event_overlaps_without_visible_time_shift(
            rebuilt, 2, lane_bounds_by_node_id=bounds,
        )

        self.assertEqual(moved, 0)
        self.assertEqual(rebuilt, before)
        self.assertIs(rebuilt[1][0][2][0], transition)
        self.assert_valid(rebuilt, bounds)

    def test_actual_overlap_is_repaired_without_changing_visible_starts(self):
        first, second = SourceClip(100), SourceClip(100)
        rebuilt = {
            0: [(100, 100, [first], 'event', 100, 'speech', 0),
                (150, 100, [second], 'event', 150, 'speech', 0)],
            1: [],
        }
        expected = {id(first): 100, id(second): 150}

        moved = layout._resolve_rebuilt_event_overlaps_without_visible_time_shift(rebuilt, 2)

        self.assertGreater(moved, 0)
        self.assert_valid(rebuilt, {})
        observed = {
            id(layout._rebuilt_event_primary_node(part)): part[4]
            for parts in rebuilt.values() for part in parts
        }
        self.assertEqual(observed, expected)
        self.assertEqual(sum(len(parts) for parts in rebuilt.values()), 2)

    def test_infeasible_bounded_overlap_leaves_input_plan_unchanged(self):
        first, second = SourceClip(100), SourceClip(100)
        rebuilt = {
            0: [(100, 100, [first], 'event', 100, 'speech', 0),
                (150, 100, [second], 'event', 150, 'speech', 0)],
            1: [],
        }
        bounds = {id(first): (0, 0), id(second): (0, 0)}
        before = {lane: list(parts) for lane, parts in rebuilt.items()}

        with self.assertRaisesRegex(RuntimeError, 'overlap'):
            layout._resolve_rebuilt_event_overlaps_without_visible_time_shift(
                rebuilt, 2, lane_bounds_by_node_id=bounds,
            )

        self.assertEqual(rebuilt, before)


    def test_disjoint_conflict_preserves_feasible_bounded_component(self):
        self._assert_disjoint_conflict_preserves_component(with_transition=False)

    def test_disjoint_conflict_preserves_transition_bearing_component(self):
        self._assert_disjoint_conflict_preserves_component(with_transition=True)

    def _assert_disjoint_conflict_preserves_component(self, *, with_transition):
        flexible, constrained = SourceClip(100), SourceClip(100)
        first, second = SourceClip(50), SourceClip(50)
        nodes = [Transition(10), flexible] if with_transition else [flexible]
        offset = 10 if with_transition else 0
        rebuilt = {
            0: [(150, 100, [constrained], 'event', 150, 'speech', 0),
                (300, 50, [first], 'event', 300, 'speech', 0),
                (325, 50, [second], 'event', 325, 'speech', 0)],
            1: [(100 + offset, 100 + offset, nodes, 'event', 100, 'speech', offset)],
        }
        bounds = {id(constrained): (0, 0)}
        preserved = (rebuilt[0][0], rebuilt[1][0])

        moved = layout._resolve_rebuilt_event_overlaps_without_visible_time_shift(
            rebuilt, 2, lane_bounds_by_node_id=bounds,
        )

        self.assertEqual(moved, 1)
        self.assertIn(preserved[0], rebuilt[0])
        self.assertIn(preserved[1], rebuilt[1])
        self.assert_valid(rebuilt, bounds)
        self.assertEqual(sum(len(parts) for parts in rebuilt.values()), 4)

    def test_transition_move_rejects_new_conflict_in_destination(self):
        moving, pinned, resident = SourceClip(100), SourceClip(50), SourceClip(100)
        rebuilt = {
            0: [(110, 110, [Transition(10), moving], 'event', 100, 'speech', 10),
                (150, 50, [pinned], 'event', 140, 'speech', 0)],
            1: [(300, 100, [resident], 'event', 300, 'speech', 0),
                (400, 10, Filler(10), 'filler', None, None, 0)],
            2: [],
        }
        bounds = {id(pinned): (0, 0)}
        preserved_destination = list(rebuilt[1])

        layout._resolve_rebuilt_event_overlaps_without_visible_time_shift(
            rebuilt, 3, lane_bounds_by_node_id=bounds,
        )

        self.assertEqual(rebuilt[1], preserved_destination)
        self.assertIs(layout._rebuilt_event_primary_node(rebuilt[2][0]), moving)
        self.assertTrue(layout._rebuilt_placement_is_feasible(rebuilt, 3, None, bounds))
        self.assert_valid(rebuilt, bounds)

    def test_negative_raw_placement_is_rejected_without_mutation(self):
        clip = SourceClip(50)
        rebuilt = {0: [(-10, 50, [clip], 'event', -10, 'speech', 0)], 1: []}
        before = {lane: list(parts) for lane, parts in rebuilt.items()}
        self.assertFalse(layout._rebuilt_placement_is_feasible(rebuilt, 2, None, None))

        with self.assertRaisesRegex(RuntimeError, 'overlap'):
            layout._resolve_rebuilt_event_overlaps_without_visible_time_shift(rebuilt, 2)

        self.assertEqual(rebuilt, before)

    def test_failure_after_repairable_conflict_is_atomic(self):
        clips = [SourceClip(50) for _ in range(4)]
        rebuilt = {
            0: [(start, 50, [clip], 'event', start, 'speech', 0)
                for start, clip in zip((100, 125, 300, 325), clips)],
            1: [],
        }
        bounds = {id(clip): (0, 0) for clip in clips[2:]}
        before = {lane: list(parts) for lane, parts in rebuilt.items()}

        with self.assertRaisesRegex(RuntimeError, 'overlap'):
            layout._resolve_rebuilt_event_overlaps_without_visible_time_shift(
                rebuilt, 2, lane_bounds_by_node_id=bounds,
            )

        self.assertEqual(rebuilt, before)

if __name__ == '__main__':
    unittest.main()
