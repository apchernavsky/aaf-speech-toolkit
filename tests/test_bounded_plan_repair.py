import unittest

from aaf_speech_filter.lane_layout_plan import (
    repair_lane_layout_plan_overlaps, validate_lane_layout_plan,
)


class BoundedPlanRepairTests(unittest.TestCase):
    def test_repair_moves_flexible_blocker_instead_of_crossing_fixed_peer(self):
        flexible = dict(src_lane=2, top_idx=0, target_lane=0, T=0, target_T=0,
                        T_edit=0, L=100, kind='speech')
        bounded = dict(src_lane=0, top_idx=0, target_lane=0, T=50, target_T=50,
                       T_edit=50, L=100, kind='noise', lane_bounds=(0, 0))
        events = [flexible, bounded]
        self.assertEqual(repair_lane_layout_plan_overlaps(events, 3), 1)
        self.assertEqual(bounded['target_lane'], 0)
        self.assertNotEqual(flexible['target_lane'], 0)
        validate_lane_layout_plan(events, 3)

    def test_validator_rejects_a_nonoverlapping_plan_outside_allowed_lanes(self):
        event = dict(src_lane=0, target_lane=2, T=10, target_T=10,
                     T_edit=10, L=100, kind='noise', lane_bounds=(0, 1))
        with self.assertRaisesRegex(RuntimeError, 'bounds'):
            validate_lane_layout_plan([event], 3)
