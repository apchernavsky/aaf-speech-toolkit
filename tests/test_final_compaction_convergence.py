import copy
import unittest

from aaf_speech_filter.lane_layout_final_compaction import final_bounded_compact_class_events
from aaf_speech_filter.lane_layout_groups import aligned_kind_group_key
from aaf_speech_filter.lane_layout_plan import validate_lane_layout_plan


def event(source, target, *, start=100, length=20, window=None):
    return dict(src_lane=source, target_lane=target, top_idx=0, T=start,
                T_edit=start, T_nuendo=start, target_T=start, L=length,
                source_start=source * 100 if window is None else window,
                source_length=length, kind='speech')


def compact(events, allowed=lambda item, lane: True):
    return final_bounded_compact_class_events(
        events, lane_count=24, target_allowed=allowed,
        target_raw_for_lane=lambda item, lane: item['T_edit'],
        aligned_group_key=aligned_kind_group_key)


class FinalCompactionConvergenceTests(unittest.TestCase):
    def pair_case(self):
        blockers = [event(i, i) for i in range(3)]
        single = event(8, 4)
        pair = [event(10, 5, window=900), event(11, 6, window=900)]
        return blockers + [single] + pair, single, pair

    def test_pair_uses_space_released_by_singleton(self):
        events, single, pair = self.pair_case()
        allowed = lambda item, lane: lane == item['src_lane'] if item['src_lane'] < 3 else True
        before = [(item['T_edit'], item['source_start'], item['L']) for item in events]
        compact(events, allowed)
        self.assertEqual(single['target_lane'], 3)
        self.assertEqual([item['target_lane'] for item in pair], [4, 5])
        self.assertEqual([(item['T_edit'], item['source_start'], item['L']) for item in events], before)
        validate_lane_layout_plan(events, 24)
        accepted = copy.deepcopy(events)
        self.assertEqual(compact(events, allowed), 0)
        self.assertEqual(events, accepted)

    def test_later_interval_releases_space_for_earlier_unit(self):
        fixed = event(0, 0, length=10)
        earlier = event(4, 3)
        later = event(5, 1, start=110, length=10)
        events = [fixed, earlier, later]
        choices = {0: {0}, 4: {1, 3}, 5: {0, 1}}
        compact(events, lambda item, lane: lane in choices[item['src_lane']])
        self.assertEqual([item['target_lane'] for item in events], [0, 1, 0])
        validate_lane_layout_plan(events, 24)

    def test_pair_respects_rendering_constraint_after_space_released(self):
        events, single, pair = self.pair_case()
        def allowed(item, lane):
            source = item['src_lane']
            if source < 3:
                return lane == source
            if source in (10, 11):
                return lane == source - 5
            return True
        compact(events, allowed)
        self.assertEqual(single['target_lane'], 3)
        self.assertEqual([item['target_lane'] for item in pair], [5, 6])
        validate_lane_layout_plan(events, 24)

    def test_pair_cannot_use_interval_occupied_outside_reported_instant(self):
        events, single, pair = self.pair_case()
        late_blocker = event(9, 4, start=115, length=5)
        events.append(late_blocker)
        def allowed(item, lane):
            source = item['src_lane']
            if source < 3:
                return lane == source
            if source == 9:
                return lane == 4
            return True
        compact(events, allowed)
        self.assertEqual(single['target_lane'], 3)
        self.assertEqual([item['target_lane'] for item in pair], [5, 6])
        validate_lane_layout_plan(events, 24)

    def test_fragmented_pair_closes_same_base_gap_after_noise_moves(self):
        pair = [event(10, 0, window=900), event(11, 2, window=900)]
        noise = event(12, 1)
        noise['kind'] = 'noise'
        events = pair + [noise]
        def allowed(item, lane):
            if item['src_lane'] == 10:
                return lane == 0
            if item['src_lane'] == 11:
                return lane in (1, 2)
            return True
        compact(events, allowed)
        self.assertEqual([item['target_lane'] for item in pair], [0, 1])
        validate_lane_layout_plan(events, 24)
        accepted = copy.deepcopy(events)
        self.assertEqual(compact(events, allowed), 0)
        self.assertEqual(events, accepted)
