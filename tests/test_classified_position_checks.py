import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from scripts import check_aaf_horizontal_positions as checker


class ClassifiedPositionChecksTests(unittest.TestCase):
    def positions(self):
        return [checker.ClipPosition('display only', f'source-{i}', 1, 5, 20, 100, 100, 20, i, 0) for i in range(3)]

    def plan(self, source, kinds):
        return checker.ClassifiedLayoutPlan('composition-id', tuple(
            checker.ClassifiedOccurrence.from_event(dict(source_id=item.source_id, source_track=item.source_track,
                source_start=item.source_start, source_length=item.source_length, T_edit=item.edit_start,
                L=item.timeline_length, src_lane=item.lane, kind=kind))
            for item, kind in zip(source, kinds)
        ))

    def compare(self, source, output, plan=None):
        with mock.patch.object(checker, 'collect_clip_positions', side_effect=[source, output]):
            return checker.compare_positions(Path('input.aaf'), Path('output.aaf'), max_report=10, classified_plan=plan)

    def test_actual_classes_allow_cross_class_reordering(self):
        source = self.positions()
        output = [replace(source[0], lane=1), replace(source[1], lane=2), replace(source[2], lane=0)]
        self.assertEqual(self.compare(source, output, self.plan(source, ['noise', 'noise', 'speech']))[0], 0)

    def test_same_class_inversion_still_fails_across_lane_bands(self):
        source = self.positions()
        output = [replace(source[0], lane=2), replace(source[1], lane=1), replace(source[2], lane=0)]
        code, report = self.compare(source, output, self.plan(source, ['noise'] * 3))
        self.assertEqual(code, 1)
        self.assertEqual(report['aligned_order_inversions'], 1)

    def test_missing_planned_output_identity_fails_explicitly(self):
        source = self.positions()
        code, report = self.compare(source, source[:-1], self.plan(source, ['noise'] * 3))
        self.assertEqual(code, 1)
        self.assertTrue(report['classified_plan_errors'])

    def test_missing_output_classification_fails_explicitly(self):
        source = self.positions()
        code, report = self.compare(source, source, self.plan(source[:-1], ['noise'] * 2))
        self.assertEqual(code, 1)
        self.assertTrue(report['classified_plan_errors'])

    def test_wrong_original_lane_fails_explicitly(self):
        source = self.positions()
        plan = self.plan(source, ['noise'] * 3)
        plan = replace(plan, events=(replace(plan.events[0], source_lane=8), *plan.events[1:]))
        code, report = self.compare(source, source, plan)
        self.assertEqual(code, 1)
        self.assertTrue(report['classified_plan_errors'])

    def test_plan_cannot_hide_temporal_or_source_window_changes(self):
        source = self.positions()
        for changed in (replace(source[0], edit_start=101), replace(source[0], source_start=6)):
            with self.subTest(changed=changed):
                code, report = self.compare(source, [changed, *source[1:]], self.plan(source, ['noise'] * 3))
                self.assertEqual(code, 1)
                self.assertTrue(report['edit_shifted_or_length_changed'] or report['new_at_unseen_identity'])

    def test_composition_selection_uses_top_level_metadata(self):
        lower = SimpleNamespace(mob_id='lower', usage='Usage_LowerLevel', slots=[object()] * 10)
        top = SimpleNamespace(mob_id='top', usage='Usage_TopLevel', slots=[])
        self.assertIs(checker._pick_main_composition([lower, top]), top)

    def test_legacy_without_plan_keeps_band_heuristic(self):
        source = self.positions()
        source = [replace(item, lane=item.lane + 3) for item in source]
        tail = replace(source[-1], source_id='unrelated', edit_start=500, timeline_start=500, lane=8)
        output = [replace(source[0], lane=5), replace(source[1], lane=4), replace(source[2], lane=3), tail]
        with mock.patch.object(checker, 'collect_clip_positions', side_effect=[[*source, tail], output]):
            code, report = checker.compare_positions(Path('input'), Path('output'), max_report=10)
        self.assertEqual(code, 1)
        self.assertEqual(report['aligned_order_inversions'], 1)

    def test_harness_captures_actual_plan_in_a_scoped_observer(self):
        from scripts.verify_local_aafs import capture_classified_layout
        from aaf_speech_filter import aaf_yamnet_lane_layout as layout
        composition = SimpleNamespace(mob_id='selected')
        event = dict(source_id='source', source_track=1, source_start=0, source_length=20,
                     T_edit=100, L=20, src_lane=2, kind='music')
        with mock.patch.object(layout, '_pick_main_composition', return_value=composition) as select, mock.patch.object(layout, '_assign_target_lanes') as assign:
            with capture_classified_layout() as capture:
                self.assertIs(layout._pick_main_composition([]), composition)
                layout._assign_target_lanes([event], 6)
            self.assertIs(layout._pick_main_composition, select)
            self.assertIs(layout._assign_target_lanes, assign)
            plan = capture.plan()
        self.assertEqual(plan.composition_id, 'selected')
        self.assertEqual(plan.events[0].source_lane, 2)
        self.assertEqual(plan.events[0].class_kind, 'music')


    def test_same_class_repeated_identity_preserves_multiplicity(self):
        source = self.positions()
        source[2] = replace(source[2], source_id=source[0].source_id)
        output = [replace(source[0], lane=4), replace(source[1], lane=6), replace(source[2], lane=8)]
        code, report = self.compare(source, output, self.plan(source, ['speech'] * 3))
        self.assertEqual(code, 0)
        self.assertEqual(report['classified_plan_errors'], [])

    def test_repeated_identity_does_not_hide_distinct_channel_inversion(self):
        source = self.positions()
        source[2] = replace(source[2], source_id=source[0].source_id)
        output = [replace(source[0], lane=4), replace(source[2], lane=6), replace(source[1], lane=8)]
        code, report = self.compare(source, output, self.plan(source, ['speech'] * 3))
        self.assertEqual(code, 1)
        self.assertEqual(report['classified_plan_errors'], [])
        self.assertEqual(report['aligned_order_inversions'], 1)

    def test_repeated_identity_with_mixed_classes_remains_ambiguous(self):
        source = self.positions()
        source[2] = replace(source[2], source_id=source[0].source_id)
        code, report = self.compare(source, source, self.plan(source, ['speech', 'speech', 'noise']))
        self.assertEqual(code, 1)
        self.assertTrue(report['classified_plan_errors'])

    def test_repeated_identity_count_mismatch_fails(self):
        source = self.positions()
        source[2] = replace(source[2], source_id=source[0].source_id)
        for output in (source[:-1], [*source, replace(source[0], lane=4)]):
            with self.subTest(count=len(output)):
                code, report = self.compare(source, output, self.plan(source, ['speech'] * 3))
                self.assertEqual(code, 1)
                self.assertTrue(report['classified_plan_errors'])

    def test_manifest_cannot_claim_same_source_occurrence_twice(self):
        source = self.positions()
        plan = self.plan(source, ['speech'] * 3)
        plan = replace(plan, events=(*plan.events, plan.events[0]))
        output = [*source, replace(source[0], lane=4)]
        code, report = self.compare(source, output, plan)
        self.assertEqual(code, 1)
        self.assertTrue(report['classified_plan_errors'])



if __name__ == '__main__':
    unittest.main()
