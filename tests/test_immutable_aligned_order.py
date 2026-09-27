import unittest
from aaf_speech_filter import lane_layout_constraints as constraints
from aaf_speech_filter import aaf_yamnet_lane_layout as layout
from tests.test_speech_yamnet import SourceClip

class ImmutableAlignedOrderTests(unittest.TestCase):
    def events(self):
        return [dict(src_lane=i, top_idx=0, T=100, T_edit=100, L=20, source_start=50, source_length=20, kind='speech') for i in (0,2,4)]

    def test_fixed_member_bounds_other_members_on_their_original_side(self):
        events = self.events()
        constraints.apply_immutable_aligned_lane_bounds(events, {2}, 6)
        self.assertEqual([e['lane_bounds'] for e in events], [(0,1),(2,2),(3,5)])
        self.assertFalse(constraints.target_lane_allowed_for_event(events[0], 3))
        self.assertFalse(constraints.target_lane_allowed_for_event(events[2], 1))
        self.assertTrue(constraints.target_lane_allowed_for_event(events[2], 3))

    def test_unrelated_class_is_not_constrained_by_fixed_member(self):
        events = self.events()
        events[1]['kind'] = 'noise'
        constraints.apply_immutable_aligned_lane_bounds(events, {2}, 6)
        self.assertNotIn('lane_bounds', events[0])
        self.assertNotIn('lane_bounds', events[2])

    def test_writer_repairs_cannot_cross_fixed_aligned_member(self):
        node = SourceClip(20, start=50)
        rebuilt = {i: [] for i in range(6)}
        part = (100,20,[node],'event',100,'speech',0)
        rebuilt[4] = [part]
        bounds = {id(node):(3,5)}
        layout._resolve_rebuilt_event_overlaps_without_visible_time_shift(rebuilt,6,immutable_lanes={2},lane_bounds_by_node_id=bounds)
        layout._finalize_rebuilt_parts_without_visible_time_shift(rebuilt,6,{id(node):4},immutable_lanes={2},lane_bounds_by_node_id=bounds)
        lane = next(i for i,parts in rebuilt.items() if parts)
        self.assertGreater(lane,2)

    def test_sdk_event_retains_bounds(self):
        event = dict(self.events()[2], target_lane=3, lane_bounds=(3,5))
        sdk_event = layout._build_sdk_lane_layout_events([event], [])[0]
        self.assertEqual(sdk_event.lane_bounds,(3,5))

    def test_replanning_recomputes_bounds_without_stale_constraints(self):
        events = self.events()
        constraints.apply_immutable_aligned_lane_bounds(events,{2},6)
        constraints.apply_immutable_aligned_lane_bounds(events,set(),6)
        self.assertTrue(all('lane_bounds' not in e for e in events))

    def test_real_pyaaf_writer_carries_bounds_into_final_compaction(self):
        import tempfile
        from pathlib import Path
        import aaf2
        from aaf_speech_filter.config import FilterConfig
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'timeline.aaf'
            with aaf2.open(str(path),'w') as aaf:
                comp = aaf.create.CompositionMob()
                aaf.content.mobs.append(comp)
                for lane in range(6):
                    slot = comp.create_timeline_slot(48000)
                    slot.segment = aaf.create.Sequence(media_kind='sound')
                    slot.segment.components.append(aaf.create.Filler(media_kind='sound',length=100 if lane==4 else 120))
                    if lane==4:
                        slot.segment.components.append(aaf.create.SourceClip(media_kind='sound',length=20))
            layout._apply_lane_layout_via_pyaaf2_rebuild(aaf_path=path,
                events=[dict(src_lane=4,top_idx=1,T=100,T_edit=100,L=20,target_lane=4,kind='speech',lane_bounds=(3,5))],
                structural_events=[],n_lanes=6,cfg=FilterConfig(),runtime_essence_paths=None,
                work_dir=Path(td),cancel_check=lambda:None,progress_callback=None,log_callback=None,immutable_lanes={2})
            with aaf2.open(str(path),'r') as aaf:
                comp = next(aaf.content.compositionmobs())
                lanes = [i for i,slot in enumerate(comp.slots) if any(type(n).__name__=='SourceClip' for n in slot.segment.components)]
                self.assertEqual(lanes,[3])
