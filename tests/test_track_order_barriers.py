"""Fixed tracks constrain their own peers, not unrelated classes."""
import unittest,itertools,random
from aaf_io.lane_order import class_ordered_lanes

class TrackBarrierTests(unittest.TestCase):
    def test_speech_crosses_unrelated_protected_track(self):
        kinds=[{'music'},{'unknown'},{'speech'}]
        self.assertEqual(class_ordered_lanes(kinds,protected={1}),[2,1,0])

    def test_fixed_peer_order_still_bounds_same_class(self):
        kinds=[set(),{'noise'},{'speech'},{'noise'},{'music'}]
        order=class_ordered_lanes(kinds,protected={1})
        self.assertEqual(order[0],2)
        self.assertEqual(order[1],1)
        self.assertGreater(order.index(3),1)

    def test_seeded_permutations_preserve_identity_fixed_tracks_and_peer_order(self):
        rng=random.Random(5739)
        for count in range(2,20):
            for repeat in range(25):
                kinds=[set(rng.choice([(),('speech',),('noise',),('music',),('unknown',),('speech','noise'),('speech','unknown'),('music','unknown'),('noise','unknown')])) for _ in range(count)]
                protected={i for i in range(count) if rng.random()<0.1}
                order=class_ordered_lanes(kinds,protected=protected)
                self.assertEqual(sorted(order),list(range(count)))
                fixed={i for i,k in enumerate(kinds) if i in protected or len(k-{'unknown'})>1 or 'unknown' in k}
                for lane in fixed:self.assertEqual(order[lane],lane)
                for kind in ('speech','noise','music','unknown'):
                    peers=[i for i,k in enumerate(kinds) if kind in k]
                    self.assertEqual(sorted(peers,key=order.index),peers)
                reordered=[kinds[i] for i in order]
                self.assertEqual(class_ordered_lanes(reordered,protected=protected),list(range(count)))

    def test_unknown_on_music_track_cannot_cross_fixed_unknown_peer(self):
        kinds=[{'music','unknown'},{'unknown'},{'speech'}]
        self.assertEqual(class_ordered_lanes(kinds,protected={1}),[0,1,2])

    def test_unknown_owners_with_different_known_roles_remain_ordered(self):
        self.assertEqual(class_ordered_lanes([{'music','unknown'},{'speech','unknown'}]),[0,1])
