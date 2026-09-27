import unittest
from types import SimpleNamespace
from aaf_speech_filter.media_resolve import _resolve_source_mob


class IndexedMobResolutionTests(unittest.TestCase):
    def test_indexed_lookup_never_scans_unrelated_mobs(self):
        mob = SimpleNamespace(mob_id='wanted')
        class IndexedCollection:
            def get(self, key):
                return mob if key == mob.mob_id else None
            def values(self):
                raise AssertionError('Unexpected full mob scan')
        aaf = SimpleNamespace(content=SimpleNamespace(mobs=IndexedCollection()))
        self.assertIs(_resolve_source_mob(aaf, 'wanted'), mob)
        self.assertIsNone(_resolve_source_mob(aaf, 'missing'))

    def test_unindexed_collection_keeps_exact_identity_lookup(self):
        wrong = SimpleNamespace(mob_id='wrong')
        wanted = SimpleNamespace(mob_id='wanted')
        aaf = SimpleNamespace(content=SimpleNamespace(mobs=[wrong, wanted]))
        self.assertIs(_resolve_source_mob(aaf, 'wanted'), wanted)
        self.assertIsNone(_resolve_source_mob(aaf, 'missing'))

    def test_lookup_observes_current_aaf_collection_without_stale_cache(self):
        first = SimpleNamespace(mob_id='wanted')
        second = SimpleNamespace(mob_id='wanted')
        mobs = {'wanted': first}
        aaf = SimpleNamespace(content=SimpleNamespace(mobs=mobs))
        self.assertIs(_resolve_source_mob(aaf, 'wanted'), first)
        mobs['wanted'] = second
        self.assertIs(_resolve_source_mob(aaf, 'wanted'), second)
