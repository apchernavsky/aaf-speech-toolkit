"""Generic integration discovery for the separately provisioned public corpus."""
from pathlib import Path
import tempfile
import unittest

from scripts.public_aaf_corpus import DEFAULT_CACHE, ROOT, load_manifest, verify_asset
from scripts.check_aaf_corpus import run_case


def make_case(case, cache):
    class PublicCorpusCase(unittest.TestCase):
        def __init__(self, case, cache):
            super().__init__('runTest')
            self.case = case
            self.cache = cache

        def id(self):
            return 'public_aaf.' + self.case['id']

        def shortDescription(self):
            return self.id()

        def runTest(self):
            with tempfile.TemporaryDirectory(prefix='corpus-test-', dir=ROOT) as directory:
                result = run_case(self.case, cache=self.cache, report=Path(directory)/'case')
                self.assertTrue(result['passed'], result.get('error', str(result)))

    return PublicCorpusCase(case, cache)


def load_tests(loader, tests, pattern):
    manifest = load_manifest()
    suite = unittest.TestSuite()
    if not DEFAULT_CACHE.exists():
        @unittest.skip('Public corpus not provisioned; run python scripts/fetch_aaf_corpus.py')
        def unavailable():
            pass
        suite.addTest(unittest.FunctionTestCase(unavailable))
        return suite
    # A partially provisioned cache must fail, rather than silently reduce coverage.
    for asset in manifest['assets']:
        verify_asset(asset, DEFAULT_CACHE)
    for case in manifest['cases']:
        suite.addTest(make_case(case, DEFAULT_CACHE))
    return suite
