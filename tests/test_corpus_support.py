from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from scripts import public_aaf_corpus as corpus

ROOT = Path(__file__).resolve().parents[1]


def asset(data=b'fixture'):
    sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    return dict(path='inputs/' + sha + '.aaf', bytes=len(data), git_blob_sha=sha, role='aaf',
                source=dict(url='https://example.invalid/pinned/data.aaf'))


class CorpusSupportTests(unittest.TestCase):
    def test_support_api_exists(self):
        for name in ('validate_manifest', 'fetch_asset', 'verify_asset', 'safe_path'):
            self.assertTrue(callable(getattr(corpus, name, None)), name)

    def test_manifest_rejects_duplicate_case_hash(self):
        a = asset()
        case = dict(id=a['git_blob_sha'], asset=a['path'], expectation='cfb', origins=[])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            corpus.validate_manifest(dict(schema=1, assets=[a], cases=[case, dict(case)]))

    def test_every_aaf_asset_requires_a_case(self):
        first = asset(); extra = asset(b'another fixture')
        case = dict(id=first['git_blob_sha'], asset=first['path'], expectation='cfb', origins=[])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            corpus.validate_manifest(dict(schema=1, assets=[first, extra], cases=[case]))

    def test_path_cannot_escape_cache(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            for path in ('../escape', '/absolute', 'C:/absolute', 'nested/../../escape', 'x\\..\\escape', 'file:stream'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    corpus.safe_path(Path(tmp), path)

    def test_corrupt_download_is_not_published(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            a = asset()
            with patch.object(corpus, 'urlopen', return_value=io.BytesIO(b'corrupt')):
                with self.assertRaisesRegex(ValueError, 'hash|size'):
                    corpus.fetch_asset(a, Path(tmp))
            self.assertFalse((Path(tmp)/a['path']).exists())
            self.assertFalse(list(Path(tmp).rglob('*.part')))

    def test_existing_bad_cache_is_reported_without_network(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            a = asset(); path = Path(tmp)/a['path']; path.parent.mkdir(); path.write_bytes(b'wrong')
            with patch.object(corpus, 'urlopen') as network:
                with self.assertRaises(ValueError): corpus.fetch_asset(a, Path(tmp))
            network.assert_not_called()
            self.assertEqual(path.read_bytes(), b'wrong')

    def test_fetch_then_verify_is_idempotent(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            a = asset()
            with patch.object(corpus, 'urlopen', return_value=io.BytesIO(b'fixture')) as network:
                path = corpus.fetch_asset(a, Path(tmp))
                self.assertEqual(path.read_bytes(), b'fixture')
                self.assertEqual(corpus.fetch_asset(a, Path(tmp)), path)
                self.assertEqual(network.call_count, 1)
            corpus.verify_asset(a, Path(tmp))

    def test_zip_reads_exact_member_without_extracting_paths(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as z:
            z.writestr('nested/input.aaf', b'fixture')
            z.writestr('../escape', b'not extracted')
        payload = stream.getvalue(); a = asset()
        a['source'].update(member='nested/input.aaf', archive_bytes=len(payload),
                           archive_sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            with patch.object(corpus, 'urlopen', return_value=io.BytesIO(payload)):
                path = corpus.fetch_asset(a, Path(tmp))
            self.assertEqual(path.read_bytes(), b'fixture')
            self.assertEqual(len(list(Path(tmp).rglob('*.aaf'))), 1)

    def test_missing_asset_is_a_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            with self.assertRaises(FileNotFoundError): corpus.verify_asset(asset(), Path(tmp))


if __name__ == '__main__':
    unittest.main()
