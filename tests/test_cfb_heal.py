from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aaf_io.heal import cfb


class CfbBookkeepingRestoreTests(unittest.TestCase):
    def test_sync_fat_header_ignores_non_cfb_files_without_opening_as_aaf(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "not-cfb.aaf"
            path.write_bytes(b"0" * 128)

            with mock.patch("aaf_io.heal.cfb.compute_fat_sector_count_from_chain") as compute:
                changed = cfb.sync_fat_sector_count_header(path)

        self.assertFalse(changed)
        compute.assert_not_called()

    def test_partial_bookkeeping_mutation_failure_propagates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, output = root / 'source.aaf', root / 'output.aaf'
            source.write_bytes(b'AB')
            output.write_bytes(b'00')
            def fake_copy(_src, dst):
                Path(dst).write_bytes(b'00')
            fake_aaf = mock.MagicMock()
            with mock.patch('aaf_io.heal.cfb.shutil.copyfile', side_effect=fake_copy), \
                 mock.patch('aaf_io.compat.pyaaf2_lenient.open_aaf_lenient', return_value=fake_aaf), \
                 mock.patch('aaf_io.heal.cfb._diff_ranges', return_value=[(0, 1), (1, 1)]), \
                 mock.patch('aaf_io.heal.cfb._range_equals', side_effect=[True, OSError('storage failed')]):
                with self.assertRaisesRegex(OSError, 'storage failed'):
                    cfb.restore_pyAAF2_noop_bookkeeping_from_source(source, output, work_dir=root)

    def test_restores_only_noop_matching_ranges_and_preserves_real_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.aaf"
            output = root / "output.aaf"
            source.write_bytes(b"abcdefghij")
            output.write_bytes(b"aXcdefZhiJNOOPTAIL")

            def fake_copyfile(_src: str, dst: str):
                Path(dst).write_bytes(b"aXcdefYhijNOOPTAIL")
                return dst

            class _FakeAaf:
                content = object()

                def __enter__(self):
                    return self

                def __exit__(self, _exc_type, _exc, _tb):
                    return False

            with (
                mock.patch("shutil.copyfile", side_effect=fake_copyfile),
                mock.patch(
                    "aaf_io.compat.pyaaf2_lenient.open_aaf_lenient",
                    return_value=_FakeAaf(),
                ),
            ):
                res = cfb.restore_pyAAF2_noop_bookkeeping_from_source(
                    source,
                    output,
                    work_dir=root,
                )

            self.assertTrue(res.applied)
            self.assertEqual(output.read_bytes(), b"abcdefZhiJ")
            self.assertEqual(res.restored_ranges, 1)
            self.assertEqual(res.skipped_ranges, 1)
            self.assertTrue(res.truncated_to_source_size)


if __name__ == "__main__":
    unittest.main()
