from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_aaf_corpus as runner

ROOT = Path(__file__).resolve().parents[1]


class CorpusRunnerTests(unittest.TestCase):
    def test_hash_works_without_file_digest(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            p = Path(directory)/'data'; p.write_bytes(b'abc')
            with patch.object(runner.hashlib, 'file_digest', side_effect=AssertionError('Python 3.10 has no file_digest'), create=True):
                self.assertEqual(runner.sha256(p), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_fixture_roots_override_user_configuration_in_worker(self):
        import aaf_pipeline
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            expected = (Path(directory),)
            with patch('aaf_pipeline.resolve_media_search_roots', return_value=(Path('unrelated'),)):
                with runner.fixture_media_roots(expected):
                    self.assertEqual(aaf_pipeline.resolve_media_search_roots(Path('input.aaf')), expected)
                self.assertEqual(aaf_pipeline.resolve_media_search_roots(), (Path('unrelated'),))

    def test_cleanup_cannot_introduce_or_edit_source_windows(self):
        before = dict(clips=[('source', 1, 10, 20, '48000', '1.0')], essence=[], compositions=1)
        after = dict(before, clips=[('source', 1, 11, 20, '48000', '1.0')])
        with self.assertRaisesRegex(AssertionError, 'source window'):
            runner.assert_cleanup_invariants(before, after)

    def test_cleanup_cannot_change_embedded_audio(self):
        before = dict(clips=[], essence=[('mob', 4, 'original')], compositions=1)
        after = dict(before, essence=[('mob', 4, 'changed')])
        with self.assertRaisesRegex(AssertionError, 'essence'):
            runner.assert_cleanup_invariants(before, after)

    def test_cleanup_allows_removing_an_occurrence(self):
        clip = ('source', 1, 10, 20, '48000', '1.0')
        before = dict(clips=[clip, clip], essence=[], compositions=1)
        runner.assert_cleanup_invariants(before, dict(before, clips=[clip]), allowed_removals=[clip])

    @unittest.skipUnless(os.name == 'nt', 'Windows termination fallback')
    def test_taskkill_failure_still_reaps_worker(self):
        with open(os.devnull, 'w') as output:
            with patch.object(runner.subprocess, 'run', side_effect=OSError('taskkill unavailable')):
                with self.assertRaisesRegex(OSError, 'taskkill unavailable'):
                    runner.run_bounded([sys.executable, '-c', 'import threading; threading.Event().wait()'],
                                       stdout=output, env=dict(os.environ), timeout=0.5)

    @unittest.skipUnless(os.name == 'nt', 'Windows process tree regression')
    def test_timeout_terminates_worker_and_descendant(self):
        import ctypes
        from ctypes import wintypes
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            pid_file = Path(directory)/'child.pid'
            code = "import subprocess,sys,pathlib; p=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.read()'],stdin=subprocess.PIPE); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); p.wait()"
            with open(os.devnull, 'w') as output:
                with self.assertRaises(subprocess.TimeoutExpired):
                    runner.run_bounded([sys.executable, '-c', code, str(pid_file)], stdout=output,
                                       env=dict(os.environ), timeout=2)
            self.assertTrue(pid_file.is_file(), 'Worker did not spawn the child')
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x00100000, False, int(pid_file.read_text()))
            if handle:
                try: self.assertEqual(kernel.WaitForSingleObject(handle, 0), 0, 'Descendant still running')
                finally: kernel.CloseHandle(handle)
