from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SourceStartupTests(unittest.TestCase):
    def run_source_probe(self, code, cwd):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        return subprocess.run(
            [sys.executable, "-E", "-c", code, str(ROOT)],
            cwd=cwd, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )

    def test_gui_preflight_before_pipeline_in_fresh_process(self):
        code = r'''
import pathlib, runpy, sys
root = pathlib.Path(sys.argv[1])
runpy.run_path(str(root / "aaf_pipeline.py"), run_name="startup_probe")
import aaf_gui
assert isinstance(aaf_gui._resolve_yamnet_download_policy(experimental_lane_layout=True), bool)
import aaf_io, aaf_speech_filter.speech_yamnet as yamnet
assert pathlib.Path(aaf_io.__file__).resolve().is_relative_to(root / "aaf_io")
assert pathlib.Path(yamnet.__file__).resolve().is_relative_to(root / "aaf_speech_filter")
assert "tensorflow" not in sys.modules
print("GUI_PREFLIGHT_OK")
'''
        with tempfile.TemporaryDirectory(dir=ROOT) as other_directory:
            for cwd in (ROOT, Path(other_directory)):
                with self.subTest(cwd=cwd):
                    result = self.run_source_probe(code, cwd)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("GUI_PREFLIGHT_OK", result.stdout)

    def test_gui_entry_dispatch_does_not_require_pipeline_first(self):
        code = r'''
import pathlib, runpy, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
import aaf_gui
calls = []
def check_gui_start():
    calls.append(aaf_gui._resolve_yamnet_download_policy(experimental_lane_layout=True))
    assert len(calls) == 1
    assert "tensorflow" not in sys.modules
    print("GUI_ENTRY_OK")
aaf_gui.run_gui = check_gui_start
sys.argv = [str(root / "aaf_pipeline.py")]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
        result = self.run_source_probe(code, ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("GUI_ENTRY_OK", result.stdout)
