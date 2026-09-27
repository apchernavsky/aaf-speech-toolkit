from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import aaf_tool_config
from aaf_io.sdk_tools import default_aaf_tools_dir as sdk_default_aaf_tools_dir


class AafToolConfigTests(unittest.TestCase):
    def test_auto_media_search_roots_keeps_scan_shallow_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "b_media").mkdir()
            (root / "a_media").mkdir()
            (root / "not_a_dir.txt").write_text("", encoding="utf-8")

            roots = aaf_tool_config.auto_media_search_roots_near_aaf(root / "project.aaf")

        self.assertEqual([p.name for p in roots], [root.name, "a_media", "b_media"])

    def test_effective_aaf_tools_dir_prefers_valid_explicit_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tools = Path(td)
            (tools / "aaffmtconv.exe").write_text("", encoding="utf-8")

            with mock.patch.object(aaf_tool_config, "default_aaf_tools_dir", return_value=Path("fallback")):
                self.assertEqual(aaf_tool_config.effective_aaf_tools_dir(tools), tools)

    def test_install_media_search_roots_env_deduplicates_existing_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = os.environ.get("AAF_MEDIA_ROOTS")
            try:
                with mock.patch.object(
                    aaf_tool_config,
                    "load_media_search_roots_from_config",
                    return_value=[root, root],
                ):
                    aaf_tool_config.install_media_search_roots_env()
                self.assertEqual(os.environ.get("AAF_MEDIA_ROOTS"), str(root))
            finally:
                if old is None:
                    os.environ.pop("AAF_MEDIA_ROOTS", None)
                else:
                    os.environ["AAF_MEDIA_ROOTS"] = old

    def test_sdk_tools_dir_accepts_aaf_tools_dir_env(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tools = Path(td)
            (tools / "aaffmtconv.exe").write_text("", encoding="utf-8")
            old_tools = os.environ.get("AAF_TOOLS")
            old_tools_dir = os.environ.get("AAF_TOOLS_DIR")
            try:
                os.environ.pop("AAF_TOOLS", None)
                os.environ["AAF_TOOLS_DIR"] = str(tools)

                self.assertEqual(sdk_default_aaf_tools_dir(), tools)
            finally:
                if old_tools is None:
                    os.environ.pop("AAF_TOOLS", None)
                else:
                    os.environ["AAF_TOOLS"] = old_tools
                if old_tools_dir is None:
                    os.environ.pop("AAF_TOOLS_DIR", None)
                else:
                    os.environ["AAF_TOOLS_DIR"] = old_tools_dir


if __name__ == "__main__":
    unittest.main()
