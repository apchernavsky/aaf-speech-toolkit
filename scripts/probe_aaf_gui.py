"""Exercise the real Tk process button on an owned copy of an AAF."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import tkinter as tk
from tkinter import messagebox
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def widgets(parent):
    for child in parent.winfo_children():
        yield child
        yield from widgets(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=False)
    original_hash = digest(source)

    # Import exactly the normal application bootstrap; no local-package paths.
    import aaf_pipeline
    import aaf_gui
    from aaf_tool_config import resolve_media_search_roots
    from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
    from aaf_speech_filter.timeline_sourceclips import total_timeline_sourceclips

    os.environ["AAF_MEDIA_ROOTS"] = ";".join(map(str, resolve_media_search_roots(source)))
    report = {"source": str(source), "python": sys.executable, "dialogs": [], "passed": False}
    real_tk = tk.Tk
    window = None
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="input-", dir=report_dir) as directory:
        copied = Path(directory) / source.name
        shutil.copyfile(source, copied)

        def record_dialog(kind, title, message, **kwargs):
            report["dialogs"].append({"kind": kind, "title": title, "message": message})
            for widget in widgets(window):
                if isinstance(widget, tk.Text):
                    (report_dir / "gui.log").write_text(widget.get("1.0", "end"), encoding="utf-8")
            window.after_idle(window.quit)

        def create_window(*args, **kwargs):
            nonlocal window
            window = real_tk(*args, **kwargs)
            window.withdraw()

            def callback_error(kind, value, traceback):
                record_dialog("callback_error", kind.__name__, str(value))

            def press_process():
                entries = [w for w in widgets(window) if w.winfo_class() == "TEntry"]
                path_entry = next(w for w in entries if int(w.grid_info()["row"]) == 1)
                path_entry.insert(0, str(copied))
                button = next(w for w in widgets(window)
                              if w.winfo_class() == "TButton" and w.cget("text") == "Обработать")
                button.invoke()
                report["button_invoked"] = True

            window.report_callback_exception = callback_error
            window.after_idle(press_process)
            return window

        try:
            with patch.object(tk, "Tk", create_window), \
                 patch.object(messagebox, "showinfo", lambda title, message, **kw: record_dialog("info", title, message, **kw)), \
                 patch.object(messagebox, "showerror", lambda title, message, **kw: record_dialog("error", title, message, **kw)), \
                 patch.object(messagebox, "showwarning", lambda title, message, **kw: record_dialog("warning", title, message, **kw)):
                aaf_gui.run_gui()
            if not report["dialogs"] or any(d["kind"] != "info" or d["title"] != "Готово" for d in report["dialogs"]):
                raise RuntimeError("GUI did not reach its successful completion dialog")
            output = aaf_pipeline.compute_processed_path(copied)
            with open_aaf_lenient(copied, "r") as aaf:
                report["input_clips"] = total_timeline_sourceclips(aaf)
            with open_aaf_lenient(output, "r") as aaf:
                report["output_clips"] = total_timeline_sourceclips(aaf)
            report["copy_unchanged"] = digest(copied) == original_hash
            report["original_unchanged"] = digest(source) == original_hash
            if not report["copy_unchanged"] or not report["original_unchanged"]:
                raise RuntimeError("Input integrity check failed")
            destination = report_dir / "processed.aaf"
            output.replace(destination)
            report["output_path"] = str(destination)
            report["source_sha256"] = original_hash
            report["passed"] = True
        finally:
            if window is not None:
                window.destroy()
            report["seconds"] = round(time.monotonic() - started, 2)
            (report_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
