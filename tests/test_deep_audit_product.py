from __future__ import annotations

import contextlib
import importlib
import io
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'aaf_io', ROOT / 'aaf_speech_filter'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from aaf_speech_filter.config import FilterConfig
from aaf_speech_filter import cli, speech_vad
import aaf_pipeline
import aaf_gui


class ThresholdContractTests(unittest.TestCase):
    def test_invalid_values_rejected_by_config_and_raw_predicate(self):
        for value in (float('nan'), float('inf'), -float('inf'), 0.01):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    FilterConfig(quiet_peak_dbfs=value)
                with self.assertRaises(ValueError):
                    speech_vad.is_quiet_clip(Path('unused.wav'), 0, 0, value)

    def test_finite_nonpositive_domain_includes_zero_and_extreme_negative(self):
        for value in (0, -40, -160, -10000):
            self.assertEqual(FilterConfig(quiet_peak_dbfs=value).quiet_peak_dbfs, value)
        self.assertEqual(speech_vad._linear_abs_peak_limit(0), 32768)
        self.assertEqual(speech_vad._linear_abs_peak_limit(-10000), 0)

    def test_pipeline_rejects_threshold_before_allocating_workspace(self):
        with mock.patch.object(aaf_pipeline, '_run_aaf_pipeline_impl') as execute:
            result = aaf_pipeline.run_aaf_pipeline(Path('unused.aaf'), quiet_peak_dbfs=float('nan'))
        execute.assert_not_called()
        self.assertIn('finite', result[2])

    def test_package_cli_rejects_threshold_before_preparation(self):
        with (
            mock.patch.object(sys, 'argv', ['tool', 'unused.aaf', '--prepare', '--quiet-peak-dbfs=nan']),
            mock.patch.object(cli, 'run_hidden') as run, contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as exit_status:
                cli.main()
        self.assertEqual(exit_status.exception.code, 2)
        run.assert_not_called()


class PrepareOptionTests(unittest.TestCase):
    def test_prepare_preserves_every_exposed_processing_option(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = Path(directory) / 'input.aaf'
            source.write_bytes(b'input')
            source.with_name('input_processed.aaf').write_bytes(b'output')
            for quiet, lanes in ((False, True), (True, False), (True, True)):
                flags = ['--remove-quiet' if quiet else '--no-remove-quiet',
                         '--experimental-yamnet-lanes' if lanes else '--no-experimental-yamnet-lanes',
                         '--quiet-peak-dbfs=-62.0', '--yamnet-score-threshold=0.008',
                         '--yamnet-frame-aggregate=mean']
                with (
                    self.subTest(quiet=quiet, lanes=lanes),
                    mock.patch.object(sys, 'argv', ['tool', str(source), '--prepare', *flags]),
                    mock.patch.object(cli, '_find_aaf_pipeline', return_value=ROOT / 'aaf_pipeline.py'),
                    mock.patch.object(cli, 'run_hidden') as run,
                ):
                    self.assertEqual(cli.main(), 0)
                    command = run.call_args.args[0]
                    for option in flags:
                        self.assertIn(option, command)
                    self.assertIn('--no-remove-duplicates', command)

    def test_prepare_preserves_numeric_precision(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = Path(directory) / 'input.aaf'
            source.write_bytes(b'input')
            source.with_name('input_processed.aaf').write_bytes(b'output')
            with mock.patch.object(sys, 'argv', ['tool', str(source), '--prepare',
                    '--quiet-peak-dbfs=-40.0000123456', '--yamnet-score-threshold=0.003512345678']),                  mock.patch.object(cli, 'run_hidden') as run:
                self.assertEqual(cli.main(), 0)
            options = dict(arg.split('=', 1) for arg in run.call_args.args[0] if '=' in arg)
            self.assertEqual(float(options['--quiet-peak-dbfs']), -40.0000123456)
            self.assertEqual(float(options['--yamnet-score-threshold']), 0.003512345678)


def _widgets(parent):
    for child in parent.winfo_children():
        yield child
        yield from _widgets(child)


class GuiLifecycleTests(unittest.TestCase):
    def _exercise_gui(self, *, close=False, failure=False, invalid_threshold=False):
        import tkinter as tk
        from tkinter import messagebox
        real_tk, real_get, real_thread = tk.Tk, tk.Variable.get, threading.Thread
        main_thread = threading.get_ident()
        windows, workers, errors, dialogs, displayed = [], [], [], [], []
        started, cleanup_done, force_release = threading.Event(), threading.Event(), threading.Event()
        cancel_events = []
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = Path(directory) / 'input.aaf'
            source.write_bytes(b'input')

            def pipeline(*args, **kwargs):
                cancel_events.append(kwargs['cancel_event'])
                started.set()
                if close:
                    while not kwargs['cancel_event'].wait(0.01):
                        if force_release.is_set():
                            break
                kwargs['log_callback']('FINAL_WARNING')
                cleanup_done.set()
                return None, source, ('FAILED' if failure else None), 0, None

            def variable_get(variable):
                if threading.get_ident() != main_thread:
                    errors.append('Tk variable read on worker')
                return real_get(variable)

            def make_thread(*args, **kwargs):
                worker = real_thread(*args, **kwargs)
                workers.append(worker)
                return worker

            def finish_dialog(title, message):
                dialogs.append((title, message))
                displayed.append(next(w for w in _widgets(windows[0]) if isinstance(w, tk.Text)).get('1.0', 'end'))
                windows[0].after_idle(windows[0].quit)

            def make_window():
                window = real_tk()
                window.withdraw()
                windows.append(window)
                window.report_callback_exception = lambda *args: errors.append(str(args[1]))
                def start():
                    entries = [w for w in _widgets(window) if w.winfo_class() == 'TEntry']
                    next(w for w in entries if int(w.cget('width')) == 56).insert(0, str(source))
                    if invalid_threshold:
                        threshold = next(w for w in entries if int(w.cget('width')) == 8)
                        threshold.delete(0, 'end')
                        threshold.insert(0, 'nan')
                    next(w for w in _widgets(window) if w.winfo_class() == 'TButton' and w.cget('text') == 'Обработать').invoke()
                    if close:
                        window.after(10, request_close)
                def request_close():
                    if not started.is_set():
                        window.after(10, request_close)
                        return
                    window.tk.call(window.protocol('WM_DELETE_WINDOW'))
                window.after_idle(start)
                window.after(5000, window.quit)
                return window

            try:
                with (
                    mock.patch.object(tk, 'Tk', make_window),
                    mock.patch.object(tk.Variable, 'get', variable_get),
                    mock.patch.object(aaf_pipeline, 'run_aaf_pipeline', pipeline),
                    mock.patch.object(aaf_gui, '_resolve_yamnet_download_policy', return_value=False),
                    mock.patch.object(threading, 'Thread', make_thread),
                    mock.patch.object(threading, 'excepthook', lambda args: errors.append(str(args.exc_value))),
                    mock.patch.object(messagebox, 'showinfo', finish_dialog),
                    mock.patch.object(messagebox, 'showerror', finish_dialog),
                ):
                    aaf_gui.run_gui()
                    if invalid_threshold:
                        self.assertFalse(started.is_set())
                        self.assertTrue(dialogs)
                    else:
                        self.assertTrue(cleanup_done.is_set(), 'GUI returned before worker cleanup')
                        self.assertFalse(workers[0].is_alive())
                        self.assertFalse(workers[0].daemon)
                        if close:
                            self.assertTrue(cancel_events[0].is_set())
                            self.assertEqual(dialogs, [])
                        else:
                            log = displayed[0]
                            terminal = 'FAILED' if failure else 'Готово:'
                            self.assertEqual(log.count('FINAL_WARNING'), 1)
                            self.assertLess(log.index('FINAL_WARNING'), log.index(terminal))
                    self.assertEqual(errors, [])
            finally:
                force_release.set()
                for event in cancel_events:
                    event.set()
                for worker in workers:
                    worker.join(3)
                for window in windows:
                    try:
                        window.destroy()
                    except tk.TclError:
                        pass

    def test_close_waits_for_cancelled_worker_cleanup(self):
        self._exercise_gui(close=True)

    def test_final_warning_before_success_and_main_thread_settings(self):
        self._exercise_gui()

    def test_final_warning_before_failure(self):
        self._exercise_gui(failure=True)

    def test_invalid_threshold_never_starts_gui_worker(self):
        self._exercise_gui(invalid_threshold=True)


class DistributionPayloadTests(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module('scripts.stage_distribution_payload')
        except ModuleNotFoundError:
            self.fail('Missing checked distribution payload boundary')

    def _fixture(self, directory, module):
        root = Path(directory)
        app, sdk, system = root / 'dist_pyinstaller.stage-test' / 'aaf_pipeline', root / 'sdk', root / 'system'
        for path in (app / '_internal', sdk, system):
            path.mkdir(parents=True)
        (app / 'aaf_pipeline.exe').write_bytes(b'fake exe')
        (app / '_internal' / 'python312.dll').write_bytes(b'python')
        (app / '_internal' / 'vcruntime140.dll').write_bytes(b'vc')
        (system / 'ucrtbase.dll').write_bytes(b'crt')
        for name in module.REQUIRED_SDK_FILES:
            (sdk / name).write_bytes(name.encode())
        return app, sdk, system

    def test_copy_failure_aborts_before_promotion_and_preserves_previous(self):
        module = self._module()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            app, sdk, system = self._fixture(directory, module)
            previous = Path(directory) / 'dist_pyinstaller'
            previous.mkdir()
            (previous / 'marker').write_text('previous')
            with mock.patch.object(module.shutil, 'copy2', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    module.stage_payload(app, sdk_dirs=[sdk], ffmpeg=None, system_dir=system)
            self.assertEqual((previous / 'marker').read_text(), 'previous')

    def test_required_sdk_dll_and_tool_must_be_present(self):
        module = self._module()
        for missing in ('AAFCOAPI.dll', 'aaffmtconv.exe', 'ComAAFInfo.exe'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                app, sdk, system = self._fixture(directory, module)
                (sdk / missing).unlink()
                with self.assertRaises(FileNotFoundError):
                    module.stage_payload(app, sdk_dirs=[sdk], ffmpeg=None, system_dir=system)

    def test_complete_payload_validates_and_copies_optional_ffmpeg(self):
        module = self._module()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            app, sdk, system = self._fixture(directory, module)
            ffmpeg = Path(directory) / 'ffmpeg.exe'
            ffmpeg.write_bytes(b'ffmpeg')
            module.stage_payload(app, sdk_dirs=[sdk], ffmpeg=ffmpeg, system_dir=system)
            self.assertEqual((app / 'bin' / 'ffmpeg.exe').read_bytes(), b'ffmpeg')
            module.validate_payload(app)

    def test_smoke_failure_is_bounded_and_never_promotes(self):
        module = self._module()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            app, sdk, system = self._fixture(directory, module)
            module.stage_payload(app, sdk_dirs=[sdk], ffmpeg=None, system_dir=system)
            with mock.patch.object(module, 'run_bounded', return_value=19) as run:
                with self.assertRaises(RuntimeError):
                    module.smoke_distribution(app, timeout=2)
            self.assertEqual(run.call_args.kwargs['timeout'], 2)

    def test_missing_python_runtime_rejected(self):
        module = self._module()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            app, sdk, system = self._fixture(directory, module)
            (app / '_internal' / 'python312.dll').unlink()
            with self.assertRaises(FileNotFoundError):
                module.stage_payload(app, sdk_dirs=[sdk], ffmpeg=None, system_dir=system)

    def test_payload_rejects_current_distribution_as_destination(self):
        module = self._module()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            app, sdk, system = self._fixture(directory, module)
            current = Path(directory) / 'dist_pyinstaller' / 'aaf_pipeline'
            module.shutil.copytree(app, current)
            before = sorted(path.relative_to(current) for path in current.rglob("*"))
            with self.assertRaises(ValueError):
                module.stage_payload(current, sdk_dirs=[sdk], ffmpeg=None, system_dir=system)
            self.assertEqual(sorted(path.relative_to(current) for path in current.rglob("*")), before)

    def test_smoke_rejects_readable_output_with_lost_audio(self):
        import aaf2
        module = self._module()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            app, sdk, system = self._fixture(directory, module)
            module.stage_payload(app, sdk_dirs=[sdk], ffmpeg=None, system_dir=system)
            def run(command, **kwargs):
                if '--help' not in command:
                    source = Path(command[1])
                    with aaf2.open(str(source.with_name('input_processed.aaf')), 'w') as aaf:
                        aaf.content.mobs.append(aaf.create.CompositionMob())
                return 0
            with mock.patch.object(module, 'run_bounded', run):
                with self.assertRaises(RuntimeError):
                    module.smoke_distribution(app, timeout=2)


if __name__ == '__main__':
    unittest.main()
