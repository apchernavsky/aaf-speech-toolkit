"""Offline external AAF checks; every case runs in its own bounded process."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.public_aaf_corpus import (
    DEFAULT_CACHE, MANIFEST, case_asset, load_manifest, safe_path, verify_asset,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def fixture_media_roots(roots):
    """Inject an explicit media snapshot only inside this test worker."""
    with patch('aaf_pipeline.resolve_media_search_roots', return_value=tuple(roots)):
        yield


@contextmanager
def windows_descendant_handles(parent_pid):
    """Hold process handles so taskkill completion can be waited on explicitly."""
    import ctypes
    from ctypes import wintypes
    class ProcessEntry(ctypes.Structure):
        _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
                    ('th32ProcessID', wintypes.DWORD), ('th32DefaultHeapID', ctypes.c_size_t),
                    ('th32ModuleID', wintypes.DWORD), ('cntThreads', wintypes.DWORD),
                    ('th32ParentProcessID', wintypes.DWORD), ('pcPriClassBase', wintypes.LONG),
                    ('dwFlags', wintypes.DWORD), ('szExeFile', wintypes.WCHAR * 260)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot_handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot_handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    parents = {}
    try:
        entry = ProcessEntry(); entry.dwSize = ctypes.sizeof(entry)
        found = kernel.Process32FirstW(snapshot_handle, ctypes.byref(entry))
        while found:
            parents[entry.th32ProcessID] = entry.th32ParentProcessID
            found = kernel.Process32NextW(snapshot_handle, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot_handle)
    descendants = {parent_pid}
    while True:
        expanded = descendants | {pid for pid, parent in parents.items() if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    handles = []
    try:
        for pid in descendants - {parent_pid}:
            handle = kernel.OpenProcess(0x00100001, False, pid)
            if handle:
                handles.append(handle)
        yield kernel, handles
    finally:
        for handle in handles:
            kernel.CloseHandle(handle)


def run_bounded(command, *, stdout, env, timeout):
    options = dict(stdout=stdout, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NO_WINDOW
    else:
        options['start_new_session'] = True
    process = subprocess.Popen(command, **options)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        try:
            # Terminate descendants while the parent PID still anchors its tree.
            if os.name == 'nt':
                with windows_descendant_handles(process.pid) as (kernel, descendants):
                    try:
                        killed = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                            capture_output=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
                        if killed.returncode != 0 and process.poll() is None:
                            raise RuntimeError('Could not terminate corpus worker process tree')
                    finally:
                        deadline = time.monotonic() + 15
                        for handle in descendants:
                            # taskkill may have failed or may still be completing.
                            if kernel.WaitForSingleObject(handle, 0) == 258:
                                kernel.TerminateProcess(handle, 1)
                            remaining = max(0, int((deadline - time.monotonic()) * 1000))
                            if kernel.WaitForSingleObject(handle, remaining) != 0:
                                raise RuntimeError('Corpus worker descendant did not terminate')
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            # Never enter Popen.__exit__: it would wait without a bound if tree
            # enumeration or the platform termination command itself failed.
            if process.poll() is None:
                process.kill()
            process.wait(timeout=15)
        raise


def snapshot(path):
    # Exercise the same source bootstrap used by the normal application.
    import aaf_pipeline
    from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
    from aaf_speech_filter.timeline_sourceclips import iter_timeline_sourceclips
    with open_aaf_lenient(path, 'r') as aaf:
        objects = Counter()
        for obj, _streams in aaf.content.walk_references():
            objects[str(obj.class_id)] += 1
        mobs = []
        for mob in aaf.content.mobs:
            slots = []
            for slot in mob.slots:
                slots.append((slot.slot_id, str(getattr(slot, 'edit_rate', None)),
                              str(slot.segment.class_id), getattr(slot.segment, 'length', None)))
            mobs.append((str(mob.mob_id), str(mob.class_id), slots))
        clips = []
        for rate, gain, clip in iter_timeline_sourceclips(aaf):
            clips.append((str(clip.mob_id), clip.slot_id, clip.start, clip.length, str(rate), str(gain)))
        essence = []
        for item in aaf.content.essencedata:
            stream = item.open('r')
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
            essence.append((str(item.mob_id), size, digest.hexdigest()))
        return dict(objects=dict(objects), mobs=sorted(mobs), clips=clips, essence=sorted(essence),
                    compositions=len(list(aaf.content.compositionmobs())))


def check_structure(source, expectation, work):
    import aaf_pipeline
    from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
    from aaf2.exceptions import CompoundFileBinaryError
    if expectation == 'unsupported_mxf':
        if source.read_bytes()[:4] != bytes.fromhex('060e2b34'):
            raise AssertionError('Negative case is no longer an MXF-style container')
        try:
            with open_aaf_lenient(source, 'r'):
                pass
        except CompoundFileBinaryError as exc:
            if 'magic signature' not in str(exc):
                raise
            return dict(expected_rejection='unsupported_mxf')
        raise AssertionError('Container became readable; update its explicit expectation')
    before = snapshot(source)
    copied = work / 'roundtrip.aaf'
    shutil.copyfile(source, copied)
    with open_aaf_lenient(copied, 'r+') as aaf:
        # Trigger ordinary property serialization, not merely a binary copy.
        for mob in aaf.content.mobs:
            if 'Name' in mob:
                mob['Name'].value = mob['Name'].value
        aaf.save()
    after = snapshot(copied)
    if before != after:
        changed = [key for key in before if before[key] != after[key]]
        raise AssertionError('Save/reopen changed semantic snapshot: ' + ', '.join(changed))
    return dict(mobs=len(before['mobs']), sound_clips=len(before['clips']),
                compositions=before['compositions'], essence_streams=len(before['essence']),
                traversed_objects=sum(before['objects'].values()))


def assert_cleanup_invariants(before, after, *, allowed_removals=()):
    if Counter(after['clips']) - Counter(before['clips']):
        raise AssertionError('Cleanup introduced or changed a retained source window')
    removed = Counter(before['clips']) - Counter(after['clips'])
    if removed != Counter(allowed_removals):
        raise AssertionError('Cleanup removals differ from independently authorized occurrences')
    if before['essence'] != after['essence']:
        raise AssertionError('Cleanup changed embedded essence')
    if before['compositions'] != after['compositions']:
        raise AssertionError('Cleanup changed composition count')


def check_processing(source, case, cache, work, *, yamnet=False):
    from aaf_pipeline import run_aaf_pipeline
    copied = work / 'input.aaf'
    shutil.copyfile(source, copied)
    before_hash = sha256(copied)
    before = snapshot(copied)
    # This worker owns its environment. Sidecars are scoped to fixture provenance.
    os.environ['AAF_MEDIA_ROOTS'] = ';'.join(str(safe_path(cache, p)) for p in case.get('media_roots', []))
    roots = tuple(safe_path(cache, p) for p in case.get('media_roots', []) if safe_path(cache, p).is_dir())
    with fixture_media_roots(roots):
        result = run_aaf_pipeline(copied, remove_quiet_clips=True, remove_duplicates=True,
                                  experimental_yamnet_lane_layout=yamnet,
                                  allow_yamnet_download=False, aaf_tools_dir=ROOT / 'sdk_bin',
                                  log_callback=lambda message: print(message, flush=True))
    if sha256(copied) != before_hash:
        raise AssertionError('Processing modified its input')
    if result[2] is not None:
        raise AssertionError('Pipeline failed: ' + str(result[2]))
    if result[1] is None or Path(result[1]).resolve() == copied.resolve():
        raise AssertionError('Pipeline did not produce a separate output')
    after = snapshot(Path(result[1]))
    from scripts.aaf_removal_oracle import validate_cleanup
    proof = validate_cleanup(copied, Path(result[1]), work_dir=work/'oracle', media_roots=roots)
    (work.parent/'removal-oracle.json').write_text(json.dumps(proof, indent=2), encoding='utf8')
    # Authorize only after the independent occurrence/audio proof succeeds.
    assert_cleanup_invariants(before, after, allowed_removals=Counter(before['clips'])-Counter(after['clips']))
    return dict(input_clips=len(before['clips']), output_clips=len(after['clips']),
                removed=result[3], layout=result[4], output_bytes=Path(result[1]).stat().st_size,
                yamnet_enabled=yamnet)


def worker(manifest, case, cache, stage, report, work_dir=None):
    report.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(id=case['id'], stage=stage, passed=False, expectation=case['expectation'])
    source = None
    before = None
    try:
        source = verify_asset(case_asset(manifest, case), cache)
        before = sha256(source)
        context = nullcontext(work_dir) if work_dir is not None else tempfile.TemporaryDirectory(prefix='work-', dir=report)
        with context as directory:
            work = Path(directory)
            if stage == 'structure' or case['expectation'] != 'cfb':
                result['checks'] = check_structure(source, case['expectation'], work)
            else:
                result['checks'] = check_processing(source, case, cache, work, yamnet=stage == 'yamnet')
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    finally:
        if source is not None and before is not None:
            result['original_unchanged'] = source.is_file() and sha256(source) == before
            if not result['original_unchanged']:
                result['passed'] = False
                result['source_error'] = 'Original fixture changed'
        result['seconds'] = round(time.monotonic() - started, 3)
        (report / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return int(not result['passed'])


def run_case(case, *, manifest_path=MANIFEST, cache=DEFAULT_CACHE, stage='structure', report, timeout=180):
    report.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest(manifest_path)
    source = verify_asset(case_asset(manifest, case), cache)
    original = sha256(source)
    result = None
    try:
        # Parent owns cleanup even if the worker is forcibly terminated.
        with tempfile.TemporaryDirectory(prefix='work-', dir=report) as work_dir:
            command = [sys.executable, str(Path(__file__).resolve()), '--manifest', str(manifest_path),
                       '--cache', str(cache), '--stage', stage, '--case', case['id'],
                       '--report-dir', str(report), '--work-dir', work_dir, '--worker']
            environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
            environment.pop('PYTHONPATH', None)
            with (report / 'run.log').open('w', encoding='utf-8') as log:
                returncode = run_bounded(command, stdout=log, env=environment, timeout=timeout)
            result_path = report / 'result.json'
            if not result_path.is_file():
                raise RuntimeError(f'Worker exited {returncode} without a result')
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if returncode != 0:
                result['passed'] = False
    except Exception:
        result = dict(id=case['id'], stage=stage, passed=False, error=traceback.format_exc())
    finally:
        unchanged = source.is_file() and sha256(source) == original
        if result is not None:
            result['original_unchanged'] = unchanged
            if not unchanged:
                result['passed'] = False
                result['source_error'] = 'Original fixture changed'
            (report/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--stage', choices=('structure', 'process', 'yamnet'), default='structure')
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--work-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    cases = [c for c in manifest['cases'] if not args.case or c['id'] in args.case]
    if not cases or (args.case and set(args.case) != {c['id'] for c in cases}):
        parser.error('unknown or empty case selection')
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    if args.worker:
        if len(cases) != 1:
            parser.error('worker requires exactly one case')
        if args.work_dir is not None and not args.work_dir.resolve().is_relative_to(args.report_dir.resolve()):
            parser.error('worker directory must belong to its report directory')
        return worker(manifest, cases[0], args.cache.resolve(), args.stage, args.report_dir.resolve(), args.work_dir)
    # Full runs never skip missing assets, including supporting media and oracles.
    for asset in manifest['assets']:
        verify_asset(asset, args.cache)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for index, case in enumerate(cases, 1):
        result = run_case(case, manifest_path=args.manifest.resolve(), cache=args.cache.resolve(),
                          stage=args.stage, report=args.report_dir.resolve()/case['id'], timeout=args.timeout)
        results.append(result)
        (args.report_dir/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(('PASS' if result['passed'] else 'FAIL'), index, '/', len(cases), case['id'], flush=True)
    summary = dict(stage=args.stage, total=len(cases), passed=sum(r['passed'] for r in results),
                   expected_rejections=sum('expected_rejection' in r.get('checks', {}) for r in results),
                   manifest_sha256=sha256(args.manifest), results=results)
    (args.report_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"TOTAL {summary['passed']}/{summary['total']}; expected format rejections: {summary['expected_rejections']}")
    return int(summary['passed'] != summary['total'])


if __name__ == '__main__':
    raise SystemExit(main())
